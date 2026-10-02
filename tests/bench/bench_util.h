// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared helpers for bench_insert / bench_query / future bench_cluster.

#pragma once

#include "shmradix/radix_client.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace bench {

// steady_clock (monotonic): all measurements are intervals; avoids NTP-step corruption.
using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
inline TimePoint now() { return Clock::now(); }
inline double seconds_since(TimePoint t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}
inline double us(TimePoint a, TimePoint b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

// ==================== Workload ====================
// Each stream: fixed shared_prefix head (每条一个) + rolling private tail.
// next() emits shared+tail hashes; streams rotate round-robin.
struct WorkloadConfig {
    int  prefix_len_blocks    = 128;
    int  shared_prefix_blocks = 64;    // must be <= prefix_len_blocks
    int  num_unique_streams   = 500;
    uint64_t seed              = 0xC0FFEE;
};

class HashGen {
public:
    explicit HashGen(const WorkloadConfig& cfg) : cfg_(cfg) {
        if (cfg_.shared_prefix_blocks > cfg_.prefix_len_blocks)
            cfg_.shared_prefix_blocks = cfg_.prefix_len_blocks;

        // Pre-build each stream's static system-prompt head.
        std::mt19937_64 rng(cfg_.seed);
        sys_heads_.resize(cfg_.num_unique_streams);
        for (int s = 0; s < cfg_.num_unique_streams; s++) {
            sys_heads_[s].resize(cfg_.shared_prefix_blocks);
            for (auto& t : sys_heads_[s]) t = (int64_t)rng();
        }
        tail_counters_.assign(cfg_.num_unique_streams, 0);

        // Reusable scratch buffer reused across next() calls.
        scratch_tokens_.resize(cfg_.prefix_len_blocks);
        scratch_hashes_.resize(cfg_.prefix_len_blocks);
    }

    const std::vector<uint64_t>& next() {
        int s = cursor_;
        cursor_ = (cursor_ + 1) % cfg_.num_unique_streams;

        int shared = cfg_.shared_prefix_blocks;
        int total  = cfg_.prefix_len_blocks;

        // [0, shared): stream's fixed system prompt
        for (int i = 0; i < shared; i++)
            scratch_tokens_[i] = sys_heads_[s][i];

        // [shared, total): rolling private tail keyed on (stream_id, counter)
        uint64_t base = ((uint64_t)s << 32) ^ tail_counters_[s];
        for (int i = shared; i < total; i++)
            scratch_tokens_[i] = (int64_t)(base * 1315423911u + (uint64_t)i);
        tail_counters_[s]++;

        auto h = shmradix::RadixClient::compute_hashes(
            scratch_tokens_.data(), total, 1);
        scratch_hashes_ = std::move(h);
        return scratch_hashes_;
    }

    // Re-emit the LAST hashes for stream `s` (for query bench).
    const std::vector<uint64_t>& replay_stream(int s) {
        if (s < 0 || s >= cfg_.num_unique_streams)
            s = cursor_ % cfg_.num_unique_streams;
        int shared = cfg_.shared_prefix_blocks;
        int total  = cfg_.prefix_len_blocks;
        for (int i = 0; i < shared; i++)
            scratch_tokens_[i] = sys_heads_[s][i];
        uint64_t counter = (tail_counters_[s] > 0) ? tail_counters_[s] - 1 : 0;
        uint64_t base = ((uint64_t)s << 32) ^ counter;
        for (int i = shared; i < total; i++)
            scratch_tokens_[i] = (int64_t)(base * 1315423911u + (uint64_t)i);
        scratch_hashes_ = shmradix::RadixClient::compute_hashes(
            scratch_tokens_.data(), total, 1);
        return scratch_hashes_;
    }

    int prefix_len() const { return cfg_.prefix_len_blocks; }
    int num_streams() const { return cfg_.num_unique_streams; }

private:
    WorkloadConfig cfg_;
    std::vector<std::vector<int64_t>> sys_heads_;
    std::vector<uint64_t> tail_counters_;
    std::vector<int64_t>  scratch_tokens_;
    std::vector<uint64_t> scratch_hashes_;
    int cursor_ = 0;
};

// ==================== Latency collector ====================
// cap_==0: unbounded full capture. cap_>0: full up to cap then Algorithm-R
// reservoir (percentiles exact below cap, estimates above); max_ always exact.
class LatencyCollector {
public:
    void clear() {
        samples_.clear(); count_ = 0;
        max_ = -std::numeric_limits<double>::infinity();
    }

    // Enable bounded/reservoir mode with capacity `cap`. Call once, before use.
    void reserve_for(size_t cap) { cap_ = cap; samples_.reserve(cap); }

    void record(double sample_us) {
        count_++;
        if (sample_us > max_) max_ = sample_us;
        if (cap_ == 0 || samples_.size() < cap_) {
            samples_.push_back(sample_us);   // full capture (reserved → no realloc)
            return;
        }
        // Reservoir (Algorithm R): retain each seen sample w.p. cap_/count_.
        uint64_t j = next_rand() % count_;
        if (j < cap_) samples_[(size_t)j] = sample_us;
    }
    void record(TimePoint a, TimePoint b) { record(us(a, b)); }

    void summarize(double& p50, double& p90, double& p99, double& maxv) {
        double p999;
        summarize(p50, p90, p99, p999, maxv);
    }

    // Also reports p999; needs >= ~1000 samples to be meaningful (see count()).
    void summarize(double& p50, double& p90, double& p99, double& p999,
                   double& maxv) {
        if (samples_.empty()) { p50 = p90 = p99 = p999 = maxv = 0; return; }
        std::sort(samples_.begin(), samples_.end());
        auto pick = [&](double q) {
            size_t i = (size_t)((double)(samples_.size() - 1) * q);
            return samples_[i];
        };
        p50  = pick(0.50);
        p90  = pick(0.90);
        p99  = pick(0.99);
        p999 = pick(0.999);
        maxv = max_;   // exact running max, never a lossy reservoir sample
    }

    size_t   count()  const { return count_; }          // total ops seen
    size_t   sampled() const { return samples_.size(); } // how many are retained
    // True once the reservoir engaged: percentiles are then estimates.
    bool     capped() const { return cap_ != 0 && count_ > cap_; }
private:
    // Cheap deterministic xorshift64 PRNG — no <random> overhead in hot loop.
    uint64_t next_rand() {
        uint64_t x = rng_; x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        return rng_ = x;
    }
    std::vector<double> samples_;
    size_t   cap_   = 0;                 // 0 = unbounded full capture
    uint64_t count_ = 0;                 // total record() calls (all ops)
    double   max_   = -std::numeric_limits<double>::infinity();  // -inf sentinel, not 0
    uint64_t rng_   = 0x9e3779b97f4a7c15ULL;  // fixed seed → reproducible
};

// ==================== Output helpers ====================
inline void print_separator() {
    printf("─────────────────────────────────────────────────────────"
           "──────────────────────────────────────────\n");
}

} // namespace bench
