// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// bench_insert — measures sustained insert throughput.
//
// Each op: allocate_slots(shortfall) + insert(hashes, slots)  (via insert_auto).
//   → matches the vLLM write path: the manager allocates CPU slots (LRU-evicting
//     when the mempool is full), then registers the staged prefix into the tree.
//
// 6 configurations exercise different pressure points:
//   - no eviction
//   - light eviction
//   - heavy eviction
//   - long prefix (big chunk count per op)
//   - high prefix sharing (split-heavy)
//   - cs-aligned (cluster-style boundary forcing)

#include "bench_util.h"
#include "test_utils.h"
#include "radix_client.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

using namespace radixshmem;
using bench::HashGen;
using bench::LatencyCollector;
using bench::WorkloadConfig;
using bench::now;
using bench::seconds_since;

struct InsertCfg {
    const char* label;
    uint32_t    max_blocks;
    WorkloadConfig wl;
    int         register_chunk_tokens;   // 0 = unaligned, >0 = cs-aligned
    double      duration_sec;
};

struct InsertStats {
    int64_t ops          = 0;
    int64_t blocks_in    = 0;
    int64_t blocks_evict = 0;   // approximate
    double  duration_sec = 0;
    double  p50_us       = 0;
    double  p90_us       = 0;
    double  p99_us       = 0;
    double  max_us       = 0;
};

static InsertStats run_one(const InsertCfg& c) {
    std::string shm_name = std::string("/bench_ins_") + c.label;
    ShmConfig shm;
    shm.max_blocks = c.max_blocks;
    shm.register_chunk_tokens = c.register_chunk_tokens;   // cs now header-borne

    ServerLauncher launcher(shm_name, shm);
    RadixClient client(shm_name);

    HashGen gen(c.wl);
    LatencyCollector lat;

    InsertStats st;

    auto t_start = now();
    while (seconds_since(t_start) < c.duration_sec) {
        const auto& h = gen.next();
        auto t0 = now();
        // insert_auto allocates the shortfall from the mempool then inserts —
        // same allocate+register work the old insert(lock=true) timed.
        int64_t before = (int64_t)client.total_blocks();
        insert_auto(client, h.data(), (int)h.size());
        auto t1 = now();
        lat.record(t0, t1);
        st.ops++;
        // Blocks newly landed = tree growth this op (evict may shrink → clamp ≥0).
        st.blocks_in += std::max<int64_t>(0, (int64_t)client.total_blocks() - before);
    }
    st.duration_sec = seconds_since(t_start);

    // Heuristic: total inserted - blocks still in tree ≈ blocks evicted.
    int64_t final_in_tree = (int64_t)client.total_blocks();
    st.blocks_evict = std::max<int64_t>(0, st.blocks_in - final_in_tree);

    lat.summarize(st.p50_us, st.p90_us, st.p99_us, st.max_us);
    return st;
}

static void print_header() {
    printf("%-22s %8s %10s %12s %12s %12s %10s %10s %10s\n",
           "config", "dur(s)", "ops", "ops/s",
           "blocks_ins", "blocks_evict",
           "p50_us", "p90_us", "p99_us");
    bench::print_separator();
}

static void print_row(const char* label, const InsertStats& st) {
    double ops_per_sec = st.duration_sec > 0
        ? (double)st.ops / st.duration_sec : 0;
    printf("%-22s %8.2f %10lld %12.0f %12lld %12lld "
           "%10.1f %10.1f %10.1f\n",
           label, st.duration_sec, (long long)st.ops, ops_per_sec,
           (long long)st.blocks_in, (long long)st.blocks_evict,
           st.p50_us, st.p90_us, st.p99_us);
}

int main() {
    printf("=== bench_insert (each op = allocate_slots + insert) ===\n\n");

    std::vector<InsertCfg> cfgs;
    // 1. No eviction: mempool large, working set << mempool.
    {
        InsertCfg c{"small-no-evict", 1'000'000, {}, 0, 5.0};
        c.wl.prefix_len_blocks    = 128;
        c.wl.shared_prefix_blocks = 0;
        c.wl.num_unique_streams   = 1000;
        cfgs.push_back(c);
    }
    // 2. Light eviction: WS ≈ mempool (slightly over).
    {
        InsertCfg c{"medium-light-evict", 50'000, {}, 0, 5.0};
        c.wl.prefix_len_blocks    = 128;
        c.wl.shared_prefix_blocks = 64;
        c.wl.num_unique_streams   = 500;   // WS ≈ 500*128 = 64k > 50k
        cfgs.push_back(c);
    }
    // 3. Heavy eviction: WS >> mempool.
    {
        InsertCfg c{"tight-heavy-evict", 5'000, {}, 0, 5.0};
        c.wl.prefix_len_blocks    = 128;
        c.wl.shared_prefix_blocks = 64;
        c.wl.num_unique_streams   = 500;
        cfgs.push_back(c);
    }
    // 4. Long prefix (1024 blocks/op).
    {
        InsertCfg c{"long-prefix", 100'000, {}, 0, 5.0};
        c.wl.prefix_len_blocks    = 1024;
        c.wl.shared_prefix_blocks = 256;
        c.wl.num_unique_streams   = 50;
        cfgs.push_back(c);
    }
    // 5. Highly shared prefix → split-heavy.
    {
        InsertCfg c{"shared-prefix", 50'000, {}, 0, 5.0};
        c.wl.prefix_len_blocks    = 128;
        c.wl.shared_prefix_blocks = 120;
        c.wl.num_unique_streams   = 200;
        cfgs.push_back(c);
    }
    // 6. cs-aligned (cluster-style boundary).
    {
        InsertCfg c{"cs-aligned-16", 50'000, {}, 16, 5.0};
        c.wl.prefix_len_blocks    = 128;
        c.wl.shared_prefix_blocks = 64;
        c.wl.num_unique_streams   = 500;
        cfgs.push_back(c);
    }

    print_header();
    for (auto& cfg : cfgs) {
        auto st = run_one(cfg);
        print_row(cfg.label, st);
        fflush(stdout);
    }
    printf("\nDone.\n");
    return 0;
}
