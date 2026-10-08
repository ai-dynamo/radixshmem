// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "peer_breaker.hpp"

#include <algorithm>
#include <cstdio>
#include <iterator>

namespace radixshmem {

PeerBreaker::PeerBreaker(std::string who, PeerFaultCounters* shared,
                         int backoff_ms, int max_backoff_ms)
    : who_(std::move(who)), shared_(shared),
      backoff_ms_(backoff_ms), max_backoff_ms_(max_backoff_ms) {}

void PeerBreaker::name(int rank, const std::string& host) {
    std::lock_guard<std::mutex> g(mu_);
    peers_[rank].host = host;
}

PeerBreaker::Gate PeerBreaker::gate(int rank) {
    if (down_count_.load(std::memory_order_acquire) == 0) return Gate::PASS;
    std::lock_guard<std::mutex> g(mu_);
    auto it = peers_.find(rank);
    if (it == peers_.end() || !it->second.down) return Gate::PASS;
    Peer& p = it->second;
    auto now = Clock::now();
    if (now < p.retry_at) return Gate::SKIP;
    p.retry_at = now + std::chrono::milliseconds(p.backoff_ms);   // one probe at a time
    return Gate::PROBE;
}

void PeerBreaker::skipped(int rank, Op op, uint64_t n) {
    if (shared_) {
        std::atomic<uint64_t>* c[kOps] = {&shared_->rht_writes, &shared_->rht_removes,
                                          &shared_->rht_lookups, &shared_->walks};
        c[(int)op]->fetch_add(n, std::memory_order_relaxed);
    }
    std::lock_guard<std::mutex> g(mu_);
    Peer& p = peers_[rank];
    p.since_report[(int)op] += n;
    p.while_down[(int)op] += n;
}

void PeerBreaker::failed(int rank, const char* op, const char* what, double secs) {
    std::lock_guard<std::mutex> g(mu_);
    Peer& p = peers_[rank];
    auto now = Clock::now();
    if (!p.down) {
        p.down = true;
        p.attempts = 1;
        p.backoff_ms = backoff_ms_;
        p.since = now;
        std::fill(std::begin(p.while_down), std::end(p.while_down), 0);
        std::fill(std::begin(p.since_report), std::end(p.since_report), 0);
        down_count_.fetch_add(1, std::memory_order_acq_rel);
        if (shared_) shared_->down_events.fetch_add(1, std::memory_order_relaxed);
        fprintf(stderr, "%s: peer rank %d%s unreachable: %s failed after %.2f s: %s; "
                "skipping it for %g s\n", who_.c_str(), rank, label(p).c_str(), op, secs,
                what, p.backoff_ms / 1000.0);
    } else {
        p.attempts++;
        p.backoff_ms = std::min(p.backoff_ms * 2, max_backoff_ms_);
        double down_s = std::chrono::duration<double>(now - p.since).count();
        fprintf(stderr, "%s: peer rank %d%s still unreachable (%d attempts, down %.0f s, "
                "%s: %s); skipped since last report: %s; next try in %g s\n",
                who_.c_str(), rank, label(p).c_str(), p.attempts, down_s, op, what,
                summary(p.since_report).c_str(), p.backoff_ms / 1000.0);
        std::fill(std::begin(p.since_report), std::end(p.since_report), 0);
    }
    p.retry_at = now + std::chrono::milliseconds(p.backoff_ms);
}

void PeerBreaker::succeeded(int rank) {
    if (down_count_.load(std::memory_order_acquire) == 0) return;
    std::lock_guard<std::mutex> g(mu_);
    auto it = peers_.find(rank);
    if (it == peers_.end() || !it->second.down) return;
    Peer& p = it->second;
    double down_s = std::chrono::duration<double>(Clock::now() - p.since).count();
    fprintf(stderr, "%s: peer rank %d%s reachable again after %.0f s; skipped while "
            "down: %s\n", who_.c_str(), rank, label(p).c_str(), down_s,
            summary(p.while_down).c_str());
    p.down = false;
    down_count_.fetch_sub(1, std::memory_order_acq_rel);
}

void PeerBreaker::left(int rank, const char* why) {
    std::lock_guard<std::mutex> g(mu_);
    Peer& p = peers_[rank];
    if (!p.down) {
        p.down = true;
        p.attempts = 0;
        p.since = Clock::now();
        std::fill(std::begin(p.while_down), std::end(p.while_down), 0);
        std::fill(std::begin(p.since_report), std::end(p.since_report), 0);
        down_count_.fetch_add(1, std::memory_order_acq_rel);
        if (shared_) shared_->down_events.fetch_add(1, std::memory_order_relaxed);
    }
    p.backoff_ms = max_backoff_ms_;
    p.retry_at = Clock::time_point::max();
    fprintf(stderr, "%s: peer rank %d%s %s; skipping it until it registers again\n",
            who_.c_str(), rank, label(p).c_str(), why);
}

void PeerBreaker::retry_now(int rank) {
    std::lock_guard<std::mutex> g(mu_);
    auto it = peers_.find(rank);
    if (it == peers_.end() || !it->second.down) return;
    it->second.backoff_ms = backoff_ms_;
    it->second.retry_at = Clock::now();
}

bool PeerBreaker::is_down(int rank) {
    std::lock_guard<std::mutex> g(mu_);
    auto it = peers_.find(rank);
    return it != peers_.end() && it->second.down;
}

std::string PeerBreaker::label(const Peer& p) {
    return p.host.empty() ? "" : " (" + p.host + ")";
}

std::string PeerBreaker::summary(const uint64_t (&n)[kOps]) {
    static const char* const kNames[kOps] = {"RHT writes", "RHT removes", "lookups", "walks"};
    std::string s;
    for (int i = 0; i < kOps; i++) {
        if (!n[i]) continue;
        if (!s.empty()) s += ", ";
        s += std::to_string(n[i]) + " " + kNames[i];
    }
    return s.empty() ? "nothing" : s;
}

} // namespace radixshmem
