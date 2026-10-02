// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Per-peer circuit breaker for the one-sided index planes (RHT, remote walks).
// A failed op marks the peer unreachable; ops to it are skipped for a backoff
// (1 s, doubling to 30 s), then one op probes it. Transitions are logged,
// skipped ops are counted here and in the node's shm counters.

#pragma once

#include "shmradix/types.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace shmradix {

class PeerBreaker {
public:
    enum class Op { RHT_WRITE, RHT_REMOVE, RHT_LOOKUP, WALK, COUNT };
    enum class Gate { PASS, SKIP, PROBE };

    // who: log prefix, e.g. "[RadixClient rank 0] RHT". shared may be null.
    PeerBreaker(std::string who, PeerFaultCounters* shared,
                int backoff_ms = 1000, int max_backoff_ms = 30000);

    // Printable identity of `rank` for the log lines (its host).
    void name(int rank, const std::string& host);

    // Before an op to `rank`. PROBE: the backoff ran out, try once.
    Gate gate(int rank);
    void skipped(int rank, Op op, uint64_t n = 1);
    void failed(int rank, const char* op, const char* what, double secs);
    void succeeded(int rank);
    // The peer has no etcd entry: down, not probed until retry_now().
    void left(int rank, const char* why);
    // The peer published a new entry: probe it on the next op.
    void retry_now(int rank);

    bool is_down(int rank);

private:
    using Clock = std::chrono::steady_clock;
    static constexpr int kOps = (int)Op::COUNT;

    struct Peer {
        std::string host;
        bool down = false;
        int attempts = 0;
        int backoff_ms = 0;
        Clock::time_point since, retry_at;
        uint64_t since_report[kOps] = {};
        uint64_t while_down[kOps] = {};
    };

    static std::string label(const Peer& p);
    static std::string summary(const uint64_t (&n)[kOps]);

    std::string who_;
    PeerFaultCounters* shared_;
    int backoff_ms_, max_backoff_ms_;
    std::atomic<int> down_count_{0};
    std::mutex mu_;
    std::unordered_map<int, Peer> peers_;
};

} // namespace shmradix
