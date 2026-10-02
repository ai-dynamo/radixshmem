// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Internal helpers shared across RDMA plane .cpp files. Not part of the public
// transport surface.

#pragma once

#include <cstdint>
#include <cstdlib>

namespace shmradix {

// Index-plane QPs report a dead peer after (retries + 1) ACK timeouts; the
// defaults (14 = 67 ms, 1 retry) measured ~0.5 s on ConnectX, whose per-attempt
// floor is ~0.25 s. Env overrides.

// QP ACK timeout exponent (4.096us * 2^t per attempt).
inline uint8_t qp_ack_timeout() {
    static uint8_t v = []() -> uint8_t {
        const char* e = ::getenv("SHMRADIX_QP_TIMEOUT");
        int t = e ? atoi(e) : 14;
        if (t < 0) t = 0;
        if (t > 31) t = 31;
        return (uint8_t)t;
    }();
    return v;
}

// Transport retries after the first attempt (0-7).
inline uint8_t qp_retry_cnt() {
    static uint8_t v = []() -> uint8_t {
        const char* e = ::getenv("SHMRADIX_QP_RETRY_CNT");
        int r = e ? atoi(e) : 1;
        if (r < 0) r = 0;
        if (r > 7) r = 7;
        return (uint8_t)r;
    }();
    return v;
}

} // namespace shmradix
