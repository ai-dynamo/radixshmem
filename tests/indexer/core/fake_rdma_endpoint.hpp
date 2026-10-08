// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Test-only IRdmaEndpoint over local memory (remote_addr = in-process
// pointer); counts ops per kind and injects failures for fail-stop tests.

#include "rdma_transport.hpp"

#include <atomic>
#include <cstring>
#include <stdexcept>

namespace radixshmem {

class FakeRdmaEndpoint : public IRdmaEndpoint {
public:
    int reads = 0;
    int faas = 0;
    int writes = 0;

    int fail_read_at = 0;
    int fail_faa_at = 0;

    void reset_counters() { reads = faas = writes = 0; }

    void rdma_read(void* local_buf, uint32_t, uint64_t remote_addr,
                   uint32_t, uint32_t length) override {
        if (++reads == fail_read_at)
            throw std::runtime_error("FakeRdmaEndpoint: injected read failure");
        std::memcpy(local_buf, (const void*)(uintptr_t)remote_addr, length);
    }

    uint64_t rdma_fetch_add(uint64_t remote_addr, uint32_t, uint64_t add_val,
                            void* local_result_buf, uint32_t) override {
        if (++faas == fail_faa_at)
            throw std::runtime_error("FakeRdmaEndpoint: injected FAA failure");
        auto* target =
            reinterpret_cast<std::atomic<uint64_t>*>((uintptr_t)remote_addr);
        uint64_t old = target->fetch_add(add_val, std::memory_order_acq_rel);
        std::memcpy(local_result_buf, &old, 8);
        return old;
    }

    void rdma_read_multi(const RdmaReadOp* ops, int count, uint32_t) override {
        for (int i = 0; i < count; i++) {
            if (++reads == fail_read_at)
                throw std::runtime_error(
                    "FakeRdmaEndpoint: injected read failure");
            std::memcpy(ops[i].local_buf,
                        (const void*)(uintptr_t)ops[i].remote_addr,
                        ops[i].length);
        }
    }

    void rdma_write_async_inline(uint64_t remote_addr, uint32_t,
                                 const void* local_buf,
                                 uint32_t length) override {
        writes++;
        std::memcpy((void*)(uintptr_t)remote_addr, local_buf, length);
    }

    void flush() override {}
};

}
