// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Transport-agnostic one-sided RDMA surface (IRdmaEndpoint): blocking read /
// fetch-add / multi-read plus 8B async-inline writes; XRC and DC implement it.

#pragma once

#include <cstddef>
#include <cstdint>
#include <infiniband/verbs.h>

namespace shmradix {

// One entry of a chained multi-read.
struct RdmaReadOp {
    void* local_buf;
    uint64_t remote_addr;
    uint32_t remote_rkey;
    uint32_t length;
};

// RC address vector material (exchanged at connect).
struct RdmaPeerInfo {
    uint32_t qp_num;
    union ibv_gid gid;
    int gid_index;
    uint16_t lid = 0;
    uint8_t  link_layer = 0;
};

// Memory-region descriptor (rkey + remote addr + length).
struct RdmaMrInfo {
    uint32_t rkey;
    uint64_t addr;
    size_t length;
};

// Longest read chain per post; callers keep at most one chain per endpoint
// in flight (the XRC SQ is sized for one on top of the async-write backlog).
static constexpr int RDMA_READ_CHAIN_MAX = 16;

class IRdmaEndpoint {
public:
    virtual ~IRdmaEndpoint() = default;

    // Blocking one-sided read of `length` bytes from (remote_addr, remote_rkey)
    // into local_buf (registered under local_lkey).
    virtual void rdma_read(void* local_buf, uint32_t local_lkey,
                           uint64_t remote_addr, uint32_t remote_rkey,
                           uint32_t length) = 0;

    // Blocking atomic fetch-and-add of add_val at (remote_addr, remote_rkey).
    // Returns the pre-add value; also written to local_result_buf.
    virtual uint64_t rdma_fetch_add(uint64_t remote_addr, uint32_t remote_rkey,
                                    uint64_t add_val,
                                    void* local_result_buf,
                                    uint32_t local_lkey) = 0;

    // Blocking multi-read of any length, RDMA_READ_CHAIN_MAX per round trip.
    virtual void rdma_read_multi(const RdmaReadOp* ops, int count,
                                 uint32_t local_lkey) = 0;

    // One-sided async-inline write of `length` bytes (RHT uses 16B cells) to
    // (remote_addr, remote_rkey). Unsignaled, non-blocking; flush() drains.
    virtual void rdma_write_async_inline(uint64_t remote_addr,
                                         uint32_t remote_rkey,
                                         const void* local_buf,
                                         uint32_t length) = 0;

    // Drain all pending async writes on this endpoint.
    virtual void flush() = 0;

    // Post one chain (1..RDMA_READ_CHAIN_MAX reads, last signaled), no poll.
    // Returns the completions owed to rdma_read_multi_wait; default: blocking
    // read, 0 owed.
    virtual int rdma_read_multi_post(const RdmaReadOp* ops, int count,
                                     uint32_t local_lkey) {
        rdma_read_multi(ops, count, local_lkey);
        return 0;
    }
    // Collect `pending` completions from any endpoint on this CQ. One polling
    // thread per CQ. No-op by default.
    virtual void rdma_read_multi_wait(int pending) { (void)pending; }

    // True once an op failed on this endpoint and it can only be replaced.
    virtual bool broken() { return false; }
};

} // namespace shmradix
