// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RC (Reliable Connected) endpoint — classic 1:1 QP, full one-sided surface.

#pragma once

#include "context.hpp"
#include "endpoint.hpp"

#include <cstdint>
#include <infiniband/verbs.h>

namespace shmradix {

class RdmaEndpoint {
public:
    RdmaEndpoint() = default;
    explicit RdmaEndpoint(RdmaContext& ctx);
    ~RdmaEndpoint();

    RdmaEndpoint(RdmaEndpoint&& o) noexcept;
    RdmaEndpoint& operator=(RdmaEndpoint&& o) noexcept;
    RdmaEndpoint(const RdmaEndpoint&) = delete;
    RdmaEndpoint& operator=(const RdmaEndpoint&) = delete;

    void connect(const RdmaPeerInfo& remote);

    // Blocking RDMA Read: read `length` bytes from remote_addr into local_buf
    void rdma_read(void* local_buf, uint32_t local_lkey,
                   uint64_t remote_addr, uint32_t remote_rkey,
                   uint32_t length);

    // Batch 2 RDMA Reads: chained WRs, one poll.
    void rdma_read_batch2(
        void* buf1, uint32_t lkey1, uint64_t addr1, uint32_t rkey1, uint32_t len1,
        void* buf2, uint32_t lkey2, uint64_t addr2, uint32_t rkey2, uint32_t len2);

    // Blocking RDMA Write
    void rdma_write(uint64_t remote_addr, uint32_t remote_rkey,
                    const void* local_buf, uint32_t local_lkey,
                    uint32_t length);

    // Async inline write (8B); selective signaling — caller MUST flush() before reading affected data.
    void rdma_write_async_inline(uint64_t remote_addr, uint32_t remote_rkey,
                                  const void* local_buf, uint32_t length);

    // Chain of inline writes in one post_send (last WR signaled); same async semantics.
    struct InlineWriteOp { uint64_t remote_addr; uint32_t remote_rkey;
                            const void* local_buf; uint32_t length; };
    void rdma_write_chain_inline(const InlineWriteOp* ops, int count);

    // Drain outstanding signaled completions; call before a read depending on a prior async write.
    void flush();

    // RDMA Fetch-and-Add: atomically add `add_val` to 8-byte word at remote_addr
    uint64_t rdma_fetch_add(uint64_t remote_addr, uint32_t remote_rkey,
                            uint64_t add_val,
                            void* local_result_buf, uint32_t local_lkey);

    // Batch RDMA Read: chain N reads, signal only last, post once, poll once
    void rdma_read_multi(const RdmaReadOp* ops, int count, uint32_t local_lkey);

    uint32_t qp_num() const;
    RdmaPeerInfo peer_info() const;
    bool valid() const { return qp_ != nullptr; }

private:
    void post_and_poll(ibv_send_wr& wr);
    void drain_one_signaled();  // poll exactly one CQE; decrements outstanding_signaled_

    ibv_qp* qp_ = nullptr;
    ibv_cq* cq_ = nullptr;
    RdmaContext* ctx_ = nullptr;
    uint64_t op_count_ = 0;

    // Selective signaling state for async writes
    static constexpr int SIGNAL_EVERY = 16;     // signal one in this many WRs
    static constexpr int MAX_SEND_WR = 64;      // matches qp_init.cap.max_send_wr
    int inflight_unsignaled_ = 0;               // WRs since last signaled
    int outstanding_signaled_ = 0;              // signaled WRs awaiting CQE
};

// Connect two endpoints in loopback (same machine)
void rdma_loopback_connect(RdmaEndpoint& a, RdmaEndpoint& b);

} // namespace shmradix
