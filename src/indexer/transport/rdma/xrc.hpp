// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// XRC (Extended Reliable Connected) — N clients share one mutex-guarded SEND QP
// per peer; mlx5 enforces 1:1 SEND↔RECV pairing, so savings = 1 QP/peer, not N.

#pragma once

#include "context.hpp"
#include "endpoint.hpp"

#include <atomic>
#include <cstdint>
#include <infiniband/verbs.h>
#include <pthread.h>

namespace radixshmem {

struct XrcPeerInfo {
    uint32_t          recv_qp_num;   // peer's XRC RECV QP num
    uint32_t          remote_srqn;   // peer's XRC SRQ num (carried per WR)
    union ibv_gid     gid;
    int               gid_index;
    uint16_t          lid;
    uint8_t           link_layer;
};

class RdmaXrcd {
public:
    explicit RdmaXrcd(RdmaContext& ctx);
    ~RdmaXrcd();

    RdmaXrcd(const RdmaXrcd&) = delete;
    RdmaXrcd& operator=(const RdmaXrcd&) = delete;

    ibv_xrcd* xrcd() const { return xrcd_; }
private:
    ibv_xrcd* xrcd_ = nullptr;
    int       fd_   = -1;
};

class XrcSrq {
public:
    XrcSrq(RdmaContext& ctx, RdmaXrcd& xrcd, ibv_cq* cq);
    ~XrcSrq();

    XrcSrq(const XrcSrq&) = delete;
    XrcSrq& operator=(const XrcSrq&) = delete;

    uint32_t srq_num() const { return srq_num_; }
    ibv_srq* srq() const { return srq_; }
private:
    ibv_srq* srq_ = nullptr;
    uint32_t srq_num_ = 0;
};

// XRC RECV QP — owned by the receiving side (server); must be RTR/RTS to terminate WRs.
class XrcRecvQp {
public:
    XrcRecvQp(RdmaContext& ctx, RdmaXrcd& xrcd, XrcSrq& srq, ibv_cq* cq);
    ~XrcRecvQp();

    XrcRecvQp(const XrcRecvQp&) = delete;
    XrcRecvQp& operator=(const XrcRecvQp&) = delete;

    void connect(uint32_t peer_send_qp_num,
                 const XrcPeerInfo& peer_av);

    uint32_t qp_num() const { return qp_num_; }
private:
    ibv_qp*  qp_ = nullptr;
    uint32_t qp_num_ = 0;
    RdmaContext* ctx_ = nullptr;
};

// POD shared state serializing one XRC SEND QP across threads/processes (mmap-safe:
// PROCESS_SHARED+ROBUST mutex + signaling counters).
struct XrcSendQpSharedState {
    pthread_mutex_t mu;               // serializes all datapath ops
    std::atomic<int> inflight_unsignaled;   // WRs since last signaled
    std::atomic<int> outstanding_signaled;  // signaled WRs awaiting CQE
    // Shared CQ: SYNC CQEs (or a WC error) of this QP taken by another poller land here.
    std::atomic<int> sync_credits;
    std::atomic<int> sync_error;

    // Init as cross-process robust mutex (creator only, exactly once).
    static void init_shared(XrcSendQpSharedState* s);
    // Init for single-process use (PRIVATE attr).
    static void init_private(XrcSendQpSharedState* s);
    // Destroy the mutex (creator only).
    static void destroy(XrcSendQpSharedState* s);
};

// Shared XRC SEND QP, API-compatible with RdmaEndpoint; datapath ops lock the
// shared state (safe N-thread, and N-process when state_in_shm).
class XrcSendQp : public IRdmaEndpoint {
public:
    // Single-process: owns a private heap XrcSendQpSharedState.
    XrcSendQp(RdmaContext& ctx, ibv_cq* cq);

    // Multi-process: caller supplies a shm pointer (init_shared'd by creator);
    // does NOT take ownership.
    XrcSendQp(RdmaContext& ctx, ibv_cq* cq,
              XrcSendQpSharedState* shared_state,
              bool state_already_initialized);

    ~XrcSendQp();

    XrcSendQp(const XrcSendQp&) = delete;
    XrcSendQp& operator=(const XrcSendQp&) = delete;

    void connect(const XrcPeerInfo& peer);
    uint32_t qp_num() const { return qp_num_; }
    uint32_t remote_srqn() const { return remote_srqn_; }

    // Same surface as RdmaEndpoint
    void rdma_read(void* local_buf, uint32_t local_lkey,
                   uint64_t remote_addr, uint32_t remote_rkey,
                   uint32_t length) override;

    void rdma_read_batch2(
        void* buf1, uint32_t lkey1, uint64_t addr1, uint32_t rkey1, uint32_t len1,
        void* buf2, uint32_t lkey2, uint64_t addr2, uint32_t rkey2, uint32_t len2);

    void rdma_write(uint64_t remote_addr, uint32_t remote_rkey,
                    const void* local_buf, uint32_t local_lkey,
                    uint32_t length);

    void rdma_write_async_inline(uint64_t remote_addr, uint32_t remote_rkey,
                                  const void* local_buf, uint32_t length) override;

    struct InlineWriteOp { uint64_t remote_addr; uint32_t remote_rkey;
                            const void* local_buf; uint32_t length; };
    void rdma_write_chain_inline(const InlineWriteOp* ops, int count);

    void flush() override;

    uint64_t rdma_fetch_add(uint64_t remote_addr, uint32_t remote_rkey,
                            uint64_t add_val,
                            void* local_result_buf, uint32_t local_lkey) override;

    void rdma_read_multi(const RdmaReadOp* ops, int count,
                         uint32_t local_lkey) override;

    // One chain of 1..RDMA_READ_CHAIN_MAX reads, no poll; returns 1.
    int rdma_read_multi_post(const RdmaReadOp* ops, int count,
                             uint32_t local_lkey) override;
    void rdma_read_multi_wait(int pending) override;
    bool broken() override;

private:
    void post_and_poll_locked(ibv_send_wr& wr);
    void drain_one_signaled_locked();
    // Chain ops[0..n), signal the last, post once.
    void post_read_chain_locked(const RdmaReadOp* ops, int n,
                                uint32_t local_lkey);
    // Poll cq_ for n SYNC completions; ASYNC ones are credited to their QP.
    void poll_sync(int n, const char* what);

    ibv_qp*  qp_ = nullptr;
    uint32_t qp_num_ = 0;
    ibv_cq*  cq_ = nullptr;
    RdmaContext* ctx_ = nullptr;
    uint32_t remote_srqn_ = 0;
    uint64_t op_count_ = 0;

    static constexpr int SIGNAL_EVERY = 16;
    static constexpr int MAX_SEND_WR = 64;

    // Shared state (mutex + counters); state_owned_ = private heap, else shm-owned.
    XrcSendQpSharedState* state_ = nullptr;
    bool state_owned_ = false;
};

} // namespace radixshmem
