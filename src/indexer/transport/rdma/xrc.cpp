// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "xrc.hpp"

#include "detail.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <infiniband/verbs.h>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace shmradix {

// ==================== XRCD ====================

RdmaXrcd::RdmaXrcd(RdmaContext& ctx) {
    ibv_xrcd_init_attr xattr = {};
    xattr.comp_mask = IBV_XRCD_INIT_ATTR_FD | IBV_XRCD_INIT_ATTR_OFLAGS;
    xattr.fd = -1;
    xattr.oflags = O_CREAT;
    xrcd_ = ibv_open_xrcd(ctx.ctx(), &xattr);
    if (!xrcd_)
        throw std::runtime_error("ibv_open_xrcd failed (errno=" +
                                 std::to_string(errno) + ")");
}

RdmaXrcd::~RdmaXrcd() {
    if (xrcd_) ibv_close_xrcd(xrcd_);
}

// ==================== XRC SRQ ====================

XrcSrq::XrcSrq(RdmaContext& ctx, RdmaXrcd& xrcd, ibv_cq* cq) {
    ibv_srq_init_attr_ex sattr = {};
    sattr.comp_mask = IBV_SRQ_INIT_ATTR_TYPE | IBV_SRQ_INIT_ATTR_PD |
                      IBV_SRQ_INIT_ATTR_CQ  | IBV_SRQ_INIT_ATTR_XRCD;
    sattr.srq_type = IBV_SRQT_XRC;
    sattr.pd       = ctx.pd();
    sattr.xrcd     = xrcd.xrcd();
    sattr.cq       = cq;
    sattr.attr.max_wr  = 16;
    sattr.attr.max_sge = 1;
    srq_ = ibv_create_srq_ex(ctx.ctx(), &sattr);
    if (!srq_)
        throw std::runtime_error("ibv_create_srq_ex(XRC) failed");
    if (ibv_get_srq_num(srq_, &srq_num_) != 0) {
        ibv_destroy_srq(srq_);
        srq_ = nullptr;
        throw std::runtime_error("ibv_get_srq_num failed");
    }
}

XrcSrq::~XrcSrq() {
    if (srq_) ibv_destroy_srq(srq_);
}

// ==================== XRC RECV QP ====================

XrcRecvQp::XrcRecvQp(RdmaContext& ctx, RdmaXrcd& xrcd, XrcSrq& srq, ibv_cq* cq)
    : ctx_(&ctx) {
    ibv_qp_init_attr_ex qiae = {};
    qiae.qp_type = IBV_QPT_XRC_RECV;
    qiae.recv_cq = cq;
    qiae.send_cq = cq;
    qiae.srq     = srq.srq();
    qiae.xrcd    = xrcd.xrcd();
    qiae.cap.max_send_wr  = 0;
    qiae.cap.max_recv_wr  = 0;
    qiae.cap.max_send_sge = 0;
    qiae.cap.max_recv_sge = 0;
    qiae.pd      = ctx.pd();
    qiae.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_XRCD;
    qp_ = ibv_create_qp_ex(ctx.ctx(), &qiae);
    if (!qp_)
        throw std::runtime_error("ibv_create_qp_ex(XRC_RECV) failed");
    qp_num_ = qp_->qp_num;
}

XrcRecvQp::~XrcRecvQp() {
    if (qp_) ibv_destroy_qp(qp_);
}

void XrcRecvQp::connect(uint32_t peer_send_qp_num,
                        const XrcPeerInfo& peer_av) {
    // INIT
    {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_INIT;
        a.pkey_index = 0;
        a.port_num = ctx_->port();
        a.qp_access_flags = IBV_ACCESS_REMOTE_READ |
                            IBV_ACCESS_REMOTE_WRITE |
                            IBV_ACCESS_REMOTE_ATOMIC;
        if (ibv_modify_qp(qp_, &a,
                          IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                              IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0)
            throw std::runtime_error("XRC RECV INIT failed");
    }
    // RTR
    {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_RTR;
        a.path_mtu = IBV_MTU_4096;
        a.dest_qp_num = peer_send_qp_num;
        a.rq_psn = 0;
        a.max_dest_rd_atomic = 16;
        a.min_rnr_timer = 12;
        a.ah_attr.dlid = peer_av.lid;
        a.ah_attr.sl = ctx_->sl();
        a.ah_attr.src_path_bits = 0;
        a.ah_attr.port_num = ctx_->port();
        if (peer_av.link_layer == IBV_LINK_LAYER_INFINIBAND) {
            a.ah_attr.is_global = 0;
        } else {
            a.ah_attr.is_global = 1;
            a.ah_attr.grh.dgid = peer_av.gid;
            a.ah_attr.grh.sgid_index = ctx_->gid_index();
            a.ah_attr.grh.hop_limit = 1;
            a.ah_attr.grh.traffic_class = ctx_->traffic_class();
        }
        if (ibv_modify_qp(qp_, &a,
                          IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                              IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                              IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0)
            throw std::runtime_error("XRC RECV RTR failed");
    }
    // RTS
    {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_RTS;
        a.timeout = qp_ack_timeout();
        a.retry_cnt = qp_retry_cnt();
        a.rnr_retry = 7;
        a.sq_psn = 0;
        a.max_rd_atomic = 16;
        if (ibv_modify_qp(qp_, &a,
                          IBV_QP_STATE | IBV_QP_TIMEOUT |
                              IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                              IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0)
            throw std::runtime_error("XRC RECV RTS failed");
    }
}

// ==================== XRC SEND QP ====================

// RAII lock for pthread_mutex_t; recovers a ROBUST mutex via EOWNERDEAD.
struct XrcLockGuard {
    pthread_mutex_t* m;
    explicit XrcLockGuard(pthread_mutex_t* mm) : m(mm) {
        int rc = pthread_mutex_lock(m);
        if (rc == EOWNERDEAD) {
            pthread_mutex_consistent(m);
        } else if (rc != 0) {
            throw std::runtime_error("XRC pthread_mutex_lock failed");
        }
    }
    ~XrcLockGuard() { pthread_mutex_unlock(m); }
    XrcLockGuard(const XrcLockGuard&) = delete;
    XrcLockGuard& operator=(const XrcLockGuard&) = delete;
};

// ---------- XrcSendQpSharedState ----------

void XrcSendQpSharedState::init_shared(XrcSendQpSharedState* s) {
    pthread_mutexattr_t attr;
    if (pthread_mutexattr_init(&attr) != 0)
        throw std::runtime_error("pthread_mutexattr_init failed");
    if (pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED) != 0)
        throw std::runtime_error("setpshared(PROCESS_SHARED) failed");
    if (pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST) != 0)
        throw std::runtime_error("setrobust(ROBUST) failed");
    if (pthread_mutex_init(&s->mu, &attr) != 0)
        throw std::runtime_error("pthread_mutex_init (shared) failed");
    pthread_mutexattr_destroy(&attr);
    s->inflight_unsignaled.store(0, std::memory_order_relaxed);
    s->outstanding_signaled.store(0, std::memory_order_relaxed);
}

void XrcSendQpSharedState::init_private(XrcSendQpSharedState* s) {
    pthread_mutexattr_t attr;
    if (pthread_mutexattr_init(&attr) != 0)
        throw std::runtime_error("pthread_mutexattr_init failed");
    if (pthread_mutex_init(&s->mu, &attr) != 0)
        throw std::runtime_error("pthread_mutex_init (private) failed");
    pthread_mutexattr_destroy(&attr);
    s->inflight_unsignaled.store(0, std::memory_order_relaxed);
    s->outstanding_signaled.store(0, std::memory_order_relaxed);
}

void XrcSendQpSharedState::destroy(XrcSendQpSharedState* s) {
    pthread_mutex_destroy(&s->mu);
}

// ---------- XrcSendQp ----------

// Shared-CQ accounting: credit each ASYNC CQE to its owning QP (wc.qp_num)
// via a process-global registry, not the polling QP.
namespace {
std::mutex g_xrc_reg_mu;
std::unordered_map<uint32_t, XrcSendQpSharedState*> g_xrc_reg;

void xrc_reg_add(uint32_t qp_num, XrcSendQpSharedState* st) {
    std::lock_guard<std::mutex> g(g_xrc_reg_mu);
    g_xrc_reg[qp_num] = st;
}
void xrc_reg_remove(uint32_t qp_num) {
    std::lock_guard<std::mutex> g(g_xrc_reg_mu);
    g_xrc_reg.erase(qp_num);
}
// Credit one ASYNC completion to its owning QP; returns true iff that QP is `self`.
bool xrc_credit_async(const ibv_wc& wc, XrcSendQpSharedState* self) {
    XrcSendQpSharedState* owner = nullptr;
    {
        std::lock_guard<std::mutex> g(g_xrc_reg_mu);
        auto it = g_xrc_reg.find(wc.qp_num);
        if (it != g_xrc_reg.end()) owner = it->second;
    }
    if (!owner) owner = self;   // defensive: unknown qp_num
    int prev = owner->outstanding_signaled.load(std::memory_order_relaxed);
    if (prev > 0)
        owner->outstanding_signaled.store(prev - 1, std::memory_order_relaxed);
    return owner == self;
}
}  // namespace

XrcSendQp::XrcSendQp(RdmaContext& ctx, ibv_cq* cq) : cq_(cq), ctx_(&ctx) {
    ibv_qp_init_attr_ex qiae = {};
    qiae.qp_type = IBV_QPT_XRC_SEND;
    qiae.send_cq = cq;
    qiae.recv_cq = cq;
    qiae.cap.max_send_wr  = MAX_SEND_WR;
    qiae.cap.max_recv_wr  = 0;
    qiae.cap.max_send_sge = 1;
    qiae.cap.max_recv_sge = 0;
    qiae.cap.max_inline_data = 64;
    qiae.pd      = ctx.pd();
    qiae.comp_mask = IBV_QP_INIT_ATTR_PD;
    qp_ = ibv_create_qp_ex(ctx.ctx(), &qiae);
    if (!qp_)
        throw std::runtime_error("ibv_create_qp_ex(XRC_SEND) failed");
    qp_num_ = qp_->qp_num;

    state_ = new XrcSendQpSharedState();
    XrcSendQpSharedState::init_private(state_);
    state_owned_ = true;
    xrc_reg_add(qp_num_, state_);
}

XrcSendQp::XrcSendQp(RdmaContext& ctx, ibv_cq* cq,
                     XrcSendQpSharedState* shared_state,
                     bool state_already_initialized)
    : cq_(cq), ctx_(&ctx) {
    ibv_qp_init_attr_ex qiae = {};
    qiae.qp_type = IBV_QPT_XRC_SEND;
    qiae.send_cq = cq;
    qiae.recv_cq = cq;
    qiae.cap.max_send_wr  = MAX_SEND_WR;
    qiae.cap.max_recv_wr  = 0;
    qiae.cap.max_send_sge = 1;
    qiae.cap.max_recv_sge = 0;
    qiae.cap.max_inline_data = 64;
    qiae.pd      = ctx.pd();
    qiae.comp_mask = IBV_QP_INIT_ATTR_PD;
    qp_ = ibv_create_qp_ex(ctx.ctx(), &qiae);
    if (!qp_)
        throw std::runtime_error("ibv_create_qp_ex(XRC_SEND) failed");
    qp_num_ = qp_->qp_num;

    state_ = shared_state;
    state_owned_ = false;
    if (!state_already_initialized) {
        XrcSendQpSharedState::init_shared(state_);
    }
    xrc_reg_add(qp_num_, state_);
}

XrcSendQp::~XrcSendQp() {
    if (qp_) xrc_reg_remove(qp_num_);
    if (qp_) ibv_destroy_qp(qp_);
    if (state_owned_ && state_) {
        XrcSendQpSharedState::destroy(state_);
        delete state_;
    }
}

void XrcSendQp::connect(const XrcPeerInfo& peer) {
    remote_srqn_ = peer.remote_srqn;
    // INIT
    {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_INIT;
        a.pkey_index = 0;
        a.port_num = ctx_->port();
        a.qp_access_flags = IBV_ACCESS_REMOTE_READ |
                            IBV_ACCESS_REMOTE_WRITE |
                            IBV_ACCESS_REMOTE_ATOMIC;
        if (ibv_modify_qp(qp_, &a,
                          IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                              IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0)
            throw std::runtime_error("XRC SEND INIT failed");
    }
    // RTR — XRC_INI mask must NOT include MAX_DEST_RD_ATOMIC/MIN_RNR_TIMER (mlx5 EINVAL).
    {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_RTR;
        a.path_mtu = IBV_MTU_4096;
        a.dest_qp_num = peer.recv_qp_num;
        a.rq_psn = 0;
        a.ah_attr.dlid = peer.lid;
        a.ah_attr.sl = ctx_->sl();
        a.ah_attr.src_path_bits = 0;
        a.ah_attr.port_num = ctx_->port();
        if (peer.link_layer == IBV_LINK_LAYER_INFINIBAND) {
            a.ah_attr.is_global = 0;
        } else {
            a.ah_attr.is_global = 1;
            a.ah_attr.grh.dgid = peer.gid;
            a.ah_attr.grh.sgid_index = ctx_->gid_index();
            a.ah_attr.grh.hop_limit = 1;
            a.ah_attr.grh.traffic_class = ctx_->traffic_class();
        }
        if (ibv_modify_qp(qp_, &a,
                          IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                              IBV_QP_DEST_QPN | IBV_QP_RQ_PSN) != 0)
            throw std::runtime_error("XRC SEND RTR failed: " +
                                     std::string(strerror(errno)));
    }
    // RTS
    {
        ibv_qp_attr a = {};
        a.qp_state = IBV_QPS_RTS;
        a.timeout = qp_ack_timeout();
        a.retry_cnt = qp_retry_cnt();
        a.rnr_retry = 7;
        a.sq_psn = 0;
        a.max_rd_atomic = 16;
        if (ibv_modify_qp(qp_, &a,
                          IBV_QP_STATE | IBV_QP_TIMEOUT |
                              IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                              IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0)
            throw std::runtime_error("XRC SEND RTS failed");
    }
}

// XRC wr tags, separate id space from RC to avoid CQ mix-ups.
static constexpr uint64_t XRC_WR_ID_SYNC  = 11;
static constexpr uint64_t XRC_WR_ID_ASYNC = 12;

void XrcSendQp::poll_sync(int n, const char* what) {
    while (n > 0) {
        ibv_wc wc = {};
        int ne = ibv_poll_cq(cq_, 1, &wc);
        if (ne == 0) continue;
        if (ne < 0) {
            char msg[128];
            snprintf(msg, sizeof(msg), "XRC %s: ibv_poll_cq failed", what);
            throw std::runtime_error(msg);
        }
        if (wc.status != IBV_WC_SUCCESS) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                     "XRC %s WC error: status=%d opcode=%d vendor=%u "
                     "qp_num=%u wr_id=%lu",
                     what, wc.status, wc.opcode, wc.vendor_err, wc.qp_num,
                     (unsigned long)wc.wr_id);
            throw std::runtime_error(msg);
        }
        if (wc.wr_id == XRC_WR_ID_ASYNC) {
            // Credit to the QP that owns this CQE (by qp_num), not necessarily us.
            xrc_credit_async(wc, state_);
            continue;
        }
        n--;   // a SYNC completion on this CQ
    }
}

void XrcSendQp::post_read_chain_locked(const RdmaReadOp* ops, int n,
                                       uint32_t local_lkey) {
    assert(n >= 1 && n <= RDMA_READ_CHAIN_MAX);
    ibv_sge sges[RDMA_READ_CHAIN_MAX];
    ibv_send_wr wrs[RDMA_READ_CHAIN_MAX];
    for (int i = 0; i < n; i++) {
        const auto& op = ops[i];
        sges[i] = {};
        sges[i].addr = (uint64_t)op.local_buf;
        sges[i].length = op.length;
        sges[i].lkey = local_lkey;

        wrs[i] = {};
        wrs[i].opcode = IBV_WR_RDMA_READ;
        wrs[i].sg_list = &sges[i];
        wrs[i].num_sge = 1;
        wrs[i].wr.rdma.remote_addr = op.remote_addr;
        wrs[i].wr.rdma.rkey = op.remote_rkey;
        wrs[i].next = (i + 1 < n) ? &wrs[i + 1] : nullptr;
        wrs[i].send_flags = 0;
        wrs[i].wr_id = XRC_WR_ID_SYNC;
        wrs[i].qp_type.xrc.remote_srqn = remote_srqn_;
    }
    wrs[n - 1].send_flags = IBV_SEND_SIGNALED;

    ibv_send_wr* bad_wr = nullptr;
    if (ibv_post_send(qp_, &wrs[0], &bad_wr) != 0)
        throw std::runtime_error("XRC read chain post_send failed");
    op_count_ += n;
}

void XrcSendQp::post_and_poll_locked(ibv_send_wr& wr) {
    wr.send_flags |= IBV_SEND_SIGNALED;
    wr.wr_id = XRC_WR_ID_SYNC;
    wr.qp_type.xrc.remote_srqn = remote_srqn_;
    op_count_++;

    ibv_send_wr* bad_wr = nullptr;
    int ret = ibv_post_send(qp_, &wr, &bad_wr);
    if (ret) {
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "XRC ibv_post_send failed: %d opcode=%d op_count=%lu",
                 ret, wr.opcode, (unsigned long)op_count_);
        throw std::runtime_error(msg);
    }
    poll_sync(1, "post_and_poll");
}

void XrcSendQp::drain_one_signaled_locked() {
    // Drain until OUR queue advances by one signaled completion; credit others' CQEs.
    while (true) {
        ibv_wc wc = {};
        int ne = ibv_poll_cq(cq_, 1, &wc);
        if (ne == 0) continue;
        if (ne < 0)
            throw std::runtime_error("XRC poll_cq (async drain) failed");
        if (wc.status != IBV_WC_SUCCESS) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                     "XRC async WC error: status=%d vendor=%u qp_num=%u",
                     wc.status, wc.vendor_err, wc.qp_num);
            throw std::runtime_error(msg);
        }
        if (wc.wr_id == XRC_WR_ID_ASYNC) {
            if (xrc_credit_async(wc, state_))
                return;          // our own queue advanced
            // else: someone else's CQE — credited to them, keep draining.
        }
    }
}

void XrcSendQp::rdma_read(void* local_buf, uint32_t local_lkey,
                          uint64_t remote_addr, uint32_t remote_rkey,
                          uint32_t length) {
    ibv_sge sge = {};
    sge.addr = (uint64_t)local_buf;
    sge.length = length;
    sge.lkey = local_lkey;

    ibv_send_wr wr = {};
    wr.opcode = IBV_WR_RDMA_READ;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = remote_rkey;

    XrcLockGuard g(&state_->mu);
    post_and_poll_locked(wr);
}

void XrcSendQp::rdma_read_batch2(
    void* buf1, uint32_t lkey1, uint64_t addr1, uint32_t rkey1, uint32_t len1,
    void* buf2, uint32_t lkey2, uint64_t addr2, uint32_t rkey2, uint32_t len2) {

    ibv_sge sge1 = {};
    sge1.addr = (uint64_t)buf1; sge1.length = len1; sge1.lkey = lkey1;
    ibv_sge sge2 = {};
    sge2.addr = (uint64_t)buf2; sge2.length = len2; sge2.lkey = lkey2;

    ibv_send_wr wr2 = {};
    wr2.opcode = IBV_WR_RDMA_READ;
    wr2.sg_list = &sge2;
    wr2.num_sge = 1;
    wr2.wr.rdma.remote_addr = addr2;
    wr2.wr.rdma.rkey = rkey2;
    wr2.next = nullptr;
    wr2.send_flags = IBV_SEND_SIGNALED;
    wr2.wr_id = XRC_WR_ID_SYNC;
    wr2.qp_type.xrc.remote_srqn = remote_srqn_;

    ibv_send_wr wr1 = {};
    wr1.opcode = IBV_WR_RDMA_READ;
    wr1.sg_list = &sge1;
    wr1.num_sge = 1;
    wr1.wr.rdma.remote_addr = addr1;
    wr1.wr.rdma.rkey = rkey1;
    wr1.next = &wr2;
    wr1.send_flags = 0;
    wr1.wr_id = XRC_WR_ID_SYNC;
    wr1.qp_type.xrc.remote_srqn = remote_srqn_;

    XrcLockGuard g(&state_->mu);
    ibv_send_wr* bad_wr = nullptr;
    if (ibv_post_send(qp_, &wr1, &bad_wr) != 0)
        throw std::runtime_error("XRC batch2 post_send failed");
    poll_sync(1, "batch2");
}

void XrcSendQp::rdma_read_multi(const RdmaReadOp* ops, int count,
                                uint32_t local_lkey) {
    if (count <= 0) return;
    XrcLockGuard g(&state_->mu);
    for (int base = 0; base < count; base += RDMA_READ_CHAIN_MAX) {
        post_read_chain_locked(ops + base,
                               std::min(RDMA_READ_CHAIN_MAX, count - base),
                               local_lkey);
        poll_sync(1, "read_multi");
    }
}

int XrcSendQp::rdma_read_multi_post(const RdmaReadOp* ops, int count,
                                    uint32_t local_lkey) {
    if (count <= 0) return 0;
    if (count > RDMA_READ_CHAIN_MAX)
        throw std::invalid_argument(
            "XrcSendQp::rdma_read_multi_post: chain longer than "
            "RDMA_READ_CHAIN_MAX");
    XrcLockGuard g(&state_->mu);
    post_read_chain_locked(ops, count, local_lkey);
    return 1;
}

// Unlocked; collects any QP's SYNC completions on this CQ.
void XrcSendQp::rdma_read_multi_wait(int pending) {
    poll_sync(pending, "read_multi_wait");
}

void XrcSendQp::rdma_write(uint64_t remote_addr, uint32_t remote_rkey,
                            const void* local_buf, uint32_t local_lkey,
                            uint32_t length) {
    ibv_sge sge = {};
    sge.addr = (uint64_t)local_buf;
    sge.length = length;
    sge.lkey = local_lkey;

    ibv_send_wr wr = {};
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = remote_rkey;

    XrcLockGuard g(&state_->mu);
    post_and_poll_locked(wr);
}

void XrcSendQp::rdma_write_async_inline(uint64_t remote_addr,
                                         uint32_t remote_rkey,
                                         const void* local_buf,
                                         uint32_t length) {
    XrcLockGuard g(&state_->mu);
    auto used_now = [&]() {
        return state_->inflight_unsignaled.load(std::memory_order_relaxed)
             + state_->outstanding_signaled.load(std::memory_order_relaxed) * SIGNAL_EVERY;
    };
    while (used_now() + SIGNAL_EVERY >= MAX_SEND_WR) {
        drain_one_signaled_locked();
    }

    ibv_sge sge = {};
    sge.addr = (uint64_t)local_buf;
    sge.length = length;
    sge.lkey = 0;

    ibv_send_wr wr = {};
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = remote_rkey;
    wr.wr_id = XRC_WR_ID_ASYNC;
    wr.qp_type.xrc.remote_srqn = remote_srqn_;

    int infl = state_->inflight_unsignaled.fetch_add(1, std::memory_order_relaxed) + 1;
    op_count_++;
    bool signal = (infl >= SIGNAL_EVERY);
    wr.send_flags = IBV_SEND_INLINE | (signal ? IBV_SEND_SIGNALED : 0);

    ibv_send_wr* bad_wr = nullptr;
    if (ibv_post_send(qp_, &wr, &bad_wr) != 0)
        throw std::runtime_error("XRC async inline post_send failed");

    if (signal) {
        state_->inflight_unsignaled.store(0, std::memory_order_relaxed);
        state_->outstanding_signaled.fetch_add(1, std::memory_order_relaxed);
    }
}

void XrcSendQp::rdma_write_chain_inline(const InlineWriteOp* ops, int count) {
    if (count <= 0) return;
    if (count == 1) {
        rdma_write_async_inline(ops[0].remote_addr, ops[0].remote_rkey,
                                ops[0].local_buf, ops[0].length);
        return;
    }

    XrcLockGuard g(&state_->mu);
    auto used_now = [&]() {
        return state_->inflight_unsignaled.load(std::memory_order_relaxed)
             + state_->outstanding_signaled.load(std::memory_order_relaxed) * SIGNAL_EVERY;
    };
    int posted = 0;
    while (posted < count) {
        while (used_now() + SIGNAL_EVERY >= MAX_SEND_WR) {
            drain_one_signaled_locked();
        }

        int remaining = count - posted;
        int chain_len = std::min(remaining, SIGNAL_EVERY);

        ibv_sge sges[SIGNAL_EVERY];
        ibv_send_wr wrs[SIGNAL_EVERY];
        for (int i = 0; i < chain_len; i++) {
            const InlineWriteOp& op = ops[posted + i];
            sges[i] = {};
            sges[i].addr = (uint64_t)op.local_buf;
            sges[i].length = op.length;
            sges[i].lkey = 0;
            wrs[i] = {};
            wrs[i].opcode = IBV_WR_RDMA_WRITE;
            wrs[i].sg_list = &sges[i];
            wrs[i].num_sge = 1;
            wrs[i].wr.rdma.remote_addr = op.remote_addr;
            wrs[i].wr.rdma.rkey = op.remote_rkey;
            wrs[i].next = (i + 1 < chain_len) ? &wrs[i + 1] : nullptr;
            wrs[i].send_flags = IBV_SEND_INLINE;
            wrs[i].wr_id = XRC_WR_ID_ASYNC;
            wrs[i].qp_type.xrc.remote_srqn = remote_srqn_;
        }
        wrs[chain_len - 1].send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;

        ibv_send_wr* bad_wr = nullptr;
        if (ibv_post_send(qp_, &wrs[0], &bad_wr) != 0)
            throw std::runtime_error("XRC chain inline post_send failed");

        state_->inflight_unsignaled.store(0, std::memory_order_relaxed);
        state_->outstanding_signaled.fetch_add(1, std::memory_order_relaxed);
        op_count_ += chain_len;
        posted += chain_len;
    }
}

bool XrcSendQp::broken() {
    ibv_qp_attr a{};
    ibv_qp_init_attr ia{};
    if (ibv_query_qp(qp_, &a, IBV_QP_STATE, &ia) != 0) return true;
    return a.qp_state == IBV_QPS_ERR || a.qp_state == IBV_QPS_SQE;
}

void XrcSendQp::flush() {
    XrcLockGuard g(&state_->mu);
    while (state_->outstanding_signaled.load(std::memory_order_relaxed) > 0)
        drain_one_signaled_locked();
}

uint64_t XrcSendQp::rdma_fetch_add(uint64_t remote_addr, uint32_t remote_rkey,
                                    uint64_t add_val,
                                    void* local_result_buf, uint32_t local_lkey) {
    ibv_sge sge = {};
    sge.addr = (uint64_t)local_result_buf;
    sge.length = 8;
    sge.lkey = local_lkey;

    ibv_send_wr wr = {};
    wr.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.wr.atomic.remote_addr = remote_addr;
    wr.wr.atomic.rkey = remote_rkey;
    wr.wr.atomic.compare_add = add_val;

    XrcLockGuard g(&state_->mu);
    post_and_poll_locked(wr);
    return *(uint64_t*)local_result_buf;
}

} // namespace shmradix
