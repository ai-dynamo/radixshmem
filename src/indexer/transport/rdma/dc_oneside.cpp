// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dc_oneside.hpp"

#include "detail.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>
#include <poll.h>
#include <string>
#include <stdexcept>
#include <unordered_map>

namespace radixshmem {

struct DciQp {
    ibv_qp*        qp   = nullptr;
    ibv_qp_ex*     qpx  = nullptr;
    mlx5dv_qp_ex*  mqpx = nullptr;
};
static DciQp dc_create_dci_qp(RdmaContext& ctx, ibv_cq* cq,
                              uint64_t send_ops_flags, const char* who) {
    ibv_qp_init_attr_ex attr{};
    attr.qp_type   = IBV_QPT_DRIVER;
    attr.send_cq   = cq;
    attr.recv_cq   = cq;
    attr.pd        = ctx.pd();
    attr.cap.max_send_wr  = 64;
    attr.cap.max_send_sge = 1;
    attr.cap.max_inline_data = 64;   // RHT write plane: 16B inline cell writes
    attr.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
    attr.send_ops_flags = send_ops_flags;

    mlx5dv_qp_init_attr dv{};
    dv.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_DC;
    dv.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCI;

    DciQp out;
    out.qp = mlx5dv_create_qp(ctx.ctx(), &attr, &dv);
    if (!out.qp)
        throw std::runtime_error(std::string(who) +
                                 ": mlx5dv_create_qp(DCI) failed");
    out.qpx  = ibv_qp_to_qp_ex(out.qp);
    out.mqpx = out.qpx ? mlx5dv_qp_ex_from_ibv_qp_ex(out.qpx) : nullptr;
    if (!out.qpx || !out.mqpx) {
        ibv_destroy_qp(out.qp);
        throw std::runtime_error(std::string(who) +
                                 ": qp_ex/mlx5dv_qp_ex unavailable");
    }
    return out;
}

static void dc_dci_to_rts(ibv_qp* qp, RdmaContext& ctx,
                          uint8_t timeout, uint8_t retry_cnt, uint8_t rnr_retry,
                          const char* who) {
    int port = ctx.port();
    {
        ibv_qp_attr a{};
        a.qp_state = IBV_QPS_RESET;
        if (ibv_modify_qp(qp, &a, IBV_QP_STATE) != 0)
            throw std::runtime_error(std::string(who) +
                                     ": DCI modify RESET failed");
    }
    {
        ibv_qp_attr a{};
        a.qp_state   = IBV_QPS_INIT;
        a.pkey_index = 0;
        a.port_num   = port;
        if (ibv_modify_qp(qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT) != 0)
            throw std::runtime_error(std::string(who) +
                                     ": DCI modify INIT failed");
    }
    {
        ibv_qp_attr a{};
        a.qp_state = IBV_QPS_RTR;
        a.path_mtu = IBV_MTU_1024;
        a.ah_attr.is_global      = 1;
        a.ah_attr.port_num       = port;
        a.ah_attr.sl             = ctx.sl();
        a.ah_attr.grh.sgid_index = ctx.gid_index();
        a.ah_attr.grh.hop_limit  = 1;
        a.ah_attr.grh.traffic_class = ctx.traffic_class();
        a.ah_attr.grh.dgid       = ctx.gid();   // placeholder; per-WR AH overrides
        if (ibv_modify_qp(qp, &a,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU) != 0)
            throw std::runtime_error(std::string(who) +
                                     ": DCI modify RTR failed");
    }
    {
        ibv_qp_attr a{};
        a.qp_state      = IBV_QPS_RTS;
        a.timeout       = timeout;
        a.retry_cnt     = retry_cnt;
        a.rnr_retry     = rnr_retry;
        a.sq_psn        = 0;
        a.max_rd_atomic = 16;
        if (ibv_modify_qp(qp, &a,
                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0)
            throw std::runtime_error(std::string(who) +
                                     ": DCI modify RTS failed");
    }
}

// Create a DCT bound to (cq, srq, key), drive INIT->RTR; dct_num valid only AFTER transitions.
static ibv_qp* dc_create_dct_qp(RdmaContext& ctx, ibv_cq* cq, ibv_srq* srq,
                                uint64_t dct_access_key, unsigned access_flags,
                                const char* who) {
    ibv_qp_init_attr_ex attr{};
    attr.qp_type   = IBV_QPT_DRIVER;
    attr.send_cq   = cq;
    attr.recv_cq   = cq;
    attr.pd        = ctx.pd();
    attr.srq       = srq;          // a DCT must bind a *basic* SRQ
    attr.comp_mask = IBV_QP_INIT_ATTR_PD;

    mlx5dv_qp_init_attr dv{};
    dv.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_DC;
    dv.dc_init_attr.dc_type        = MLX5DV_DCTYPE_DCT;
    dv.dc_init_attr.dct_access_key = dct_access_key;

    ibv_qp* qp = mlx5dv_create_qp(ctx.ctx(), &attr, &dv);
    if (!qp)
        throw std::runtime_error(std::string(who) +
                                 ": mlx5dv_create_qp(DCT) failed");
    int port = ctx.port();
    {
        ibv_qp_attr a{};
        a.qp_state        = IBV_QPS_INIT;
        a.pkey_index      = 0;
        a.port_num        = port;
        a.qp_access_flags = access_flags;
        if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
            ibv_destroy_qp(qp);
            throw std::runtime_error(std::string(who) +
                                     ": DCT modify INIT failed");
        }
    }
    {
        ibv_qp_attr a{};
        a.qp_state        = IBV_QPS_RTR;
        a.path_mtu        = IBV_MTU_1024;
        a.min_rnr_timer   = 12;
        a.ah_attr.is_global       = 1;
        a.ah_attr.port_num        = port;
        a.ah_attr.grh.hop_limit   = 1;
        a.ah_attr.grh.sgid_index  = ctx.gid_index();
        a.ah_attr.grh.dgid        = ctx.gid();
        a.ah_attr.sl              = ctx.sl();
        a.ah_attr.grh.traffic_class = ctx.traffic_class();
        if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_AV |
                IBV_QP_PATH_MTU | IBV_QP_MIN_RNR_TIMER) != 0) {
            ibv_destroy_qp(qp);
            throw std::runtime_error(std::string(who) +
                                     ": DCT modify RTR failed");
        }
    }
    return qp;
}

// Data-plane flavor: a dead target errors the DCI within the index-plane
// budget (detail.hpp); the DCI is reset to RTS after every error. Its own CQ
// keeps its completions apart from the XRC endpoints'.
DcInitiator::DcInitiator(RdmaContext& ctx)
    : ctx_(&ctx), timeout_(qp_ack_timeout()), retry_cnt_(qp_retry_cnt()), rnr_retry_(7) {
    cq_ = ibv_create_cq(ctx.ctx(), 256, nullptr, nullptr, 0);
    if (!cq_)
        throw std::runtime_error("DcInitiator: ibv_create_cq failed");
    try {
        DciQp h = dc_create_dci_qp(ctx, cq_,
                                   IBV_QP_EX_WITH_RDMA_READ |
                                   IBV_QP_EX_WITH_ATOMIC_FETCH_AND_ADD |
                                   IBV_QP_EX_WITH_RDMA_WRITE,   // RHT write plane
                                   who_);
        qp_ = h.qp; qpx_ = h.qpx; mqpx_ = h.mqpx;
        reset_to_rts();
    } catch (...) {
        if (qp_) { ibv_destroy_qp(qp_); qp_ = nullptr; }
        ibv_destroy_cq(cq_); cq_ = nullptr;   // throwing ctor runs no dtor
        throw;
    }
}

DcInitiator::DcInitiator(RdmaContext& ctx, uint64_t dc_key)
    : ctx_(&ctx), dc_key_(dc_key),
      timeout_(14), retry_cnt_(2), rnr_retry_(6), who_("DcInitiator(ctrl)") {
    cq_ = ibv_create_cq(ctx.ctx(), 64, nullptr, nullptr, 0);
    if (!cq_)
        throw std::runtime_error("DcInitiator(ctrl): ibv_create_cq failed");
    try {
        DciQp h = dc_create_dci_qp(ctx, cq_, IBV_QP_EX_WITH_SEND, who_);
        qp_ = h.qp; qpx_ = h.qpx; mqpx_ = h.mqpx;
        reset_to_rts();
    } catch (...) {
        if (qp_) { ibv_destroy_qp(qp_); qp_ = nullptr; }
        ibv_destroy_cq(cq_); cq_ = nullptr;   // throwing ctor runs no dtor
        throw;
    }
}

DcInitiator::~DcInitiator() {
    if (qp_) ibv_destroy_qp(qp_);
    if (cq_) ibv_destroy_cq(cq_);
}

// Bring the DCI RESET->INIT->RTR->RTS; also recovers from ERR (DC errors poison
// the QP). Completions of the failed WRs are dropped with it.
void DcInitiator::reset_to_rts() {
    dc_dci_to_rts(qp_, *ctx_, timeout_, retry_cnt_, rnr_retry_, who_);
    ibv_wc wc[16];
    while (ibv_poll_cq(cq_, 16, wc) > 0) {}
    unsignaled_ = 0;
}

void DcInitiator::add_owner(DcOwner* o) {
    std::lock_guard<std::mutex> g(mu_);
    owners_.insert(o);
}

void DcInitiator::remove_owner(DcOwner* o) {
    std::lock_guard<std::mutex> g(mu_);
    owners_.erase(o);
}

void DcInitiator::flag(uint64_t wr_id) {
    auto* o = reinterpret_cast<DcOwner*>((uintptr_t)wr_id);
    if (!(wr_id & kSignaled) && owners_.count(o))
        o->failed.store(true, std::memory_order_release);
}

void DcInitiator::await(uint64_t wrid, const DcTargetInfo& dst, const char* what) {
    char earlier[160] = "";   // an unsignaled write before this op failed
    for (;;) {
        ibv_wc wc{};
        int ne = ibv_poll_cq(cq_, 1, &wc);
        if (ne == 0) continue;
        if (ne < 0) {
            reset_to_rts();
            throw std::runtime_error(std::string(who_) + "::" + what + ": ibv_poll_cq failed");
        }
        if (wc.wr_id != wrid) {
            if (wc.status != IBV_WC_SUCCESS && !earlier[0]) {
                flag(wc.wr_id);
                snprintf(earlier, sizeof(earlier),
                         "%s::%s: an earlier write to another target failed: WC error "
                         "status=%d vendor=%u", who_, what, wc.status, wc.vendor_err);
            }
            continue;
        }
        if (wc.status == IBV_WC_SUCCESS && !earlier[0]) return;
        reset_to_rts();
        if (earlier[0] && (wc.status == IBV_WC_SUCCESS || wc.status == IBV_WC_WR_FLUSH_ERR))
            throw std::runtime_error(earlier);
        if (dst.owner) dst.owner->failed.store(true, std::memory_order_release);
        char m[160];
        snprintf(m, sizeof(m), "%s::%s WC error status=%d vendor=%u",
                 who_, what, wc.status, wc.vendor_err);
        throw std::runtime_error(m);
    }
}

void DcInitiator::rdma_read(const DcTargetInfo& dst,
                            void* local_buf, uint32_t local_lkey,
                            uint64_t remote_addr, uint32_t remote_rkey,
                            uint32_t length) {
    std::lock_guard<std::mutex> g(mu_);
    const uint64_t wrid = next_wrid();
    auto* mqp = static_cast<mlx5dv_qp_ex*>(mqpx_);
    ibv_wr_start(qpx_);
    qpx_->wr_id    = wrid;
    qpx_->wr_flags = IBV_SEND_SIGNALED;
    ibv_wr_rdma_read(qpx_, remote_rkey, remote_addr);
    ibv_wr_set_sge(qpx_, local_lkey, (uintptr_t)local_buf, length);
    mlx5dv_wr_set_dc_addr(mqp, dst.ah, dst.dct_num, dst.dc_key);
    if (ibv_wr_complete(qpx_) != 0) {
        reset_to_rts();
        throw std::runtime_error("DcInitiator::rdma_read post failed");
    }
    unsignaled_ = 0;
    await(wrid, dst, "rdma_read");
}

uint64_t DcInitiator::rdma_fetch_add(const DcTargetInfo& dst,
                                     uint64_t remote_addr, uint32_t remote_rkey,
                                     uint64_t add_val,
                                     void* local_result_buf, uint32_t local_lkey) {
    std::lock_guard<std::mutex> g(mu_);
    const uint64_t wrid = next_wrid();
    auto* mqp = static_cast<mlx5dv_qp_ex*>(mqpx_);
    ibv_wr_start(qpx_);
    qpx_->wr_id    = wrid;
    qpx_->wr_flags = IBV_SEND_SIGNALED;
    ibv_wr_atomic_fetch_add(qpx_, remote_rkey, remote_addr, add_val);
    ibv_wr_set_sge(qpx_, local_lkey, (uintptr_t)local_result_buf, 8);
    mlx5dv_wr_set_dc_addr(mqp, dst.ah, dst.dct_num, dst.dc_key);
    if (ibv_wr_complete(qpx_) != 0) {
        reset_to_rts();
        throw std::runtime_error("DcInitiator::rdma_fetch_add post failed");
    }
    unsignaled_ = 0;
    await(wrid, dst, "fetch_add");
    return *(uint64_t*)local_result_buf;
}

void DcInitiator::rdma_read_multi(const DcTargetInfo& dst,
                                  const RdmaReadOp* ops, int count,
                                  uint32_t local_lkey) {
    if (count <= 0) return;
    std::lock_guard<std::mutex> g(mu_);
    auto* mqp = static_cast<mlx5dv_qp_ex*>(mqpx_);

    // Bounded batches: a chain must not exceed the DCI send-queue depth; one signaled CQE per batch.
    static constexpr int BATCH = 16;
    for (int base = 0; base < count; base += BATCH) {
        int end = std::min(base + BATCH, count);
        int n = end - base;
        const uint64_t wrid = next_wrid();   // on every WR: a failed read reports it

        ibv_wr_start(qpx_);
        for (int i = 0; i < n; i++) {
            const auto& op = ops[base + i];
            qpx_->wr_id    = wrid;
            qpx_->wr_flags = (i == n - 1) ? IBV_SEND_SIGNALED : 0;
            ibv_wr_rdma_read(qpx_, op.remote_rkey, op.remote_addr);
            ibv_wr_set_sge(qpx_, local_lkey, (uintptr_t)op.local_buf, op.length);
            mlx5dv_wr_set_dc_addr(mqp, dst.ah, dst.dct_num, dst.dc_key);
        }
        if (ibv_wr_complete(qpx_) != 0) {
            reset_to_rts();
            throw std::runtime_error("DcInitiator::rdma_read_multi post failed");
        }
        unsignaled_ = 0;
        await(wrid, dst, "read_multi");
    }
}

void DcInitiator::rdma_write_async_inline(const DcTargetInfo& dst,
                                          uint64_t remote_addr, uint32_t remote_rkey,
                                          const void* local_buf, uint32_t length) {
    std::lock_guard<std::mutex> g(mu_);
    const bool signal = ++unsignaled_ >= kSignalEvery;
    const uint64_t wrid = signal ? next_wrid() : (uint64_t)(uintptr_t)dst.owner;
    auto* mqp = static_cast<mlx5dv_qp_ex*>(mqpx_);
    ibv_wr_start(qpx_);
    qpx_->wr_id    = wrid;
    qpx_->wr_flags = signal ? IBV_SEND_SIGNALED : 0;
    ibv_wr_rdma_write(qpx_, remote_rkey, remote_addr);
    ibv_wr_set_inline_data(qpx_, (void*)local_buf, length);   // copied into WQE
    mlx5dv_wr_set_dc_addr(mqp, dst.ah, dst.dct_num, dst.dc_key);
    if (ibv_wr_complete(qpx_) != 0) {
        reset_to_rts();
        throw std::runtime_error("DcInitiator::rdma_write_async_inline post failed");
    }
    if (signal) {
        unsignaled_ = 0;
        await(wrid, dst, "rdma_write");
    }
}

void DcInitiator::fence(const DcTargetInfo& dst, uint64_t remote_addr,
                        uint32_t remote_rkey, const void* value, uint32_t length) {
    std::lock_guard<std::mutex> g(mu_);
    const uint64_t wrid = next_wrid();
    auto* mqp = static_cast<mlx5dv_qp_ex*>(mqpx_);
    ibv_wr_start(qpx_);
    qpx_->wr_id    = wrid;
    qpx_->wr_flags = IBV_SEND_SIGNALED;
    ibv_wr_rdma_write(qpx_, remote_rkey, remote_addr);
    ibv_wr_set_inline_data(qpx_, (void*)value, length);
    mlx5dv_wr_set_dc_addr(mqp, dst.ah, dst.dct_num, dst.dc_key);
    if (ibv_wr_complete(qpx_) != 0) {
        reset_to_rts();
        throw std::runtime_error("DcInitiator::flush fence post failed");
    }
    unsignaled_ = 0;
    await(wrid, dst, "flush");
}

void DcReader::rdma_write_async_inline(uint64_t remote_addr, uint32_t remote_rkey,
                                       const void* local_buf, uint32_t length) {
    dci_->rdma_write_async_inline(dst_, remote_addr, remote_rkey, local_buf, length);
    last_.addr = remote_addr;
    last_.rkey = remote_rkey;
    last_.len  = length < sizeof(last_.value) ? length : (uint32_t)sizeof(last_.value);
    memcpy(last_.value, local_buf, last_.len);
    last_.dirty = true;
}

void DcReader::flush() {
    if (!last_.dirty) return;
    last_.dirty = false;
    dci_->fence(dst_, last_.addr, last_.rkey, last_.value, last_.len);
}

void DcInitiator::send(ibv_ah* ah, uint32_t dctn,
                       const void* buf, uint32_t lkey, uint32_t len) {
    std::lock_guard<std::mutex> g(mu_);
    const uint64_t wrid = next_wrid();
    auto* mqp = static_cast<mlx5dv_qp_ex*>(mqpx_);
    ibv_wr_start(qpx_);
    qpx_->wr_id    = wrid;
    qpx_->wr_flags = IBV_SEND_SIGNALED;
    ibv_wr_send(qpx_);
    ibv_wr_set_sge(qpx_, lkey, (uintptr_t)buf, len);
    mlx5dv_wr_set_dc_addr(mqp, ah, dctn, dc_key_);
    if (ibv_wr_complete(qpx_) != 0) {
        reset_to_rts();
        throw std::runtime_error("DcInitiator::send: post failed");
    }
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    for (;;) {
        ibv_wc wc{};
        int ne = ibv_poll_cq(cq_, 1, &wc);
        if (ne < 0) {
            reset_to_rts();
            throw std::runtime_error("DcInitiator::send: ibv_poll_cq failed");
        }
        if (ne == 0) {
            if (std::chrono::steady_clock::now() >= deadline) {
                reset_to_rts();
                throw std::runtime_error(
                    "DcInitiator::send: completion timed out");
            }
            continue;
        }
        if (wc.wr_id != wrid) continue;   // stale flushed completion
        if (wc.status != IBV_WC_SUCCESS) {
            reset_to_rts();   // an errored SEND poisons the DCI
            char m[128];
            snprintf(m, sizeof(m),
                     "DcInitiator::send: WC error status=%d vendor=%u",
                     wc.status, wc.vendor_err);
            throw std::runtime_error(m);
        }
        return;
    }
}

// ==================== DC addressing helper ====================

ibv_ah* dc_create_ah(RdmaContext& ctx, const ibv_gid& dgid, uint16_t dlid,
                     uint8_t link_layer) {
    ibv_ah_attr a{};
    a.is_global = 1;
    a.port_num  = ctx.port();
    a.sl        = ctx.sl();
    a.grh.hop_limit     = 1;
    a.grh.sgid_index    = ctx.gid_index();
    a.grh.traffic_class = ctx.traffic_class();
    a.grh.dgid          = dgid;
    a.dlid              = dlid;
    if (link_layer == IBV_LINK_LAYER_INFINIBAND)
        a.is_global = 0;
    return ibv_create_ah(ctx.pd(), &a);
}

// ==================== DcTarget (receive half + DCTs) ====================

DcTarget::DcTarget(RdmaContext& ctx, uint64_t dc_key, int srq_max_wr)
    : ctx_(&ctx), dc_key_(dc_key) {
    int cq_depth = srq_max_wr + 16;
    if (cq_depth < 64) cq_depth = 64;
    cq_ = ibv_create_cq(ctx.ctx(), cq_depth, nullptr, nullptr, 0);
    if (!cq_) throw std::runtime_error("DcTarget: ibv_create_cq failed");

    ibv_srq_init_attr sattr{};
    sattr.attr.max_wr  = (uint32_t)(srq_max_wr < 1 ? 1 : srq_max_wr);
    sattr.attr.max_sge = 1;
    srq_ = ibv_create_srq(ctx.pd(), &sattr);
    if (!srq_) {
        ibv_destroy_cq(cq_); cq_ = nullptr;
        throw std::runtime_error("DcTarget: ibv_create_srq failed");
    }
}

DcTarget::~DcTarget() {
    for (auto& mb : dcts_)                 // DCTs before their SRQ/CQ
        if (mb.second) ibv_destroy_qp(mb.second);
    if (srq_) ibv_destroy_srq(srq_);
    if (cq_)  ibv_destroy_cq(cq_);
}

uint32_t DcTarget::open_dct() {
    ibv_qp* qp = dc_create_dct_qp(*ctx_, cq_, srq_, dc_key_,
                                  IBV_ACCESS_REMOTE_READ |
                                  IBV_ACCESS_REMOTE_ATOMIC |
                                  IBV_ACCESS_REMOTE_WRITE,   // RHT write plane
                                  "DcTarget");
    uint32_t dctn = qp->qp_num;   // valid only after INIT->RTR (helper did both)
    dcts_.emplace_back(dctn, qp);
    return dctn;
}

void DcTarget::close_dct(uint32_t dctn) {
    try { drain_all(); } catch (const std::exception&) {}
    for (auto it = recv_ready_.begin(); it != recv_ready_.end();) {
        if (it->dctn == dctn) {
            if (it->wr_id >= POOL_WR_BASE &&
                it->wr_id - POOL_WR_BASE < pool_.size()) {
                try { repost_recv(it->wr_id); } catch (const std::exception&) {}
            }
            it = recv_ready_.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = dcts_.begin(); it != dcts_.end(); ++it) {
        if (it->first == dctn) {
            if (it->second) ibv_destroy_qp(it->second);
            dcts_.erase(it);
            return;
        }
    }
}

void DcTarget::post_recv(void* buf, uint32_t lkey, uint32_t len,
                            uint64_t wr_id) {
    ibv_sge sge{};
    sge.addr   = (uint64_t)buf;
    sge.length = len;
    sge.lkey   = lkey;
    ibv_recv_wr wr{};
    wr.wr_id   = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    ibv_recv_wr* bad = nullptr;
    if (ibv_post_srq_recv(srq_, &wr, &bad) != 0)
        throw std::runtime_error("DcTarget: ibv_post_srq_recv failed");
}

void DcTarget::provision_recv_pool(int count, uint32_t buf_bytes) {
    // Incremental-post + graceful-cap (queried SRQ max_wr is unreliable as a post limit).
    pool_.resize(count);
    int posted = 0;
    for (int i = 0; i < count; i++) {
        pool_[i].mem.resize(buf_bytes);
        pool_[i].mr = RdmaMemoryRegion(ctx_->pd(), pool_[i].mem.data(),
                                       buf_bytes, IBV_ACCESS_LOCAL_WRITE);
        try {
            post_recv(pool_[i].mem.data(), pool_[i].mr.lkey(), buf_bytes,
                      POOL_WR_BASE + i);
        } catch (const std::exception&) {
            fprintf(stderr, "[DcTarget] recv pool capped at %d of %d "
                    "(SRQ full)\n", i, count);
            break;
        }
        posted = i + 1;
    }
    if (posted < count) pool_.resize(posted);
}

void DcTarget::repost_recv(uint64_t wr_id) {
    size_t i = wr_id - POOL_WR_BASE;
    if (wr_id < POOL_WR_BASE || i >= pool_.size())
        throw std::runtime_error("DcTarget::repost_recv: bad wr_id");
    post_recv(pool_[i].mem.data(), pool_[i].mr.lkey(),
              (uint32_t)pool_[i].mem.size(), wr_id);
}

int DcTarget::drain_all() {
    int n = 0;
    for (;;) {
        ibv_wc wc{};
        int ne = ibv_poll_cq(cq_, 1, &wc);
        if (ne == 0) break;
        if (ne < 0)
            throw std::runtime_error("DcTarget: ibv_poll_cq failed");
        if (wc.status != IBV_WC_SUCCESS) {
            if (wc_err_logged_ < 8) {
                fprintf(stderr,
                        "[DcTarget] swallowing WC error: status=%d opcode=%d "
                        "vendor=%u qp_num=%u wr_id=%lu%s\n",
                        wc.status, wc.opcode, wc.vendor_err, wc.qp_num,
                        (unsigned long)wc.wr_id,
                        ++wc_err_logged_ == 8 ? " (suppressing further)" : "");
            }
            if (wc.wr_id >= POOL_WR_BASE &&
                wc.wr_id - POOL_WR_BASE < pool_.size()) {
                repost_recv(wc.wr_id);      // errored RECV consumed a pool buffer
            }
            continue;
        }
        if (wc.opcode == IBV_WC_RECV) {
            DcRecvCompletion c;
            c.ok       = true;
            c.dctn     = wc.qp_num;         // receiving DCT (demux key)
            c.byte_len = wc.byte_len;
            c.wr_id    = wc.wr_id;
            if (wc.wr_id >= POOL_WR_BASE &&
                wc.wr_id - POOL_WR_BASE < pool_.size())
                c.data = pool_[wc.wr_id - POOL_WR_BASE].mem.data();
            recv_ready_.push_back(c);
        }
        n++;
    }
    return n;
}

bool DcTarget::pump(int timeout_ms) {
    if (timeout_ms <= 0)
        return drain_all() > 0;
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(timeout_ms);
    for (;;) {
        if (drain_all() > 0) return true;
        if (std::chrono::steady_clock::now() >= deadline) return false;
    }
}

DcRecvCompletion DcTarget::poll(int timeout_ms, uint32_t want_dctn) {
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(timeout_ms);
    for (;;) {
        for (auto it = recv_ready_.begin(); it != recv_ready_.end(); ++it) {
            if (want_dctn == 0 || it->dctn == want_dctn) {
                DcRecvCompletion c = *it;
                recv_ready_.erase(it);
                return c;
            }
        }
        int left = (int)std::chrono::duration_cast<std::chrono::milliseconds>(
                       deadline - std::chrono::steady_clock::now()).count();
        if (left < 0) left = 0;
        bool pumped = pump(left);
        if (left <= 0 && !pumped) return DcRecvCompletion{};   // timeout
        if (left <= 0 && pumped) {
            for (auto it = recv_ready_.begin(); it != recv_ready_.end(); ++it) {
                if (want_dctn == 0 || it->dctn == want_dctn) {
                    DcRecvCompletion c = *it;
                    recv_ready_.erase(it);
                    return c;
                }
            }
            return DcRecvCompletion{};
        }
    }
}

} // namespace radixshmem
