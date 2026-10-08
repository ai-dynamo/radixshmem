// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "rc_endpoint.hpp"

#include "detail.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <infiniband/verbs.h>
#include <stdexcept>

namespace radixshmem {

RdmaEndpoint::RdmaEndpoint(RdmaContext& ctx) : ctx_(&ctx) {
    cq_ = ibv_create_cq(ctx.ctx(), 128, nullptr, nullptr, 0);
    if (!cq_)
        throw std::runtime_error("ibv_create_cq (per-endpoint) failed");

    ibv_qp_init_attr qp_init = {};
    qp_init.send_cq = cq_;
    qp_init.recv_cq = cq_;
    qp_init.qp_type = IBV_QPT_RC;
    qp_init.cap.max_send_wr = 64;
    qp_init.cap.max_recv_wr = 1;
    qp_init.cap.max_send_sge = 1;
    qp_init.cap.max_recv_sge = 1;
    // Enable IBV_SEND_INLINE for small payloads (64B covers 8-byte RHT writes).
    qp_init.cap.max_inline_data = 64;

    qp_ = ibv_create_qp(ctx.pd(), &qp_init);
    if (!qp_) {
        ibv_destroy_cq(cq_);
        cq_ = nullptr;
        throw std::runtime_error("ibv_create_qp failed");
    }
}

RdmaEndpoint::~RdmaEndpoint() {
    if (qp_) ibv_destroy_qp(qp_);
    if (cq_) ibv_destroy_cq(cq_);
}

RdmaEndpoint::RdmaEndpoint(RdmaEndpoint&& o) noexcept
    : qp_(o.qp_), cq_(o.cq_), ctx_(o.ctx_) {
    o.qp_ = nullptr;
    o.cq_ = nullptr;
}

RdmaEndpoint& RdmaEndpoint::operator=(RdmaEndpoint&& o) noexcept {
    if (this != &o) {
        if (qp_) ibv_destroy_qp(qp_);
        if (cq_) ibv_destroy_cq(cq_);
        qp_ = o.qp_;
        cq_ = o.cq_;
        ctx_ = o.ctx_;
        o.qp_ = nullptr;
        o.cq_ = nullptr;
    }
    return *this;
}

uint32_t RdmaEndpoint::qp_num() const { return qp_->qp_num; }

RdmaPeerInfo RdmaEndpoint::peer_info() const {
    RdmaPeerInfo info;
    info.qp_num = qp_->qp_num;
    info.gid = ctx_->gid();
    info.gid_index = ctx_->gid_index();
    ibv_port_attr pa = {};
    if (ibv_query_port(ctx_->ctx(), ctx_->port(), &pa) == 0) {
        info.lid = pa.lid;
        info.link_layer = pa.link_layer;
    }
    return info;
}

void RdmaEndpoint::connect(const RdmaPeerInfo& remote) {
    // RESET → INIT
    {
        ibv_qp_attr attr = {};
        attr.qp_state = IBV_QPS_INIT;
        attr.port_num = ctx_->port();
        attr.pkey_index = 0;
        attr.qp_access_flags = IBV_ACCESS_REMOTE_READ |
                               IBV_ACCESS_REMOTE_WRITE |
                               IBV_ACCESS_REMOTE_ATOMIC;
        int ret = ibv_modify_qp(qp_, &attr,
                                IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                                    IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
        if (ret)
            throw std::runtime_error("ibv_modify_qp INIT failed: " +
                                     std::to_string(ret));
    }

    // INIT → RTR
    {
        ibv_qp_attr attr = {};
        attr.qp_state = IBV_QPS_RTR;
        attr.path_mtu = IBV_MTU_4096;
        attr.dest_qp_num = remote.qp_num;
        attr.rq_psn = 0;
        attr.max_dest_rd_atomic = 16;
        attr.min_rnr_timer = 12;
        attr.ah_attr.dlid = remote.lid;
        attr.ah_attr.sl = ctx_->sl();
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num = ctx_->port();
        if (remote.link_layer == IBV_LINK_LAYER_INFINIBAND) {
            attr.ah_attr.is_global = 0;
        } else {
            attr.ah_attr.is_global = 1;
            attr.ah_attr.grh.dgid = remote.gid;
            attr.ah_attr.grh.sgid_index = ctx_->gid_index();
            attr.ah_attr.grh.hop_limit = 1;
            attr.ah_attr.grh.traffic_class = ctx_->traffic_class();
        }

        int ret = ibv_modify_qp(
            qp_, &attr,
            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
        if (ret)
            throw std::runtime_error("ibv_modify_qp RTR failed: " +
                                     std::to_string(ret));
    }

    // RTR → RTS
    {
        ibv_qp_attr attr = {};
        attr.qp_state = IBV_QPS_RTS;
        attr.timeout = qp_ack_timeout();
        attr.retry_cnt = qp_retry_cnt();
        attr.rnr_retry = 7;
        attr.sq_psn = 0;
        attr.max_rd_atomic = 16;

        int ret = ibv_modify_qp(qp_, &attr,
                                IBV_QP_STATE | IBV_QP_TIMEOUT |
                                    IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                    IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
        if (ret)
            throw std::runtime_error("ibv_modify_qp RTS failed: " +
                                     std::to_string(ret));
    }
}

static const char* qp_state_str(ibv_qp_state s) {
    switch (s) {
        case IBV_QPS_RESET: return "RESET";
        case IBV_QPS_INIT:  return "INIT";
        case IBV_QPS_RTR:   return "RTR";
        case IBV_QPS_RTS:   return "RTS";
        case IBV_QPS_ERR:   return "ERR";
        case IBV_QPS_SQD:   return "SQD";
        default:            return "UNKNOWN";
    }
}

// wr_id tags: distinguish sync completions from background async-write completions.
static constexpr uint64_t WR_ID_SYNC = 1;
static constexpr uint64_t WR_ID_ASYNC = 2;

void RdmaEndpoint::post_and_poll(ibv_send_wr& wr) {
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr_id = WR_ID_SYNC;
    op_count_++;

    ibv_send_wr* bad_wr = nullptr;
    int ret = ibv_post_send(qp_, &wr, &bad_wr);
    if (ret) {
        ibv_qp_attr qattr = {};
        ibv_qp_init_attr qinit = {};
        ibv_query_qp(qp_, &qattr, IBV_QP_STATE | IBV_QP_CUR_STATE, &qinit);
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "ibv_post_send failed: %d opcode=%d op_count=%lu qp_state=%s",
                 ret, wr.opcode, (unsigned long)op_count_,
                 qp_state_str(qattr.qp_state));
        throw std::runtime_error(msg);
    }

    // Drain CQEs until our sync completion; credit async ones along the way.
    while (true) {
        ibv_wc wc = {};
        int ne = ibv_poll_cq(cq_, 1, &wc);
        if (ne == 0) continue;
        if (ne < 0)
            throw std::runtime_error("ibv_poll_cq failed");
        if (wc.status != IBV_WC_SUCCESS) {
            ibv_qp_attr qattr = {};
            ibv_qp_init_attr qinit = {};
            ibv_query_qp(qp_, &qattr, IBV_QP_STATE | IBV_QP_CUR_STATE, &qinit);
            char msg[512];
            snprintf(msg, sizeof(msg),
                     "RDMA WC error: status=%d opcode=%d vendor_err=%u "
                     "qp_num=%u wr_id=%lu op_count=%lu qp_state=%s",
                     wc.status, wc.opcode, wc.vendor_err, wc.qp_num,
                     (unsigned long)wc.wr_id,
                     (unsigned long)op_count_, qp_state_str(qattr.qp_state));
            throw std::runtime_error(msg);
        }
        if (wc.wr_id == WR_ID_ASYNC) {
            if (outstanding_signaled_ > 0) outstanding_signaled_--;
            continue;
        }
        break;
    }
}

void RdmaEndpoint::rdma_read(void* local_buf, uint32_t local_lkey,
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

    post_and_poll(wr);
}

void RdmaEndpoint::rdma_read_batch2(
    void* buf1, uint32_t lkey1, uint64_t addr1, uint32_t rkey1, uint32_t len1,
    void* buf2, uint32_t lkey2, uint64_t addr2, uint32_t rkey2, uint32_t len2) {

    ibv_sge sge1 = {};
    sge1.addr = (uint64_t)buf1;
    sge1.length = len1;
    sge1.lkey = lkey1;

    ibv_sge sge2 = {};
    sge2.addr = (uint64_t)buf2;
    sge2.length = len2;
    sge2.lkey = lkey2;

    ibv_send_wr wr2 = {};
    wr2.opcode = IBV_WR_RDMA_READ;
    wr2.sg_list = &sge2;
    wr2.num_sge = 1;
    wr2.wr.rdma.remote_addr = addr2;
    wr2.wr.rdma.rkey = rkey2;
    wr2.next = nullptr;

    ibv_send_wr wr1 = {};
    wr1.opcode = IBV_WR_RDMA_READ;
    wr1.sg_list = &sge1;
    wr1.num_sge = 1;
    wr1.wr.rdma.remote_addr = addr1;
    wr1.wr.rdma.rkey = rkey1;
    wr1.next = &wr2;
    wr1.send_flags = 0;  // unsignaled

    wr2.send_flags = IBV_SEND_SIGNALED;
    wr1.wr_id = wr2.wr_id = WR_ID_SYNC;

    ibv_send_wr* bad_wr = nullptr;
    int ret = ibv_post_send(qp_, &wr1, &bad_wr);
    if (ret)
        throw std::runtime_error("ibv_post_send batch2 failed: " +
                                 std::to_string(ret));

    // Drain CQEs until our sync completion; credit any async ones along the way.
    while (true) {
        ibv_wc wc = {};
        int ne = ibv_poll_cq(cq_, 1, &wc);
        if (ne == 0) continue;
        if (ne < 0)
            throw std::runtime_error("ibv_poll_cq failed");
        if (wc.status != IBV_WC_SUCCESS)
            throw std::runtime_error("RDMA batch2 WC error: " +
                                     std::to_string(wc.status));
        if (wc.wr_id == WR_ID_ASYNC) {
            if (outstanding_signaled_ > 0) outstanding_signaled_--;
            continue;
        }
        break;
    }
}

void RdmaEndpoint::rdma_read_multi(const RdmaReadOp* ops, int count,
                                   uint32_t local_lkey) {
    if (count <= 0) return;
    if (count == 1) {
        rdma_read(ops[0].local_buf, local_lkey,
                  ops[0].remote_addr, ops[0].remote_rkey, ops[0].length);
        return;
    }

    static constexpr int BATCH = 16;  // must not exceed max_rd_atomic

    for (int base = 0; base < count; base += BATCH) {
        int end = std::min(base + BATCH, count);
        int n = end - base;

        ibv_sge sges[BATCH];
        ibv_send_wr wrs[BATCH];

        for (int i = 0; i < n; i++) {
            const auto& op = ops[base + i];
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
            wrs[i].wr_id = WR_ID_SYNC;
        }

        wrs[n - 1].send_flags = IBV_SEND_SIGNALED;

        ibv_send_wr* bad_wr = nullptr;
        int ret = ibv_post_send(qp_, &wrs[0], &bad_wr);
        if (ret) {
            ibv_qp_attr qattr = {};
            ibv_qp_init_attr qinit = {};
            ibv_query_qp(qp_, &qattr, IBV_QP_STATE | IBV_QP_CUR_STATE, &qinit);
            char msg[512];
            snprintf(msg, sizeof(msg),
                     "ibv_post_send read_multi failed: %d "
                     "(total=%d batch=%d/%d n=%d) qp_state=%s",
                     ret, count, base / BATCH, (count + BATCH - 1) / BATCH, n,
                     qp_state_str(qattr.qp_state));
            throw std::runtime_error(msg);
        }

        while (true) {
            ibv_wc wc = {};
            int ne = ibv_poll_cq(cq_, 1, &wc);
            if (ne == 0) continue;
            if (ne < 0)
                throw std::runtime_error("ibv_poll_cq failed");
            if (wc.status != IBV_WC_SUCCESS) {
                ibv_qp_attr qattr = {};
                ibv_qp_init_attr qinit = {};
                ibv_query_qp(qp_, &qattr,
                             IBV_QP_STATE | IBV_QP_CUR_STATE, &qinit);
                char msg[512];
                snprintf(msg, sizeof(msg),
                         "RDMA read_multi WC error: status=%d opcode=%d "
                         "vendor_err=%u qp_num=%u batch=%d/%d qp_state=%s",
                         wc.status, wc.opcode, wc.vendor_err, wc.qp_num,
                         base / BATCH, (count + BATCH - 1) / BATCH,
                         qp_state_str(qattr.qp_state));
                throw std::runtime_error(msg);
            }
            if (wc.wr_id == WR_ID_ASYNC) {
                if (outstanding_signaled_ > 0) outstanding_signaled_--;
                continue;
            }
            break;
        }
    }

    op_count_ += count;
}

void RdmaEndpoint::rdma_write(uint64_t remote_addr, uint32_t remote_rkey,
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

    post_and_poll(wr);
}

// ============= Async inline-write path (Opt 1) =============
// Backpressure: signal every SIGNAL_EVERY WRs; drain one near MAX_SEND_WR.

void RdmaEndpoint::drain_one_signaled() {
    // Wait for one async-write completion; skip unexpected wr_ids defensively.
    while (true) {
        ibv_wc wc = {};
        int ne = ibv_poll_cq(cq_, 1, &wc);
        if (ne == 0) continue;
        if (ne < 0)
            throw std::runtime_error("ibv_poll_cq (async drain) failed");
        if (wc.status != IBV_WC_SUCCESS) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                     "RDMA async WC error: status=%d opcode=%d vendor_err=%u "
                     "qp_num=%u wr_id=%lu",
                     wc.status, wc.opcode, wc.vendor_err, wc.qp_num,
                     (unsigned long)wc.wr_id);
            throw std::runtime_error(msg);
        }
        if (wc.wr_id == WR_ID_ASYNC) {
            outstanding_signaled_--;
            return;
        }
        // Stray sync completion (shouldn't happen — drop it).
    }
}

void RdmaEndpoint::flush() {
    while (outstanding_signaled_ > 0)
        drain_one_signaled();
    // Non-signaled tail WRs are drained by the next SIGNAL_EVERY-th WR / flush.
}

static void check_post_send(int ret, ibv_qp* qp, int wr_count,
                             uint64_t op_count) {
    if (!ret) return;
    ibv_qp_attr qattr = {};
    ibv_qp_init_attr qinit = {};
    ibv_query_qp(qp, &qattr, IBV_QP_STATE | IBV_QP_CUR_STATE, &qinit);
    char msg[256];
    snprintf(msg, sizeof(msg),
             "ibv_post_send (async) failed: %d wr_count=%d op_count=%lu "
             "qp_state=%s",
             ret, wr_count, (unsigned long)op_count,
             qp_state_str(qattr.qp_state));
    throw std::runtime_error(msg);
}

void RdmaEndpoint::rdma_write_async_inline(uint64_t remote_addr,
                                            uint32_t remote_rkey,
                                            const void* local_buf,
                                            uint32_t length) {
    // Backpressure: leave room for one new WR (signaled or not).
    int used = inflight_unsignaled_ + outstanding_signaled_ * SIGNAL_EVERY;
    while (used + SIGNAL_EVERY >= MAX_SEND_WR) {
        drain_one_signaled();
        used = inflight_unsignaled_ + outstanding_signaled_ * SIGNAL_EVERY;
    }

    ibv_sge sge = {};
    sge.addr = (uint64_t)local_buf;
    sge.length = length;
    sge.lkey = 0;  // ignored when IBV_SEND_INLINE is set

    ibv_send_wr wr = {};
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = remote_rkey;
    wr.wr_id = WR_ID_ASYNC;

    inflight_unsignaled_++;
    op_count_++;
    bool signal = (inflight_unsignaled_ >= SIGNAL_EVERY);
    wr.send_flags = IBV_SEND_INLINE | (signal ? IBV_SEND_SIGNALED : 0);

    ibv_send_wr* bad_wr = nullptr;
    int ret = ibv_post_send(qp_, &wr, &bad_wr);
    check_post_send(ret, qp_, 1, op_count_);

    if (signal) {
        inflight_unsignaled_ = 0;
        outstanding_signaled_++;
    }
}

void RdmaEndpoint::rdma_write_chain_inline(const InlineWriteOp* ops, int count) {
    if (count <= 0) return;
    if (count == 1) {
        rdma_write_async_inline(ops[0].remote_addr, ops[0].remote_rkey,
                                ops[0].local_buf, ops[0].length);
        return;
    }

    // Process in chunks bounded by the remaining send-queue capacity.
    int posted = 0;
    while (posted < count) {
        // Drain until we have room for at least one new chain.
        int used = inflight_unsignaled_ + outstanding_signaled_ * SIGNAL_EVERY;
        // Reserve up to SIGNAL_EVERY per chain (one signaled WR at the tail).
        while (used + SIGNAL_EVERY >= MAX_SEND_WR) {
            drain_one_signaled();
            used = inflight_unsignaled_ + outstanding_signaled_ * SIGNAL_EVERY;
        }

        int remaining = count - posted;
        int chain_len = std::min(remaining, SIGNAL_EVERY);

        // Stack-allocated chain (chain_len <= SIGNAL_EVERY = 16).
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
            wrs[i].send_flags = IBV_SEND_INLINE;  // unsignaled by default
            wrs[i].wr_id = WR_ID_ASYNC;
        }
        // Tail of the chain is signaled — one CQE per chain.
        wrs[chain_len - 1].send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;

        ibv_send_wr* bad_wr = nullptr;
        int ret = ibv_post_send(qp_, &wrs[0], &bad_wr);
        check_post_send(ret, qp_, chain_len, op_count_);

        // Bookkeeping: chain_len - 1 unsignaled + 1 signaled was posted.
        inflight_unsignaled_ = 0;       // tail signal clears the unsignaled count
        outstanding_signaled_++;
        op_count_ += chain_len;
        posted += chain_len;
    }
}

uint64_t RdmaEndpoint::rdma_fetch_add(uint64_t remote_addr,
                                      uint32_t remote_rkey,
                                      uint64_t add_val,
                                      void* local_result_buf,
                                      uint32_t local_lkey) {
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

    post_and_poll(wr);
    return *(uint64_t*)local_result_buf;
}

// ==================== Loopback connect ====================

void rdma_loopback_connect(RdmaEndpoint& a, RdmaEndpoint& b) {
    auto info_a = a.peer_info();
    auto info_b = b.peer_info();
    a.connect(info_b);
    b.connect(info_a);
}

} // namespace radixshmem
