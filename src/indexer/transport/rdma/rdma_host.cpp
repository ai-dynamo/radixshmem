// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "rdma_host.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace radixshmem {

RdmaHost::RdmaHost(const RdmaHostConfig& cfg) : cfg_(cfg) {
    const IndexDevice d = index_device_env();
    ctx_  = std::make_unique<RdmaContext>(d.dev.empty() ? nullptr : d.dev.c_str(), 1,
                                          d.gid_idx >= 0 ? d.gid_idx : 3);
    ctx_->set_qos(cfg_.sl, cfg_.tc);   // the cluster's QoS before any QP connects
    xrcd_ = std::make_unique<RdmaXrcd>(*ctx_);
    cq_.reset(ibv_create_cq(ctx_->ctx(), 256, nullptr, nullptr, 0));
    if (!cq_) throw std::runtime_error("RdmaHost: ibv_create_cq failed");
    srq_  = std::make_unique<XrcSrq>(*ctx_, *xrcd_, cq_.get());
    if (cfg_.dc) {
        dct_ = std::make_unique<DcTarget>(*ctx_, cfg_.dc_access_key, /*srq_max_wr=*/1);
        dct_num_ = dct_->open_dct();
        fprintf(stderr, "[RadixServer rank %d] DC target up: dctn=%u\n", cfg_.rank, dct_num_);
    }
}

RdmaHost::~RdmaHost() {
    std::lock_guard<std::mutex> g(clients_mu_);
    clients_.clear();   // the RECV QPs, before the SRQ, CQ and context they use
}

void RdmaHost::expose_tree(void* base, size_t size) {
    // READ + ATOMIC only (no REMOTE_WRITE); RHT writes use the shard's own MR.
    tree_mr_ = std::make_unique<RdmaMemoryRegion>(
        ctx_->pd(), base, size,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
}

void RdmaHost::publish_to(PeerEntry& e) const {
    if (tree_mr_) {
        e.shm_base_addr = (uint64_t)tree_mr_->addr();
        e.shm_rkey      = tree_mr_->rkey();
    }
    std::strncpy(e.index_dev, ctx_->name(), sizeof(e.index_dev) - 1);
    memcpy(e.gid_raw, &ctx_->gid(), 16);
    e.gid_index = ctx_->gid_index();
    ibv_port_attr pa{};
    if (ibv_query_port(ctx_->ctx(), ctx_->port(), &pa) == 0) {
        e.lid        = pa.lid;
        e.link_layer = (uint8_t)pa.link_layer;
    }
    e.srq_num = srq_->srq_num();
    e.dct_num = dct_num_;
}

XrcConnInfo RdmaHost::accept_xrc(const XrcConnInfo& caller) {
    std::lock_guard<std::mutex> g(clients_mu_);
    auto recv_qp = std::make_unique<XrcRecvQp>(*ctx_, *xrcd_, *srq_, cq_.get());
    XrcPeerInfo peer{};
    memcpy(&peer.gid, caller.gid, sizeof(peer.gid));
    peer.gid_index  = caller.gid_index;
    peer.lid        = (uint16_t)caller.lid;
    peer.link_layer = (uint8_t)caller.link_layer;
    recv_qp->connect(caller.qp_num, peer);

    ibv_port_attr pa{};
    if (ibv_query_port(ctx_->ctx(), ctx_->port(), &pa) != 0)
        throw std::runtime_error("ibv_query_port failed");
    XrcConnInfo me;
    me.rank       = cfg_.rank;
    me.qp_num     = recv_qp->qp_num();
    me.srq_num    = srq_->srq_num();
    me.link_layer = pa.link_layer;
    me.lid        = pa.lid;
    me.gid_index  = ctx_->gid_index();
    memcpy(me.gid, &ctx_->gid(), sizeof(me.gid));
    clients_.push_back(std::move(recv_qp));
    return me;
}

} // namespace radixshmem
