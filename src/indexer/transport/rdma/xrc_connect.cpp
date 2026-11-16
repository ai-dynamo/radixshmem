// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "xrc_connect.hpp"

#include "remote_op_factory.hpp"

#include <cstdio>
#include <cstring>
#include <exception>
#include <string>

namespace radixshmem {

std::unique_ptr<XrcSendQp> xrc_connect(RdmaContext& ctx, ibv_cq* cq, const PeerEntry& peer,
                                       int my_rank, int timeout_sec) {
    auto ep = std::make_unique<XrcSendQp>(ctx, cq);
    ibv_port_attr pa{};
    if (ibv_query_port(ctx.ctx(), ctx.port(), &pa) != 0) {
        fprintf(stderr, "[xrc_connect rank %d] ibv_query_port failed\n", my_rank);
        return nullptr;
    }
    XrcConnInfo mine;
    mine.rank       = my_rank;
    mine.qp_num     = ep->qp_num();
    mine.link_layer = pa.link_layer;
    mine.lid        = pa.lid;
    mine.gid_index  = ctx.gid_index();
    memcpy(mine.gid, &ctx.gid(), sizeof(mine.gid));

    XrcConnInfo theirs;
    RemoteStatus st = RemoteStatus::UNAVAILABLE;
    std::string endpoint;
    try {
        st = make_remote_op_client(peer, endpoint, timeout_sec * 1000)->xrc_connect(mine, theirs);
    } catch (const std::exception& e) {
        fprintf(stderr, "[xrc_connect rank %d] peer rank %d: %s\n", my_rank, peer.rank, e.what());
        return nullptr;
    }
    if (st != RemoteStatus::OK) {
        fprintf(stderr, "[xrc_connect rank %d] XrcConnect to %s failed: status %d\n", my_rank,
                endpoint.c_str(), (int)st);
        return nullptr;
    }

    XrcPeerInfo xpeer{};
    xpeer.recv_qp_num = theirs.qp_num;
    xpeer.remote_srqn = theirs.srq_num;
    memcpy(&xpeer.gid, theirs.gid, sizeof(xpeer.gid));
    xpeer.gid_index   = theirs.gid_index;
    xpeer.lid         = (uint16_t)theirs.lid;
    xpeer.link_layer  = (uint8_t)theirs.link_layer;
    ep->connect(xpeer);
    return ep;
}

} // namespace radixshmem
