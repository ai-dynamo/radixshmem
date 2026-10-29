// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Backend selection for both halves of the remote-op plane.
#include "remote_op_factory.hpp"

#ifndef RADIXSHMEM_NO_RDMA

#include "radix_server.h"        // RemoteOpAddr
#include "cluster_manifest.h"    // PeerEntry, REMOTE_OP_TRANSPORT_*
#include "rdma_transport.hpp"    // DcInitiator, DcTarget, dc_create_ah
#ifdef RADIXSHMEM_REMOTE_OP_RDMA
#include "dc_bilateral.hpp"                 // DcClientTransport, DcServerTransport
#endif
#ifdef RADIXSHMEM_HAS_ZMQ
#include "zmq_transport.hpp"                // ZmqClientTransport, ZmqServerTransport
#endif
#ifdef RADIXSHMEM_HAS_GRPC
#include "grpc_remote_op.hpp"               // GrpcRemoteOp{ClientTransport,Server}
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace radixshmem {

namespace {

int rc_resolve_lanes() {
    int lanes = 1;
    if (const char* e = ::getenv("RADIXSHMEM_RC_LANES")) {
        int v = std::atoi(e);
        if (v >= 1 && v <= 8) lanes = v;
    }
    return lanes;
}

}  // namespace

std::unique_ptr<IRemoteOpServerTransport> make_remote_op_server(
        uint32_t rot, uint16_t listen_port, RdmaContext& ctx,
        uint64_t dc_access_key, RemoteOpAddr& addr) {
    addr = RemoteOpAddr{};
    if (rot == REMOTE_OP_TRANSPORT_DC) {
#ifdef RADIXSHMEM_REMOTE_OP_RDMA
        int k_pool = 64;
        if (const char* e = ::getenv("RADIXSHMEM_RC_KPOOL")) {
            int v = std::atoi(e);
            if (v >= 1 && v <= 4096) k_pool = v;
        }
        auto t = std::make_unique<DcServerTransport>(
            ctx, dc_access_key, std::min(rc_resolve_lanes(), 4), k_pool);
        for (int i = 0; i < t->lanes() && i < 4; ++i)
            addr.dctns[i] = t->dct_num(i);
        return t;   // addr.port stays 0 — DC has no TCP endpoint
#else
        throw std::runtime_error(
            "remote-op: cluster authoritative transport is dc but this build "
            "has no RDMA control-plane backend");
#endif
    }
    if (rot == REMOTE_OP_TRANSPORT_GRPC) {
#ifdef RADIXSHMEM_HAS_GRPC
        auto g = std::make_unique<GrpcRemoteOpServer>(listen_port);
        addr.port = g->port();   // actual bound port (throws on failure)
        return g;
#else
        throw std::runtime_error(
            "remote-op: cluster authoritative transport is grpc but gRPC is "
            "not compiled into this build");
#endif
    }
    if (rot != REMOTE_OP_TRANSPORT_ZMQ) {
        throw std::runtime_error(
            "remote-op: cluster authoritative transport enum " +
            std::to_string(rot) + " is not served by this build "
            "(removed xrc/rdma backend?) — redeploy rank 0 with "
            "remote_op_transport = grpc, dc or zmq");
    }
#ifdef RADIXSHMEM_HAS_ZMQ
    auto z = std::make_unique<ZmqServerTransport>(listen_port);
    addr.port = z->bind();   // actual bound port (throws on failure)
    return z;
#else
    throw std::runtime_error(
        "remote-op: cluster authoritative transport is zmq but ZMQ is not "
        "compiled into this build");
#endif
}

RemoteOpClientFactory::RemoteOpClientFactory(uint32_t rot, RdmaContext& ctx,
                                             uint64_t dc_access_key)
    : rot_(rot), ctx_(ctx), dc_access_key_(dc_access_key) {}

RemoteOpClientFactory::~RemoteOpClientFactory() {
    if (dc_stack_mu_ && dc_stack_mu_.use_count() > 1) {
        fprintf(stderr,
                "[PeerConnector] WARNING: %ld dc remote-op session(s) (or "
                "retained lock=true results) outlive this client — their "
                "destruction is use-after-free. Drop all RemotePeer handles AND "
                "locked QueryResults before the RadixClient.\n",
                (long)(dc_stack_mu_.use_count() - 1));
    }
    dc_ini_.reset();
    dc_tgt_.reset();
}

std::unique_ptr<IRemoteOpClientTransport> RemoteOpClientFactory::open(
        const PeerEntry& p, int client_rank, std::string& endpoint) {
    if (rot_ == REMOTE_OP_TRANSPORT_DC) {
#ifdef RADIXSHMEM_REMOTE_OP_RDMA
        if (p.rc_dct_num == 0)
            throw std::runtime_error(
                "open_remote: peer rank " + std::to_string(p.rank) +
                " has remote ops disabled (rc_dct_num == 0)");
        // DC sessions share one initiator+target stack; ZMQ/gRPC sessions own theirs.
        if (!dc_stack_mu_) dc_stack_mu_ = std::make_shared<std::mutex>();
        std::lock_guard<std::mutex> g(*dc_stack_mu_);
        if (!dc_tgt_) {
            auto tgt = std::make_unique<DcTarget>(
                ctx_, dc_access_key_, /*srq_max_wr=*/16);
            auto ini = std::make_unique<DcInitiator>(ctx_, dc_access_key_);
            dc_tgt_ = std::move(tgt);
            dc_ini_ = std::move(ini);
        }
        int p_eff = 1;
        for (int i = 0; i < 3 && p.rc_dct_num_extra[i] != 0; ++i) ++p_eff;
        int lane = (int)((unsigned)client_rank % (unsigned)p_eff);
        uint32_t dctn = lane == 0 ? p.rc_dct_num : p.rc_dct_num_extra[lane - 1];
        ibv_gid dgid;
        std::memcpy(&dgid, p.gid_raw, 16);
        ibv_ah* ah = dc_create_ah(ctx_, dgid, p.lid, p.link_layer);
        if (!ah)
            throw std::runtime_error(
                "open_remote: ibv_create_ah to rank " +
                std::to_string(p.rank) + " failed");
        endpoint = "dc://" + std::string(p.host) + "#" + std::to_string(dctn);
        return std::make_unique<DcClientTransport>(
            *dc_ini_, *dc_tgt_, ah, dctn, client_rank, dc_stack_mu_);
#else
        throw std::runtime_error(
            "open_remote: cluster remote-op transport is dc but "
            "this build has no RDMA control-plane backend");
#endif
    }

    if (p.remote_op_listen_port == 0)
        throw std::runtime_error(
            "open_remote: peer rank " + std::to_string(p.rank) +
            " has remote ops disabled (remote-op listen port == 0)");
    std::string hostport = std::string(p.host) + ":" +
                           std::to_string(p.remote_op_listen_port);

    if (rot_ == REMOTE_OP_TRANSPORT_GRPC) {
#ifdef RADIXSHMEM_HAS_GRPC
        endpoint = "grpc://" + hostport;
        return std::make_unique<GrpcRemoteOpClientTransport>(
            hostport, grpc_remote_op_timeout_ms());
#else
        throw std::runtime_error(
            "open_remote: cluster remote-op transport is grpc but "
            "gRPC is not compiled into this build");
#endif
    }
    if (rot_ != REMOTE_OP_TRANSPORT_ZMQ) {
        throw std::runtime_error(
            "open_remote: cluster advertises remote-op transport enum " +
            std::to_string(rot_) + ", which this build "
            "does not serve (removed xrc/rdma backend?) — redeploy rank 0 "
            "with remote_op_transport = grpc, dc or zmq");
    }
#ifdef RADIXSHMEM_HAS_ZMQ
    endpoint = "tcp://" + hostport;
    return std::make_unique<ZmqClientTransport>(endpoint);
#else
    throw std::runtime_error(
        "open_remote: cluster remote-op transport is zmq but "
        "ZMQ is not compiled into this build");
#endif
}

}  // namespace radixshmem

#endif  // !RADIXSHMEM_NO_RDMA
