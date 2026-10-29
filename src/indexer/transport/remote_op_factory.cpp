// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Backend selection for both halves of the remote-op plane.
#include "remote_op_factory.hpp"

#ifndef RADIXSHMEM_NO_RDMA

#include "radix_server.h"        // RemoteOpAddr
#include "cluster_manifest.h"    // PeerEntry, REMOTE_OP_TRANSPORT_*
#ifdef RADIXSHMEM_REMOTE_OP_RDMA
#include "dc_bilateral.hpp"                 // DcClientStack, make_dc_server_transport
#endif
#ifdef RADIXSHMEM_HAS_ZMQ
#include "zmq_transport.hpp"                // ZmqClientTransport, ZmqServerTransport
#endif
#ifdef RADIXSHMEM_HAS_GRPC
#include "grpc_remote_op.hpp"               // GrpcRemoteOp{ClientTransport,Server}
#endif

#include <stdexcept>
#include <string>
#include <utility>

namespace radixshmem {

std::unique_ptr<IRemoteOpServerTransport> make_remote_op_server(
        uint32_t rot, uint16_t listen_port, RdmaContext& ctx,
        uint64_t dc_access_key, RemoteOpAddr& addr) {
    addr = RemoteOpAddr{};
    if (rot == REMOTE_OP_TRANSPORT_DC) {
#ifdef RADIXSHMEM_REMOTE_OP_RDMA
        auto t = make_dc_server_transport(ctx, dc_access_key);
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

RemoteOpClientFactory::~RemoteOpClientFactory() = default;

std::unique_ptr<IRemoteOpClientTransport> RemoteOpClientFactory::open(
        const PeerEntry& p, int client_rank, std::string& endpoint) {
    if (rot_ == REMOTE_OP_TRANSPORT_DC) {
#ifdef RADIXSHMEM_REMOTE_OP_RDMA
        if (!dc_) dc_ = std::make_unique<DcClientStack>(ctx_, dc_access_key_);
        return dc_->open(p, client_rank, endpoint);
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
