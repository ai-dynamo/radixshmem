// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// gRPC transport construction for both halves of the remote-op plane.
#include "remote_op_factory.hpp"

#ifndef RADIXSHMEM_NO_RDMA

#include "cluster_manifest.h"
#ifdef RADIXSHMEM_HAS_GRPC
#include "grpc_remote_op.hpp"               // GrpcRemoteOp{ClientTransport,Server}
#endif

#include <stdexcept>
#include <string>

namespace radixshmem {

std::unique_ptr<IRemoteOpServerTransport> make_remote_op_server(
        uint16_t listen_port, RemoteOpAddr& addr) {
    addr = RemoteOpAddr{};
#ifdef RADIXSHMEM_HAS_GRPC
    auto server = std::make_unique<GrpcRemoteOpServer>(listen_port);
    addr.port = server->port();
    return server;
#else
    (void)listen_port;
    throw std::runtime_error(
        "remote-op: gRPC is the only supported transport but is not compiled into this build");
#endif
}

std::unique_ptr<IRemoteOpClientTransport> make_remote_op_client(
        const PeerEntry& peer, std::string& endpoint) {
    if (peer.remote_op_listen_port == 0)
        throw std::runtime_error(
            "open_remote: peer rank " + std::to_string(peer.rank) +
            " has remote ops disabled (remote-op listen port == 0)");

#ifdef RADIXSHMEM_HAS_GRPC
    const std::string hostport = std::string(peer.host) + ":" +
                                std::to_string(peer.remote_op_listen_port);
    endpoint = "grpc://" + hostport;
    return std::make_unique<GrpcRemoteOpClientTransport>(
        hostport, grpc_remote_op_timeout_ms());
#else
    (void)endpoint;
    throw std::runtime_error(
        "open_remote: gRPC is the only supported transport but is not compiled into this build");
#endif
}

}  // namespace radixshmem

#endif  // !RADIXSHMEM_NO_RDMA
