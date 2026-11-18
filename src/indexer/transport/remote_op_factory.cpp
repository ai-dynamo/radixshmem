// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// gRPC transport construction for both halves of the remote-op plane.
#include "remote_op_factory.hpp"


#include "cluster_manifest.h"
#include "grpc_remote_op.hpp"               // GrpcRemoteOp{ClientTransport,Server}

#include <stdexcept>
#include <string>
#include <utility>

namespace radixshmem {

std::unique_ptr<IRemoteOpServerTransport> make_remote_op_server(
        uint16_t listen_port, uint64_t instance_id, XrcAcceptor xrc, RemoteOpAddr& addr) {
    addr = RemoteOpAddr{};
    auto server = std::make_unique<GrpcRemoteOpServer>(listen_port, instance_id, std::move(xrc));
    addr.port = server->port();
    return server;
}

std::unique_ptr<IRemoteOpClientTransport> make_remote_op_client(
        const PeerEntry& peer, std::string& endpoint, int timeout_ms) {
    if (peer.rpc_port == 0)
        throw std::runtime_error(
            "open_remote: peer rank " + std::to_string(peer.rank) +
            " registered no RPC port");

    const std::string hostport = std::string(peer.host) + ":" + std::to_string(peer.rpc_port);
    endpoint = "grpc://" + hostport;
    return std::make_unique<GrpcRemoteOpClientTransport>(
        hostport, timeout_ms > 0 ? timeout_ms : grpc_remote_op_timeout_ms(), peer.instance_id);
}

}  // namespace radixshmem

