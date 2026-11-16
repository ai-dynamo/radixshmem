// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "peer_index.h"

#include "cluster_protocol.h"
#include "grpc_peer_index.hpp"
#include "rdma_peer_index.hpp"

namespace radixshmem {

std::unique_ptr<PeerIndexBackend> make_peer_index(const ShardMapMsg& sm, PeerRegistry& peers,
                                                  const PeerIndexConfig& cfg) {
    if (sm.peer_index_backend == CLUSTER_PEER_INDEX_GRPC)
        return std::make_unique<GrpcPeerIndex>(peers, cfg);
    return std::make_unique<RdmaPeerIndex>(sm, peers, cfg);
}

} // namespace radixshmem
