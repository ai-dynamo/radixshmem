// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The shard map rank 0 resolves and publishes in etcd /config. Fixed-size, native
// endian; version mismatch is fatal.

#pragma once

#include <cstdint>
#include <limits>

#include "cluster_manifest.h"   // CLUSTER_MAX_SHARDS
#include "types.h"              // RemoteStatus (shared with QueryResult)

namespace radixshmem {

constexpr uint32_t CLUSTER_PROTOCOL_VERSION = 16;  // lockstep with manifest v16
constexpr uint32_t CLUSTER_SHARDMAP_MAGIC = 0x52584d50; // "RXMP"

// rank 0 → worker: authoritative shard map, sent after rank 0 resolves it.
struct ShardMapMsg {
    uint32_t magic;          // CLUSTER_SHARDMAP_MAGIC
    uint32_t version;        // CLUSTER_PROTOCOL_VERSION
    uint32_t num_shards;     // M
    uint32_t rht_capacity;   // per-shard bucket count
    int32_t  rdma_sl;        // authoritative QoS (rank 0's env), see manifest
    int32_t  rdma_tc;
    uint32_t rht_transport;         // authoritative RHT-plane transport (ClusterTransport)
    uint32_t peer_index_transport;  // rdma peer index's transport (ClusterTransport)
    uint32_t rht_slots;      // authoritative W (RHT slots per bucket)
    uint64_t dc_access_key;  // cluster-wide DCT key (when either plane is DC)
    uint32_t peer_index_backend;    // ClusterPeerIndexBackend
    uint32_t register_chunk_size;   // cs; node-boundary / probe stride alignment
    uint32_t holder_rank[CLUSTER_MAX_SHARDS];  // shard s → holder rank
    uint32_t rht_backend;    // ClusterRhtBackend; REDIS leaves the shard map empty
    uint32_t rht_redis_timeout_ms;
    char     rht_redis_seeds[CLUSTER_RHT_SEEDS_LEN];
};

constexpr int      MAX_REMOTE_OP_MESSAGE_BYTES = std::numeric_limits<int>::max();
constexpr int      MAX_REMOTE_OP_BLOCKS =
    (MAX_REMOTE_OP_MESSAGE_BYTES - 4096) / (2 * (int)sizeof(uint64_t));

} // namespace radixshmem
