// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// PeerEntry, the etcd /peers wire struct (base64'd, not laid out in shm), and the
// cluster enums and limits the shard map in /config uses.

#pragma once

#include "types.h"

#include <cstdint>
#include <cstring>

namespace radixshmem {

static constexpr uint32_t CLUSTER_MANIFEST_VERSION  = 16; // 16: one RPC port, no reserved fields
static constexpr uint32_t CLUSTER_MANIFEST_HOST_LEN = 64;
static constexpr uint32_t CLUSTER_MANIFEST_ENTRY_SIZE = 344;
static constexpr uint32_t CLUSTER_RHT_NAME_LEN = 64;

// Max RHT shards (M) the inline holder map can describe; M <= this cap.
static constexpr uint32_t CLUSTER_MAX_SHARDS = 64;
// Redis RHT seed list ("host:port,...") in the shard map.
static constexpr uint32_t CLUSTER_RHT_SEEDS_LEN = 256;

enum ClusterTransport : uint32_t {
    CLUSTER_TRANSPORT_XRC = 0,
    CLUSTER_TRANSPORT_DC  = 1,
};

enum ClusterRhtBackend : uint32_t {
    CLUSTER_RHT_RDMA  = 0,
    CLUSTER_RHT_REDIS = 1,
};

// How a query reaches the tree of the peer the RHT picked (peer_index.h).
enum ClusterPeerIndexBackend : uint32_t {
    CLUSTER_PEER_INDEX_RDMA = 0,
    CLUSTER_PEER_INDEX_GRPC = 1,
};

// One entry per rank (each server fills its own; master broadcasts the union).
struct PeerEntry {
    // Identity
    int32_t  rank;
    int32_t  _pad0;

    // Reachability: the gRPC endpoint of remote ops, peer queries and XRC setup
    char     host[CLUSTER_MANIFEST_HOST_LEN];
    uint16_t rpc_port;
    uint16_t _pad1;
    uint32_t _pad2;

    // Shared memory layout (this peer's local shm region)
    uint64_t shm_base_addr;                    // peer's virtual addr of shm base
    uint64_t shm_size;
    uint32_t shm_rkey;                         // rkey of the shm-spanning MR
    uint32_t node_pool_capacity;               // NodePool slot count (remote node_id range check)
    uint64_t node_pool_offset;
    uint64_t data_pool_offset;
    uint64_t global_ht_offset;
    uint32_t ht_bucket_count;
    uint32_t ht_entry_capacity;
    // RHT shard — separate shm (/<name>_rht_shard{S}) with own READ|WRITE MR;
    // holders only, non-holders leave 0. rht_shard_rkey distinct from shm_rkey.
    uint64_t rht_shard_base_addr;
    uint64_t rht_shard_size;
    uint32_t rht_shard_rkey;
    int32_t  rht_shard_id;                     // shard index S this rank holds, or -1
    // POSIX shm name of this rank's RHT shard (holders only); same-host shm_open shortcut
    char     rht_shard_name[CLUSTER_RHT_NAME_LEN];
    uint64_t rdma_rwlock_offset;
    uint32_t root_node_id;
    uint32_t _pad4;

    // RDMA address vector (raw GID bytes; cast to ibv_gid in callers)
    uint8_t  gid_raw[16];
    int32_t  gid_index;
    uint16_t lid;
    uint8_t  link_layer;
    uint8_t  _pad5;

    // XRC bootstrap-time hints
    uint32_t srq_num;                          // peer's SRQ number
    uint32_t dct_num;                          // DC target for RHT / peer-index reads (0 when neither plane is DC)

    // RDMA device name (client's local entry picks the same device)
    char     index_dev[32];

    uint64_t instance_id;

    // Component ABI (from ShmHeader at bootstrap); layout sizes are
    // binary-generation-gated, so peers validate values, not layout.
    ComponentConfig component_config;
    uint64_t component_state_offset;
};
static_assert(sizeof(PeerEntry) == CLUSTER_MANIFEST_ENTRY_SIZE,
              "PeerEntry size budget exceeded — adjust CLUSTER_MANIFEST_ENTRY_SIZE");

// Peer-entry checks mirroring create/attach; config must EQUAL the local
// one. false => component walks fail closed, legacy Full keeps working.
inline bool peer_component_abi_valid(const PeerEntry& e,
                                     const ComponentConfig& local) {
    const ComponentConfig& c = e.component_config;
    if (c.abi_version != COMPONENT_ABI_VERSION) return false;
    if (c.component_mask == 0 || (c.component_mask & ~COMPONENT_MASK_ALL) != 0)
        return false;
    if ((c.component_mask & COMPONENT_MASK_SWA) != 0 &&
        c.swa_window_blocks == 0)
        return false;
    if (c.component_mask != local.component_mask ||
        c.logical_block_tokens != local.logical_block_tokens ||
        c.swa_window_blocks != local.swa_window_blocks ||
        c.mamba_state_bytes != local.mamba_state_bytes)
        return false;
    // The CST exists iff an aux component does; Full-only peers publish 0.
    const bool aux = component_has_aux(c.component_mask);
    if (aux != (e.component_state_offset != 0)) return false;
    if (aux) {
        const uint64_t cst_end =
            e.component_state_offset +
            (uint64_t)e.node_pool_capacity * sizeof(NodeComponentState);
        if (e.shm_size != 0 &&
            (e.component_state_offset >= e.shm_size || cst_end > e.shm_size))
            return false;
    }
    return true;
}

} // namespace radixshmem
