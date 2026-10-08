// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Cluster wire structs (ClusterManifestHeader + PeerEntry) — base64'd into etcd
// /config and /peers, decoded by RadixServer and RadixClient.

#pragma once

#include "types.h"

#include <cstdint>
#include <cstring>

namespace radixshmem {

static constexpr uint32_t CLUSTER_MANIFEST_VERSION  = 11; // 11: rht / peer-index transports split
static constexpr uint32_t CLUSTER_MANIFEST_HOST_LEN = 64;
static constexpr uint32_t CLUSTER_MANIFEST_ENTRY_SIZE = 352;
static constexpr uint32_t CLUSTER_RHT_NAME_LEN = 64;

// Max RHT shards (M) the inline holder map can describe; M <= this cap.
static constexpr uint32_t CLUSTER_MAX_SHARDS = 64;

struct ClusterManifestHeader {
    uint64_t magic;           // CLUSTER_MANIFEST_MAGIC
    uint32_t version;         // CLUSTER_MANIFEST_VERSION
    uint32_t world_size;
    uint32_t entry_size;      // CLUSTER_MANIFEST_ENTRY_SIZE
                              // NOTE: this header is not on the wire today —
                              // peer blobs carry their own version framing.
    uint32_t entry_count;     // == world_size

    // ---- Authoritative RHT shard map (rank 0, broadcast) ----
    uint32_t num_shards;      // M (number of RHT shards)
    uint32_t rht_capacity;    // requested per-shard bucket count, global/uniform (RHT normalizes)
    uint32_t shard_holder_rank[CLUSTER_MAX_SHARDS];   // shard s → holder rank; [0,num_shards) valid

    // ---- Authoritative RDMA QoS (rank 0, broadcast); applied to index-plane QPs ----
    int32_t  rdma_sl;
    int32_t  rdma_tc;

    // ---- RDMA transports (rank 0, broadcast) ----
    uint32_t peer_index_transport;   // remote walk into peer trees, see ClusterTransport
    uint32_t rht_slots;          // W: RHT slots per bucket (chaining depth)
    uint64_t dc_access_key;      // cluster-wide DCT key (either plane DC)

    // ---- Remote-op control-plane transport (rank 0, broadcast) ----
    uint32_t remote_op_transport;   // see RemoteOpTransport
    uint32_t rht_transport;         // client -> RHT shard holder, see ClusterTransport
};
static_assert(sizeof(ClusterManifestHeader) == 320,
              "ClusterManifestHeader size changed — update version + readers");

enum ClusterTransport : uint32_t {
    CLUSTER_TRANSPORT_XRC = 0,
    CLUSTER_TRANSPORT_DC  = 1,
};

enum RemoteOpTransport : uint32_t {
    REMOTE_OP_TRANSPORT_ZMQ  = 0,
    REMOTE_OP_TRANSPORT_RDMA = 1,   // removed backend — reserved, rejected on read
    REMOTE_OP_TRANSPORT_DC   = 2,   // DCI/DCT
};

// One entry per rank (each server fills its own; master broadcasts the union).
struct PeerEntry {
    // Identity
    int32_t  rank;
    int32_t  _pad0;

    // Reachability — TCP bootstrap endpoint for this rank's XRC listener
    char     host[CLUSTER_MANIFEST_HOST_LEN];
    uint16_t xrc_listen_port;
    // Remote-op endpoint port: zmq REP TCP port (0 = disabled); 0 under dc (rc_dct_num addresses)
    uint16_t zmq_listen_port;
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
    // Remote-op DCT number under DC; 0 = remote ops disabled
    uint32_t rc_dct_num;

    // RDMA device name (client's local entry picks the same device)
    char     index_dev[32];

    // DC control-plane lanes 1..3 (lane 0 = rc_dct_num above)
    uint32_t rc_dct_num_extra[3];

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

// ClusterManifestHeader / PeerEntry are the etcd /config and /peers wire structs
// (base64'd); they are not laid out in shm.

} // namespace radixshmem
