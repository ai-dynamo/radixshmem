// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// TCP bootstrap wire format between RadixClient and RadixServer (Hello/Reply
// per peer, then pure RDMA). Fixed-size, native endian; version mismatch is fatal.

#pragma once

#include <cstdint>
#include <infiniband/verbs.h>

#include "cluster_manifest.h"   // CLUSTER_MAX_SHARDS
#include "types.h"              // RemoteStatus (shared with QueryResult)

namespace radixshmem {

constexpr uint32_t CLUSTER_PROTOCOL_VERSION = 12;  // lockstep with manifest v12
constexpr uint32_t CLUSTER_HELLO_MAGIC = 0x52584348; // "RXCH"
constexpr uint32_t CLUSTER_REPLY_MAGIC = 0x52584352; // "RXCR"
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
    uint32_t peer_index_transport;  // authoritative peer-index-plane transport
    uint32_t rht_slots;      // authoritative W (RHT slots per bucket)
    uint64_t dc_access_key;  // cluster-wide DCT key (when either plane is DC)
    uint32_t remote_op_transport;   // authoritative remote-op transport (RemoteOpTransport)
    uint32_t register_chunk_size;   // cs; node-boundary / probe stride alignment
    uint32_t holder_rank[CLUSTER_MAX_SHARDS];  // shard s → holder rank
};

// Client → Server, sent after TCP connect: client's RESET XRC SEND QP asks for
// a paired RECV QP on the server's shared SRQ.
struct ClusterHelloMsg {
    uint32_t      magic;
    uint32_t      version;
    int32_t       client_rank;          // RadixClient rank (informational)
    uint32_t      client_send_qp_num;   // XRC SEND QP the client just created

    uint8_t       link_layer;
    uint8_t       _pad0[3];
    uint16_t      lid;
    uint16_t      _pad1;
    int32_t       gid_index;
    union ibv_gid gid;
};

// Server → Client, after the server brings up the per-client XRC RECV QP.
struct ClusterReplyMsg {
    uint32_t      magic;
    uint32_t      version;
    int32_t       server_rank;          // peer's rank (informational; client also has it from manifest)
    uint32_t      server_recv_qp_num;   // client SEND QP RTR-targets this
    uint32_t      server_srq_num;       // wr.qp_type.xrc.remote_srqn

    uint8_t       link_layer;
    uint8_t       _pad0[3];
    uint16_t      lid;
    uint16_t      _pad1;
    int32_t       gid_index;
    union ibv_gid gid;
};

constexpr int      MAX_REMOTE_OP_BLOCKS = 512;

} // namespace radixshmem
