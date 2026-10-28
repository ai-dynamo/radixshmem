// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "radixshmem_config.h"   // feature macros (CMake-generated, build tree)
#include "health_check.h"
#include "types.h"

#include <cstdint>
#include <memory>
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifndef RADIXSHMEM_NO_RDMA
#include "bilateral_transport.h"   // IRemoteOpServerTransport
#include <unordered_map>
#endif

namespace radixshmem {

class RadixShmemTree;

#ifndef RADIXSHMEM_NO_RDMA
class LocalTreeConnector;
class RadixClient;
class RdmaContext;
class RdmaXrcd;
class XrcSrq;
class RdmaMemoryRegion;
class RhtHolder;   // src/indexer/rht/rht_endpoint.h
class DcTarget;
class XrcRecvQp;
class EtcdRegistry;
struct PeerEntry;
struct ShardMapMsg;
class RemoteOpService;

struct RemoteOpAddr {
    uint16_t port = 0;        // zmq: actual bound TCP port (0 = not zmq / off)
    uint32_t dctns[4] = {};   // dc: lane DCT numbers, lane 0 first, 0-padded
};
#endif

struct RadixServerConfig {
    // POSIX shm name; identity-based, no rank suffix.
    std::string name = "/radixshmem";

    // Shm sizing / eviction policy.
    ShmConfig shm;

    // rank: etcd-assigned dense rank, filled after bootstrap.
    int          rank        = 0;
    int          world_size  = 1;   // observed count(/peers) overrides at bootstrap
    std::string  index_dev;          // empty = first available device
    int          gid_idx     = 3;
    int          bootstrap_timeout_sec = 60;
    // TTL of this node's etcd lease and of the shared cluster lease: how long a
    // killed node's keys, or a fully stopped cluster's /config, outlive it.
    int          lease_ttl_sec = 30;

    std::string  rpc_address;              // explicit IPv4, e.g. "192.168.1.11"
    std::string  rpc_interface;            // NIC name, e.g. "eth0"
    uint16_t     rpc_port    = 0;          // 0 = OS-assigned bootstrap / XRC listen port
    int          register_chunk_tokens = DEFAULT_REGISTER_CHUNK_TOKENS;   // copied onto shm at bootstrap

    int          expected_min_nodes = 0;   // 0 = gate on num_rht_shards only
    int          settle_ms          = 500;

    int               num_rht_shards = 0;      // M
    std::vector<int>  rht_shard_holders;       // candidate holder ranks

    std::string       rht_transport = "xrc";
    std::string       peer_index_transport = "xrc";

    // RHT slots per bucket (W): must be in {1,2,4,8}. rank 0 authoritative (broadcast).
    int               rht_slots_per_bucket = 4;

    bool              enable_remote_insert = false;   // remote-insert control plane
    bool              enable_remote_query = false;    // remote-query control plane
    uint16_t          zmq_listen_port = 0;            // remote-op endpoint TCP port

    // Control-plane backend: "zmq" | "dc" ("dc" needs world_size>1 + mlx5). rank 0 authoritative.
    std::string       remote_op_transport = "zmq";

    // Cluster bootstrap registry: "etcd://host:port" (etcd dynamic discovery).
    std::string       registry   = "etcd://127.0.0.1:2379";
    std::string       cluster_id = "default";   // etcd key namespace: radix/<cluster_id>/...
    std::string       node_name  = "";          // etcd identity; empty → auto "node<host-ip>"
};

class RadixServer {
#ifndef RADIXSHMEM_NO_RDMA
    friend class EtcdRegistry;
#endif
public:
    explicit RadixServer(const RadixServerConfig& cfg);

    // Convenience overload: constructs and immediately bootstraps (single-node).
    RadixServer(const std::string& name, const ShmConfig& cfg = ShmConfig{});

    ~RadixServer();

    RadixServer(const RadixServer&) = delete;
    RadixServer& operator=(const RadixServer&) = delete;

    // Bring the server up; world_size>1 does etcd+RDMA bootstrap (false on failure).
    bool bootstrap();

    bool is_distributed() const { return cfg_.world_size > 1; }
    int  rank()       const { return cfg_.rank; }
    int  world_size() const { return cfg_.world_size; }
    const std::string& shm_name() const;
    // etcd identity, resolved by bootstrap (auto "node<ip>" when unset).
    const std::string& node_name() const { return cfg_.node_name; }
    // etcd lease every key of this node hangs on; 0 when standalone.
    int64_t lease_id() const;

    // Attach a LocalTreeConnector on shm_name() to read or write the tree.
    RadixShmemTree& tree();
    const RadixShmemTree& tree() const;

    // Stats (forwarded to the tree).
    uint32_t total_radix_nodes() const;
    uint64_t total_blocks() const;
    float    data_pool_utilization() const;
    uint32_t mempool_total() const;
    uint32_t mempool_free() const;
    uint32_t mempool_used() const;
    uint32_t swa_mempool_total() const;
    uint32_t mamba_mempool_total() const;
    ComponentConfig component_config() const;

    uint64_t insert_success_count() const;
    PeerFaultCounters* peer_faults() const;
    uint64_t insert_error_node_pool_count() const;
    uint64_t insert_error_data_pool_count() const;
    uint64_t insert_error_ht_pool_count() const;
    uint64_t insert_error_mempool_count() const;

    // Lock-free leak audit of the owned tree (see RadixClient::check_health).
    int check_health(HealthReport* report = nullptr) const;

private:
    RadixServerConfig cfg_;

    // ---- shm region + background evictor ----
    std::unique_ptr<RadixShmemTree> tree_;
    std::string shm_name_;
    std::thread evictor_thread_;
    std::atomic<bool> evictor_running_{false};

    void start_evictor();
    void stop_evictor();
    void evictor_loop();

#ifndef RADIXSHMEM_NO_RDMA
    // ---- embedded eviction / remote-op service client ----
    std::unique_ptr<LocalTreeConnector>    svc_tree_;
    std::unique_ptr<RadixClient>   svc_client_;
    std::mutex                     svc_client_mu_;

    // ---- cluster bring-up ----
    std::string bind_ip_;   // published as PeerEntry.host
#ifdef RADIXSHMEM_HAS_ETCD
    std::unique_ptr<EtcdRegistry> registry_;   // holds the lease for the server's lifetime
#endif

    std::unique_ptr<RdmaContext>      ctx_;
    std::unique_ptr<RdmaXrcd>         xrcd_;
    std::unique_ptr<XrcSrq>           srq_;
    ibv_cq*                            cq_ = nullptr;
    std::unique_ptr<RdmaMemoryRegion> shm_mr_;       // tree shm MR: READ|ATOMIC

    // RHT shard (holders only): owns the shard shm + its MR (READ|WRITE).
    std::unique_ptr<RhtHolder>         rht_holder_;

    // DC read target — only when either plane is DC and no lane-0 DCT donated.
    std::unique_ptr<DcTarget>          dct_;
    uint32_t                           rc_lane0_dctn_ = 0;   // remote-op lane-0 dctn (0 = none)
    uint32_t                           rht_transport_ = CLUSTER_TRANSPORT_XRC;
    uint32_t                           peer_index_transport_ = CLUSTER_TRANSPORT_XRC;
    uint64_t                           dc_access_key_ = 0;
    uint32_t                           remote_op_transport_ = REMOTE_OP_TRANSPORT_ZMQ;

    // Authoritative shard map (resolved on rank 0, broadcast to all).
    int                  num_shards_ = 0;       // M
    uint32_t             rht_capacity_ = 0;     // requested per-shard buckets (RHT normalizes)
    uint32_t             rht_slots_ = 4;        // W: slots per bucket
    std::vector<int>     holder_of_shard_;      // size M: shard s → holder rank
    int                  my_local_shard_ = -1;  // shard this rank holds, or -1
    int                  rdma_sl_ = 5;          // authoritative QoS (rank 0)
    int                  rdma_tc_ = 160;

    // TCP XRC listener
    int       xrc_listen_fd_ = -1;
    uint16_t  actual_listen_port_ = 0;
    std::thread listener_thread_;
    std::atomic<bool> stopping_{false};

    // Per-client XRC RECV QP table, keyed by an internal counter.
    struct ClientSlot {
        std::unique_ptr<XrcRecvQp> recv_qp;
        int32_t  client_rank   = -1;
        uint32_t client_send_qp_num = 0;
        uint32_t recv_qp_num       = 0;
    };
    mutable std::mutex clients_mu_;
    std::unordered_map<uint64_t, std::unique_ptr<ClientSlot>> clients_;
    uint64_t next_client_id_ = 1;

    // etcd rendezvous, install manifest, start XRC accept loop; true when all peers ready.
    bool cluster_bootstrap();

    void fill_local_entry(PeerEntry& e) const;
    int  register_chunk_size() const;
    void resolve_shard_map(int n_obs);
    void adopt_shard_map(const ShardMapMsg& sm);
    // First field where `sm` disagrees with this node's own settings, "" if none.
    std::string shard_map_mismatch(const ShardMapMsg& sm) const;
    void stamp_shm_ready(int assigned_rank, int n_obs);
    void setup_local_rht_shard(PeerEntry& e);
    void setup_dct(PeerEntry& e);
    void setup_remote_op(PeerEntry& e);
    void xrc_listener_loop();
    void handle_client(int conn_fd);

    // ---- remote-op control plane (remote_op_service.cpp) ----
    std::unique_ptr<IRemoteOpServerTransport> rc_server_;
    std::unique_ptr<RemoteOpService> rc_service_;   // built by rc_spawn on svc_*

    // build_remote_op callback body: create the authoritative transport, return its address material.
    RemoteOpAddr rc_build_endpoint(uint32_t rot_auth, uint64_t dc_access_key,
                                   RdmaContext& ctx);
    // Start serving once svc_* exist; rc_stop stops serving, then drops the service.
    void     rc_spawn();
    void     rc_stop();
#endif
};

} // namespace radixshmem
