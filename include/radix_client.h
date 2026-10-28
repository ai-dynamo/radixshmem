// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "radixshmem_config.h"   // feature macros (CMake-generated, build tree)
#include "component_types.h"
#include "health_check.h"
#include "types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace radixshmem {

struct RemoteHit;

class LocalTreeConnector;
class RadixServer;
#ifndef RADIXSHMEM_NO_RDMA
class RdmaTransportClient;
class RdmaContext;
class PeerRegistry;
class PeerConnector;
class RhtConsumer;   // src/indexer/rht/rht_endpoint.h
class IRdmaEndpoint;
struct PeerEntry;
struct PeerFaultCounters;
struct ShardMapMsg;

class IRemoteOpClientTransport;

struct AllocateReply {
    RemoteStatus status = RemoteStatus::OK;
    InsertError  error  = InsertError::OK;   // MEMPOOL_FULL when status is OK but nothing was allocated
    std::vector<int32_t> slots;   // target-local reserved slot ids (empty on failure)
};

struct QueryReply {
    RemoteStatus status = RemoteStatus::OK;
    int hit_length       = 0;   // B local-tree total matched blocks
    std::vector<int32_t> slots; // target-local physical slot ids (len == hit_length)
};

class RemotePeer {
public:
#ifdef RADIXSHMEM_HAS_ZMQ
    explicit RemotePeer(const std::string& endpoint,
                        int recv_timeout_ms = 5000,
                        int max_retry = 3);
#endif
    RemotePeer(std::unique_ptr<IRemoteOpClientTransport> transport,
               std::string endpoint);

    ~RemotePeer();

    RemotePeer(RemotePeer&&) noexcept;
    RemotePeer& operator=(RemotePeer&&) noexcept;
    RemotePeer(const RemotePeer&) = delete;
    RemotePeer& operator=(const RemotePeer&) = delete;

    AllocateReply allocate(int n);   // pure slot alloc; no hashes, no dedup

    InsertResult insert(const uint64_t* hashes, int n, int start, int length,
                        const int32_t* slots, bool auto_recycle = true);

    QueryReply query(const uint64_t* hashes, int n, bool lock = false,
                     bool update_meta = true);

    // Both return the target's comms status: OK on success, BAD_REQUEST/TIMEOUT/... on failure.
    RemoteStatus recycle(const int32_t* slots, int num_slots);
    RemoteStatus unlock(const uint64_t* hashes, int n, int start, int length);

    const std::string& endpoint() const { return endpoint_; }

    // The peer rank this session talks to (set by peer_for; -1 when built directly).
    int  peer_rank() const { return peer_rank_; }
    void set_peer_rank(int r) { peer_rank_ = r; }

private:
    std::unique_ptr<IRemoteOpClientTransport> transport_;
    std::string endpoint_;
    int         peer_rank_    = -1;
};

#endif  // !RADIXSHMEM_NO_RDMA

// Client config resolving the local shm from /shm/<node_name> and cluster
// topology (peers/config) via etcd.
struct RadixClientConfig {
    std::string registry   = "etcd://127.0.0.1:2379";
    std::string cluster_id = "default";
    std::string node_name  = "";     // identity whose shm to attach (/shm/<node_name>)
    std::string index_dev;            // empty = first available
    int         gid_idx    = 3;
    int         bootstrap_timeout_sec = 60;
    bool        rht_only   = false;  // build only RHT-plane QPs (eviction client)
    bool        no_local_tree = false;  // observer: no local server, rank -1, remote-only

    // Membership gate + settle window (mirror of the server-side gate).
    int         expected_min_nodes = 0;
    int         settle_ms          = 500;
    int         register_chunk_tokens = DEFAULT_REGISTER_CHUNK_TOKENS;  // informational; authoritative cs in /config
};

class RadixClient {
public:
    // Attach to an existing shm region by name. local_read_only=true attaches
    // the tree and nothing else: no etcd, RDMA or RHT plane even when the
    // region belongs to a cluster, so it is cheap to hold in every worker
    // process. Such a client reads only: query() is always local,
    // is_distributed() is false, and insert / allocate_slots / recycle_slots /
    // reset throw, since topology changes made without the RHT plane would
    // leave peers routing to nodes this process never unpublished.
    explicit RadixClient(const std::string& name, bool local_read_only = false);

    // etcd-native form: discover shm via /shm/<node_name>, join cluster from etcd.
    explicit RadixClient(const RadixClientConfig& cfg);

    // Embedded form: piggy-back on a same-process RadixServer.
    explicit RadixClient(RadixServer& server);

    // Embedded eviction service: borrow an already-attached local tree instead
    // of re-attaching the shm. `borrowed` is owned by the caller (not freed here).
    // Builds the RHT plane only (rht_only implied). Distributed always.
    RadixClient(const RadixClientConfig& cfg, LocalTreeConnector* borrowed);

    ~RadixClient();

    RadixClient(const RadixClient&) = delete;
    RadixClient& operator=(const RadixClient&) = delete;

    // ==================== Datapath ====================

    // One call publishes one component; FULL is the positional prefix insert.
    // SWA/MAMBA use component-specific payload and pool contracts.
    InsertResult insert(const uint64_t* hashes, int n,
                        const int32_t* slots, int num_slots,
                        int start = 0, bool auto_recycle = true,
                        ComponentType component = ComponentType::FULL);

    // lock=true pins whole-node coverage, released only via the one-shot
    // result.finalize(); local_only=false adds at most one remote continuation.
    ComponentQueryResult query(const uint64_t* hashes, int n,
                               uint32_t component_mask = COMPONENT_MASK_FULL,
                               bool local_only  = false,
                               bool lock        = false,
                               bool update_meta = true);

    int allocate_slots(int num_slots, std::vector<int32_t>& out,
                       ComponentType component = ComponentType::FULL);
    void recycle_slots(const int32_t* slots, int num,
                       ComponentType component = ComponentType::FULL);

    // Clear cached data in place (node-local tree + owned RHT shards); per-node.
    // Invalidates every slot ID / query result obtained before the call.
    void reset();

    // Distributed-only helpers — no-op on single-node attach.
    void flush();

    // Node-wide counts of index-plane ops lost to unreachable peers; null
    // without a local tree.
    PeerFaultCounters* peer_faults() const;

    // Server-side eviction client: remove RHT entries for locally-evicted nodes
    // (only cells still matching this rank). Empty list ok; flushes at end.
    struct EvictedKey { uint64_t first_hash; uint32_t tree_node_id; };
    void remove_evicted_from_rht(const EvictedKey* entries, int count);
    // Zero every RHT shard THIS rank holds (local memset, no RDMA).
    void reset_owned_rht_shard();

#ifndef RADIXSHMEM_NO_RDMA
    // Remote ops are rank-first and stateless from the caller's view: the client
    // opens/reuses one cached session per rank internally. An illegal rank yields
    // BAD_TARGET / REMOTE_FAILED in the op result (never throws).

    // Phase 1: allocate `n` slots on the peer (no hashes, no dedup).
    AllocateReply remote_allocate(int rank, int n);

    // Phase 2: publish `slots` on the peer; dedup + surplus recycle happen here.
    InsertResult remote_insert(int rank, const uint64_t* hashes, int n,
                               int start, int length, const int32_t* slots,
                               bool auto_recycle = true);

    // Recycle unpublished slots on the peer (abort / auto_recycle=false surplus).
    RemoteStatus remote_recycle(int rank, const int32_t* slots, int num_slots);

    QueryResult remote_query(int rank, const uint64_t* hashes, int n,
                             bool lock = false, bool update_meta = true);

    // Drop a remote_query(lock=true) pin: dec_ref over [0, hit_length).
    RemoteStatus remote_unlock(int rank, const uint64_t* hashes, int n, int hit_length);
#endif

    // ==================== Stats / accessors ====================

    bool is_distributed() const;
    int  rank() const;
    int  world_size() const;

    static std::vector<HashType> compute_hashes(const int64_t* token_ids,
                                                 int num_tokens,
                                                 int tokens_per_block);

    uint32_t total_radix_nodes() const;
    uint64_t total_blocks() const;
    // tokens per block, recovered from the region's ShmHeader.
    int      block_size() const;
    float    data_pool_utilization() const;
    uint32_t mempool_total() const;
    uint32_t mempool_free() const;
    uint32_t mempool_used() const;
    uint32_t swa_mempool_total() const;
    uint32_t mamba_mempool_total() const;
    ComponentConfig component_config() const;   // stamped at create time
    void     print_tree() const;
    const std::string& name() const;

    // Lock-free leak audit of the attached local tree: lock holders, mempool
    // vs tree slot accounting, pins, node accounting. Returns HealthFlag bits
    // (0 == healthy); HEALTH_NOT_ATTACHED for an observer. Never blocks.
    int check_health(HealthReport* report = nullptr) const;

private:
    // `local_` attaches the node-local shm (null for an observer). The cluster
    // machinery below is built (bootstrap_cluster) only when distributed.
    // owns_local_ is false when a same-process RadixServer lends its connector
    // (embedded eviction service): the dtor releases without freeing it.
    std::unique_ptr<LocalTreeConnector> local_;
    bool owns_local_ = true;
    bool local_read_only_ = false;   // read-only attach, never joins the cluster
    void require_writable(const char* op) const;

    // Pin release, reachable only through a query(lock=true) finalize:
    // one PinRecord on the local tree / on peer `remote_node`.
    void dec_ref(const uint64_t* hashes, int n, PinRecord pin);
    void dec_ref_remote(int remote_node, const uint64_t* hashes, int n,
                        PinRecord pin);

#ifndef RADIXSHMEM_NO_RDMA
    // ---- Cluster bring-up + teardown ----
    RadixClientConfig cfg_;            // set on the distributed ctors
    bool distributed_ = false;         // cluster machinery live

    // Run the etcd rendezvous + RDMA/RHT bring-up. Throws on failure.
    void bootstrap_cluster();
    // Ordered teardown of the RDMA/RHT plane (watcher → peers → RHT → transport).
    void teardown_cluster();
    bool wait_for_ready();
    void read_topology();
    IRdmaEndpoint* connect_rht_holder(int holder);
    void admit_peer(const PeerEntry& e);
    void remove_peer(int rank);
    void setup_rht();
    void update_rht(const uint64_t* hashes, int n, const InsertResult& r);
    // Distributed datapath (renamed to avoid colliding with the public facade).
    InsertResult dist_insert(const uint64_t* hashes, int n,
                             const int32_t* slots, int num_slots,
                             int start, bool auto_recycle);
    void         dist_flush();
    // Cached per-rank session; nullptr on illegal rank (self/oob/negative).
    RemotePeer*  peer_for(int rank);

    // RHT probe from the deepest position not past the incumbent, with the
    // chunk-gap early-stop; returns best peer rank (-1 = none) plus its hints.
    int find_component_source(const uint64_t* hashes, int n, int incumbent,
                              std::vector<RemoteHit>& hits_out);
    // Authoritative local query/pin + RHT republish, then at most one remote
    // continuation folded as local prefix + remote suffix.
    ComponentQueryResult dist_query_components(const uint64_t* hashes, int n,
                                               uint32_t component_mask,
                                               bool lock, bool update_meta,
                                               std::vector<PinRecord>& pins);
    // Legacy shape: takes the local public result, returns it (folded) after
    // at most one remote continuation.
    ComponentQueryResult query_components_remote_from(
        ComponentQueryResult result, const uint64_t* hashes, int n,
        uint32_t component_mask, bool lock, std::vector<PinRecord>& pins);

    struct Impl;                       // holds the RegistryClient (etcd topology)
    std::unique_ptr<Impl> etcd_;

    std::unique_ptr<PeerRegistry> peer_reg_;
    uint32_t remote_op_transport_ = 0;   // authoritative control-plane transport (/config)
    int      rdma_sl_ = 5;
    int      rdma_tc_ = 160;
    int      cluster_cs_ = 0;

    std::unique_ptr<RdmaTransportClient> tx_;
    std::unique_ptr<PeerConnector> peer_conn_;

    // Per-rank remote-op sessions, opened lazily by peer_for(), reused across
    // allocate→insert / query→unlock. Cleared before peer_conn_ in teardown.
    std::unordered_map<int, std::shared_ptr<RemotePeer>> peer_cache_;
    std::mutex peer_cache_mu_;
    bool debug_route_ = ::getenv("RADIXSHMEM_DEBUG_ROUTE") != nullptr;
    bool lazy_connect_ = ::getenv("RADIXSHMEM_LAZY_CONNECT") &&
                         ::getenv("RADIXSHMEM_LAZY_CONNECT")[0] == '1';
    void preconnect_peers();

    uint32_t rht_transport_ = 0;         // ClusterTransport, client -> RHT holders
    uint32_t peer_index_transport_ = 0;  // ClusterTransport, remote walks
    uint64_t dc_access_key_ = 0;
    int64_t  peers_snapshot_rev_ = 0;

    // ---- RHT plane (from /config) ----
    int              num_shards_ = 0;
    uint32_t         rht_capacity_ = 0;
    uint32_t         rht_slots_ = 4;
    std::vector<int> holder_of_shard_;
    std::unordered_map<int, std::unique_ptr<IRdmaEndpoint>> rht_holder_qps_;
    std::unique_ptr<RhtConsumer> rht_conn_;
#endif
};

} // namespace radixshmem
