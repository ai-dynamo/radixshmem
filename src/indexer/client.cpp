// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radix_client.h"       // RadixClient / RadixClientConfig / RemotePeer
#include "radix_server.h"       // RadixClient(RadixServer&) ctor

#include "local.h"
#include "cluster_protocol.h"
#include "cluster_manifest.h"
#include "rht_endpoint.h"        // RhtConsumer / RhtTopology (+ rht_types.h)
#include "rht_chain.h"
#include "rdma_remote_walker.h"
#include "rdma_transport.hpp"
#include "rdma_transport_client.hpp"
#include "peer_registry.hpp"
#include "peer_connector.hpp"
#include "xrc_connect.hpp"
#ifdef RADIXSHMEM_HAS_ETCD
#include "etcd_client.hpp"
#include "registry_client.hpp"
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>

namespace radixshmem {

#ifdef RADIXSHMEM_HAS_ETCD
struct RadixClient::Impl {
    std::unique_ptr<RegistryClient> reg;
};
#else
struct RadixClient::Impl {};
#endif

// ==================== Cluster plane ====================

void RadixClient::teardown_cluster() {
    // Stop the /peers watcher first: its callback admits into the registry and
    // resizes the connector's slots, so it must be joined before RDMA teardown.
#ifdef RADIXSHMEM_HAS_ETCD
    if (etcd_ && etcd_->reg) etcd_->reg->stop_watch();
#endif
    // Cached per-rank sessions hold DC AHs + reference the connector's shared
    // DCI/DCT, so they must die before the connector.
    { std::lock_guard<std::mutex> g(peer_cache_mu_); peer_cache_.clear(); }
    // Connector (data-plane endpoints + DC control stack) before the RHT plane,
    // then the RHT plane, then the transport. DcReaders reference the shared DCI
    // + AHs the transport frees in its dtor, so the transport must outlive them.
    peer_conn_.reset();
    rht_conn_.reset();
    rht_holder_qps_.clear();
    tx_.reset();
}

bool RadixClient::wait_for_ready() {
#ifdef RADIXSHMEM_HAS_ETCD
    // Same-host fast path: spin the local ready flag briefly, then defer to etcd.
    // Observers have no local shm, so they go straight to etcd.
    if (local_) {
        auto* hdr = static_cast<ShmHeader*>(local_->tree().shm_base());
        for (int i = 0; i < 50; i++) {
            if (hdr->cluster_ready_magic.load(std::memory_order_acquire)
                    == CLUSTER_MANIFEST_MAGIC)
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    return etcd_->reg->wait_for_ready();
#else
    return false;
#endif
}

void RadixClient::admit_peer(const PeerEntry& e) {
    if (!peer_reg_->admit(e) || e.rank == peer_reg_->my_rank()) return;
    drop_peer_session(e.rank);
    if (rht_conn_) rht_conn_->holder_changed(e);
    if (!peer_conn_) return;
    peer_conn_->peer_changed(e.rank);
    if (!lazy_connect_ && !cfg_.rht_only && !peer_conn_->preconnect(e.rank))
        fprintf(stderr, "[RadixClient rank %d] preconnect to peer rank %d failed\n",
                peer_reg_->my_rank(), e.rank);
}

void RadixClient::remove_peer(int rank) {
    if (rank == peer_reg_->my_rank()) return;
    peer_reg_->remove(rank);
    if (rht_conn_) rht_conn_->holder_left(rank);
    if (peer_conn_) peer_conn_->peer_left(rank);
}

void RadixClient::preconnect_peers() {
    if (lazy_connect_ || cfg_.rht_only) return;
    int me = peer_reg_->my_rank(), ws = peer_reg_->world_size();
    for (int r = 0; r < ws; r++) {
        if (r == me) continue;
        {
            std::shared_lock<std::shared_mutex> lk(peer_reg_->mutex());
            if (peer_reg_->peers_locked()[r].rank != r) continue;   // node down
        }
        if (!peer_conn_->preconnect(r))
            fprintf(stderr, "[RadixClient rank %d] preconnect to peer rank %d failed\n",
                    me, r);
    }
}

void RadixClient::read_topology() {
#ifdef RADIXSHMEM_HAS_ETCD
    ClusterTopology t = etcd_->reg->read_topology();

    const ShardMapMsg& sm = t.sm;
    rht_transport_        = sm.rht_transport;
    peer_index_transport_ = sm.peer_index_transport;
    dc_access_key_       = sm.dc_access_key;
    remote_op_transport_ = sm.remote_op_transport;
    rdma_sl_             = sm.rdma_sl;
    rdma_tc_             = sm.rdma_tc;
    cluster_cs_          = (int)sm.register_chunk_size;

    num_shards_   = (int)sm.num_shards;
    rht_capacity_ = sm.rht_capacity;
    rht_slots_    = sm.rht_slots ? sm.rht_slots : 4;
    holder_of_shard_.assign(num_shards_, 0);
    for (int s = 0; s < num_shards_; s++)
        holder_of_shard_[s] = (int)sm.holder_rank[s];

    if (t.my_rank < 0 && !cfg_.no_local_tree)
        throw std::runtime_error("RadixClient: own node '" + cfg_.node_name +
                                 "' not found in /peers");
    peer_reg_->seed(std::move(t.peers), t.world_size, t.my_rank);
    peers_snapshot_rev_  = t.peers_rev;
#else
    throw std::runtime_error("RadixClient: built without etcd");
#endif
}

IRdmaEndpoint* RadixClient::connect_rht_holder(int holder) {
    auto it = rht_holder_qps_.find(holder);
    if (it != rht_holder_qps_.end()) return it->second.get();

    // The watcher may reallocate the peer table; lock around reads and writes.
    std::unique_lock<std::shared_mutex> lk(peer_reg_->mutex());
    auto it2 = rht_holder_qps_.find(holder);
    if (it2 != rht_holder_qps_.end()) return it2->second.get();

    const PeerEntry& h = peer_reg_->peers_locked()[holder];
    int my_rank = peer_reg_->my_rank();
    std::unique_ptr<IRdmaEndpoint> ep;

    if (rht_transport_ == CLUSTER_TRANSPORT_DC) {
        DcInitiator* dci = tx_->shared_dci();
        ibv_gid dgid;
        memcpy(&dgid, h.gid_raw, 16);
        int ws = peer_reg_->world_size();
        ibv_ah* ah = tx_->ah_for(holder, dgid, h.lid, h.link_layer, ws);
        if (!ah) {
            fprintf(stderr,
                    "[RadixClient rank %d] RHT-plane AH to holder rank %d failed\n",
                    my_rank, holder);
            return nullptr;
        }
        DcTargetInfo dst{ah, h.dct_num, dc_access_key_};
        ep = std::make_unique<DcReader>(dci, dst);
    } else {
        // One try: a holder that does not answer is skipped and probed later
        // by the RHT fault handling, at attach as on the query path.
        ep = xrc_connect(tx_->ctx(), tx_->cq(), h.host, h.xrc_listen_port, my_rank,
                         rht_conn_ ? 2 : 10);
        if (!ep) return nullptr;
    }

    IRdmaEndpoint* raw = ep.get();
    rht_holder_qps_.emplace(holder, std::move(ep));
    return raw;
}

void RadixClient::setup_rht() {
    RhtTopology topo;
    topo.num_shards       = num_shards_;
    topo.capacity         = rht_capacity_;
    topo.slots_per_bucket = rht_slots_;
    topo.my_rank          = peer_reg_->my_rank();
    topo.holder_of_shard  = holder_of_shard_;
    topo.peers            = &peer_reg_->peers_locked();
    topo.faults           = peer_faults();

    rht_conn_ = std::make_unique<RhtConsumer>(
        topo, tx_->ctx().pd(),
        [this](int holder) { return connect_rht_holder(holder); },
        [this](int holder) {
            std::unique_lock<std::shared_mutex> lk(peer_reg_->mutex());
            rht_holder_qps_.erase(holder);
        },
        cfg_.bootstrap_timeout_sec);
}

// Build the etcd rendezvous + RDMA/RHT plane. Constructs peer_reg_/etcd_ then
// runs the topology read + transport bring-up. Sets distributed_ on success;
// leaves it false (single-node local shm) or throws on failure.
void RadixClient::bootstrap_cluster() {
    peer_reg_ = std::make_unique<PeerRegistry>();
    etcd_     = std::make_unique<Impl>();

    // A local server may declare itself single-node; observers are always
    // distributed (they exist only to reach remote peers).
    if (local_) {
        auto* hdr = static_cast<ShmHeader*>(local_->tree().shm_base());
        if (hdr->is_distributed == 0) {
            peer_reg_->set_my_rank(0);
            return;   // single-node: distributed_ stays false
        }
    }
#ifdef RADIXSHMEM_HAS_ETCD
    RegistryClientConfig rc;
    rc.registry             = cfg_.registry;
    rc.cluster_id           = cfg_.cluster_id;
    rc.node_name            = cfg_.node_name;
    rc.expected_min_nodes   = cfg_.expected_min_nodes;
    rc.bootstrap_timeout_sec = cfg_.bootstrap_timeout_sec;
    etcd_->reg = std::make_unique<RegistryClient>(rc);
    etcd_->reg->connect();
#else
    throw std::runtime_error(
        "RadixClient: distributed shm but built without etcd");
#endif
    if (!wait_for_ready())
        throw std::runtime_error("RadixClient: cluster not ready (etcd timeout)");
    read_topology();

    int my_rank = peer_reg_->my_rank();

    // Use the server-advertised index_dev / gid_idx unless caller overrode.
    // Observers (rank < 0) have no /peers entry, so they rely on config only.
    std::string dev_str = cfg_.index_dev;
    int          gid_idx = cfg_.gid_idx;
    if (my_rank >= 0) {
        PeerEntry self;
        {
            std::shared_lock<std::shared_mutex> lk(peer_reg_->mutex());
            self = peer_reg_->peers_locked()[my_rank];
        }
        if (dev_str.empty() && self.index_dev[0] != '\0') dev_str = self.index_dev;
        if (self.gid_index != gid_idx) gid_idx = self.gid_index;
    }
    const char* dev = dev_str.empty() ? nullptr : dev_str.c_str();
    int ws = peer_reg_->world_size();
    // QoS (rank 0's SL/TC) is adopted inside the transport before any QP connect.
    tx_ = std::make_unique<RdmaTransportClient>(dev, gid_idx, rdma_sl_, rdma_tc_, ws);

    // Peer plane: lazy data-plane endpoints + remote-op factory. Its slots track
    // the peer table via the registry grow hook.
    PeerConnectorConfig pcfg;
    pcfg.transport            = peer_index_transport_;
    pcfg.remote_op_transport  = remote_op_transport_;
    pcfg.dc_access_key        = dc_access_key_;
    pcfg.bootstrap_timeout_sec = cfg_.bootstrap_timeout_sec;
    pcfg.faults               = peer_faults();
    peer_conn_ = std::make_unique<PeerConnector>(*peer_reg_, *tx_, pcfg);
    if (local_) {
        peer_conn_->set_expected_component_config(
            local_->tree().component_config());
    } else {
        // Observer: adopt the config only if every peer publishes the same
        // self-valid tuple; otherwise component walks stay fail-closed.
        std::shared_lock<std::shared_mutex> lk(peer_reg_->mutex());
        const auto& ps = peer_reg_->peers_locked();
        const PeerEntry* ref = nullptr;
        bool consensus = true;
        for (size_t i = 0; consensus && i < ps.size(); i++) {
            if (ps[i].rank < 0) continue;   // node down
            if (!ref) ref = &ps[i];
            consensus = peer_component_abi_valid(ps[i], ref->component_config);
        }
        consensus = consensus && ref;
        if (consensus)
            peer_conn_->set_expected_component_config(ref->component_config);
        else
            fprintf(stderr,
                    "[RadixClient observer] no component-config consensus "
                    "across %d peer(s); component queries disabled\n",
                    (int)ps.size());
    }
    peer_conn_->resize(ws);
    peer_reg_->set_grow_hook([this](int n) { peer_conn_->resize(n); });

    if (rht_transport_ == CLUSTER_TRANSPORT_DC ||
        (!cfg_.rht_only && peer_index_transport_ == CLUSTER_TRANSPORT_DC)) {
        tx_->shared_dci();
        fprintf(stderr, "[RadixClient rank %d] DC initiator up\n", my_rank);
    }
    if (cfg_.rht_only) {
        fprintf(stderr,
                "[RadixClient rank %d] RHT-only mode (no data-plane QPs)\n",
                my_rank);
    }

    setup_rht();
    preconnect_peers();

    // Watch /peers for members that join, restart or leave. Anchor at the
    // snapshot rev so a PUT before establishment isn't lost.
#ifdef RADIXSHMEM_HAS_ETCD
    etcd_->reg->start_watch(peers_snapshot_rev_ + 1,
                            [this](const PeerEntry& e) { admit_peer(e); },
                            [this](int rank) { remove_peer(rank); });
#endif

    // Evict/merge on the connector's own tree must clean RHT, else the recycled node id orphans a stale anchor.
    if (local_) {
        local_->tree().set_evict_callback(
            [this](const std::vector<EvictedNode>& evicted) {
                std::vector<EvictedKey> keys;
                keys.reserve(evicted.size());
                for (const auto& e : evicted)
                    keys.push_back({e.first_hash, e.tree_node_id});
                remove_evicted_from_rht(keys.data(), (int)keys.size());
            });
    }

    distributed_ = true;
}

// ==================== Datapath ====================

void RadixClient::update_rht(const uint64_t* hashes, int n, const InsertResult& r) {
    // RHT write half: needs a local tree to derive boundaries and a valid rank
    // to stamp ownership. Observers (rank < 0) publish nothing.
    int my_rank = peer_reg_->my_rank();
    if (my_rank < 0 || !local_) return;
    auto& tree = local_->tree();
    auto boundaries = tree.get_node_boundaries(hashes, n);
    if (boundaries.empty()) return;
    std::vector<RhtInsertEntry> entries;
    entries.reserve(boundaries.size() + 1);
    for (auto& b : boundaries)
        entries.push_back({b.first_hash, (int32_t)my_rank, b.tree_node_id,
                           b.node_blocks});
    if (r.split_node_id != INVALID_NODE_ID)   // suffix split off the [0, n) path
        entries.push_back({r.split_first_hash, (int32_t)my_rank, r.split_node_id,
                           r.split_num_blocks});
    rht_conn_->insert_batch(entries.data(), (int)entries.size());
}

InsertResult RadixClient::dist_insert(const uint64_t* hashes, int n,
                                      const int32_t* slots, int num_slots,
                                      int start, bool auto_recycle) {
    auto& tree = local_->tree();
    // Node boundaries align to cs so peer RHT probes land on a node first_hash.
    InsertResult result = tree.insert(hashes, n, slots, num_slots,
                                      start, auto_recycle);

    int my_rank = peer_reg_->my_rank();
    // Republish the whole root→leaf chain (idempotent), so merge/collision can't leave any node unrouted.
    if (rht_conn_ && my_rank >= 0 && result.error == InsertError::OK) {
        std::vector<NodeBoundary> chain = tree.get_node_boundaries(hashes, n);
        if (!chain.empty()) {
            std::vector<RhtInsertEntry> entries;
            entries.reserve(chain.size() + 1);
            for (const auto& b : chain)
                entries.push_back({b.first_hash, (int32_t)my_rank,
                                   b.tree_node_id, b.node_blocks});
            if (result.split_node_id != INVALID_NODE_ID)   // suffix off the path
                entries.push_back({result.split_first_hash, (int32_t)my_rank,
                                   result.split_node_id, result.split_num_blocks});
            rht_conn_->insert_batch(entries.data(), (int)entries.size());
            if (debug_route_) {
                fprintf(stderr, "[pub r%d] n=%d start=%d h0=%08llx chain:",
                        my_rank, n, start, (unsigned long long)(hashes[0] & 0xffffffffu));
                for (const auto& b : chain)
                    fprintf(stderr, " %d+%d:%08llx:t%u/%u", b.offset, b.num_blocks,
                            (unsigned long long)(b.first_hash & 0xffffffffu),
                            b.tree_node_id, b.node_blocks);
                fprintf(stderr, "\n");
            }
        }
    }
    return result;
}

void RadixClient::dist_flush() { if (rht_conn_) rht_conn_->flush(); }

PeerFaultCounters* RadixClient::peer_faults() const {
    return local_ ? local_->tree().peer_faults() : nullptr;
}

void RadixClient::remove_evicted_from_rht(const EvictedKey* entries, int count) {
    if (!rht_conn_ || count <= 0) return;
    int my_rank = peer_reg_->my_rank();
    if (my_rank < 0) return;   // write half: observers publish nothing
    std::vector<RhtRemoveEntry> rem;
    rem.reserve(count);
    for (int i = 0; i < count; i++) {
        // Skip ids already re-registered for the same prefix (LIFO reuse
        // between evict and this removal): the cell is live again.
        if (local_ && local_->tree().node_head_matches(entries[i].tree_node_id,
                                                       entries[i].first_hash)) {
            if (debug_route_)
                fprintf(stderr, "[unevict r%d] keep %08llx:t%u (reborn)\n", my_rank,
                        (unsigned long long)(entries[i].first_hash & 0xffffffffu),
                        entries[i].tree_node_id);
            continue;
        }
        rem.push_back({entries[i].first_hash, (int32_t)my_rank,
                       entries[i].tree_node_id});
    }
    if (rem.empty()) return;
    rht_conn_->remove_batch(rem.data(), (int)rem.size());
    rht_conn_->flush();
}

void RadixClient::reset_owned_rht_shard() {
    if (rht_conn_ && peer_reg_->my_rank() >= 0) rht_conn_->reset_owned_shard();
}

// Lazily open (and cache) the per-rank session. Returns nullptr on any illegal
// rank (single-node, self, out of range, remote ops disabled) — callers map that
// to BAD_TARGET / REMOTE_FAILED, never a throw.
std::shared_ptr<RemotePeer> RadixClient::peer_for(int rank) {
    if (!distributed_ || !peer_conn_) return nullptr;
    std::lock_guard<std::mutex> g(peer_cache_mu_);
    auto it = peer_cache_.find(rank);
    if (it != peer_cache_.end()) return it->second;
    try {
        auto rc = peer_conn_->open_remote(rank);
        peer_cache_.emplace(rank, rc);
        return rc;
    } catch (const std::exception&) {
        return nullptr;   // self / oob / disabled peer
    }
}

// The rank re-registered with a new entry: the next op reopens from it; in-flight ops keep the old session alive.
void RadixClient::drop_peer_session(int rank) {
    std::lock_guard<std::mutex> g(peer_cache_mu_);
    peer_cache_.erase(rank);
}



// =====================================================================
// RadixClient
// =====================================================================

RadixClient::RadixClient(const std::string& name, bool local_read_only)
    : local_(std::make_unique<LocalTreeConnector>(name)),
      local_read_only_(local_read_only) {
    if (local_read_only_) return;   // tree only; the cluster plane is never built
#ifndef RADIXSHMEM_NO_RDMA
    // The shm header carries the cluster identity; bootstrap_cluster reads
    // is_distributed and either sets distributed_ or returns single-node.
    auto* hdr = static_cast<ShmHeader*>(local_->tree().shm_base());
    cfg_.registry   = hdr->registry;
    cfg_.cluster_id = hdr->cluster_id;
    cfg_.node_name  = hdr->node_name;
    bootstrap_cluster();
#endif
}

void RadixClient::require_writable(const char* op) const {
    if (local_read_only_)
        throw std::runtime_error(std::string("RadixClient::") + op +
                                 ": local_read_only client cannot write");
}

namespace {
#ifdef RADIXSHMEM_HAS_ETCD
// Resolve /shm/<node_name> → local shm name via etcd.
std::string resolve_shm_name(const RadixClientConfig& cfg) {
    etcd::Client cli;
    cli.init(etcd::strip_scheme(cfg.registry));
    etcd::Keys keys;
    keys.cluster_id = cfg.cluster_id;
    std::string shm;
    if (!etcd::poll_until(cfg.bootstrap_timeout_sec,
                          [&] { return cli.get(keys.shm(cfg.node_name), shm); }))
        throw std::runtime_error(
            "RadixClient: /shm/" + cfg.node_name + " not published");
    return shm;
}
#endif
} // namespace

RadixClient::RadixClient(const RadixClientConfig& cfg)
#ifdef RADIXSHMEM_HAS_ETCD
    : cfg_(cfg)
{
    // Observer (no local server): skip the shm attach, join the cluster as a
    // remote-only reader with rank -1.
    if (!cfg_.no_local_tree)
        local_ = std::make_unique<LocalTreeConnector>(resolve_shm_name(cfg_));
#ifndef RADIXSHMEM_NO_RDMA
    // Observers are always distributed; a local server may be single-node.
    // bootstrap_cluster resolves is_distributed off the attached header.
    bootstrap_cluster();
#endif
}
#else
{
    throw std::runtime_error(
        "RadixClient: etcd-config ctor requires an etcd-enabled build");
}
#endif

RadixClient::RadixClient(RadixServer& server)
    : RadixClient(server.shm_name()) {
    // Embedded form == name-attach form; client builds its own RDMA plane.
}

RadixClient::RadixClient(const RadixClientConfig& cfg, LocalTreeConnector* borrowed)
    : local_(borrowed), owns_local_(false), cfg_(cfg) {
    // Borrow the caller's already-attached tree; build the RHT plane only.
    cfg_.rht_only = true;
    bootstrap_cluster();
}

RadixClient::~RadixClient() {
#ifndef RADIXSHMEM_NO_RDMA
    teardown_cluster();
#endif
    // Borrowed local tree is owned by the caller — don't free it.
    if (!owns_local_) (void)local_.release();
}

InsertResult RadixClient::insert(const uint64_t* hashes, int n,
                                 const int32_t* slots, int num_slots,
                                 int start, bool auto_recycle,
                                 ComponentType component) {
    if (!local_)
        throw std::runtime_error(
            "RadixClient::insert: no local tree (observer client is read-only)");
    require_writable("insert");
#ifndef RADIXSHMEM_NO_RDMA
    if (distributed_ && component == ComponentType::FULL)
        return dist_insert(hashes, n, slots, num_slots, start, auto_recycle);
#endif
    InsertResult result = local_->insert(hashes, n, slots, num_slots, start,
                                         auto_recycle, component);
#ifndef RADIXSHMEM_NO_RDMA
    if (distributed_ && component != ComponentType::FULL &&
        result.topology_changed)
        update_rht(hashes, n, result);
#endif
    return result;
}

// Probes local node boundaries + cs-aligned positions (the latter flagged
// for the gap early-stop); scored from the deepest probe not past incumbent,
// then the winner's route chain is followed to its end (rht_chain.h).
int RadixClient::find_component_source(const uint64_t* hashes, int n,
                                       int incumbent,
                                       std::vector<RemoteHit>& hits_out) {
    // Probe stride aligns to cs; /config is authoritative, local shm is fallback.
    int local_cs = local_ ? local_->tree().register_chunk_size() : 0;
    int cs = cluster_cs_ > 0 ? cluster_cs_ : local_cs;
    assert(!local_ || cluster_cs_ <= 0 || cluster_cs_ == local_cs);
    if (cs <= 0) cs = 1;   // keeps the stride sane for observers w/o /config

    int my_rank = peer_reg_->my_rank();
    int local_hit = incumbent;
    int best_remote_node = -1;
    std::vector<RemoteHit> remote_hits;

    if (local_hit < n) {
        std::vector<std::pair<int, bool>> probes;
        if (local_) {
            auto local_bounds = local_->tree().get_node_boundaries(hashes, n);
            for (auto& b : local_bounds)
                probes.push_back({b.offset, b.offset % cs == 0});
        }
        for (int pos = 0; pos < n; pos += cs)
            probes.push_back({pos, true});

        std::sort(probes.begin(), probes.end());
        std::vector<std::pair<int, bool>> merged;
        for (auto& p : probes) {
            if (!merged.empty() && merged.back().first == p.first)
                merged.back().second = merged.back().second || p.second;
            else
                merged.push_back(p);
        }
        probes = std::move(merged);

        // Probes from the deepest one <= local_hit on, plus the deepest
        // cs-aligned one <= local_hit: peers cut nodes on cs multiples, so
        // that key always names a peer node when the peer holds the prefix.
        int start_idx = 0;
        for (int i = (int)probes.size() - 1; i >= 0; i--) {
            if (probes[i].first <= local_hit) {
                start_idx = i; break;
            }
        }
        std::vector<int> sel;
        for (int i = start_idx - 1; i >= 0; i--) {
            if (probes[i].second) { sel.push_back(i); break; }
        }
        for (int i = start_idx; i < (int)probes.size(); i++) {
            if (probes[i].first >= n) break;
            sel.push_back(i);
        }
        int probe_count = (int)sel.size();
        std::vector<uint64_t> probe_keys(probe_count);
        for (int i = 0; i < probe_count; i++)
            probe_keys[i] = hashes[probes[sel[i]].first];

        // Each probe may return multiple candidates (W-way bucket chaining).
        std::vector<std::vector<RHTValue>> probe_results(probe_count);
        rht_conn_->lookup_batch(probe_keys.data(), probe_count,
                                probe_results.data());

        // Collect remote candidates per position; tally positions hit per peer.
        struct FullHit { int pos; uint32_t tnid; int32_t remote; uint32_t len; };
        std::vector<FullHit> all_hits;
        std::unordered_map<int, int> peer_hit_positions;  // peer → #positions
        int chunk_misses = 0;
        for (int i = 0; i < probe_count; i++) {
            int pos = probes[sel[i]].first;
            bool is_chunk = probes[sel[i]].second;
            bool any_remote_here = false;
            // De-dup peers within this position so one peer counts once per pos.
            std::unordered_map<int, size_t> seen_here;  // peer → all_hits index
            for (auto& v : probe_results[i]) {
                if (v.empty() || v.node_id == my_rank) continue;
                any_remote_here = true;
                auto it = seen_here.find(v.node_id);
                if (it == seen_here.end()) {
                    seen_here[v.node_id] = all_hits.size();
                    all_hits.push_back({pos, v.tree_node_id, v.node_id,
                                        v.num_blocks});
                    peer_hit_positions[v.node_id]++;
                } else if (v.num_blocks > all_hits[it->second].len) {
                    all_hits[it->second].tnid = v.tree_node_id;
                    all_hits[it->second].len  = v.num_blocks;
                }
            }
            // Stop only if no remote node found yet: a cs-aligned probe can
            // land inside a node (RHT holds only first_hash).
            if (!any_remote_here && is_chunk && all_hits.empty() &&
                ++chunk_misses >= 2)
                break;
        }

        // Deepest probe position wins (hash chains: a hit at pos proves the
        // whole prefix). Among peers there: longest node, then most positions
        // covered, then lowest rank.
        int deepest = -1;
        for (auto& h : all_hits) deepest = std::max(deepest, h.pos);
        uint32_t best_len = 0; int best_cov = -1;
        for (auto& h : all_hits) {
            if (h.pos != deepest) continue;
            int cov = peer_hit_positions[h.remote];
            if (best_remote_node < 0 || h.len > best_len ||
                (h.len == best_len &&
                 (cov > best_cov ||
                  (cov == best_cov && h.remote < best_remote_node)))) {
                best_len = h.len; best_cov = cov; best_remote_node = h.remote;
            }
        }
        // Follow the winner's chain by RHT lengths from its deepest hit: the
        // chain end decides miss-without-walk; the hops become walker hints.
        int chain_end = -1;
        std::vector<RhtHop> hops;
        if (best_remote_node >= 0) {
            std::unordered_map<uint64_t, FullHit> known;
            for (auto& h : all_hits)
                if (h.remote == best_remote_node) {
                    remote_hits.push_back({h.pos, h.tnid});
                    known.emplace(hashes[h.pos], h);
                }
            auto lookup = [&](uint64_t key, RhtHop& hop) {
                auto it = known.find(key);
                if (it != known.end()) {
                    hop.tree_node_id = it->second.tnid;
                    hop.num_blocks = it->second.len;
                    return true;
                }
                std::vector<RHTValue> vals;
                rht_conn_->lookup_batch(&key, 1, &vals);
                for (auto& v : vals)
                    if (!v.empty() && v.node_id == best_remote_node &&
                        v.num_blocks > hop.num_blocks) {
                        hop.tree_node_id = v.tree_node_id;
                        hop.num_blocks = v.num_blocks;
                    }
                return hop.num_blocks > 0;
            };
            chain_end = follow_rht_chain(hashes, n, deepest, best_len, lookup, hops);
            for (auto& h : hops) remote_hits.push_back({h.position, h.tree_node_id});
            if (chain_end <= local_hit) {
                best_remote_node = -1;
                remote_hits.clear();
            }
        }
        if (debug_route_) {
            fprintf(stderr, "[route r%d] n=%d inc=%d h0=%08llx probes:", my_rank, n,
                    incumbent, (unsigned long long)(hashes[0] & 0xffffffffu));
            for (int i = 0; i < probe_count; i++) {
                int pos = probes[sel[i]].first;
                fprintf(stderr, " %d%s:%08llx=[", pos,
                        probes[sel[i]].second ? "c" : "",
                        (unsigned long long)(hashes[pos] & 0xffffffffu));
                for (auto& v : probe_results[i])
                    fprintf(stderr, "r%d:t%u/%u,", v.node_id, v.tree_node_id,
                            v.num_blocks);
                fprintf(stderr, "]");
                if (::getenv("RADIXSHMEM_DEBUG_ROUTE")[0] == '2') {
                    std::vector<RhtRawCell> raw;
                    rht_conn_->dump_bucket(probe_keys[i], raw);
                    fprintf(stderr, "{");
                    for (auto& c : raw)
                        fprintf(stderr, "%016llx/%016llx ", (unsigned long long)c.head,
                                (unsigned long long)c.ext);
                    fprintf(stderr, "}");
                }
            }
            fprintf(stderr, " chain_end=%d (+%zu hops) -> best=r%d\n", chain_end,
                    hops.size(), best_remote_node);
        }
    }

    hits_out = std::move(remote_hits);
    return best_remote_node;
}

// Tree-level → public mapping; the Full run stays uncropped at [0, f) so a
// remote fold can cut it at c — the component query exit crops to common_hit.
static ComponentQueryResult to_public(TreeComponentQueryResult&& r,
                                      int32_t source_rank,
                                      std::vector<PinRecord>& pins) {
    ComponentQueryResult o;
    o.status = r.status;
    if (r.status != RemoteStatus::OK) return o;
    o.common_hit = (int)r.common_hit;
    if (!r.full_slots.empty())
        o.full_fragments.push_back(
            ComponentSlotFragment{source_rank, 0, std::move(r.full_slots)});
    o.swa = ComponentSlotFragment{r.swa_slots.empty() ? -1 : source_rank,
                             (int)r.swa_start, std::move(r.swa_slots)};
    if (r.mamba_checkpoint_slot != INVALID_COMPONENT_SLOT)
        o.mamba = ComponentCheckpoint{source_rank, r.mamba_checkpoint_slot};
    if (!r.pin.empty())
        pins.push_back(std::move(r.pin));   // source_rank stays -1: local
    return o;
}

ComponentQueryResult RadixClient::query(const uint64_t* hashes, int n,
                                        uint32_t component_mask,
                                        bool local_only, bool lock,
                                        bool update_meta) {
    ComponentQueryResult result;
    std::vector<PinRecord> pins;
    if (component_mask == 0 ||
        (component_mask & ~COMPONENT_MASK_ALL) != 0 ||
        n < 0 || (n > 0 && hashes == nullptr)) {
        result.status = RemoteStatus::BAD_REQUEST;
        return result;
    }
#ifndef RADIXSHMEM_NO_RDMA
    if (distributed_ && !local_only) {
        result = dist_query_components(hashes, n, component_mask, lock,
                                       update_meta, pins);
    } else
#endif
    {
        if (!local_) {
            // local-only query needs a local tree (observer client is
            // remote-only); components fail closed instead of throwing.
            result.status = RemoteStatus::BAD_REQUEST;
            return result;
        }
        result = to_public(local_->tree().query_components(
                               hashes, n, component_mask, lock, update_meta),
                           (int32_t)rank(), pins);
    }

    // The local run is uncropped [0, f); crop to the final common_hit (a
    // remote fold already cut its runs to c/L and never exceeds it).
    if (result.full_fragments.size() == 1 &&
        result.full_fragments[0].slot_ids.size() > (size_t)result.common_hit) {
        result.full_fragments[0].slot_ids.resize((size_t)result.common_hit);
        if (result.full_fragments[0].slot_ids.empty())
            result.full_fragments.clear();
    }

    if (lock && !pins.empty()) {
        // finalize dec_refs every pinned tree, routed by source_rank.
        std::vector<uint64_t> hashes_copy(hashes, hashes + n);
        auto* self = this;
        result.finalize = FinalizeFn([self,
                                       hc = std::move(hashes_copy),
                                       pins = std::move(pins)]() {
            int hn = (int)hc.size();
            for (const PinRecord& p : pins) {
                if (p.source_rank < 0)
                    self->dec_ref(hc.data(), hn, p);
                else
                    self->dec_ref_remote(p.source_rank, hc.data(), hn, p);
            }
        });
    }
    return result;
}

#ifndef RADIXSHMEM_NO_RDMA

ComponentQueryResult RadixClient::dist_query_components(
        const uint64_t* hashes, int n, uint32_t component_mask, bool lock,
        bool update_meta, std::vector<PinRecord>& pins) {
    // Observers have no local tree: their query is purely remote (RHT route +
    // peer walk over the walker engine).
    if (!local_)
        return query_components_remote_from(ComponentQueryResult{}, hashes, n,
                                            component_mask, lock, pins);

    // Local query; update_meta bumps local LRU.
    TreeComponentQueryResult result = local_->tree().query_components(
        hashes, n, component_mask, lock, update_meta);
    if (result.status != RemoteStatus::OK) {
        ComponentQueryResult out;
        out.status = result.status;
        return out;
    }

    return query_components_remote_from(
        to_public(std::move(result), (int32_t)rank(), pins), hashes, n,
        component_mask, lock, pins);
}

ComponentQueryResult RadixClient::query_components_remote_from(
        ComponentQueryResult result, const uint64_t* hashes, int n,
        uint32_t component_mask, bool lock, std::vector<PinRecord>& pins) {
    // The local run is still uncropped: h = joint incumbent, f = Full extent.
    const int h = result.common_hit;
    const int f = result.full_fragments.empty()
                      ? 0
                      : (int)result.full_fragments[0].slot_ids.size();
    if (!rht_conn_ || h >= n) {
        return result;
    }

    std::vector<RemoteHit> remote_hits;
    int best_remote_node = find_component_source(hashes, n, h, remote_hits);

    if (best_remote_node >= 0 &&
        best_remote_node < peer_reg_->world_size() &&
        !remote_hits.empty()) {
        return peer_conn_->walk_components(best_remote_node, hashes, n,
                                           component_mask, lock, h, f,
                                           remote_hits, std::move(result),
                                           pins);
    }
    return result;
}

#endif  // !RADIXSHMEM_NO_RDMA

int RadixClient::allocate_slots(int num_slots, std::vector<int32_t>& out,
                                ComponentType component) {
    if (!local_)
        throw std::runtime_error(
            "RadixClient::allocate_slots: no local tree (observer client)");
    require_writable("allocate_slots");
    return local_->allocate_slots(num_slots, out, component);
}

void RadixClient::recycle_slots(const int32_t* slots, int num,
                                ComponentType component) {
    if (!local_)
        throw std::runtime_error(
            "RadixClient::recycle_slots: no local tree (observer client)");
    require_writable("recycle_slots");
    local_->recycle_slots(slots, num, component);
}

void RadixClient::reset() {
    require_writable("reset");
    if (local_) local_->reset();
#ifndef RADIXSHMEM_NO_RDMA
    if (distributed_) reset_owned_rht_shard();
#endif
}


void RadixClient::dec_ref(const uint64_t* hashes, int n, PinRecord pin) {
    if (!local_) return;   // observer: nothing local to unpin
    (void)local_->tree().dec_ref(hashes, n, std::move(pin));
}

void RadixClient::flush() {
#ifndef RADIXSHMEM_NO_RDMA
    if (distributed_) dist_flush();
#endif
}

void RadixClient::dec_ref_remote(int remote_node, const uint64_t* hashes,
                                 int n, PinRecord pin) {
#ifndef RADIXSHMEM_NO_RDMA
    if (distributed_ && peer_conn_)
        (void)peer_conn_->dec_ref_components_remote(remote_node, hashes, n,
                                                    std::move(pin));
#endif
    (void)remote_node; (void)hashes; (void)n;
}

#ifndef RADIXSHMEM_NO_RDMA

AllocateReply RadixClient::remote_allocate(int rank, int n) {
    AllocateReply out;
    if (n <= 0) return out;                       // nothing to allocate, status OK
    auto p = peer_for(rank);
    if (!p) { out.status = RemoteStatus::BAD_TARGET; return out; }
    return p->allocate(n);            // serializes + caps internally
}

InsertResult RadixClient::remote_insert(int rank,
                                        const uint64_t* hashes, int n,
                                        int start, int length,
                                        const int32_t* slots, bool auto_recycle) {
    InsertResult out;
    auto p = peer_for(rank);
    if (!p) {
        out.error = InsertError::REMOTE_FAILED;
        out.remote_status = RemoteStatus::BAD_TARGET;
        return out;
    }
    return p->insert(hashes, n, start, length, slots, auto_recycle);
}

RemoteStatus RadixClient::remote_recycle(int rank, const int32_t* slots, int num_slots) {
    if (!slots || num_slots <= 0) return RemoteStatus::BAD_REQUEST;
    auto p = peer_for(rank);
    if (!p) return RemoteStatus::BAD_TARGET;
    return p->recycle(slots, num_slots);
}

QueryResult RadixClient::remote_query(int rank, const uint64_t* hashes, int n,
                                      bool lock, bool update_meta) {
    QueryResult out;
    if (n <= 0) return out;                       // empty input, status OK
    auto p = peer_for(rank);
    if (!p) { out.status = RemoteStatus::BAD_TARGET; return out; }
    QueryReply rep = p->query(hashes, n, lock, update_meta);

    out.status            = rep.status;
    out.total_hit_length  = rep.hit_length;
    out.remote_hit_length = rep.hit_length;
    out.local_hit_length  = 0;
    out.remote_slots      = std::move(rep.slots);  // len == hit_length
    out.remote_node_id    = p->peer_rank();
    if (lock && rep.status == RemoteStatus::OK && rep.hit_length > 0) {
        std::vector<uint64_t> hc(hashes, hashes + n);
        int  len  = rep.hit_length;
        auto* self = this;
        out.finalize = FinalizeFn([self, rank, hc = std::move(hc), len]() {
            self->remote_unlock(rank, hc.data(), (int)hc.size(), len);
        });
    }
    return out;
}

RemoteStatus RadixClient::remote_unlock(int rank, const uint64_t* hashes, int n,
                                int hit_length) {
    auto p = peer_for(rank);
    if (!p) return RemoteStatus::BAD_TARGET;
    return p->unlock(hashes, n, 0, hit_length);
}

#endif  // !RADIXSHMEM_NO_RDMA

bool RadixClient::is_distributed() const {
#ifndef RADIXSHMEM_NO_RDMA
    return distributed_;
#else
    return false;
#endif
}

int RadixClient::rank() const {
#ifndef RADIXSHMEM_NO_RDMA
    if (distributed_) return peer_reg_->my_rank();
#endif
    // Without a cluster plane the tree's own stamp is the answer: 0 for a
    // standalone region, the owner's etcd rank for a local_read_only attach.
    if (local_)
        return (int)static_cast<const ShmHeader*>(local_->tree().shm_base())->my_rank;
    return 0;
}

int RadixClient::world_size() const {
#ifndef RADIXSHMEM_NO_RDMA
    if (distributed_) return peer_reg_->world_size();
#endif
    return 1;
}

std::vector<HashType> RadixClient::compute_hashes(const int64_t* token_ids,
                                                    int num_tokens,
                                                    int tokens_per_block) {
    return LocalTreeConnector::compute_hashes(token_ids, num_tokens, tokens_per_block);
}

// Node-local stats: 0 / empty for an observer (no local tree).
uint32_t RadixClient::total_radix_nodes() const  { return local_ ? local_->total_radix_nodes() : 0; }
uint64_t RadixClient::total_blocks() const { return local_ ? local_->total_blocks() : 0; }
int RadixClient::block_size() const {
    if (!local_)
        throw std::runtime_error("RadixClient::block_size: no local tree (observer client)");
    return static_cast<ShmHeader*>(local_->tree().shm_base())->block_size;
}
float    RadixClient::data_pool_utilization() const { return local_ ? local_->data_pool_utilization() : 0.f; }
uint32_t RadixClient::mempool_total() const { return local_ ? local_->mempool_total() : 0; }
uint32_t RadixClient::mempool_free()  const { return local_ ? local_->mempool_free()  : 0; }
uint32_t RadixClient::mempool_used()  const { return local_ ? local_->mempool_used()  : 0; }
uint32_t RadixClient::swa_mempool_total()   const { return local_ ? local_->tree().swa_mempool_total()   : 0; }
uint32_t RadixClient::mamba_mempool_total() const { return local_ ? local_->tree().mamba_mempool_total() : 0; }
ComponentConfig RadixClient::component_config() const {
    return local_ ? local_->tree().component_config() : ComponentConfig{};
}
void     RadixClient::print_tree()    const { if (local_) local_->print_tree(); }

int RadixClient::check_health(HealthReport* report) const {
    HealthReport r;
    if (local_) r = radixshmem::check_health(local_->tree());
    else        r.flags = HEALTH_NOT_ATTACHED;
    if (report) *report = r;
    return (int)r.flags;
}
const std::string& RadixClient::name() const {
    static const std::string kNone;
    return local_ ? local_->name() : kNone;
}

} // namespace radixshmem
