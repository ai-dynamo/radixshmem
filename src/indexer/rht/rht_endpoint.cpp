// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "rht_endpoint.h"
#include "rht_core.hpp"
#include "cluster_protocol.h"
#include "peer_registry.hpp"
#include "rdma_transport_client.hpp"
#include "xrc_connect.hpp"

#include <algorithm>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <infiniband/verbs.h>

namespace radixshmem {

// ==================== RhtShardAddr <-> PeerEntry ====================
//
// The only place the rht_shard_* field mapping is spelled out. capacity and
// slots_per_bucket are cluster-global (ShardManifest), not per-peer, so from()
// leaves them 0; RdmaRht takes them from the shard map.

void RhtShardAddr::publish_to(PeerEntry& e) const {
    e.rht_shard_base_addr = base_addr;
    e.rht_shard_size      = size;
    e.rht_shard_rkey      = rkey;
    e.rht_shard_id        = shard_id;
    std::strncpy(e.rht_shard_name, shm_name, sizeof(e.rht_shard_name) - 1);
    e.rht_shard_name[sizeof(e.rht_shard_name) - 1] = '\0';
}

RhtShardAddr RhtShardAddr::from(const PeerEntry& e) {
    RhtShardAddr a;
    a.shard_id  = e.rht_shard_id;
    a.base_addr = e.rht_shard_base_addr;
    a.rkey      = e.rht_shard_rkey;
    a.size      = e.rht_shard_size;
    std::strncpy(a.shm_name, e.rht_shard_name, sizeof(a.shm_name) - 1);
    a.shm_name[sizeof(a.shm_name) - 1] = '\0';
    return a;
}

// ==================== RhtHolder — write side ====================

struct RhtHolder::Impl {
    RhtShard                          shard;
    std::unique_ptr<RdmaMemoryRegion> mr;      // declared after shard: torn down first
    RhtShardAddr                      addr;
};

RhtHolder::RhtHolder(ibv_pd* pd, const std::string& shm_name_base,
                     int shard_id, int holder_rank,
                     uint32_t capacity, uint32_t slots_per_bucket)
    : impl_(std::make_unique<Impl>()) {
    std::string shard_name = shm_name_base + "_rht_shard"
                           + std::to_string(shard_id);
    impl_->shard = RhtShard::create(shard_name, capacity, slots_per_bucket,
                                    shard_id, holder_rank);
    impl_->mr = std::make_unique<RdmaMemoryRegion>(
        pd, impl_->shard.base(), impl_->shard.size(),
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
        IBV_ACCESS_REMOTE_WRITE);

    // Publish what the shard actually allocated (normalized capacity), not
    // the request: peers size their reads/MR bounds from these fields.
    const RhtShard& sh = impl_->shard;
    RhtShardAddr& a = impl_->addr;
    a.shard_id         = shard_id;
    a.base_addr        = (uint64_t)sh.base() + sh.data_offset();
    a.rkey             = impl_->mr->rkey();
    a.size             = (uint64_t)sh.capacity() * sh.slots_per_bucket() * sizeof(RhtSlot);
    a.capacity         = sh.capacity();
    a.slots_per_bucket = sh.slots_per_bucket();
    std::strncpy(a.shm_name, shard_name.c_str(), sizeof(a.shm_name) - 1);
}

RhtHolder::~RhtHolder() {
    if (impl_) impl_->mr.reset();   // deregister before the shard shm unmaps/unlinks
}

const RhtShardAddr& RhtHolder::addr() const { return impl_->addr; }

void RhtHolder::reset() { impl_->shard.reset(); }

// ==================== RdmaRht — read/write side ====================

// Private scratch for remote bucket reads. 64 KiB holds 512 probes per round
// at W=8 (4096 at W=1); RhtEngine::lookup_batch splits larger batches into
// rounds, so this bounds per-round work, not batch size.
static constexpr size_t RDMA_RHT_SCRATCH_BYTES = 64 * 1024;

struct RdmaRht::Impl {
    std::unique_ptr<RdmaTransportClient> tx;   // first: outlives the endpoints below
    PeerRegistry&        peers;
    uint32_t             transport = CLUSTER_TRANSPORT_XRC;
    uint64_t             dc_access_key = 0;
    bool                 attached = false;
    std::unordered_map<int, std::unique_ptr<IRdmaEndpoint>> holder_eps;

    int my_rank = -1;
    // This rank's own shard attached over shm, keyed by shard id (empty on
    // non-holder ranks).
    std::unordered_map<int, RhtShard> local_shards;
    // Page-aligned scratch + its MR; the MR is declared after the buffer so it
    // is deregistered first, then ~Impl frees the memory.
    void*                             scratch = nullptr;
    RdmaMemoryRegion                  scratch_mr;
    std::unique_ptr<PeerBreaker>      breaker;
    std::unique_ptr<RhtEngine>        rht;
    std::vector<int>                  holder_of_shard;
    uint64_t                          shard_bytes = 0;

    // Watcher updates, applied on the caller's thread (the engine has no lock).
    struct Change { int rank; bool left; PeerEntry e; };
    std::mutex                        changes_mu;
    std::vector<Change>               changes;
    std::atomic<bool>                 has_changes{false};
    std::mutex                        op_mu;

    explicit Impl(PeerRegistry& p) : peers(p) {}

    bool holds_shard(int rank) const {
        return rank != my_rank &&
               std::find(holder_of_shard.begin(), holder_of_shard.end(), rank) !=
                   holder_of_shard.end();
    }
    void apply_changes();
    IRdmaEndpoint* connect(int holder);
    void release(int holder);

    ~Impl() {
        rht.reset();                          // stop using scratch
        scratch_mr = RdmaMemoryRegion();      // deregister
        std::free(scratch);
    }
};

RdmaRht::RdmaRht(const ShardMapMsg& sm, const std::string& dev, int gid_idx,
                 PeerRegistry& peers_reg, PeerFaultCounters* faults, int bootstrap_timeout_sec)
    : impl_(std::make_unique<Impl>(peers_reg)) {
    const int num_shards = (int)sm.num_shards;
    if (num_shards <= 0 || num_shards > (int)CLUSTER_MAX_SHARDS)
        throw std::runtime_error("RdmaRht: num_shards == " + std::to_string(num_shards));
    // QoS (rank 0's SL/TC) is adopted inside the transport before any QP connect.
    impl_->tx = std::make_unique<RdmaTransportClient>(dev.empty() ? nullptr : dev.c_str(),
                                                      gid_idx, sm.rdma_sl, sm.rdma_tc,
                                                      peers_reg.world_size(), 0);
    RdmaTransportClient& tx = *impl_->tx;
    const uint32_t slots_per_bucket = sm.rht_slots ? sm.rht_slots : 4;
    const std::vector<int> holder_of_shard(sm.holder_rank, sm.holder_rank + num_shards);
    impl_->my_rank       = peers_reg.my_rank();
    impl_->transport     = sm.rht_transport;
    impl_->dc_access_key = sm.dc_access_key;
    std::vector<PeerEntry> peers;
    {
        std::shared_lock<std::shared_mutex> lk(peers_reg.mutex());
        peers = peers_reg.peers_locked();
    }

    impl_->scratch = std::aligned_alloc(4096, RDMA_RHT_SCRATCH_BYTES);
    if (!impl_->scratch)
        throw std::runtime_error("RdmaRht: scratch allocation failed");
    std::memset(impl_->scratch, 0, RDMA_RHT_SCRATCH_BYTES);
    impl_->scratch_mr = RdmaMemoryRegion(tx.ctx().pd(), impl_->scratch,
                                         RDMA_RHT_SCRATCH_BYTES,
                                         IBV_ACCESS_LOCAL_WRITE);
    if (!impl_->scratch_mr.valid())
        throw std::runtime_error("RdmaRht: scratch MR registration failed");

    // The bucket count the engine will route over. Every holder built its
    // shard from the same manifest value through the same normalization, so
    // its published extent must equal this; a mismatch means the engine's
    // mask would index past the holder's MR.
    const uint32_t capacity = rht_normalize_capacity(sm.rht_capacity);
    const uint64_t shard_bytes =
        (uint64_t)capacity * slots_per_bucket * RHT_SLOT_BYTES;

    std::vector<void*>          shard_ptrs(num_shards, nullptr);
    std::vector<int>            unreachable, missing;
    std::vector<IRdmaEndpoint*> eps(num_shards, nullptr);
    std::vector<RdmaMrInfo>     mrs(num_shards, RdmaMrInfo{});

    for (int s = 0; s < num_shards; s++) {
        int holder = holder_of_shard[s];
        RhtShardAddr h = RhtShardAddr::from(peers[holder]);

        // One rank == one physical node here: a rank attaches only the shard it
        // holds; every peer shard goes over the NIC. This also makes the
        // one-box-many-ranks test a faithful multi-node simulation (each rank
        // RDMA-reads the others' shards) without an env override.
        if (holder == impl_->my_rank) {
            RhtShard shard = RhtShard::attach(h.shm_name, bootstrap_timeout_sec);
            if (shard.valid() && shard.shard_id() == s &&
                shard.holder_rank() == holder &&
                shard.capacity() == capacity &&
                shard.slots_per_bucket() == slots_per_bucket) {
                shard_ptrs[s] = shard.buckets();
                impl_->local_shards.emplace(s, std::move(shard));
                continue;
            }
            throw std::runtime_error(
                "RdmaRht: failed to attach own RHT shard");
        }

        // Peer shard: RHT-plane QP + RDMA. A holder that is not registered
        // is skipped until the watcher sees its entry.
        if (peers[holder].rank != holder) {
            if (std::find(missing.begin(), missing.end(), holder) == missing.end())
                missing.push_back(holder);
            continue;
        }
        if (h.size != shard_bytes)
            throw std::runtime_error(
                "RdmaRht: holder rank " + std::to_string(holder) +
                " published RHT shard of " + std::to_string(h.size) +
                " bytes, expected " + std::to_string(shard_bytes));
        if (std::find(unreachable.begin(), unreachable.end(), holder) == unreachable.end()) {
            eps[s] = impl_->connect(holder);
            if (!eps[s]) unreachable.push_back(holder);   // marked down below
        }
        mrs[s].rkey   = h.rkey;
        mrs[s].addr   = h.base_addr;
        mrs[s].length = h.size;
    }

    impl_->rht = std::make_unique<RhtEngine>(
        num_shards, sm.rht_capacity, slots_per_bucket,
        std::move(shard_ptrs));
    impl_->rht->enable_rdma(std::move(eps), std::move(mrs),
                            impl_->scratch, impl_->scratch_mr.lkey(),
                            RDMA_RHT_SCRATCH_BYTES);

    impl_->breaker = std::make_unique<PeerBreaker>(
        "[RadixClient rank " + std::to_string(impl_->my_rank) + "] RHT", faults);
    for (int s = 0; s < num_shards; s++)
        impl_->breaker->name(holder_of_shard[s], peers[holder_of_shard[s]].host);
    Impl* im = impl_.get();
    impl_->rht->set_fault_handling(holder_of_shard, impl_->breaker.get(),
                                   [im](int h) { return im->connect(h); },
                                   [im](int h) { im->release(h); });
    for (int holder : unreachable)
        if (!impl_->breaker->is_down(holder))
            impl_->breaker->failed(holder, "connect", "no RHT-plane endpoint", 0.0);
    for (int holder : missing)
        impl_->breaker->left(holder, "is not registered");
    impl_->holder_of_shard = holder_of_shard;
    impl_->shard_bytes = shard_bytes;
    impl_->attached = true;
}

// peers.mutex() guards the peer table and tx's DCI and AH cache.
IRdmaEndpoint* RdmaRht::Impl::connect(int holder) {
    auto it = holder_eps.find(holder);
    if (it != holder_eps.end()) return it->second.get();

    std::unique_lock<std::shared_mutex> lk(peers.mutex());
    const PeerEntry& h = peers.peers_locked()[holder];
    std::unique_ptr<IRdmaEndpoint> ep;
    if (transport == CLUSTER_TRANSPORT_DC) {
        DcInitiator* dci = tx->shared_dci();
        ibv_gid dgid;
        memcpy(&dgid, h.gid_raw, 16);
        ibv_ah* ah = tx->ah_for(holder, dgid, h.lid, h.link_layer, peers.world_size());
        if (!ah) {
            fprintf(stderr, "[RadixClient rank %d] RHT-plane AH to holder rank %d failed\n",
                    my_rank, holder);
            return nullptr;
        }
        ep = std::make_unique<DcReader>(dci, DcTargetInfo{ah, h.dct_num, dc_access_key});
    } else {
        // One try; a holder that does not answer is probed again by the breaker.
        ep = xrc_connect(tx->ctx(), tx->cq(), h.host, h.xrc_listen_port, my_rank,
                         attached ? 2 : 10);
        if (!ep) return nullptr;
    }
    IRdmaEndpoint* raw = ep.get();
    holder_eps.emplace(holder, std::move(ep));
    return raw;
}

void RdmaRht::Impl::release(int holder) {
    std::unique_lock<std::shared_mutex> lk(peers.mutex());
    holder_eps.erase(holder);
}

void RdmaRht::Impl::apply_changes() {
    std::vector<Change> todo;
    {
        std::lock_guard<std::mutex> g(changes_mu);
        todo.swap(changes);
        has_changes.store(false, std::memory_order_release);
    }
    for (const Change& c : todo) {
        RhtShardAddr h = RhtShardAddr::from(c.e);
        if (c.left || h.size != shard_bytes) {
            rht->holder_changed(c.rank, nullptr);
            breaker->left(c.rank, c.left ? "left the cluster (etcd entry removed)"
                                         : "published an RHT shard of another size");
            continue;
        }
        RdmaMrInfo mr{h.rkey, h.base_addr, h.size};
        rht->holder_changed(c.rank, &mr);
        breaker->name(c.rank, c.e.host);
        breaker->retry_now(c.rank);
    }
}

void RdmaRht::peer_changed(const PeerEntry& e) {
    if (!impl_->holds_shard(e.rank)) return;
    std::lock_guard<std::mutex> g(impl_->changes_mu);
    impl_->changes.push_back({e.rank, false, e});
    impl_->has_changes.store(true, std::memory_order_release);
}

void RdmaRht::peer_left(int rank) {
    if (!impl_->holds_shard(rank)) return;
    std::lock_guard<std::mutex> g(impl_->changes_mu);
    impl_->changes.push_back({rank, true, PeerEntry{}});
    impl_->has_changes.store(true, std::memory_order_release);
}

void RdmaRht::sync_changes() {
    if (impl_->has_changes.load(std::memory_order_acquire)) impl_->apply_changes();
}

RdmaRht::~RdmaRht() = default;

void RdmaRht::insert_batch(const RhtInsertEntry* entries, int count) {
    std::lock_guard<std::mutex> g(impl_->op_mu);
    sync_changes();
    impl_->rht->insert_batch(entries, count);
}

void RdmaRht::lookup_batch(const uint64_t* keys, int count,
                           std::vector<RHTValue>* results) {
    std::lock_guard<std::mutex> g(impl_->op_mu);
    sync_changes();
    impl_->rht->lookup_batch(keys, count, results);
}

void RdmaRht::remove_batch(const RhtRemoveEntry* entries, int count) {
    std::lock_guard<std::mutex> g(impl_->op_mu);
    sync_changes();
    impl_->rht->remove_batch(entries, count);
    impl_->rht->flush();
}

void RdmaRht::dump(uint64_t key, std::vector<std::string>& lines) {
    std::lock_guard<std::mutex> g(impl_->op_mu);
    sync_changes();
    std::vector<RhtSlot> raw;
    impl_->rht->dump_bucket(key, raw);
    lines.clear();
    char buf[40];
    for (const RhtSlot& c : raw) {
        snprintf(buf, sizeof(buf), "%016llx/%016llx", (unsigned long long)c.head,
                 (unsigned long long)c.ext);
        lines.emplace_back(buf);
    }
}

void RdmaRht::flush() {
    std::lock_guard<std::mutex> g(impl_->op_mu);
    sync_changes();
    impl_->rht->flush();
}

void RdmaRht::reset(const RhtRemoveEntry*, int) {
    // local_shards only ever holds this rank's own shard (see ctor).
    for (auto& kv : impl_->local_shards)
        if (kv.second.valid()) kv.second.reset();
}

} // namespace radixshmem
