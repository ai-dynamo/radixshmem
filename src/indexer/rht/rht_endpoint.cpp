// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "shmradix/rht_endpoint.h"
#include "rht_core.hpp"

#include <algorithm>
#include <mutex>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <infiniband/verbs.h>

namespace shmradix {

// ==================== RhtShardAddr <-> PeerEntry ====================
//
// The only place the rht_shard_* field mapping is spelled out. capacity and
// slots_per_bucket are cluster-global (ShardManifest), not per-peer, so from()
// leaves them 0; RhtConsumer takes them from RhtTopology.

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

// ==================== RhtConsumer — read/write side ====================

// Private scratch for remote bucket reads. 64 KiB holds 512 probes per round
// at W=8 (4096 at W=1); RhtEngine::lookup_batch splits larger batches into
// rounds, so this bounds per-round work, not batch size.
static constexpr size_t RHT_CONSUMER_SCRATCH_BYTES = 64 * 1024;

struct RhtConsumer::Impl {
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

    bool holds_shard(int rank) const {
        return rank != my_rank &&
               std::find(holder_of_shard.begin(), holder_of_shard.end(), rank) !=
                   holder_of_shard.end();
    }
    void apply_changes();

    ~Impl() {
        rht.reset();                          // stop using scratch
        scratch_mr = RdmaMemoryRegion();      // deregister
        std::free(scratch);
    }
};

RhtConsumer::RhtConsumer(const RhtTopology& topo, ibv_pd* pd,
                         const RhtEndpointFactory& connect_holder,
                         const RhtEndpointRelease& release_holder,
                         int bootstrap_timeout_sec)
    : impl_(std::make_unique<Impl>()) {
    impl_->my_rank = topo.my_rank;
    if (topo.num_shards <= 0)
        throw std::runtime_error("RhtConsumer: num_shards == 0");
    if (!pd)
        throw std::runtime_error("RhtConsumer: null protection domain");
    const std::vector<PeerEntry>& peers = *topo.peers;

    impl_->scratch = std::aligned_alloc(4096, RHT_CONSUMER_SCRATCH_BYTES);
    if (!impl_->scratch)
        throw std::runtime_error("RhtConsumer: scratch allocation failed");
    std::memset(impl_->scratch, 0, RHT_CONSUMER_SCRATCH_BYTES);
    impl_->scratch_mr = RdmaMemoryRegion(pd, impl_->scratch,
                                         RHT_CONSUMER_SCRATCH_BYTES,
                                         IBV_ACCESS_LOCAL_WRITE);
    if (!impl_->scratch_mr.valid())
        throw std::runtime_error("RhtConsumer: scratch MR registration failed");

    // The bucket count the engine will route over. Every holder built its
    // shard from the same manifest value through the same normalization, so
    // its published extent must equal this; a mismatch means the engine's
    // mask would index past the holder's MR.
    const uint32_t capacity = rht_normalize_capacity(topo.capacity);
    const uint64_t shard_bytes =
        (uint64_t)capacity * topo.slots_per_bucket * RHT_SLOT_BYTES;

    std::vector<void*>          shard_ptrs(topo.num_shards, nullptr);
    std::vector<int>            unreachable, missing;
    std::vector<IRdmaEndpoint*> eps(topo.num_shards, nullptr);
    std::vector<RdmaMrInfo>     mrs(topo.num_shards, RdmaMrInfo{});

    for (int s = 0; s < topo.num_shards; s++) {
        int holder = topo.holder_of_shard[s];
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
                shard.slots_per_bucket() == topo.slots_per_bucket) {
                shard_ptrs[s] = shard.buckets();
                impl_->local_shards.emplace(s, std::move(shard));
                continue;
            }
            throw std::runtime_error(
                "RhtConsumer: failed to attach own RHT shard");
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
                "RhtConsumer: holder rank " + std::to_string(holder) +
                " published RHT shard of " + std::to_string(h.size) +
                " bytes, expected " + std::to_string(shard_bytes));
        if (std::find(unreachable.begin(), unreachable.end(), holder) == unreachable.end()) {
            eps[s] = connect_holder(holder);
            if (!eps[s]) unreachable.push_back(holder);   // marked down below
        }
        mrs[s].rkey   = h.rkey;
        mrs[s].addr   = h.base_addr;
        mrs[s].length = h.size;
    }

    impl_->rht = std::make_unique<RhtEngine>(
        topo.num_shards, topo.capacity, topo.slots_per_bucket,
        std::move(shard_ptrs));
    impl_->rht->enable_rdma(std::move(eps), std::move(mrs),
                            impl_->scratch, impl_->scratch_mr.lkey(),
                            RHT_CONSUMER_SCRATCH_BYTES);

    impl_->breaker = std::make_unique<PeerBreaker>(
        "[RadixClient rank " + std::to_string(topo.my_rank) + "] RHT", topo.faults);
    for (int s = 0; s < topo.num_shards; s++)
        impl_->breaker->name(topo.holder_of_shard[s], peers[topo.holder_of_shard[s]].host);
    impl_->rht->set_fault_handling(topo.holder_of_shard, impl_->breaker.get(),
                                   connect_holder, release_holder);
    for (int holder : unreachable)
        if (!impl_->breaker->is_down(holder))
            impl_->breaker->failed(holder, "connect", "no RHT-plane endpoint", 0.0);
    for (int holder : missing)
        impl_->breaker->left(holder, "is not registered");
    impl_->holder_of_shard = topo.holder_of_shard;
    impl_->shard_bytes = shard_bytes;
}

void RhtConsumer::Impl::apply_changes() {
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

void RhtConsumer::holder_changed(const PeerEntry& e) {
    if (!impl_->holds_shard(e.rank)) return;
    std::lock_guard<std::mutex> g(impl_->changes_mu);
    impl_->changes.push_back({e.rank, false, e});
    impl_->has_changes.store(true, std::memory_order_release);
}

void RhtConsumer::holder_left(int rank) {
    if (!impl_->holds_shard(rank)) return;
    std::lock_guard<std::mutex> g(impl_->changes_mu);
    impl_->changes.push_back({rank, true, PeerEntry{}});
    impl_->has_changes.store(true, std::memory_order_release);
}

void RhtConsumer::sync_changes() {
    if (impl_->has_changes.load(std::memory_order_acquire)) impl_->apply_changes();
}

RhtConsumer::~RhtConsumer() = default;

void RhtConsumer::insert_batch(const RhtInsertEntry* entries, int count) {
    sync_changes();
    impl_->rht->insert_batch(entries, count);
}

void RhtConsumer::lookup_batch(const uint64_t* keys, int count,
                               std::vector<RHTValue>* results) {
    sync_changes();
    impl_->rht->lookup_batch(keys, count, results);
}

void RhtConsumer::remove_batch(const RhtRemoveEntry* entries, int count) {
    sync_changes();
    impl_->rht->remove_batch(entries, count);
}

void RhtConsumer::dump_bucket(uint64_t key, std::vector<RhtRawCell>& out) {
    sync_changes();
    std::vector<RhtSlot> raw;
    impl_->rht->dump_bucket(key, raw);
    out.clear();
    for (const RhtSlot& c : raw) out.push_back({c.head, c.ext});
}

void RhtConsumer::flush() {
    sync_changes();
    impl_->rht->flush();
}

void RhtConsumer::reset_owned_shard() {
    // local_shards only ever holds this rank's own shard (see ctor).
    for (auto& kv : impl_->local_shards)
        if (kv.second.valid()) kv.second.reset();
}

} // namespace shmradix
