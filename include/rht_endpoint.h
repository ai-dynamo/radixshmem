// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cluster_manifest.h"   // PeerEntry
#include "rht_types.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct ibv_pd;

namespace radixshmem {

class IRdmaEndpoint;   // src/indexer/transport/rdma/endpoint.hpp
struct PeerFaultCounters;

// ==================== RhtHolder — write side ====================

class RhtHolder {
public:
    // Create the shard shm (name = shm_name_base + "_rht_shard" + shard_id) and
    // register its MR (READ|WRITE). `capacity` is a REQUEST: the module rounds
    // it to its internal bucket-count rule; addr().capacity / addr().size
    // report what was actually allocated.
    RhtHolder(ibv_pd* pd, const std::string& shm_name_base,
              int shard_id, int holder_rank,
              uint32_t capacity, uint32_t slots_per_bucket);
    ~RhtHolder();                     // MR deregistered before shard shm unlink

    RhtHolder(const RhtHolder&) = delete;
    RhtHolder& operator=(const RhtHolder&) = delete;

    // What this rank publishes for its shard; see RhtShardAddr::publish_to.
    const RhtShardAddr& addr() const;

    // Zero every slot in the shard (empty entry == 0); identity preserved.
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ==================== RhtConsumer — read/write side ====================

// Resolved shard map + peer table the cluster read from etcd.
struct RhtTopology {
    int      num_shards       = 0;
    uint32_t capacity         = 0;   // requested per-shard buckets (ShardManifest value)
    uint32_t slots_per_bucket = 0;   // W
    int      my_rank          = -1;
    std::vector<int>              holder_of_shard;   // size M: shard s → holder rank
    const std::vector<PeerEntry>* peers = nullptr;   // rank-indexed, holds rht_shard_*
    PeerFaultCounters*            faults = nullptr;  // node-wide fault counters, may be null
};

// "Connect an RHT-plane endpoint to `holder`", returning a non-owning pointer
// (QP ownership stays with the caller). nullptr on failure.
using RhtEndpointFactory = std::function<IRdmaEndpoint*(int holder)>;
// "Destroy the endpoint to `holder`": it failed; the factory builds a new one.
using RhtEndpointRelease = std::function<void(int holder)>;

class RhtConsumer {
public:
    // pd: protection domain the RHT-plane endpoints live in. The consumer
    // registers its own private scratch MR there for remote bucket reads;
    // its size is an internal detail and lookups of any batch size are
    // split to fit it, so callers share nothing with the data plane.
    // A holder that fails an op is skipped for a while (its writes and
    // removes are dropped, its lookups miss), then reconnected.
    RhtConsumer(const RhtTopology& topo, ibv_pd* pd,
                const RhtEndpointFactory& connect_holder,
                const RhtEndpointRelease& release_holder,
                int bootstrap_timeout_sec);
    ~RhtConsumer();

    RhtConsumer(const RhtConsumer&) = delete;
    RhtConsumer& operator=(const RhtConsumer&) = delete;

    // Publish routing entries. Writes are async; call flush() to drain.
    void insert_batch(const RhtInsertEntry* entries, int count);
    // results[i] receives every candidate for keys[i] (cleared first). With
    // W>1 a bucket may hold several peers' entries; the caller picks.
    void lookup_batch(const uint64_t* keys, int count,
                      std::vector<RHTValue>* results);
    // Retract entries; each cell is cleared only if it still names exactly
    // the (node_id, tree_node_id) given.
    void remove_batch(const RhtRemoveEntry* entries, int count);
    // Raw cells of key's bucket (debug only; no tag/checksum filtering).
    void dump_bucket(uint64_t key, std::vector<RhtRawCell>& out);
    // Drain all pending async writes across every remote endpoint.
    void flush();

    // Zero the shard THIS rank holds (local memset, no RDMA); no-op for
    // non-holder ranks.
    void reset_owned_shard();

    // From the etcd watcher (any thread): a peer published a new entry, or
    // its entry went away. Applied by the next call above.
    void holder_changed(const PeerEntry& e);
    void holder_left(int rank);

private:
    void sync_changes();   // every call above starts here

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace radixshmem
