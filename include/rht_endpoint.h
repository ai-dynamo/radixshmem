// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cluster_manifest.h"   // PeerEntry
#include "rht_backend.h"
#include "rht_types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ibv_pd;

namespace radixshmem {

class PeerRegistry;
class RdmaTransportClient;
struct PeerFaultCounters;
struct ShardMapMsg;

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

// ==================== RdmaRht — read/write side ====================

// Shards on the holder ranks, read and written with one-sided RDMA (the own
// shard through shm). A holder that fails an op is skipped for a while.
class RdmaRht final : public RhtBackend {
public:
    // Opens `dev` (empty = first device) and connects to the holders `sm` names.
    RdmaRht(const ShardMapMsg& sm, const std::string& dev, int gid_idx,
            PeerRegistry& peers, PeerFaultCounters* faults, int bootstrap_timeout_sec);
    ~RdmaRht() override;

    RdmaRht(const RdmaRht&) = delete;
    RdmaRht& operator=(const RdmaRht&) = delete;

    void insert_batch(const RhtInsertEntry* entries, int count) override;
    void lookup_batch(const uint64_t* keys, int count,
                      std::vector<RHTValue>* results) override;
    void remove_batch(const RhtRemoveEntry* entries, int count) override;
    void flush() override;
    // Zeroes the shard this rank holds; `own` is not needed.
    void reset(const RhtRemoveEntry* own, int count) override;
    void peer_changed(const PeerEntry& e) override;
    void peer_left(int rank) override;
    void dump(uint64_t key, std::vector<std::string>& lines) override;

private:
    void sync_changes();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace radixshmem
