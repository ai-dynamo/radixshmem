// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Live, rank-indexed peer table shared by the data plane, RHT plane, and the
// remote-op factory. Owns the dense PeerEntry[] and its guard; the /peers
// watcher updates it (admit / remove) under the write lock, dependents read
// under it.

#pragma once

#include "cluster_manifest.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <shared_mutex>
#include <vector>

namespace radixshmem {

class PeerRegistry {
  public:
    // Invoked under the write lock when the table grows to `n`, so a dependent's
    // lockstep rank-indexed structure (e.g. per-peer endpoint slots) resizes
    // atomically with the peer table.
    using GrowFn = std::function<void(int n)>;

    void set_grow_hook(GrowFn fn) { on_grow_ = std::move(fn); }

    std::shared_mutex& mutex() const { return mu_; }

    int world_size() const { return world_size_.load(std::memory_order_acquire); }
    int my_rank()    const { return my_rank_; }
    void set_my_rank(int r) { my_rank_ = r; }

    // Seed the dense table from a bootstrap snapshot.
    void seed(std::vector<PeerEntry> peers, int world_size, int my_rank) {
        std::unique_lock<std::shared_mutex> lk(mu_);
        peers_ = std::move(peers);
        if ((int)peers_.size() < world_size) peers_.resize(world_size);
        world_size_.store(world_size, std::memory_order_release);
        my_rank_ = my_rank;
        if (on_grow_) on_grow_((int)peers_.size());
    }

    // Take a published entry: grows the table (and lockstep structures via
    // the hook) for a new rank, else fills or replaces the rank's entry.
    // false when the entry is unchanged.
    bool admit(const PeerEntry& e) {
        std::unique_lock<std::shared_mutex> lk(mu_);
        if (e.rank < (int)peers_.size() &&
            std::memcmp(&peers_[e.rank], &e, sizeof(PeerEntry)) == 0)
            return false;
        if (e.rank >= (int)peers_.size()) {
            peers_.resize(e.rank + 1);
            if (on_grow_) on_grow_(e.rank + 1);
        }
        peers_[e.rank] = e;
        int ws = world_size_.load(std::memory_order_acquire);
        world_size_.store(std::max(ws, e.rank + 1), std::memory_order_release);
        return true;
    }

    // The rank's entry went away: it reads as down (rank -1).
    void remove(int rank) {
        std::unique_lock<std::shared_mutex> lk(mu_);
        if (rank >= 0 && rank < (int)peers_.size()) peers_[rank].rank = -1;
    }

    // Caller must hold at least a shared lock on mutex().
    const std::vector<PeerEntry>& peers_locked() const { return peers_; }
    int size_locked() const { return (int)peers_.size(); }

  private:
    mutable std::shared_mutex mu_;
    std::vector<PeerEntry>    peers_;
    std::atomic<int>          world_size_{1};
    int                       my_rank_ = 0;
    GrowFn                    on_grow_;
};

} // namespace radixshmem
