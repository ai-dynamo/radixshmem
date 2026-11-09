// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cluster_manifest.h"   // PeerEntry
#include "rht_types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace radixshmem {

// Soft state: ops may be dropped without throwing, writes land by flush(),
// lookups may miss or return stale routes (callers validate them), and remove
// only clears a route that still names the given tree_node_id.
class RhtBackend {
public:
    virtual ~RhtBackend() = default;

    virtual void insert_batch(const RhtInsertEntry* entries, int count) = 0;
    virtual void remove_batch(const RhtRemoveEntry* entries, int count) = 0;
    virtual void lookup_batch(const uint64_t* keys, int count,
                              std::vector<RHTValue>* results) = 0;
    virtual void flush() = 0;

    // Drops this rank's routes. `own` lists them when the caller still has its
    // tree; count < 0 leaves finding them to the backend.
    virtual void reset(const RhtRemoveEntry* own, int count) = 0;
    virtual void peer_changed(const PeerEntry&) {}
    virtual void peer_left(int) {}

    virtual void dump(uint64_t key, std::vector<std::string>& lines) = 0;
};

} // namespace radixshmem
