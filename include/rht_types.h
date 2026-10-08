// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cluster_manifest.h"   // PeerEntry, CLUSTER_RHT_NAME_LEN

#include <cstdint>

namespace radixshmem {

struct RHTValue {
    int32_t node_id = -1;
    uint32_t tree_node_id = UINT32_MAX;
    uint32_t num_blocks = 0;   // node length as published (soft state)

    bool empty() const { return node_id < 0; }
};

struct RhtInsertEntry {
    uint64_t key;
    int32_t  node_id;
    uint32_t tree_node_id;
    uint32_t num_blocks;
};

struct RhtRawCell {
    uint64_t head;
    uint64_t ext;
};

struct RhtRemoveEntry {
    uint64_t key;
    int32_t  node_id;
    uint32_t tree_node_id;
};

struct RhtShardAddr {
    int      shard_id     = -1;
    uint64_t base_addr    = 0;        // slot[0] address; peers index addr + idx*16
    uint32_t rkey         = 0;
    uint64_t size         = 0;        // capacity * W * RHT_SLOT_BYTES
    uint32_t capacity     = 0;
    uint32_t slots_per_bucket = 0;    // W
    char     shm_name[CLUSTER_RHT_NAME_LEN] = {0};

    void publish_to(PeerEntry& e) const;
    static RhtShardAddr from(const PeerEntry& e);
};

} // namespace radixshmem
