// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.h"
#include <cstdint>

namespace radixshmem {

class DataPool {
public:
    DataPool() = default;
    // swa_lane: this shm carries the SWA slot lane (component_has_swa).
    DataPool(void* shm_base, BuddyHeader* header, bool swa_lane);

    uint64_t alloc(int num_blocks);
    void free(uint64_t offset, int num_blocks);
    float utilization() const;

    // Payload lanes (see data_block_bytes): hashes @0, full slots @8n, swa @12n.
    uint64_t* get_hashes(uint64_t offset) const;
    int32_t* get_slots(uint64_t offset, int num_blocks) const;      // FULL lane
    int32_t* get_swa_slots(uint64_t offset, int num_blocks) const;  // SWA lane; nullptr without it
    bool has_swa_lane() const { return swa_lane_; }

    int blocks_to_order(int num_blocks) const;

    static void init(void* shm_base, BuddyHeader* header,
                     uint64_t data_pool_offset,
                     uint64_t buddy_hdr_size, uint64_t bitmap_bytes,
                     uint64_t omap_bytes, uint64_t blocks_size);

    // Clear all allocations in place (dirty-memory safe). Layout fields in the
    // BuddyHeader are untouched.
    void reset();

private:
    // Reset bitmap / order map / free lists to the all-free state (init/reset).
    void rebuild_free_lists();

    void free_list_push(int order, uint64_t offset);
    void free_list_remove(int order, uint64_t offset);
    uint64_t free_list_pop(int order);

    void bitmap_set(uint32_t start_idx, int order);
    void bitmap_clear(uint32_t start_idx, int order);
    bool bitmap_all_free(uint32_t start_idx, int order) const;

    uint32_t offset_to_index(uint64_t offset) const;
    uint64_t index_to_offset(uint32_t index) const;

    uint8_t* bitmap() const;
    uint8_t* order_map() const;

    void* base_ = nullptr;
    BuddyHeader* header_ = nullptr;
    bool swa_lane_ = false;
};

} // namespace radixshmem
