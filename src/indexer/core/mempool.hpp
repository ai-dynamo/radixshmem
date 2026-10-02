// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "shmradix/types.h"
#include <cstdint>

namespace shmradix {

class Mempool {
public:
    Mempool() = default;
    Mempool(MempoolHeader* header, uint32_t* free_stack, uint64_t* used_bitmap);

    int allocate(int num, int32_t* out_ids);
    // Idempotent: skips out-of-range ids and any slot already free (bit==0),
    // so a double-recycle is a no-op. Returns the count actually freed.
    int recycle(const int32_t* ids, int num);

    // Return every slot to the free stack (dirty-memory safe).
    void reset();

    uint32_t num_free() const;
    uint32_t num_used() const;
    uint32_t total() const;
    bool enabled() const;
    // Read-only bitmap probe (health audit); false for out-of-range ids.
    bool is_used(uint32_t id) const {
        return header_ && id < header_->total_blocks &&
               ((used_bitmap_[id >> 6] >> (id & 63)) & 1ull);
    }

    // shm-resident used-bitmap: 0 = free, 1 = used. words = ceil(total/64).
    static uint64_t bitmap_words(uint32_t total_blocks) {
        return ((uint64_t)total_blocks + 63) / 64;
    }

    static void init(MempoolHeader* header, uint32_t* free_stack,
                     uint64_t* used_bitmap, uint32_t total_blocks);

    // Region: header | free_stack[u32×N] | padding | used_bitmap.
    // Shared by compute_layout/create/attach for all component pools.
    static uint64_t region_bytes(uint32_t total_blocks) {
        if (total_blocks == 0) return 0;
        return align_up(align_up(sizeof(MempoolHeader), 32)
                            + (uint64_t)total_blocks * sizeof(uint32_t), 8)
             + bitmap_words(total_blocks) * sizeof(uint64_t);
    }
    // Wire a Mempool over an existing region (attach path).
    static Mempool attach_region(void* region_base, uint32_t total_blocks);
    // Initialize a fresh region (create path).
    static void init_region(void* region_base, uint32_t total_blocks);

private:
    void mark_used(uint32_t id) {
        used_bitmap_[id >> 6] |= (1ull << (id & 63));
    }
    void mark_free(uint32_t id) {
        used_bitmap_[id >> 6] &= ~(1ull << (id & 63));
    }

    MempoolHeader* header_ = nullptr;
    uint32_t* free_stack_ = nullptr;
    uint64_t* used_bitmap_ = nullptr;
};

} // namespace shmradix
