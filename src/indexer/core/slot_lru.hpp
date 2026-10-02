// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "shmradix/types.h"

namespace shmradix {

// Per-component LRU over slot ids; entries live in shm, indexed by slot.
class SlotLru {
public:
    SlotLru() = default;
    SlotLru(SlotLruEntry* entries, uint32_t capacity, AuxListAnchors* anchors);

    static uint64_t region_bytes(uint32_t capacity) {
        return (uint64_t)capacity * sizeof(SlotLruEntry);
    }
    static void init_region(void* base, uint32_t capacity);

    bool enabled() const { return entries_ != nullptr; }
    void reset();

    bool linked(int32_t slot) const {
        return entries_[slot].node_id != INVALID_NODE_ID;
    }
    SlotLruEntry& entry(int32_t slot) { return entries_[slot]; }
    const SlotLruEntry& entry(int32_t slot) const { return entries_[slot]; }
    uint32_t head() const { return anchors_->head; }

    void push_tail(int32_t slot, uint32_t node_id, uint32_t idx, uint64_t ts);
    void move_tail(int32_t slot, uint64_t ts);
    void unlink(int32_t slot);

private:
    void link_tail(int32_t slot);

    SlotLruEntry* entries_ = nullptr;
    uint32_t capacity_ = 0;
    AuxListAnchors* anchors_ = nullptr;
};

} // namespace shmradix
