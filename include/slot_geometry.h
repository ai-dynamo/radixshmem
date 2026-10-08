// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "component_kind.h"

#include <cstdint>

namespace radixshmem {

struct PoolGeometry {
    uint64_t base       = 0;
    uint64_t num_slots  = 0;   // 0 = absent
    uint64_t slot_bytes = 0;   // stride
};

struct StoreGeometry {
    uint32_t     pool_mask = 0;
    PoolGeometry pools[NUM_COMPONENT_TYPES];   // indexed by ComponentType

    bool has(ComponentType t) const noexcept {
        return component_type_valid(t) && (pool_mask & component_mask_of(t)) != 0;
    }
    const PoolGeometry& operator[](ComponentType t) const noexcept {
        return pools[static_cast<uint32_t>(t)];
    }
    PoolGeometry& operator[](ComponentType t) noexcept {
        return pools[static_cast<uint32_t>(t)];
    }

    StoreGeometry rebased(uint64_t segment_base) const noexcept {
        StoreGeometry g = *this;
        for (int i = 0; i < NUM_COMPONENT_TYPES; ++i)
            if (pool_mask & (1u << i)) g.pools[i].base += segment_base;
        return g;
    }

    bool same_strides(const StoreGeometry& o) const noexcept {
        if (pool_mask != o.pool_mask) return false;
        for (int i = 0; i < NUM_COMPONENT_TYPES; ++i)
            if ((pool_mask & (1u << i)) && pools[i].slot_bytes != o.pools[i].slot_bytes)
                return false;
        return true;
    }
};

using SlotGeometry = PoolGeometry;

} // namespace radixshmem
