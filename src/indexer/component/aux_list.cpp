// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "component_set.hpp"
#include "shmradix/radix_tree.h"

#include <algorithm>

namespace shmradix {

void ShmRadixTree::aux_touch(const std::vector<MatchedNodeSpanView>& spans,
                             uint32_t mask, uint32_t common_hit,
                             uint32_t swa_start, uint64_t ts) {
    if (common_hit == 0 ||
        (mask & (COMPONENT_MASK_SWA | COMPONENT_MASK_MAMBA)) == 0)
        return;
    lru_lock();
    for (const MatchedNodeSpanView& v : spans) {
        if (v.absolute_start >= common_hit) break;
        const uint32_t end = v.absolute_start + v.node_blocks;
        if ((mask & COMPONENT_MASK_SWA) != 0) {
            const uint32_t lo = std::max(v.absolute_start, swa_start);
            const uint32_t hi = std::min(end, common_hit);
            for (uint32_t b = lo; b < hi; b++)
                swa_lru_.move_tail(v.swa_slots[b - v.absolute_start], ts);
        }
        if ((mask & COMPONENT_MASK_MAMBA) != 0 && end == common_hit &&
            v.mamba_checkpoint_slot != INVALID_COMPONENT_SLOT)
            mamba_lru_.move_tail(v.mamba_checkpoint_slot, ts);
    }
    lru_unlock();
}

} // namespace shmradix
