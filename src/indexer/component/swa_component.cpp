// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "component_set.hpp"
#include "radix_tree.h"

#include <cassert>
#include <cstdio>

namespace radixshmem {

void RadixShmemTree::relink_swa(uint32_t id) {
    TreeNode* node = nodes_.get(id);
    const int nb = (int)node->num_blocks;
    const int32_t* lane = data_.get_swa_slots(node->data_offset, nb);
    uint32_t live = 0;
    for (int i = 0; i < nb; i++) {
        if (lane[i] == INVALID_COMPONENT_SLOT) continue;
        SlotLruEntry& e = swa_lru_.entry(lane[i]);
        e.node_id = id;
        e.idx = (uint32_t)i;
        live++;
    }
    nodes_.component_state(id)->swa_live_blocks = live;
}

// Positional publish of the trailing lane cells [n - num_slots, n); live cells
// keep their slot and only move to the LRU tail. W is not involved: the window
// is a query-side acceptance rule, so a publisher may cover one position (a
// decode step) or a whole node. A whole window is the special case
// num_slots == min(W, n).
InsertResult RadixShmemTree::insert_swa(const uint64_t* hashes, int n,
                                      const int32_t* slots, int num_slots,
                                      bool auto_recycle) {
    set_write_op("insert_swa");
    InsertResult result;

    auto finish = [&]() -> InsertResult {
        if (auto_recycle && !result.unused_slots.empty())
            recycle_slots(result.unused_slots.data(),
                          (int)result.unused_slots.size(), ComponentType::SWA);
        return result;
    };
    auto refuse = [&](InsertError e) -> InsertResult {
        result.error = e;
        if (slots != nullptr && num_slots > 0)
            result.unused_slots.assign(slots, slots + num_slots);
        return finish();
    };

    const ComponentConfig& config = header()->component_config;
    const int start = n - num_slots;
    if (n <= 0 || hashes == nullptr || slots == nullptr ||
        num_slots <= 0 || num_slots > n ||
        !is_component_config_valid(config.component_mask,
                                   config.swa_window_blocks) ||
        !(config.component_mask & COMPONENT_MASK_SWA))
        return refuse(InsertError::BAD_REQUEST);

    result.unused_slots.reserve((size_t)num_slots);

    write_lock();
    QueryInternalDetail m = query_internal(hashes, n, 0, false);
    if (m.matched_blocks < n) {
        write_unlock();
        return refuse(InsertError::FULL_PATH_MISSING);
    }

    const uint64_t now = now_ts();
    for (const MatchedNodeSpanView& v : path_spans(m)) {
        const int before = (int)v.absolute_start;
        const int nb = (int)v.node_blocks;
        const int lo = before > start ? before : start;
        const int hi = before + nb < n ? before + nb : n;
        if (lo >= hi) continue;
        int32_t* lane = data_.get_swa_slots(
            nodes_.get(v.node_id)->data_offset, nb);
        NodeComponentState* cs = nodes_.component_state(v.node_id);
        for (int p = lo; p < hi; p++) {
            int32_t& cell = lane[p - before];
            if (cell != INVALID_COMPONENT_SLOT) {
                result.unused_slots.push_back(slots[p - start]);
                swa_lru_.move_tail(cell, now);
                continue;
            }
            cell = slots[p - start];
            swa_lru_.push_tail(cell, v.node_id, (uint32_t)(p - before), now);
            cs->swa_live_blocks++;
        }
    }
    write_unlock();
    return finish();
}

void RadixShmemTree::retire_swa(uint32_t id, std::vector<int32_t>& out) {
    if (!has_swa_) return;
    NodeComponentState* cs = nodes_.component_state(id);
    if (cs->swa_live_blocks == 0) return;
    TreeNode* node = nodes_.get(id);
    const int nb = (int)node->num_blocks;
    int32_t* lane = data_.get_swa_slots(node->data_offset, nb);
    for (int i = 0; i < nb; i++) {
        if (lane[i] == INVALID_COMPONENT_SLOT) continue;
        swa_lru_.unlink(lane[i]);
        out.push_back(lane[i]);
        lane[i] = INVALID_COMPONENT_SLOT;
    }
    cs->swa_live_blocks = 0;
}

int RadixShmemTree::evict_swa_walk(int need_slots) {
    if (!has_swa_) return 0;
    std::vector<int32_t> tmp;
    uint32_t cur = swa_lru_.head();
    while (cur != INVALID_SLOT_LINK && (int)tmp.size() < need_slots) {
        const int32_t slot = (int32_t)cur;
        const SlotLruEntry e = swa_lru_.entry(slot);
        cur = e.next;
        NodeComponentState* cs = nodes_.component_state(e.node_id);
        if (cs->swa_ref_count.load(std::memory_order_relaxed) != 0) continue;
        TreeNode* node = nodes_.get(e.node_id);
        data_.get_swa_slots(node->data_offset, (int)node->num_blocks)[e.idx] =
            INVALID_COMPONENT_SLOT;
        cs->swa_live_blocks--;
        swa_lru_.unlink(slot);
        tmp.push_back(slot);
    }
    int got = swa_mempool_.recycle(tmp.data(), (int)tmp.size());
    if (got != (int)tmp.size())
        fprintf(stderr, "[radixshmem] WARNING: SWA evict recycled %d/%d slots "
                "(duplicate or foreign slot id)\n", got, (int)tmp.size());
    return got;
}

} // namespace radixshmem
