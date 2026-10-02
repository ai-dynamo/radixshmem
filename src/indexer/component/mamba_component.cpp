// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "component_set.hpp"
#include "shmradix/radix_tree.h"

#include <cassert>
#include <cstdio>

namespace shmradix {

// Deliberate sglang divergence (design §6.3): no checkpoint materialization
// at deeper Full hits — checkpoint density is entirely the publisher's choice.
InsertResult ShmRadixTree::insert_mamba(const uint64_t* hashes, int n,
                                        const int32_t* slots, int num_slots,
                                        bool auto_recycle) {
    set_write_op("insert_mamba");
    InsertResult result;

    // Recycle collected unused slots after unlock; returns result (used at exits).
    auto finish = [&]() -> InsertResult {
        if (auto_recycle && !result.unused_slots.empty())
            recycle_slots(result.unused_slots.data(),
                          (int)result.unused_slots.size(),
                          ComponentType::MAMBA);
        return result;
    };
    // Any refusal returns the whole batch unused with zero mutation.
    auto refuse = [&](InsertError e) -> InsertResult {
        result.error = e;
        if (slots != nullptr && num_slots > 0)
            result.unused_slots.assign(slots, slots + num_slots);
        return finish();
    };

    const ComponentConfig& config = header()->component_config;
    if (n <= 0 || hashes == nullptr || slots == nullptr || num_slots != 1 ||
        !is_component_config_valid(config.component_mask,
                                   config.swa_window_blocks) ||
        !(config.component_mask & COMPONENT_MASK_MAMBA))
        return refuse(InsertError::BAD_REQUEST);

    result.unused_slots.reserve(1);
    write_lock();
    QueryInternalDetail m = query_internal(hashes, n, 0, false);
    if (m.matched_blocks < n) {
        write_unlock();
        return refuse(InsertError::FULL_PATH_MISSING);
    }
    uint32_t end_id = m.last_node_id;
    const int end_before = n - m.last_node_matched_blocks;
    const int end_nb = (int)nodes_.get(end_id)->num_blocks;
    if (end_before + end_nb > n) {
        const uint32_t suffix_id = end_id;   // keeps its id past the split
        end_id = split_node(end_id, n - end_before);
        if (end_id == INVALID_NODE_ID) {
            write_unlock();
            return refuse(InsertError::DATA_POOL_FULL);
        }
        result.topology_changed = true;
        TreeNode* suffix = nodes_.get(suffix_id);
        result.split_node_id = suffix_id;
        result.split_num_blocks = suffix->num_blocks;
        result.split_first_hash = data_.get_hashes(suffix->data_offset)[0];
    }
    TreeNode* node = nodes_.get(end_id);
    if (node->mamba_checkpoint_slot == INVALID_COMPONENT_SLOT) {
        node->mamba_checkpoint_slot = slots[0];
        mamba_lru_.push_tail(slots[0], end_id, 0, now_ts());
    } else {
        result.unused_slots.push_back(slots[0]);   // FWW: the incumbent stays
    }
    write_unlock();
    return finish();
}

void ShmRadixTree::retire_mamba(uint32_t id, std::vector<int32_t>& out) {
    if (!has_mamba_) return;
    TreeNode* node = nodes_.get(id);
    if (node->mamba_checkpoint_slot == INVALID_COMPONENT_SLOT) return;
    mamba_lru_.unlink(node->mamba_checkpoint_slot);
    out.push_back(node->mamba_checkpoint_slot);
    node->mamba_checkpoint_slot = INVALID_COMPONENT_SLOT;
}

// Merge is gated on the parent END carrying no checkpoint; a leaf END
// checkpoint moves onto the merged node.
void ShmRadixTree::aux_on_merge(uint32_t parent_id, uint32_t leaf_id) {
    if (!has_mamba_) return;
    assert(nodes_.get(parent_id)->mamba_checkpoint_slot ==
           INVALID_COMPONENT_SLOT);
    TreeNode* leaf = nodes_.get(leaf_id);
    if (leaf->mamba_checkpoint_slot == INVALID_COMPONENT_SLOT)
        return;
    nodes_.get(parent_id)->mamba_checkpoint_slot = leaf->mamba_checkpoint_slot;
    mamba_lru_.entry(leaf->mamba_checkpoint_slot).node_id = parent_id;
    leaf->mamba_checkpoint_slot = INVALID_COMPONENT_SLOT;
}

int ShmRadixTree::evict_mamba_walk(int need_slots) {
    if (!has_mamba_) return 0;
    std::vector<int32_t> tmp;
    uint32_t cur = mamba_lru_.head();
    while (cur != INVALID_SLOT_LINK && (int)tmp.size() < need_slots) {
        const int32_t slot = (int32_t)cur;
        const SlotLruEntry e = mamba_lru_.entry(slot);
        cur = e.next;
        if (nodes_.component_state(e.node_id)->mamba_ref_count.load(
                std::memory_order_relaxed) != 0)
            continue;
        nodes_.get(e.node_id)->mamba_checkpoint_slot = INVALID_COMPONENT_SLOT;
        mamba_lru_.unlink(slot);
        tmp.push_back(slot);
    }
    int got = mamba_mempool_.recycle(tmp.data(), (int)tmp.size());
    if (got != (int)tmp.size())
        fprintf(stderr, "[shmradix] WARNING: Mamba evict recycled %d/%d slots "
                "(duplicate or foreign slot id)\n", got, (int)tmp.size());
    return got;
}

} // namespace shmradix
