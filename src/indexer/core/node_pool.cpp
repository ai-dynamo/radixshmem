// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "node_pool.hpp"

#include <cassert>
#include <cstring>
#include <stdexcept>

namespace radixshmem {

NodePool::NodePool(TreeNode* nodes, uint32_t capacity,
                   uint32_t* free_head, std::atomic<uint32_t>* pool_size,
                   NodeComponentState* cst)
    : nodes_(nodes), capacity_(capacity),
      free_head_(free_head), pool_size_(pool_size), cst_(cst) {}

void NodePool::init_free_list(TreeNode* nodes, uint32_t capacity,
                              uint32_t& free_head) {
    free_head = 0;
    for (uint32_t i = 0; i < capacity; i++) {
        memset(&nodes[i], 0, sizeof(TreeNode));
        // 0 is a valid Mamba slot id; free-list nodes must not claim it.
        nodes[i].mamba_checkpoint_slot = INVALID_COMPONENT_SLOT;
        nodes[i].next_free = (i + 1 < capacity) ? i + 1 : INVALID_NODE_ID;
    }
}

void NodePool::reset_component_state(NodeComponentState& s) {
    s.swa_ref_count.store(0, std::memory_order_relaxed);
    s.mamba_ref_count.store(0, std::memory_order_relaxed);
    s.swa_live_blocks = 0;
}

uint32_t NodePool::alloc() {
    if (*free_head_ == INVALID_NODE_ID)
        return INVALID_NODE_ID;

    uint32_t id = *free_head_;
    TreeNode* node = &nodes_[id];
    *free_head_ = node->next_free;
    pool_size_->fetch_add(1, std::memory_order_relaxed);

    memset(node, 0, sizeof(TreeNode));
    node->parent_id = INVALID_NODE_ID;
    node->data_offset = INVALID_OFFSET;
    node->mamba_checkpoint_slot = INVALID_COMPONENT_SLOT;
    node->leaf_prev = INVALID_NODE_ID;
    node->leaf_next = INVALID_NODE_ID;
    node->first_child_entry = HT_END;
    node->ref_count.store(0, std::memory_order_relaxed);
    node->last_access_time.store(0, std::memory_order_relaxed);
    if (cst_)
        reset_component_state(cst_[id]);
    return id;
}

void NodePool::free(uint32_t id) {
    TreeNode* node = &nodes_[id];
    // Invalidate structural fields so stale node IDs fail revalidation.
    node->parent_id = INVALID_NODE_ID;
    node->num_blocks = 0;
    node->data_offset = INVALID_OFFSET;
    node->mamba_checkpoint_slot = INVALID_COMPONENT_SLOT;
    if (cst_)
        reset_component_state(cst_[id]);
    node->next_free = *free_head_;
    *free_head_ = id;
    pool_size_->fetch_sub(1, std::memory_order_relaxed);
}

void NodePool::reset() {
    init_free_list(nodes_, capacity_, *free_head_);
    if (cst_)
        for (uint32_t i = 0; i < capacity_; i++)
            reset_component_state(cst_[i]);
    pool_size_->store(0, std::memory_order_relaxed);
}

TreeNode* NodePool::get(uint32_t id) const {
    return &nodes_[id];
}

NodeComponentState* NodePool::component_state(uint32_t id) const {
    return cst_ ? &cst_[id] : nullptr;
}

uint32_t NodePool::size() const {
    return pool_size_->load(std::memory_order_relaxed);
}

} // namespace radixshmem
