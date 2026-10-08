// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.h"
#include <atomic>
#include <cstdint>

namespace radixshmem {

class NodePool {
public:
    NodePool() = default;
    // cst: ComponentStateTable base (entry i belongs to node id i); entry
    // lifecycle strictly follows node alloc/free. nullptr = table absent.
    NodePool(TreeNode* nodes, uint32_t capacity,
             uint32_t* free_head, std::atomic<uint32_t>* pool_size,
             NodeComponentState* cst = nullptr);

    uint32_t alloc();
    void free(uint32_t id);

    // Clear all nodes in place (caller re-allocates the root afterwards).
    void reset();

    TreeNode* get(uint32_t id) const;
    NodeComponentState* component_state(uint32_t id) const;
    uint32_t size() const;

    static void init_free_list(TreeNode* nodes, uint32_t capacity,
                               uint32_t& free_head);

    // Idle state for an unused/new entry (refs 0, LRU links INVALID).
    static void reset_component_state(NodeComponentState& s);

private:
    TreeNode* nodes_ = nullptr;
    uint32_t capacity_ = 0;
    uint32_t* free_head_ = nullptr;
    std::atomic<uint32_t>* pool_size_ = nullptr;
    NodeComponentState* cst_ = nullptr;
};

} // namespace radixshmem
