// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "shmradix/types.h"
#include <cstdint>
#include <vector>

namespace shmradix {

class ChildTable {
public:
    ChildTable() = default;
    ChildTable(uint32_t* buckets, ChainHTEntry* entries,
           uint32_t bucket_count, uint32_t entry_capacity,
           uint32_t* free_head);

    uint32_t lookup(uint32_t parent_id, uint64_t token_hash) const;
    bool insert(uint32_t parent_id, uint64_t token_hash,
                uint32_t child_node_id, TreeNode* parent_node);
    void remove(uint32_t parent_id, uint64_t token_hash,
                TreeNode* parent_node);
    void update_child(uint32_t parent_id, uint64_t token_hash,
                      uint32_t new_child_id);

    uint32_t find_any_child(const TreeNode* parent) const;
    void collect_children(const TreeNode* parent,
                          std::vector<uint32_t>& out) const;

    static void init_free_list(ChainHTEntry* entries, uint32_t capacity,
                               uint32_t& free_head);

    // Clear all children in place (empty buckets, rebuild the entry free list).
    void reset();

private:
    uint32_t bucket_index(uint64_t token_hash) const;
    uint32_t alloc_entry();
    void free_entry(uint32_t idx);

    uint32_t* buckets_ = nullptr;
    ChainHTEntry* entries_ = nullptr;
    uint32_t bucket_count_ = 0;
    uint32_t mask_ = 0;
    uint32_t entry_capacity_ = 0;
    uint32_t* free_head_ = nullptr;
};

} // namespace shmradix
