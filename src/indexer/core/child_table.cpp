// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "child_table.hpp"

#include <cassert>

namespace radixshmem {

ChildTable::ChildTable(uint32_t* buckets, ChainHTEntry* entries,
               uint32_t bucket_count, uint32_t entry_capacity,
               uint32_t* free_head)
    : buckets_(buckets), entries_(entries),
      bucket_count_(bucket_count), mask_(bucket_count - 1),
      entry_capacity_(entry_capacity), free_head_(free_head) {}

void ChildTable::init_free_list(ChainHTEntry* entries, uint32_t capacity,
                            uint32_t& free_head) {
    for (uint32_t i = 0; i < capacity - 1; i++)
        entries[i].hash_next = i + 1;
    entries[capacity - 1].hash_next = HT_END;
    free_head = 0;
}

void ChildTable::reset() {
    for (uint32_t i = 0; i < bucket_count_; i++)
        buckets_[i] = HT_END;
    init_free_list(entries_, entry_capacity_, *free_head_);
}

uint32_t ChildTable::bucket_index(uint64_t token_hash) const {
    return (uint32_t)(token_hash ^ (token_hash >> 32)) & mask_;
}

uint32_t ChildTable::alloc_entry() {
    if (*free_head_ == HT_END) return HT_END;
    uint32_t idx = *free_head_;
    *free_head_ = entries_[idx].hash_next;
    return idx;
}

void ChildTable::free_entry(uint32_t idx) {
    entries_[idx].hash_next = *free_head_;
    *free_head_ = idx;
}

// ---- lookup ----

uint32_t ChildTable::lookup(uint32_t parent_id, uint64_t token_hash) const {
    uint32_t bi = bucket_index(token_hash);
    uint32_t idx = buckets_[bi];
    while (idx != HT_END) {
        const ChainHTEntry& e = entries_[idx];
        if (e.token_hash == token_hash)
            return e.child_node_id;
        idx = e.hash_next;
    }
    return INVALID_NODE_ID;
}

// ---- insert ----

bool ChildTable::insert(uint32_t parent_id, uint64_t token_hash,
                    uint32_t child_node_id, TreeNode* parent_node) {
    uint32_t ei = alloc_entry();
    if (ei == HT_END) return false;
    ChainHTEntry& e = entries_[ei];
    e.token_hash = token_hash;
    e.parent_id = parent_id;
    e.child_node_id = child_node_id;

    uint32_t bi = bucket_index(token_hash);
    e.hash_next = buckets_[bi];
    buckets_[bi] = ei;

    e.sibling_next = parent_node->first_child_entry;
    parent_node->first_child_entry = ei;
    return true;
}

// ---- remove ----

void ChildTable::remove(uint32_t parent_id, uint64_t token_hash,
                    TreeNode* parent_node) {
    uint32_t bi = bucket_index(token_hash);
    uint32_t* prev_ptr = &buckets_[bi];
    uint32_t idx = *prev_ptr;
    uint32_t found = HT_END;

    while (idx != HT_END) {
        ChainHTEntry& e = entries_[idx];
        if (e.token_hash == token_hash) {
            found = idx;
            *prev_ptr = e.hash_next;
            break;
        }
        prev_ptr = &e.hash_next;
        idx = e.hash_next;
    }
    if (found == HT_END) return;

    uint32_t* sprev = &parent_node->first_child_entry;
    uint32_t sidx = *sprev;
    while (sidx != HT_END) {
        if (sidx == found) {
            *sprev = entries_[sidx].sibling_next;
            break;
        }
        sprev = &entries_[sidx].sibling_next;
        sidx = entries_[sidx].sibling_next;
    }

    free_entry(found);
}

// ---- update_child ----

void ChildTable::update_child(uint32_t parent_id, uint64_t token_hash,
                           uint32_t new_child_id) {
    uint32_t bi = bucket_index(token_hash);
    uint32_t idx = buckets_[bi];
    while (idx != HT_END) {
        ChainHTEntry& e = entries_[idx];
        if (e.token_hash == token_hash) {
            e.child_node_id = new_child_id;
            return;
        }
        idx = e.hash_next;
    }
}

// ---- child traversal via sibling chain ----

uint32_t ChildTable::find_any_child(const TreeNode* parent) const {
    uint32_t idx = parent->first_child_entry;
    if (idx == HT_END) return INVALID_NODE_ID;
    return entries_[idx].child_node_id;
}

void ChildTable::collect_children(const TreeNode* parent,
                              std::vector<uint32_t>& out) const {
    uint32_t idx = parent->first_child_entry;
    while (idx != HT_END) {
        out.push_back(entries_[idx].child_node_id);
        idx = entries_[idx].sibling_next;
    }
}

} // namespace radixshmem
