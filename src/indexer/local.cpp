// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "shmradix/local.h"
#include "shmradix/hash.h"

namespace shmradix {

// ==================== LocalTreeConnector ====================

LocalTreeConnector::LocalTreeConnector(const std::string& name)
    : tree_(ShmRadixTree::attach(name)), name_(name) {}

LocalTreeConnector::~LocalTreeConnector() {
    tree_.detach();
}

LocalTreeConnector::LocalTreeConnector(LocalTreeConnector&& other) noexcept
    : tree_(std::move(other.tree_)), name_(std::move(other.name_)) {}

LocalTreeConnector& LocalTreeConnector::operator=(LocalTreeConnector&& other) noexcept {
    if (this != &other) {
        tree_.detach();
        tree_ = std::move(other.tree_);
        name_ = std::move(other.name_);
    }
    return *this;
}

InsertResult LocalTreeConnector::insert(const uint64_t* hashes, int n,
                                const int32_t* slots, int num_slots,
                                int start, bool auto_recycle,
                                ComponentType component) {
    return tree_.insert(hashes, n, slots, num_slots, start, auto_recycle,
                        component);
}

int LocalTreeConnector::allocate_slots(int num_slots, std::vector<int32_t>& out,
                                       ComponentType component) {
    return tree_.allocate_slots(num_slots, out, component);
}

void LocalTreeConnector::recycle_slots(const int32_t* slots, int num,
                                       ComponentType component) {
    tree_.recycle_slots(slots, num, component);
}

void LocalTreeConnector::reset() {
    tree_.reset();
}

std::vector<HashType> LocalTreeConnector::compute_hashes(const int64_t* token_ids,
                                                  int num_tokens,
                                                  int tokens_per_block) {
    int num_blocks = num_tokens / tokens_per_block;
    std::vector<HashType> hashes(num_blocks);
    if (num_blocks > 0)
        gen_hashes(token_ids, num_blocks * tokens_per_block, tokens_per_block,
                   hashes.data());
    return hashes;
}

uint32_t LocalTreeConnector::total_radix_nodes() const { return tree_.total_radix_nodes(); }
uint64_t LocalTreeConnector::total_blocks() const { return tree_.total_blocks(); }
float LocalTreeConnector::data_pool_utilization() const { return tree_.data_pool_utilization(); }
uint32_t LocalTreeConnector::mempool_total() const { return tree_.mempool_total(); }
uint32_t LocalTreeConnector::mempool_free() const { return tree_.mempool_free(); }
uint32_t LocalTreeConnector::mempool_used() const { return tree_.mempool_used(); }
void LocalTreeConnector::print_tree() const { tree_.print_tree(); }

int LocalTreeConnector::check_health(HealthReport* report) const {
    HealthReport r = shmradix::check_health(tree_);
    if (report) *report = r;
    return (int)r.flags;
}

const std::string& LocalTreeConnector::name() const { return name_; }

} // namespace shmradix
