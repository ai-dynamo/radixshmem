// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hash.h"
#include "health_check.h"
#include "radix_tree.h"
#include "types.h"
#include <string>
#include <utility>
#include <vector>

namespace radixshmem {

class LocalTreeConnector {
public:
    explicit LocalTreeConnector(const std::string& name);
    ~LocalTreeConnector();

    LocalTreeConnector(LocalTreeConnector&& other) noexcept;
    LocalTreeConnector& operator=(LocalTreeConnector&& other) noexcept;
    LocalTreeConnector(const LocalTreeConnector&) = delete;
    LocalTreeConnector& operator=(const LocalTreeConnector&) = delete;

    InsertResult insert(const uint64_t* hashes, int n,
                        const int32_t* slots, int num_slots,
                        int start = 0, bool auto_recycle = true,
                        ComponentType component = ComponentType::FULL);
    int allocate_slots(int num_slots, std::vector<int32_t>& out,
                       ComponentType component = ComponentType::FULL);
    void recycle_slots(const int32_t* slots, int num,
                       ComponentType component = ComponentType::FULL);

    void reset();

    static std::vector<HashType> compute_hashes(const int64_t* token_ids,
                                                 int num_tokens,
                                                 int tokens_per_block);

    uint32_t total_radix_nodes() const;
    uint64_t total_blocks() const;
    float data_pool_utilization() const;
    uint32_t mempool_total() const;
    uint32_t mempool_free() const;
    uint32_t mempool_used() const;
    void print_tree() const;
    int check_health(HealthReport* report = nullptr) const;

    const std::string& name() const;

    RadixShmemTree& tree() { return tree_; }
    const RadixShmemTree& tree() const { return tree_; }

private:
    RadixShmemTree tree_;
    std::string name_;
};

} // namespace radixshmem
