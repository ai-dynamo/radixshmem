// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <vector>

namespace radixshmem {

struct RhtHop {
    int      position;
    uint32_t tree_node_id;
    uint32_t num_blocks;
};

inline int follow_rht_chain(const uint64_t* hashes, int n, int pos, uint32_t len,
                            const std::function<bool(uint64_t, RhtHop&)>& lookup,
                            std::vector<RhtHop>& hops) {
    int end = pos + (int)len;
    while (len > 0 && end < n) {
        RhtHop hop{end, 0, 0};
        if (!lookup(hashes[end], hop) || hop.num_blocks == 0) break;
        hops.push_back(hop);
        len = hop.num_blocks;
        end += (int)len;
    }
    return end;
}

}  // namespace radixshmem
