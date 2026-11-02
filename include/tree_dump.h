// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace radixshmem {

class RadixShmemTree;

// A copy of one tree for monitoring. Nodes are in BFS order, root first, so a
// node's parent always comes before it.
struct TreeDump {
    std::vector<int32_t>  parent;        // index into these arrays, -1 for the root
    std::vector<uint32_t> level;         // edges from the root
    std::vector<uint32_t> depth;         // blocks on the path before the node's first block
    std::vector<uint32_t> num_blocks;
    std::vector<uint32_t> swa_blocks;    // blocks holding an SWA slot
    std::vector<int64_t>  ref_count;     // FULL + SWA + Mamba pins
    std::vector<uint64_t> last_access;   // steady_clock ns
    std::vector<uint8_t>  mamba;         // 1 = a Mamba checkpoint at the node's end
    std::vector<uint64_t> hashes;        // num_blocks per node, concatenated in node order

    uint64_t now = 0;                    // steady_clock ns when the walk started
    uint32_t nodes_in_pool = 0;
    uint64_t total_blocks = 0;           // the header's counter
    uint32_t torn = 0;                   // nodes / child entries skipped as inconsistent
    bool     truncated = false;          // max_nodes or max_blocks reached
    int32_t  block_size = 0;             // tokens per block
    int32_t  register_chunk_size = 0;
    uint64_t reset_generation = 0;
    bool     is_distributed = false;
    uint32_t rank = 0;
    std::string node_name;
    std::string cluster_id;
};

// Copies the tree without taking a lock or writing shm, like check_health(): a
// writer racing the walk can leave the copy slightly inconsistent, never out of
// bounds. 0 = no limit.
TreeDump dump_tree(const RadixShmemTree& tree, uint32_t max_nodes = 0, uint64_t max_blocks = 0);

} // namespace radixshmem
