// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Internal matched-path views used by ComponentSet and tree walkers.

#include "component_types.h"

#include <cstdint>

namespace radixshmem {

// One matched node fragment. Lane pointers cover the FULL node while
// matched_blocks marks the consumed prefix; borrowed, read-lock lifetime only.
struct MatchedNodeSpanView {
    uint32_t node_id;
    uint32_t absolute_start;          // block offset of this node from root
    uint32_t node_blocks;             // full node length
    uint32_t matched_blocks;          // consumed prefix, 1..node_blocks
    const uint64_t* hashes;
    const int32_t* full_slots;
    const int32_t* swa_slots;
    int32_t mamba_checkpoint_slot;    // INVALID_COMPONENT_SLOT = none
};

// Ordered root-to-terminal matched path. Borrowed; never stored past the call.
struct MatchedPathView {
    const MatchedNodeSpanView* spans = nullptr;
    uint32_t num_spans = 0;
    uint32_t topology_match_length = 0;   // sum of matched_blocks
};

struct ComponentQueryContext {
    ComponentConfig config{};   // region ComponentConfig (window W etc.)
    uint32_t mask = 0;          // requested COMPONENT_MASK_* bits
    MatchedPathView path;
};

} // namespace radixshmem
