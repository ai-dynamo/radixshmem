// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "finalize_fn.h"
#include "types.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace shmradix {

struct PinBlockRange {
    uint32_t start = 0;
    uint32_t end = 0;
    bool empty() const noexcept { return start == end; }
};

// Release data; swa.start records A. source_rank tags the pinned tree
// (-1 = local). The release re-walks CALLER-supplied hashes; `tail` holds the
// tree's own hashes for the deepest pinned node's blocks past the caller's
// keys (positions [end - size, end), end = the deepest lane END), which the
// caller lacks. Any lane may pin such a terminal, so the tail is lane-neutral.
struct PinRecord {
    PinBlockRange full;
    PinBlockRange swa;
    int32_t mamba_end = INVALID_COMPONENT_SLOT;
    int32_t source_rank = -1;
    std::vector<uint64_t> tail;

    bool empty() const noexcept {
        return full.empty() && swa.empty() &&
               mamba_end == INVALID_COMPONENT_SLOT;
    }
};

struct TreeComponentQueryResult {
    TreeComponentQueryResult() = default;

    RemoteStatus status = RemoteStatus::OK;
    uint32_t topology_match_length = 0;
    uint32_t common_hit = 0;
    std::vector<int32_t> full_slots;
    uint32_t swa_start = 0;
    std::vector<int32_t> swa_slots;
    int32_t mamba_checkpoint_slot = INVALID_COMPONENT_SLOT;
    PinRecord pin;
};

// ---- Public component query surface (RadixClient::query, mask overload). ----

// Owning slot copy; slot IDs are only valid on source_rank's SlotStore.
// offset = the run's block position within [0, common_hit).
struct ComponentSlotFragment {
    int32_t source_rank = -1;
    int offset = 0;
    std::vector<int32_t> slot_ids;
};

// Mamba checkpoint at the exact node END common_hit, on source_rank —
// always the tail source of the result.
struct ComponentCheckpoint {
    int32_t source_rank = -1;
    int32_t slot_id = INVALID_COMPONENT_SLOT;
};

// status != OK: no other field is valid. full_fragments tile [0, common_hit)
// sorted/gapless; SWA is one whole window; finalize one-shot on lock hits.
struct ComponentQueryResult {
    RemoteStatus status = RemoteStatus::OK;
    int common_hit = 0;
    std::vector<ComponentSlotFragment> full_fragments;
    ComponentSlotFragment swa;
    std::optional<ComponentCheckpoint> mamba;
    int rdma_read_count = 0;
    int rdma_atomic_count = 0;
    FinalizeFn finalize;
};

} // namespace shmradix
