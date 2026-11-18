// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Slot-granular transfer planning: the peer's geometry checked against ours, and slot ids turned into
// address ranges. radixshmem.transfer.SlotTransfer drives a transfer backend with what these return.

#include "component_kind.h"
#include "slot_geometry.h"
#include "slot_store.h"

#include <cstdint>
#include <string>
#include <vector>

namespace radixshmem {

inline constexpr uint64_t PEER_SLOT_BYTES_UNCHECKED = UINT64_MAX;

// The peer's absolute geometry, or false with `why`. `published` is its layout from the registry
// (offsets into its segment), or nullptr to take the count from the segment (FULL-only stores) or
// assume ours (multi-pool stores). seg_base / seg_len describe the peer's registered segment.
bool resolve_peer_geometry(const StoreGeometry& mine, const StoreGeometry* published,
                           uint64_t seg_base, uint64_t seg_len, StoreGeometry& out, std::string& why);

struct SlotPlan {
    std::vector<uint64_t> local;    // local addresses
    std::vector<uint64_t> remote;   // peer addresses
    std::vector<uint64_t> length;
};

// local_slots[i] <-> remote_slots[i] in pool `c`, runs contiguous on both sides fused into one range.
// All or nothing: false with `why` if any slot is out of range on its side or a pool is missing.
bool plan_slots(const SlotStore& store, const StoreGeometry& peer, ComponentType c,
                const int64_t* local_slots, const int64_t* remote_slots, int n, SlotPlan& out,
                std::string& why);

} // namespace radixshmem
