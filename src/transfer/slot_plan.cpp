// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "slot_plan.h"

#include <string>

namespace radixshmem {

bool resolve_peer_geometry(const StoreGeometry& mine, const StoreGeometry* published,
                           uint64_t seg_base, uint64_t seg_len, StoreGeometry& out, std::string& why) {
    StoreGeometry layout;
    if (published == nullptr) {
        if (mine.pool_mask == COMPONENT_MASK_FULL) {
            // Single-pool contract: count = whatever the segment holds.
            const uint64_t sb = mine[ComponentType::FULL].slot_bytes;
            if (seg_len < sb || seg_len % sb != 0) {
                why = "publishes " + std::to_string(seg_len) +
                      " bytes, which is not a whole number of " + std::to_string(sb) + "-byte slots";
                return false;
            }
            layout.pool_mask = COMPONENT_MASK_FULL;
            layout[ComponentType::FULL] = PoolGeometry{0, seg_len / sb, sb};
        } else {
            layout = mine;   // assumed identical; still checked to fit
        }
    } else {
        layout = *published;
        if (layout.pool_mask != mine.pool_mask) {
            why = "has pool mask " + std::to_string(layout.pool_mask) + ", we have " +
                  std::to_string(mine.pool_mask) + " — the two stores hold different KV components";
            return false;
        }
        for (int i = 0; i < NUM_COMPONENT_TYPES; ++i) {
            const ComponentType t = static_cast<ComponentType>(i);
            if (!layout.has(t)) continue;
            if (layout[t].slot_bytes != mine[t].slot_bytes) {
                why = std::string("has a ") + std::to_string(layout[t].slot_bytes) + "-byte " +
                      component_type_name(t) + " slot stride, we have " +
                      std::to_string(mine[t].slot_bytes) +
                      " — slot ids would not mean the same bytes on both sides";
                return false;
            }
            if (layout[t].num_slots == 0) {
                why = std::string("publishes an empty ") + component_type_name(t) + " pool";
                return false;
            }
        }
    }

    // Pools in order, disjoint, inside the segment.
    uint64_t prev_end = 0;
    for (int i = 0; i < NUM_COMPONENT_TYPES; ++i) {
        const ComponentType t = static_cast<ComponentType>(i);
        if (!layout.has(t)) continue;
        const PoolGeometry& p = layout[t];
        uint64_t bytes = 0, end = 0;
        if (__builtin_mul_overflow(p.num_slots, p.slot_bytes, &bytes) ||
            __builtin_add_overflow(p.base, bytes, &end) || p.base < prev_end || end > seg_len) {
            why = std::string("its ") + component_type_name(t) + " pool [" + std::to_string(p.base) +
                  ", +" + std::to_string(p.num_slots) + " x " + std::to_string(p.slot_bytes) +
                  ") does not fit its " + std::to_string(seg_len) + "-byte segment";
            return false;
        }
        prev_end = end;
    }

    out = layout.rebased(seg_base);
    return true;
}

bool plan_slots(const SlotStore& store, const StoreGeometry& peer, ComponentType c,
                const int64_t* local_slots, const int64_t* remote_slots, int n, SlotPlan& out,
                std::string& why) {
    out = SlotPlan{};
    if (n <= 0 || local_slots == nullptr || remote_slots == nullptr) {
        why = "no slots";
        return false;
    }
    if (!component_type_valid(c)) {
        why = "invalid ComponentType " + std::to_string(static_cast<int>(c));
        return false;
    }
    if (!store.has_pool(c)) {
        why = std::string("this store has no ") + component_type_name(c) + " pool";
        return false;
    }
    if (!peer.has(c)) {
        why = std::string("the peer has no ") + component_type_name(c) + " pool";
        return false;
    }
    const PoolGeometry& rp = peer[c];
    const uint64_t sb = rp.slot_bytes;   // == store.slot_bytes(c), checked at attach
    const uint64_t local_n = store.num_slots(c);

    for (int i = 0; i < n; ++i) {
        if (local_slots[i] < 0 || (uint64_t)local_slots[i] >= local_n) {
            why = std::string("local ") + component_type_name(c) + " slot " +
                  std::to_string(local_slots[i]) + " out of range [0," + std::to_string(local_n) + ")";
            return false;
        }
        if (remote_slots[i] < 0 || (uint64_t)remote_slots[i] >= rp.num_slots) {
            why = std::string("remote ") + component_type_name(c) + " slot " +
                  std::to_string(remote_slots[i]) + " out of range [0," + std::to_string(rp.num_slots) +
                  ")";
            return false;
        }
    }

    for (int i = 0; i < n;) {
        int run = 1;
        while (i + run < n && local_slots[i + run] == local_slots[i + run - 1] + 1 &&
               remote_slots[i + run] == remote_slots[i + run - 1] + 1)
            ++run;
        out.local.push_back(reinterpret_cast<uint64_t>(store.slot_addr(c, (uint64_t)local_slots[i])));
        out.remote.push_back(rp.base + (uint64_t)remote_slots[i] * sb);
        out.length.push_back((uint64_t)run * sb);
        i += run;
    }
    return true;
}

} // namespace radixshmem
