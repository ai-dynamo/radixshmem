// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "component_set.hpp"

#include <cassert>

namespace radixshmem {

uint32_t ComponentSet::mamba_at_or_below(const MatchedPathView& path,
                                         uint32_t limit) noexcept {
    for (uint32_t i = path.num_spans; i-- > 0;) {
        const MatchedNodeSpanView& v = path.spans[i];
        const uint32_t end = v.absolute_start + v.node_blocks;
        if (end <= limit && v.mamba_checkpoint_slot != INVALID_COMPONENT_SLOT)
            return end;
    }
    return 0;
}

uint32_t ComponentSet::swa_at_or_below(const MatchedPathView& path,
                                       uint32_t window, uint32_t limit) noexcept {
    if (limit == 0) return 0;
    uint32_t si = path.num_spans - 1;
    auto live = [&](uint32_t p) {
        while (path.spans[si].absolute_start > p) si--;
        const MatchedNodeSpanView& v = path.spans[si];
        return v.swa_slots[p - v.absolute_start] != INVALID_COMPONENT_SLOT;
    };
    uint32_t hi = limit;
    while (hi > 0) {
        uint32_t lo = hi;
        while (lo > 0 && live(lo - 1)) lo--;
        if (hi - lo >= window || lo == 0) return hi;
        hi = lo;
        while (hi > 0 && !live(hi - 1)) hi--;
    }
    return 0;
}

TreeComponentQueryResult
ComponentSet::query(const ComponentQueryContext& ctx) const {
    TreeComponentQueryResult result;
    const MatchedPathView& path = ctx.path;
    const bool want_full  = (ctx.mask & COMPONENT_MASK_FULL) != 0;
    const bool want_swa   = (ctx.mask & COMPONENT_MASK_SWA) != 0;
    const bool want_mamba = (ctx.mask & COMPONENT_MASK_MAMBA) != 0;
    const uint32_t window = ctx.config.swa_window_blocks;
    result.topology_match_length = path.topology_match_length;
    if (want_full) {
        result.full_slots.reserve(path.topology_match_length);
        for (uint32_t i = 0; i < path.num_spans; i++) {
            const MatchedNodeSpanView& v = path.spans[i];
            result.full_slots.insert(result.full_slots.end(), v.full_slots,
                                     v.full_slots + v.matched_blocks);
        }
    }

    // Intersection by alternating descent; each step scans a disjoint range.
    uint32_t b = path.topology_match_length;
    while (b > 0 && (want_swa || want_mamba)) {
        if (want_mamba) b = mamba_at_or_below(path, b);
        if (!want_swa || b == 0) break;
        const uint32_t s = swa_at_or_below(path, window, b);
        const bool settled = (s == b) || !want_mamba;
        b = s;
        if (settled) break;
    }
    result.common_hit = b;
    if (b == 0) return result;

    if (want_mamba) {
        for (uint32_t i = path.num_spans; i-- > 0;) {
            const MatchedNodeSpanView& v = path.spans[i];
            if (v.absolute_start + v.node_blocks == b) {
                result.mamba_checkpoint_slot = v.mamba_checkpoint_slot;
                break;
            }
        }
    }
    if (want_swa) {
        result.swa_start = b > window ? b - window : 0;
        result.swa_slots.reserve(b - result.swa_start);
        for (uint32_t s = 0; s < path.num_spans; s++) {
            const MatchedNodeSpanView& span = path.spans[s];
            const uint32_t lo = span.absolute_start > result.swa_start
                                    ? span.absolute_start : result.swa_start;
            uint32_t hi = span.absolute_start + span.matched_blocks;
            if (hi > b) hi = b;
            for (uint32_t p = lo; p < hi; p++) {
                const int32_t slot = span.swa_slots[p - span.absolute_start];
                assert(slot != INVALID_COMPONENT_SLOT);
                result.swa_slots.push_back(slot);
            }
        }
    }
    return result;
}

// Full pins every node overlapping [full_lo, full_hi) whole, partial terminal
// included; SWA/Mamba pin the nodes overlapping their window / END whole.
PinCoverage compute_pin_coverage(const MatchedNodeSpanView* spans, size_t num,
                                 uint32_t mask, uint32_t L, uint32_t swa_start,
                                 uint32_t full_lo, uint32_t full_hi) {
    PinCoverage cov;
    cov.swa_from = num;
    if (num == 0) return cov;
    if ((mask & COMPONENT_MASK_FULL) != 0 && full_lo < full_hi) {
        cov.full_from = num;
        for (size_t i = 0; i < num; i++) {
            const uint32_t s = spans[i].absolute_start;
            if (s >= full_hi) break;                          // spans tile upward
            if (s + spans[i].node_blocks <= full_lo) continue; // ends before lo
            if (cov.full_from == num) cov.full_from = i;
            cov.full_k = i;
        }
        if (cov.full_from < num) {
            cov.has_full = true;
            cov.full_cover_end = spans[cov.full_k].absolute_start +
                                 spans[cov.full_k].node_blocks;
        }
    }
    if ((mask & (COMPONENT_MASK_SWA | COMPONENT_MASK_MAMBA)) != 0 && L > 0) {
        for (size_t i = 0; i < num; i++) {
            if (spans[i].absolute_start < L) cov.aux_k = i;
            else break;
        }
        cov.aux_cover_end =
            spans[cov.aux_k].absolute_start + spans[cov.aux_k].node_blocks;
        if ((mask & COMPONENT_MASK_SWA) != 0 && swa_start < L) {
            for (size_t i = 0; i <= cov.aux_k; i++) {
                if (spans[i].absolute_start + spans[i].node_blocks > swa_start) {
                    cov.swa_from = i;
                    break;
                }
            }
        }
        cov.pin_mamba =
            (mask & COMPONENT_MASK_MAMBA) != 0 &&
            spans[cov.aux_k].mamba_checkpoint_slot != INVALID_COMPONENT_SLOT;
    }
    return cov;
}

void record_pin_tail(const MatchedNodeSpanView* spans, size_t num,
                     const PinCoverage& cov, uint32_t full_hi, uint32_t L,
                     PinRecord& pin) {
    size_t k;
    uint32_t hi, cover_end;
    if (cov.has_full) {
        // full_k is at or below aux_k (full_hi >= L), so its tail covers the
        // SWA range's END as well.
        pin.full = {spans[cov.full_from].absolute_start, cov.full_cover_end};
        k = cov.full_k;
        hi = full_hi;
        cover_end = cov.full_cover_end;
    } else if (cov.swa_from < num) {
        // SWA-only: the caller's keys equal the tree's on [0, L).
        k = cov.aux_k;
        hi = L;
        cover_end = cov.aux_cover_end;
    } else {
        return;
    }
    pin.tail.clear();
    if (cover_end > hi) {
        const MatchedNodeSpanView& v = spans[k];
        const uint32_t from = hi - v.absolute_start;
        pin.tail.assign(v.hashes + from, v.hashes + v.node_blocks);
    }
}

} // namespace radixshmem
