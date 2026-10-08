// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "component_types.hpp"

namespace radixshmem {

// Stateless per-call facade over the fixed three components. Full is the
// topology hit itself; SWA/Mamba are resolved bottom-up from the terminal and
// common_hit is the deepest boundary all requested components accept.
class ComponentSet {
public:
    // Caller holds the read lock that keeps ctx.path's borrowed views alive.
    TreeComponentQueryResult query(const ComponentQueryContext& ctx) const;

private:
    // Deepest span END <= limit carrying a checkpoint; 0 if none.
    static uint32_t mamba_at_or_below(const MatchedPathView& path,
                                      uint32_t limit) noexcept;
    // Deepest b <= limit whose window [max(0,b-W), b) is all live; 0 if none.
    static uint32_t swa_at_or_below(const MatchedPathView& path, uint32_t window,
                                    uint32_t limit) noexcept;
};

// Pin coverage, pure index math: Full pins every span overlapping [full_lo,
// full_hi) whole (a partial terminal included); SWA/Mamba cover the aux
// boundary L. Caller supplies hashes.
struct PinCoverage {
    bool has_full = false;
    size_t full_from = 0;      // first span overlapping [full_lo, full_hi)
    size_t full_k = 0;         // deepest span overlapping [full_lo, full_hi)
    uint32_t full_cover_end = 0;
    size_t aux_k = 0;          // deepest span with absolute_start < L
    uint32_t aux_cover_end = 0;
    size_t swa_from = 0;       // first SWA-pinned span; num when none
    bool pin_mamba = false;
};

PinCoverage compute_pin_coverage(const MatchedNodeSpanView* spans, size_t num,
                                 uint32_t mask, uint32_t L, uint32_t swa_start,
                                 uint32_t full_lo, uint32_t full_hi);

// Write the Full range and the release tail: the deepest pinned span's hashes
// past the caller's keyed prefix (full_hi with a Full pin, else L), so a
// release can re-walk every pinned node whole. Mamba alone needs no tail
// (its pinned node ENDs exactly at L).
void record_pin_tail(const MatchedNodeSpanView* spans, size_t num,
                     const PinCoverage& cov, uint32_t full_hi, uint32_t L,
                     PinRecord& pin);

} // namespace radixshmem
