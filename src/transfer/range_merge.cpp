// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "range_merge.hpp"

#include <algorithm>

namespace shmradix {

std::vector<AddrRange> merge_contiguous(std::vector<AddrRange> in) {
    std::vector<AddrRange> out;
    in.erase(std::remove_if(in.begin(), in.end(),
                            [](const AddrRange& r) { return r.len == 0; }),
             in.end());
    if (in.empty()) return out;

    std::sort(in.begin(), in.end(),
              [](const AddrRange& a, const AddrRange& b) { return a.addr < b.addr; });

    out.push_back(in[0]);
    for (size_t i = 1; i < in.size(); ++i) {
        AddrRange& last = out.back();
        // `<=` fuses on touch, not just on overlap: two chunks split out of one
        // registration are exactly adjacent.
        if (in[i].addr <= last.end()) {
            const uint64_t end = std::max(last.end(), in[i].end());
            last.len = end - last.addr;
        } else {
            out.push_back(in[i]);
        }
    }
    return out;
}

} // namespace shmradix
