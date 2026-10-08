// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Address-range fusing, shared by every backend.
//
// A registration is not guaranteed to reach a peer as one buffer: Mooncake
// splits anything longer than max_mr_size into several chunks and publishes one
// BufferDesc per chunk (rdma_transport.cpp:303). A 40 GiB store therefore shows
// up remotely as a handful of descriptors that happen to be adjacent. Peers must
// fuse them back before computing addresses, or they would only ever see the
// first chunk.

#include <cstdint>
#include <vector>

namespace radixshmem {

struct AddrRange {
    uint64_t addr = 0;
    uint64_t len  = 0;
    uint64_t end() const { return addr + len; }
};

// Sorts by address and fuses ranges that touch or overlap. Zero-length ranges
// are dropped. The result is sorted and disjoint, with a gap between every pair.
std::vector<AddrRange> merge_contiguous(std::vector<AddrRange> in);

} // namespace radixshmem
