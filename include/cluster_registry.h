// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cluster_manifest.h"   // PeerEntry

#include <vector>

namespace radixshmem {

struct IClusterRegistry {
    virtual bool run_rendezvous(std::vector<PeerEntry>& out) = 0;
    virtual ~IClusterRegistry() = default;
};

} // namespace radixshmem
