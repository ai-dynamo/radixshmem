// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The NIXL backend's only public surface.
//
// NIXL's headers require C++20 (nixl.h pulls in <span>), same constraint as
// Mooncake, so nixl_backend.cpp is compiled as a self-contained C++20
// translation unit and every NIXL type stays inside it. Nothing but this
// factory crosses the boundary.

#include "transfer.h"

#include <memory>

namespace radixshmem {

// Returns nullptr when the agent fails to initialise (the reason is logged).
std::unique_ptr<ITransferBackend> create_nixl_backend(const TransferConfig& cfg);

} // namespace radixshmem
