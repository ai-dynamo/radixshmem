// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The Mooncake backend's only public surface.
//
// Mooncake's headers require C++20 (transfer_engine.h pulls in <span>) while the
// rest of this project is C++17, so mooncake_backend.cpp is compiled as a
// self-contained C++20 translation unit and every Mooncake type stays inside it.
// Nothing but this factory crosses the boundary, which is also what keeps the
// abstraction honest: callers cannot reach the engine even if they wanted to.

#include "shmradix/transfer.h"

#include <memory>

namespace shmradix {

// Returns nullptr when the engine fails to initialise (the reason is logged).
std::unique_ptr<ITransferBackend> create_mooncake_backend(const TransferConfig& cfg);

} // namespace shmradix
