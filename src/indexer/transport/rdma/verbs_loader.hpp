// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// rdma-core is loaded on first use, so a process that opens no RDMA device needs none installed.

#pragma once

#include <string>

namespace radixshmem::verbs {

// Throw, saying what to install, unless libibverbs (libmlx5 for DC) loads with every function used.
void require_ibverbs();
void require_mlx5();

// Why libibverbs cannot be used, or empty when it can.
std::string ibverbs_error();

// A libibverbs function this build does not require, or nullptr when the library or the function is missing.
void* ibverbs_symbol(const char* name);

}  // namespace radixshmem::verbs
