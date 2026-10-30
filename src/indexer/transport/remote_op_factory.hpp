// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "radixshmem_config.h"   // feature macros (CMake-generated, build tree)

#ifndef RADIXSHMEM_NO_RDMA

#include "bilateral_transport.h"   // IRemoteOp{Client,Server}Transport

#include <cstdint>
#include <memory>
#include <string>

namespace radixshmem {

struct PeerEntry;

// Creates the gRPC transport behind the typed service interface.
std::unique_ptr<IRemoteOpServerTransport> make_remote_op_server(
    uint16_t listen_port, RemoteOpAddr& addr);

std::unique_ptr<IRemoteOpClientTransport> make_remote_op_client(
    const PeerEntry& peer, std::string& endpoint);

}  // namespace radixshmem

#endif  // !RADIXSHMEM_NO_RDMA
