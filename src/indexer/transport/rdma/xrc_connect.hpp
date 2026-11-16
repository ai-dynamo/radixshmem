// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cluster_manifest.h"   // PeerEntry
#include "rdma_transport.hpp"

#include <memory>

namespace radixshmem {

// XRC SEND-QP bring-up: the XrcConnect RPC of `peer` pairs a RECV QP with it, then
// the QP goes to RTS. nullptr on failure, after a log line.
std::unique_ptr<XrcSendQp> xrc_connect(RdmaContext& ctx, ibv_cq* cq, const PeerEntry& peer,
                                       int my_rank, int timeout_sec);

} // namespace radixshmem
