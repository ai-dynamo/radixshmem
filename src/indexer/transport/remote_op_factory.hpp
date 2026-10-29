// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "radixshmem_config.h"   // feature macros (CMake-generated, build tree)

#ifndef RADIXSHMEM_NO_RDMA

#include "bilateral_transport.h"   // IRemoteOp{Client,Server}Transport

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace radixshmem {

class RdmaContext;
class DcInitiator;
class DcTarget;
struct PeerEntry;
struct RemoteOpAddr;

// The only place that names concrete remote-op backends; `rot` is a REMOTE_OP_TRANSPORT_* value.

// Server backend for `rot`; fills `addr` with what peers dial (TCP port or DC lane DCT numbers).
std::unique_ptr<IRemoteOpServerTransport> make_remote_op_server(
    uint32_t rot, uint16_t listen_port, RdmaContext& ctx,
    uint64_t dc_access_key, RemoteOpAddr& addr);

// Client sessions of one connector; owns the DC initiator/target stack its DC sessions share.
class RemoteOpClientFactory {
public:
    RemoteOpClientFactory(uint32_t rot, RdmaContext& ctx, uint64_t dc_access_key);
    ~RemoteOpClientFactory();

    RemoteOpClientFactory(const RemoteOpClientFactory&) = delete;
    RemoteOpClientFactory& operator=(const RemoteOpClientFactory&) = delete;

    // Session to `peer`; `endpoint` gets its name. Throws when the peer serves no remote ops.
    std::unique_ptr<IRemoteOpClientTransport> open(const PeerEntry& peer,
                                                   int client_rank,
                                                   std::string& endpoint);

private:
    uint32_t     rot_;
    RdmaContext& ctx_;
    uint64_t     dc_access_key_;
    std::unique_ptr<DcInitiator> dc_ini_;
    std::unique_ptr<DcTarget>    dc_tgt_;
    std::shared_ptr<std::mutex>  dc_stack_mu_;
};

}  // namespace radixshmem

#endif  // !RADIXSHMEM_NO_RDMA
