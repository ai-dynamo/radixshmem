// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "component_types.h"   // QueryReply, RemoteQuery, PinRecord
#include "types.h"             // AllocateReply, InsertResult, RemoteStatus

#include <cstdint>
#include <vector>

namespace radixshmem {

// Typed remote operations, independent of gRPC encoding.
class IRemoteOpClientTransport {
public:
    virtual ~IRemoteOpClientTransport() = default;
    virtual AllocateReply allocate(int n, ComponentType component = ComponentType::FULL) = 0;
    virtual AllocateComponentsReply allocate_components(
        const std::vector<ComponentCount>& components) = 0;
    virtual QueryReply query(const uint64_t* hashes, int n, const RemoteQuery& q) = 0;
    virtual InsertResult insert(const uint64_t* hashes, int n, int start,
                                int length, const int32_t* slots,
                                bool auto_recycle,
                                ComponentType component = ComponentType::FULL) = 0;
    virtual InsertPathResult insert_path(const uint64_t* hashes, int n, int start, int length,
                                         const int32_t* full_slots,
                                         const std::vector<ComponentRun>& runs,
                                         bool auto_recycle) = 0;
    virtual RemoteStatus recycle(const int32_t* slots, int num_slots,
                                 ComponentType component = ComponentType::FULL) = 0;
    // Drops a query(lock) pin; never replayed, so a lost reply may leave it.
    virtual RemoteStatus unlock(const uint64_t* hashes, int n, const PinRecord& pin) = 0;
    virtual DedupReply dedup_prefix(const uint64_t* hashes, int n,
                                    const DedupRequest& req = {}) = 0;
    virtual bool connected() = 0;
    // Starts connecting now instead of on the first op.
    virtual void connect() {}
};

// The gRPC server's TCP endpoint.
struct RemoteOpAddr {
    uint16_t port = 0;
};

class RemoteOpService;

// Backend lifecycle: rc_build_endpoint publishes the endpoint; call stop() before destruction.
class IRemoteOpServerTransport {
public:
    virtual ~IRemoteOpServerTransport() = default;

    // Serve `svc` until stop(); `svc` must outlive stop().
    virtual void start(RemoteOpService& svc) = 0;
    // Stop admitting ops and wait for in-flight handlers. Idempotent.
    virtual void stop() = 0;
};

} // namespace radixshmem
