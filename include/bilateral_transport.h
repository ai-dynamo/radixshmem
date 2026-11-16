// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "component_types.h"   // QueryReply, RemoteQuery, PinRecord
#include "types.h"             // AllocateReply, InsertResult, RemoteStatus

#include <cstdint>
#include <functional>
#include <vector>

namespace radixshmem {

// One end of an XRC connection: the caller's SEND QP, or the RECV QP and SRQ the
// target paired with it, and the address of that end's HCA port.
struct XrcConnInfo {
    int32_t  rank = -1;
    uint32_t qp_num = 0;
    uint32_t srq_num = 0;   // target end only
    uint32_t link_layer = 0;
    uint32_t lid = 0;
    int32_t  gid_index = 0;
    uint8_t  gid[16] = {};
};

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
    // RDMA connection setup: the target pairs a RECV QP with the caller's XRC SEND
    // QP `mine` and returns its end. UNKNOWN_OP if the target has no RDMA side.
    virtual RemoteStatus xrc_connect(const XrcConnInfo& mine, XrcConnInfo& theirs) = 0;
    virtual bool connected() = 0;
    // Starts connecting now instead of on the first op.
    virtual void connect() {}
};

// The gRPC server's TCP endpoint.
struct RemoteOpAddr {
    uint16_t port = 0;
};

class RemoteOpService;

// The server's side of XrcConnect; throws on failure.
using XrcAcceptor = std::function<XrcConnInfo(const XrcConnInfo& caller)>;

// Built in round 2 of the rendezvous, when it starts serving XrcConnect; the other
// ops wait for start(). Call stop() before destruction.
class IRemoteOpServerTransport {
public:
    virtual ~IRemoteOpServerTransport() = default;

    // Serve `svc` until stop(); `svc` must outlive stop().
    virtual void start(RemoteOpService& svc) = 0;
    // Stop admitting ops and wait for in-flight handlers. Idempotent.
    virtual void stop() = 0;
};

} // namespace radixshmem
