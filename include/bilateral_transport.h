// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.h"   // AllocateReply, QueryReply, InsertResult, RemoteStatus

#include <cstdint>

namespace radixshmem {

// One typed method per remote op; each backend owns its own wire encoding.
class IRemoteOpClientTransport {
public:
    virtual ~IRemoteOpClientTransport() = default;
    virtual AllocateReply allocate(int n) = 0;
    virtual QueryReply query(const uint64_t* hashes, int n, bool lock,
                             bool update_meta) = 0;
    virtual InsertResult insert(const uint64_t* hashes, int n, int start,
                                int length, const int32_t* slots,
                                bool auto_recycle) = 0;
    virtual RemoteStatus recycle(const int32_t* slots, int num_slots) = 0;
    virtual RemoteStatus unlock(const uint64_t* hashes, int n, int start,
                                int length) = 0;
};

// What peers need to reach a server: its TCP port (grpc/zmq) or lane DCT numbers (dc).
struct RemoteOpAddr {
    uint16_t port = 0;        // grpc/zmq: actual bound TCP port (0 = dc / off)
    uint32_t dctns[4] = {};   // dc: lane DCT numbers, lane 0 first, 0-padded
};

class RemoteOpService;

// Serving lifecycle of one backend; endpoint creation/publication is done by
// rc_build_endpoint. stop() must run before destruction.
class IRemoteOpServerTransport {
public:
    virtual ~IRemoteOpServerTransport() = default;

    // Serve `svc` until stop(); `svc` must outlive stop().
    virtual void start(RemoteOpService& svc) = 0;
    // Stop admitting ops and wait for in-flight handlers. Idempotent.
    virtual void stop() = 0;
};

} // namespace radixshmem
