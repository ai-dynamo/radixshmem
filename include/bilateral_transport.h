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
