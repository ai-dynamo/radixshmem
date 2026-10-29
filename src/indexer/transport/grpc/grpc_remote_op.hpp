// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "radixshmem_config.h"   // feature macros (CMake-generated, build tree)

#ifdef RADIXSHMEM_HAS_GRPC

#include "bilateral_transport.h"   // IRemoteOp{Client,Server}Transport

#include <cstdint>
#include <memory>
#include <string>

namespace radixshmem {

// Default per-op deadline (RADIXSHMEM_REMOTE_OP_TIMEOUT_MS, else 5000 ms).
int grpc_remote_op_timeout_ms();

// Concurrent per-target channel, no retries: TIMEOUT/UNAVAILABLE does not imply the op was skipped.
class GrpcRemoteOpClientTransport : public IRemoteOpClientTransport {
public:
    // `target` is "host:port"; the channel connects lazily.
    GrpcRemoteOpClientTransport(const std::string& target, int timeout_ms);
    ~GrpcRemoteOpClientTransport() override;

    GrpcRemoteOpClientTransport(const GrpcRemoteOpClientTransport&) = delete;
    GrpcRemoteOpClientTransport& operator=(const GrpcRemoteOpClientTransport&) = delete;

    AllocateReply allocate(int n) override;
    QueryReply query(const uint64_t* hashes, int n, bool lock,
                     bool update_meta) override;
    InsertResult insert(const uint64_t* hashes, int n, int start, int length,
                        const int32_t* slots, bool auto_recycle) override;
    RemoteStatus recycle(const int32_t* slots, int num_slots) override;
    RemoteStatus unlock(const uint64_t* hashes, int n, int start,
                        int length) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Binds on construction; UNAVAILABLE until start(svc), RESOURCE_EXHAUSTED at max_inflight; stop() drains handlers.
class GrpcRemoteOpServer : public IRemoteOpServerTransport {
public:
    // listen_port 0 = OS-assigned; max_inflight <= 0 = 32; bind failure throws.
    explicit GrpcRemoteOpServer(uint16_t listen_port, int max_inflight = 0);
    ~GrpcRemoteOpServer() override;

    GrpcRemoteOpServer(const GrpcRemoteOpServer&) = delete;
    GrpcRemoteOpServer& operator=(const GrpcRemoteOpServer&) = delete;

    uint16_t port() const;   // the bound port

    void start(RemoteOpService& svc) override;
    void stop() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace radixshmem

#endif  // RADIXSHMEM_HAS_GRPC
