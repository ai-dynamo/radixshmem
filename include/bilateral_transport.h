// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cluster_protocol.h"   // RemoteStatus

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace radixshmem {

struct AllocateReply;
struct QueryReply;
struct InsertResult;

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

// DC/ZMQ: the binary remote-op protocol over a request/reply byte transport.
// `op_mu` serializes every op (and the subclass dtor) of sessions sharing one
// transport stack; null gives this session a private mutex.
class LegacyRemoteOpClientTransport : public IRemoteOpClientTransport {
public:
    explicit LegacyRemoteOpClientTransport(
        std::shared_ptr<std::mutex> op_mu = nullptr);

    AllocateReply allocate(int n) override;
    QueryReply query(const uint64_t* hashes, int n, bool lock,
                     bool update_meta) override;
    InsertResult insert(const uint64_t* hashes, int n, int start, int length,
                        const int32_t* slots, bool auto_recycle) override;
    RemoteStatus recycle(const int32_t* slots, int num_slots) override;
    RemoteStatus unlock(const uint64_t* hashes, int n, int start,
                        int length) override;

protected:
    virtual RemoteStatus send_request(const void* req, size_t len,
                                      uint64_t req_id,
                                      std::vector<char>& reply_out) = 0;
    std::unique_lock<std::mutex> guard() const {
        return std::unique_lock<std::mutex>(*op_mu_);
    }

private:
    RemoteStatus send_ack_op(uint16_t op, const uint64_t* hashes, int n,
                             int start, int length, const int32_t* slots);

    std::shared_ptr<std::mutex> op_mu_;
    uint64_t next_req_id_ = 0;
};

// Reply-routing token from recv_request, handed back to send_reply.
struct ReplyContext {
    uint16_t lane = 0;
    uint8_t  gid[16] = {}; // requester's port GID → reply AH
    uint32_t dct_num = 0;  // requester's reply DCT
    uint64_t incarnation = 0;   // echoed in the reply ctrl header
    uint16_t lid = 0;           // requester's LID (IB only)
    uint8_t  link_layer = 0;    // IBV_LINK_LAYER_* (drives AH is_global)
};

// Pure recv/reply; endpoint creation/publication done by rc_build_endpoint.
class IRemoteOpServerTransport {
public:
    virtual ~IRemoteOpServerTransport() = default;

    virtual int lanes() const { return 1; }
    virtual int recv_request(std::vector<char>& buf, ReplyContext& ctx_out,
                             int timeout_ms, int lane = 0) = 0;

    virtual void send_reply(const ReplyContext& ctx,
                            const void* data, size_t len) = 0;
};

} // namespace radixshmem
