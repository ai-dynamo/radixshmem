// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "shmradix/cluster_protocol.h"   // RemoteStatus

#include <cstddef>
#include <cstdint>
#include <vector>

namespace shmradix {

class IRemoteOpClientTransport {
public:
    virtual ~IRemoteOpClientTransport() = default;
    virtual RemoteStatus send_request(const void* req, size_t len,
                                      uint64_t req_id,
                                      std::vector<char>& reply_out) = 0;
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

} // namespace shmradix
