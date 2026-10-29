// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "radixshmem_config.h"   // feature macros (CMake-generated, build tree)

#ifdef RADIXSHMEM_HAS_ZMQ

#include "cluster_protocol.h"      // RemoteStatus
#include "remote_op_wire.hpp"   // WireRemoteOp{Client,Server}Transport

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace radixshmem {

class ZmqClientTransport : public WireRemoteOpClientTransport {
public:
    explicit ZmqClientTransport(const std::string& endpoint,
                                int recv_timeout_ms = 5000, int max_retry = 3);
    ~ZmqClientTransport() override;

    ZmqClientTransport(const ZmqClientTransport&) = delete;
    ZmqClientTransport& operator=(const ZmqClientTransport&) = delete;

protected:
    RemoteStatus send_request(const void* req, size_t len, uint64_t req_id,
                              std::vector<char>& reply_out) override;

private:
    void reconnect();    // close + rebuild the wedged REQ socket
    void close();        // tear down socket + ctx (idempotent; dtor/move)

    void*       zmq_ctx_  = nullptr;
    void*       req_sock_ = nullptr;
    std::string endpoint_;
    int         recv_timeout_ms_ = 5000;
    int         max_retry_       = 3;
};

class ZmqServerTransport : public WireRemoteOpServerTransport {
public:
    explicit ZmqServerTransport(uint16_t listen_port) : listen_port_(listen_port) {}
    ~ZmqServerTransport() override;

    ZmqServerTransport(const ZmqServerTransport&) = delete;
    ZmqServerTransport& operator=(const ZmqServerTransport&) = delete;

    // Bind the REP socket, return the actual port (OS-assigned when the
    // requested port is 0). Not part of the transport interface — called on
    // the concrete type by rc_build_endpoint, which publishes the port.
    uint16_t bind();

protected:
    int recv_request(std::vector<char>& buf, ReplyContext& ctx_out,
                     int timeout_ms, int lane = 0) override;

    void send_reply(const ReplyContext& ctx, const void* data,
                    size_t len) override;

private:
    void rebuild_rep();   // recover a REP socket wedged by a failed send

    void*    ctx_      = nullptr;
    void*    rep_sock_ = nullptr;
    uint16_t listen_port_ = 0;
    uint16_t bound_port_  = 0;   // actual port from bind() (rebuild rebinds it)
    int      cur_timeo_   = 50;
};

}  // namespace radixshmem

#endif  // RADIXSHMEM_HAS_ZMQ
