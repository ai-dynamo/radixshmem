// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "zmq_transport.hpp"

#ifdef SHMRADIX_HAS_ZMQ

#include "shmradix/radix_client.h"   // RemotePeer decl (convenience ctor below)

#include <zmq.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace shmradix {

namespace {
// Upper bound on a single reply message — derived from the protocol
// constants (cluster_protocol.h), never a restated literal.
constexpr uint32_t kMaxReplyBytes = REMOTE_OP_MAX_REPLY_BYTES;

void apply_sockopts(void* sock, int recv_timeout_ms) {
    zmq_setsockopt(sock, ZMQ_RCVTIMEO, &recv_timeout_ms, sizeof(recv_timeout_ms));
    int linger = 0;  // don't block on close/term with queued messages
    zmq_setsockopt(sock, ZMQ_LINGER, &linger, sizeof(linger));
}

// Parse the OS-assigned port out of "tcp://host:port" from ZMQ_LAST_ENDPOINT.
uint16_t parse_bound_port(void* sock) {
    char ep[256];
    size_t len = sizeof(ep);
    if (zmq_getsockopt(sock, ZMQ_LAST_ENDPOINT, ep, &len) != 0) return 0;
    if (len < sizeof(ep)) ep[len] = '\0'; else ep[sizeof(ep) - 1] = '\0';
    const char* colon = std::strrchr(ep, ':');
    if (!colon) return 0;
    return (uint16_t)atoi(colon + 1);
}
}  // namespace

// ==================== ZmqClientTransport ====================

ZmqClientTransport::ZmqClientTransport(const std::string& endpoint,
                                       int recv_timeout_ms, int max_retry)
    : endpoint_(endpoint),
      recv_timeout_ms_(recv_timeout_ms),
      max_retry_(max_retry) {
    zmq_ctx_  = zmq_ctx_new();
    req_sock_ = zmq_socket(zmq_ctx_, ZMQ_REQ);
    apply_sockopts(req_sock_, recv_timeout_ms_);
    // Lazy connect: no TCP handshake here; the first send triggers it.
    zmq_connect(req_sock_, endpoint_.c_str());
}

void ZmqClientTransport::close() {
    if (req_sock_) { zmq_close(req_sock_); req_sock_ = nullptr; }
    if (zmq_ctx_)  { zmq_ctx_term(zmq_ctx_); zmq_ctx_ = nullptr; }
}

ZmqClientTransport::~ZmqClientTransport() { close(); }

void ZmqClientTransport::reconnect() {
    // A timed-out REQ is stuck; rebuild and reconnect the socket to recover.
    zmq_close(req_sock_);
    req_sock_ = zmq_socket(zmq_ctx_, ZMQ_REQ);
    apply_sockopts(req_sock_, recv_timeout_ms_);
    zmq_connect(req_sock_, endpoint_.c_str());
}

RemoteStatus ZmqClientTransport::send_request(const void* req, size_t req_len,
                                              uint64_t req_id,
                                              std::vector<char>& reply_out) {
    // The caller has already stamped this request's req_id into the header.
    for (int attempt = 0; attempt < max_retry_; ++attempt) {
        if (zmq_send(req_sock_, req, req_len, 0) < 0) {
            reconnect();
            continue;
        }
        char buf[kMaxReplyBytes];
        int got = zmq_recv(req_sock_, buf, sizeof(buf), 0);
        if (got >= (int)sizeof(RemoteOpReplyHeader)) {
            auto* h = reinterpret_cast<RemoteOpReplyHeader*>(buf);
            if (h->magic == REMOTE_OP_REPLY_MAGIC &&
                h->version == REMOTE_OP_PROTOCOL_VERSION &&
                h->req_id == req_id) {
                reply_out.assign(buf, buf + got);
                return RemoteStatus::OK;
            }
        }
        // Timeout (got < 0) or bad/late packet → rebuild the wedged socket.
        reconnect();
    }
    return RemoteStatus::TIMEOUT;
}

// ZMQ convenience ctor. The transport-agnostic RemotePeer body lives in
// remote_op_endpoint.cpp.
RemotePeer::RemotePeer(const std::string& endpoint,
                               int recv_timeout_ms, int max_retry)
    : transport_(std::make_unique<ZmqClientTransport>(endpoint, recv_timeout_ms,
                                                      max_retry)),
      endpoint_(endpoint) {}

// ==================== ZmqServerTransport ====================

ZmqServerTransport::~ZmqServerTransport() {
    if (rep_sock_) { zmq_close(rep_sock_); rep_sock_ = nullptr; }
    if (ctx_)      { zmq_ctx_term(ctx_);   ctx_ = nullptr; }
}

uint16_t ZmqServerTransport::bind() {
    ctx_      = zmq_ctx_new();
    rep_sock_ = zmq_socket(ctx_, ZMQ_REP);
    cur_timeo_ = 50;   // poll cadence so rc_loop can observe rc_running_
    zmq_setsockopt(rep_sock_, ZMQ_RCVTIMEO, &cur_timeo_, sizeof(cur_timeo_));
    int linger = 0;
    zmq_setsockopt(rep_sock_, ZMQ_LINGER, &linger, sizeof(linger));
    std::string bind_ep = "tcp://*:" + std::to_string(listen_port_);
    if (zmq_bind(rep_sock_, bind_ep.c_str()) != 0) {
        zmq_close(rep_sock_); rep_sock_ = nullptr;
        zmq_ctx_term(ctx_);   ctx_ = nullptr;
        throw std::runtime_error(
            "ZmqServerTransport::bind: zmq_bind failed on " + bind_ep);
    }
    bound_port_ = parse_bound_port(rep_sock_);   // rebuild_rep rebinds this
    return bound_port_;
}


void ZmqServerTransport::rebuild_rep() {
    zmq_close(rep_sock_);
    rep_sock_ = zmq_socket(ctx_, ZMQ_REP);
    zmq_setsockopt(rep_sock_, ZMQ_RCVTIMEO, &cur_timeo_, sizeof(cur_timeo_));
    int linger = 0;
    zmq_setsockopt(rep_sock_, ZMQ_LINGER, &linger, sizeof(linger));
    std::string bind_ep = "tcp://*:" + std::to_string(bound_port_);
    if (zmq_bind(rep_sock_, bind_ep.c_str()) != 0)
        throw std::runtime_error(
            "ZmqServerTransport::rebuild_rep: rebind failed on " + bind_ep);
}

int ZmqServerTransport::recv_request(std::vector<char>& buf,
                                     ReplyContext& ctx_out,
                                     int timeout_ms, int /*lane*/) {
    ctx_out = ReplyContext{};   // ZMQ REP self-remembers the peer; ctx unused
                                // (single-lane backend: ctx.lane stays 0)
    // Honor the caller's timeout (only touch the sockopt when it changes).
    if (timeout_ms != cur_timeo_) {
        cur_timeo_ = timeout_ms;
        zmq_setsockopt(rep_sock_, ZMQ_RCVTIMEO, &cur_timeo_, sizeof(cur_timeo_));
    }
    return zmq_recv(rep_sock_, buf.data(), buf.size(), 0);
}

void ZmqServerTransport::send_reply(const ReplyContext& /*ctx*/,
                                    const void* data, size_t len) {
    for (;;) {
        if (zmq_send(rep_sock_, data, len, 0) >= 0) return;
        if (errno == EINTR) continue;   // interrupted before sending; FSM
                                        // still expects this send — retry
        fprintf(stderr, "[ZmqServerTransport] send_reply failed (%s) — "
                "rebuilding REP socket\n", zmq_strerror(errno));
        rebuild_rep();                  // throws on rebind failure
        return;                         // reply lost; peer retries
    }
}

}  // namespace shmradix

#endif  // SHMRADIX_HAS_ZMQ
