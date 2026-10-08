// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// XRC SEND-QP bring-up: TCP-dial a holder's XRC listener, exchange the
// hello/reply handshake, and drive the SEND QP to RTS.

#pragma once

#include "cluster_protocol.h"
#include "rdma_transport.hpp"

#include <arpa/inet.h>
#include <cstdio>
#include <cstring>
#include <memory>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

namespace radixshmem {

inline bool xrc_send_all(int fd, const void* data, size_t len) {
    const char* p = static_cast<const char*>(data);
    while (len > 0) {
        ssize_t n = ::send(fd, p, len, MSG_NOSIGNAL);
        if (n <= 0) return false;
        p += n; len -= (size_t)n;
    }
    return true;
}

inline bool xrc_recv_all(int fd, void* data, size_t len) {
    char* p = static_cast<char*>(data);
    while (len > 0) {
        ssize_t n = ::recv(fd, p, len, 0);
        if (n <= 0) return false;
        p += n; len -= (size_t)n;
    }
    return true;
}

inline int xrc_tcp_connect(const char* host, uint16_t port, int timeout_sec) {
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%u", port);
    if (getaddrinfo(host, port_str, &hints, &res) != 0) return -1;

    int fd = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) { freeaddrinfo(res); return -1; }

    timeval tv;
    tv.tv_sec = timeout_sec;
    tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    if (::connect(fd, res->ai_addr, res->ai_addrlen) < 0) {
        ::close(fd);
        freeaddrinfo(res);
        return -1;
    }
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    freeaddrinfo(res);
    return fd;
}

inline std::unique_ptr<XrcSendQp> xrc_connect(RdmaContext& ctx, ibv_cq* cq,
                                              const char* host, uint16_t port,
                                              int my_rank, int timeout_sec) {
    auto ep = std::make_unique<XrcSendQp>(ctx, cq);

    int conn = xrc_tcp_connect(host, port, timeout_sec);
    if (conn < 0) {
        fprintf(stderr, "[xrc_connect rank %d] TCP connect to %s:%u failed: %s\n",
                my_rank, host, port, strerror(errno));
        return nullptr;
    }

    ibv_port_attr pa{};
    if (ibv_query_port(ctx.ctx(), ctx.port(), &pa) != 0) {
        ::close(conn);
        return nullptr;
    }

    ClusterHelloMsg hello{};
    hello.magic              = CLUSTER_HELLO_MAGIC;
    hello.version            = CLUSTER_PROTOCOL_VERSION;
    hello.client_rank        = my_rank;
    hello.client_send_qp_num = ep->qp_num();
    hello.link_layer         = (uint8_t)pa.link_layer;
    hello.lid                = pa.lid;
    hello.gid_index          = ctx.gid_index();
    hello.gid                = ctx.gid();
    if (!xrc_send_all(conn, &hello, sizeof(hello))) { ::close(conn); return nullptr; }

    ClusterReplyMsg reply{};
    if (!xrc_recv_all(conn, &reply, sizeof(reply))) { ::close(conn); return nullptr; }
    ::close(conn);

    if (reply.magic != CLUSTER_REPLY_MAGIC ||
        reply.version != CLUSTER_PROTOCOL_VERSION) {
        return nullptr;
    }

    XrcPeerInfo xpeer{};
    xpeer.recv_qp_num = reply.server_recv_qp_num;
    xpeer.remote_srqn = reply.server_srq_num;
    xpeer.gid         = reply.gid;
    xpeer.gid_index   = reply.gid_index;
    xpeer.lid         = reply.lid;
    xpeer.link_layer  = reply.link_layer;
    ep->connect(xpeer);
    return ep;
}

} // namespace radixshmem
