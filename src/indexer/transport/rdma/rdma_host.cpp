// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "rdma_host.hpp"

#include "cluster_protocol.h"
#include "xrc_connect.hpp"   // xrc_send_all / xrc_recv_all

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdexcept>
#include <sys/socket.h>
#include <unistd.h>

namespace radixshmem {

namespace {

// Bind+listen on `port` (0 = OS-assigned); returns fd, writes bound port to *out_port.
int tcp_listen(uint16_t port, uint16_t* out_port) {
    constexpr int kMaxAttempts = 12;
    constexpr auto kBackoff = std::chrono::milliseconds(500);
    int last_errno = 0;
    for (int attempt = 0; attempt < kMaxAttempts; attempt++) {
        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) { last_errno = errno; continue; }
        int opt = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port);
        if (::bind(fd, (sockaddr*)&addr, sizeof(addr)) == 0 &&
            ::listen(fd, 64) == 0) {
            socklen_t alen = sizeof(addr);
            ::getsockname(fd, (sockaddr*)&addr, &alen);
            if (out_port) *out_port = ntohs(addr.sin_port);
            return fd;
        }
        last_errno = errno;
        ::close(fd);
        std::this_thread::sleep_for(kBackoff);
    }
    fprintf(stderr, "[RadixServer] tcp_listen(:%u) gave up: %s\n",
            port, strerror(last_errno));
    errno = last_errno;
    return -1;
}

} // namespace

RdmaHost::RdmaHost(const RdmaHostConfig& cfg) : cfg_(cfg) {
    ctx_  = std::make_unique<RdmaContext>(cfg_.dev.empty() ? nullptr : cfg_.dev.c_str(), 1,
                                          cfg_.gid_idx);
    ctx_->set_qos(cfg_.sl, cfg_.tc);   // the cluster's QoS before any QP connects
    xrcd_ = std::make_unique<RdmaXrcd>(*ctx_);
    cq_.reset(ibv_create_cq(ctx_->ctx(), 256, nullptr, nullptr, 0));
    if (!cq_) throw std::runtime_error("RdmaHost: ibv_create_cq failed");
    srq_  = std::make_unique<XrcSrq>(*ctx_, *xrcd_, cq_.get());
    if (cfg_.dc) {
        dct_ = std::make_unique<DcTarget>(*ctx_, cfg_.dc_access_key, /*srq_max_wr=*/1);
        dct_num_ = dct_->open_dct();
        fprintf(stderr, "[RadixServer rank %d] DC target up: dctn=%u\n", cfg_.rank, dct_num_);
    }
    listen_fd_ = tcp_listen(cfg_.port, &port_);
    if (listen_fd_ < 0) throw std::runtime_error("RdmaHost: no XRC listen port");
    listener_ = std::thread(&RdmaHost::accept_loop, this);
    fprintf(stderr, "[RadixServer rank %d] XRC accept loop on :%u\n", cfg_.rank, port_);
}

RdmaHost::~RdmaHost() {
    stopping_.store(true);
    if (listen_fd_ >= 0) {
        ::shutdown(listen_fd_, SHUT_RDWR);
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
    if (listener_.joinable()) listener_.join();
    std::lock_guard<std::mutex> g(clients_mu_);
    clients_.clear();   // the RECV QPs, before the SRQ, CQ and context they use
}

void RdmaHost::expose_tree(void* base, size_t size) {
    // READ + ATOMIC only (no REMOTE_WRITE); RHT writes use the shard's own MR.
    tree_mr_ = std::make_unique<RdmaMemoryRegion>(
        ctx_->pd(), base, size,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
}

void RdmaHost::publish_to(PeerEntry& e) const {
    e.xrc_listen_port = port_;
    if (tree_mr_) {
        e.shm_base_addr = (uint64_t)tree_mr_->addr();
        e.shm_rkey      = tree_mr_->rkey();
    }
    memcpy(e.gid_raw, &ctx_->gid(), 16);
    e.gid_index = ctx_->gid_index();
    ibv_port_attr pa{};
    if (ibv_query_port(ctx_->ctx(), ctx_->port(), &pa) == 0) {
        e.lid        = pa.lid;
        e.link_layer = (uint8_t)pa.link_layer;
    }
    e.srq_num = srq_->srq_num();
    e.dct_num = dct_num_;
}

// ==================== Listener / hello-reply ====================

void RdmaHost::accept_loop() {
    while (!stopping_.load()) {
        int conn = ::accept(listen_fd_, nullptr, nullptr);
        if (conn < 0) {
            if (stopping_.load()) break;
            if (errno == EINTR) continue;
            fprintf(stderr, "[RadixServer] accept failed: %s\n", strerror(errno));
            break;
        }
        int one = 1;
        setsockopt(conn, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        try {
            handle_client(conn);
        } catch (const std::exception& e) {
            fprintf(stderr, "[RadixServer] handle_client: %s\n", e.what());
        }
        ::close(conn);
    }
}

void RdmaHost::handle_client(int conn_fd) {
    ClusterHelloMsg hello{};
    if (!xrc_recv_all(conn_fd, &hello, sizeof(hello)))
        throw std::runtime_error("hello recv short");
    if (hello.magic != CLUSTER_HELLO_MAGIC)
        throw std::runtime_error("bad hello magic");
    if (hello.version != CLUSTER_PROTOCOL_VERSION)
        throw std::runtime_error("hello version mismatch");

    auto recv_qp = std::make_unique<XrcRecvQp>(*ctx_, *xrcd_, *srq_, cq_.get());

    XrcPeerInfo peer{};
    peer.recv_qp_num = 0;
    peer.remote_srqn = 0;
    peer.gid         = hello.gid;
    peer.gid_index   = hello.gid_index;
    peer.lid         = hello.lid;
    peer.link_layer  = hello.link_layer;
    recv_qp->connect(hello.client_send_qp_num, peer);

    ibv_port_attr pa{};
    if (ibv_query_port(ctx_->ctx(), ctx_->port(), &pa) != 0)
        throw std::runtime_error("ibv_query_port failed");

    ClusterReplyMsg reply{};
    reply.magic              = CLUSTER_REPLY_MAGIC;
    reply.version            = CLUSTER_PROTOCOL_VERSION;
    reply.server_rank        = cfg_.rank;
    reply.server_recv_qp_num = recv_qp->qp_num();
    reply.server_srq_num     = srq_->srq_num();
    reply.link_layer         = (uint8_t)pa.link_layer;
    reply.lid                = pa.lid;
    reply.gid_index          = ctx_->gid_index();
    reply.gid                = ctx_->gid();
    if (!xrc_send_all(conn_fd, &reply, sizeof(reply)))
        throw std::runtime_error("reply send short");

    std::lock_guard<std::mutex> g(clients_mu_);
    clients_.push_back(std::move(recv_qp));
}

} // namespace radixshmem
