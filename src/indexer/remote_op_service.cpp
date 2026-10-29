// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Server half of the remote-op plane: RemoteOpService (the ops against our
// local tree) and RadixServer::rc_* (endpoint + serving lifecycle). Backends
// live under transport/; the client half (RemotePeer) in peer/remote_op.cpp.
#include "radix_server.h"

#ifndef RADIXSHMEM_NO_RDMA

#include "remote_op_service.hpp"
#include "local.h"                 // LocalTreeConnector (svc_tree_)
#include "radix_client.h"          // RadixClient (svc_client_)
#include "cluster_protocol.h"      // MAX_REMOTE_OP_BLOCKS
#include "bilateral_transport.h"   // IRemoteOpServerTransport
#include "cluster_manifest.h"      // REMOTE_OP_TRANSPORT_*
#ifdef RADIXSHMEM_REMOTE_OP_RDMA
#include "dc_bilateral.hpp"                 // DcServerTransport
#endif
#ifdef RADIXSHMEM_HAS_ZMQ
#include "zmq_transport.hpp"                // ZmqServerTransport
#endif
#ifdef RADIXSHMEM_HAS_GRPC
#include "grpc_remote_op.hpp"               // GrpcRemoteOpServer
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace radixshmem {

namespace {

int rc_resolve_lanes() {
    int lanes = 1;
    if (const char* e = ::getenv("RADIXSHMEM_RC_LANES")) {
        int v = std::atoi(e);
        if (v >= 1 && v <= 8) lanes = v;
    }
    return lanes;
}

bool valid_count(int n) { return n >= 0 && n <= MAX_REMOTE_OP_BLOCKS; }

// [start, start + length) inside [0, n).
bool valid_range(int start, int length, int n) {
    return start >= 0 && length >= 0 && start <= n - length;
}

}  // namespace

// ==================== RemoteOpService ====================

AllocateReply RemoteOpService::allocate(int n) {
    // Pure slot alloc (no hashes, no dedup); dedup happens in insert, whose
    // local insert consumes only the unmatched tail.
    AllocateReply out;
    if (!valid_count(n)) { out.status = RemoteStatus::BAD_REQUEST; return out; }
    if (n == 0) return out;
    std::lock_guard<std::mutex> g(insert_mu_);
    int got = tree_.allocate_slots(n, out.slots);
    if (got < n) {
        tree_.recycle_slots(out.slots.data(), got);
        out.slots.clear();
        out.error = InsertError::MEMPOOL_FULL;
    }
    return out;
}

QueryReply RemoteOpService::query(const uint64_t* hashes, int n, bool lock,
                                  bool update_meta) {
    QueryReply out;
    if (!valid_count(n) || (n > 0 && !hashes)) {
        out.status = RemoteStatus::BAD_REQUEST;
        return out;
    }
    TreeComponentQueryResult r = tree_.tree().query_components(
        hashes, n, COMPONENT_MASK_FULL, lock, update_meta);
    out.hit_length = (int)r.common_hit;
    out.slots      = std::move(r.full_slots);   // len == hit_length
    return out;
}

InsertResult RemoteOpService::insert(const uint64_t* hashes, int n, int start,
                                     int length, const int32_t* slots,
                                     bool auto_recycle) {
    InsertResult out;
    if (!valid_count(n) || !valid_range(start, length, n) ||
        (n > 0 && !hashes) || (length > 0 && !slots)) {
        out.error = InsertError::REMOTE_FAILED;
        out.remote_status = RemoteStatus::BAD_REQUEST;
        return out;
    }
    if (length == 0) return out;
    std::lock_guard<std::mutex> g(insert_mu_);
    // auto_recycle: the target recycles the surplus; else the initiator does.
    return client_.insert(hashes, n, slots, length, start, auto_recycle);
}

RemoteStatus RemoteOpService::recycle(const int32_t* slots, int num_slots) {
    if (!valid_count(num_slots) || (num_slots > 0 && !slots))
        return RemoteStatus::BAD_REQUEST;
    if (num_slots > 0) tree_.recycle_slots(slots, num_slots);
    return RemoteStatus::OK;
}

RemoteStatus RemoteOpService::unlock(const uint64_t* hashes, int n, int start,
                                     int length) {
    if (!valid_count(n) || !valid_range(start, length, n) ||
        (n > 0 && !hashes))
        return RemoteStatus::BAD_REQUEST;
    if (length == 0) return RemoteStatus::OK;
    // Rebuild the fully-inside record from the wire range and release
    // through the component pair.
    PinRecord pin;
    pin.full = {(uint32_t)start, (uint32_t)(start + length)};
    (void)tree_.tree().dec_ref(hashes, n, std::move(pin));
    return RemoteStatus::OK;
}

// ==================== RadixServer remote-op endpoint ====================

RemoteOpAddr RadixServer::rc_build_endpoint(uint32_t rot_auth,
                                            uint64_t dc_access_key,
                                            RdmaContext& ctx) {
    // Body of the build_remote_op callback.
    RemoteOpAddr addr{};
    if (rot_auth == REMOTE_OP_TRANSPORT_DC) {
#ifdef RADIXSHMEM_REMOTE_OP_RDMA
        int k_pool = 64;
        if (const char* e = ::getenv("RADIXSHMEM_RC_KPOOL")) {
            int v = std::atoi(e);
            if (v >= 1 && v <= 4096) k_pool = v;
        }
        auto t = std::make_unique<DcServerTransport>(
            ctx, dc_access_key, std::min(rc_resolve_lanes(), 4), k_pool);
        for (int i = 0; i < t->lanes() && i < 4; ++i)
            addr.dctns[i] = t->dct_num(i);
        rc_server_ = std::move(t);
        return addr;   // addr.port stays 0 — DC has no TCP endpoint
#else
        throw std::runtime_error(
            "remote-op: cluster authoritative transport is dc but this build "
            "has no RDMA control-plane backend");
#endif
    }
    if (rot_auth == REMOTE_OP_TRANSPORT_GRPC) {
#ifdef RADIXSHMEM_HAS_GRPC
        auto g = std::make_unique<GrpcRemoteOpServer>(cfg_.remote_op_listen_port);
        addr.port = g->port();   // actual bound port (throws on failure)
        rc_server_ = std::move(g);
        return addr;
#else
        throw std::runtime_error(
            "remote-op: cluster authoritative transport is grpc but gRPC is "
            "not compiled into this build");
#endif
    }
    if (rot_auth != REMOTE_OP_TRANSPORT_ZMQ) {
        throw std::runtime_error(
            "remote-op: cluster authoritative transport enum " +
            std::to_string(rot_auth) + " is not served by this build "
            "(removed xrc/rdma backend?) — redeploy rank 0 with "
            "remote_op_transport = grpc, dc or zmq");
    }
#ifdef RADIXSHMEM_HAS_ZMQ
    auto z = std::make_unique<ZmqServerTransport>(cfg_.remote_op_listen_port);
    addr.port = z->bind();   // actual bound port (throws on failure)
    rc_server_ = std::move(z);
    return addr;
#else
    throw std::runtime_error(
        "remote-op: cluster authoritative transport is zmq but ZMQ is not "
        "compiled into this build");
#endif
}

void RadixServer::rc_spawn() {
    if (!rc_server_) return;
    if (!svc_client_ || !svc_tree_) {
        throw std::runtime_error(
            "rc_spawn: enable_remote_insert/_query set but svc_* not built "
            "(need world_size>1 + successful svc RadixClient bootstrap)");
    }
    rc_service_ = std::make_unique<RemoteOpService>(*svc_tree_, *svc_client_,
                                                    svc_client_mu_);
    rc_server_->start(*rc_service_);
}

void RadixServer::rc_stop() {
    if (rc_server_) rc_server_->stop();
    rc_server_.reset();
    rc_service_.reset();
}

}  // namespace radixshmem

#endif  // !RADIXSHMEM_NO_RDMA
