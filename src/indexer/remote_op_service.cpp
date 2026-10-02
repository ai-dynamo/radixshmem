// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Server half of the remote-op plane: RadixServer::rc_* endpoint + op handlers,
// the pollers that serve a peer's insert/query/release against our local tree.
// Client half (RemotePeer) lives in peer/remote_op.cpp.
#include "shmradix/radix_server.h"

#ifndef SHMRADIX_NO_RDMA

#include "shmradix/local.h"                 // LocalTreeConnector (svc_tree_)
#include "shmradix/radix_client.h"          // RadixClient (svc_client_)
#include "shmradix/cluster_protocol.h"      // remote-op wire structs
#include "shmradix/bilateral_transport.h"   // IRemoteOpServerTransport
#include "shmradix/cluster_manifest.h"      // REMOTE_OP_TRANSPORT_*
#ifdef SHMRADIX_REMOTE_OP_RDMA
#include "dc_bilateral.hpp"                 // DcServerTransport
#endif
#ifdef SHMRADIX_HAS_ZMQ
#include "zmq_transport.hpp"                // ZmqServerTransport (fallback)
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace shmradix {

namespace {

int rc_resolve_lanes() {
    int lanes = 1;
    if (const char* e = ::getenv("SHMRADIX_RC_LANES")) {
        int v = std::atoi(e);
        if (v >= 1 && v <= 8) lanes = v;
    }
    return lanes;
}

// Header-only reply carrying a request-level status (BAD_REQUEST etc.).
std::vector<char> make_status_reply(const RemoteOpReqHeader& req, RemoteStatus st) {
    std::vector<char> buf(sizeof(RemoteOpReplyHeader));
    auto* r = reinterpret_cast<RemoteOpReplyHeader*>(buf.data());
    *r = {};
    r->magic   = REMOTE_OP_REPLY_MAGIC;
    r->version = REMOTE_OP_PROTOCOL_VERSION;
    r->op      = req.op;
    r->status  = static_cast<uint16_t>(st);
    r->req_id  = req.req_id;
    return buf;
}

}  // namespace

RemoteOpAddr RadixServer::rc_build_endpoint(uint32_t rot_auth,
                                            uint64_t dc_access_key,
                                            RdmaContext& ctx) {
    // Body of the build_remote_op callback.
    RemoteOpAddr addr{};
    if (rot_auth == REMOTE_OP_TRANSPORT_DC) {
#ifdef SHMRADIX_REMOTE_OP_RDMA
        int k_pool = 64;
        if (const char* e = ::getenv("SHMRADIX_RC_KPOOL")) {
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
    if (rot_auth != REMOTE_OP_TRANSPORT_ZMQ) {
        throw std::runtime_error(
            "remote-op: cluster authoritative transport enum " +
            std::to_string(rot_auth) + " is not served by this build "
            "(removed xrc/rdma backend?) — redeploy rank 0 with "
            "remote_op_transport = dc or zmq");
    }
#ifdef SHMRADIX_HAS_ZMQ
    auto z = std::make_unique<ZmqServerTransport>(cfg_.zmq_listen_port);
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
    int lanes = rc_resolve_lanes();
    if (lanes > rc_server_->lanes()) {
        fprintf(stderr, "[RadixServer] rc_lanes=%d exceeds backend lanes()=%d "
                "— capping\n", lanes, rc_server_->lanes());
        lanes = rc_server_->lanes();
    }
    // Every poller dispatches inline
    rc_running_.store(true);
    rc_threads_.reserve(lanes);
    for (int i = 0; i < lanes; ++i)
        rc_threads_.emplace_back(&RadixServer::rc_loop, this, i);   // pollers
}

void RadixServer::rc_stop() {
    if (rc_running_.exchange(false)) {
        // The pollers exit on rc_running_ == false; nothing else to wake.
        for (auto& t : rc_threads_)
            if (t.joinable()) t.join();
        rc_threads_.clear();
    }
    rc_server_.reset();
}

std::vector<char> RadixServer::handle_insert(const RemoteOpReqHeader& req,
                                             const uint64_t* /*hashes*/) {
    // ALLOCATE: pure slot alloc of req.n (no hashes, no dedup). Dedup happens
    // in the phase-2 INSERT (local insert consumes only the unmatched tail).
    std::vector<int32_t> slots;
    uint16_t error = static_cast<uint16_t>(InsertError::OK);
    if (req.n > 0) {
        std::lock_guard<std::mutex> g(svc_client_mu_);
        int got = svc_tree_->allocate_slots(req.n, slots);
        if (got < req.n) {
            svc_tree_->recycle_slots(slots.data(), got);
            slots.clear();
            error = static_cast<uint16_t>(InsertError::MEMPOOL_FULL);
        }
    }
    int matched = 0;   // no server-side dedup
    int num_slots = (int)slots.size();
    std::vector<char> buf(sizeof(RemoteOpReplyHeader) +
                          (size_t)num_slots * sizeof(int32_t));
    auto* rep = reinterpret_cast<RemoteOpReplyHeader*>(buf.data());
    *rep = {};
    rep->magic   = REMOTE_OP_REPLY_MAGIC;
    rep->version = REMOTE_OP_PROTOCOL_VERSION;
    rep->op      = REMOTE_OP_INSERT_RESERVE;
    rep->status  = static_cast<uint16_t>(RemoteStatus::OK);
    rep->error   = error;
    rep->req_id  = req.req_id;
    rep->u.reserve.matched_prefix = matched;
    rep->u.reserve.inserted_count = num_slots;
    rep->u.reserve.num_slots      = num_slots;
    if (num_slots > 0)
        std::memcpy(rep + 1, slots.data(), (size_t)num_slots * sizeof(int32_t));
    return buf;
}

std::vector<char> RadixServer::handle_finalize_like(const RemoteOpReqHeader& req,
                                                    const uint64_t* hashes) {
    const auto& f = req.u.insert;
    const int32_t* slots = reinterpret_cast<const int32_t*>(hashes + req.n);
    uint16_t error = 0;
    std::vector<int32_t> unused;   // INSERT: slots this publish did not consume
    if (f.length > 0) {
        if (req.op == REMOTE_OP_INSERT) {
            std::lock_guard<std::mutex> g(svc_client_mu_);
            // Recycle by flag: unset → server recycles; set → initiator recycles.
            bool ar = !(req.flags & REMOTE_OP_FLAG_NO_AUTO_RECYCLE);
            InsertResult r = svc_client_->insert(hashes, req.n, slots, f.length,
                                                 f.start, /*auto_recycle=*/ar);
            error  = static_cast<uint16_t>(r.error);
            unused = std::move(r.unused_slots);
        } else if (req.op == REMOTE_OP_INSERT_RELEASE) {
            svc_tree_->recycle_slots(slots, f.length);
        } else {   // REMOTE_OP_QUERY_RELEASE
            // Drop the pin a query(lock=true) placed: rebuild the fully-inside
            // record from the wire range and release through the component pair.
            PinRecord pin;
            pin.full = {(uint32_t)f.start, (uint32_t)(f.start + f.length)};
            (void)svc_tree_->tree().dec_ref(hashes, req.n, std::move(pin));
        }
    }
    int num_unused = (int)unused.size();   // 0 for *_RELEASE
    std::vector<char> buf(sizeof(RemoteOpReplyHeader) +
                          (size_t)num_unused * sizeof(int32_t));
    auto* rep = reinterpret_cast<RemoteOpReplyHeader*>(buf.data());
    *rep = {};
    rep->magic   = REMOTE_OP_REPLY_MAGIC;
    rep->version = REMOTE_OP_PROTOCOL_VERSION;
    rep->op      = req.op;   // echo (INSERT or *_RELEASE), not hardcoded
    rep->status  = static_cast<uint16_t>(RemoteStatus::OK);
    rep->error   = error;
    rep->req_id  = req.req_id;
    rep->u.insert.num_unused = num_unused;
    if (num_unused > 0)
        std::memcpy(rep + 1, unused.data(),
                    (size_t)num_unused * sizeof(int32_t));
    return buf;
}

std::vector<char> RadixServer::handle_query(const RemoteOpReqHeader& req,
                                            const uint64_t* hashes) {
    const auto& q = req.u.query;   // only legal after op == REMOTE_OP_QUERY
    TreeComponentQueryResult r = svc_tree_->tree().query_components(
        hashes, req.n, COMPONENT_MASK_FULL,
        /*lock=*/q.lock != 0,
        /*update_meta=*/q.update_meta != 0);
    int num_slots = (int)r.full_slots.size();
    std::vector<char> buf(sizeof(RemoteOpReplyHeader) +
                          (size_t)num_slots * sizeof(int32_t));
    auto* rep = reinterpret_cast<RemoteOpReplyHeader*>(buf.data());
    *rep = {};
    rep->magic   = REMOTE_OP_REPLY_MAGIC;
    rep->version = REMOTE_OP_PROTOCOL_VERSION;
    rep->op      = REMOTE_OP_QUERY;
    rep->status  = static_cast<uint16_t>(RemoteStatus::OK);
    rep->error   = 0;          // query never yields InsertError
    rep->req_id  = req.req_id;
    rep->u.query.hit_length = (int)r.common_hit;
    rep->u.query.num_slots  = num_slots;   // == hit_length
    if (num_slots > 0)
        std::memcpy(rep + 1, r.full_slots.data(),
                    (size_t)num_slots * sizeof(int32_t));
    return buf;
}

// Poller for one lane — the SOLE thread that polls that lane of rc_server_.
void RadixServer::rc_loop(int lane) {
    std::vector<char> rbuf(REMOTE_OP_MAX_REQ_BYTES);

    int idle = 0;                 // consecutive no-progress iterations
    constexpr int kSpinBudget = 2048;  // spin ~this many idle iters before sleeping

    while (rc_running_.load(std::memory_order_relaxed)) {
        bool progress = false;
        try {
            ReplyContext ctx;
            int got = rc_server_->recv_request(rbuf, ctx, /*non-blocking*/0, lane);
            if (got >= 0) {
                progress = true;
                if (got < (int)sizeof(RemoteOpReqHeader)) {
                    RemoteOpReqHeader dummy{};
                    auto reply = make_status_reply(dummy, RemoteStatus::BAD_REQUEST);
                    rc_server_->send_reply(ctx, reply.data(), reply.size());
                } else {
                    auto* h = reinterpret_cast<RemoteOpReqHeader*>(rbuf.data());
                    bool ok = h->magic == REMOTE_OP_REQ_MAGIC &&
                              h->version == REMOTE_OP_PROTOCOL_VERSION &&
                              h->n >= 0 && h->n <= MAX_REMOTE_OP_BLOCKS;
                    if (ok) {
                        // RESERVE carries no hashes (h->n is the alloc count);
                        // every other op is preceded by n hashes.
                        size_t want = sizeof(RemoteOpReqHeader) +
                            (h->op == REMOTE_OP_INSERT_RESERVE
                                 ? 0
                                 : (size_t)h->n * sizeof(uint64_t));
                        // INSERT / INSERT_RELEASE carry the phase-1 slots
                        // behind the hashes (u.insert.length int32s).
                        if (h->op == REMOTE_OP_INSERT ||
                            h->op == REMOTE_OP_INSERT_RELEASE) {
                            int ns = h->u.insert.length;
                            ok = ns >= 0 &&
                                 (h->op == REMOTE_OP_INSERT
                                      ? ns <= h->n
                                      : ns <= MAX_REMOTE_OP_BLOCKS);
                            want += (size_t)(ns > 0 ? ns : 0) * sizeof(int32_t);
                        }
                        ok = ok && got == (int)want;
                    }
                    if (!ok) {
                        auto reply = make_status_reply(*h, RemoteStatus::BAD_REQUEST);
                        rc_server_->send_reply(ctx, reply.data(), reply.size());
                    } else {
                        // Dispatch + reply in place. hashes point into rbuf —
                        // valid for the duration of the handler, no copy needed.
                        const uint64_t* hashes =
                            reinterpret_cast<const uint64_t*>(h + 1);
                        std::vector<char> reply;
                        try {
                            switch (h->op) {
                                case REMOTE_OP_INSERT_RESERVE:
                                    reply = handle_insert(*h, hashes); break;
                                case REMOTE_OP_QUERY:
                                    reply = handle_query(*h, hashes); break;
                                case REMOTE_OP_INSERT:
                                case REMOTE_OP_QUERY_RELEASE:
                                case REMOTE_OP_INSERT_RELEASE:
                                    reply = handle_finalize_like(*h, hashes);
                                    break;
                                default:
                                    reply = make_status_reply(
                                        *h, RemoteStatus::UNKNOWN_OP);
                            }
                        } catch (const std::exception& e) {
                            static std::atomic<uint64_t> thrown{0};
                            uint64_t k = thrown.fetch_add(1,
                                             std::memory_order_relaxed);
                            if (k < 8 || (k & 1023) == 0)
                                fprintf(stderr, "[RadixServer] rc handler threw "
                                        "(op=%u req_id=%llu n=%d, occ %llu): %s\n",
                                        (unsigned)h->op,
                                        (unsigned long long)h->req_id, h->n,
                                        (unsigned long long)(k + 1), e.what());
                            reply = make_status_reply(*h, RemoteStatus::INTERNAL);
                        }
                        rc_server_->send_reply(ctx, reply.data(), reply.size());
                    }
                }
            }
        } catch (const std::exception& e) {
            static std::atomic<uint64_t> thrown{0};
            uint64_t k = thrown.fetch_add(1, std::memory_order_relaxed);
            if (k < 8 || (k & 1023) == 0)
                fprintf(stderr, "[RadixServer] rc_loop lane %d transport error "
                        "(occurrence %llu): %s\n", lane,
                        (unsigned long long)(k + 1), e.what());
            // Do not hot-spin a persistently failing transport.
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (progress) {
            idle = 0;
        } else if (++idle < kSpinBudget) {
            // hot: keep polling (event caught within µs)
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }
}

}  // namespace shmradix

#endif  // !SHMRADIX_NO_RDMA
