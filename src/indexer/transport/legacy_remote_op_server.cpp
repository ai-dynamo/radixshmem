// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Server half of the legacy (DC/ZMQ) remote-op backends: lane pollers that
// decode the binary protocol, run the op on RemoteOpService and send the
// encoded reply. Client half lives in peer/remote_op.cpp.
#include "radixshmem_config.h"

#ifndef RADIXSHMEM_NO_RDMA

#include "bilateral_transport.h"
#include "cluster_protocol.h"
#include "remote_op_service.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <thread>
#include <vector>

namespace radixshmem {

namespace {

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

// OK reply with `error` and a trailing int32 array.
std::vector<char> make_slots_reply(const RemoteOpReqHeader& req, uint16_t error,
                                   const std::vector<int32_t>& slots,
                                   RemoteOpReplyHeader*& rep) {
    std::vector<char> buf(sizeof(RemoteOpReplyHeader) +
                          slots.size() * sizeof(int32_t));
    rep = reinterpret_cast<RemoteOpReplyHeader*>(buf.data());
    *rep = {};
    rep->magic   = REMOTE_OP_REPLY_MAGIC;
    rep->version = REMOTE_OP_PROTOCOL_VERSION;
    rep->op      = req.op;
    rep->status  = static_cast<uint16_t>(RemoteStatus::OK);
    rep->error   = error;
    rep->req_id  = req.req_id;
    if (!slots.empty())
        std::memcpy(rep + 1, slots.data(), slots.size() * sizeof(int32_t));
    return buf;
}

}  // namespace

void LegacyRemoteOpServerTransport::start(RemoteOpService& svc) {
    svc_ = &svc;
    running_.store(true);
    int n = lanes();
    threads_.reserve(n);
    for (int i = 0; i < n; ++i)
        threads_.emplace_back(&LegacyRemoteOpServerTransport::loop, this, i);
}

void LegacyRemoteOpServerTransport::stop() {
    // The pollers exit on running_ == false; nothing else to wake.
    running_.store(false);
    for (auto& t : threads_)
        if (t.joinable()) t.join();
    threads_.clear();
}

std::vector<char> LegacyRemoteOpServerTransport::dispatch(
        const RemoteOpReqHeader& req, const uint64_t* hashes) {
    RemoteOpReplyHeader* rep = nullptr;
    switch (req.op) {
    case REMOTE_OP_INSERT_RESERVE: {
        // RESERVE carries no hashes: req.n is the slot count.
        AllocateReply r = svc_->allocate(req.n);
        if (r.status != RemoteStatus::OK) return make_status_reply(req, r.status);
        auto buf = make_slots_reply(req, static_cast<uint16_t>(r.error),
                                    r.slots, rep);
        rep->u.reserve.matched_prefix = 0;   // no server-side dedup
        rep->u.reserve.inserted_count = (int)r.slots.size();
        rep->u.reserve.num_slots      = (int)r.slots.size();
        return buf;
    }
    case REMOTE_OP_QUERY: {
        const auto& q = req.u.query;
        QueryReply r = svc_->query(hashes, req.n, q.lock != 0,
                                   q.update_meta != 0);
        if (r.status != RemoteStatus::OK) return make_status_reply(req, r.status);
        auto buf = make_slots_reply(req, 0, r.slots, rep);
        rep->u.query.hit_length = r.hit_length;
        rep->u.query.num_slots  = (int)r.slots.size();   // == hit_length
        return buf;
    }
    case REMOTE_OP_INSERT: {
        // The phase-1 slots trail the hashes (u.insert.length int32s).
        const auto& f = req.u.insert;
        const int32_t* slots = reinterpret_cast<const int32_t*>(hashes + req.n);
        bool ar = !(req.flags & REMOTE_OP_FLAG_NO_AUTO_RECYCLE);
        InsertResult r = svc_->insert(hashes, req.n, f.start, slots, f.length, ar);
        if (r.remote_status != RemoteStatus::OK)
            return make_status_reply(req, r.remote_status);
        auto buf = make_slots_reply(req, static_cast<uint16_t>(r.error),
                                    r.unused_slots, rep);
        rep->u.insert.num_unused = (int)r.unused_slots.size();
        return buf;
    }
    case REMOTE_OP_INSERT_RELEASE: {
        const int32_t* slots = reinterpret_cast<const int32_t*>(hashes + req.n);
        return make_status_reply(req, svc_->recycle(slots, req.u.insert.length));
    }
    case REMOTE_OP_QUERY_RELEASE:
        return make_status_reply(req, svc_->unlock(hashes, req.n,
                                                   req.u.insert.start,
                                                   req.u.insert.length));
    default:
        return make_status_reply(req, RemoteStatus::UNKNOWN_OP);
    }
}

void LegacyRemoteOpServerTransport::loop(int lane) {
    std::vector<char> rbuf(REMOTE_OP_MAX_REQ_BYTES);

    int idle = 0;                 // consecutive no-progress iterations
    constexpr int kSpinBudget = 2048;  // spin ~this many idle iters before sleeping

    while (running_.load(std::memory_order_relaxed)) {
        bool progress = false;
        try {
            ReplyContext ctx;
            int got = recv_request(rbuf, ctx, /*non-blocking*/0, lane);
            if (got >= 0) {
                progress = true;
                if (got < (int)sizeof(RemoteOpReqHeader)) {
                    RemoteOpReqHeader dummy{};
                    auto reply = make_status_reply(dummy, RemoteStatus::BAD_REQUEST);
                    send_reply(ctx, reply.data(), reply.size());
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
                        send_reply(ctx, reply.data(), reply.size());
                    } else {
                        // Dispatch + reply in place. hashes point into rbuf —
                        // valid for the duration of the handler, no copy needed.
                        const uint64_t* hashes =
                            reinterpret_cast<const uint64_t*>(h + 1);
                        std::vector<char> reply;
                        try {
                            reply = dispatch(*h, hashes);
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
                        send_reply(ctx, reply.data(), reply.size());
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

}  // namespace radixshmem

#endif  // !RADIXSHMEM_NO_RDMA
