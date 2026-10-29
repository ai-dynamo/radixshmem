// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Client half of the binary remote-op protocol shared by the DC and ZMQ
// backends: encode the request, send it over the subclass byte transport,
// decode the reply. Server half lives in remote_op_wire_server.cpp.
#include "radixshmem_config.h"

#ifndef RADIXSHMEM_NO_RDMA

#include "remote_op_wire.hpp"
#include "cluster_protocol.h"       // remote-op wire structs

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace radixshmem {

namespace {

std::vector<char> build_req(uint16_t op, const uint64_t* hashes, int n,
                            uint64_t req_id, RemoteOpReqHeader*& hdr) {
    hdr = nullptr;
    if (n < 0 || n > MAX_REMOTE_OP_BLOCKS) return {};
    std::vector<char> req(sizeof(RemoteOpReqHeader) + (size_t)n * sizeof(uint64_t));
    hdr = reinterpret_cast<RemoteOpReqHeader*>(req.data());
    *hdr = {};
    hdr->magic   = REMOTE_OP_REQ_MAGIC;
    hdr->version = REMOTE_OP_PROTOCOL_VERSION;
    hdr->op      = op;
    hdr->n       = n;
    hdr->req_id  = req_id;
    if (n > 0)
        std::memcpy(hdr + 1, hashes, (size_t)n * sizeof(uint64_t));
    return req;
}

// Guard against a reply that under-delivers the trailing slots[] array.
bool read_slots(const std::vector<char>& reply, int num_slots,
                std::vector<int32_t>& out) {
    size_t need = sizeof(RemoteOpReplyHeader) + (size_t)num_slots * sizeof(int32_t);
    if (num_slots < 0 || reply.size() < need) return false;
    auto* slots = reinterpret_cast<const int32_t*>(
        reinterpret_cast<const RemoteOpReplyHeader*>(reply.data()) + 1);
    out.assign(slots, slots + num_slots);
    return true;
}

}  // namespace

WireRemoteOpClientTransport::WireRemoteOpClientTransport(
    std::shared_ptr<std::mutex> op_mu)
    : op_mu_(op_mu ? std::move(op_mu) : std::make_shared<std::mutex>()) {}

AllocateReply WireRemoteOpClientTransport::allocate(int n) {
    auto g = guard();
    AllocateReply out;
    // ALLOCATE carries no hashes: n is the slot count, request is header-only.
    if (n < 0 || n > MAX_REMOTE_OP_BLOCKS) {
        out.status = RemoteStatus::BAD_REQUEST; return out;
    }
    std::vector<char> req(sizeof(RemoteOpReqHeader));
    auto* h = reinterpret_cast<RemoteOpReqHeader*>(req.data());
    *h = {};
    h->magic   = REMOTE_OP_REQ_MAGIC;
    h->version = REMOTE_OP_PROTOCOL_VERSION;
    h->op      = REMOTE_OP_INSERT_RESERVE;
    h->n       = n;
    h->req_id  = next_req_id_ + 1;
    ++next_req_id_;
    // flags stays 0: bit0 is reserved.

    std::vector<char> reply;
    RemoteStatus st = send_request(req.data(), req.size(), next_req_id_, reply);
    if (st != RemoteStatus::OK) { out.status = st; return out; }

    auto* r = reinterpret_cast<RemoteOpReplyHeader*>(reply.data());
    out.status = static_cast<RemoteStatus>(r->status);
    if (out.status != RemoteStatus::OK) return out;   // BAD_REQUEST / UNKNOWN_OP
    out.error = static_cast<InsertError>(r->error);
    if (!read_slots(reply, r->u.reserve.num_slots, out.slots))
        out.status = RemoteStatus::BAD_REPLY;
    return out;
}

RemoteStatus WireRemoteOpClientTransport::send_ack_op(
        uint16_t op, const uint64_t* hashes, int n, int start, int length,
        const int32_t* slots) {
    auto g = guard();
    // Only the two RELEASE ops route here now. INSERT_RELEASE carries a slot
    // count; QUERY_RELEASE length is a hash range (<=n).
    const bool slot_op = (op == REMOTE_OP_INSERT_RELEASE);
    if (length < 0 || (slot_op ? length > MAX_REMOTE_OP_BLOCKS
                               : length > n)) return RemoteStatus::BAD_REQUEST;
    RemoteOpReqHeader* h = nullptr;
    auto req = build_req(op, hashes, n, next_req_id_ + 1, h);
    if (!h) return RemoteStatus::BAD_REQUEST; // over the cap: nothing sent
    if (slots && length > 0) {
        size_t off = req.size();
        req.resize(off + (size_t)length * sizeof(int32_t));
        h = reinterpret_cast<RemoteOpReqHeader*>(req.data());
        std::memcpy(req.data() + off, slots, (size_t)length * sizeof(int32_t));
    }
    ++next_req_id_;
    h->u.insert.start  = start;
    h->u.insert.length = length;

    std::vector<char> reply;
    RemoteStatus st = send_request(req.data(), req.size(), next_req_id_, reply);
    if (st != RemoteStatus::OK) return st;
    auto* r = reinterpret_cast<RemoteOpReplyHeader*>(reply.data());
    RemoteStatus rst = static_cast<RemoteStatus>(r->status);
    if (rst != RemoteStatus::OK) return rst;   // target-side reject
    return r->error == 0 ? RemoteStatus::OK : RemoteStatus::INTERNAL;
}

InsertResult WireRemoteOpClientTransport::insert(
        const uint64_t* hashes, int n, int start, int length,
        const int32_t* slots, bool auto_recycle) {
    auto g = guard();
    InsertResult out;
    if (length < 0 || length > MAX_REMOTE_OP_BLOCKS ||
        (length > 0 && !slots)) {              // slots are the payload now
        out.error = InsertError::REMOTE_FAILED;
        out.remote_status = RemoteStatus::BAD_REQUEST;
        return out;
    }
    RemoteOpReqHeader* h = nullptr;
    auto req = build_req(REMOTE_OP_INSERT, hashes, n, next_req_id_ + 1, h);
    if (!h) {
        out.error = InsertError::REMOTE_FAILED;
        out.remote_status = RemoteStatus::BAD_REQUEST;
        return out;
    }
    if (slots && length > 0) {
        size_t off = req.size();
        req.resize(off + (size_t)length * sizeof(int32_t));
        h = reinterpret_cast<RemoteOpReqHeader*>(req.data());
        std::memcpy(req.data() + off, slots, (size_t)length * sizeof(int32_t));
    }
    ++next_req_id_;
    h->flags           = auto_recycle ? (uint16_t)0
                                      : REMOTE_OP_FLAG_NO_AUTO_RECYCLE;
    h->u.insert.start  = start;
    h->u.insert.length = length;

    std::vector<char> reply;
    RemoteStatus st = send_request(req.data(), req.size(), next_req_id_, reply);
    if (st != RemoteStatus::OK) {
        out.error = InsertError::REMOTE_FAILED;
        out.remote_status = st;
        return out;
    }

    auto* r = reinterpret_cast<RemoteOpReplyHeader*>(reply.data());
    if (static_cast<RemoteStatus>(r->status) != RemoteStatus::OK) {
        out.error = InsertError::REMOTE_FAILED;   // request-level reject (BAD_REQUEST / UNKNOWN_OP)
        out.remote_status = static_cast<RemoteStatus>(r->status);
        return out;
    }
    out.error = static_cast<InsertError>(r->error);
    read_slots(reply, r->u.insert.num_unused, out.unused_slots);
    return out;
}

RemoteStatus WireRemoteOpClientTransport::recycle(const int32_t* slots,
                                                    int num_slots) {
    return send_ack_op(REMOTE_OP_INSERT_RELEASE, nullptr, 0, 0, num_slots,
                       slots);
}

RemoteStatus WireRemoteOpClientTransport::unlock(const uint64_t* hashes,
                                                   int n, int start,
                                                   int length) {
    return send_ack_op(REMOTE_OP_QUERY_RELEASE, hashes, n, start, length,
                       nullptr);
}

QueryReply WireRemoteOpClientTransport::query(const uint64_t* hashes, int n,
                                                bool lock, bool update_meta) {
    auto g = guard();
    QueryReply out;
    RemoteOpReqHeader* h = nullptr;
    auto req = build_req(REMOTE_OP_QUERY, hashes, n, next_req_id_ + 1, h);
    if (!h) { out.status = RemoteStatus::BAD_REQUEST; return out; }
    ++next_req_id_;
    h->u.query.local_only  = 1;
    h->u.query.lock        = lock ? 1 : 0;
    h->u.query.update_meta = update_meta ? 1 : 0;

    std::vector<char> reply;
    RemoteStatus st = send_request(req.data(), req.size(), next_req_id_, reply);
    if (st != RemoteStatus::OK) { out.status = st; return out; }

    auto* r = reinterpret_cast<RemoteOpReplyHeader*>(reply.data());
    out.status = static_cast<RemoteStatus>(r->status);
    if (out.status != RemoteStatus::OK) return out;   // BAD_REQUEST / UNKNOWN_OP
    out.hit_length = r->u.query.hit_length;
    if (!read_slots(reply, r->u.query.num_slots, out.slots))
        out.status = RemoteStatus::BAD_REPLY;
    return out;
}

}  // namespace radixshmem

#endif  // !RADIXSHMEM_NO_RDMA
