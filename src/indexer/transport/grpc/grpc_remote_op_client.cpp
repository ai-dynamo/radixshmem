// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Client half of the gRPC remote-op backend: method arguments → protobuf,
// one unary call per op, response checks, gRPC status → RemoteStatus.
#include "grpc_remote_op.hpp"

#ifdef RADIXSHMEM_HAS_GRPC

#include "radix_client.h"       // AllocateReply, QueryReply
#include "cluster_protocol.h"   // MAX_REMOTE_OP_BLOCKS
#include "remote_ops.grpc.pb.h"

#include <grpcpp/grpcpp.h>

#include <chrono>
#include <cstdlib>
#include <vector>

namespace radixshmem {

namespace pb = ::radixshmem::remoteops::v1;

namespace {

// A request or reply never exceeds MAX_REMOTE_OP_BLOCKS hashes + slots.
constexpr int kMaxMessageBytes = 64 * 1024;

RemoteStatus to_remote_status(const grpc::Status& s) {
    switch (s.error_code()) {
    case grpc::StatusCode::OK:                 return RemoteStatus::OK;
    case grpc::StatusCode::DEADLINE_EXCEEDED:  return RemoteStatus::TIMEOUT;
    case grpc::StatusCode::UNAVAILABLE:
    case grpc::StatusCode::CANCELLED:          return RemoteStatus::UNAVAILABLE;
    case grpc::StatusCode::RESOURCE_EXHAUSTED: return RemoteStatus::RESOURCE_EXHAUSTED;
    case grpc::StatusCode::INVALID_ARGUMENT:   return RemoteStatus::BAD_REQUEST;
    case grpc::StatusCode::UNIMPLEMENTED:      return RemoteStatus::UNKNOWN_OP;
    default:                                   return RemoteStatus::INTERNAL;
    }
}

bool to_insert_error(pb::InsertError e, InsertError& out) {
    if (e < pb::INSERT_ERROR_OK || e > pb::INSERT_ERROR_FULL_PATH_MISSING)
        return false;
    out = static_cast<InsertError>(e);
    return true;
}

bool valid_count(int n) { return n >= 0 && n <= MAX_REMOTE_OP_BLOCKS; }

}  // namespace

int grpc_remote_op_timeout_ms() {
    if (const char* e = ::getenv("RADIXSHMEM_REMOTE_OP_TIMEOUT_MS")) {
        int v = std::atoi(e);
        if (v > 0) return v;
    }
    return 5000;
}

struct GrpcRemoteOpClientTransport::Impl {
    std::unique_ptr<pb::RemoteOps::Stub> stub;
    std::chrono::milliseconds timeout;

    // Fresh context per call; the deadline covers connect + execute.
    void prepare(grpc::ClientContext& ctx) const {
        ctx.set_deadline(std::chrono::system_clock::now() + timeout);
    }
};

GrpcRemoteOpClientTransport::GrpcRemoteOpClientTransport(
        const std::string& target, int timeout_ms)
    : impl_(std::make_unique<Impl>()) {
    grpc::ChannelArguments args;
    args.SetInt(GRPC_ARG_ENABLE_RETRIES, 0);   // never replay an op
    args.SetMaxSendMessageSize(kMaxMessageBytes);
    args.SetMaxReceiveMessageSize(kMaxMessageBytes);
    auto channel = grpc::CreateCustomChannel(
        target, grpc::InsecureChannelCredentials(), args);
    impl_->stub = pb::RemoteOps::NewStub(channel);
    impl_->timeout = std::chrono::milliseconds(timeout_ms);
}

GrpcRemoteOpClientTransport::~GrpcRemoteOpClientTransport() = default;

AllocateReply GrpcRemoteOpClientTransport::allocate(int n) {
    AllocateReply out;
    if (!valid_count(n)) { out.status = RemoteStatus::BAD_REQUEST; return out; }
    pb::AllocateRequest req;
    req.set_num_blocks(n);
    pb::AllocateResponse rep;
    grpc::ClientContext ctx;
    impl_->prepare(ctx);
    out.status = to_remote_status(impl_->stub->Allocate(&ctx, req, &rep));
    if (out.status != RemoteStatus::OK) return out;
    if (!to_insert_error(rep.error(), out.error) ||
        rep.slots_size() != (out.error == InsertError::OK ? n : 0)) {
        out.status = RemoteStatus::BAD_REPLY;
        out.error  = InsertError::OK;
        return out;
    }
    out.slots.assign(rep.slots().begin(), rep.slots().end());
    return out;
}

QueryReply GrpcRemoteOpClientTransport::query(const uint64_t* hashes, int n,
                                              bool lock, bool update_meta) {
    QueryReply out;
    if (!valid_count(n) || (n > 0 && !hashes)) {
        out.status = RemoteStatus::BAD_REQUEST;
        return out;
    }
    pb::QueryRequest req;
    req.mutable_hashes()->Add(hashes, hashes + n);
    req.set_lock(lock);
    req.set_update_meta(update_meta);
    pb::QueryResponse rep;
    grpc::ClientContext ctx;
    impl_->prepare(ctx);
    out.status = to_remote_status(impl_->stub->Query(&ctx, req, &rep));
    if (out.status != RemoteStatus::OK) return out;
    if (rep.hit_length() < 0 || rep.hit_length() > n ||
        rep.slots_size() != rep.hit_length()) {
        out.status = RemoteStatus::BAD_REPLY;
        return out;
    }
    out.hit_length = rep.hit_length();
    out.slots.assign(rep.slots().begin(), rep.slots().end());
    return out;
}

InsertResult GrpcRemoteOpClientTransport::insert(
        const uint64_t* hashes, int n, int start, int length,
        const int32_t* slots, bool auto_recycle) {
    InsertResult out;
    if (!valid_count(n) || length < 0 || start < 0 || start > n - length ||
        (n > 0 && !hashes) || (length > 0 && !slots)) {
        out.error = InsertError::REMOTE_FAILED;
        out.remote_status = RemoteStatus::BAD_REQUEST;
        return out;
    }
    pb::InsertRequest req;
    req.mutable_hashes()->Add(hashes, hashes + n);
    req.mutable_slots()->Add(slots, slots + length);
    req.set_start(start);
    req.set_auto_recycle(auto_recycle);
    pb::InsertResponse rep;
    grpc::ClientContext ctx;
    impl_->prepare(ctx);
    RemoteStatus st = to_remote_status(impl_->stub->Insert(&ctx, req, &rep));
    if (st == RemoteStatus::OK &&
        (!to_insert_error(rep.error(), out.error) ||
         rep.unused_slots_size() > length))
        st = RemoteStatus::BAD_REPLY;
    if (st != RemoteStatus::OK) {
        out.error = InsertError::REMOTE_FAILED;
        out.remote_status = st;
        return out;
    }
    out.unused_slots.assign(rep.unused_slots().begin(),
                            rep.unused_slots().end());
    return out;
}

RemoteStatus GrpcRemoteOpClientTransport::recycle(const int32_t* slots,
                                                  int num_slots) {
    if (!valid_count(num_slots) || (num_slots > 0 && !slots))
        return RemoteStatus::BAD_REQUEST;
    pb::RecycleRequest req;
    req.mutable_slots()->Add(slots, slots + num_slots);
    pb::RecycleResponse rep;
    grpc::ClientContext ctx;
    impl_->prepare(ctx);
    return to_remote_status(impl_->stub->Recycle(&ctx, req, &rep));
}

RemoteStatus GrpcRemoteOpClientTransport::unlock(const uint64_t* hashes, int n,
                                                 int start, int length) {
    if (!valid_count(n) || length < 0 || start < 0 || start > n - length ||
        (n > 0 && !hashes))
        return RemoteStatus::BAD_REQUEST;
    pb::UnlockRequest req;
    req.mutable_hashes()->Add(hashes, hashes + n);
    req.set_start(start);
    req.set_length(length);
    pb::UnlockResponse rep;
    grpc::ClientContext ctx;
    impl_->prepare(ctx);
    return to_remote_status(impl_->stub->Unlock(&ctx, req, &rep));
}

}  // namespace radixshmem

#endif  // RADIXSHMEM_HAS_GRPC
