// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// gRPC client: encode arguments, issue unary calls, validate replies and map statuses to RemoteStatus.
#include "grpc_remote_op.hpp"

#ifdef RADIXSHMEM_HAS_GRPC

#include "cluster_protocol.h"
#include "remote_ops.grpc.pb.h"

#include <grpcpp/grpcpp.h>

#include <chrono>
#include <cstdlib>
#include <vector>

namespace radixshmem {

namespace pb = ::radixshmem::remoteops::v1;

namespace {

RemoteStatus to_remote_status(const grpc::Status& s) {
    switch (s.error_code()) {
    case grpc::StatusCode::OK:                 return RemoteStatus::OK;
    case grpc::StatusCode::DEADLINE_EXCEEDED:  return RemoteStatus::TIMEOUT;
    case grpc::StatusCode::UNAVAILABLE:
    case grpc::StatusCode::CANCELLED:          return RemoteStatus::UNAVAILABLE;
    case grpc::StatusCode::RESOURCE_EXHAUSTED: return RemoteStatus::RESOURCE_EXHAUSTED;
    case grpc::StatusCode::INVALID_ARGUMENT:   return RemoteStatus::BAD_REQUEST;
    case grpc::StatusCode::UNIMPLEMENTED:      return RemoteStatus::UNKNOWN_OP;
    case grpc::StatusCode::FAILED_PRECONDITION: return RemoteStatus::STALE_TARGET;
    default:                                   return RemoteStatus::INTERNAL;
    }
}

bool to_insert_error(pb::InsertError e, InsertError& out) {
    if (e < pb::INSERT_ERROR_OK || e > pb::INSERT_ERROR_FULL_PATH_MISSING)
        return false;
    out = static_cast<InsertError>(e);
    return true;
}

// The array can be copied into a request; range checks are the server's.
bool copyable(const void* p, int n) {
    return n >= 0 && n <= MAX_REMOTE_OP_BLOCKS && (n == 0 || p);
}

}  // namespace

int grpc_remote_op_timeout_ms() {
    if (const char* e = ::getenv("RADIXSHMEM_REMOTE_OP_TIMEOUT_MS")) {
        int v = std::atoi(e);
        if (v > 0) return v;
    }
    return 5000;
}

struct GrpcRemoteOpClientTransport::Impl {
    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<pb::RemoteOps::Stub> stub;
    std::chrono::milliseconds timeout;
    std::string instance;

    // Fresh context per call; the deadline covers connect + execute.
    void prepare(grpc::ClientContext& ctx) const {
        ctx.set_deadline(std::chrono::system_clock::now() + timeout);
        ctx.AddMetadata(kGrpcRemoteOpInstanceKey, instance);
    }
};

GrpcRemoteOpClientTransport::GrpcRemoteOpClientTransport(
        const std::string& target, int timeout_ms, uint64_t instance_id)
    : impl_(std::make_unique<Impl>()) {
    grpc::ChannelArguments args;
    args.SetInt(GRPC_ARG_ENABLE_RETRIES, 0);   // never replay an op
    args.SetInt(GRPC_ARG_ENABLE_HTTP_PROXY, 0);
    args.SetMaxSendMessageSize(kGrpcRemoteOpMaxMessageBytes);
    args.SetMaxReceiveMessageSize(kGrpcRemoteOpMaxMessageBytes);
    args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, kGrpcRemoteOpKeepaliveMs);
    args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 5000);
    args.SetInt(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
    args.SetInt(GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA, 0);
    args.SetInt(GRPC_ARG_HTTP2_MIN_SENT_PING_INTERVAL_WITHOUT_DATA_MS,
                kGrpcRemoteOpKeepaliveMs);
    impl_->channel = grpc::CreateCustomChannel(
        target, grpc::InsecureChannelCredentials(), args);
    impl_->stub = pb::RemoteOps::NewStub(impl_->channel);
    impl_->timeout = std::chrono::milliseconds(timeout_ms);
    impl_->instance = std::to_string(instance_id);
}

GrpcRemoteOpClientTransport::~GrpcRemoteOpClientTransport() = default;

AllocateReply GrpcRemoteOpClientTransport::allocate(int n, ComponentType component) {
    AllocateReply out;
    if (n < 0 || n > MAX_REMOTE_OP_BLOCKS) { out.status = RemoteStatus::BAD_REQUEST; return out; }
    pb::AllocateRequest req;
    req.set_num_blocks(n);
    req.set_component(static_cast<pb::Component>(component));
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

AllocateComponentsReply GrpcRemoteOpClientTransport::allocate_components(
        const std::vector<ComponentCount>& components) {
    AllocateComponentsReply out;
    bool sane = true;
    int64_t total = 0;
    for (const ComponentCount& c : components) {
        sane = sane && c.n >= 0;
        total += c.n;
    }
    if (!sane || total > MAX_REMOTE_OP_BLOCKS) { out.status = RemoteStatus::BAD_REQUEST; return out; }
    pb::AllocateComponentsRequest req;
    for (const ComponentCount& c : components) {
        pb::ComponentCount* m = req.add_components();
        m->set_component(static_cast<pb::Component>(c.component));
        m->set_num_blocks(c.n);
    }
    pb::AllocateComponentsResponse rep;
    grpc::ClientContext ctx;
    impl_->prepare(ctx);
    out.status = to_remote_status(impl_->stub->AllocateComponents(&ctx, req, &rep));
    if (out.status != RemoteStatus::OK) return out;
    sane = to_insert_error(rep.error(), out.error) &&
                rep.components_size() == (out.error == InsertError::OK ? (int)components.size() : 0);
    for (int i = 0; sane && i < rep.components_size(); i++)
        sane = rep.components(i).slots_size() == components[i].n;
    if (!sane) {
        out.status = RemoteStatus::BAD_REPLY;
        out.error = InsertError::OK;
        return out;
    }
    out.slots.resize(rep.components_size());
    for (int i = 0; i < rep.components_size(); i++)
        out.slots[i].assign(rep.components(i).slots().begin(), rep.components(i).slots().end());
    return out;
}

QueryReply GrpcRemoteOpClientTransport::query(const uint64_t* hashes, int n,
                                              bool lock, bool update_meta) {
    QueryReply out;
    if (!copyable(hashes, n)) { out.status = RemoteStatus::BAD_REQUEST; return out; }
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
        rep.slots_size() != rep.hit_length() ||
        (lock && rep.hit_length() > 0 && rep.pin_end() < rep.hit_length())) {
        out.status = RemoteStatus::BAD_REPLY;
        return out;
    }
    out.hit_length = rep.hit_length();
    out.slots.assign(rep.slots().begin(), rep.slots().end());
    out.pin_end = rep.pin_end();
    out.pin_tail.assign(rep.pin_tail().begin(), rep.pin_tail().end());
    return out;
}

InsertResult GrpcRemoteOpClientTransport::insert(
        const uint64_t* hashes, int n, int start, int length,
        const int32_t* slots, bool auto_recycle, ComponentType component) {
    InsertResult out;
    if (!copyable(hashes, n) || !copyable(slots, length)) {
        out.error = InsertError::REMOTE_FAILED;
        out.remote_status = RemoteStatus::BAD_REQUEST;
        return out;
    }
    pb::InsertRequest req;
    req.mutable_hashes()->Add(hashes, hashes + n);
    req.mutable_slots()->Add(slots, slots + length);
    req.set_start(start);
    req.set_auto_recycle(auto_recycle);
    req.set_component(static_cast<pb::Component>(component));
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

InsertPathResult GrpcRemoteOpClientTransport::insert_path(
        const uint64_t* hashes, int n, int start, int length, const int32_t* full_slots,
        const std::vector<ComponentRun>& runs, bool auto_recycle) {
    InsertPathResult out;
    out.runs.resize(runs.size());
    auto fail = [&](RemoteStatus st) {
        out.remote_status = st;
        out.full = InsertResult{};
        out.full.error = InsertError::REMOTE_FAILED;
        out.full.remote_status = st;
        out.runs.assign(runs.size(), out.full);
        return out;
    };
    bool sane = copyable(hashes, n) && copyable(full_slots, length) &&
                runs.size() <= (size_t)MAX_REMOTE_OP_BLOCKS;
    for (const ComponentRun& r : runs)
        sane = sane && copyable(r.slots.data(), (int)r.slots.size());
    if (!sane) return fail(RemoteStatus::BAD_REQUEST);
    pb::InsertPathRequest req;
    req.mutable_hashes()->Add(hashes, hashes + n);
    req.set_start(start);
    req.mutable_full_slots()->Add(full_slots, full_slots + length);
    req.set_auto_recycle(auto_recycle);
    for (const ComponentRun& r : runs) {
        pb::ComponentRun* m = req.add_runs();
        m->set_component(static_cast<pb::Component>(r.component));
        m->set_end(r.end);
        m->mutable_slots()->Add(r.slots.begin(), r.slots.end());
    }
    if (req.ByteSizeLong() > (size_t)kGrpcRemoteOpMaxMessageBytes)
        return fail(RemoteStatus::BAD_REQUEST);
    pb::InsertPathResponse rep;
    grpc::ClientContext ctx;
    impl_->prepare(ctx);
    RemoteStatus st = to_remote_status(impl_->stub->InsertPath(&ctx, req, &rep));
    if (st == RemoteStatus::OK &&
        (rep.runs_size() != (int)runs.size() ||
         !to_insert_error(rep.full().error(), out.full.error) ||
         rep.full().unused_slots_size() > length))
        st = RemoteStatus::BAD_REPLY;
    for (int i = 0; st == RemoteStatus::OK && i < rep.runs_size(); i++)
        if (!to_insert_error(rep.runs(i).error(), out.runs[i].error) ||
            rep.runs(i).unused_slots_size() > (int)runs[i].slots.size())
            st = RemoteStatus::BAD_REPLY;
    if (st != RemoteStatus::OK) return fail(st);
    out.full.unused_slots.assign(rep.full().unused_slots().begin(), rep.full().unused_slots().end());
    for (int i = 0; i < rep.runs_size(); i++)
        out.runs[i].unused_slots.assign(rep.runs(i).unused_slots().begin(),
                                        rep.runs(i).unused_slots().end());
    return out;
}

RemoteStatus GrpcRemoteOpClientTransport::recycle(const int32_t* slots,
                                                  int num_slots,
                                                  ComponentType component) {
    if (!copyable(slots, num_slots)) return RemoteStatus::BAD_REQUEST;
    pb::RecycleRequest req;
    req.mutable_slots()->Add(slots, slots + num_slots);
    req.set_component(static_cast<pb::Component>(component));
    pb::RecycleResponse rep;
    grpc::ClientContext ctx;
    impl_->prepare(ctx);
    return to_remote_status(impl_->stub->Recycle(&ctx, req, &rep));
}

RemoteStatus GrpcRemoteOpClientTransport::unlock(
        const uint64_t* hashes, int n, int start, int length, int pin_end,
        const std::vector<uint64_t>& pin_tail) {
    if (!copyable(hashes, n) || !copyable(pin_tail.data(), (int)pin_tail.size()))
        return RemoteStatus::BAD_REQUEST;
    pb::UnlockRequest req;
    req.mutable_hashes()->Add(hashes, hashes + n);
    req.set_start(start);
    req.set_length(length);
    req.set_pin_end(pin_end);
    req.mutable_pin_tail()->Add(pin_tail.begin(), pin_tail.end());
    pb::UnlockResponse rep;
    grpc::ClientContext ctx;
    impl_->prepare(ctx);
    return to_remote_status(impl_->stub->Unlock(&ctx, req, &rep));
}

bool GrpcRemoteOpClientTransport::connected() {
    return impl_->channel->GetState(false) == GRPC_CHANNEL_READY;
}

}  // namespace radixshmem

#endif  // RADIXSHMEM_HAS_GRPC
