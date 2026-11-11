// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// gRPC client: encode arguments, issue unary calls, validate replies and map statuses to RemoteStatus.
#include "grpc_remote_op.hpp"

#ifdef RADIXSHMEM_HAS_GRPC

#include "cluster_protocol.h"
#include "remote_ops.grpc.pb.h"
#include "remote_ops_codec.hpp"

#include <grpcpp/grpcpp.h>

#include <algorithm>
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
                                              const RemoteQuery& q) {
    QueryReply out;
    if (!copyable(hashes, n)) { out.status = RemoteStatus::BAD_REQUEST; return out; }
    pb::QueryRequest req;
    req.mutable_hashes()->Add(hashes, hashes + n);
    req.set_lock(q.lock);
    req.set_update_meta(q.update_meta);
    req.set_mask(q.mask);
    req.set_incumbent(q.window.incumbent);
    req.set_full_from(q.window.full_from);
    pb::QueryResponse rep;
    grpc::ClientContext ctx;
    impl_->prepare(ctx);
    out.status = to_remote_status(impl_->stub->Query(&ctx, req, &rep));
    if (out.status != RemoteStatus::OK) return out;
    const int64_t hit = rep.hit_length();
    const bool swa = (q.mask & COMPONENT_MASK_SWA) != 0 && hit > 0;
    const pb::Pin& pin = rep.pin();
    const int64_t pin_end = std::max({(int64_t)pin.full_end(), (int64_t)pin.swa_end(),
                                      (int64_t)pin.mamba_end()});
    if (hit < 0 || hit > n ||
        rep.slots_size() != ((q.mask & COMPONENT_MASK_FULL) != 0
                                 ? hit - std::min<int64_t>(q.window.full_from, hit) : 0) ||
        (swa ? rep.swa_start() > hit || rep.swa_slots_size() != hit - rep.swa_start()
             : rep.swa_slots_size() != 0) ||
        (rep.has_mamba() && (q.mask & COMPONENT_MASK_MAMBA) == 0) ||
        (rep.has_pin() && !q.lock) || pin.full_start() > pin.full_end() ||
        pin.swa_start() > pin.swa_end() || pin_end > hit + pin.tail_size()) {
        out.status = RemoteStatus::BAD_REPLY;
        return out;
    }
    out.hit_length = (int)hit;
    out.slots.assign(rep.slots().begin(), rep.slots().end());
    out.swa_start = rep.swa_start();
    out.swa_slots.assign(rep.swa_slots().begin(), rep.swa_slots().end());
    if (rep.has_mamba()) out.mamba_slot = rep.mamba().slot();
    out.pin = pin_from_pb(pin);
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

RemoteStatus GrpcRemoteOpClientTransport::unlock(const uint64_t* hashes, int n,
                                                const PinRecord& pin) {
    if (!copyable(hashes, n) || !copyable(pin.tail.data(), (int)pin.tail.size()))
        return RemoteStatus::BAD_REQUEST;
    pb::UnlockRequest req;
    req.mutable_hashes()->Add(hashes, hashes + n);
    pin_to_pb(pin, req.mutable_pin());
    pb::UnlockResponse rep;
    grpc::ClientContext ctx;
    impl_->prepare(ctx);
    return to_remote_status(impl_->stub->Unlock(&ctx, req, &rep));
}

DedupReply GrpcRemoteOpClientTransport::dedup_prefix(const uint64_t* hashes, int n,
                                                     const DedupRequest& in) {
    DedupReply out;
    // Not sent or refused before it ran: no landing slot was written.
    auto unsent = [&](RemoteStatus st) {
        out.status = st;
        for (int k = 0; k < NUM_COMPONENT_TYPES; k++) out.unused[k] = in.landing[k];
        return out;
    };
    if (!copyable(hashes, n)) return unsent(RemoteStatus::BAD_REQUEST);
    pb::DedupPrefixRequest req;
    req.mutable_hashes()->Add(hashes, hashes + n);
    req.set_rank(in.rank);
    req.set_tail_cap(in.tail_cap);
    req.set_branch_cap(in.branch_cap);
    bool landing = false;
    for (const std::vector<int32_t>& l : in.landing) landing |= !l.empty();
    if (landing)
        for (const std::vector<int32_t>& l : in.landing)
            req.add_landing()->mutable_slots()->Add(l.begin(), l.end());
    pb::DedupPrefixResponse rep;
    grpc::ClientContext ctx;
    impl_->prepare(ctx);
    const RemoteStatus st = to_remote_status(impl_->stub->DedupPrefix(&ctx, req, &rep));
    if (st == RemoteStatus::BAD_REQUEST || st == RemoteStatus::UNKNOWN_OP ||
        st == RemoteStatus::RESOURCE_EXHAUSTED || st == RemoteStatus::STALE_TARGET)
        return unsent(st);
    out.status = st;
    if (st != RemoteStatus::OK) return out;
    out.freed = rep.freed();
    if (rep.unused_size() == 0) {   // a target without moves wrote nothing
        for (int k = 0; k < NUM_COMPONENT_TYPES; k++) out.unused[k] = in.landing[k];
        return out;
    }
    for (int k = 0; k < NUM_COMPONENT_TYPES && k < rep.unused_size(); k++)
        out.unused[k].assign(rep.unused(k).slots().begin(), rep.unused(k).slots().end());
    out.moved.resize(rep.moved_size());
    for (int i = 0; i < rep.moved_size(); i++) {
        const pb::MovedPath& m = rep.moved(i);
        MovedPath& p = out.moved[i];
        p.base = m.base();
        p.hashes.assign(m.hashes().begin(), m.hashes().end());
        p.full_from = m.full_from();
        p.full.assign(m.full().begin(), m.full().end());
        p.full_age_ns.assign(m.full_age_ns().begin(), m.full_age_ns().end());
        for (const pb::MovedCell& c : m.cells())
            p.cells.push_back({static_cast<ComponentType>(c.component()), c.pos(), c.slot(),
                               c.age_ns()});
    }
    return out;
}

bool GrpcRemoteOpClientTransport::connected() {
    return impl_->channel->GetState(false) == GRPC_CHANNEL_READY;
}

}  // namespace radixshmem

#endif  // RADIXSHMEM_HAS_GRPC
