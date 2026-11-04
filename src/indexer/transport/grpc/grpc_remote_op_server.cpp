// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Gated, bounded RemoteOpService RPCs: business results in replies, request failures in gRPC status codes.
#include "grpc_remote_op.hpp"

#ifdef RADIXSHMEM_HAS_GRPC

#include "remote_op_service.hpp"
#include "remote_ops.grpc.pb.h"

#include <grpcpp/grpcpp.h>
#include <grpcpp/resource_quota.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>

namespace radixshmem {

namespace pb = ::radixshmem::remoteops::v1;

namespace {

constexpr int kNumCqs     = 4;
constexpr int kMinPollers = 4;   // per CQ
constexpr int kMaxPollers = 8;   // per CQ

int remote_op_max_threads() {
    const char* e = ::getenv("RADIXSHMEM_REMOTE_OP_MAX_THREADS");
    const int v = e ? std::atoi(e) : 0;
    return std::max(v > 0 ? v : 256, kNumCqs * kMaxPollers + 1);
}

// A service-level reject: BAD_REQUEST is the caller's fault, the rest ours.
grpc::Status to_grpc_status(RemoteStatus st) {
    if (st == RemoteStatus::BAD_REQUEST)
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                            "remote-op: invalid arguments");
    if (st == RemoteStatus::TIMEOUT)
        return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED,
                            "remote-op: deadline passed before the op ran");
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        "remote-op: status " + std::to_string((int)st));
}

}  // namespace

struct GrpcRemoteOpServer::Impl final : pb::RemoteOps::Service {
    std::unique_ptr<grpc::Server> server;
    int port = 0;
    std::string instance;

    RemoteOpService*  svc = nullptr;
    std::atomic<bool> ready{false};
    std::once_flag    stopped;

    template <typename Op>
    grpc::Status run(grpc::ServerContext* ctx, Op&& op) {
        if (!ready.load())
            return grpc::Status(grpc::StatusCode::UNAVAILABLE,
                                "remote-op: server not serving");
        const auto& md = ctx->client_metadata();
        auto it = md.find(kGrpcRemoteOpInstanceKey);
        if (it == md.end() || it->second != grpc::string_ref(instance))
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                "remote-op: request for another server instance");
        try {
            return op();
        } catch (const std::exception& e) {
            return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
        }
    }

    grpc::Status Allocate(grpc::ServerContext* ctx, const pb::AllocateRequest* req,
                          pb::AllocateResponse* rep) override {
        return run(ctx, [&] {
            AllocateReply r = svc->allocate(req->num_blocks(), ctx->deadline(),
                                            static_cast<ComponentType>(req->component()));
            if (r.status != RemoteStatus::OK) return to_grpc_status(r.status);
            rep->set_error(static_cast<pb::InsertError>(r.error));
            rep->mutable_slots()->Add(r.slots.begin(), r.slots.end());
            return grpc::Status::OK;
        });
    }

    grpc::Status AllocateComponents(grpc::ServerContext* ctx,
                                    const pb::AllocateComponentsRequest* req,
                                    pb::AllocateComponentsResponse* rep) override {
        return run(ctx, [&] {
            std::vector<ComponentCount> components(req->components_size());
            for (int i = 0; i < req->components_size(); i++) {
                components[i].component = static_cast<ComponentType>(req->components(i).component());
                components[i].n = req->components(i).num_blocks();
            }
            AllocateComponentsReply r = svc->allocate_components(components, ctx->deadline());
            if (r.status != RemoteStatus::OK) return to_grpc_status(r.status);
            rep->set_error(static_cast<pb::InsertError>(r.error));
            for (const std::vector<int32_t>& s : r.slots)
                rep->add_components()->mutable_slots()->Add(s.begin(), s.end());
            return grpc::Status::OK;
        });
    }

    grpc::Status Query(grpc::ServerContext* ctx, const pb::QueryRequest* req,
                       pb::QueryResponse* rep) override {
        return run(ctx, [&] {
            QueryReply r = svc->query(req->hashes().data(), req->hashes_size(),
                                      req->lock(), req->update_meta(),
                                      ctx->deadline());
            if (r.status != RemoteStatus::OK) return to_grpc_status(r.status);
            rep->set_hit_length(r.hit_length);
            rep->mutable_slots()->Add(r.slots.begin(), r.slots.end());
            rep->set_pin_end(r.pin_end);
            rep->mutable_pin_tail()->Add(r.pin_tail.begin(), r.pin_tail.end());
            return grpc::Status::OK;
        });
    }

    grpc::Status Insert(grpc::ServerContext* ctx, const pb::InsertRequest* req,
                        pb::InsertResponse* rep) override {
        return run(ctx, [&] {
            InsertResult r = svc->insert(
                req->hashes().data(), req->hashes_size(), req->start(),
                req->slots_size(), req->slots().data(), req->auto_recycle(),
                static_cast<ComponentType>(req->component()));
            if (r.remote_status != RemoteStatus::OK)
                return to_grpc_status(r.remote_status);
            rep->set_error(static_cast<pb::InsertError>(r.error));
            rep->mutable_unused_slots()->Add(r.unused_slots.begin(),
                                             r.unused_slots.end());
            return grpc::Status::OK;
        });
    }

    grpc::Status InsertPath(grpc::ServerContext* ctx, const pb::InsertPathRequest* req,
                            pb::InsertPathResponse* rep) override {
        return run(ctx, [&] {
            std::vector<ComponentRun> runs(req->runs_size());
            for (int i = 0; i < req->runs_size(); i++) {
                const pb::ComponentRun& m = req->runs(i);
                runs[i].component = static_cast<ComponentType>(m.component());
                runs[i].end = m.end();
                runs[i].slots.assign(m.slots().begin(), m.slots().end());
            }
            InsertPathResult r = svc->insert_path(
                req->hashes().data(), req->hashes_size(), req->start(),
                req->full_slots_size(), req->full_slots().data(), runs, req->auto_recycle());
            if (r.remote_status != RemoteStatus::OK)
                return to_grpc_status(r.remote_status);
            auto put = [](const InsertResult& in, pb::InsertResponse* o) {
                o->set_error(static_cast<pb::InsertError>(in.error));
                o->mutable_unused_slots()->Add(in.unused_slots.begin(), in.unused_slots.end());
            };
            put(r.full, rep->mutable_full());
            for (const InsertResult& x : r.runs) put(x, rep->add_runs());
            return grpc::Status::OK;
        });
    }

    grpc::Status Recycle(grpc::ServerContext* ctx, const pb::RecycleRequest* req,
                         pb::RecycleResponse*) override {
        return run(ctx, [&] {
            RemoteStatus st = svc->recycle(req->slots().data(),
                                           req->slots_size(),
                                           static_cast<ComponentType>(req->component()));
            return st == RemoteStatus::OK ? grpc::Status::OK
                                          : to_grpc_status(st);
        });
    }

    grpc::Status Unlock(grpc::ServerContext* ctx, const pb::UnlockRequest* req,
                        pb::UnlockResponse*) override {
        return run(ctx, [&] {
            RemoteStatus st = svc->unlock(req->hashes().data(),
                                          req->hashes_size(), req->start(),
                                          req->length(), req->pin_end(),
                                          req->pin_tail().data(),
                                          req->pin_tail_size());
            return st == RemoteStatus::OK ? grpc::Status::OK
                                          : to_grpc_status(st);
        });
    }
};

GrpcRemoteOpServer::GrpcRemoteOpServer(uint16_t listen_port, uint64_t instance_id)
    : impl_(std::make_unique<Impl>()) {
    impl_->instance = std::to_string(instance_id);

    grpc::ServerBuilder builder;
    builder.AddListeningPort("0.0.0.0:" + std::to_string(listen_port),
                             grpc::InsecureServerCredentials(), &impl_->port);
    builder.RegisterService(impl_.get());
    builder.SetMaxReceiveMessageSize(kGrpcRemoteOpMaxMessageBytes);
    builder.SetMaxSendMessageSize(kGrpcRemoteOpMaxMessageBytes);
    builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
    builder.AddChannelArgument(GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS,
                               kGrpcRemoteOpKeepaliveMs / 2);
    // A second server on a taken port must fail, not share it.
    builder.AddChannelArgument(GRPC_ARG_ALLOW_REUSEPORT, 0);
    // Four CQs with 4-8 pollers each cut per-call thread handoffs (measured: 3.5x query throughput, 40% less CPU per op).
    builder.SetSyncServerOption(grpc::ServerBuilder::NUM_CQS, kNumCqs);
    builder.SetSyncServerOption(grpc::ServerBuilder::MIN_POLLERS, kMinPollers);
    builder.SetSyncServerOption(grpc::ServerBuilder::MAX_POLLERS, kMaxPollers);
    grpc::ResourceQuota quota("radixshmem-remote-op");
    quota.SetMaxThreads(remote_op_max_threads());
    builder.SetResourceQuota(quota);
    impl_->server = builder.BuildAndStart();
    if (!impl_->server || impl_->port == 0)
        throw std::runtime_error("remote-op: gRPC server failed to bind port " +
                                 std::to_string(listen_port));
}

GrpcRemoteOpServer::~GrpcRemoteOpServer() { stop(); }

uint16_t GrpcRemoteOpServer::port() const { return (uint16_t)impl_->port; }

void GrpcRemoteOpServer::start(RemoteOpService& svc) {
    impl_->svc = &svc;
    impl_->ready.store(true);
}

void GrpcRemoteOpServer::stop() {
    std::call_once(impl_->stopped, [this] {
        // Close admission and drain service-holding handlers; deadline expiry cancels pending calls on the wire.
        impl_->ready.store(false);
        impl_->server->Shutdown(std::chrono::system_clock::now() +
                                std::chrono::seconds(1));
        impl_->server->Wait();
    });
}

}  // namespace radixshmem

#endif  // RADIXSHMEM_HAS_GRPC
