// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Server half of the gRPC remote-op backend: one handler per named RPC,
// guarded by a ready gate and a bounded in-flight count, calling
// RemoteOpService. Business results go in the response; request-level
// failures are gRPC status codes.
#include "grpc_remote_op.hpp"

#ifdef RADIXSHMEM_HAS_GRPC

#include "remote_op_service.hpp"
#include "remote_ops.grpc.pb.h"

#include <grpcpp/grpcpp.h>
#include <grpcpp/resource_quota.h>

#include <atomic>
#include <chrono>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>

namespace radixshmem {

namespace pb = ::radixshmem::remoteops::v1;

namespace {

constexpr int kMaxMessageBytes = 64 * 1024;
constexpr int kNumCqs     = 4;
constexpr int kMinPollers = 4;   // per CQ
constexpr int kMaxPollers = 8;   // per CQ

// A service-level reject: BAD_REQUEST is the caller's fault, the rest ours.
grpc::Status to_grpc_status(RemoteStatus st) {
    if (st == RemoteStatus::BAD_REQUEST)
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                            "remote-op: invalid arguments");
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        "remote-op: status " + std::to_string((int)st));
}

}  // namespace

struct GrpcRemoteOpServer::Impl final : pb::RemoteOps::Service {
    std::unique_ptr<grpc::Server> server;
    int port = 0;
    int max_inflight = 0;

    RemoteOpService*  svc = nullptr;
    std::atomic<bool> ready{false};
    std::atomic<int>  inflight{0};
    std::once_flag    stopped;

    // Ready gate, then the in-flight bound; run `op` (which fills the
    // response) and map a throw to INTERNAL.
    template <typename Op>
    grpc::Status run(Op&& op) {
        if (!ready.load())
            return grpc::Status(grpc::StatusCode::UNAVAILABLE,
                                "remote-op: server not serving");
        grpc::Status st;
        if (inflight.fetch_add(1) >= max_inflight) {
            st = grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED,
                              "remote-op: too many ops in flight");
        } else {
            try {
                st = op();
            } catch (const std::exception& e) {
                st = grpc::Status(grpc::StatusCode::INTERNAL, e.what());
            }
        }
        inflight.fetch_sub(1);
        return st;
    }

    grpc::Status Allocate(grpc::ServerContext*, const pb::AllocateRequest* req,
                          pb::AllocateResponse* rep) override {
        return run([&] {
            AllocateReply r = svc->allocate(req->num_blocks());
            if (r.status != RemoteStatus::OK) return to_grpc_status(r.status);
            rep->set_error(static_cast<pb::InsertError>(r.error));
            rep->mutable_slots()->Add(r.slots.begin(), r.slots.end());
            return grpc::Status::OK;
        });
    }

    grpc::Status Query(grpc::ServerContext*, const pb::QueryRequest* req,
                       pb::QueryResponse* rep) override {
        return run([&] {
            QueryReply r = svc->query(req->hashes().data(), req->hashes_size(),
                                      req->lock(), req->update_meta());
            if (r.status != RemoteStatus::OK) return to_grpc_status(r.status);
            rep->set_hit_length(r.hit_length);
            rep->mutable_slots()->Add(r.slots.begin(), r.slots.end());
            return grpc::Status::OK;
        });
    }

    grpc::Status Insert(grpc::ServerContext*, const pb::InsertRequest* req,
                        pb::InsertResponse* rep) override {
        return run([&] {
            InsertResult r = svc->insert(
                req->hashes().data(), req->hashes_size(), req->start(),
                req->slots_size(), req->slots().data(), req->auto_recycle());
            if (r.remote_status != RemoteStatus::OK)
                return to_grpc_status(r.remote_status);
            rep->set_error(static_cast<pb::InsertError>(r.error));
            rep->mutable_unused_slots()->Add(r.unused_slots.begin(),
                                             r.unused_slots.end());
            return grpc::Status::OK;
        });
    }

    grpc::Status Recycle(grpc::ServerContext*, const pb::RecycleRequest* req,
                         pb::RecycleResponse*) override {
        return run([&] {
            RemoteStatus st = svc->recycle(req->slots().data(),
                                           req->slots_size());
            return st == RemoteStatus::OK ? grpc::Status::OK
                                          : to_grpc_status(st);
        });
    }

    grpc::Status Unlock(grpc::ServerContext*, const pb::UnlockRequest* req,
                        pb::UnlockResponse*) override {
        return run([&] {
            RemoteStatus st = svc->unlock(req->hashes().data(),
                                          req->hashes_size(), req->start(),
                                          req->length());
            return st == RemoteStatus::OK ? grpc::Status::OK
                                          : to_grpc_status(st);
        });
    }
};

GrpcRemoteOpServer::GrpcRemoteOpServer(uint16_t listen_port, int max_inflight)
    : impl_(std::make_unique<Impl>()) {
    impl_->max_inflight = max_inflight > 0 ? max_inflight : 32;

    grpc::ServerBuilder builder;
    builder.AddListeningPort("0.0.0.0:" + std::to_string(listen_port),
                             grpc::InsecureServerCredentials(), &impl_->port);
    builder.RegisterService(impl_.get());
    builder.SetMaxReceiveMessageSize(kMaxMessageBytes);
    builder.SetMaxSendMessageSize(kMaxMessageBytes);
    // A second server on a taken port must fail, not share it.
    builder.AddChannelArgument(GRPC_ARG_ALLOW_REUSEPORT, 0);
    // The default single CQ with 1-2 pollers hands every call to another
    // thread; 4 CQs with 4-8 pollers each cut that cost (measured: 3.5x the
    // query throughput, 40% less CPU per op).
    builder.SetSyncServerOption(grpc::ServerBuilder::NUM_CQS, kNumCqs);
    builder.SetSyncServerOption(grpc::ServerBuilder::MIN_POLLERS, kMinPollers);
    builder.SetSyncServerOption(grpc::ServerBuilder::MAX_POLLERS, kMaxPollers);
    // Bounds the threads: every poller plus one per admitted op.
    grpc::ResourceQuota quota("radixshmem-remote-op");
    quota.SetMaxThreads(impl_->max_inflight + kNumCqs * kMaxPollers);
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
        // Close the gate; Shutdown returns only after the handlers already
        // running (they hold the service) have returned. Calls still open
        // at the deadline are cancelled on the wire.
        impl_->ready.store(false);
        impl_->server->Shutdown(std::chrono::system_clock::now() +
                                std::chrono::seconds(1));
        impl_->server->Wait();
    });
}

}  // namespace radixshmem

#endif  // RADIXSHMEM_HAS_GRPC
