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
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
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

int resolve_max_inflight(int requested) {
    if (requested > 0) return requested;
    if (const char* e = ::getenv("RADIXSHMEM_GRPC_MAX_INFLIGHT")) {
        int v = std::atoi(e);
        if (v > 0) return v;
    }
    return 32;
}

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

    RemoteOpService*        svc = nullptr;
    std::atomic<bool>       ready{false};
    std::atomic<int>        inflight{0};
    std::mutex              drain_mu;
    std::condition_variable drained;
    std::once_flag          stopped;

    // Admission: ready gate, then the in-flight bound. The gate is re-read
    // after the increment so stop() never misses a handler it must wait for.
    grpc::Status enter() {
        if (!ready.load())
            return grpc::Status(grpc::StatusCode::UNAVAILABLE,
                                "remote-op: server not serving");
        if (inflight.fetch_add(1) >= max_inflight) {
            leave();
            return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED,
                                "remote-op: too many ops in flight");
        }
        if (!ready.load()) {
            leave();
            return grpc::Status(grpc::StatusCode::UNAVAILABLE,
                                "remote-op: server not serving");
        }
        return grpc::Status::OK;
    }

    void leave() {
        if (inflight.fetch_sub(1) == 1) {
            std::lock_guard<std::mutex> g(drain_mu);
            drained.notify_all();
        }
    }

    // Admit, run `op` (which fills the response), map a throw to INTERNAL.
    template <typename Op>
    grpc::Status run(const char* name, Op&& op) {
        grpc::Status st = enter();
        if (!st.ok()) return st;
        try {
            st = op();
        } catch (const std::exception& e) {
            static std::atomic<uint64_t> thrown{0};
            uint64_t k = thrown.fetch_add(1, std::memory_order_relaxed);
            if (k < 8 || (k & 1023) == 0)
                fprintf(stderr, "[RadixServer] grpc %s handler threw "
                        "(occ %llu): %s\n", name,
                        (unsigned long long)(k + 1), e.what());
            st = grpc::Status(grpc::StatusCode::INTERNAL, e.what());
        }
        leave();
        return st;
    }

    grpc::Status Allocate(grpc::ServerContext*, const pb::AllocateRequest* req,
                          pb::AllocateResponse* rep) override {
        return run("Allocate", [&] {
            AllocateReply r = svc->allocate(req->num_blocks());
            if (r.status != RemoteStatus::OK) return to_grpc_status(r.status);
            rep->set_error(static_cast<pb::InsertError>(r.error));
            rep->mutable_slots()->Add(r.slots.begin(), r.slots.end());
            return grpc::Status::OK;
        });
    }

    grpc::Status Query(grpc::ServerContext*, const pb::QueryRequest* req,
                       pb::QueryResponse* rep) override {
        return run("Query", [&] {
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
        return run("Insert", [&] {
            InsertResult r = svc->insert(
                req->hashes().data(), req->hashes_size(), req->start(),
                req->slots().data(), req->slots_size(), req->auto_recycle());
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
        return run("Recycle", [&] {
            RemoteStatus st = svc->recycle(req->slots().data(),
                                           req->slots_size());
            return st == RemoteStatus::OK ? grpc::Status::OK
                                          : to_grpc_status(st);
        });
    }

    grpc::Status Unlock(grpc::ServerContext*, const pb::UnlockRequest* req,
                        pb::UnlockResponse*) override {
        return run("Unlock", [&] {
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
    impl_->max_inflight = resolve_max_inflight(max_inflight);

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
        // Close the gate, wait out the handlers already running (they hold
        // the service), then stop the server.
        impl_->ready.store(false);
        {
            std::unique_lock<std::mutex> g(impl_->drain_mu);
            impl_->drained.wait(g, [this] { return impl_->inflight.load() == 0; });
        }
        impl_->server->Shutdown(std::chrono::system_clock::now() +
                                std::chrono::seconds(1));
        impl_->server->Wait();
    });
}

}  // namespace radixshmem

#endif  // RADIXSHMEM_HAS_GRPC
