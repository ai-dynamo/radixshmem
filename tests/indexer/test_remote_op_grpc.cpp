// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// test_remote_op_grpc — loopback gate for the gRPC remote-op backend: a
// GrpcRemoteOpServer over a real single-node tree, driven by a
// GrpcRemoteOpClientTransport on 127.0.0.1. Needs no RDMA device or etcd.
// Covers the five ops, business vs request-level errors, deadlines,
// admission, the ready gate, shutdown with ops in flight, and reply checks.

#include "test_utils.h"
#include "grpc_remote_op.hpp"
#include "remote_op_service.hpp"
#include "remote_ops.grpc.pb.h"
#include "local.h"

#include <grpcpp/grpcpp.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using namespace radixshmem;
namespace pb = ::radixshmem::remoteops::v1;

namespace {

constexpr int kMaxBlocks = 64;

std::string target_of(uint16_t port) {
    return "127.0.0.1:" + std::to_string(port);
}

std::vector<uint64_t> make_hashes(int64_t base, int n) {
    std::vector<int64_t> tok(n);
    std::iota(tok.begin(), tok.end(), base);
    return RadixClient::compute_hashes(tok.data(), n, 1);
}

// A single-node tree plus the service the gRPC server runs on it. Tests
// hold `insert_mu` to park allocate/insert handlers inside the service.
struct Node {
    std::string name;
    std::unique_ptr<RadixServer> srv;
    std::unique_ptr<LocalTreeConnector> tree;
    std::unique_ptr<RadixClient> client;
    std::mutex insert_mu;
    std::unique_ptr<RemoteOpService> svc;

    explicit Node(const char* tag)
        : name(std::string("/test_grpc_") + tag + "_" + std::to_string(::getpid())) {
        ::shm_unlink(name.c_str());
        ShmConfig cfg;
        cfg.max_blocks = kMaxBlocks;
        cfg.hugepage_path = "";
        cfg.background_evict_ratio = 0.0f;   // no async eviction under the test
        srv = std::make_unique<RadixServer>(name, cfg);
        tree = std::make_unique<LocalTreeConnector>(name);
        client = std::make_unique<RadixClient>(name);
        svc = std::make_unique<RemoteOpService>(*tree, *client, insert_mu);
    }
    ~Node() {
        svc.reset();
        client.reset();
        tree.reset();
        srv.reset();
        ::shm_unlink(name.c_str());
    }
    uint32_t used() const { return tree->mempool_used(); }
};

template <typename Pred>
bool wait_until(Pred p, int timeout_ms = 5000) {
    auto end = std::chrono::steady_clock::now() +
               std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < end) {
        if (p()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return p();
}

// ---- the five ops end to end ----
void test_roundtrip() {
    TEST_BEGIN("allocate/insert/query/unlock/recycle round trip");
    Node node("rt");
    GrpcRemoteOpServer server(0);
    server.start(*node.svc);
    GrpcRemoteOpClientTransport cli(target_of(server.port()), 5000);

    const int n = 8;
    auto h = make_hashes(100, n);
    AllocateReply a = cli.allocate(n);
    assert(a.status == RemoteStatus::OK && a.error == InsertError::OK);
    assert((int)a.slots.size() == n && node.used() == (uint32_t)n);

    InsertResult ins = cli.insert(h.data(), n, 0, n, a.slots.data(), true);
    assert(ins.error == InsertError::OK && ins.remote_status == RemoteStatus::OK);
    assert(ins.unused_slots.empty());

    QueryReply q = cli.query(h.data(), n, /*lock=*/true, /*update_meta=*/true);
    assert(q.status == RemoteStatus::OK && q.hit_length == n);
    assert(q.slots == a.slots);
    assert(cli.unlock(h.data(), n, 0, n) == RemoteStatus::OK);

    // Re-insert of the published prefix consumes nothing; surplus recycled.
    AllocateReply a2 = cli.allocate(n);
    assert(a2.status == RemoteStatus::OK && (int)a2.slots.size() == n);
    InsertResult dup = cli.insert(h.data(), n, 0, n, a2.slots.data(), true);
    assert(dup.error == InsertError::OK && (int)dup.unused_slots.size() == n);
    assert(node.used() == (uint32_t)n);

    // Abort path: recycle an unpublished allocation.
    AllocateReply a3 = cli.allocate(4);
    assert(a3.status == RemoteStatus::OK && node.used() == (uint32_t)n + 4);
    assert(cli.recycle(a3.slots.data(), 4) == RemoteStatus::OK);
    assert(node.used() == (uint32_t)n);
    server.stop();
    TEST_PASS();
}

// ---- a business failure is an OK call; bad arguments are request-level ----
void test_business_vs_request_errors() {
    TEST_BEGIN("MEMPOOL_FULL vs BAD_REQUEST");
    Node node("err");
    GrpcRemoteOpServer server(0);
    server.start(*node.svc);
    GrpcRemoteOpClientTransport cli(target_of(server.port()), 5000);

    AllocateReply full = cli.allocate(kMaxBlocks + 1);
    assert(full.status == RemoteStatus::OK);
    assert(full.error == InsertError::MEMPOOL_FULL && full.slots.empty());
    assert(node.used() == 0);

    // Rejected before the wire.
    auto h = make_hashes(200, 4);
    std::vector<int32_t> s(4, 0);
    assert(cli.allocate(MAX_REMOTE_OP_BLOCKS + 1).status == RemoteStatus::BAD_REQUEST);
    InsertResult bad = cli.insert(h.data(), 4, 2, 3, s.data(), true);
    assert(bad.error == InsertError::REMOTE_FAILED &&
           bad.remote_status == RemoteStatus::BAD_REQUEST);
    assert(cli.unlock(h.data(), 4, 3, 2) == RemoteStatus::BAD_REQUEST);

    // Rejected by the server (raw stub bypasses the client checks).
    auto stub = pb::RemoteOps::NewStub(grpc::CreateChannel(
        target_of(server.port()), grpc::InsecureChannelCredentials()));
    {
        grpc::ClientContext ctx;
        pb::AllocateRequest req; pb::AllocateResponse rep;
        req.set_num_blocks(-1);
        assert(stub->Allocate(&ctx, req, &rep).error_code() ==
               grpc::StatusCode::INVALID_ARGUMENT);
    }
    {
        grpc::ClientContext ctx;
        pb::InsertRequest req; pb::InsertResponse rep;
        req.mutable_hashes()->Add(h.begin(), h.end());
        req.mutable_slots()->Add(s.begin(), s.begin() + 3);
        req.set_start(2);   // 2 + 3 > 4
        assert(stub->Insert(&ctx, req, &rep).error_code() ==
               grpc::StatusCode::INVALID_ARGUMENT);
    }
    {
        grpc::ClientContext ctx;
        pb::QueryRequest req; pb::QueryResponse rep;
        for (int i = 0; i <= MAX_REMOTE_OP_BLOCKS; ++i) req.add_hashes(i);
        assert(stub->Query(&ctx, req, &rep).error_code() ==
               grpc::StatusCode::INVALID_ARGUMENT);
    }
    assert(node.used() == 0);
    server.stop();
    TEST_PASS();
}

// ---- ready gate: bound but not serving → UNAVAILABLE, then OK ----
void test_not_ready() {
    TEST_BEGIN("ops before start() fail UNAVAILABLE");
    Node node("ready");
    GrpcRemoteOpServer server(0);
    GrpcRemoteOpClientTransport cli(target_of(server.port()), 5000);
    auto h = make_hashes(300, 4);
    assert(cli.query(h.data(), 4, false, false).status == RemoteStatus::UNAVAILABLE);
    assert(cli.allocate(4).status == RemoteStatus::UNAVAILABLE);
    assert(node.used() == 0);
    server.start(*node.svc);
    assert(cli.query(h.data(), 4, false, false).status == RemoteStatus::OK);
    server.stop();
    TEST_PASS();
}

// ---- deadline: TIMEOUT, the op still runs once, the client never replays ----
void test_deadline_no_replay() {
    TEST_BEGIN("deadline → TIMEOUT, op runs exactly once");
    Node node("dl");
    GrpcRemoteOpServer server(0);
    server.start(*node.svc);
    GrpcRemoteOpClientTransport cli(target_of(server.port()), 200);

    const int n = 8;
    auto h = make_hashes(400, n);
    AllocateReply a = cli.allocate(n);
    assert(a.status == RemoteStatus::OK);

    std::unique_lock<std::mutex> park(node.insert_mu);
    auto t0 = std::chrono::steady_clock::now();
    InsertResult r = cli.insert(h.data(), n, 0, n, a.slots.data(), true);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - t0).count();
    assert(r.error == InsertError::REMOTE_FAILED &&
           r.remote_status == RemoteStatus::TIMEOUT);
    assert(ms < 2000);
    // The handler is still parked; releasing it lands the insert anyway.
    park.unlock();
    assert(wait_until([&] {
        return cli.query(h.data(), n, false, false).hit_length == n;
    }));
    // Exactly the allocated slots are in use: nothing replayed or leaked.
    assert(node.used() == (uint32_t)n);
    server.stop();
    TEST_PASS();
}

// ---- admission: a full server answers RESOURCE_EXHAUSTED ----
void test_admission() {
    TEST_BEGIN("in-flight bound → RESOURCE_EXHAUSTED");
    Node node("adm");
    GrpcRemoteOpServer server(0, /*max_inflight=*/1);
    server.start(*node.svc);
    GrpcRemoteOpClientTransport cli(target_of(server.port()), 5000);

    std::unique_lock<std::mutex> park(node.insert_mu);
    AllocateReply parked;
    std::thread t([&] { parked = cli.allocate(4); });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    auto h = make_hashes(500, 4);
    assert(cli.query(h.data(), 4, false, false).status ==
           RemoteStatus::RESOURCE_EXHAUSTED);
    park.unlock();
    t.join();
    assert(parked.status == RemoteStatus::OK && parked.slots.size() == 4);
    assert(cli.query(h.data(), 4, false, false).status == RemoteStatus::OK);
    server.stop();
    TEST_PASS();
}

// ---- stop() waits for running handlers, then refuses; idempotent ----
void test_stop_with_inflight() {
    TEST_BEGIN("stop drains in-flight ops, repeat stop is a no-op");
    Node node("stop");
    auto server = std::make_unique<GrpcRemoteOpServer>(0);
    server->start(*node.svc);
    uint16_t port = server->port();
    GrpcRemoteOpClientTransport cli(target_of(port), 5000);

    std::unique_lock<std::mutex> park(node.insert_mu);
    AllocateReply inflight;
    std::thread op([&] { inflight = cli.allocate(4); });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::atomic<bool> stopped{false};
    std::thread stopper([&] { server->stop(); stopped = true; });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    assert(!stopped.load());     // still waiting for the parked handler
    park.unlock();
    stopper.join();
    op.join();
    assert(inflight.status == RemoteStatus::OK && inflight.slots.size() == 4);

    auto h = make_hashes(600, 4);
    assert(cli.query(h.data(), 4, false, false).status == RemoteStatus::UNAVAILABLE);
    server->stop();
    server.reset();
    // Nothing listens on the port any more.
    GrpcRemoteOpClientTransport gone(target_of(port), 5000);
    assert(gone.allocate(1).status == RemoteStatus::UNAVAILABLE);
    TEST_PASS();
}

// ---- a taken port fails the bind instead of being shared ----
void test_bind_conflict() {
    TEST_BEGIN("bind conflict throws");
    GrpcRemoteOpServer first(0);
    bool threw = false;
    try {
        GrpcRemoteOpServer second(first.port());
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
    TEST_PASS();
}

// Replies that break the protocol; Recycle stays unimplemented.
struct BadService final : pb::RemoteOps::Service {
    grpc::Status Allocate(grpc::ServerContext*, const pb::AllocateRequest*,
                          pb::AllocateResponse* rep) override {
        rep->add_slots(1);   // OK with one slot whatever was asked
        return grpc::Status::OK;
    }
    grpc::Status Query(grpc::ServerContext*, const pb::QueryRequest*,
                       pb::QueryResponse* rep) override {
        rep->set_hit_length(3);
        rep->add_slots(1);   // fewer slots than hit_length
        return grpc::Status::OK;
    }
    grpc::Status Insert(grpc::ServerContext*, const pb::InsertRequest*,
                        pb::InsertResponse* rep) override {
        rep->set_error(static_cast<pb::InsertError>(99));
        return grpc::Status::OK;
    }
};

void test_bad_reply() {
    TEST_BEGIN("malformed replies → BAD_REPLY, unknown RPC → UNKNOWN_OP");
    BadService svc;
    int port = 0;
    grpc::ServerBuilder b;
    b.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    b.RegisterService(&svc);
    auto server = b.BuildAndStart();
    assert(server && port > 0);
    GrpcRemoteOpClientTransport cli(target_of((uint16_t)port), 5000);

    auto h = make_hashes(700, 4);
    std::vector<int32_t> s(4, 0);
    assert(cli.allocate(4).status == RemoteStatus::BAD_REPLY);
    assert(cli.query(h.data(), 4, false, false).status == RemoteStatus::BAD_REPLY);
    InsertResult r = cli.insert(h.data(), 4, 0, 4, s.data(), true);
    assert(r.error == InsertError::REMOTE_FAILED &&
           r.remote_status == RemoteStatus::BAD_REPLY);
    assert(cli.recycle(s.data(), 4) == RemoteStatus::UNKNOWN_OP);
    server->Shutdown();
    TEST_PASS();
}

// ---- one channel, many ops in flight ----
void test_concurrent_ops() {
    TEST_BEGIN("concurrent ops over one client transport");
    Node node("conc");
    GrpcRemoteOpServer server(0);
    server.start(*node.svc);
    GrpcRemoteOpClientTransport cli(target_of(server.port()), 5000);

    const int n = 8;
    auto h = make_hashes(800, n);
    AllocateReply a = cli.allocate(n);
    assert(cli.insert(h.data(), n, 0, n, a.slots.data(), true).error ==
           InsertError::OK);

    std::atomic<int> ok{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < 8; ++t)
        ts.emplace_back([&] {
            for (int i = 0; i < 50; ++i) {
                QueryReply q = cli.query(h.data(), n, true, false);
                if (q.status == RemoteStatus::OK && q.hit_length == n &&
                    cli.unlock(h.data(), n, 0, n) == RemoteStatus::OK)
                    ++ok;
            }
        });
    for (auto& t : ts) t.join();
    assert(ok.load() == 8 * 50);
    expect_healthy(*node.client);   // every pin was released
    server.stop();
    TEST_PASS();
}

}  // namespace

int main() {
    printf("test_remote_op_grpc\n");
    test_roundtrip();
    test_business_vs_request_errors();
    test_not_ready();
    test_deadline_no_replay();
    test_admission();
    test_stop_with_inflight();
    test_bind_conflict();
    test_bad_reply();
    test_concurrent_ops();
    return tests_report();
}
