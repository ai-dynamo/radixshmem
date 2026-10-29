// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Loopback gRPC tests on a real tree: five ops, errors, deadlines, admission, lifecycle and replies; no RDMA/etcd.

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

// Single-node tree/service fixture; holding insert_mu parks allocate/insert handlers.
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

// ---- the five ops end to end, business vs request-level errors ----
void test_roundtrip() {
    TEST_BEGIN("five ops round trip, MEMPOOL_FULL, BAD_REQUEST");
    Node node("rt");
    GrpcRemoteOpServer server(0);
    server.start(*node.svc);
    GrpcRemoteOpClientTransport cli(target_of(server.port()), 5000);

    // An exhausted pool is a business result, not a request error.
    AllocateReply full = cli.allocate(kMaxBlocks + 1);
    assert(full.status == RemoteStatus::OK);
    assert(full.error == InsertError::MEMPOOL_FULL && full.slots.empty());
    assert(node.used() == 0);

    const int n = 8;
    auto h = make_hashes(100, n);
    AllocateReply a = cli.allocate(n);
    assert(a.status == RemoteStatus::OK && a.error == InsertError::OK);
    assert((int)a.slots.size() == n && node.used() == (uint32_t)n);

    InsertResult ins = cli.insert(h.data(), n, 0, n, a.slots.data(), true);
    assert(ins.error == InsertError::OK && ins.unused_slots.empty());

    QueryReply q = cli.query(h.data(), n, /*lock=*/true, /*update_meta=*/true);
    assert(q.status == RemoteStatus::OK && q.hit_length == n && q.slots == a.slots);
    assert(cli.unlock(h.data(), n, 0, n) == RemoteStatus::OK);

    // Re-insert of the published prefix consumes nothing; surplus recycled.
    AllocateReply a2 = cli.allocate(n);
    InsertResult dup = cli.insert(h.data(), n, 0, n, a2.slots.data(), true);
    assert(dup.error == InsertError::OK && (int)dup.unused_slots.size() == n);

    // Abort path: recycle an unpublished allocation.
    AllocateReply a3 = cli.allocate(4);
    assert(cli.recycle(a3.slots.data(), 4) == RemoteStatus::OK);
    assert(node.used() == (uint32_t)n);

    // Bad arguments are a request error, rejected by the server.
    assert(cli.allocate(MAX_REMOTE_OP_BLOCKS + 1).status == RemoteStatus::BAD_REQUEST);
    InsertResult bad = cli.insert(h.data(), 4, 2, 3, a.slots.data(), true);
    assert(bad.error == InsertError::REMOTE_FAILED &&
           bad.remote_status == RemoteStatus::BAD_REQUEST);
    assert(node.used() == (uint32_t)n);
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
    InsertResult r = cli.insert(h.data(), n, 0, n, a.slots.data(), true);
    assert(r.error == InsertError::REMOTE_FAILED &&
           r.remote_status == RemoteStatus::TIMEOUT);
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

// ---- lifecycle: not ready → serving → stop with an op in flight → gone ----
void test_lifecycle() {
    TEST_BEGIN("ready gate, stop with an op in flight, repeat stop");
    Node node("life");
    auto server = std::make_unique<GrpcRemoteOpServer>(0);
    uint16_t port = server->port();
    GrpcRemoteOpClientTransport cli(target_of(port), 5000);
    auto h = make_hashes(600, 4);
    assert(cli.allocate(4).status == RemoteStatus::UNAVAILABLE);
    assert(node.used() == 0);
    server->start(*node.svc);

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

    assert(cli.query(h.data(), 4, false, false).status == RemoteStatus::UNAVAILABLE);
    server->stop();
    server.reset();
    GrpcRemoteOpClientTransport gone(target_of(port), 5000);
    assert(gone.allocate(1).status == RemoteStatus::UNAVAILABLE);
    TEST_PASS();
}

// Replies that break the protocol; Recycle stays unimplemented.
struct BadService final : pb::RemoteOps::Service {
    grpc::Status Allocate(grpc::ServerContext*, const pb::AllocateRequest*,
                          pb::AllocateResponse* rep) override {
        rep->add_slots(1);   // one slot whatever was asked
        return grpc::Status::OK;
    }
    grpc::Status Query(grpc::ServerContext*, const pb::QueryRequest*,
                       pb::QueryResponse* rep) override {
        rep->set_hit_length(3);   // fewer slots than hit_length
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
    GrpcRemoteOpClientTransport cli(target_of((uint16_t)port), 5000);

    auto h = make_hashes(700, 4);
    std::vector<int32_t> s(4, 0);
    assert(cli.allocate(4).status == RemoteStatus::BAD_REPLY);
    assert(cli.query(h.data(), 4, false, false).status == RemoteStatus::BAD_REPLY);
    assert(cli.recycle(s.data(), 4) == RemoteStatus::UNKNOWN_OP);
    server->Shutdown();
    TEST_PASS();
}

}  // namespace

int main() {
    printf("test_remote_op_grpc\n");
    test_roundtrip();
    test_deadline_no_replay();
    test_admission();
    test_lifecycle();
    test_bad_reply();
    return tests_report();
}
