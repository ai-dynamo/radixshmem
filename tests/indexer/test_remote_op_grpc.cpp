// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Loopback gRPC tests on a real tree: five ops, errors, deadlines, lifecycle and replies; no RDMA/etcd.

#include "test_utils.h"
#include "grpc_remote_op.hpp"
#include "remote_op_service.hpp"
#include "remote_op_factory.hpp"
#include "remote_ops.grpc.pb.h"
#include "local.h"

#include <grpcpp/grpcpp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace radixshmem;
namespace pb = ::radixshmem::remoteops::v1;

namespace {

constexpr int kMaxBlocks = 64;
constexpr uint64_t kInstance = 42;

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

    explicit Node(const char* tag, int max_blocks = kMaxBlocks)
        : name(std::string("/test_grpc_") + tag + "_" + std::to_string(::getpid())) {
        ::shm_unlink(name.c_str());
        ShmConfig cfg;
        cfg.max_blocks = max_blocks;
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

void test_transport_factory() {
    TEST_BEGIN("transport factory serves gRPC and rejects an entry without a port");
    Node node("factory");
    RemoteOpAddr address;
    auto server = make_remote_op_server(0, kInstance, address);
    assert(address.port != 0);
    server->start(*node.svc);
    PeerEntry peer{};
    peer.rank = 1;
    std::strcpy(peer.host, "127.0.0.1");
    peer.remote_op_listen_port = address.port;
    peer.instance_id = kInstance;
    std::string endpoint;
    auto client = make_remote_op_client(peer, endpoint);
    assert(endpoint == "grpc://" + target_of(address.port));
    auto allocated = client->allocate(2);
    assert(allocated.status == RemoteStatus::OK && allocated.slots.size() == 2);
    assert(client->recycle(allocated.slots.data(), 2) == RemoteStatus::OK);
    assert(node.used() == 0);
    peer.remote_op_listen_port = 0;
    bool rejected = false;
    try {
        make_remote_op_client(peer, endpoint);
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("no remote-op port") != std::string::npos;
    }
    assert(rejected);
    server->stop();
    expect_healthy(*node.client);
    TEST_PASS();
}

// ---- the five ops end to end, business vs request-level errors ----
void test_roundtrip() {
    TEST_BEGIN("five ops round trip, MEMPOOL_FULL, BAD_REQUEST");
    Node node("rt");
    GrpcRemoteOpServer server(0, kInstance);
    server.start(*node.svc);
    GrpcRemoteOpClientTransport cli(target_of(server.port()), 5000, kInstance);

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
    assert(cli.unlock(h.data(), n, 0, q.hit_length, q.pin_end, q.pin_tail) ==
           RemoteStatus::OK);

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
    const auto bogus = static_cast<ComponentType>(NUM_COMPONENT_TYPES);
    assert(cli.allocate(1, bogus).status == RemoteStatus::BAD_REQUEST);
    assert(cli.insert(h.data(), n, 0, 1, a.slots.data(), true, bogus).remote_status ==
           RemoteStatus::BAD_REQUEST);
    assert(cli.recycle(a.slots.data(), 1, bogus) == RemoteStatus::BAD_REQUEST);
    assert(node.used() == (uint32_t)n);
    server.stop();
    TEST_PASS();
}

void test_largest_request() {
    TEST_BEGIN("ops of MAX_REMOTE_OP_BLOCKS blocks fit the message limit");
    Node node("max", MAX_REMOTE_OP_BLOCKS);
    GrpcRemoteOpServer server(0, kInstance);
    server.start(*node.svc);
    GrpcRemoteOpClientTransport cli(target_of(server.port()), 5000, kInstance);
    const int n = MAX_REMOTE_OP_BLOCKS;
    auto h = make_hashes(100000, n);
    AllocateReply a = cli.allocate(n);
    assert(a.status == RemoteStatus::OK && (int)a.slots.size() == n);
    InsertResult ins = cli.insert(h.data(), n, 0, n, a.slots.data(), true);
    assert(ins.remote_status == RemoteStatus::OK && ins.error == InsertError::OK);
    QueryReply q = cli.query(h.data(), n, true, false);
    assert(q.status == RemoteStatus::OK && q.hit_length == n && q.slots == a.slots);
    assert(cli.unlock(h.data(), n, 0, q.hit_length, q.pin_end, q.pin_tail) == RemoteStatus::OK);
    assert(node.used() == (uint32_t)n);
    server.stop();
    expect_healthy(*node.client);
    TEST_PASS();
}

// ---- deadline: TIMEOUT, the op still runs once, the client never replays ----
void test_deadline_no_replay() {
    TEST_BEGIN("deadline → TIMEOUT, op runs exactly once");
    Node node("dl");
    GrpcRemoteOpServer server(0, kInstance);
    server.start(*node.svc);
    GrpcRemoteOpClientTransport cli(target_of(server.port()), 200, kInstance);

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

void test_no_admission_cap() {
    TEST_BEGIN("40 parked allocates; query/unlock/recycle still served");
    Node node("cap");
    GrpcRemoteOpServer server(0, kInstance);
    server.start(*node.svc);
    GrpcRemoteOpClientTransport cli(target_of(server.port()), 5000, kInstance);

    const int n = 8;
    auto h = make_hashes(500, n);
    AllocateReply a = cli.allocate(n);
    assert(cli.insert(h.data(), n, 0, n, a.slots.data(), true).error == InsertError::OK);
    AllocateReply spare = cli.allocate(2);
    assert(spare.status == RemoteStatus::OK);

    std::unique_lock<std::mutex> park(node.insert_mu);
    std::vector<AllocateReply> parked(40);
    std::vector<std::thread> threads;
    for (auto& r : parked) threads.emplace_back([&cli, &r] { r = cli.allocate(1); });
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    QueryReply q = cli.query(h.data(), n, true, false);
    assert(q.status == RemoteStatus::OK && q.hit_length == n);
    assert(cli.unlock(h.data(), n, 0, q.hit_length, q.pin_end, q.pin_tail) ==
           RemoteStatus::OK);
    assert(cli.recycle(spare.slots.data(), 2) == RemoteStatus::OK);
    park.unlock();
    for (auto& t : threads) t.join();
    for (auto& r : parked) {
        assert(r.status == RemoteStatus::OK && r.slots.size() == 1);
        assert(cli.recycle(r.slots.data(), 1) == RemoteStatus::OK);
    }
    assert(node.used() == (uint32_t)n);
    server.stop();
    expect_healthy(*node.client);
    TEST_PASS();
}

void test_deadline_skips_allocate_and_query() {
    TEST_BEGIN("past the deadline → allocate/query skipped, nothing held");
    Node node("skip");
    auto past = std::chrono::system_clock::now();
    assert(node.svc->allocate(4, past).status == RemoteStatus::TIMEOUT);
    assert(node.used() == 0);

    GrpcRemoteOpServer server(0, kInstance);
    server.start(*node.svc);
    GrpcRemoteOpClientTransport cli(target_of(server.port()), 5000, kInstance);
    const int n = 8;
    auto h = make_hashes(550, n);
    AllocateReply a = cli.allocate(n);
    assert(cli.insert(h.data(), n, 0, n, a.slots.data(), true).error == InsertError::OK);
    QueryReply late = node.svc->query(h.data(), n, true, false, past);
    assert(late.status == RemoteStatus::TIMEOUT && late.hit_length == 0);
    expect_healthy(*node.client);

    GrpcRemoteOpClientTransport impatient(target_of(server.port()), 200, kInstance);
    {
        std::unique_lock<std::mutex> park(node.insert_mu);
        assert(impatient.allocate(4).status == RemoteStatus::TIMEOUT);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    assert(node.used() == (uint32_t)n);
    server.stop();
    TEST_PASS();
}

void test_unlock_partial_node() {
    TEST_BEGIN("query(lock) ending inside a node; unlock with its tail");
    Node node("tail");
    GrpcRemoteOpServer server(0, kInstance);
    server.start(*node.svc);
    GrpcRemoteOpClientTransport cli(target_of(server.port()), 5000, kInstance);

    const int n = 8;
    auto h = make_hashes(800, n);
    AllocateReply a = cli.allocate(n);
    assert(cli.insert(h.data(), n, 0, n, a.slots.data(), true).error == InsertError::OK);

    QueryReply q = cli.query(h.data(), 4, true, false);
    assert(q.status == RemoteStatus::OK && q.hit_length == 4);
    assert(q.pin_end == n && q.pin_tail.size() == 4);
    assert(std::equal(q.pin_tail.begin(), q.pin_tail.end(), h.begin() + 4));
    assert(cli.unlock(h.data(), 4, 0, q.hit_length, q.pin_end, q.pin_tail) ==
           RemoteStatus::OK);
    expect_healthy(*node.client);
    server.stop();
    TEST_PASS();
}

void test_stale_instance() {
    TEST_BEGIN("another instance id → STALE_TARGET, nothing changes");
    Node node("stale");
    GrpcRemoteOpServer server(0, kInstance);
    server.start(*node.svc);
    GrpcRemoteOpClientTransport cli(target_of(server.port()), 5000, kInstance + 1);

    const int n = 4;
    auto h = make_hashes(900, n);
    std::vector<int32_t> s(n, 0);
    assert(cli.allocate(n).status == RemoteStatus::STALE_TARGET);
    assert(cli.query(h.data(), n, true, false).status == RemoteStatus::STALE_TARGET);
    InsertResult ins = cli.insert(h.data(), n, 0, n, s.data(), true);
    assert(ins.error == InsertError::REMOTE_FAILED &&
           ins.remote_status == RemoteStatus::STALE_TARGET);
    assert(cli.recycle(s.data(), n) == RemoteStatus::STALE_TARGET);
    assert(cli.unlock(h.data(), n, 0, n, n, {}) == RemoteStatus::STALE_TARGET);
    assert(node.used() == 0);
    server.stop();
    expect_healthy(*node.client);
    TEST_PASS();
}

void test_local_limits() {
    TEST_BEGIN("oversized requests → BAD_REQUEST without a server");
    GrpcRemoteOpClientTransport cli("127.0.0.1:1", 1000, kInstance);
    const int big = 9000;
    auto h = make_hashes(1000, big);
    std::vector<int32_t> s(big, 0);
    assert(cli.allocate(MAX_REMOTE_OP_BLOCKS + 1).status == RemoteStatus::BAD_REQUEST);
    assert(cli.query(h.data(), big, false, false).status == RemoteStatus::BAD_REQUEST);
    assert(cli.query(h.data(), MAX_REMOTE_OP_BLOCKS + 1, false, false).status ==
           RemoteStatus::BAD_REQUEST);
    assert(cli.insert(h.data(), big, 0, big, s.data(), true).remote_status ==
           RemoteStatus::BAD_REQUEST);
    assert(cli.recycle(s.data(), big) == RemoteStatus::BAD_REQUEST);
    std::vector<uint64_t> tail(h.begin(), h.begin() + MAX_REMOTE_OP_BLOCKS + 1);
    assert(cli.unlock(h.data(), 4, 0, 4, 4 + (int)tail.size(), tail) ==
           RemoteStatus::BAD_REQUEST);
    TEST_PASS();
}

// ---- lifecycle: not ready → serving → stop with an op in flight → gone ----
void test_lifecycle() {
    TEST_BEGIN("ready gate, stop with an op in flight, repeat stop");
    Node node("life");
    auto server = std::make_unique<GrpcRemoteOpServer>(0, kInstance);
    uint16_t port = server->port();
    GrpcRemoteOpClientTransport cli(target_of(port), 5000, kInstance);
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
    GrpcRemoteOpClientTransport gone(target_of(port), 5000, kInstance);
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
    GrpcRemoteOpClientTransport cli(target_of((uint16_t)port), 5000, kInstance);

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
    test_transport_factory();
    test_roundtrip();
    test_largest_request();
    test_deadline_no_replay();
    test_no_admission_cap();
    test_deadline_skips_allocate_and_query();
    test_unlock_partial_node();
    test_stale_instance();
    test_local_limits();
    test_lifecycle();
    test_bad_reply();
    return tests_report();
}
