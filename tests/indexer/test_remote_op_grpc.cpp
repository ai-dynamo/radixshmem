// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Loopback gRPC tests on a real tree: five ops, errors, deadlines, lifecycle and replies,
// the peer index over them, and the XRC setup RPC; no RDMA/etcd.

#include "test_utils.h"
#include "grpc_remote_op.hpp"
#include "grpc_peer_index.hpp"
#include "remote_op_service.hpp"
#include "remote_op_factory.hpp"
#include "remote_ops.grpc.pb.h"
#include "local.h"
#include "peer_breaker.hpp"
#include "peer_registry.hpp"

#include <grpcpp/grpcpp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
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

RemoteQuery rq(bool lock, bool update_meta = false) {
    RemoteQuery q;
    q.lock = lock;
    q.update_meta = update_meta;
    return q;
}

PinRecord full_pin(int end) {
    PinRecord p;
    p.full = {0, (uint32_t)end};
    return p;
}

// Single-node tree/service fixture; holding insert_mu parks allocate/insert handlers.
struct Node {
    std::string name;
    std::unique_ptr<RadixServer> srv;
    std::unique_ptr<LocalTreeConnector> tree;
    std::unique_ptr<RadixClient> client;
    std::mutex insert_mu;
    std::unique_ptr<RemoteOpService> svc;

    explicit Node(const char* tag, int max_blocks = kMaxBlocks, bool aux = false)
        : name(std::string("/test_grpc_") + tag + "_" + std::to_string(::getpid())) {
        ::shm_unlink(name.c_str());
        ShmConfig cfg;
        cfg.max_blocks = max_blocks;
        cfg.hugepage_path = "";
        cfg.background_evict_ratio = 0.0f;   // no async eviction under the test
        if (aux) {
            cfg.register_chunk_tokens = 4;
            cfg.component_mask = COMPONENT_MASK_ALL;
            cfg.swa_window_blocks = 4;
            cfg.mamba_state_bytes = 256;
            cfg.swa_max_blocks = max_blocks;
            cfg.mamba_max_slots = 16;
        }
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
    auto server = make_remote_op_server(0, kInstance, {}, address);
    assert(address.port != 0);
    server->start(*node.svc);
    PeerEntry peer{};
    peer.rank = 1;
    std::strcpy(peer.host, "127.0.0.1");
    peer.rpc_port = address.port;
    peer.instance_id = kInstance;
    std::string endpoint;
    auto client = make_remote_op_client(peer, endpoint);
    assert(endpoint == "grpc://" + target_of(address.port));
    auto allocated = client->allocate(2);
    assert(allocated.status == RemoteStatus::OK && allocated.slots.size() == 2);
    assert(client->recycle(allocated.slots.data(), 2) == RemoteStatus::OK);
    assert(node.used() == 0);
    peer.rpc_port = 0;
    bool rejected = false;
    try {
        make_remote_op_client(peer, endpoint);
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("no RPC port") != std::string::npos;
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

    QueryReply q = cli.query(h.data(), n, rq(true, true));
    assert(q.status == RemoteStatus::OK && q.hit_length == n && q.slots == a.slots);
    assert(cli.unlock(h.data(), n, q.pin) ==
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
    assert(cli.connected());
    server.stop();
    TEST_PASS();
}

void test_large_request() {
    TEST_BEGIN("ops past gRPC's default 4 MiB message limit round trip");
    const int n = 1 << 19;
    Node node("large", n);
    GrpcRemoteOpServer server(0, kInstance);
    server.start(*node.svc);
    GrpcRemoteOpClientTransport cli(target_of(server.port()), 5000, kInstance);
    auto h = make_hashes(100000, n);
    AllocateReply a = cli.allocate(n);
    assert(a.status == RemoteStatus::OK && (int)a.slots.size() == n);
    InsertResult ins = cli.insert(h.data(), n, 0, n, a.slots.data(), true);
    assert(ins.remote_status == RemoteStatus::OK && ins.error == InsertError::OK);
    QueryReply q = cli.query(h.data(), n, rq(true));
    assert(q.status == RemoteStatus::OK && q.hit_length == n && q.slots == a.slots);
    assert(cli.unlock(h.data(), n, q.pin) == RemoteStatus::OK);
    assert(node.used() == (uint32_t)n);
    server.stop();
    expect_healthy(*node.client);
    TEST_PASS();
}

std::vector<int32_t> reserve(GrpcRemoteOpClientTransport& cli, int n, ComponentType kind) {
    AllocateReply a = cli.allocate(n, kind);
    assert(a.status == RemoteStatus::OK && a.error == InsertError::OK && (int)a.slots.size() == n);
    return a.slots;
}

std::vector<ComponentRun> path_runs(GrpcRemoteOpClientTransport& cli) {
    auto swa = reserve(cli, 6, ComponentType::SWA);
    auto mamba = reserve(cli, 2, ComponentType::MAMBA);
    return {{ComponentType::SWA, 6, {swa.begin(), swa.begin() + 4}},
            {ComponentType::SWA, 8, {swa.begin() + 4, swa.end()}},
            {ComponentType::MAMBA, 6, {mamba[0]}},
            {ComponentType::MAMBA, 8, {mamba[1]}}};
}

void test_insert_path() {
    TEST_BEGIN("InsertPath publishes FULL and every SWA run and checkpoint in one request");
    Node node("path", kMaxBlocks, true);
    GrpcRemoteOpServer server(0, kInstance);
    server.start(*node.svc);
    GrpcRemoteOpClientTransport cli(target_of(server.port()), 5000, kInstance);
    const int n = 8;
    auto h = make_hashes(700, n);

    auto full = reserve(cli, n, ComponentType::FULL);
    auto runs = path_runs(cli);
    InsertPathResult r = cli.insert_path(h.data(), n, 0, n, full.data(), runs, true);
    assert(r.remote_status == RemoteStatus::OK && r.full.error == InsertError::OK &&
           r.full.unused_slots.empty() && r.runs.size() == runs.size());
    for (const InsertResult& x : r.runs)
        assert(x.error == InsertError::OK && x.unused_slots.empty());
    for (int end : {6, 8})
        assert((int)node.client->query(h.data(), end, COMPONENT_MASK_ALL, true, false, false)
                   .common_hit == end);

    auto again = reserve(cli, n, ComponentType::FULL);
    auto runs2 = path_runs(cli);
    r = cli.insert_path(h.data(), n, 0, n, again.data(), runs2, true);
    assert(r.remote_status == RemoteStatus::OK && (int)r.full.unused_slots.size() == n);
    for (size_t i = 0; i < runs2.size(); i++)
        assert(r.runs[i].error == InsertError::OK &&
               r.runs[i].unused_slots.size() == runs2[i].slots.size());
    assert(node.used() == (uint32_t)n);

    auto g = make_hashes(900, 4);
    auto orphan = reserve(cli, 2, ComponentType::SWA);
    r = cli.insert_path(g.data(), 4, 0, 0, nullptr, {{ComponentType::SWA, 4, orphan}}, true);
    assert(r.remote_status == RemoteStatus::OK &&
           r.runs[0].error == InsertError::FULL_PATH_MISSING && r.runs[0].unused_slots.size() == 2);

    server.stop();
    expect_healthy(*node.client);
    TEST_PASS();
}

void test_insert_path_rejects_bad_runs() {
    TEST_BEGIN("InsertPath rejects a malformed request and changes nothing");
    Node node("pathbad", kMaxBlocks, true);
    GrpcRemoteOpServer server(0, kInstance);
    server.start(*node.svc);
    GrpcRemoteOpClientTransport cli(target_of(server.port()), 5000, kInstance);
    const int n = 8;
    auto h = make_hashes(800, n);
    std::vector<int32_t> full(n, 0), one{1}, two{1, 2};
    const auto bogus = static_cast<ComponentType>(NUM_COMPONENT_TYPES);
    auto rejected = [&](std::vector<ComponentRun> runs, int start, int length) {
        InsertPathResult r = cli.insert_path(h.data(), n, start, length, full.data(), runs, true);
        bool all_failed = r.full.error == InsertError::REMOTE_FAILED && r.runs.size() == runs.size();
        for (const InsertResult& x : r.runs)
            all_failed = all_failed && x.error == InsertError::REMOTE_FAILED &&
                         x.remote_status == RemoteStatus::BAD_REQUEST;
        return r.remote_status == RemoteStatus::BAD_REQUEST && all_failed;
    };
    assert(rejected({{ComponentType::FULL, 4, one}}, 0, 0));
    assert(rejected({{bogus, 4, one}}, 0, 0));
    assert(rejected({{ComponentType::SWA, 0, one}}, 0, 0));
    assert(rejected({{ComponentType::SWA, n + 1, one}}, 0, 0));
    assert(rejected({{ComponentType::SWA, 4, {}}}, 0, 0));
    assert(rejected({{ComponentType::SWA, 1, two}}, 0, 0));
    assert(rejected({{ComponentType::MAMBA, 4, two}}, 0, 0));
    assert(rejected({}, 4, 5));
    assert(node.used() == 0);
    server.stop();
    expect_healthy(*node.client);
    TEST_PASS();
}

using Counts = std::vector<ComponentCount>;

void test_allocate_components() {
    TEST_BEGIN("AllocateComponents reserves every pool or none");
    Node node("alloccomp", kMaxBlocks, true);
    GrpcRemoteOpServer server(0, kInstance);
    server.start(*node.svc);
    GrpcRemoteOpClientTransport cli(target_of(server.port()), 5000, kInstance);
    const auto FULL = ComponentType::FULL, SWA = ComponentType::SWA, MAMBA = ComponentType::MAMBA;
    auto give_back = [&](const Counts& counts, const AllocateComponentsReply& r) {
        for (size_t i = 0; i < counts.size(); i++)
            if (!r.slots[i].empty())
                assert(cli.recycle(r.slots[i].data(), (int)r.slots[i].size(), counts[i].component) ==
                       RemoteStatus::OK);
    };
    auto whole = [&] {
        const Counts all = {{FULL, kMaxBlocks}, {SWA, kMaxBlocks}, {MAMBA, 16}};
        AllocateComponentsReply r = cli.allocate_components(all);
        bool ok = r.status == RemoteStatus::OK && r.error == InsertError::OK;
        if (ok) give_back(all, r);
        return ok && node.used() == 0;
    };

    const Counts counts = {{MAMBA, 2}, {FULL, 8}, {SWA, 4}};
    AllocateComponentsReply r = cli.allocate_components(counts);
    assert(r.status == RemoteStatus::OK && r.error == InsertError::OK && r.slots.size() == 3);
    assert(r.slots[0].size() == 2 && r.slots[1].size() == 8 && r.slots[2].size() == 4);
    assert(node.used() == 8);
    const int n = 8;
    auto h = make_hashes(1100, n);
    InsertPathResult path = cli.insert_path(h.data(), n, 0, n, r.slots[1].data(),
                                            {{SWA, n, r.slots[2]}, {MAMBA, 4, {r.slots[0][0]}},
                                             {MAMBA, n, {r.slots[0][1]}}}, true);
    assert(path.remote_status == RemoteStatus::OK && path.full.error == InsertError::OK);
    for (const InsertResult& x : path.runs) assert(x.error == InsertError::OK && x.unused_slots.empty());
    assert((int)node.client->query(h.data(), n, COMPONENT_MASK_ALL, true, false, false).common_hit == n);
    node.client->reset();
    assert(whole());

    r = cli.allocate_components({{FULL, 0}, {SWA, 3}});
    assert(r.status == RemoteStatus::OK && r.slots.size() == 2 && r.slots[0].empty() &&
           r.slots[1].size() == 3);
    give_back({{FULL, 0}, {SWA, 3}}, r);
    r = cli.allocate_components({});
    assert(r.status == RemoteStatus::OK && r.error == InsertError::OK && r.slots.empty());

    for (const Counts& short_one : {Counts{{FULL, 8}, {SWA, 4}, {MAMBA, 17}},
                                    Counts{{SWA, 4}, {FULL, kMaxBlocks + 1}, {MAMBA, 1}},
                                    Counts{{MAMBA, 1}, {SWA, kMaxBlocks + 1}}}) {
        r = cli.allocate_components(short_one);
        assert(r.status == RemoteStatus::OK && r.error == InsertError::MEMPOOL_FULL && r.slots.empty());
        assert(whole());
    }

    const auto bogus = static_cast<ComponentType>(NUM_COMPONENT_TYPES);
    for (const Counts& bad : {Counts{{FULL, 1}, {FULL, 1}}, Counts{{bogus, 1}}, Counts{{SWA, -1}},
                              Counts{{FULL, MAX_REMOTE_OP_BLOCKS}, {SWA, 1}}}) {
        assert(node.svc->allocate_components(bad).status == RemoteStatus::BAD_REQUEST);
        assert(whole());
    }
    for (const Counts& bad : {Counts{{FULL, 1}, {FULL, 1}}, Counts{{bogus, 1}}}) {
        r = cli.allocate_components(bad);
        assert(r.status == RemoteStatus::BAD_REQUEST && r.slots.empty());
    }
    assert(whole());

    auto past = std::chrono::system_clock::now();
    assert(node.svc->allocate_components({{FULL, 4}, {SWA, 1}}, past).status == RemoteStatus::TIMEOUT);
    GrpcRemoteOpClientTransport impatient(target_of(server.port()), 200, kInstance);
    {
        std::unique_lock<std::mutex> park(node.insert_mu);
        assert(impatient.allocate_components({{FULL, 4}, {MAMBA, 1}}).status == RemoteStatus::TIMEOUT);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    assert(whole());
    server.stop();
    expect_healthy(*node.client);
    TEST_PASS();
}

// ---- deadline: TIMEOUT, the op still runs once, the client never replays ----
void test_dedup_prefix() {
    TEST_BEGIN("DedupPrefix drops a covered leaf; RADIXSHMEM_DISABLE_AUTO_DEDUP=1 keeps it");
    auto h = make_hashes(300, 8);
    auto longer = make_hashes(300, 12);
    {
        Node node("dedup");
        GrpcRemoteOpServer server(0, kInstance);
        server.start(*node.svc);
        GrpcRemoteOpClientTransport cli(target_of(server.port()), 5000, kInstance);
        insert_auto(*node.client, h.data(), 8);
        DedupReply r = cli.dedup_prefix(longer.data(), 12);
        assert(r.status == RemoteStatus::OK && r.freed == 8 && node.used() == 0);
        r = cli.dedup_prefix(longer.data(), 12);
        assert(r.status == RemoteStatus::OK && r.freed == 0);
        assert(cli.dedup_prefix(nullptr, 3).status == RemoteStatus::BAD_REQUEST);
        server.stop();
        expect_healthy(*node.client);
    }
    ::setenv("RADIXSHMEM_DISABLE_AUTO_DEDUP", "1", 1);
    Node node("dedupoff");
    ::unsetenv("RADIXSHMEM_DISABLE_AUTO_DEDUP");
    GrpcRemoteOpServer server(0, kInstance);
    server.start(*node.svc);
    GrpcRemoteOpClientTransport cli(target_of(server.port()), 5000, kInstance);
    insert_auto(*node.client, h.data(), 8);
    DedupReply r = cli.dedup_prefix(h.data(), 8);
    assert(r.status == RemoteStatus::OK && r.freed == 0 && node.used() == 8);
    DedupRequest req;
    req.rank = 1;
    req.landing[0] = {5};
    r = cli.dedup_prefix(h.data(), 8, req);
    assert(r.status == RemoteStatus::OK && r.freed == 0 && r.unused[0] == (std::vector<int32_t>{5}));
    server.stop();
    expect_healthy(*node.client);
    TEST_PASS();
}

// Stands in for the server's transfer loop.
struct FakeWriter : DedupWriter {
    Result next = OK;
    bool on = true;
    int calls = 0, rank = -1;
    std::function<void()> during;
    std::vector<int32_t> local[NUM_COMPONENT_TYPES], remote[NUM_COMPONENT_TYPES];
    bool active() override { return on; }
    Result write(int r, const std::vector<int32_t> (&l)[NUM_COMPONENT_TYPES],
                 const std::vector<int32_t> (&rm)[NUM_COMPONENT_TYPES], int) override {
        calls++;
        rank = r;
        if (during) during();
        for (int k = 0; k < NUM_COMPONENT_TYPES; k++) {
            local[k] = l[k];
            remote[k] = rm[k];
        }
        return next;
    }
};

void test_dedup_move() {
    TEST_BEGIN("DedupPrefix with landing slots moves a side branch; failed and late writes keep it");
    Node node("dmove");
    FakeWriter w;
    node.svc->set_dedup_writer(&w);
    GrpcRemoteOpServer server(0, kInstance);
    server.start(*node.svc);
    GrpcRemoteOpClientTransport cli(target_of(server.port()), 5000, kInstance);
    std::vector<int64_t> tok(8), side_tok(6);
    std::iota(tok.begin(), tok.end(), 500);
    std::iota(side_tok.begin(), side_tok.end(), 500);
    side_tok[4] = 900;
    side_tok[5] = 901;
    auto h = RadixClient::compute_hashes(tok.data(), 8, 1);
    auto side = RadixClient::compute_hashes(side_tok.data(), 6, 1);
    DedupRequest req;
    req.rank = 3;
    req.tail_cap = 16;
    req.branch_cap = 16;
    req.landing[0] = {40, 41, 42, 43};
    auto both = [&] {
        insert_auto(*node.client, h.data(), 8);
        insert_auto(*node.client, side.data(), 6);
    };

    both();
    DedupReply r = cli.dedup_prefix(h.data(), 8, req);
    assert(r.status == RemoteStatus::OK && r.freed == 10 && node.used() == 0);
    assert(w.calls == 1 && w.rank == 3 && w.local[0].size() == 2 &&
           w.remote[0] == (std::vector<int32_t>{40, 41}));
    assert(r.moved.size() == 1 && r.moved[0].base == 4 && r.moved[0].full_from == 4 &&
           r.moved[0].full == (std::vector<int32_t>{40, 41}) &&
           r.moved[0].hashes == std::vector<uint64_t>(side.begin() + 4, side.end()) &&
           r.moved[0].full_age_ns.size() == 2);
    assert(r.unused[0] == (std::vector<int32_t>{42, 43}) && r.unused[1].empty());

    both();
    w.next = DedupWriter::FAILED;
    r = cli.dedup_prefix(h.data(), 8, req);
    assert(r.status == RemoteStatus::OK && r.freed == 0 && node.used() == 10 && r.moved.empty());
    assert(r.unused[0] == (std::vector<int32_t>{42, 43, 40, 41}));
    w.next = DedupWriter::LATE;
    r = cli.dedup_prefix(h.data(), 8, req);
    assert(r.freed == 0 && node.used() == 10 && r.moved.empty() &&
           r.unused[0] == (std::vector<int32_t>{42, 43}));

    DedupRequest drop_only;
    r = cli.dedup_prefix(h.data(), 8, drop_only);
    assert(r.freed == 4 && w.calls == 3 && node.used() == 6);
    server.stop();
    expect_healthy(*node.client);
    TEST_PASS();
}

void test_dedup_move_fallbacks() {
    TEST_BEGIN("DedupPrefix with landing slots: every drop-only or failed case hands them all back");
    Node node("dmfb");
    FakeWriter w;
    node.svc->set_dedup_writer(&w);
    GrpcRemoteOpServer server(0, kInstance);
    server.start(*node.svc);
    GrpcRemoteOpClientTransport cli(target_of(server.port()), 5000, kInstance);
    auto h = make_hashes(700, 8);
    auto longer = make_hashes(700, 12);
    DedupRequest req;
    req.rank = 3;
    req.tail_cap = 16;
    req.branch_cap = 16;
    req.landing[0] = {40, 41, 42, 43};
    const std::vector<int32_t> all = req.landing[0];

    w.on = false;
    insert_auto(*node.client, h.data(), 8);
    DedupReply r = cli.dedup_prefix(h.data(), 8, req);
    assert(r.status == RemoteStatus::OK && r.freed == 8 && r.unused[0] == all && node.used() == 0);
    w.on = true;
    DedupRequest anon = req;
    anon.rank = -1;
    insert_auto(*node.client, h.data(), 8);
    r = cli.dedup_prefix(h.data(), 8, anon);
    assert(r.freed == 8 && r.unused[0] == all && node.used() == 0);
    insert_auto(*node.client, h.data(), 8);
    r = cli.dedup_prefix(h.data(), 8, req);
    assert(r.freed == 8 && r.moved.empty() && r.unused[0] == all && node.used() == 0);
    assert(w.calls == 0);

    insert_auto(*node.client, longer.data(), 12);
    DedupRequest small = req;
    small.tail_cap = 2;
    r = cli.dedup_prefix(h.data(), 8, small);
    assert(r.freed == 0 && r.unused[0] == all && node.used() == 12);
    GrpcRemoteOpClientTransport hurried(target_of(server.port()), 200, kInstance);
    r = hurried.dedup_prefix(h.data(), 8, req);
    assert(r.status == RemoteStatus::OK && r.freed == 0 && r.moved.empty() && r.unused[0] == all &&
           node.used() == 12 && w.calls == 0);
    r = cli.dedup_prefix(nullptr, 3, req);
    assert(r.status == RemoteStatus::BAD_REQUEST && r.unused[0] == all);
    r = cli.dedup_prefix(h.data(), 8, req);
    assert(r.freed == 12 && w.calls == 1 && r.moved.size() == 1 && node.used() == 0);
    server.stop();
    expect_healthy(*node.client);
    TEST_PASS();
}

void test_dedup_move_cells() {
    TEST_BEGIN("DedupPrefix with landing slots carries SWA cells and Mamba checkpoints per pool");
    Node node("dmcell", kMaxBlocks, true);
    FakeWriter w;
    node.svc->set_dedup_writer(&w);
    GrpcRemoteOpServer server(0, kInstance);
    server.start(*node.svc);
    GrpcRemoteOpClientTransport cli(target_of(server.port()), 5000, kInstance);
    auto h = make_hashes(800, 8);
    insert_auto(*node.client, h.data(), 8);
    std::vector<int32_t> s;
    assert(node.client->allocate_slots(4, s, ComponentType::SWA) == 4);
    assert(node.client->insert(h.data(), 8, s.data(), 4, 0, true, ComponentType::SWA).error ==
           InsertError::OK);
    assert(node.client->allocate_slots(1, s, ComponentType::MAMBA) == 1);
    assert(node.client->insert(h.data(), 8, s.data(), 1, 0, true, ComponentType::MAMBA).error ==
           InsertError::OK);
    DedupRequest req;
    req.rank = 2;
    req.tail_cap = 16;
    req.branch_cap = 16;
    req.landing[1] = {60, 61, 62, 63, 64};
    req.landing[2] = {70, 71};
    DedupReply r = cli.dedup_prefix(h.data(), 8, req);
    assert(r.status == RemoteStatus::OK && r.freed == 8 && r.moved.size() == 1);
    const MovedPath& p = r.moved[0];
    assert(p.base == 8 && p.hashes.empty() && p.full.empty() && p.full_from == 8 && p.cells.size() == 5);
    for (int i = 0; i < 4; i++)
        assert(p.cells[i].component == ComponentType::SWA && p.cells[i].pos == 4 + i &&
               p.cells[i].slot == 60 + i && p.cells[i].age_ns < 10000000000ull);
    assert(p.cells[4].component == ComponentType::MAMBA && p.cells[4].pos == 7 && p.cells[4].slot == 70);
    assert(r.unused[0].empty() && r.unused[1] == (std::vector<int32_t>{64}) &&
           r.unused[2] == (std::vector<int32_t>{71}));
    assert(w.local[1].size() == 4 && w.remote[1] == (std::vector<int32_t>{60, 61, 62, 63}) &&
           w.local[2].size() == 1 && w.remote[2] == (std::vector<int32_t>{70}) && w.local[0].empty());
    server.stop();
    expect_healthy(*node.client);
    TEST_PASS();
}

void test_dedup_move_races() {
    TEST_BEGIN("DedupPrefix: a reset during the copy sends no manifest; no drop close to the deadline");
    Node node("dmrace");
    FakeWriter w;
    node.svc->set_dedup_writer(&w);
    auto h = make_hashes(950, 8);
    auto tail = make_hashes(950, 12);
    DedupRequest req;
    req.rank = 3;
    req.tail_cap = 16;
    req.branch_cap = 16;
    req.landing[0] = {40, 41, 42, 43};
    const std::vector<int32_t> all = req.landing[0];

    insert_auto(*node.client, tail.data(), 12);
    w.during = [&] { node.tree->tree().reset(); };
    DedupReply r = node.svc->dedup_prefix(h.data(), 8, req);
    assert(r.status == RemoteStatus::OK && r.freed == 0 && r.moved.empty() && r.unused[0] == all);
    assert(w.calls == 1 && node.used() == 0);

    insert_auto(*node.client, tail.data(), 12);
    w.during = [] { std::this_thread::sleep_for(std::chrono::milliseconds(1600)); };
    r = node.svc->dedup_prefix(h.data(), 8, req,
                               std::chrono::system_clock::now() + std::chrono::milliseconds(1700));
    assert(r.status == RemoteStatus::OK && r.freed == 0 && r.moved.size() == 1 && node.used() == 12);
    assert(r.unused[0].empty() && w.calls == 2);
    w.during = nullptr;
    r = node.svc->dedup_prefix(h.data(), 8, req);
    assert(r.freed == 12 && r.moved.size() == 1 && node.used() == 0);
    expect_healthy(*node.client);
    TEST_PASS();
}

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
        return cli.query(h.data(), n, rq(false)).hit_length == n;
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
    QueryReply q = cli.query(h.data(), n, rq(true));
    assert(q.status == RemoteStatus::OK && q.hit_length == n);
    assert(cli.unlock(h.data(), n, q.pin) ==
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
    QueryReply late = node.svc->query(h.data(), n, rq(true), past);
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

    QueryReply q = cli.query(h.data(), 4, rq(true));
    assert(q.status == RemoteStatus::OK && q.hit_length == 4);
    assert(q.pin.full.end == (uint32_t)n && q.pin.tail.size() == 4);
    assert(std::equal(q.pin.tail.begin(), q.pin.tail.end(), h.begin() + 4));
    assert(cli.unlock(h.data(), 4, q.pin) ==
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
    assert(cli.query(h.data(), n, rq(true)).status == RemoteStatus::STALE_TARGET);
    InsertResult ins = cli.insert(h.data(), n, 0, n, s.data(), true);
    assert(ins.error == InsertError::REMOTE_FAILED &&
           ins.remote_status == RemoteStatus::STALE_TARGET);
    assert(cli.recycle(s.data(), n) == RemoteStatus::STALE_TARGET);
    assert(cli.unlock(h.data(), n, full_pin(n)) == RemoteStatus::STALE_TARGET);
    DedupRequest req;
    req.rank = 1;
    req.landing[0] = {7};
    DedupReply d = cli.dedup_prefix(h.data(), n, req);
    assert(d.status == RemoteStatus::STALE_TARGET && d.unused[0] == (std::vector<int32_t>{7}));
    assert(node.used() == 0);
    server.stop();
    expect_healthy(*node.client);
    TEST_PASS();
}

void test_local_limits() {
    TEST_BEGIN("oversized requests → BAD_REQUEST without a server");
    GrpcRemoteOpClientTransport cli("127.0.0.1:1", 1000, kInstance);
    const int big = MAX_REMOTE_OP_BLOCKS + 1;
    auto h = make_hashes(1000, 4);
    std::vector<int32_t> s(4, 0);
    assert(cli.allocate(big).status == RemoteStatus::BAD_REQUEST);
    assert(cli.allocate_components({{ComponentType::FULL, big}}).status == RemoteStatus::BAD_REQUEST);
    assert(cli.allocate_components({{ComponentType::FULL, MAX_REMOTE_OP_BLOCKS},
                                    {ComponentType::SWA, 1}}).status == RemoteStatus::BAD_REQUEST);
    assert(cli.allocate_components({{ComponentType::SWA, -1}}).status == RemoteStatus::BAD_REQUEST);
    assert(cli.query(h.data(), big, rq(false)).status == RemoteStatus::BAD_REQUEST);
    assert(cli.insert(h.data(), big, 0, big, s.data(), true).remote_status ==
           RemoteStatus::BAD_REQUEST);
    assert(cli.recycle(s.data(), big) == RemoteStatus::BAD_REQUEST);
    assert(cli.unlock(h.data(), big, full_pin(4)) == RemoteStatus::BAD_REQUEST);
    assert(cli.allocate(1).status == RemoteStatus::UNAVAILABLE && !cli.connected());
    TEST_PASS();
}

struct FakeTransport final : IRemoteOpClientTransport {
    RemoteStatus next = RemoteStatus::OK;
    bool up = true;
    int calls = 0;

    AllocateReply allocate(int, ComponentType) override {
        calls++;
        AllocateReply r;
        r.status = next;
        return r;
    }
    AllocateComponentsReply allocate_components(const std::vector<ComponentCount>&) override {
        calls++;
        AllocateComponentsReply r;
        r.status = next;
        return r;
    }
    QueryReply query(const uint64_t*, int, const RemoteQuery&) override {
        calls++;
        QueryReply r;
        r.status = next;
        return r;
    }
    InsertResult insert(const uint64_t*, int, int, int, const int32_t*, bool,
                        ComponentType) override {
        calls++;
        InsertResult r;
        r.remote_status = next;
        return r;
    }
    InsertPathResult insert_path(const uint64_t*, int, int, int, const int32_t*,
                                 const std::vector<ComponentRun>& runs, bool) override {
        calls++;
        InsertPathResult r;
        r.remote_status = next;
        r.runs.resize(runs.size());
        return r;
    }
    RemoteStatus recycle(const int32_t*, int, ComponentType) override {
        calls++;
        return next;
    }
    RemoteStatus unlock(const uint64_t*, int, const PinRecord&) override {
        calls++;
        return next;
    }
    DedupReply dedup_prefix(const uint64_t*, int, const DedupRequest&) override {
        calls++;
        DedupReply r;
        r.status = next;
        return r;
    }
    RemoteStatus xrc_connect(const XrcConnInfo&, XrcConnInfo&) override {
        calls++;
        return next;
    }
    bool connected() override { return up; }
};

void test_breaker_gates_remote_ops() {
    TEST_BEGIN("remote ops: an unreachable peer is skipped until its backoff ends");
    auto owned = std::make_unique<FakeTransport>();
    FakeTransport& t = *owned;
    auto breaker = std::make_shared<PeerBreaker>("[test] remote ops", nullptr, 100, 400);
    RemotePeer peer(std::move(owned), "fake");
    peer.set_peer_rank(1);
    peer.set_breaker(breaker);
    auto h = make_hashes(100, 4);
    std::vector<int32_t> s(4, 0);

    t.next = RemoteStatus::TIMEOUT;
    assert(peer.allocate(1).status == RemoteStatus::TIMEOUT && !breaker->is_down(1));
    t.next = RemoteStatus::BAD_REQUEST;
    assert(peer.recycle(s.data(), 4) == RemoteStatus::BAD_REQUEST && !breaker->is_down(1));

    t.next = RemoteStatus::UNAVAILABLE;
    assert(peer.allocate(1).status == RemoteStatus::UNAVAILABLE && breaker->is_down(1));
    const int calls = t.calls;
    assert(peer.allocate(1).status == RemoteStatus::UNAVAILABLE);
    assert(peer.allocate_components({{ComponentType::FULL, 1}}).status == RemoteStatus::UNAVAILABLE);
    assert(peer.query(h.data(), 4, RemoteQuery{}).status == RemoteStatus::UNAVAILABLE);
    InsertResult ins = peer.insert(h.data(), 4, 0, 4, s.data());
    assert(ins.error == InsertError::REMOTE_FAILED && ins.remote_status == RemoteStatus::UNAVAILABLE);
    InsertPathResult path = peer.insert_path(h.data(), 4, 0, 4, s.data(), {{ComponentType::SWA, 4, {1}}});
    assert(path.remote_status == RemoteStatus::UNAVAILABLE && path.runs.size() == 1 &&
           path.full.error == InsertError::REMOTE_FAILED &&
           path.runs[0].error == InsertError::REMOTE_FAILED);
    assert(peer.recycle(s.data(), 4) == RemoteStatus::UNAVAILABLE);
    assert(peer.unlock(h.data(), 4, full_pin(4)) == RemoteStatus::UNAVAILABLE);
    assert(peer.dedup_prefix(h.data(), 4).status == RemoteStatus::UNAVAILABLE);
    DedupRequest landing;
    landing.landing[1] = {9};
    DedupReply skipped = peer.dedup_prefix(h.data(), 4, landing);
    assert(skipped.status == RemoteStatus::UNAVAILABLE && skipped.unused[1] == (std::vector<int32_t>{9}));
    assert(t.calls == calls);

    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    assert(peer.query(h.data(), 4, RemoteQuery{}).status == RemoteStatus::UNAVAILABLE && t.calls == calls + 1);
    assert(peer.query(h.data(), 4, RemoteQuery{}).status == RemoteStatus::UNAVAILABLE && t.calls == calls + 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    t.next = RemoteStatus::OK;
    assert(peer.allocate_components({{ComponentType::SWA, 1}}).status == RemoteStatus::OK &&
           t.calls == calls + 2 && !breaker->is_down(1));
    assert(peer.unlock(h.data(), 4, full_pin(4)) == RemoteStatus::OK && t.calls == calls + 3);

    t.next = RemoteStatus::UNAVAILABLE;
    assert(peer.allocate_components({{ComponentType::FULL, 1}}).status == RemoteStatus::UNAVAILABLE &&
           breaker->is_down(1) && t.calls == calls + 4);
    breaker->retry_now(1);
    t.next = RemoteStatus::TIMEOUT;
    t.up = false;
    assert(peer.query(h.data(), 4, RemoteQuery{}).status == RemoteStatus::TIMEOUT && breaker->is_down(1));
    assert(peer.query(h.data(), 4, RemoteQuery{}).status == RemoteStatus::UNAVAILABLE && t.calls == calls + 5);
    breaker->retry_now(1);
    t.next = RemoteStatus::OK;
    t.up = true;
    assert(peer.query(h.data(), 4, RemoteQuery{}).status == RemoteStatus::OK && !breaker->is_down(1));
    TEST_PASS();
}

// ---- the peer index over gRPC: a one-peer cluster seen from rank 0 ----

double secs_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

PeerEntry entry_of(Node& node, uint16_t port, uint64_t instance) {
    auto* hdr = static_cast<const ShmHeader*>(node.tree->tree().shm_base());
    PeerEntry e{};
    e.rank = 1;
    std::strcpy(e.host, "127.0.0.1");
    e.rpc_port = port;
    e.instance_id = instance;
    e.shm_size = node.tree->tree().shm_size();
    e.node_pool_capacity = hdr->node_pool_capacity;
    e.component_config = hdr->component_config;
    e.component_state_offset = hdr->component_state_offset;
    return e;
}

// Rank 1 is `node` behind `port`.
struct OnePeer {
    PeerRegistry reg;
    std::unique_ptr<GrpcPeerIndex> index;

    OnePeer(Node& node, uint16_t port, uint64_t instance, const ComponentConfig& expected) {
        std::vector<PeerEntry> peers(2);
        peers[0].rank = 0;
        peers[1] = entry_of(node, port, instance);
        reg.seed(std::move(peers), 2, 0);
        PeerIndexConfig pc;
        pc.has_expected = true;
        pc.expected = expected;
        index = std::make_unique<GrpcPeerIndex>(reg, pc);
    }
    OnePeer(Node& node, uint16_t port, uint64_t instance)
        : OnePeer(node, port, instance, node.tree->tree().component_config()) {}
};

RemoteQuery locked_window(uint32_t mask, uint32_t incumbent, uint32_t full_from) {
    RemoteQuery q;
    q.mask = mask;
    q.lock = true;
    q.window = {incumbent, full_from};
    return q;
}

bool refs_held(const Node& node) {
    return (node.client->check_health() & HEALTH_REF_LEAK) != 0;
}

const std::vector<RemoteHit> kNoHints;

void test_peer_index_components() {
    TEST_BEGIN("peer index: Query returns the part past the incumbent with SWA and Mamba; Unlock drops it");
    Node node("pi_cmp", kMaxBlocks, true);
    GrpcRemoteOpServer server(0, kInstance);
    server.start(*node.svc);
    GrpcRemoteOpClientTransport cli(target_of(server.port()), 5000, kInstance);
    const int n = 8;
    auto h = make_hashes(1100, n);
    auto full = reserve(cli, n, ComponentType::FULL);
    InsertPathResult ins = cli.insert_path(h.data(), n, 0, n, full.data(), path_runs(cli), true);
    assert(ins.remote_status == RemoteStatus::OK && ins.full.error == InsertError::OK);
    auto want = node.tree->tree().query_components(h.data(), n, COMPONENT_MASK_ALL, false, false);
    assert(want.common_hit == (uint32_t)n && want.mamba_checkpoint_slot != INVALID_COMPONENT_SLOT);

    OnePeer c(node, server.port(), kInstance);
    for (uint32_t inc : {0u, 3u, 6u, 8u}) {
        QueryReply r = c.index->query(1, h.data(), n, locked_window(COMPONENT_MASK_ALL, inc, inc),
                                      kNoHints);
        assert(r.status == RemoteStatus::OK && r.hit_length == n);
        assert(r.slots == std::vector<int32_t>(want.full_slots.begin() + inc,
                                               want.full_slots.begin() + n));
        assert(r.swa_start == want.swa_start && r.swa_slots == want.swa_slots);
        assert(r.mamba_slot == want.mamba_checkpoint_slot);
        assert(r.pin.source_rank == 1 && r.pin.source_instance == kInstance);
        assert(r.pin.empty() == (inc == (uint32_t)n) && refs_held(node) == !r.pin.empty());
        if (!r.pin.empty()) c.index->unlock(h.data(), n, std::move(r.pin));
        expect_healthy(*node.client);
    }

    // A peer whose tree has another layout yields nothing, without a call.
    ComponentConfig other = node.tree->tree().component_config();
    other.swa_window_blocks += 1;
    OnePeer mismatch(node, server.port(), kInstance, other);
    QueryReply r = mismatch.index->query(1, h.data(), n, locked_window(COMPONENT_MASK_ALL, 0, 0),
                                         kNoHints);
    assert(r.status == RemoteStatus::BAD_REQUEST && r.pin.empty() && !refs_held(node));
    server.stop();
    TEST_PASS();
}

void test_peer_index_dead_peer() {
    TEST_BEGIN("peer index: a dead peer fails fast and is skipped; its pin stays");
    Node node("pi_dead");
    auto server = std::make_unique<GrpcRemoteOpServer>(0, kInstance);
    server->start(*node.svc);
    GrpcRemoteOpClientTransport cli(target_of(server->port()), 5000, kInstance);
    const int n = 8;
    auto h = make_hashes(1200, n);
    AllocateReply a = cli.allocate(n);
    assert(cli.insert(h.data(), n, 0, n, a.slots.data(), true).error == InsertError::OK);

    OnePeer c(node, server->port(), kInstance);
    QueryReply held = c.index->query(1, h.data(), n, rq(true), kNoHints);
    assert(held.status == RemoteStatus::OK && !held.pin.empty());
    PinRecord leaked = held.pin;

    server->stop();
    server.reset();
    const auto t0 = std::chrono::steady_clock::now();
    QueryReply r = c.index->query(1, h.data(), n, rq(true), kNoHints);
    assert(r.status == RemoteStatus::UNAVAILABLE || r.status == RemoteStatus::TIMEOUT);
    assert(c.index->query(1, h.data(), n, rq(true), kNoHints).status == RemoteStatus::UNAVAILABLE);
    c.index->unlock(h.data(), n, std::move(held.pin));
    assert(secs_since(t0) < 1.0);
    assert(refs_held(node));   // nobody could release it: the leak a dead peer is allowed
    assert(node.tree->tree().dec_ref(h.data(), n, std::move(leaked)));
    expect_healthy(*node.client);
    TEST_PASS();
}

void test_peer_index_hung_peer() {
    TEST_BEGIN("peer index: a hung peer times out; it drops a pin taken after its caller left");
    setenv("RADIXSHMEM_PEER_INDEX_TIMEOUT_MS", "300", 1);
    Node node("pi_hung");
    GrpcRemoteOpServer server(0, kInstance);
    server.start(*node.svc);
    GrpcRemoteOpClientTransport cli(target_of(server.port()), 5000, kInstance);
    const int n = 8;
    auto h = make_hashes(1300, n);
    AllocateReply a = cli.allocate(n);
    assert(cli.insert(h.data(), n, 0, n, a.slots.data(), true).error == InsertError::OK);
    OnePeer c(node, server.port(), kInstance);
    assert(c.index->preconnect(1));

    node.tree->tree().write_lock();   // every query on the peer waits behind it
    auto t0 = std::chrono::steady_clock::now();
    QueryReply r = c.index->query(1, h.data(), n, rq(true), kNoHints);
    const double waited = secs_since(t0);
    assert(r.status == RemoteStatus::TIMEOUT && waited >= 0.25 && waited < 1.0);
    t0 = std::chrono::steady_clock::now();
    assert(c.index->query(1, h.data(), n, rq(true), kNoHints).status == RemoteStatus::UNAVAILABLE);
    assert(secs_since(t0) < 0.05);
    node.tree->tree().write_unlock();
    server.stop();   // returns once the parked handler is done
    expect_healthy(*node.client);
    unsetenv("RADIXSHMEM_PEER_INDEX_TIMEOUT_MS");
    TEST_PASS();
}

void test_peer_index_restarted_peer() {
    TEST_BEGIN("peer index: a restarted peer is stale until its new entry; old pins skip it");
    Node node("pi_restart");
    auto first = std::make_unique<GrpcRemoteOpServer>(0, kInstance);
    const uint16_t port = first->port();
    first->start(*node.svc);
    GrpcRemoteOpClientTransport cli(target_of(port), 5000, kInstance);
    const int n = 8;
    auto h = make_hashes(1400, n);
    AllocateReply a = cli.allocate(n);
    assert(cli.insert(h.data(), n, 0, n, a.slots.data(), true).error == InsertError::OK);

    OnePeer before(node, port, kInstance);
    QueryReply held = before.index->query(1, h.data(), n, rq(true), kNoHints);
    assert(held.status == RemoteStatus::OK && !held.pin.empty());
    PinRecord old_pin = held.pin;
    first->stop();
    first.reset();
    GrpcRemoteOpServer second(port, kInstance + 1);   // the same node, a new run
    second.start(*node.svc);

    // Its /peers entry has not reached this client yet.
    OnePeer c(node, port, kInstance);
    assert(c.index->query(1, h.data(), n, rq(false), kNoHints).status ==
           RemoteStatus::STALE_TARGET);
    assert(c.index->query(1, h.data(), n, rq(false), kNoHints).status ==
           RemoteStatus::STALE_TARGET);
    PeerEntry renewed = entry_of(node, port, kInstance + 1);
    assert(c.reg.admit(renewed));
    c.index->peer_changed(renewed);
    QueryReply fresh = c.index->query(1, h.data(), n, rq(true), kNoHints);
    assert(fresh.status == RemoteStatus::OK && fresh.pin.source_instance == kInstance + 1);
    c.index->unlock(h.data(), n, std::move(fresh.pin));
    // Here both runs share one tree, so the old pin shows; a real restart took it along.
    c.index->unlock(h.data(), n, std::move(held.pin));
    assert(refs_held(node));
    assert(node.tree->tree().dec_ref(h.data(), n, std::move(old_pin)));
    expect_healthy(*node.client);
    second.stop();
    TEST_PASS();
}

// ---- XRC setup: served before start(), for this instance, absent without RDMA ----
void test_xrc_connect() {
    TEST_BEGIN("XrcConnect before start(), instance-checked, UNKNOWN_OP without an RDMA side");
    std::atomic<bool> fail{false};
    // Answers with the caller's end shifted, so the reply shows both directions decoded.
    GrpcRemoteOpServer server(0, kInstance, [&](const XrcConnInfo& c) {
        if (fail.load()) throw std::runtime_error("no RECV QP");
        XrcConnInfo t = c;
        t.rank = c.rank + 1;
        t.qp_num = c.qp_num + 1;
        t.srq_num = 77;
        t.lid = c.lid + 1;
        t.gid[15] = (uint8_t)(c.gid[15] + 1);
        return t;
    });
    GrpcRemoteOpClientTransport cli(target_of(server.port()), 5000, kInstance);
    XrcConnInfo mine;
    mine.rank = 2;
    mine.qp_num = 0x4242;
    mine.link_layer = 2;
    mine.lid = 9;
    mine.gid_index = 3;
    for (int i = 0; i < 16; i++) mine.gid[i] = (uint8_t)(0xa0 + i);
    XrcConnInfo theirs;
    assert(cli.xrc_connect(mine, theirs) == RemoteStatus::OK);
    assert(theirs.rank == 3 && theirs.qp_num == 0x4243 && theirs.srq_num == 77 &&
           theirs.link_layer == 2 && theirs.lid == 10 && theirs.gid_index == 3);
    assert(std::memcmp(theirs.gid, mine.gid, 15) == 0 && theirs.gid[15] == 0xb0);
    assert(cli.allocate(1).status == RemoteStatus::UNAVAILABLE);   // the rest waits for start()

    GrpcRemoteOpClientTransport other(target_of(server.port()), 5000, kInstance + 1);
    assert(other.xrc_connect(mine, theirs) == RemoteStatus::STALE_TARGET);
    fail = true;
    assert(cli.xrc_connect(mine, theirs) == RemoteStatus::INTERNAL);
    server.stop();

    GrpcRemoteOpServer plain(0, kInstance);
    GrpcRemoteOpClientTransport no_rdma(target_of(plain.port()), 5000, kInstance);
    assert(no_rdma.xrc_connect(mine, theirs) == RemoteStatus::UNKNOWN_OP);
    plain.stop();
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

    assert(cli.query(h.data(), 4, rq(false)).status == RemoteStatus::UNAVAILABLE);
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
    grpc::Status AllocateComponents(grpc::ServerContext*, const pb::AllocateComponentsRequest* req,
                                    pb::AllocateComponentsResponse* rep) override {
        const int mode = req->components(0).num_blocks();
        for (int i = 0; i < req->components_size() - (mode == 1 ? 1 : 0); i++) {
            pb::ComponentSlots* c = rep->add_components();
            for (int j = 0; j < req->components(i).num_blocks() + (mode == 2 ? 1 : 0); j++) c->add_slots(j);
        }
        if (mode == 3) rep->set_error(pb::INSERT_ERROR_MEMPOOL_FULL);
        if (mode == 4) rep->set_error(static_cast<pb::InsertError>(99));
        return grpc::Status::OK;
    }
    grpc::Status Query(grpc::ServerContext*, const pb::QueryRequest*,
                       pb::QueryResponse* rep) override {
        rep->set_hit_length(3);   // fewer slots than hit_length
        return grpc::Status::OK;
    }
    grpc::Status InsertPath(grpc::ServerContext*, const pb::InsertPathRequest* req,
                            pb::InsertPathResponse* rep) override {
        rep->mutable_full();
        for (int i = 0; i < req->runs_size() - (req->start() == 0 ? 1 : 0); i++) rep->add_runs();
        if (req->start() == 1)
            for (int i = 0; i <= req->full_slots_size(); i++) rep->mutable_full()->add_unused_slots(i);
        if (req->start() == 2) rep->mutable_runs(0)->set_error(static_cast<pb::InsertError>(99));
        if (req->start() == 3)
            for (int i = 0; i <= req->runs(0).slots_size(); i++) rep->mutable_runs(0)->add_unused_slots(i);
        return grpc::Status::OK;
    }
    grpc::Status DedupPrefix(grpc::ServerContext*, const pb::DedupPrefixRequest*,
                             pb::DedupPrefixResponse* rep) override {
        rep->set_freed(3);   // a server without moves
        return grpc::Status::OK;
    }
    grpc::Status XrcConnect(grpc::ServerContext*, const pb::XrcConnectRequest*,
                            pb::XrcConnectResponse* rep) override {
        rep->mutable_target()->set_gid("short");
        return grpc::Status::OK;
    }
};

void test_bad_reply() {
    TEST_BEGIN("malformed replies → BAD_REPLY, unknown RPC → UNKNOWN_OP, no unused list → all landing back");
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
    assert(cli.query(h.data(), 4, rq(false)).status == RemoteStatus::BAD_REPLY);
    assert(cli.recycle(s.data(), 4) == RemoteStatus::UNKNOWN_OP);
    const std::vector<ComponentRun> runs = {{ComponentType::SWA, 4, {1, 2}}, {ComponentType::MAMBA, 4, {3}}};
    for (int mode : {1, 2, 3, 4}) {
        AllocateComponentsReply r = cli.allocate_components({{ComponentType::FULL, mode}, {ComponentType::SWA, 1}});
        assert(r.status == RemoteStatus::BAD_REPLY && r.error == InsertError::OK && r.slots.empty());
    }
    for (int start : {0, 1, 2, 3}) {
        InsertPathResult r = cli.insert_path(h.data(), 4, start, 4 - start, s.data(), runs, true);
        assert(r.remote_status == RemoteStatus::BAD_REPLY && r.full.error == InsertError::REMOTE_FAILED &&
               r.runs.size() == runs.size() && r.runs[1].error == InsertError::REMOTE_FAILED &&
               r.full.unused_slots.empty());
    }
    DedupRequest req;
    req.rank = 1;
    req.landing[0] = {5, 6};
    DedupReply d = cli.dedup_prefix(h.data(), 4, req);
    assert(d.status == RemoteStatus::OK && d.freed == 3 && d.moved.empty() &&
           d.unused[0] == (std::vector<int32_t>{5, 6}));
    XrcConnInfo theirs;
    assert(cli.xrc_connect(XrcConnInfo{}, theirs) == RemoteStatus::BAD_REPLY);
    server->Shutdown();
    TEST_PASS();
}

}  // namespace

int main() {
    printf("test_remote_op_grpc\n");
    test_transport_factory();
    test_roundtrip();
    test_large_request();
    test_insert_path();
    test_insert_path_rejects_bad_runs();
    test_allocate_components();
    test_dedup_prefix();
    test_dedup_move();
    test_dedup_move_fallbacks();
    test_dedup_move_cells();
    test_dedup_move_races();
    test_deadline_no_replay();
    test_no_admission_cap();
    test_deadline_skips_allocate_and_query();
    test_unlock_partial_node();
    test_stale_instance();
    test_local_limits();
    test_breaker_gates_remote_ops();
    test_peer_index_components();
    test_peer_index_dead_peer();
    test_peer_index_hung_peer();
    test_peer_index_restarted_peer();
    test_xrc_connect();
    test_lifecycle();
    test_bad_reply();
    return tests_report();
}
