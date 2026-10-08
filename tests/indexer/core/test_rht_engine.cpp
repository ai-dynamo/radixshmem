// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// test_rht_engine — RhtEngine over a fake one-sided endpoint (no NIC). Pins
// the remote read/write plumbing that the cluster tests only exercise with a
// handful of probes per shard:
//
//   * lookup_batch reads remote probes in rounds: one chain (<= 16, sized
//     down to the scratch) per endpoint, one wait, collect after the wait;
//   * shards on one holder share the endpoint's chain;
//   * every round's results are collected before the scratch is reused, so a
//     batch far larger than the scratch still returns every entry;
//   * enable_rdma refuses a scratch smaller than one bucket;
//   * with fault handling a dead holder never throws: its ops are dropped
//     and counted, lookups miss, and a probe after the backoff reconnects;
//     a holder that restarted is reached at its new address.
//
// RDMA builds only: the engine header pulls in the verbs transport surface.

#include "test_utils.h"

#ifndef RADIXSHMEM_NO_RDMA
#include "rht_core.hpp"   // private module header: this test drives RhtEngine directly
#include "rht_chain.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace radixshmem;

static void sleep_for_ms(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

// Shared "CQ": posted reads are served (memcpy) only at wait time, so
// collecting before the wait sees zeros.
struct FakeCq {
    std::vector<RdmaReadOp> inflight;
    int chains = 0;         // posted since the last wait
    int waits = 0;
};

// A "remote" holder over heap arrays; records how lookup_batch chained.
class FakeEndpoint : public IRdmaEndpoint {
public:
    explicit FakeEndpoint(FakeCq& cq) : cq_(cq) {}

    int calls = 0;          // chains posted
    int max_ops = 0;        // longest chain
    int total_ops = 0;      // sum of chain lengths
    int single_reads = 0;   // rdma_read invocations (remove/insert path)

    void rdma_read(void* local_buf, uint32_t, uint64_t remote_addr,
                   uint32_t, uint32_t length) override {
        single_reads++;
        std::memcpy(local_buf, (const void*)remote_addr, length);
    }
    uint64_t rdma_fetch_add(uint64_t, uint32_t, uint64_t, void*,
                            uint32_t) override {
        assert(false && "RHT never uses fetch_add");
        return 0;
    }
    void rdma_read_multi(const RdmaReadOp*, int, uint32_t) override {
        assert(false && "lookup_batch posts chains and waits; no sync read");
    }
    int rdma_read_multi_post(const RdmaReadOp* ops, int count,
                             uint32_t) override {
        assert(count >= 1 && count <= RDMA_READ_CHAIN_MAX);
        assert(cq_.chains == 0 || last_waited_);   // one chain per endpoint
        last_waited_ = false;
        calls++;
        max_ops = std::max(max_ops, count);
        total_ops += count;
        cq_.inflight.insert(cq_.inflight.end(), ops, ops + count);
        cq_.chains++;
        return 1;
    }
    void rdma_read_multi_wait(int pending) override {
        assert(pending == cq_.chains);   // owed exactly what was posted
        cq_.waits++;
        for (const RdmaReadOp& op : cq_.inflight)
            std::memcpy(op.local_buf, (const void*)op.remote_addr, op.length);
        cq_.inflight.clear();
        cq_.chains = 0;
        last_waited_ = true;
    }
    void rdma_write_async_inline(uint64_t remote_addr, uint32_t,
                                 const void* local_buf,
                                 uint32_t length) override {
        std::memcpy((void*)remote_addr, local_buf, length);
    }
    void flush() override {}

private:
    FakeCq& cq_;
    bool last_waited_ = true;
};

namespace {

constexpr int      M        = 2;      // shards: 0 local, 1 remote
constexpr uint32_t CAPACITY = 1024;
constexpr uint32_t W        = 4;
constexpr int      N_KEYS   = 200;    // 100 per shard, distinct buckets

// Keys carry a non-zero tag (bits 48..63) and map to distinct buckets:
// shard = key % 2, bucket = (key / 2) & mask.
uint64_t key_of(int i) {
    return ((uint64_t)(0xA5 + i) << 48) | (uint64_t)i;
}

// Shard 0 local; shards 1..m-1 behind one fake endpoint (one holder).
struct Fixture {
    int                               m;
    std::vector<RhtSlot>              local;
    std::vector<std::vector<RhtSlot>> remote;   // shards 1..m-1
    FakeCq                            cq;
    FakeEndpoint                      ep;
    std::vector<char>                 scratch;
    RhtEngine                         eng;

    explicit Fixture(size_t scratch_bytes, int num_shards = M)
        : m(num_shards),
          local((size_t)CAPACITY * W, RhtSlot{0, 0}),
          remote((size_t)num_shards - 1,
                 std::vector<RhtSlot>((size_t)CAPACITY * W, RhtSlot{0, 0})),
          ep(cq),
          scratch(scratch_bytes ? scratch_bytes : 1),
          eng(num_shards, CAPACITY, W, shard_ptrs()) {
        std::vector<IRdmaEndpoint*> eps(m, &ep);
        std::vector<RdmaMrInfo>     mrs(m, RdmaMrInfo{});
        eps[0] = nullptr;
        for (int s = 1; s < m; s++) {
            mrs[s].rkey   = 0x1234 + s;
            mrs[s].addr   = (uint64_t)remote[s - 1].data();
            mrs[s].length = remote[s - 1].size() * sizeof(RhtSlot);
        }
        eng.enable_rdma(std::move(eps), std::move(mrs),
                        scratch.data(), /*lkey=*/0, scratch_bytes);
    }

    std::vector<void*> shard_ptrs() {
        std::vector<void*> p(m, nullptr);
        p[0] = local.data();
        return p;
    }

    // Keys routed to a remote shard (shard_of = key % m).
    int remote_keys() const {
        int n = 0;
        for (int i = 0; i < N_KEYS; i++) n += key_of(i) % (uint64_t)m != 0;
        return n;
    }

    void publish_all() {
        std::vector<RhtInsertEntry> e;
        for (int i = 0; i < N_KEYS; i++)
            e.push_back({key_of(i), /*node_id=*/1, (uint32_t)i, (uint32_t)(i + 1)});
        eng.insert_batch(e.data(), (int)e.size());
        eng.flush();
    }

    // Look up every key; assert each comes back exactly once with its payload.
    void lookup_all_and_check() {
        std::vector<uint64_t> keys(N_KEYS);
        for (int i = 0; i < N_KEYS; i++) keys[i] = key_of(i);
        std::vector<std::vector<RHTValue>> res(N_KEYS);
        eng.lookup_batch(keys.data(), N_KEYS, res.data());
        for (int i = 0; i < N_KEYS; i++) {
            assert(res[i].size() == 1);
            assert(res[i][0].node_id == 1);
            assert(res[i][0].tree_node_id == (uint32_t)i);
            assert(res[i][0].num_blocks == (uint32_t)(i + 1));
        }
    }
};

constexpr size_t BUCKET_BYTES = (size_t)W * RHT_SLOT_BYTES;

} // namespace

// ==================================================================
static void test_large_scratch_chain_rounds() {
    TEST_BEGIN("scratch >= batch: rounds of one full chain, one wait each");
    Fixture f(64 * 1024);
    f.publish_all();
    f.lookup_all_and_check();
    const int chains =
        (N_KEYS / 2 + RDMA_READ_CHAIN_MAX - 1) / RDMA_READ_CHAIN_MAX;
    assert(f.ep.calls == chains);            // ceil(100/16) == 7
    assert(f.ep.max_ops == RDMA_READ_CHAIN_MAX);
    assert(f.ep.total_ops == N_KEYS / 2);
    assert(f.cq.waits == chains);            // one wait per round
    TEST_PASS();
}

static void test_two_shards_one_holder_share_chains() {
    TEST_BEGIN("two remote shards on one holder: mixed chains, still <= 16");
    Fixture f(64 * 1024, /*num_shards=*/3);
    f.publish_all();
    f.lookup_all_and_check();                // every entry of both shards
    const int rk = f.remote_keys();
    assert(rk > 2 * RDMA_READ_CHAIN_MAX);    // enough to need several rounds
    const int chains = (rk + RDMA_READ_CHAIN_MAX - 1) / RDMA_READ_CHAIN_MAX;
    assert(f.ep.total_ops == rk);
    assert(f.ep.max_ops == RDMA_READ_CHAIN_MAX);
    assert(f.ep.calls == chains);            // both shards fill one chain/round
    assert(f.cq.waits == chains);
    TEST_PASS();
}

static void test_two_shards_one_holder_small_scratch() {
    TEST_BEGIN("two remote shards on one holder, scratch of 5 buckets");
    Fixture f(5 * BUCKET_BYTES, /*num_shards=*/3);
    f.publish_all();
    f.lookup_all_and_check();
    const int rk = f.remote_keys();
    assert(f.ep.total_ops == rk);
    assert(f.ep.max_ops == 5);
    assert(f.ep.calls == (rk + 4) / 5);
    assert(f.cq.waits == f.ep.calls);
    TEST_PASS();
}

static void test_small_scratch_many_rounds() {
    TEST_BEGIN("scratch of 3 buckets: 100 probes -> 34 rounds, none > 3");
    Fixture f(3 * BUCKET_BYTES);
    f.publish_all();
    f.lookup_all_and_check();                // every entry still found
    assert(f.ep.total_ops == N_KEYS / 2);
    assert(f.ep.max_ops == 3);
    assert(f.ep.calls == (N_KEYS / 2 + 2) / 3);   // ceil(100/3) == 34
    assert(f.cq.waits == f.ep.calls);
    TEST_PASS();
}

static void test_one_bucket_scratch() {
    TEST_BEGIN("scratch of exactly one bucket: one probe per round");
    Fixture f(BUCKET_BYTES);
    f.publish_all();
    f.lookup_all_and_check();
    assert(f.ep.max_ops == 1);
    assert(f.ep.calls == N_KEYS / 2);
    TEST_PASS();
}

static void test_odd_scratch_rounds_down() {
    TEST_BEGIN("scratch not a bucket multiple: rounds use the floor");
    Fixture f(2 * BUCKET_BYTES + BUCKET_BYTES / 2);   // fits 2, not 3
    f.publish_all();
    f.lookup_all_and_check();
    assert(f.ep.max_ops == 2);
    assert(f.ep.calls == N_KEYS / 2 / 2);
    TEST_PASS();
}

static void test_scratch_below_one_bucket_rejected() {
    TEST_BEGIN("enable_rdma rejects scratch smaller than one bucket");
    bool threw = false;
    try {
        Fixture f(BUCKET_BYTES - 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    TEST_PASS();
}

static void test_remove_over_remote_shard() {
    TEST_BEGIN("remove_batch clears remote entries via single-bucket reads");
    Fixture f(BUCKET_BYTES);
    f.publish_all();                         // insert also reads each bucket once
    const int reads_before = f.ep.single_reads;
    std::vector<RhtRemoveEntry> rem;
    for (int i = 1; i < N_KEYS; i += 2)      // odd i -> shard 1 (remote)
        rem.push_back({key_of(i), 1, (uint32_t)i});
    f.eng.remove_batch(rem.data(), (int)rem.size());
    f.eng.flush();
    // One validating read per removed cell, each into the one-bucket scratch.
    assert(f.ep.single_reads - reads_before == (int)rem.size());

    std::vector<uint64_t> keys(N_KEYS);
    for (int i = 0; i < N_KEYS; i++) keys[i] = key_of(i);
    std::vector<std::vector<RHTValue>> res(N_KEYS);
    f.eng.lookup_batch(keys.data(), N_KEYS, res.data());
    for (int i = 0; i < N_KEYS; i++)
        assert(res[i].size() == ((i % 2 == 0) ? 1u : 0u));
    TEST_PASS();
}

static void test_follow_rht_chain() {
    TEST_BEGIN("follow_rht_chain: hops by published length until a gap or n");
    std::vector<uint64_t> h(20);
    for (int i = 0; i < 20; i++) h[i] = 1000 + (uint64_t)i;
    // Peer nodes start at 0 (4 blocks), 4 (4), 8 (6); nothing at 14.
    std::vector<uint64_t> asked;
    auto lookup = [&](uint64_t key, RhtHop& hop) {
        asked.push_back(key);
        if (key == h[4])  { hop.tree_node_id = 41; hop.num_blocks = 4; return true; }
        if (key == h[8])  { hop.tree_node_id = 81; hop.num_blocks = 6; return true; }
        return false;
    };
    std::vector<RhtHop> hops;
    assert(follow_rht_chain(h.data(), 20, 0, 4, lookup, hops) == 14);
    assert(hops.size() == 2 && hops[0].position == 4 && hops[0].tree_node_id == 41 &&
           hops[1].position == 8 && hops[1].num_blocks == 6);
    assert(asked.size() == 3 && asked.back() == h[14]);   // the gap costs one lookup

    hops.clear(); asked.clear();
    assert(follow_rht_chain(h.data(), 12, 0, 4, lookup, hops) == 14);   // last hop crosses n
    assert(hops.size() == 2 && asked.size() == 2);

    hops.clear(); asked.clear();
    assert(follow_rht_chain(h.data(), 20, 8, 6, lookup, hops) == 14 && hops.empty());
    assert(follow_rht_chain(h.data(), 20, 0, 0, lookup, hops) == 0 && hops.empty());
    assert(asked.size() == 1);   // a zero-length start never chains
    TEST_PASS();
}


// ==================================================================
// Fault handling: shard 0 local, shard 1 on holder 1, shard 2 on holder 2.

namespace {

// A holder that can die. Sync reads by default; `async` posts chains to the
// shared FakeCq like XRC, and a wait fails when a dead holder's chain is in it.
class FlakyEndpoint : public IRdmaEndpoint {
public:
    FlakyEndpoint(FakeCq& cq, std::vector<FlakyEndpoint*>& owners)
        : cq_(cq), owners_(owners) {}
    bool dead = false;
    bool flush_fails = false;
    bool async = false;
    int  ops = 0;           // remote ops attempted

    void rdma_read(void* local_buf, uint32_t, uint64_t remote_addr,
                   uint32_t, uint32_t length) override {
        hit();
        std::memcpy(local_buf, (const void*)remote_addr, length);
    }
    uint64_t rdma_fetch_add(uint64_t, uint32_t, uint64_t, void*, uint32_t) override {
        assert(false && "RHT never uses fetch_add");
        return 0;
    }
    void rdma_read_multi(const RdmaReadOp* o, int n, uint32_t) override {
        hit();
        for (int i = 0; i < n; i++)
            std::memcpy(o[i].local_buf, (const void*)o[i].remote_addr, o[i].length);
    }
    int rdma_read_multi_post(const RdmaReadOp* o, int n, uint32_t lkey) override {
        if (!async) {
            rdma_read_multi(o, n, lkey);
            return 0;
        }
        ops++;
        cq_.inflight.insert(cq_.inflight.end(), o, o + n);
        owners_.insert(owners_.end(), (size_t)n, this);
        return 1;
    }
    void rdma_read_multi_wait(int) override {
        bool failed = false;
        for (size_t i = 0; i < cq_.inflight.size(); i++) {
            const RdmaReadOp& op = cq_.inflight[i];
            if (owners_[i]->dead) failed = true;
            else std::memcpy(op.local_buf, (const void*)op.remote_addr, op.length);
        }
        cq_.inflight.clear();
        owners_.clear();
        if (failed) throw std::runtime_error("injected WC error status=12");
    }
    void rdma_write_async_inline(uint64_t remote_addr, uint32_t,
                                 const void* local_buf, uint32_t length) override {
        ops++;   // like DC: a dead target only shows at flush
        if (!dead) std::memcpy((void*)remote_addr, local_buf, length);
    }
    void flush() override {
        hit();
        if (flush_fails) throw std::runtime_error("injected flush WC error status=12");
    }
    bool broken() override { return dead; }

private:
    void hit() {
        ops++;
        if (dead) throw std::runtime_error("injected WC error status=12");
    }
    FakeCq& cq_;
    std::vector<FlakyEndpoint*>& owners_;   // owner of each cq_.inflight op
};

struct FaultFixture {
    static constexpr int kM = 3;
    std::vector<RhtSlot>              local;
    std::vector<std::vector<RhtSlot>> remote;
    FakeCq                            cq;
    std::vector<FlakyEndpoint*>       owners;
    FlakyEndpoint                     a, b;          // holders 1 and 2
    std::vector<char>                 scratch;
    PeerFaultCounters                 counters;
    PeerBreaker                       breaker;
    int                               connects[kM] = {}, releases[kM] = {};
    RhtEngine                         eng;

    explicit FaultFixture(bool with_faults = true)
        : local((size_t)CAPACITY * W, RhtSlot{0, 0}),
          remote(2, std::vector<RhtSlot>((size_t)CAPACITY * W, RhtSlot{0, 0})),
          a(cq, owners), b(cq, owners), scratch(64 * 1024),
          breaker("[test] RHT", &counters, /*backoff_ms=*/100, /*max_backoff_ms=*/400),
          eng(kM, CAPACITY, W, {local.data(), nullptr, nullptr}) {
        counters.clear();
        std::vector<RdmaMrInfo> mrs(kM, RdmaMrInfo{});
        for (int s = 1; s < kM; s++) {
            mrs[s].addr   = (uint64_t)remote[s - 1].data();
            mrs[s].length = remote[s - 1].size() * sizeof(RhtSlot);
        }
        eng.enable_rdma({nullptr, &a, &b}, std::move(mrs), scratch.data(), 0,
                        scratch.size());
        if (with_faults)
            eng.set_fault_handling({0, 1, 2}, &breaker,
                                   [this](int h) { connects[h]++; return h == 1 ? &a : &b; },
                                   [this](int h) { releases[h]++; });
    }

    void publish() {
        std::vector<RhtInsertEntry> e;
        for (int i = 0; i < N_KEYS; i++)
            e.push_back({key_of(i), 1, (uint32_t)i, (uint32_t)(i + 1)});
        eng.insert_batch(e.data(), (int)e.size());
        eng.flush();
    }

    // Found entries per shard.
    std::vector<int> lookup() {
        std::vector<uint64_t> keys(N_KEYS);
        for (int i = 0; i < N_KEYS; i++) keys[i] = key_of(i);
        std::vector<std::vector<RHTValue>> res(N_KEYS);
        eng.lookup_batch(keys.data(), N_KEYS, res.data());
        std::vector<int> found(kM, 0);
        for (int i = 0; i < N_KEYS; i++) found[key_of(i) % kM] += (int)res[i].size();
        return found;
    }

    int keys_on(int shard) const {
        int n = 0;
        for (int i = 0; i < N_KEYS; i++) n += key_of(i) % kM == (uint64_t)shard;
        return n;
    }
};

} // namespace

static void test_dead_holder_never_throws() {
    TEST_BEGIN("dead holder: publish/lookup/remove go on, its ops dropped and counted");
    FaultFixture f;
    f.publish();
    f.a.dead = true;
    auto found = f.lookup();                          // sync reads: holder 1 throws
    assert(found[0] == f.keys_on(0) && found[1] == 0 && found[2] == f.keys_on(2));
    assert(f.breaker.is_down(1) && !f.breaker.is_down(2));
    assert(f.releases[1] == 1 && f.counters.down_events.load() == 1);
    assert(f.counters.rht_lookups.load() == (uint64_t)f.keys_on(1));

    const int ops = f.a.ops;
    found = f.lookup();                               // within the backoff: not tried
    assert(f.a.ops == ops && found[1] == 0 && found[2] == f.keys_on(2));
    assert(f.counters.rht_lookups.load() == 2 * (uint64_t)f.keys_on(1));

    f.publish();                                      // W>1: the bucket read is skipped
    assert(f.counters.rht_writes.load() == (uint64_t)f.keys_on(1));
    std::vector<RhtRemoveEntry> rem;
    for (int i = 0; i < N_KEYS; i++) rem.push_back({key_of(i), 1, (uint32_t)i});
    f.eng.remove_batch(rem.data(), (int)rem.size());  // the evictor's path
    f.eng.flush();
    assert(f.counters.rht_removes.load() == (uint64_t)f.keys_on(1));
    assert(f.a.ops == ops);
    TEST_PASS();
}

static void test_probe_reconnects() {
    TEST_BEGIN("after the backoff one probe reconnects; success reopens the holder");
    FaultFixture f;
    f.publish();
    f.a.dead = true;
    f.lookup();
    assert(f.breaker.is_down(1));
    f.a.dead = false;
    sleep_for_ms(120);
    auto found = f.lookup();       // the first probe of holder 1 reconnects and reads
    assert(f.connects[1] == 1 && !f.breaker.is_down(1));
    assert(found[1] >= 1);
    found = f.lookup();
    assert(found[1] == f.keys_on(1));
    TEST_PASS();
}

static void test_flush_fault() {
    TEST_BEGIN("a failed flush (DC style) marks the holder down, no throw");
    FaultFixture f;
    f.a.flush_fails = true;
    f.publish();
    assert(f.breaker.is_down(1) && !f.breaker.is_down(2));
    assert(f.releases[1] == 1 && f.counters.down_events.load() == 1);
    TEST_PASS();
}

static void test_shared_cq_wait_fault() {
    TEST_BEGIN("a failed wait on a shared CQ faults only the broken endpoint");
    FaultFixture f;
    f.publish();
    f.a.async = f.b.async = true;
    f.a.dead = true;
    auto found = f.lookup();       // both chains posted, the wait throws
    assert(f.breaker.is_down(1) && !f.breaker.is_down(2));
    assert(f.releases[1] == 1 && f.releases[2] == 1);   // healthy one released, not down
    assert(found[0] == f.keys_on(0));
    found = f.lookup();            // holder 2 reconnects, holder 1 still skipped
    assert(f.connects[2] >= 1 && found[2] == f.keys_on(2) && found[1] == 0);
    TEST_PASS();
}

static void test_holder_changed() {
    TEST_BEGIN("holder_changed: a restarted holder is reached at its new address");
    FaultFixture f;
    f.publish();
    std::vector<RhtSlot> fresh((size_t)CAPACITY * W, RhtSlot{0, 0});   // the new, empty shard
    RdmaMrInfo mr{0, (uint64_t)fresh.data(), fresh.size() * sizeof(RhtSlot)};
    f.eng.holder_changed(1, &mr);
    assert(f.releases[1] == 1);
    auto found = f.lookup();
    assert(f.connects[1] == 1 && found[1] == 0 && found[2] == f.keys_on(2));
    f.publish();                                      // lands in the new shard
    assert(f.lookup()[1] == f.keys_on(1));
    f.eng.holder_changed(2, nullptr);                 // went away: endpoint dropped
    assert(f.releases[2] == 1);
    TEST_PASS();
}

static void test_no_fault_handling_throws() {
    TEST_BEGIN("without fault handling remote errors still throw");
    FaultFixture f(/*with_faults=*/false);
    f.publish();
    f.a.dead = true;
    bool threw = false;
    try {
        f.lookup();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
    TEST_PASS();
}

int main() {
    printf("=== test_rht_engine ===\n\n");
    test_large_scratch_chain_rounds();
    test_two_shards_one_holder_share_chains();
    test_two_shards_one_holder_small_scratch();
    test_small_scratch_many_rounds();
    test_one_bucket_scratch();
    test_odd_scratch_rounds_down();
    test_scratch_below_one_bucket_rejected();
    test_remove_over_remote_shard();
    test_follow_rht_chain();
    test_dead_holder_never_throws();
    test_probe_reconnects();
    test_flush_fault();
    test_shared_cq_wait_fault();
    test_holder_changed();
    test_no_fault_handling_throws();
    return tests_report();
}

#else // RADIXSHMEM_NO_RDMA

int main() {
    printf("=== test_rht_engine ===\n\n  (skipped: built without RDMA)\n");
    return 0;
}

#endif
