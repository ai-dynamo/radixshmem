// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// test_rht_redis — the Redis RHT against a live Redis Cluster at
// RADIXSHMEM_TEST_REDIS (host:port of any node); skips (77) when unset.

#include "test_utils.h"

#ifndef RADIXSHMEM_NO_RDMA
#include "rht_redis.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

using namespace radixshmem;

static std::string g_seeds;

static RedisRhtConfig config(const std::string& tag) {
    return {g_seeds, "rsh:test_rht_redis_" + std::to_string(::getpid()) + "_" + tag + ":", 1000};
}

static std::vector<RHTValue> lookup(RhtBackend& b, uint64_t key) {
    std::vector<RHTValue> out;
    b.lookup_batch(&key, 1, &out);
    std::sort(out.begin(), out.end(),
              [](const RHTValue& x, const RHTValue& y) { return x.node_id < y.node_id; });
    return out;
}

static void test_ranks_and_replace() {
    TEST_BEGIN("routes of several ranks; a rank replaces its own");
    auto r0 = make_redis_rht(config("ranks"), 0, nullptr);
    auto r1 = make_redis_rht(config("ranks"), 1, nullptr);
    auto obs = make_redis_rht(config("ranks"), -1, nullptr);
    RhtInsertEntry a{42, 0, 5, 64}, b{42, 1, 9, 128}, a2{42, 0, 6, 32};
    r0->insert_batch(&a, 1);
    r1->insert_batch(&b, 1);
    r0->insert_batch(&a2, 1);
    r0->flush();
    r1->flush();
    auto v = lookup(*obs, 42);
    assert(v.size() == 2);
    assert(v[0].node_id == 0 && v[0].tree_node_id == 6 && v[0].num_blocks == 32);
    assert(v[1].node_id == 1 && v[1].tree_node_id == 9 && v[1].num_blocks == 128);
    assert(lookup(*obs, 43).empty());
    r0->reset(nullptr, -1);
    r1->reset(nullptr, -1);
    TEST_PASS();
}

static void test_conditional_remove() {
    TEST_BEGIN("remove clears a route only while it names the node");
    auto r0 = make_redis_rht(config("remove"), 0, nullptr);
    auto r1 = make_redis_rht(config("remove"), 1, nullptr);
    RhtInsertEntry a{7, 0, 70, 1}, b{7, 1, 71, 1};
    r0->insert_batch(&a, 1);
    r1->insert_batch(&b, 1);
    RhtRemoveEntry stale{7, 0, 99};
    r0->remove_batch(&stale, 1);
    r0->flush();
    r1->flush();
    assert(lookup(*r0, 7).size() == 2);
    RhtRemoveEntry mine{7, 0, 70};
    r0->remove_batch(&mine, 1);
    r0->flush();
    auto v = lookup(*r0, 7);
    assert(v.size() == 1 && v[0].node_id == 1 && v[0].tree_node_id == 71);
    r1->reset(nullptr, -1);
    TEST_PASS();
}

static void test_batches_and_reset() {
    TEST_BEGIN("large batches over every master; reset by list or by scan drops one rank");
    const int n = 5000;
    auto r0 = make_redis_rht(config("batch"), 0, nullptr);
    auto r1 = make_redis_rht(config("batch"), 1, nullptr);
    std::vector<RhtInsertEntry> e0(n), e1(n);
    std::vector<uint64_t> keys(n + 1000);
    for (int i = 0; i < n + 1000; i++) keys[i] = 0x9E3779B97F4A7C15ULL * (i + 1);
    for (int i = 0; i < n; i++) {
        e0[i] = {keys[i], 0, (uint32_t)i, (uint32_t)(i % 64 + 1)};
        e1[i] = {keys[i], 1, (uint32_t)(i + 1), 64};
    }
    r0->insert_batch(e0.data(), n);
    r1->insert_batch(e1.data(), n);
    r0->flush();
    r1->flush();
    std::vector<std::vector<RHTValue>> res(n + 1000);
    r0->lookup_batch(keys.data(), n + 1000, res.data());
    for (int i = 0; i < n; i++) {
        assert(res[i].size() == 2);
        for (const RHTValue& v : res[i])
            assert(v.node_id == 0 ? v.tree_node_id == (uint32_t)i && v.num_blocks == (uint32_t)(i % 64 + 1)
                                  : v.node_id == 1 && v.tree_node_id == (uint32_t)(i + 1));
    }
    for (int i = n; i < n + 1000; i++) assert(res[i].empty());
    std::vector<RhtRemoveEntry> own(n / 2);
    for (int i = 0; i < n / 2; i++) own[i] = {keys[i], 0, (uint32_t)i};
    r0->reset(own.data(), n / 2);
    r1->lookup_batch(keys.data(), n, res.data());
    for (int i = 0; i < n; i++) assert(res[i].size() == (i < n / 2 ? 1u : 2u));
    r0->reset(nullptr, -1);
    r1->lookup_batch(keys.data(), n, res.data());
    for (int i = 0; i < n; i++) assert(res[i].size() == 1 && res[i][0].node_id == 1);
    r1->reset(nullptr, -1);
    r1->lookup_batch(keys.data(), n, res.data());
    for (int i = 0; i < n; i++) assert(res[i].empty());
    TEST_PASS();
}

static void test_dump() {
    TEST_BEGIN("dump lists the fields behind a key");
    auto r0 = make_redis_rht(config("dump"), 3, nullptr);
    RhtInsertEntry a{11, 3, 12, 13};
    r0->insert_batch(&a, 1);
    r0->flush();
    std::vector<std::string> lines;
    r0->dump(11, lines);
    assert(lines.size() == 1 && lines[0] == "r3:t12/13");
    r0->reset(nullptr, -1);
    TEST_PASS();
}

static void test_unreachable() {
    TEST_BEGIN("an unreachable cluster holds writes, drops lookups, never throws");
    PeerFaultCounters c;
    c.clear();
    auto r = make_redis_rht({"127.0.0.1:1", "rsh:none:", 50}, 0, &c);
    RhtInsertEntry a{1, 0, 1, 1};
    RhtRemoveEntry d{1, 0, 1};
    r->insert_batch(&a, 1);
    r->remove_batch(&d, 1);
    r->flush();
    assert(lookup(*r, 1).empty());
    assert(c.rht_writes.load() == 0 && c.rht_removes.load() == 0 && c.rht_lookups.load() == 1);
    TEST_PASS();
}

static void test_gives_up() {
    TEST_BEGIN("reset and clear give up on a cluster whose slot map cannot load");
    auto run = [](const std::string& seeds) {
        RedisRhtConfig cfg{seeds, "rsh:test_rht_redis_" + std::to_string(::getpid()) + "_giveup:", 50};
        cfg.retry_secs = 1;
        auto r = make_redis_rht(cfg, 0, nullptr);
        std::vector<RhtRemoveEntry> own(10);
        for (int i = 0; i < 10; i++) own[i] = {0x1234567800000000ULL + i, 0, (uint32_t)i};
        auto t0 = std::chrono::steady_clock::now();
        r->reset(own.data(), 10);
        r->reset(nullptr, -1);
        RhtInsertEntry e{0x42, 0, 1, 4};
        r->insert_batch(&e, 1);
        r->flush();
        assert(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5));
        assert(!redis_rht_clear(cfg) && !redis_rht_drop_rank(cfg, 0));
    };
    run("127.0.0.1:1");
    setenv("RADIXSHMEM_REDIS_PASSWORD", "wrong", 1);
    run(g_seeds);
    unsetenv("RADIXSHMEM_REDIS_PASSWORD");
    TEST_PASS();
}

static void test_prefix_match() {
    TEST_BEGIN("clear and scan reset keep to their own prefix, glob characters included");
    const std::string base = "rsh:test_rht_redis_" + std::to_string(::getpid()) + "_";
    auto cfg = [&](const std::string& id) { return RedisRhtConfig{g_seeds, base + id + ":", 1000}; };
    const std::vector<std::string> ids = {"a", "a:b", "x[1]", "x1"};
    std::vector<std::unique_ptr<RhtBackend>> r;
    for (const std::string& id : ids) {
        r.push_back(make_redis_rht(cfg(id), 1, nullptr));
        std::vector<RhtInsertEntry> e(100);
        for (int i = 0; i < 100; i++) e[i] = {0x5000 + (uint64_t)i, 1, (uint32_t)i, 4};
        r.back()->insert_batch(e.data(), 100);
        r.back()->flush();
    }
    auto found = [&](size_t k) {
        int n = 0;
        for (int i = 0; i < 100; i++) n += !lookup(*r[k], 0x5000 + (uint64_t)i).empty();
        return n;
    };
    assert(redis_rht_clear(cfg("a")));
    assert(found(0) == 0 && found(1) == 100);
    r[2]->reset(nullptr, -1);
    assert(found(2) == 0 && found(3) == 100);
    redis_rht_clear(cfg("a:b"));
    redis_rht_clear(cfg("x1"));
    TEST_PASS();
}

static void test_bad_config() {
    TEST_BEGIN("bad seeds or a hash-tag prefix are refused");
    auto throws = [](const RedisRhtConfig& cfg) {
        try {
            make_redis_rht(cfg, 0, nullptr);
        } catch (const std::runtime_error&) {
            return true;
        }
        return false;
    };
    assert(throws({"", "rsh:x:", 50}));
    assert(throws({g_seeds, "rsh:{x}:", 50}));
    assert(throws({g_seeds, "rsh:x:", 0}));
    TEST_PASS();
}

int main() {
    printf("=== test_rht_redis ===\n\n");
    const char* seeds = std::getenv("RADIXSHMEM_TEST_REDIS");
    if (!seeds || !*seeds) {
        printf("  (skipped: RADIXSHMEM_TEST_REDIS is not set)\n");
        return 77;
    }
    g_seeds = seeds;
    std::string why = redis_rht_check({g_seeds, "", 1000});
    if (why.find("without hiredis") != std::string::npos) {
        printf("  (skipped: %s)\n", why.c_str());
        return 77;
    }
    if (!why.empty()) {
        fprintf(stderr, "Redis Cluster at %s is not usable: %s\n", seeds, why.c_str());
        return 1;
    }
    test_ranks_and_replace();
    test_conditional_remove();
    test_batches_and_reset();
    test_dump();
    test_unreachable();
    test_gives_up();
    test_prefix_match();
    test_bad_config();
    return tests_report();
}

#else // RADIXSHMEM_NO_RDMA

int main() {
    printf("=== test_rht_redis ===\n\n  (skipped: built without RDMA)\n");
    return 77;
}

#endif
