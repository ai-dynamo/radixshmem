// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// test_rht_redis_faults — the Redis RHT while masters hang, fail over and
// while async replies pile up. Starts its own cluster of 3 masters with one
// replica each on 127.0.0.1:7351 with RADIXSHMEM_REDIS_SCRIPT and the
// binaries in RADIXSHMEM_TEST_REDIS_BIN; skips (77) when unset.

#include "test_utils.h"

#ifndef RADIXSHMEM_NO_RDMA
#include "rht_redis.h"

#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <string>
#include <thread>
#include <vector>

using namespace radixshmem;
using Clock = std::chrono::steady_clock;

static const int kPort = 7351, kMasters = 3, kReplicas = 1, kKeys = 3000;
static std::string g_bin, g_script;

static void cluster(const char* verb) {
    std::string cmd = "REDIS_BIN=" + g_bin + " bash " + g_script + " " + verb + " " +
                      std::to_string(kPort) + " " + std::to_string(kMasters) + " " +
                      std::to_string(kReplicas) + " >/dev/null 2>&1";
    if (std::system(cmd.c_str()) != 0) fprintf(stderr, "  %s failed\n", cmd.c_str());
}

static void on_abort(int) {
    cluster("stop");
    std::signal(SIGABRT, SIG_DFL);
    std::abort();
}

static void signal_masters(int sig, int first, int count) {
    for (int p = kPort + first; p < kPort + first + count; p++) {
        std::ifstream f("/tmp/radixshmem-redis-test/" + std::to_string(p) + "/redis.pid");
        int pid = 0;
        if (f >> pid && pid > 0) ::kill(pid, sig);
    }
}

// Waits until every master's replica is online and has applied all it was sent.
static bool replicas_synced() {
    for (int i = 0; i < kMasters; i++) {
        std::string cmd = g_bin + "/redis-cli -p " + std::to_string(kPort + i) + " info replication";
        std::string out;
        if (FILE* f = popen(cmd.c_str(), "r")) {
            char buf[512];
            while (fgets(buf, sizeof(buf), f)) out += buf;
            pclose(f);
        }
        size_t m = out.find("master_repl_offset:"), r = out.find("state=online,offset=");
        if (m == std::string::npos || r == std::string::npos) return false;
        if (std::stoll(out.substr(m + 19)) != std::stoll(out.substr(r + 20))) return false;
    }
    return true;
}

static uint64_t key_of(uint64_t i) {
    uint64_t z = i + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static std::unique_ptr<RhtBackend> filled(const std::string& seeds, const char* tag,
                                          PeerFaultCounters* faults) {
    auto rht = make_redis_rht({seeds, "rsh:test_rht_redis_faults_" + std::to_string(::getpid()) +
                               "_" + tag + ":", 50}, 1, faults);
    std::vector<RhtInsertEntry> e(kKeys);
    for (int i = 0; i < kKeys; i++) e[i] = {key_of(i), 1, (uint32_t)i, 4};
    rht->insert_batch(e.data(), kKeys);
    rht->flush();
    return rht;
}

struct Call { double t, ms; int found; };

// Four callers look up 8 keys at a time and one republishes and flushes, as
// RadixClient does, for secs seconds while `faults` runs alongside.
static std::vector<Call> probe(RhtBackend& rht, double secs, const std::function<void()>& faults) {
    auto t0 = Clock::now();
    auto since = [](Clock::time_point a) {
        return std::chrono::duration<double>(Clock::now() - a).count();
    };
    std::vector<std::vector<Call>> per(4);
    std::vector<std::thread> ts;
    for (int t = 0; t < 4; t++)
        ts.emplace_back([&, t] {
            std::vector<uint64_t> keys(8);
            std::vector<std::vector<RHTValue>> res(8);
            for (uint64_t i = t * 7919; since(t0) < secs;) {
                for (auto& k : keys) k = key_of(i++ % kKeys);
                double at = since(t0);
                auto a = Clock::now();
                rht.lookup_batch(keys.data(), 8, res.data());
                int found = 0;
                for (auto& r : res) found += !r.empty();
                per[t].push_back({at, since(a) * 1e3, found});
            }
        });
    ts.emplace_back([&] {
        std::vector<RhtInsertEntry> e(25);
        for (uint64_t i = 0; since(t0) < secs;) {
            for (auto& x : e) { uint64_t k = i++ % kKeys; x = {key_of(k), 1, (uint32_t)k, 4}; }
            rht.insert_batch(e.data(), (int)e.size());
            rht.flush();
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });
    faults();
    for (auto& t : ts) t.join();
    std::vector<Call> calls;
    for (auto& v : per) calls.insert(calls.end(), v.begin(), v.end());
    return calls;
}

static double slowest_ms(const std::vector<Call>& calls) {
    double m = 0;
    for (const Call& c : calls) m = std::max(m, c.ms);
    return m;
}

static double found_share(const std::vector<Call>& calls, double from, double to) {
    long found = 0, asked = 0;
    for (const Call& c : calls)
        if (c.t >= from && c.t < to) found += c.found, asked += 8;
    assert(asked > 0);
    return (double)found / asked;
}

static void pause_for(double s) { std::this_thread::sleep_for(std::chrono::duration<double>(s)); }

static void test_hung_master() {
    TEST_BEGIN("a hung master costs one deadline, is skipped, then comes back");
    PeerFaultCounters c{};
    auto rht = filled("127.0.0.1:" + std::to_string(kPort), "master", &c);
    auto calls = probe(*rht, 5.0, [] {
        pause_for(1.0);
        signal_masters(SIGSTOP, 1, 1);
        pause_for(1.5);
        signal_masters(SIGCONT, 1, 1);
    });
    assert(slowest_ms(calls) < 250);
    assert(found_share(calls, 1.3, 2.4) < 0.9);
    assert(found_share(calls, 4.5, 5.0) == 1.0);
    assert(c.down_events.load() == 1);
    TEST_PASS();
}

static void test_hung_cluster() {
    TEST_BEGIN("a hung cluster and an unreachable seed never block a request");
    PeerFaultCounters c{};
    auto rht = filled("127.0.0.1:" + std::to_string(kPort) + ",10.255.255.1:" +
                      std::to_string(kPort), "cluster", &c);
    auto calls = probe(*rht, 5.0, [] {
        pause_for(1.0);
        signal_masters(SIGSTOP, 0, kMasters);
        pause_for(1.5);
        signal_masters(SIGCONT, 0, kMasters);
    });
    assert(slowest_ms(calls) < 250);
    assert(found_share(calls, 1.3, 2.4) == 0.0);
    assert(found_share(calls, 4.5, 5.0) == 1.0);
    TEST_PASS();
}

static void test_unread_replies() {
    TEST_BEGIN("unflushed writes keep their replies drained; flush drops nothing");
    PeerFaultCounters c{};
    auto rht = filled("127.0.0.1:" + std::to_string(kPort), "backlog", &c);
    std::vector<RhtInsertEntry> e(1000);
    for (int round = 0; round < 2000; round++) {
        for (int i = 0; i < 1000; i++) e[i] = {key_of(round * 1000 + i), 1, (uint32_t)i, 4};
        rht->insert_batch(e.data(), 1000);
    }
    rht->flush();
    assert(c.rht_writes.load() == 0 && c.down_events.load() == 0);
    TEST_PASS();
}

static void test_failover() {
    TEST_BEGIN("a killed master's replica takes over with every route");
    PeerFaultCounters c{};
    auto rht = filled("127.0.0.1:" + std::to_string(kPort), "failover", &c);
    for (int i = 0; i < 300 && !replicas_synced(); i++) pause_for(0.1);   // a resyncing replica cannot take over
    auto calls = probe(*rht, 14.0, [] {
        pause_for(1.0);
        signal_masters(SIGKILL, 1, 1);
    });
    assert(slowest_ms(calls) < 250);
    assert(found_share(calls, 1.3, 5.0) < 0.9);
    assert(found_share(calls, 12.5, 14.0) == 1.0);
    std::vector<uint64_t> keys(kKeys);
    std::vector<std::vector<RHTValue>> res(kKeys);
    for (int i = 0; i < kKeys; i++) keys[i] = key_of(i);
    rht->lookup_batch(keys.data(), kKeys, res.data());
    for (auto& r : res) assert(r.size() == 1);
    TEST_PASS();
}

int main() {
    printf("=== test_rht_redis_faults ===\n\n");
    const char* bin = std::getenv("RADIXSHMEM_TEST_REDIS_BIN");
    const char* script = std::getenv("RADIXSHMEM_REDIS_SCRIPT");
    if (!bin || !*bin || !script || !*script) {
        printf("  (skipped: RADIXSHMEM_TEST_REDIS_BIN is not set)\n");
        return 77;
    }
    g_bin = bin;
    g_script = script;
    std::string seeds = "127.0.0.1:" + std::to_string(kPort);
    std::string why = redis_rht_check({seeds, "", 1000});
    if (why.find("without hiredis") != std::string::npos) {
        printf("  (skipped: %s)\n", why.c_str());
        return 77;
    }
    cluster("stop");
    cluster("start");
    std::signal(SIGABRT, on_abort);
    why = redis_rht_check({seeds, "", 1000});
    if (!why.empty()) {
        fprintf(stderr, "test cluster at %s did not come up: %s\n", seeds.c_str(), why.c_str());
        cluster("stop");
        return 1;
    }
    test_hung_master();
    test_hung_cluster();
    test_unread_replies();
    test_failover();
    cluster("stop");
    return tests_report();
}

#else // RADIXSHMEM_NO_RDMA

int main() {
    printf("=== test_rht_redis_faults ===\n\n  (skipped: built without RDMA)\n");
    return 77;
}

#endif
