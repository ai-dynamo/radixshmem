// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// test_rht_redis_faults — the Redis RHT while masters hang, fail over or
// give up slots, and while async replies pile up. Starts its own cluster of 3 masters with one
// replica each from the binaries in RADIXSHMEM_TEST_REDIS_BIN, on free ports and in a
// directory of its own; skips (77) when unset. The nodes are its children and die with it.

#include "test_utils.h"

#ifndef RADIXSHMEM_NO_RDMA
#include "rht_redis.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <thread>
#include <vector>

using namespace radixshmem;
using Clock = std::chrono::steady_clock;

static const int kMasters = 3, kReplicas = 1, kNodes = kMasters * (1 + kReplicas), kKeys = 3000;
static std::string g_bin, g_cli, g_dir, g_seeds;
static int g_port[kNodes];     // the first kMasters start as masters
static pid_t g_pid[kNodes];

static sockaddr_in loopback(int port) {
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    return a;
}

// Ports whose cluster bus port (+10000) is free too; every socket stays bound
// until all are picked, so no port is picked twice.
static bool pick_ports() {
    std::vector<int> held;
    auto bind_port = [&](int port) {
        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return 0;
        held.push_back(fd);
        sockaddr_in a = loopback(port);
        socklen_t len = sizeof(a);
        if (::bind(fd, (sockaddr*)&a, len) != 0 || ::getsockname(fd, (sockaddr*)&a, &len) != 0)
            return 0;
        return (int)ntohs(a.sin_port);
    };
    int n = 0;
    for (int tries = 0; n < kNodes && tries < 1000; tries++) {
        int p = bind_port(0);
        if (p > 0 && p + 10000 <= 65535 && bind_port(p + 10000) > 0) g_port[n++] = p;
    }
    for (int fd : held) ::close(fd);
    return n == kNodes;
}

static bool reachable(int port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a = loopback(port);
    bool ok = fd >= 0 && ::connect(fd, (sockaddr*)&a, sizeof(a)) == 0;
    if (fd >= 0) ::close(fd);
    return ok;
}

// Node i as a child that the kernel kills when the test dies, however it dies.
static pid_t start_node(int i) {
    const std::string port = std::to_string(g_port[i]), dir = g_dir + "/" + port;
    const std::string conf = dir + "/nodes.conf", log = dir + "/redis.log";
    std::string server = g_bin + "/redis-server";
    if (::access(server.c_str(), X_OK) != 0) server = g_bin + "/valkey-server";
    const char* argv[] = {server.c_str(), "--port", port.c_str(), "--bind", "127.0.0.1",
                          "--cluster-enabled", "yes", "--cluster-config-file", conf.c_str(),
                          "--cluster-node-timeout", "5000", "--cluster-require-full-coverage", "no",
                          "--save", "", "--appendonly", "no", "--maxmemory", "2gb",
                          "--maxmemory-policy", "allkeys-lru", "--dir", dir.c_str(),
                          "--logfile", log.c_str(), nullptr};
    if (::mkdir(dir.c_str(), 0700) != 0) return -1;
    const pid_t parent = ::getpid();
    const pid_t pid = ::fork();
    if (pid == 0) {
        ::prctl(PR_SET_PDEATHSIG, SIGKILL);
        if (::getppid() != parent) _exit(1);
        int null = ::open("/dev/null", O_WRONLY);
        if (null >= 0) {
            ::dup2(null, 1);
            ::dup2(null, 2);
        }
        ::execv(argv[0], const_cast<char* const*>(argv));
        _exit(127);
    }
    return pid;
}

static bool start_cluster() {
    g_dir = "/tmp/radixshmem-redis-faults-" + std::to_string(::getpid());
    if (std::system(("rm -rf " + g_dir).c_str()) != 0 || ::mkdir(g_dir.c_str(), 0700) != 0 ||
        !pick_ports())
        return false;
    g_seeds = "127.0.0.1:" + std::to_string(g_port[0]);
    std::string create = g_cli + " --cluster create";
    for (int i = 0; i < kNodes; i++) {
        if ((g_pid[i] = start_node(i)) < 0) return false;
        create += " 127.0.0.1:" + std::to_string(g_port[i]);
    }
    for (int i = 0; i < kNodes; i++)
        for (int t = 0; !reachable(g_port[i]); t++) {
            if (t == 100) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    create += " --cluster-replicas " + std::to_string(kReplicas) + " --cluster-yes >/dev/null 2>&1";
    return std::system(create.c_str()) == 0;
}

// "" once every node reports cluster_state:ok, else the first complaint.
static std::string cluster_ok() {
    for (int i = 0; i < kNodes; i++) {
        std::string why = redis_rht_check({"127.0.0.1:" + std::to_string(g_port[i]), "", 1000});
        if (!why.empty()) return why;
    }
    return "";
}

static void stop_cluster() {
    for (int i = 0; i < kNodes; i++)
        if (g_pid[i] > 0) {
            ::kill(g_pid[i], SIGKILL);
            ::waitpid(g_pid[i], nullptr, 0);
            g_pid[i] = 0;
        }
    if (std::system(("rm -rf " + g_dir).c_str()) != 0) fprintf(stderr, "  rm -rf %s failed\n", g_dir.c_str());
}

static void signal_masters(int sig, int first, int count) {
    for (int i = first; i < first + count; i++)
        if (g_pid[i] > 0) ::kill(g_pid[i], sig);
}

static std::string node_id(int port) {
    std::string cmd = g_cli + " -p " + std::to_string(port) + " cluster myid";
    std::string id;
    if (FILE* f = popen(cmd.c_str(), "r")) {
        char buf[128] = {};
        if (fgets(buf, sizeof(buf), f)) id = buf;
        pclose(f);
    }
    while (!id.empty() && (id.back() == '\n' || id.back() == '\r')) id.pop_back();
    return id;
}

// Moves `slots` slots from one master to another with redis-cli.
static void move_slots(int from, int to, int slots) {
    std::string cmd = g_cli + " --cluster reshard 127.0.0.1:" + std::to_string(from) +
                      " --cluster-from " + node_id(from) + " --cluster-to " + node_id(to) +
                      " --cluster-slots " + std::to_string(slots) + " --cluster-yes >/dev/null 2>&1";
    if (std::system(cmd.c_str()) != 0) fprintf(stderr, "  %s failed\n", cmd.c_str());
}

// Waits until every master's replica is online and has applied all it was sent.
static bool replicas_synced() {
    for (int i = 0; i < kMasters; i++) {
        std::string cmd = g_cli + " -p " + std::to_string(g_port[i]) + " info replication";
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

static std::string prefix_of(const char* tag) {
    return "rsh:test_rht_redis_faults_" + std::to_string(::getpid()) + "_" + tag + ":";
}

// Index of the first key from `from` on whose Redis key falls in slots [lo, hi].
static uint64_t index_in_slots(const std::string& prefix, int lo, int hi, uint64_t from) {
    for (uint64_t i = from;; i++) {
        char name[128];
        snprintf(name, sizeof(name), "%s%016llx", prefix.c_str(), (unsigned long long)key_of(i));
        uint16_t crc = 0;
        for (const char* q = name; *q; q++) {
            crc ^= (uint16_t)((uint8_t)*q << 8);
            for (int b = 0; b < 8; b++) crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
        }
        if ((crc & 16383) >= lo && (crc & 16383) <= hi) return i;
    }
}

static std::unique_ptr<RhtBackend> filled(const std::string& seeds, const char* tag,
                                          PeerFaultCounters* faults,
                                          std::function<bool(uint64_t, uint32_t)> live = {}) {
    auto rht = make_redis_rht({seeds, prefix_of(tag), 50, std::move(live)}, 1, faults);
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
    auto rht = filled(g_seeds, "master", &c);
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
    auto rht = filled(g_seeds + ",10.255.255.1:" + std::to_string(g_port[0]), "cluster", &c);
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
    auto rht = filled(g_seeds, "backlog", &c);
    std::vector<RhtInsertEntry> e(1000);
    for (int round = 0; round < 2000; round++) {
        for (int i = 0; i < 1000; i++) e[i] = {key_of(round * 1000 + i), 1, (uint32_t)i, 4};
        rht->insert_batch(e.data(), 1000);
    }
    rht->flush();
    assert(c.rht_writes.load() == 0 && c.down_events.load() == 0);
    TEST_PASS();
}

static void test_reset_after_move() {
    TEST_BEGIN("a reset that meets moved slots retries them, by list and by scan");
    const std::string& seeds = g_seeds;
    for (int by_list = 1; by_list >= 0; by_list--) {
        const std::string prefix = "rsh:test_rht_redis_faults_" + std::to_string(::getpid()) +
                                   "_moved" + std::to_string(by_list) + ":";
        auto r0 = make_redis_rht({seeds, prefix, 50}, 0, nullptr);
        auto r1 = make_redis_rht({seeds, prefix, 50}, 1, nullptr);
        std::vector<RhtInsertEntry> e(kKeys);
        std::vector<RhtRemoveEntry> own(kKeys);
        std::vector<uint64_t> keys(kKeys);
        for (int k = 0; k < 2; k++) {
            for (int i = 0; i < kKeys; i++) e[i] = {key_of(i), k, (uint32_t)i, 4};
            (k ? r1 : r0)->insert_batch(e.data(), kKeys);
            (k ? r1 : r0)->flush();
        }
        for (int i = 0; i < kKeys; i++) own[i] = {key_of(i), 0, (uint32_t)i}, keys[i] = key_of(i);
        move_slots(by_list ? g_port[0] : g_port[2], by_list ? g_port[2] : g_port[0], 1000);   // r0's map is stale now
        r0->reset(own.data(), by_list ? kKeys : -1);
        auto check = make_redis_rht({seeds, prefix, 1000}, 2, nullptr);
        std::vector<std::vector<RHTValue>> res(kKeys);
        check->lookup_batch(keys.data(), kKeys, res.data());
        for (auto& r : res) assert(r.size() == 1 && r[0].node_id == 1);
    }
    TEST_PASS();
}

static void test_killed_connections() {
    TEST_BEGIN("connections the servers closed are reopened, not taken for down masters");
    PeerFaultCounters c{};
    auto rht = filled(g_seeds, "killed", &c);
    std::vector<uint64_t> keys(2 * kKeys);
    std::vector<std::vector<RHTValue>> res(2 * kKeys);
    for (int i = 0; i < kKeys; i++) keys[i] = key_of(i);
    rht->lookup_batch(keys.data(), kKeys, res.data());   // pooled connections to every master
    std::vector<RhtInsertEntry> e(kKeys);
    for (int i = 0; i < kKeys; i++) e[i] = {key_of(kKeys + i), 1, (uint32_t)i, 4};
    rht->insert_batch(e.data(), kKeys);                   // replies still owed
    for (int i = 0; i < kMasters; i++)
        if (std::system((g_cli + " -p " + std::to_string(g_port[i]) +
                         " client kill type normal skipme yes >/dev/null 2>&1").c_str()) != 0)
            fprintf(stderr, "  CLIENT KILL on %d failed\n", g_port[i]);
    pause_for(0.2);
    for (int round = 0; round < 2; round++) {
        for (int i = 0; i < kKeys; i++) e[i] = {key_of((2 + round) * kKeys + i), 1, (uint32_t)i, 4};
        rht->insert_batch(e.data(), kKeys);
    }
    rht->flush();
    for (int i = 0; i < 2 * kKeys; i++) keys[i] = key_of(2 * kKeys + i);
    rht->lookup_batch(keys.data(), 2 * kKeys, res.data());
    for (auto& r : res) assert(r.size() == 1);
    assert(c.down_events.load() == 0);
    TEST_PASS();
}

static void test_slow_master_write() {
    TEST_BEGIN("a master too slow to take its share of a batch fails alone");
    PeerFaultCounters c{};
    const std::string prefix = prefix_of("slow") + std::string(1000, 'p') + ":";
    auto rht = make_redis_rht({g_seeds, prefix, 50}, 0, &c);
    const int n = 3 * 16384;
    std::vector<RhtRemoveEntry> own(n);
    for (int i = 0; i < n; i++) own[i] = {key_of(i), 0, (uint32_t)i};
    signal_masters(SIGSTOP, 1, 1);
    std::thread resume([] {
        pause_for(1.5);
        signal_masters(SIGCONT, 1, 1);
    });
    rht->reset(own.data(), n);   // ~5 MB per master per chunk: the stopped one cannot take it
    resume.join();
    assert(c.down_events.load() == 1);
    TEST_PASS();
}

static void test_reset_held_remove() {
    TEST_BEGIN("a list reset also removes the routes whose removal was held");
    const std::string& seeds = g_seeds;
    const uint64_t k = key_of(index_in_slots(prefix_of("heldreset"), 6000, 10000, 0));
    auto rht = make_redis_rht({seeds, prefix_of("heldreset"), 50}, 1, nullptr);
    RhtInsertEntry e{k, 1, 7, 4};
    rht->insert_batch(&e, 1);
    rht->flush();
    signal_masters(SIGSTOP, 1, 1);
    std::vector<RHTValue> res;
    rht->lookup_batch(&k, 1, &res);   // times out: the master is marked down
    RhtRemoveEntry d{k, 1, 7};
    rht->remove_batch(&d, 1);         // held
    std::thread resume([] {
        pause_for(1.0);
        signal_masters(SIGCONT, 1, 1);
    });
    rht->reset(nullptr, 0);
    resume.join();
    auto check = make_redis_rht({seeds, prefix_of("heldreset"), 1000}, 2, nullptr);
    check->lookup_batch(&k, 1, &res);
    assert(res.empty());
    TEST_PASS();
}

static void test_failover() {
    TEST_BEGIN("a killed master's replica takes over; writes held meanwhile land after");
    PeerFaultCounters c{};
    // Keys of the master killed below (it starts with slots 5461-10922): one whose
    // node the tree drops while the master is gone, one whose node comes back, and
    // one that a client started after the kill writes before it knows.
    const uint64_t ie = index_in_slots(prefix_of("failover"), 6000, 10000, 1u << 20);
    const uint64_t ir = index_in_slots(prefix_of("failover"), 6000, 10000, ie + 1);
    const uint64_t evicted = key_of(ie), reborn = key_of(ir);
    const uint64_t late = key_of(index_in_slots(prefix_of("failover"), 6000, 10000, ir + 1));
    PeerFaultCounters c2{};
    std::unique_ptr<RhtBackend> newcomer;
    auto rht = filled(g_seeds, "failover", &c,
                      [evicted](uint64_t k, uint32_t) { return k != evicted; });
    RhtInsertEntry r{reborn, 1, 7, 4};
    rht->insert_batch(&r, 1);
    rht->flush();
    for (int i = 0; i < 300 && !replicas_synced(); i++) pause_for(0.1);   // a resyncing replica cannot take over
    std::vector<RhtInsertEntry> fresh(kKeys);
    for (int i = 0; i < kKeys; i++) fresh[i] = {key_of(kKeys + i), 1, (uint32_t)i, 4};
    auto calls = probe(*rht, 14.0, [&] {
        pause_for(1.0);
        signal_masters(SIGKILL, 1, 1);
        pause_for(0.5);
        RhtInsertEntry e{evicted, 1, 8, 4};
        RhtRemoveEntry d{reborn, 1, 7};
        rht->insert_batch(&e, 1);
        rht->remove_batch(&d, 1);
        rht->insert_batch(fresh.data(), kKeys);
        newcomer = make_redis_rht({g_seeds, prefix_of("failover"), 50}, 1, &c2);
        RhtInsertEntry x{late, 1, 9, 4};
        newcomer->insert_batch(&x, 1);
    });
    rht->flush();
    newcomer->flush();
    assert(slowest_ms(calls) < 250);
    assert(found_share(calls, 1.3, 5.0) < 0.9);
    assert(found_share(calls, 12.5, 14.0) == 1.0);
    std::vector<uint64_t> keys(2 * kKeys + 2);
    std::vector<std::vector<RHTValue>> res(2 * kKeys + 2);
    for (int i = 0; i < 2 * kKeys; i++) keys[i] = key_of(i);
    keys[2 * kKeys] = evicted;
    keys[2 * kKeys + 1] = reborn;
    rht->lookup_batch(keys.data(), 2 * kKeys + 2, res.data());
    for (int i = 0; i < 2 * kKeys; i++) assert(res[i].size() == 1);
    assert(res[2 * kKeys].empty());
    assert(res[2 * kKeys + 1].size() == 1 && res[2 * kKeys + 1][0].tree_node_id == 7);
    std::vector<RHTValue> got;
    newcomer->lookup_batch(&late, 1, &got);
    assert(got.size() == 1 && got[0].tree_node_id == 9);
    assert(c2.rht_writes.load() == 0 && c2.down_events.load() == 1);
    TEST_PASS();
}

int main() {
    printf("=== test_rht_redis_faults ===\n\n");
    const char* bin = std::getenv("RADIXSHMEM_TEST_REDIS_BIN");
    if (!bin || !*bin) {
        printf("  (skipped: RADIXSHMEM_TEST_REDIS_BIN is not set)\n");
        return 77;
    }
    g_bin = bin;
    g_cli = g_bin + (::access((g_bin + "/redis-cli").c_str(), X_OK) == 0 ? "/redis-cli" : "/valkey-cli");
    std::string why = redis_rht_check({"127.0.0.1:1", "", 1000});
    if (why.find("without hiredis") != std::string::npos) {
        printf("  (skipped: %s)\n", why.c_str());
        return 77;
    }
    why = "its nodes did not start";
    if (start_cluster())   // a new master reports ok only after up to the node timeout
        for (int t = 0; t < 200 && !(why = cluster_ok()).empty(); t++) pause_for(0.1);
    if (!why.empty()) {
        fprintf(stderr, "test cluster at %s did not come up: %s\n", g_seeds.c_str(), why.c_str());
        stop_cluster();
        return 1;
    }
    test_hung_master();
    test_hung_cluster();
    test_unread_replies();
    test_reset_after_move();
    test_killed_connections();
    test_slow_master_write();
    test_reset_held_remove();
    test_failover();
    stop_cluster();
    return tests_report();
}

#else // RADIXSHMEM_NO_RDMA

int main() {
    printf("=== test_rht_redis_faults ===\n\n  (skipped: built without RDMA)\n");
    return 77;
}

#endif
