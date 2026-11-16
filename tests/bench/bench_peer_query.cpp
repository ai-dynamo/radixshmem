// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// bench_peer_query — latency of step two of a distributed query, the peer index:
// one-sided RDMA walks against the peer's gRPC Query/Unlock remote ops.
//
// node0 holds every prefix, the reader on node1 has an empty tree. Two parts:
//   index   PeerIndexBackend::query and ::unlock alone, both backends in one
//           process on the same servers and data. The servers run the rdma
//           backend, so their trees are registered for walks; every server
//           serves Query/Unlock over gRPC anyway. The walk gets the per-node
//           hints the RHT hands it in a real query.
//   client  RadixClient::query(lock) + finalize end to end, RHT probe included,
//           on a cluster started with the backend under test.
//
// Single host (private etcd, both nodes local, the RDMA RHT unless RADIXSHMEM_RHT_BACKEND):
//     ./bench_peer_query <index_dev> [gid_idx]
// Two hosts, one etcd: run the roles by hand (the reader waits for the hints file),
//     --server <node> <registry> <cluster> <rpc_addr>
//     --writer <node> <registry> <cluster> <hints_file>
//     --reader <node> <registry> <cluster> <hints_file> index|client [owner pid]
// Every role takes RADIXSHMEM_PEER_INDEX_BACKEND, RADIXSHMEM_RHT_* and the RDMA device
// (RADIXSHMEM_INDEX_DEV / RADIXSHMEM_INDEX_GID_IDX) from the env.

#include "bench_util.h"
#include "cluster_protocol.h"
#include "etcd_test_harness.hpp"
#include "local.h"
#include "peer_index.h"
#include "peer_registry.hpp"
#include "radix_client.h"
#include "radix_server.h"
#include "registry_client.hpp"

#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <numeric>
#include <string>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace radixshmem;

namespace {

constexpr int         kBootstrapTO = 60;
constexpr int         kChunk = 64;                      // blocks per node (block_size 1)
constexpr const char* kShm = "/bench_pq";
const int             kLens[] = {64, 512, 2048, 8192};  // 1, 8, 32, 128 nodes
const int             kThreads[] = {1, 8, 32};
constexpr double      kSecs = 3.0;

int prefixes_for(int len) { return std::max(8, std::min(64, (1 << 18) / len)); }

std::vector<uint64_t> prefix_hashes(int len, int i) {
    std::vector<int64_t> tok(len);
    std::iota(tok.begin(), tok.end(), (int64_t)len * 1000000 + (int64_t)i * 20000);
    return RadixClient::compute_hashes(tok.data(), len, 1);
}

RadixClientConfig client_cfg(const std::string& node, const std::string& registry,
                             const std::string& cluster) {
    RadixClientConfig cc;
    cc.registry = registry;
    cc.cluster_id = cluster;
    cc.node_name = node;
    cc.bootstrap_timeout_sec = kBootstrapTO;
    return cc;
}

volatile sig_atomic_t g_term = 0;
void on_term(int) { g_term = 1; }

int run_server(const std::string& node, const std::string& registry, const std::string& cluster,
               const std::string& addr) {
    ::signal(SIGTERM, on_term);
    ::signal(SIGINT, on_term);
    RadixServerConfig cfg;
    cfg.name = kShm;
    cfg.shm.hugepage_path = "";
    cfg.shm.max_blocks = 1 << 19;
    cfg.shm.max_nodes = 1 << 16;
    cfg.shm.block_size = 1;
    cfg.shm.background_evict_ratio = 0.0f;   // no evictor thread in the server's CPU time
    cfg.register_chunk_tokens = kChunk;
    cfg.registry = registry;
    cfg.cluster_id = cluster;
    cfg.node_name = node;
    cfg.rpc_address = addr;
    cfg.expected_min_nodes = 2;
    cfg.bootstrap_timeout_sec = kBootstrapTO;
    RadixServer srv(cfg);
    if (!srv.bootstrap()) return 10;
    fprintf(stderr, "[server %s] rank %d up, pid %d\n", node.c_str(), srv.rank(), (int)::getpid());
    while (!g_term) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    return 0;
}

// Hints file: the owner's rank, then per prefix (kLens order) the node starts on its path.
int run_writer(const std::string& node, const std::string& registry, const std::string& cluster,
               const std::string& hints_file) {
    RadixClient cli(client_cfg(node, registry, cluster));
    LocalTreeConnector tree(cli.name());
    const std::string tmp = hints_file + ".tmp";
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f) return 11;
    const int32_t owner = cli.rank();
    fwrite(&owner, sizeof(owner), 1, f);
    for (int len : kLens) {
        for (int i = 0; i < prefixes_for(len); i++) {
            auto h = prefix_hashes(len, i);
            std::vector<int32_t> slots;
            if (cli.allocate_slots(len, slots) != len) return 12;
            InsertResult r = cli.insert(h.data(), len, slots.data(), len);
            if (r.error != InsertError::OK || !r.unused_slots.empty()) return 13;
            std::vector<NodeBoundary> b = tree.tree().get_node_boundaries(h.data(), len);
            const int32_t count = (int32_t)b.size();
            fwrite(&count, sizeof(count), 1, f);
            for (const NodeBoundary& x : b) {
                RemoteHit hit{x.offset, x.tree_node_id};
                fwrite(&hit, sizeof(hit), 1, f);
            }
        }
    }
    cli.flush();
    fclose(f);
    return ::rename(tmp.c_str(), hints_file.c_str()) == 0 ? 0 : 14;
}

struct Prefix {
    int len;
    std::vector<uint64_t> hashes;
    std::vector<RemoteHit> hints;
};

bool load_hints(const std::string& file, int& owner, std::vector<Prefix>& out) {
    for (int i = 0; i < 600 && ::access(file.c_str(), R_OK) != 0; i++)
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    FILE* f = fopen(file.c_str(), "rb");
    if (!f) return false;
    int32_t o = -1;
    bool ok = fread(&o, sizeof(o), 1, f) == 1;
    for (int len : kLens) {
        for (int i = 0; ok && i < prefixes_for(len); i++) {
            Prefix p{len, prefix_hashes(len, i), {}};
            int32_t count = 0;
            ok = fread(&count, sizeof(count), 1, f) == 1 && count > 0;
            p.hints.resize(ok ? count : 0);
            ok = ok && fread(p.hints.data(), sizeof(RemoteHit), count, f) == (size_t)count;
            out.push_back(std::move(p));
        }
    }
    fclose(f);
    owner = o;
    return ok;
}

double cpu_secs_self() {
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_stime.tv_sec +
           (ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) / 1e6;
}

double cpu_secs_of(int pid) {
    if (pid <= 0) return 0;
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/stat", pid);
    FILE* f = fopen(path, "r");
    if (!f) return 0;
    char buf[1024];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = '\0';
    const char* p = strrchr(buf, ')');   // fields after the command name
    unsigned long utime = 0, stime = 0;
    if (!p || sscanf(p + 2, "%*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %lu %lu",
                     &utime, &stime) != 2)
        return 0;
    return (double)(utime + stime) / (double)sysconf(_SC_CLK_TCK);
}

struct Pct { double p50 = 0, p90 = 0, p99 = 0, mean = 0; };

Pct summarize(std::vector<double>& v) {
    Pct p;
    if (v.empty()) return p;
    std::sort(v.begin(), v.end());
    auto at = [&](double q) { return v[(size_t)((double)(v.size() - 1) * q)]; };
    p.p50 = at(0.50);
    p.p90 = at(0.90);
    p.p99 = at(0.99);
    p.mean = std::accumulate(v.begin(), v.end(), 0.0) / (double)v.size();
    return p;
}

void print_header(const char* what) {
    printf("\n-- %s --\n", what);
    printf("%-7s %6s %5s %4s %9s %8s %8s %8s %8s %9s %9s %10s %10s\n", "backend", "blocks",
           "lock", "thr", "ops/s", "q_p50", "q_p90", "q_p99", "q_mean", "unl_p50", "unl_p99",
           "cli_cpu/op", "srv_cpu/op");
    fflush(stdout);
}

// `op` runs one query (and unlock) on prefix k, adding its latencies in us; false = wrong reply.
template <typename Op>
void measure(const char* backend, int len, bool lock, int threads, int owner_pid,
             const std::vector<const Prefix*>& set, Op op) {
    std::vector<std::vector<double>> q(threads), u(threads);
    std::atomic<bool> stop{false};
    std::atomic<long> bad{0};
    const double cli0 = cpu_secs_self(), srv0 = cpu_secs_of(owner_pid);
    const auto t0 = bench::now();
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; t++)
        pool.emplace_back([&, t] {
            for (size_t k = (size_t)t; !stop.load(std::memory_order_relaxed); k++)
                if (!op(*set[k % set.size()], q[t], u[t])) bad++;
        });
    std::this_thread::sleep_for(std::chrono::duration<double>(kSecs));
    stop = true;
    for (auto& th : pool) th.join();
    const double secs = bench::seconds_since(t0);
    const double cli = cpu_secs_self() - cli0, srv = cpu_secs_of(owner_pid) - srv0;
    std::vector<double> qa, ua;
    for (int t = 0; t < threads; t++) {
        qa.insert(qa.end(), q[t].begin(), q[t].end());
        ua.insert(ua.end(), u[t].begin(), u[t].end());
    }
    const double ops = (double)qa.size();
    Pct qp = summarize(qa), up = summarize(ua);
    printf("%-7s %6d %5s %4d %9.0f %8.1f %8.1f %8.1f %8.1f %9.1f %9.1f %10.1f %10s\n", backend,
           len, lock ? "yes" : "no", threads, ops / secs, qp.p50, qp.p90, qp.p99, qp.mean,
           up.p50, up.p99, ops > 0 ? cli * 1e6 / ops : 0.0,
           owner_pid > 0 ? std::to_string((int)(srv * 1e6 / std::max(ops, 1.0))).c_str() : "-");
    if (bad) printf("        !! %ld queries came back short or failed\n", bad.load());
    fflush(stdout);
}

int bench_index(const std::string& node, const std::string& registry, const std::string& cluster,
                int owner, int owner_pid, const std::vector<Prefix>& prefixes) {
    RegistryClientConfig rc;
    rc.registry = registry;
    rc.cluster_id = cluster;
    rc.node_name = node;
    rc.bootstrap_timeout_sec = kBootstrapTO;
    RegistryClient reg(rc);
    reg.connect();
    if (!reg.wait_for_ready()) return 21;
    ClusterTopology topo = reg.read_topology();
    PeerRegistry peers;
    peers.seed(topo.peers, topo.world_size, topo.my_rank);
    PeerIndexConfig pc;
    pc.has_expected = true;
    pc.expected = topo.peers[owner].component_config;

    print_header("index: PeerIndexBackend::query / unlock, same servers and data");
    for (uint32_t backend : {CLUSTER_PEER_INDEX_RDMA, CLUSTER_PEER_INDEX_GRPC}) {
        ShardMapMsg sm = topo.sm;
        sm.peer_index_backend = backend;
        std::unique_ptr<PeerIndexBackend> index = make_peer_index(sm, peers, pc);
        const char* name = backend == CLUSTER_PEER_INDEX_GRPC ? "grpc" : "rdma";
        if (!index->preconnect(owner)) return 22;
        for (int len : kLens) {
            std::vector<const Prefix*> set;
            for (const Prefix& p : prefixes)
                if (p.len == len) set.push_back(&p);
            for (bool lock : {false, true}) {
                RemoteQuery rq;
                rq.lock = lock;
                auto op = [&](const Prefix& p, std::vector<double>& q, std::vector<double>& u) {
                    auto a = bench::now();
                    QueryReply r = index->query(owner, p.hashes.data(), len, rq, p.hints);
                    auto b = bench::now();
                    q.push_back(bench::us(a, b));
                    const bool ok = r.status == RemoteStatus::OK && r.hit_length == len &&
                                    (int)r.slots.size() == len && r.pin.empty() == !lock;
                    if (!r.pin.empty()) {
                        index->unlock(p.hashes.data(), len, std::move(r.pin));
                        u.push_back(bench::us(b, bench::now()));
                    }
                    return ok;
                };
                for (int i = 0; i < 20; i++) {   // warm the channel / endpoint
                    std::vector<double> q, u;
                    op(*set[i % set.size()], q, u);
                }
                for (int threads : kThreads)
                    measure(name, len, lock, threads, owner_pid, set, op);
            }
        }
    }
    return 0;
}

int bench_client(const std::string& node, const std::string& registry, const std::string& cluster,
                 int owner_pid, const std::vector<Prefix>& prefixes) {
    RadixClient cli(client_cfg(node, registry, cluster));
    const char* env = ::getenv("RADIXSHMEM_PEER_INDEX_BACKEND");
    const std::string name = env ? env : "grpc";
    print_header(("client: RadixClient::query(lock) + finalize, " + name + " cluster").c_str());
    for (int len : kLens) {
        std::vector<const Prefix*> set;
        for (const Prefix& p : prefixes)
            if (p.len == len) set.push_back(&p);
        auto op = [&](const Prefix& p, std::vector<double>& q, std::vector<double>& u) {
            auto a = bench::now();
            ComponentQueryResult r = cli.query(p.hashes.data(), len, COMPONENT_MASK_FULL,
                                               /*local_only=*/false, /*lock=*/true,
                                               /*update_meta=*/false);
            auto b = bench::now();
            q.push_back(bench::us(a, b));
            const bool ok = r.status == RemoteStatus::OK && r.common_hit == len;
            if (r.finalize) {
                r.finalize();
                u.push_back(bench::us(b, bench::now()));
            }
            return ok;
        };
        for (int i = 0; i < 20; i++) {
            std::vector<double> q, u;
            op(*set[i % set.size()], q, u);
        }
        for (int threads : kThreads) measure(name.c_str(), len, true, threads, owner_pid, set, op);
    }
    return 0;
}

int run_reader(const std::string& node, const std::string& registry, const std::string& cluster,
               const std::string& hints_file, const std::string& mode, int owner_pid) {
    int owner = -1;
    std::vector<Prefix> prefixes;
    if (!load_hints(hints_file, owner, prefixes) || owner < 0) return 20;
    try {
        if (mode == "index")
            return bench_index(node, registry, cluster, owner, owner_pid, prefixes);
        return bench_client(node, registry, cluster, owner_pid, prefixes);
    } catch (const std::exception& e) {
        fprintf(stderr, "[reader] %s\n", e.what());
        return 23;
    }
}

pid_t spawn(const std::vector<std::string>& argv) {
    pid_t p = ::fork();
    if (p != 0) return p;
    std::vector<char*> av;
    for (auto& s : argv) av.push_back(const_cast<char*>(s.c_str()));
    av.push_back(nullptr);
    ::execv(av[0], av.data());
    _exit(127);
}

int wait_one(pid_t pid) {
    int status = 0;
    if (::waitpid(pid, &status, 0) != pid) return -1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : -2;
}

bool wait_shm_ready(const std::string& shm, int timeout_sec) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
    while (std::chrono::steady_clock::now() < deadline) {
        int fd = ::shm_open(shm.c_str(), O_RDONLY, 0);
        if (fd >= 0) {
            void* p = ::mmap(nullptr, sizeof(ShmHeader), PROT_READ, MAP_SHARED, fd, 0);
            ::close(fd);
            if (p != MAP_FAILED) {
                const uint64_t magic = static_cast<ShmHeader*>(p)->cluster_ready_magic.load();
                ::munmap(p, sizeof(ShmHeader));
                if (magic == CLUSTER_MANIFEST_MAGIC) return true;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return false;
}

} // namespace

int main(int argc, char** argv) {
    auto arg = [&](int i) { return std::string(argv[i]); };
    if (argc >= 6 && arg(1) == "--server")
        return run_server(arg(2), arg(3), arg(4), arg(5));
    if (argc >= 6 && arg(1) == "--writer")
        return run_writer(arg(2), arg(3), arg(4), arg(5));
    if (argc >= 7 && arg(1) == "--reader")
        return run_reader(arg(2), arg(3), arg(4), arg(5), arg(6),
                          argc >= 8 ? std::atoi(argv[7]) : 0);
    if (argc < 2) {
        fprintf(stderr, "usage: %s <index_dev> [gid_idx]   (or a --server/--writer/--reader role)\n",
                argv[0]);
        return 2;
    }
    const std::string dev = argv[1];
    const std::string gid = argc >= 3 ? argv[2] : "3";
    setenv("RADIXSHMEM_INDEX_DEV", dev.c_str(), 1);
    setenv("RADIXSHMEM_INDEX_GID_IDX", gid.c_str(), 1);
    if (!::getenv("RADIXSHMEM_RHT_BACKEND")) setenv("RADIXSHMEM_RHT_BACKEND", "rdma", 1);

    radixshmem_test::EtcdHarness etcd;
    if (!etcd.usable()) {
        fprintf(stderr, "no etcd binary — skipping\n");
        return radixshmem_test::kSkipExitCode;
    }
    printf("=== bench_peer_query (single host, dev=%s gid=%s, RHT %s, %g s per point) ===\n",
           dev.c_str(), gid.c_str(), ::getenv("RADIXSHMEM_RHT_BACKEND"), kSecs);
    int failures = 0;
    for (const char* backend : {"rdma", "grpc"}) {
        setenv("RADIXSHMEM_PEER_INDEX_BACKEND", backend, 1);
        const std::string cluster = etcd.cluster_id() + "_" + backend;
        const std::string hints = "/tmp/bench_pq_" + std::to_string(::getpid()) + "_" + backend;
        std::vector<pid_t> servers;
        for (int r = 0; r < 2; r++) {
            const std::string node = "node" + std::to_string(r);
            ::shm_unlink((std::string(kShm) + "_" + node).c_str());
            servers.push_back(spawn({argv[0], "--server", node, etcd.registry(), cluster,
                                     "127.0.0.1"}));
        }
        bool ready = true;
        for (int r = 0; r < 2; r++)
            ready = ready && wait_shm_ready(std::string(kShm) + "_node" + std::to_string(r),
                                            kBootstrapTO);
        int rc = ready ? wait_one(spawn({argv[0], "--writer", "node0", etcd.registry(), cluster,
                                         hints}))
                       : -3;
        const std::string owner_pid = std::to_string(servers[0]);
        if (rc == 0 && std::string(backend) == "rdma")
            rc = wait_one(spawn({argv[0], "--reader", "node1", etcd.registry(), cluster, hints,
                                 "index", owner_pid}));
        if (rc == 0)
            rc = wait_one(spawn({argv[0], "--reader", "node1", etcd.registry(), cluster, hints,
                                 "client", owner_pid}));
        if (rc != 0) {
            fprintf(stderr, "FAIL: %s cluster rc=%d\n", backend, rc);
            failures++;
        }
        for (pid_t p : servers) ::kill(p, SIGTERM);
        for (pid_t p : servers) wait_one(p);
        for (int r = 0; r < 2; r++) ::shm_unlink((std::string(kShm) + "_node" + std::to_string(r)).c_str());
        ::unlink(hints.c_str());
    }
    return failures ? 1 : 0;
}
