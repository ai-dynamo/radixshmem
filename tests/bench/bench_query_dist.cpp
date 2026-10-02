// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// bench_query_dist — cross-node (RDMA remote-walk) query latency vs tree depth.
//
// Single-host simulation of two ranks: rank 0 holds the data, rank 1 is
// locally empty, so every hit is served by an RDMA one-sided remote walk.
// The local baseline is bench_query (scenario 6, long prefix hits).
//
// Two sweeps:
//   1. DEPTH   — one stream extended by D inserts of one B-block node each,
//                giving a D-node chain (B divides the 128-block probe stride).
//   2. CONTEXT — one insert of an L-block context (10k..200k); the tree splits
//                it into 128-block nodes, so node count is L/128.
//
// Usage (the RDMA device is required):
//     ./bench_query_dist <index_dev> [gid_idx]
//     ./bench_query_dist mlx5_0 3
//
// Reported per point: nodes walked, coverage, ops/s, latency percentiles and
// RDMA reads / atomics per query.

#include "bench_util.h"
#include "shmradix/radix_client.h"
#include "shmradix/radix_server.h"
#include "shmradix/types.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace shmradix;
using bench::HashGen;
using bench::LatencyCollector;
using bench::WorkloadConfig;
using bench::now;
using bench::seconds_since;

namespace {

constexpr int         kWorldSize      = 2;
constexpr int         kBlockSize      = 1;
constexpr int         kRegisterChunk  = 128;   // blocks per RHT registration chunk (= node cap)
constexpr int         kBootstrapTO    = 30;
constexpr const char* kShmPrefix      = "/bench_qd";
constexpr int         kMasterPortBase = 27100;
constexpr double      kDurationSec    = 5.0;

// Constant node size (blocks per node). Must divide kRegisterChunk
// (=128) so every 128-block reader probe lands on a node boundary. With B=8 the
// deepest sweep point (5000 nodes = 40000 blocks) yields ~312 RHT hints — well
// past the old 64-hint cap, exercising the all-hints batched walk.
constexpr int kBlocksPerNode = 8;

// Depth sweep: number of radix nodes the remote walk must traverse.
const int kDepths[] = {100, 500, 1000, 2000, 5000};

// Context sweep: total context length in BLOCKS, inserted in ONE call (not by
// progressive inserts). The tree auto-chunks each insert into cap=128-block
// nodes, so node count is emergent (L/128): 10k→79, 50k→391, 100k→782,
// 200k→1563 nodes. "Don't think about depth" — just a long single insert.
const int kCtxBlocks[] = {10000, 50000, 100000, 200000};

// RDMA config from argv (required device — see file header).
std::string g_index_dev;
int         g_gid_idx = 3;

std::string shm_name_for(const std::string& base, int rank) {
    return base + "_" + std::to_string(rank);
}

// allocate the shortfall then insert; slots bind hashes[matched:] via
// start=matched, insert auto-recycles any unused tail slots.
InsertResult insert_auto(RadixClient& cli, const uint64_t* h, int n) {
    int matched = (int)cli.query(h, n, COMPONENT_MASK_FULL,
                                 /*local_only=*/true,
                                 /*lock=*/false, /*update_meta=*/false)
                      .common_hit;
    int need = n - matched;
    std::vector<int32_t> slots;
    if (need > 0) cli.allocate_slots(need, slots);
    return cli.insert(h, n, slots.data(), (int)slots.size(),
                      /*start=*/matched, /*auto_recycle=*/true);
}

// hugepage-first probe, matching ShmRadixTree::attach() — a shm_open-only
// probe misses segments the server built on /dev/hugepages.
int open_cluster_shm(const std::string& shm_name) {
    std::string base = (!shm_name.empty() && shm_name[0] == '/')
                     ? shm_name.substr(1) : shm_name;
    int fd = ::open(("/dev/hugepages/" + base).c_str(), O_RDONLY);
    if (fd >= 0) return fd;
    return ::shm_open(shm_name.c_str(), O_RDONLY, 0);
}

bool wait_for_cluster_ready(const std::string& shm_name, int timeout_sec) {
    auto deadline = std::chrono::steady_clock::now()
                  + std::chrono::seconds(timeout_sec);
    int fd = -1;
    while (true) {
        fd = open_cluster_shm(shm_name);
        if (fd >= 0) break;
        if (std::chrono::steady_clock::now() > deadline) {
            fprintf(stderr, "[parent] timed out opening %s\n", shm_name.c_str());
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    void* base = ::mmap(nullptr, sizeof(ShmHeader), PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (base == MAP_FAILED) {
        fprintf(stderr, "[parent] mmap %s failed: %s\n",
                shm_name.c_str(), strerror(errno));
        return false;
    }
    auto* hdr = static_cast<ShmHeader*>(base);
    while (true) {
        if (hdr->cluster_ready_magic.load(std::memory_order_acquire)
                == CLUSTER_MANIFEST_MAGIC) {
            ::munmap(base, sizeof(ShmHeader));
            return true;
        }
        if (std::chrono::steady_clock::now() > deadline) {
            ::munmap(base, sizeof(ShmHeader));
            fprintf(stderr, "[parent] timed out on ready_magic for %s\n",
                    shm_name.c_str());
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

volatile sig_atomic_t g_term = 0;
void on_term(int) { g_term = 1; }

// ---- Workload: ONE deterministic stream of L blocks (a single deep context) ----
WorkloadConfig make_wl_len(int L) {
    WorkloadConfig w;
    w.prefix_len_blocks    = L;
    w.shared_prefix_blocks = 0;   // single stream: whole path is deterministic
    w.num_unique_streams   = 1;
    // w.seed defaults to 0xC0FFEE — identical in writer + reader.
    return w;
}

int run_server_process(const std::string& base, int rank, int master_port,
                       uint32_t max_blocks, uint32_t max_nodes) {
    ::signal(SIGTERM, on_term);
    ::signal(SIGINT,  on_term);

    RadixServerConfig cfg;
    cfg.name           = base;
    cfg.shm.max_blocks = max_blocks;
    cfg.shm.max_nodes  = max_nodes;
    cfg.shm.block_size = kBlockSize;
    cfg.register_chunk_tokens = kRegisterChunk * kBlockSize;
    cfg.rank           = rank;
    cfg.world_size     = kWorldSize;
    cfg.registry       = "etcd://127.0.0.1:2379";
    cfg.cluster_id     = "bench_query_dist";
    cfg.node_name      = "node" + std::to_string(rank);
    (void)master_port;
    cfg.index_dev       = g_index_dev;
    cfg.gid_idx        = g_gid_idx;
    cfg.bootstrap_timeout_sec = kBootstrapTO;

    RadixServer srv(cfg);
    if (!srv.bootstrap()) {
        fprintf(stderr, "[server rank %d] bootstrap failed (dev=%s gid=%d)\n",
                rank, g_index_dev.c_str(), g_gid_idx);
        return 10;
    }
    while (!g_term) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    return 0;
}

// Writer: attach rank-0 shm and build the matched chain from ONE hash array.
//
//   single_insert=false (depth sweep): D progressively-longer prefixes, each
//     appending exactly one B-block node (path compression) → a D-node chain of
//     B-block nodes. B is the per-step increment.
//   single_insert=true (context sweep): ONE insert of the full L-block context.
//     The tree chunks it internally into cap=min(register_chunk_size=128,
//     MAX_DATA_BLOCKS)=128-block nodes → ~L/128 nodes, built in a single call.
//     Depth is emergent (L/128), not controlled — the point of this mode.
//
// Cluster insert publishes every new node's first_hash to rank-1's RHT shard
// over RDMA. Then flush.
int run_writer_process(const std::string& base, const WorkloadConfig& wl,
                       int B, bool single_insert) {
    try {
        RadixClient cli(shm_name_for(base, 0));
        if (!cli.is_distributed()) {
            fprintf(stderr, "[writer] expected distributed mode\n");
            return 11;
        }
        HashGen gen(wl);
        const auto& h = gen.next();          // stream 0 @ counter 0, full length
        int L = wl.prefix_len_blocks;
        if (single_insert) {
            insert_auto(cli, h.data(), L);
        } else {
            for (int len = B; len <= L; len += B)
                insert_auto(cli, h.data(), len);
        }
        cli.flush();

        // Sanity: full-length query on the writer must now see the whole chain.
        auto qr = cli.query(h.data(), L, COMPONENT_MASK_FULL, /*local_only=*/true, false, false);
        fprintf(stderr, "[writer] built %d-block context (local hit=%d/%d)\n",
                L, qr.common_hit, L);
        return (qr.common_hit == L) ? 0 : 12;
    } catch (const std::exception& e) {
        fprintf(stderr, "[writer] exception: %s\n", e.what());
        return 13;
    }
}

struct RemoteStats {
    int64_t ops = 0, misses = 0, matched_sum = 0;
    int64_t rdma_reads = 0, rdma_atomics = 0;
    double  dur = 0, p50 = 0, p90 = 0, p99 = 0, maxv = 0;
};

// Reader: attach rank-1 shm (locally empty) and replay the SAME deep stream.
// Each query(local_only=false) misses locally and is served by an RDMA
// remote-walk against rank 0 — the path under measurement.
int run_reader_process(const std::string& base, const WorkloadConfig& wl,
                       double duration_sec, int result_fd) {
    try {
        RadixClient cli(shm_name_for(base, 1));
        if (!cli.is_distributed()) {
            fprintf(stderr, "[reader] expected distributed mode\n");
            return 21;
        }
        HashGen gen(wl);
        const auto& hr = gen.replay_stream(0);      // counter 0 → writer's array
        std::vector<uint64_t> h(hr.begin(), hr.end());
        int L = wl.prefix_len_blocks;

        LatencyCollector lat;
        RemoteStats st;
        auto t_start = now();
        while (seconds_since(t_start) < duration_sec) {
            auto t0 = now();
            auto qr = cli.query(h.data(), L, COMPONENT_MASK_FULL,
                                 /*local_only=*/false,
                                 /*lock=*/false, /*update_meta=*/true);
            auto t1 = now();
            lat.record(t0, t1);

            st.ops++;
            st.matched_sum += qr.common_hit;
            if (qr.common_hit == 0) st.misses++;
            st.rdma_reads   += qr.rdma_read_count;
            st.rdma_atomics += qr.rdma_atomic_count;
        }
        st.dur = seconds_since(t_start);
        lat.summarize(st.p50, st.p90, st.p99, st.maxv);

        if (::write(result_fd, &st, sizeof(st)) != (ssize_t)sizeof(st)) {
            fprintf(stderr, "[reader] result write failed\n");
            return 22;
        }
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[reader] exception: %s\n", e.what());
        return 23;
    }
}

int wait_one(pid_t pid, const char* who) {
    int status = 0;
    if (::waitpid(pid, &status, 0) != pid) {
        fprintf(stderr, "[parent] waitpid(%s) failed: %s\n", who, strerror(errno));
        return -1;
    }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    fprintf(stderr, "[parent] %s abnormal exit (status=%d)\n", who, status);
    return -2;
}

// Build a fresh 2-rank cluster for one sweep point, run writer + reader, and
// return the reader's stats. Every point gets its own shm base + master port so
// stale segments never collide across sweep iterations.
//
//   L            = context length in blocks (the matched prefix).
//   node_size    = blocks per radix node → expected node count = L / node_size.
//                  depth sweep: B (progressive inserts). context sweep: 128
//                  (single insert auto-chunks at cap=128).
//   single_insert= false: D progressive inserts. true: one insert of L blocks.
bool run_case(const std::string& tag, int L, int B, int node_size,
              bool single_insert, int port, RemoteStats& out) {
    std::string base = std::string(kShmPrefix) + "_" + tag;
    WorkloadConfig wl = make_wl_len(L);
    int est_nodes = L / node_size + 1;
    // rank-0 mempool holds the whole chain (4x headroom → no eviction); node
    // pool must hold ~est_nodes nodes (auto sizing assumes far fewer nodes/block).
    uint32_t max_blocks = (uint32_t)std::max(4096, 4 * L);
    uint32_t max_nodes  = (uint32_t)(2 * est_nodes + 4096);

    for (int r = 0; r < kWorldSize; r++)
        ::shm_unlink(shm_name_for(base, r).c_str());

    std::vector<pid_t> server_pids(kWorldSize);
    for (int r = 0; r < kWorldSize; r++) {
        pid_t p = ::fork();
        if (p < 0) { fprintf(stderr, "[parent] fork server %d failed\n", r); return false; }
        if (p == 0) _exit(run_server_process(base, r, port, max_blocks, max_nodes));
        server_pids[r] = p;
    }
    auto kill_servers = [&]() {
        for (pid_t p : server_pids) if (p > 0) ::kill(p, SIGTERM);
        for (pid_t p : server_pids) if (p > 0) ::waitpid(p, nullptr, 0);
        for (int r = 0; r < kWorldSize; r++)
            ::shm_unlink(shm_name_for(base, r).c_str());
    };

    for (int r = 0; r < kWorldSize; r++) {
        if (!wait_for_cluster_ready(shm_name_for(base, r), kBootstrapTO)) {
            kill_servers();
            return false;
        }
    }

    pid_t writer = ::fork();
    if (writer < 0) { kill_servers(); return false; }
    if (writer == 0) _exit(run_writer_process(base, wl, B, single_insert));
    int writer_rc = wait_one(writer, "writer");
    if (writer_rc != 0) {
        fprintf(stderr, "FAIL: writer rc=%d (%s)\n", writer_rc, tag.c_str());
        kill_servers();
        return false;
    }

    int pipefd[2];
    if (::pipe(pipefd) != 0) { kill_servers(); return false; }
    pid_t reader = ::fork();
    if (reader < 0) { kill_servers(); return false; }
    if (reader == 0) {
        ::close(pipefd[0]);
        int rc = run_reader_process(base, wl, kDurationSec, pipefd[1]);
        ::close(pipefd[1]);
        _exit(rc);
    }
    ::close(pipefd[1]);

    RemoteStats st{};
    bool got = (::read(pipefd[0], &st, sizeof(st)) == (ssize_t)sizeof(st));
    ::close(pipefd[0]);
    int reader_rc = wait_one(reader, "reader");
    kill_servers();

    if (reader_rc != 0 || !got) {
        fprintf(stderr, "FAIL: reader rc=%d got=%d (%s)\n",
                reader_rc, got, tag.c_str());
        return false;
    }
    out = st;
    return true;
}

void print_header() {
    printf("%-9s %5s %7s %8s %6s %8s %10s %11s %9s %9s %9s %11s %11s\n",
           "ctx_blk", "nd_sz", "exp_nd", "walk_nd", "cov%", "dur(s)", "ops",
           "ops/s", "p50_us", "p90_us", "p99_us", "rdma_rd/op", "rdma_at/op");
    bench::print_separator();
}

// ctx_blk = context length in blocks; node_size = blocks/node; exp_nd = expected
// node count on the path. walk_nd is derived from matched blocks / node_size.
void print_row(int ctx_blk, int node_size, int exp_nd, const RemoteStats& st) {
    double ops_ps  = st.dur > 0 ? (double)st.ops / st.dur : 0;
    double denom   = (double)std::max<int64_t>(1, st.ops);
    double walk_nd = ((double)st.matched_sum / denom) / node_size;
    double cov     = exp_nd > 0 ? 100.0 * walk_nd / exp_nd : 0;
    printf("%-9d %5d %7d %8.0f %6.1f %8.2f %10lld %11.0f %9.1f %9.1f %9.1f "
           "%11.2f %11.2f\n",
           ctx_blk, node_size, exp_nd, walk_nd, cov, st.dur,
           (long long)st.ops, ops_ps, st.p50, st.p90, st.p99,
           (double)st.rdma_reads / denom, (double)st.rdma_atomics / denom);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <index_dev> [gid_idx]\n"
                "  <index_dev>  RDMA device for the remote walk, e.g. mlx5_0\n"
                "  [gid_idx]   GID index (default 3)\n", argv[0]);
        return 2;
    }
    g_index_dev = argv[1];
    if (argc >= 3) g_gid_idx = std::atoi(argv[2]);

    printf("=== bench_query_dist (rank1 queries rank0 over RDMA; dev=%s gid=%d) ===\n",
           g_index_dev.c_str(), g_gid_idx);
    printf("    ctx_blk = context length in blocks; walk_nd = radix nodes walked remotely\n");

    int base_port = kMasterPortBase + (int)(::getpid() % 400);
    int port_seq  = 0;
    bool any_rdma = false;

    // Sweep 1 — DEPTH: chain built by D progressive inserts of B-block nodes.
    printf("\n-- depth sweep (progressive inserts, B=%d blocks/node) --\n",
           kBlocksPerNode);
    print_header();
    fflush(stdout);
    for (size_t i = 0; i < sizeof(kDepths) / sizeof(kDepths[0]); i++) {
        int depth = kDepths[i];
        int B = kBlocksPerNode;
        int L = depth * B;
        std::string tag = "d" + std::to_string(depth);
        fprintf(stderr, "[parent] depth=%d B=%d (%d blocks): building + measuring...\n",
                depth, B, L);
        RemoteStats st{};
        bool ok = run_case(tag, L, B, /*node_size=*/B, /*single_insert=*/false,
                           base_port + port_seq++, st);
        if (!ok) { printf("%-9d  (failed)\n", L); fflush(stdout); continue; }
        print_row(L, B, depth, st);
        fflush(stdout);
        if (st.rdma_reads > 0) any_rdma = true;
    }

    // Sweep 2 — CONTEXT: one long insert; tree auto-chunks at cap=128 blocks.
    printf("\n-- context sweep (single insert, auto-chunked to %d blocks/node) --\n",
           kRegisterChunk);
    print_header();
    fflush(stdout);
    for (size_t i = 0; i < sizeof(kCtxBlocks) / sizeof(kCtxBlocks[0]); i++) {
        int L = kCtxBlocks[i];
        int ns = kRegisterChunk;   // emergent node size
        int exp_nd = L / ns + 1;
        std::string tag = "c" + std::to_string(L);
        fprintf(stderr, "[parent] ctx=%d blocks (~%d nodes): building + measuring...\n",
                L, exp_nd);
        RemoteStats st{};
        bool ok = run_case(tag, L, /*B=*/ns, /*node_size=*/ns,
                           /*single_insert=*/true, base_port + port_seq++, st);
        if (!ok) { printf("%-9d  (failed)\n", L); fflush(stdout); continue; }
        print_row(L, ns, exp_nd, st);
        fflush(stdout);
        if (st.rdma_reads > 0) any_rdma = true;
    }

    if (!any_rdma) {
        fprintf(stderr,
                "\nWARNING: rdma_rd/op == 0 across all points — queries were NOT "
                "served over the NIC. Check that rank1 has no local copy and "
                "dev=%s is correct.\n", g_index_dev.c_str());
    }
    printf("\nDone.\n");
    return 0;
}
