// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Node-join integration test over etcd dynamic membership.
//
// Brings up a 2-node cluster (node0, node1; both RHT holders, num_shards=2),
// then spawns a third node as a data-only peer (join_existing). Verifies:
//
//   1. A reader on node0, attached while the cluster was 2 nodes, senses node2
//      through its /peers watcher (world_size() grows 2→3) with no re-attach.
//   2. That reader RDMA remote-walks a prefix written by node2, connecting to
//      node2's endpoint on first use.
//   3. remote_node_id == 2, common_hit covers the prefix, rdma reads > 0.
//
// Re-exec discipline (as test_cluster.cpp): the etcd Go-client wrapper starts
// its runtime at library-load, so every server AND client is a fresh exec.
//   --server <node> <registry> <cluster_id> <num_shards> <min_nodes>
//   --joined-reader <node> <registry> <cluster_id>
//   --writer2       <node> <registry> <cluster_id>
//
// Loopback-only — all processes run on the host that owns the RDMA NIC.

#include "radix_client.h"
#include "radix_server.h"
#include "types.h"

#include "etcd_test_harness.hpp"
#include "test_utils.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <numeric>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

constexpr int    kNumShards    = 2;    // holders freeze at {node0,node1}
constexpr int    kBlockSize    = 1;
constexpr int    kPrefixLen    = 16;   // prefix B written by node2
constexpr int    kBootstrapTO  = 30;
constexpr int    kJoinTO       = 40;   // reader's budget to sense node2 + hit
constexpr const char* kShmPrefix = "/test_join";
constexpr const char* kRdmaDev   = "mlx5_bond_0";
constexpr int    kGidIdx        = 3;

std::string node_name_for(int rank) { return "node" + std::to_string(rank); }
std::string shm_name_for(int rank) {
    return std::string(kShmPrefix) + "_" + node_name_for(rank);
}

// Prefix B, inserted by node2, remote-walked by node0. Distinct token base so it
// never collides with anything the baseline path might touch.
std::vector<uint64_t> prefix_b() {
    std::vector<int64_t> tok(kPrefixLen);
    std::iota(tok.begin(), tok.end(), int64_t{90000});
    return radixshmem::RadixClient::compute_hashes(
        tok.data(), (int)tok.size(), kBlockSize);
}

radixshmem::InsertResult
insert_auto(radixshmem::RadixClient& cli, const uint64_t* h, int n) {
    int matched = (int)cli.query(h, n, radixshmem::COMPONENT_MASK_FULL, /*local_only=*/true,
                                 /*lock=*/false, /*update_meta=*/false)
                      .common_hit;
    int need = n - matched;
    std::vector<int32_t> slots;
    if (need > 0) cli.allocate_slots(need, slots);
    return cli.insert(h, n, slots.data(), (int)slots.size(),
                      /*start=*/matched, /*auto_recycle=*/true);
}

radixshmem::RadixClientConfig client_cfg(const std::string& node,
                                       const std::string& registry,
                                       const std::string& cluster_id) {
    radixshmem::RadixClientConfig cc;
    cc.registry   = registry;
    cc.cluster_id = cluster_id;
    cc.node_name  = node;
    cc.index_dev   = kRdmaDev;
    cc.gid_idx    = kGidIdx;
    cc.bootstrap_timeout_sec = kBootstrapTO;
    return cc;
}

bool wait_shm_ready(const std::string& shm_name, int timeout_sec) {
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::seconds(timeout_sec);
    while (std::chrono::steady_clock::now() < deadline) {
        int fd = ::shm_open(shm_name.c_str(), O_RDONLY, 0);
        if (fd >= 0) {
            void* p = ::mmap(nullptr, sizeof(radixshmem::ShmHeader), PROT_READ,
                             MAP_SHARED, fd, 0);
            ::close(fd);
            if (p != MAP_FAILED) {
                auto* hdr = static_cast<radixshmem::ShmHeader*>(p);
                uint64_t magic =
                    hdr->cluster_ready_magic.load(std::memory_order_acquire);
                ::munmap(p, sizeof(radixshmem::ShmHeader));
                if (magic == radixshmem::CLUSTER_MANIFEST_MAGIC) return true;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return false;
}

volatile sig_atomic_t g_term = 0;
void on_term(int) { g_term = 1; }

int run_server_process(const std::string& node, const std::string& registry,
                       const std::string& cluster_id, int num_shards,
                       int min_nodes) {
    ::signal(SIGTERM, on_term);
    ::signal(SIGINT,  on_term);

    radixshmem::RadixServerConfig cfg;
    cfg.name        = kShmPrefix;
    cfg.shm.hugepage_path = "";
    cfg.shm.max_blocks = 4096;
    cfg.shm.block_size = kBlockSize;
    cfg.registry    = registry;
    cfg.cluster_id  = cluster_id;
    cfg.node_name   = node;
    cfg.rpc_address = "127.0.0.1";
    cfg.num_rht_shards  = num_shards;
    cfg.expected_min_nodes = min_nodes;
    cfg.index_dev    = kRdmaDev;
    cfg.gid_idx     = kGidIdx;
    cfg.bootstrap_timeout_sec = kBootstrapTO;

    radixshmem::RadixServer srv(cfg);
    if (!srv.bootstrap()) {
        fprintf(stderr, "[server %s] bootstrap failed\n", node.c_str());
        return 10;
    }
    fprintf(stderr, "[server %s] up; rank=%d/%d shm=%s\n",
            node.c_str(), srv.rank(), srv.world_size(), srv.shm_name().c_str());

    while (!g_term) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    fprintf(stderr, "[server %s] shutting down\n", node.c_str());
    return server_healthy(srv, node.c_str()) ? 0 : kHealthExitCode;
}

// node2 writer: insert prefix B into node2's local tree + flush the RHT write
// (node2 is a data peer; its RHT hint lands on a holder shard 0/1).
int run_writer2_process(const radixshmem::RadixClientConfig& cc) {
    try {
        radixshmem::RadixClient cli(cc);
        if (!cli.is_distributed()) {
            fprintf(stderr, "[writer2] expected distributed mode\n");
            return 11;
        }
        if (cli.rank() != 2) {
            fprintf(stderr, "[writer2] expected rank 2, got %d\n", cli.rank());
            return 12;
        }
        auto hashes = prefix_b();
        auto ins = insert_auto(cli, hashes.data(), (int)hashes.size());
        if (ins.error != radixshmem::InsertError::OK ||
            !ins.unused_slots.empty()) {
            fprintf(stderr, "[writer2] insert incomplete unused=%zu (err=%u)\n",
                    ins.unused_slots.size(), (unsigned)ins.error);
            return 13;
        }
        cli.flush();
        fprintf(stderr, "[writer2] inserted %d hashes on rank 2 and flushed\n",
                kPrefixLen);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[writer2] exception: %s\n", e.what());
        return 14;
    }
}

// Long-lived reader on node0: attaches while the cluster is 2 nodes, then must
// (a) sense node2 via its /peers watcher (world_size 2→3) and (b) remote-walk
// prefix B on node2 through the lazily-built data-peer endpoint.
int run_joined_reader_process(const radixshmem::RadixClientConfig& cc) {
    try {
        radixshmem::RadixClient cli(cc);
        if (!cli.is_distributed()) {
            fprintf(stderr, "[joined_reader] expected distributed mode\n");
            return 21;
        }
        if (cli.world_size() != 2) {
            fprintf(stderr, "[joined_reader] expected initial world=2, got %d\n",
                    cli.world_size());
            return 22;
        }

        // (a) Wait for the watcher to grow the peer table to include node2.
        auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(kJoinTO);
        while (cli.world_size() < 3 &&
               std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (cli.world_size() < 3) {
            fprintf(stderr, "[joined_reader] watcher never saw node2 "
                            "(world stuck at %d)\n", cli.world_size());
            return 23;
        }
        fprintf(stderr, "[joined_reader] watcher grew world → %d\n",
                cli.world_size());

        // (b) Remote-walk prefix B; node2's endpoint is built lazily here on
        // first query. Retry until node2's writer flush is visible.
        auto hashes = prefix_b();
        radixshmem::ComponentQueryResult qr;
        bool hit = false;
        while (std::chrono::steady_clock::now() < deadline) {
            qr = cli.query(hashes.data(), (int)hashes.size(), radixshmem::COMPONENT_MASK_FULL);
            if (qr.common_hit == kPrefixLen && remote_src(qr) == 2) {
                hit = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (!hit) {
            fprintf(stderr, "[joined_reader] no node2 hit: total=%d "
                            "remote_src=%d rdma_reads=%d\n",
                    qr.common_hit, remote_src(qr), qr.rdma_read_count);
            return 24;
        }
        if (qr.rdma_read_count <= 0) {
            fprintf(stderr, "[joined_reader] expected rdma_read_count>0 got %d\n",
                    qr.rdma_read_count);
            return 26;
        }
        fprintf(stderr, "[joined_reader] OK hit node2: total=%d remote_src=%d "
                        "rdma_reads=%d\n",
                qr.common_hit, remote_src(qr), qr.rdma_read_count);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[joined_reader] exception: %s\n", e.what());
        return 27;
    }
}

pid_t spawn(const std::vector<std::string>& argv) {
    pid_t p = ::fork();
    if (p != 0) return p;
    std::vector<char*> av;
    av.reserve(argv.size() + 1);
    for (auto& s : argv) av.push_back(const_cast<char*>(s.c_str()));
    av.push_back(nullptr);
    ::execv(av[0], av.data());
    fprintf(stderr, "[spawn] execv(%s) failed: %s\n", av[0], strerror(errno));
    _exit(127);
}

int wait_one(pid_t pid, const char* who) {
    int status = 0;
    if (::waitpid(pid, &status, 0) != pid) {
        fprintf(stderr, "[parent] waitpid(%s) failed: %s\n", who, strerror(errno));
        return -1;
    }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    fprintf(stderr, "[parent] %s did not exit normally (status=%d)\n", who, status);
    return -2;
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 7 && std::strcmp(argv[1], "--server") == 0) {
        return run_server_process(argv[2], argv[3], argv[4],
                                  std::atoi(argv[5]), std::atoi(argv[6]));
    }
    if (argc >= 5 && std::strcmp(argv[1], "--joined-reader") == 0) {
        return run_joined_reader_process(client_cfg(argv[2], argv[3], argv[4]));
    }
    if (argc >= 5 && std::strcmp(argv[1], "--writer2") == 0) {
        return run_writer2_process(client_cfg(argv[2], argv[3], argv[4]));
    }

    radixshmem_test::EtcdHarness etcd;
    if (!etcd.usable()) {
        fprintf(stderr, "[parent] no etcd binary (set RADIXSHMEM_ETCD_BIN or "
                        "put etcd on PATH) — skipping\n");
        return radixshmem_test::kSkipExitCode;
    }
    const std::string registry   = etcd.registry();
    const std::string cluster_id = etcd.cluster_id();

    for (int r = 0; r < 3; r++) ::shm_unlink(shm_name_for(r).c_str());

    // ---- node0, node1 (min_nodes=2, both holders) ----
    std::vector<pid_t> servers;
    for (int r = 0; r < 2; r++) {
        pid_t p = spawn({argv[0], "--server", node_name_for(r), registry,
                         cluster_id, std::to_string(kNumShards),
                         std::to_string(2)});
        if (p < 0) { fprintf(stderr, "[parent] spawn server %d failed\n", r); return 30; }
        servers.push_back(p);
    }

    int unhealthy = 0;   // servers whose end-of-run check_health failed
    auto cleanup = [&]() {
        for (pid_t p : servers) if (p > 0) ::kill(p, SIGTERM);
        unhealthy += reap_servers(servers);
        for (int r = 0; r < 3; r++) ::shm_unlink(shm_name_for(r).c_str());
    };

    for (int r = 0; r < 2; r++) {
        if (!wait_shm_ready(shm_name_for(r), kBootstrapTO)) {
            fprintf(stderr, "[parent] server %d never became cluster-ready\n", r);
            cleanup();
            return 31;
        }
    }
    fprintf(stderr, "[parent] initial 2-node cluster ready\n");

    // ---- long-lived reader on node0 (attaches at world=2) ----
    pid_t reader = spawn({argv[0], "--joined-reader", node_name_for(0),
                          registry, cluster_id});
    if (reader < 0) { fprintf(stderr, "[parent] spawn reader failed\n"); cleanup(); return 32; }

    // ---- node2 joins (min_nodes=3; config already present → join_existing) ----
    pid_t node2 = spawn({argv[0], "--server", node_name_for(2), registry,
                         cluster_id, std::to_string(kNumShards),
                         std::to_string(3)});
    if (node2 < 0) { fprintf(stderr, "[parent] spawn node2 failed\n"); cleanup(); return 33; }
    servers.push_back(node2);

    if (!wait_shm_ready(shm_name_for(2), kBootstrapTO)) {
        fprintf(stderr, "[parent] node2 never became cluster-ready\n");
        ::kill(reader, SIGTERM); ::waitpid(reader, nullptr, 0);
        cleanup();
        return 34;
    }
    fprintf(stderr, "[parent] node2 joined as data peer\n");

    // ---- write prefix B on node2, then the reader must remote-walk it ----
    int wrc = wait_one(spawn({argv[0], "--writer2", node_name_for(2),
                              registry, cluster_id}), "writer2");
    if (wrc != 0) {
        fprintf(stderr, "FAIL: writer2 rc=%d\n", wrc);
        ::kill(reader, SIGTERM); ::waitpid(reader, nullptr, 0);
        cleanup();
        return 41;
    }

    int rrc = wait_one(reader, "joined_reader");
    cleanup();
    if (rrc != 0) {
        fprintf(stderr, "FAIL: joined_reader rc=%d\n", rrc);
        return 42;
    }
    if (unhealthy) {
        fprintf(stderr, "FAIL: %d server(s) leaked (check_health)\n", unhealthy);
        return 43;
    }
    fprintf(stderr, "PASS: test_cluster_node_join\n");
    return 0;
}
