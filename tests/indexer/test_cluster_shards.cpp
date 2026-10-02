// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// M < N RHT-shard decoupling test.
//
// Verifies the split-shm + shard→holder design:
//   * world_size N = 3, num_shards M = 2, holders = {rank 0, rank 1}.
//   * rank 2 holds NO RHT shard — it must reach both holders' RHT shards via
//     RHT-plane SEND QPs and route inserts/lookups by key % M.
//   * A writer on rank 2 inserts a prefix; its RHT registrations land on
//     holder 0 and/or holder 1 (sharded by key % 2), NOT on rank 2.
//   * A reader on a different rank finds the full prefix via RHT lookup +
//     RDMA remote-walk against rank 2 (the tree owner).
//
// Loopback-only, same single-host RDMA NIC. Rank is registry-assigned from the
// sorted /peers order, so node0/node1/node2 map to ranks 0/1/2 deterministically
// (holders {0,1}, tree owner 2). Every server AND client is a re-exec'd image —
// the etcd Go wrapper's runtime does not survive a bare fork.

#include "shmradix/radix_client.h"
#include "shmradix/radix_server.h"
#include "shmradix/types.h"

#include "etcd_test_harness.hpp"
#include "test_utils.h"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <numeric>
#include <string>
#include <sys/mman.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

constexpr int    kWorldSize     = 3;
constexpr int    kNumShards     = 2;          // M < N
constexpr int    kBlockSize     = 1;
constexpr int    kPrefixLen     = 64;
constexpr int    kBootstrapTO   = 30;
constexpr const char* kShmPrefix = "/test_dcshard";
constexpr const char* kRdmaDev   = "mlx5_bond_0";
constexpr int    kGidIdx        = 3;

std::string node_name_for(int rank) { return "node" + std::to_string(rank); }

std::string shm_name_for(int rank) {
    return std::string(kShmPrefix) + "_" + node_name_for(rank);
}

// allocate the shortfall then insert; slots bind to hashes[matched:] via
// start=matched and insert auto-recycles anything unused.
shmradix::InsertResult
insert_auto(shmradix::RadixClient& cli, const uint64_t* h, int n) {
    int matched = (int)cli.query(h, n, shmradix::COMPONENT_MASK_FULL, /*local_only=*/true,
                                 /*lock=*/false, /*update_meta=*/false)
                      .common_hit;
    int need = n - matched;
    std::vector<int32_t> slots;
    if (need > 0) cli.allocate_slots(need, slots);
    return cli.insert(h, n, slots.data(), (int)slots.size(),
                      /*start=*/matched, /*auto_recycle=*/true);
}

shmradix::RadixClientConfig client_cfg(const std::string& node,
                                       const std::string& registry,
                                       const std::string& cluster_id) {
    shmradix::RadixClientConfig cc;
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
            void* p = ::mmap(nullptr, sizeof(shmradix::ShmHeader), PROT_READ,
                             MAP_SHARED, fd, 0);
            ::close(fd);
            if (p != MAP_FAILED) {
                auto* hdr = static_cast<shmradix::ShmHeader*>(p);
                uint64_t magic =
                    hdr->cluster_ready_magic.load(std::memory_order_acquire);
                ::munmap(p, sizeof(shmradix::ShmHeader));
                if (magic == shmradix::CLUSTER_MANIFEST_MAGIC) return true;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return false;
}

volatile sig_atomic_t g_term = 0;
void on_term(int) { g_term = 1; }

int run_server_process(const std::string& node, const std::string& registry,
                       const std::string& cluster_id) {
    ::signal(SIGTERM, on_term);
    ::signal(SIGINT,  on_term);

    shmradix::RadixServerConfig cfg;
    cfg.name        = kShmPrefix;
    cfg.shm.hugepage_path = "";
    cfg.shm.max_blocks = 4096;
    cfg.shm.block_size = kBlockSize;
    cfg.num_rht_shards  = kNumShards;            // M = 2
    cfg.rht_shard_holders = {0, 1};              // holders are ranks 0 and 1 (leader-authoritative)
    cfg.expected_min_nodes = kWorldSize;
    cfg.registry    = registry;
    cfg.cluster_id  = cluster_id;
    cfg.node_name   = node;
    cfg.rpc_address = "127.0.0.1";   // single-host peers connect over loopback
    cfg.index_dev    = kRdmaDev;
    cfg.gid_idx     = kGidIdx;
    cfg.bootstrap_timeout_sec = kBootstrapTO;

    shmradix::RadixServer srv(cfg);
    if (!srv.bootstrap()) {
        fprintf(stderr, "[server %s] bootstrap failed\n", node.c_str());
        return 10;
    }
    fprintf(stderr, "[server %s] up; rank=%d/%d shm=%s\n",
            node.c_str(), srv.rank(), srv.world_size(), srv.shm_name().c_str());
    while (!g_term) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    return server_healthy(srv, node.c_str()) ? 0 : kHealthExitCode;
}

int run_writer_process(const shmradix::RadixClientConfig& cc) {
    try {
        shmradix::RadixClient cli(cc);
        if (!cli.is_distributed()) { fprintf(stderr, "[writer] not distributed\n"); return 11; }
        if (cli.world_size() != kWorldSize) { fprintf(stderr, "[writer] world=%d\n", cli.world_size()); return 12; }

        std::vector<int64_t> tokens(kPrefixLen);
        std::iota(tokens.begin(), tokens.end(), int64_t{3000});
        auto hashes = shmradix::RadixClient::compute_hashes(
            tokens.data(), (int)tokens.size(), kBlockSize);

        auto ins = insert_auto(cli, hashes.data(), (int)hashes.size());
        if (ins.error != shmradix::InsertError::OK ||
            !ins.unused_slots.empty()) {
            fprintf(stderr, "[writer] insert incomplete unused=%zu\n",
                    ins.unused_slots.size());
            return 13;
        }
        cli.flush();
        fprintf(stderr, "[writer] inserted %d hashes from non-holder rank 2\n", kPrefixLen);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[writer] exception: %s\n", e.what());
        return 14;
    }
}

int run_reader_process(const shmradix::RadixClientConfig& cc) {
    try {
        shmradix::RadixClient cli(cc);
        if (!cli.is_distributed()) { fprintf(stderr, "[reader] not distributed\n"); return 21; }

        std::vector<int64_t> tokens(kPrefixLen);
        std::iota(tokens.begin(), tokens.end(), int64_t{3000});
        auto hashes = shmradix::RadixClient::compute_hashes(
            tokens.data(), (int)tokens.size(), kBlockSize);

        auto qr = cli.query(hashes.data(), (int)hashes.size(), shmradix::COMPONENT_MASK_FULL);
        bool ok = true;
        if (qr.common_hit != kPrefixLen) {
            fprintf(stderr, "[reader] total_hit=%d expected %d\n", qr.common_hit, kPrefixLen);
            ok = false;
        }
        if (remote_src(qr) != 2) {
            fprintf(stderr, "[reader] remote_src=%d expected 2\n", remote_src(qr));
            ok = false;
        }
        if (qr.rdma_read_count <= 0) {
            fprintf(stderr, "[reader] rdma_reads=%d expected >0\n", qr.rdma_read_count);
            ok = false;
        }
        if (!ok) return 22;
        fprintf(stderr, "[reader] OK total=%d remote_src=%d rdma_reads=%d (M=%d)\n",
                qr.common_hit, remote_src(qr), qr.rdma_read_count, kNumShards);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[reader] exception: %s\n", e.what());
        return 23;
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

int run_client_case(const char* self, const std::string& cs, int node_rank,
                    const std::string& registry, const std::string& cluster_id) {
    pid_t p = spawn({self, "--client", cs, node_name_for(node_rank),
                     registry, cluster_id});
    if (p < 0) return -1;
    int status = 0;
    if (::waitpid(p, &status, 0) != p) return -1;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    fprintf(stderr, "[parent] client %s abnormal status=%d\n", cs.c_str(), status);
    return -2;
}

int client_main(const std::string& cs, const std::string& node,
                const std::string& registry, const std::string& cluster_id) {
    auto cc = client_cfg(node, registry, cluster_id);
    if (cs == "writer") return run_writer_process(cc);
    if (cs == "reader") return run_reader_process(cc);
    fprintf(stderr, "[client] unknown case '%s'\n", cs.c_str());
    return 90;
}

} // namespace

int main(int argc, char** argv) {
    // Re-exec'd server: --server <node> <registry> <cluster_id>.
    if (argc >= 5 && std::strcmp(argv[1], "--server") == 0)
        return run_server_process(argv[2], argv[3], argv[4]);
    // Re-exec'd client: --client <case> <node> <registry> <cluster_id>.
    if (argc >= 6 && std::strcmp(argv[1], "--client") == 0)
        return client_main(argv[2], argv[3], argv[4], argv[5]);

    shmradix_test::EtcdHarness etcd;
    if (!etcd.usable()) {
        fprintf(stderr, "[parent] no etcd binary (set SHMRADIX_ETCD_BIN or "
                        "put etcd on PATH) — skipping\n");
        return shmradix_test::kSkipExitCode;
    }
    const std::string registry   = etcd.registry();
    const std::string cluster_id = etcd.cluster_id();

    for (int r = 0; r < kWorldSize; r++) ::shm_unlink(shm_name_for(r).c_str());

    std::vector<pid_t> server_pids(kWorldSize);
    for (int r = 0; r < kWorldSize; r++) {
        pid_t p = spawn({argv[0], "--server", node_name_for(r), registry, cluster_id});
        if (p < 0) return 30;
        server_pids[r] = p;
    }
    int unhealthy = 0;   // servers whose end-of-run check_health failed
    auto kill_servers = [&]() {
        for (pid_t p : server_pids) if (p > 0) ::kill(p, SIGTERM);
        unhealthy += reap_servers(server_pids);
        for (int r = 0; r < kWorldSize; r++) ::shm_unlink(shm_name_for(r).c_str());
    };

    for (int r = 0; r < kWorldSize; r++) {
        if (!wait_shm_ready(shm_name_for(r), kBootstrapTO)) {
            fprintf(stderr, "[parent] rank %d not ready\n", r);
            kill_servers();
            return 31;
        }
    }

    int writer_rc = run_client_case(argv[0], "writer", 2, registry, cluster_id);
    int reader_rc = run_client_case(argv[0], "reader", 0, registry, cluster_id);

    kill_servers();

    if (writer_rc != 0) { fprintf(stderr, "FAIL: writer rc=%d\n", writer_rc); return writer_rc ? writer_rc : 40; }
    if (reader_rc != 0) { fprintf(stderr, "FAIL: reader rc=%d\n", reader_rc); return reader_rc ? reader_rc : 41; }
    if (unhealthy) { fprintf(stderr, "FAIL: %d server(s) leaked (check_health)\n", unhealthy); return 42; }
    fprintf(stderr, "PASS: test_cluster_shards (N=%d, M=%d)\n", kWorldSize, kNumShards);
    return 0;
}
