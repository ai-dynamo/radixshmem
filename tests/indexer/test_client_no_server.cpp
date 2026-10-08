// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Pure-client (observer) smoke test over etcd dynamic membership.
//
// Verifies the no-local-tree path of RadixClient: a client with
// no_local_tree=true and a node_name that never appears in /peers attaches no
// shm, joins the cluster as a remote-only reader, and gets rank -1.
//
//   1. Two RadixServer processes bootstrap through a private etcd.
//   2. A writer client (node0) inserts a prefix and updates the distributed RHT.
//   3. An OBSERVER client (node_name="observer", not a /peers member) queries
//      the same prefix: rank()==-1, and the hit comes back via RDMA remote-walk
//      (common_hit covers everything, remote_node_id >= 0, rdma reads > 0).
//   4. The observer's local-only ops (insert / allocate_slots / local query)
//      all raise a clear error instead of crashing or mis-inserting.
//
// Every process is a RE-EXEC'd image — the etcd Go-client wrapper starts its
// runtime at library-load, so a bare fork()'d child deadlocks on the first cgo
// call. main() dispatches on argv:
//   --server <node> <registry> <cluster_id> <num_shards> <min_nodes>
//   --client <case> <node> <registry> <cluster_id>
//
// Loopback-only — all processes run on the host that owns the RDMA NIC.

#include "radix_client.h"
#include "radix_server.h"
#include "types.h"

#include "etcd_test_harness.hpp"
#include "test_utils.h"

#include <atomic>
#include <cassert>
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

constexpr int    kWorldSize     = 2;
constexpr int    kBlockSize     = 1;
constexpr int    kPrefixLen     = 16;
constexpr int    kBootstrapTO   = 30;
constexpr const char* kShmPrefix = "/test_nosrv";
constexpr const char* kRdmaDev   = "mlx5_bond_0";
constexpr int    kGidIdx        = 3;
constexpr const char* kObserverNode = "observer";

std::string node_name_for(int rank) { return "node" + std::to_string(rank); }

std::string shm_name_for(int rank) {
    return std::string(kShmPrefix) + "_" + node_name_for(rank);
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

// Writer client (node0): inserts kPrefixLen hashes, flushes RHT writes, exits.
int run_writer_process(const radixshmem::RadixClientConfig& cc) {
    try {
        radixshmem::RadixClient cli(cc);
        if (!cli.is_distributed()) {
            fprintf(stderr, "[writer] expected distributed mode\n");
            return 11;
        }
        std::vector<int64_t> tokens(kPrefixLen);
        std::iota(tokens.begin(), tokens.end(), int64_t{1000});
        auto hashes = radixshmem::RadixClient::compute_hashes(
            tokens.data(), (int)tokens.size(), kBlockSize);

        auto ins = insert_auto(cli, hashes.data(), (int)hashes.size());
        if (ins.error != radixshmem::InsertError::OK ||
            !ins.unused_slots.empty()) {
            fprintf(stderr,
                    "[writer] insert incomplete (unused=%zu, err=%u)\n",
                    ins.unused_slots.size(), (unsigned)ins.error);
            return 12;
        }
        cli.flush();
        fprintf(stderr, "[writer] inserted %d hashes and flushed\n", kPrefixLen);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[writer] exception: %s\n", e.what());
        return 13;
    }
}

// Observer client: no local tree, node_name not in /peers. Must reach rank -1,
// hit the writer's prefix over RDMA remote-walk, and reject every local op.
int run_observer_process(const std::string& registry,
                         const std::string& cluster_id) {
    radixshmem::RadixClientConfig cc;
    cc.registry      = registry;
    cc.cluster_id    = cluster_id;
    cc.node_name     = kObserverNode;
    cc.index_dev      = kRdmaDev;
    cc.gid_idx       = kGidIdx;
    cc.bootstrap_timeout_sec = kBootstrapTO;
    cc.no_local_tree = true;

    try {
        radixshmem::RadixClient cli(cc);

        if (!cli.is_distributed()) {
            fprintf(stderr, "[observer] expected distributed mode\n");
            return 30;
        }
        if (cli.rank() != -1) {
            fprintf(stderr, "[observer] expected rank -1, got %d\n", cli.rank());
            return 31;
        }

        std::vector<int64_t> tokens(kPrefixLen);
        std::iota(tokens.begin(), tokens.end(), int64_t{1000});
        auto hashes = radixshmem::RadixClient::compute_hashes(
            tokens.data(), (int)tokens.size(), kBlockSize);

        auto qr = cli.query(hashes.data(), (int)hashes.size(), radixshmem::COMPONENT_MASK_FULL);
        bool ok = true;
        if (qr.common_hit != kPrefixLen) {
            fprintf(stderr, "[observer] expected total_hit=%d, got %d\n",
                    kPrefixLen, qr.common_hit);
            ok = false;
        }
        if (remote_src(qr) < 0) {
            fprintf(stderr, "[observer] expected remote fragment, got src=%d\n",
                    remote_src(qr));
            ok = false;
        }
        if (qr.rdma_read_count <= 0) {
            fprintf(stderr, "[observer] expected rdma_read_count>0, got %d\n",
                    qr.rdma_read_count);
            ok = false;
        }
        if (!ok) return 32;

        // Every local-tree op must raise, not crash or silently mis-insert.
        auto raises = [](auto&& fn, const char* what) -> bool {
            try {
                fn();
            } catch (const std::exception&) {
                return true;
            }
            fprintf(stderr, "[observer] %s did not raise on observer\n", what);
            return false;
        };

        bool guarded = true;
        guarded &= raises([&] {
            std::vector<int32_t> out;
            cli.allocate_slots(1, out);
        }, "allocate_slots");
        guarded &= raises([&] {
            int32_t slot = 0;
            cli.insert(hashes.data(), (int)hashes.size(), &slot, 1);
        }, "insert");
        {
            auto lr = cli.query(hashes.data(), (int)hashes.size(),
                                radixshmem::COMPONENT_MASK_FULL, /*local_only=*/true);
            if (lr.status != radixshmem::RemoteStatus::BAD_REQUEST) {
                fprintf(stderr,
                        "[observer] local-only query on observer: expected "
                        "BAD_REQUEST, got %d\n", (int)lr.status);
                guarded = false;
            }
        }
        if (!guarded) return 33;

        fprintf(stderr,
                "[observer] OK rank=-1 total=%d remote_src=%d rdma_reads=%d; "
                "local ops all rejected\n",
                qr.common_hit, remote_src(qr), qr.rdma_read_count);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[observer] exception: %s\n", e.what());
        return 34;
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
    fprintf(stderr, "[parent] %s did not exit normally (status=%d)\n",
            who, status);
    return -2;
}

int run_client_case(const char* self, const std::string& cs,
                    const std::string& node, const std::string& registry,
                    const std::string& cluster_id) {
    pid_t p = spawn({self, "--client", cs, node, registry, cluster_id});
    if (p < 0) return -1;
    return wait_one(p, cs.c_str());
}

} // namespace

int main(int argc, char** argv) {
    // Re-exec'd server: --server <node> <registry> <cluster_id> <num_shards> <min_nodes>
    if (argc >= 7 && std::strcmp(argv[1], "--server") == 0) {
        return run_server_process(argv[2], argv[3], argv[4],
                                  std::atoi(argv[5]), std::atoi(argv[6]));
    }
    // Re-exec'd client: --client <case> <node> <registry> <cluster_id>
    if (argc >= 6 && std::strcmp(argv[1], "--client") == 0) {
        const std::string cs = argv[2];
        if (cs == "writer")
            return run_writer_process(client_cfg(argv[3], argv[4], argv[5]));
        if (cs == "observer")
            return run_observer_process(argv[4], argv[5]);
        fprintf(stderr, "[client] unknown case '%s'\n", cs.c_str());
        return 90;
    }

    radixshmem_test::EtcdHarness etcd;
    if (!etcd.usable()) {
        fprintf(stderr, "[parent] no etcd binary (set RADIXSHMEM_ETCD_BIN or "
                        "put etcd on PATH) — skipping\n");
        return radixshmem_test::kSkipExitCode;
    }
    const std::string registry   = etcd.registry();
    const std::string cluster_id = etcd.cluster_id();

    for (int r = 0; r < kWorldSize; r++)
        ::shm_unlink(shm_name_for(r).c_str());

    std::vector<pid_t> server_pids(kWorldSize);
    for (int r = 0; r < kWorldSize; r++) {
        pid_t p = spawn({argv[0], "--server", node_name_for(r), registry,
                         cluster_id, std::to_string(kWorldSize),
                         std::to_string(kWorldSize)});
        if (p < 0) {
            fprintf(stderr, "[parent] spawn(server %d) failed: %s\n",
                    r, strerror(errno));
            return 30;
        }
        server_pids[r] = p;
    }

    int unhealthy = 0;   // servers whose end-of-run check_health failed
    auto kill_servers = [&]() {
        for (pid_t p : server_pids) if (p > 0) ::kill(p, SIGTERM);
        unhealthy += reap_servers(server_pids);
        for (int r = 0; r < kWorldSize; r++)
            ::shm_unlink(shm_name_for(r).c_str());
    };

    for (int r = 0; r < kWorldSize; r++) {
        if (!wait_shm_ready(shm_name_for(r), kBootstrapTO)) {
            fprintf(stderr, "[parent] server %d never became cluster-ready\n", r);
            kill_servers();
            return 31;
        }
    }

    int failures = 0;
    int rc_w = run_client_case(argv[0], "writer", node_name_for(0),
                               registry, cluster_id);
    if (rc_w != 0) { fprintf(stderr, "FAIL: writer rc=%d\n", rc_w); failures++; }

    int rc_o = run_client_case(argv[0], "observer", kObserverNode,
                               registry, cluster_id);
    if (rc_o != 0) { fprintf(stderr, "FAIL: observer rc=%d\n", rc_o); failures++; }

    kill_servers();
    if (unhealthy) {
        fprintf(stderr, "FAIL: %d server(s) leaked (check_health)\n", unhealthy);
        failures += unhealthy;
    }

    if (failures) {
        fprintf(stderr, "FAIL: test_client_no_server (%d case(s))\n", failures);
        return 40;
    }
    fprintf(stderr, "PASS: test_client_no_server\n");
    return 0;
}
