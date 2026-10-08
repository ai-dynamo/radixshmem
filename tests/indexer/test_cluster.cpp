// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Cross-process distributed smoke test over etcd dynamic membership.
//
// Uses the public surface — RadixServer (identity-named shm, registry-assigned
// rank) + RadixClient (etcd-config ctor, discovers its shm via /shm/<node>) —
// to verify:
//
//   1. Two RadixServer processes bootstrap through a private etcd.
//   2. A writer client (node0) inserts hashes and updates the distributed RHT
//      via one-sided RDMA writes to the rank-1 shard.
//   3. A reader client (node1) sees the prefix on RDMA remote-walk:
//      common_hit covers everything inserted, remote_node_id == 0,
//      rdma_read_count > 0.
//
// Every server and client runs as a RE-EXEC'd process image: the etcd Go-client
// wrapper starts its runtime at library-load, so a bare fork()'d child deadlocks
// on the first cgo call. main() dispatches on argv:
//   --server <node> <registry> <cluster_id> <num_shards> <min_nodes>
//   --client <case> <node> <registry> <cluster_id> <min_nodes>
//
// Loopback-only — all processes run on the host that owns the RDMA NIC.

#include "shmradix/radix_client.h"
#include "shmradix/radix_server.h"
#include "shmradix/types.h"

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
#include <stdexcept>
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
constexpr int    kPrefixLen     = 16;          // hashes inserted by writer
constexpr int    kBootstrapTO   = 30;
constexpr const char* kShmPrefix = "/test_dc";
constexpr const char* kRdmaDev   = "mlx5_bond_0";
constexpr int    kGidIdx        = 3;

std::string node_name_for(int rank) { return "node" + std::to_string(rank); }

// Server-side shm name (identity-based, matching RadixServer::bootstrap).
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

// Build the etcd-config client for `node`; discovers its shm via /shm/<node>.
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

// Parent-side readiness probe: mmap a server's shm header (no cgo — the parent
// must not touch the Go etcd client before forking, or forked children inherit
// a broken runtime) and spin on cluster_ready_magic, which the owner stamps
// together with is_distributed once bootstrap completes.
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

// Long-lived server process: hold the RadixServer until SIGTERM. Rank is assigned
// by the registry; node_name is this process's stable identity. min_nodes gates
// bootstrap on the full expected membership so the run terminates deterministically.
int run_server_process(const std::string& node, const std::string& registry,
                       const std::string& cluster_id, int num_shards,
                       int min_nodes) {
    ::signal(SIGTERM, on_term);
    ::signal(SIGINT,  on_term);

    shmradix::RadixServerConfig cfg;
    cfg.name        = kShmPrefix;
    cfg.shm.hugepage_path = "";
    cfg.shm.max_blocks = 4096;
    cfg.shm.block_size = kBlockSize;
    cfg.registry    = registry;
    cfg.cluster_id  = cluster_id;
    cfg.node_name   = node;
    cfg.rpc_address = "127.0.0.1";   // single-host peers connect over loopback
    cfg.num_rht_shards  = num_shards;
    cfg.expected_min_nodes = min_nodes;
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
    fprintf(stderr, "[server %s] shutting down\n", node.c_str());
    return server_healthy(srv, node.c_str()) ? 0 : kHealthExitCode;
}

// Writer client (node0): inserts kPrefixLen hashes, flushes RHT writes, exits.
int run_writer_process(const shmradix::RadixClientConfig& cc) {
    try {
        shmradix::RadixClient cli(cc);
        if (!cli.is_distributed()) {
            fprintf(stderr, "[writer] expected distributed mode\n");
            return 11;
        }
        std::vector<int64_t> tokens(kPrefixLen);
        std::iota(tokens.begin(), tokens.end(), int64_t{1000});
        auto hashes = shmradix::RadixClient::compute_hashes(
            tokens.data(), (int)tokens.size(), kBlockSize);

        auto ins = insert_auto(cli, hashes.data(), (int)hashes.size());
        if (ins.error != shmradix::InsertError::OK ||
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

// Reader client (node1): queries the same prefix; result comes back via RDMA
// remote-walk against rank 0.
int run_reader_process(const shmradix::RadixClientConfig& cc) {
    try {
        shmradix::RadixClient cli(cc);
        if (!cli.is_distributed()) {
            fprintf(stderr, "[reader] expected distributed mode\n");
            return 21;
        }
        std::vector<int64_t> tokens(kPrefixLen);
        std::iota(tokens.begin(), tokens.end(), int64_t{1000});
        auto hashes = shmradix::RadixClient::compute_hashes(
            tokens.data(), (int)tokens.size(), kBlockSize);

        auto qr = cli.query(hashes.data(), (int)hashes.size(), shmradix::COMPONENT_MASK_FULL);

        bool ok = true;
        if (remote_src(qr) != 0) {
            fprintf(stderr, "[reader] expected remote fragment from rank 0, got %d\n",
                    remote_src(qr));
            ok = false;
        }
        if (qr.common_hit != kPrefixLen) {
            fprintf(stderr, "[reader] expected total_hit=%d, got %d\n",
                    kPrefixLen, qr.common_hit);
            ok = false;
        }
        if (qr.rdma_read_count <= 0) {
            fprintf(stderr, "[reader] expected rdma_read_count > 0, got %d\n",
                    qr.rdma_read_count);
            ok = false;
        }
        if (!ok) return 22;

        fprintf(stderr,
                "[reader] OK total=%d remote_src=%d rdma_reads=%d\n",
                qr.common_hit, remote_src(qr), qr.rdma_read_count);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[reader] exception: %s\n", e.what());
        return 23;
    }
}

// Lock reader (node1): full query(lock=true) → dec_ref_remote cycle.
int run_lock_dec_reader_process(const shmradix::RadixClientConfig& cc) {
    try {
        shmradix::RadixClient cli(cc);
        if (!cli.is_distributed()) {
            fprintf(stderr, "[lockdec] expected distributed mode\n");
            return 51;
        }
        std::vector<int64_t> tokens(kPrefixLen);
        std::iota(tokens.begin(), tokens.end(), int64_t{1000});
        auto hashes = shmradix::RadixClient::compute_hashes(
            tokens.data(), (int)tokens.size(), kBlockSize);

        auto qr = cli.query(hashes.data(), (int)hashes.size(), shmradix::COMPONENT_MASK_FULL,
                              /*local_only=*/false, /*lock=*/true);
        if (qr.common_hit != kPrefixLen) {
            fprintf(stderr, "[lockdec] expected total_hit=%d got %d\n",
                    kPrefixLen, qr.common_hit);
            return 52;
        }
        if (remote_src(qr) != 0) {
            fprintf(stderr, "[lockdec] expected remote fragment from rank 0 got %d\n",
                    remote_src(qr));
            return 53;
        }
        qr.finalize();
        fprintf(stderr, "[lockdec] OK total_hit=%d atomics(query)=%d\n",
                qr.common_hit, qr.rdma_atomic_count);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[lockdec] exception: %s\n", e.what());
        return 55;
    }
}

// Split-remote: writer inserts prefix A on rank 0, then B diverging inside A
// (forces a remote split). Reader on rank 1 query(lock=true) over A must
// inc_ref both fragments; dec_ref_remote must reach both. Verified by
// force-evicting on rank 0 afterward — the prefix must be fully gone.
int run_split_remote_process(const shmradix::RadixClientConfig& cc0,
                             const shmradix::RadixClientConfig& cc1) {
    try {
        {
            shmradix::RadixClient w(cc0);
            if (!w.is_distributed()) return 71;
            std::vector<int64_t> ta(16);
            std::iota(ta.begin(), ta.end(), int64_t{7000});
            auto ha = shmradix::RadixClient::compute_hashes(
                ta.data(), 16, kBlockSize);
            insert_auto(w, ha.data(), 16);
            std::vector<int64_t> tb(16);
            for (int i = 0; i < 8; i++) tb[i] = 7000 + i;
            for (int i = 8; i < 16; i++) tb[i] = 7100 + (i - 8);
            auto hb = shmradix::RadixClient::compute_hashes(
                tb.data(), 16, kBlockSize);
            insert_auto(w, hb.data(), 16);   // triggers split on rank 0
            w.flush();
        }

        shmradix::RadixClient r(cc1);
        if (!r.is_distributed()) return 72;
        std::vector<int64_t> ta(16);
        std::iota(ta.begin(), ta.end(), int64_t{7000});
        auto ha = shmradix::RadixClient::compute_hashes(
            ta.data(), 16, kBlockSize);

        auto qr = r.query(ha.data(), 16, shmradix::COMPONENT_MASK_FULL, /*local_only=*/false, /*lock=*/true);
        if (qr.common_hit != 16) {
            fprintf(stderr, "[split_remote] hit=%d expected 16\n",
                    qr.common_hit);
            return 73;
        }
        if (remote_src(qr) != 0) return 74;
        qr.finalize();

        {
            shmradix::RadixClient w(cc0);
            std::vector<int32_t> tmp;
            w.allocate_slots((int)w.mempool_total(), tmp);
            if (!tmp.empty()) w.recycle_slots(tmp.data(), (int)tmp.size());
            auto q = w.query(ha.data(), 16, shmradix::COMPONENT_MASK_FULL, /*local_only=*/true,
                              /*lock=*/false, /*update_meta=*/false);
            if (q.common_hit != 0) {
                fprintf(stderr,
                        "[split_remote] residual ref on rank 0: hit=%d\n",
                        q.common_hit);
                return 75;
            }
        }
        fprintf(stderr, "[split_remote] OK split-invariant dec_ref_remote\n");
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[split_remote] exception: %s\n", e.what());
        return 76;
    }
}

// RHT-aligned probe: writer inserts a long cs-aligned prefix on rank 0; reader
// on rank 1 queries it back in full via RDMA remote walk.
int run_rht_aligned_probe_process(const shmradix::RadixClientConfig& cc0,
                                  const shmradix::RadixClientConfig& cc1) {
    try {
        constexpr int kLen = 256;  // 2 * cs (128 default)
        std::vector<int64_t> tok(kLen);
        std::iota(tok.begin(), tok.end(), int64_t{80000});
        auto hashes = shmradix::RadixClient::compute_hashes(
            tok.data(), kLen, kBlockSize);

        {
            shmradix::RadixClient w(cc0);
            if (!w.is_distributed()) return 81;
            auto ins = insert_auto(w, hashes.data(), kLen);
            if (ins.error != shmradix::InsertError::OK ||
                !ins.unused_slots.empty()) {
                fprintf(stderr, "[rht_probe] writer insert short (unused=%zu)\n",
                        ins.unused_slots.size());
                return 82;
            }
            w.flush();
        }

        shmradix::RadixClient r(cc1);
        if (!r.is_distributed()) return 83;
        auto qr = r.query(hashes.data(), kLen, shmradix::COMPONENT_MASK_FULL);
        if (qr.common_hit != kLen) {
            fprintf(stderr, "[rht_probe] expected hit=%d got %d\n",
                    kLen, qr.common_hit);
            return 84;
        }
        if (remote_src(qr) != 0) return 85;
        fprintf(stderr,
                "[rht_probe] OK %d-block cs-aligned prefix found via RHT\n",
                kLen);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[rht_probe] exception: %s\n", e.what());
        return 86;
    }
}

// Multi-chunk query-lock reader (node0): insert a multi-node prefix, pin via
// query(lock=true), finalize, and verify every fragment released (no ref leak
// blocking eviction).
int run_multichunk_finalize_process(const shmradix::RadixClientConfig& cc0) {
    try {
        shmradix::RadixClient cli(cc0);
        if (!cli.is_distributed()) {
            fprintf(stderr, "[mchunk] expected distributed mode\n");
            return 61;
        }
        constexpr int kLen = 320;
        std::vector<int64_t> tokens(kLen);
        std::iota(tokens.begin(), tokens.end(), int64_t{50000});
        auto hashes = shmradix::RadixClient::compute_hashes(
            tokens.data(), (int)tokens.size(), kBlockSize);

        std::vector<int32_t> slots;
        cli.allocate_slots(kLen, slots);
        auto ins = cli.insert(hashes.data(), (int)hashes.size(),
                              slots.data(), (int)slots.size());
        if (ins.error != shmradix::InsertError::OK) {
            fprintf(stderr, "[mchunk] insert error=%u\n", (unsigned)ins.error);
            return 62;
        }
        if (!ins.unused_slots.empty()) {
            fprintf(stderr, "[mchunk] expected full land, unused=%zu\n",
                    ins.unused_slots.size());
            return 63;
        }
        auto qlock = cli.query(hashes.data(), (int)hashes.size(), shmradix::COMPONENT_MASK_FULL,
                               /*local_only=*/true, /*lock=*/true);
        if (!(bool)qlock.finalize) {
            fprintf(stderr, "[mchunk] finalize unexpectedly empty\n");
            return 64;
        }
        qlock.finalize();
        auto qr = cli.query(hashes.data(), (int)hashes.size(), shmradix::COMPONENT_MASK_FULL,
                              /*local_only=*/true);
        if (qr.common_hit != kLen) {
            fprintf(stderr,
                    "[mchunk] expected total_hit=%d after finalize got %d\n",
                    kLen, qr.common_hit);
            return 65;
        }
        std::vector<int32_t> tmp;
        cli.allocate_slots((int)cli.mempool_total(), tmp);
        if (!tmp.empty()) cli.recycle_slots(tmp.data(), (int)tmp.size());
        auto qr2 = cli.query(hashes.data(), (int)hashes.size(), shmradix::COMPONENT_MASK_FULL,
                               /*local_only=*/true);
        if (qr2.common_hit != 0) {
            fprintf(stderr,
                    "[mchunk] expected fully evicted after finalize, "
                    "still hit %d (ref leak across chunks)\n",
                    qr2.common_hit);
            return 66;
        }
        fprintf(stderr, "[mchunk] OK %d blocks across %d chunks released\n",
                kLen, (kLen + 127) / 128);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[mchunk] exception: %s\n", e.what());
        return 67;
    }
}

// local_read_only client (node0): attaches the tree without the cluster plane
// (no etcd, no RDMA), sees the writer's prefix and the node's rank, and
// refuses every mutation.
int run_local_read_only_process() {
    try {
        shmradix::RadixClient ro(shm_name_for(0), /*local_read_only=*/true);
        if (ro.is_distributed() || ro.rank() != 0 || ro.world_size() != 1) {
            fprintf(stderr, "[local_read_only]expected a standalone rank-0 view, got "
                            "dist=%d rank=%d world=%d\n",
                    (int)ro.is_distributed(), ro.rank(), ro.world_size());
            return 111;
        }
        std::vector<int64_t> tokens(kPrefixLen);
        std::iota(tokens.begin(), tokens.end(), int64_t{1000});
        auto hashes = shmradix::RadixClient::compute_hashes(
            tokens.data(), (int)tokens.size(), kBlockSize);
        auto qr = ro.query(hashes.data(), (int)hashes.size(), shmradix::COMPONENT_MASK_FULL);
        if (qr.common_hit != kPrefixLen || qr.rdma_read_count != 0) {
            fprintf(stderr, "[local_read_only]expected local hit %d with no RDMA, got %d "
                            "(%d reads)\n",
                    kPrefixLen, qr.common_hit, qr.rdma_read_count);
            return 112;
        }
        int refused = 0;
        std::vector<int32_t> slots;
        int32_t one = 0;
        try { ro.allocate_slots(1, slots); } catch (const std::runtime_error&) { refused++; }
        try { ro.insert(hashes.data(), 1, &one, 1); } catch (const std::runtime_error&) { refused++; }
        try { ro.reset(); } catch (const std::runtime_error&) { refused++; }
        if (refused != 3) {
            fprintf(stderr, "[local_read_only]expected 3 refused mutations, got %d\n", refused);
            return 113;
        }
        fprintf(stderr, "[local_read_only]OK hit=%d rank=%d, mutations refused\n",
                qr.common_hit, ro.rank());
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[local_read_only]exception: %s\n", e.what());
        return 114;
    }
}

// Re-exec a fresh process image for one role. The etcd Go-client wrapper starts
// its runtime at library-load, so a bare fork()'d child deadlocks on the first
// cgo call; every server AND client must be a fresh exec.
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

// Spawn a client `case` for `node` and wait for it. Returns its exit code.
int run_client_case(const char* self, const std::string& cs, int node_rank,
                    const std::string& registry, const std::string& cluster_id) {
    pid_t p = spawn({self, "--client", cs, node_name_for(node_rank),
                     registry, cluster_id});
    if (p < 0) return -1;
    int status = 0;
    if (::waitpid(p, &status, 0) != p) return -1;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    fprintf(stderr, "[parent] client %s did not exit normally (status=%d)\n",
            cs.c_str(), status);
    return -2;
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

// Client-mode dispatch: build the etcd-config client(s) for the case and run it.
int client_main(const std::string& cs, const std::string& node,
                const std::string& registry, const std::string& cluster_id) {
    auto cc0 = client_cfg(node_name_for(0), registry, cluster_id);
    auto cc1 = client_cfg(node_name_for(1), registry, cluster_id);
    auto cc  = client_cfg(node, registry, cluster_id);

    if (cs == "writer")       return run_writer_process(cc);
    if (cs == "local_read_only") return run_local_read_only_process();
    if (cs == "reader")       return run_reader_process(cc);
    if (cs == "lockdec")      return run_lock_dec_reader_process(cc);
    if (cs == "mchunk")       return run_multichunk_finalize_process(cc0);
    if (cs == "rht_probe")    return run_rht_aligned_probe_process(cc0, cc1);
    if (cs == "split_remote") return run_split_remote_process(cc0, cc1);
    fprintf(stderr, "[client] unknown case '%s'\n", cs.c_str());
    return 90;
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
        return client_main(argv[2], argv[3], argv[4], argv[5]);
    }

    // Private etcd for this run — the sole cluster bootstrap backend.
    shmradix_test::EtcdHarness etcd;
    if (!etcd.usable()) {
        fprintf(stderr, "[parent] no etcd binary (set SHMRADIX_ETCD_BIN or "
                        "put etcd on PATH) — skipping\n");
        return shmradix_test::kSkipExitCode;
    }
    const std::string registry   = etcd.registry();
    const std::string cluster_id = etcd.cluster_id();

    for (int r = 0; r < kWorldSize; r++)
        ::shm_unlink(shm_name_for(r).c_str());

    // ---- spawn servers (re-exec) — min_nodes gates on full membership ----
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

    // Gate on full cluster readiness before any client attaches — the owner
    // stamps is_distributed only after bootstrap settles, so an early attach
    // would see a standalone header and skip the RDMA plane.
    for (int r = 0; r < kWorldSize; r++) {
        if (!wait_shm_ready(shm_name_for(r), kBootstrapTO)) {
            fprintf(stderr, "[parent] server %d never became cluster-ready\n", r);
            kill_servers();
            return 31;
        }
    }

    // ---- run client cases (each a fresh exec; see spawn) ----
    struct Case { const char* name; int node; };
    const Case cases[] = {
        {"writer",          0},
        {"local_read_only", 0},
        {"reader",          1},
        {"lockdec",         1},
        {"mchunk",          0},
        {"rht_probe",       0},
        {"split_remote",    0},
    };

    int failures = 0;
    for (const auto& c : cases) {
        int rc = run_client_case(argv[0], c.name, c.node, registry, cluster_id);
        if (rc != 0) {
            fprintf(stderr, "FAIL: %s rc=%d\n", c.name, rc);
            failures++;
        }
    }

    kill_servers();
    if (unhealthy) {
        fprintf(stderr, "FAIL: %d server(s) leaked (check_health)\n", unhealthy);
        failures += unhealthy;
    }

    if (failures) {
        fprintf(stderr, "FAIL: test_cluster (%d case(s))\n", failures);
        return 40;
    }
    fprintf(stderr, "PASS: test_cluster all scenarios\n");
    return 0;
}
