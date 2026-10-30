// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// test_remote_api — e2e gate for RadixClient rank-first remote ops:
// remote_query, two-phase remote_allocate→remote_insert, remote_recycle, remote_unlock.
//
// Servers are registry-named by node identity (node0/node1 → ranks 0/1 in
// sorted /peers order). Every server AND client case is a re-exec'd image: the
// etcd Go wrapper's runtime doesn't survive a bare fork, and the name-attach
// RadixClient ctor now builds a ClientCluster (etcd client) from the shm header.
// Client cases attach by shm name and read their cluster coordinates from the
// header the server stamped, so they need no etcd argv.

#include "rdma_test_env.h"
#include "etcd_test_harness.hpp"
#include "test_utils.h"
#include "radix_client.h"
#include "radix_server.h"
#include "types.h"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
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

constexpr int    kWorldSize      = 2;
constexpr int    kBlockSize      = 1;
constexpr int    kBootstrapTO    = 30;
constexpr const char* kShmPrefix = "/test_remote_routing";
std::string kRdmaDev = "mlx5_bond_0";   // default; resolved in main() (env override + SKIP)
constexpr int    kGidIdx         = 3;
constexpr int    kTargetRank     = 0;   // B: remote insert + query enabled
constexpr int    kInitiatorRank  = 1;   // A: remote ops off

std::string node_name_for(int rank) { return "node" + std::to_string(rank); }

std::string shm_name_for(int rank) {
    return std::string(kShmPrefix) + "_" + node_name_for(rank);
}

volatile sig_atomic_t g_term = 0;
void on_term(int) { g_term = 1; }

int run_server_process(const std::string& node, const std::string& registry,
                       const std::string& cluster_id) {
    ::signal(SIGTERM, on_term);
    ::signal(SIGINT,  on_term);

    radixshmem::RadixServerConfig cfg;
    cfg.name        = kShmPrefix;
    cfg.shm.hugepage_path = "";
    cfg.shm.max_blocks = 4096;
    cfg.shm.block_size = kBlockSize;
    cfg.expected_min_nodes = kWorldSize;
    cfg.registry    = registry;
    cfg.cluster_id  = cluster_id;
    cfg.node_name   = node;
    cfg.rpc_address = "127.0.0.1";   // single-host peers connect over loopback
    cfg.index_dev    = kRdmaDev;
    cfg.gid_idx     = kGidIdx;
    cfg.bootstrap_timeout_sec = kBootstrapTO;
    if (node == node_name_for(kTargetRank)) {
        cfg.enable_remote_insert = true;
        cfg.enable_remote_query  = true;
    }

    try {
        radixshmem::RadixServer srv(cfg);
        if (!srv.bootstrap()) {
            fprintf(stderr, "[server %s] bootstrap failed\n", node.c_str());
            return 10;
        }
        fprintf(stderr, "[server %s] up rank=%d%s\n", node.c_str(), srv.rank(),
                node == node_name_for(kTargetRank) ? " (B: remote insert+query ON)" : "");
        while (!g_term) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        // Report only: lock_closure/seed(pin) leave pins on B by design.
        (void)server_healthy(srv, node.c_str());
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[server %s] exception: %s\n", node.c_str(), e.what());
        return 11;
    }
}

// Parent-side readiness probe: mmap the owner's shm header and spin on
// cluster_ready_magic. No cgo before forking.
bool wait_shm_ready(const std::string& shm_name, int timeout_sec) {
    auto deadline = std::chrono::steady_clock::now()
                  + std::chrono::seconds(timeout_sec);
    while (std::chrono::steady_clock::now() < deadline) {
        int fd = ::shm_open(shm_name.c_str(), O_RDONLY, 0);
        if (fd >= 0) {
            void* base = ::mmap(nullptr, sizeof(radixshmem::ShmHeader),
                                PROT_READ, MAP_SHARED, fd, 0);
            ::close(fd);
            if (base != MAP_FAILED) {
                auto* hdr = static_cast<radixshmem::ShmHeader*>(base);
                uint64_t v =
                    hdr->cluster_ready_magic.load(std::memory_order_acquire);
                ::munmap(base, sizeof(radixshmem::ShmHeader));
                if (v == radixshmem::CLUSTER_MANIFEST_MAGIC) return true;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return false;
}

std::vector<radixshmem::HashType> make_hashes(int64_t base, int n) {
    std::vector<int64_t> tok(n);
    std::iota(tok.begin(), tok.end(), base);
    return radixshmem::RadixClient::compute_hashes(tok.data(), n, kBlockSize);
}

int wait_one(pid_t pid, const char* who) {
    int status = 0;
    if (::waitpid(pid, &status, 0) != pid) return -1;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    fprintf(stderr, "[parent] %s abnormal (status=%d)\n", who, status);
    return -2;
}

// Re-exec a fresh process image (server or client case). The etcd Go wrapper's
// runtime doesn't survive a bare fork, so nothing that touches cgo may run in a
// bare-forked child — every role is a fresh exec.
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

// Local-tree insert (upstream insert_auto): peek LOCAL matched prefix, allocate
// shortfall, bind slots to hashes[matched:] via start=matched, auto-recycle
// surplus. Returns the number of blocks this call newly landed.
int insert_auto_local(radixshmem::RadixClient& c,
                      const radixshmem::HashType* h, int n) {
    int matched = c.query(h, n, radixshmem::COMPONENT_MASK_FULL,
                          /*local_only=*/true, /*lock=*/false,
                          /*update_meta=*/false).common_hit;
    std::vector<int32_t> slots;
    if (n - matched > 0) c.allocate_slots(n - matched, slots);
    auto ins = c.insert(h, n, slots.data(), (int)slots.size(),
                        /*start=*/matched, /*auto_recycle=*/true);
    return (int)slots.size() - (int)ins.unused_slots.size();
}

// Seed a prefix into B's local tree. pin=false: evictable insert; pin=true: inc_ref path (unevictable).
int seed_prefix_on_B(const std::vector<radixshmem::HashType>& h, bool pin) {
    radixshmem::RadixClient b(shm_name_for(kTargetRank));
    int n = (int)h.size();
    int inserted = insert_auto_local(b, h.data(), n);
    if (pin) (void)pin_path(b, h.data(), n);   // handle dropped: stays pinned
    b.flush();   // drain any pending RHT publish before peers probe
    return inserted;
}

// Eviction pressure on B: allocate whole mempool (evicts unpinned leaves) then recycle (upstream force_evict).
void evict_pressure_on_B() {
    radixshmem::RadixClient b(shm_name_for(kTargetRank));
    std::vector<int32_t> tmp;
    b.allocate_slots((int)b.mempool_total(), tmp);
    if (!tmp.empty()) b.recycle_slots(tmp.data(), (int)tmp.size());
}

int b_local_hit(const std::vector<radixshmem::HashType>& h) {
    radixshmem::RadixClient b(shm_name_for(kTargetRank));
    return b.query(h.data(), (int)h.size(), radixshmem::COMPONENT_MASK_FULL,
                   /*local_only=*/true).common_hit;
}

// ---- #1: rank-first session reuse — repeated ops to one rank stay live ----
int run_session_reuse() {
    try {
        const int n = 8;
        auto h = make_hashes(1000, n);
        if (seed_prefix_on_B(h, /*pin=*/false) != n) return 60;

        radixshmem::RadixClient cli(shm_name_for(kInitiatorRank));
        if (!cli.is_distributed()) return 61;

        // No handle: repeated rank-first queries reuse the client's cached session.
        auto q1 = cli.remote_query(kTargetRank, h.data(), n);
        auto q2 = cli.remote_query(kTargetRank, h.data(), n);
        if (q1.status != radixshmem::RemoteStatus::OK || q1.total_hit_length != n ||
            q2.status != radixshmem::RemoteStatus::OK || q2.total_hit_length != n) {
            fprintf(stderr, "[reuse] q1 status=%d hit=%d / q2 status=%d hit=%d\n",
                    (int)q1.status, q1.total_hit_length,
                    (int)q2.status, q2.total_hit_length);
            return 63;
        }
        fprintf(stderr, "[reuse] OK rank-first session reused across calls\n");
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[reuse] exception: %s\n", e.what());
        return 64;
    }
}

// ---- #4: remote_query(target) ≡ B-local golden, field by field ----
int run_control_vs_golden() {
    try {
        const int n = 16, head_n = 8;
        auto h = make_hashes(2000, n);
        std::vector<radixshmem::HashType> head(h.begin(), h.begin() + head_n);
        if (seed_prefix_on_B(head, /*pin=*/false) != head_n) return 70;
        // Extend with PINNED blocks — query reports pinned/unpinned uniformly.
        if (seed_prefix_on_B(h, /*pin=*/true) != n - head_n) return 71;

        radixshmem::ComponentQueryResult golden;
        {
            radixshmem::RadixClient b(shm_name_for(kTargetRank));
            golden = b.query(h.data(), n, radixshmem::COMPONENT_MASK_FULL,
                             /*local_only=*/true);
        }

        radixshmem::RadixClient cli(shm_name_for(kInitiatorRank));
        auto r = cli.remote_query(kTargetRank, h.data(), n);
        if (r.status != radixshmem::RemoteStatus::OK) {
            fprintf(stderr, "[golden] status=%d\n", (int)r.status); return 72;
        }
        if (r.total_hit_length != golden.common_hit) {
            fprintf(stderr, "[golden] hit=%d/%d\n",
                    r.total_hit_length, golden.common_hit);
            return 73;
        }
        if (r.local_hit_length != 0 || r.remote_node_id != kTargetRank) return 74;
        if (r.remote_hit_length != golden.common_hit) return 75;
        if (r.remote_slots != hit_slots(golden)) {
            fprintf(stderr, "[golden] slots mismatch (%zu vs %zu)\n",
                    r.remote_slots.size(), hit_slots(golden).size());
            return 76;
        }
        // Cross-check: a second rank-first query reuses the session, same answer.
        auto r2 = cli.remote_query(kTargetRank, h.data(), n);
        if (r2.total_hit_length != r.total_hit_length ||
            r2.remote_slots != r.remote_slots) return 77;
        fprintf(stderr, "[golden] OK hit=%d slots match\n",
                r.total_hit_length);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[golden] exception: %s\n", e.what());
        return 78;
    }
}

// ---- #5: BAD_TARGET — illegal ranks report inline, no throw, no wire effects ----
int run_bad_target() {
    try {
        const int n = 4;
        auto h = make_hashes(3000, n);
        std::vector<int32_t> dummy_slots(n, 0);
        radixshmem::RadixClient cli(shm_name_for(kInitiatorRank));

        // Illegal ranks (self / out of range / negative): every op reports
        // BAD_TARGET or REMOTE_FAILED inline — never a throw, nothing on the wire.
        int bad_ranks[] = { kInitiatorRank /*self*/, kWorldSize, 99, -1 };
        for (int t : bad_ranks) {
            auto r = cli.remote_query(t, h.data(), n);
            if (r.status != radixshmem::RemoteStatus::BAD_TARGET ||
                r.rdma_read_count != 0 || r.total_hit_length != 0) {
                fprintf(stderr, "[badtgt] query(rank=%d) status=%d\n",
                        t, (int)r.status);
                return 81;
            }
            if (cli.remote_allocate(t, n).status
                    != radixshmem::RemoteStatus::BAD_TARGET) return 82;
            if (cli.remote_insert(t, h.data(), n, 0, n, dummy_slots.data()).error
                    != radixshmem::InsertError::REMOTE_FAILED) return 83;
            if (cli.remote_unlock(t, h.data(), n, n)
                    != radixshmem::RemoteStatus::BAD_TARGET) return 84;
            if (cli.remote_recycle(t, dummy_slots.data(), n)
                    != radixshmem::RemoteStatus::BAD_TARGET) return 84;
        }
        // Self-guard also holds on a rank that SERVES remote ops.
        {
            radixshmem::RadixClient b(shm_name_for(kTargetRank));
            if (b.remote_query(kTargetRank, h.data(), n).status
                    != radixshmem::RemoteStatus::BAD_TARGET) return 86;
        }
        fprintf(stderr, "[badtgt] OK self/oob/negative → BAD_TARGET inline\n");
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[badtgt] exception: %s\n", e.what());
        return 85;
    }
}

// ---- #6: control-plane lock closure — pin defends, finalize releases ----
int run_lock_closure() {
    try {
        const int n = 16;
        auto h = make_hashes(4000, n);
        if (seed_prefix_on_B(h, /*pin=*/false) != n) return 90;

        radixshmem::RadixClient cli(shm_name_for(kInitiatorRank));
        auto r = cli.remote_query(kTargetRank, h.data(), n, /*lock=*/true);
        if (r.status != radixshmem::RemoteStatus::OK || r.total_hit_length != n)
            return 91;
        if (!r.finalize) return 92;   // lock=true must arm the closure

        evict_pressure_on_B();
        if (b_local_hit(h) != n) {
            fprintf(stderr, "[lock] pinned prefix was evicted\n");
            return 93;
        }
        r.finalize();                 // QUERY_RELEASE via the session
        evict_pressure_on_B();
        if (b_local_hit(h) != 0) {
            fprintf(stderr, "[lock] released prefix survived eviction\n");
            return 94;
        }
        fprintf(stderr, "[lock] OK pin defended, finalize released\n");
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[lock] exception: %s\n", e.what());
        return 95;
    }
}

// ---- #7: remote_allocate → remote_insert two-phase flow (rank-first) ----
int run_insert_finalize() {
    try {
        const int n = 16;
        auto h = make_hashes(5000, n);
        radixshmem::RadixClient cli(shm_name_for(kInitiatorRank));

        auto rep = cli.remote_allocate(kTargetRank, n);
        if (rep.status != radixshmem::RemoteStatus::OK) {
            fprintf(stderr, "[2phase] allocate status=%d\n", (int)rep.status);
            return 100;
        }
        // Pure allocate: exactly n slots, no server-side dedup.
        if ((int)rep.slots.size() != n) return 101;

        // Between phases: nothing published, allocation lives outside tree — B must NOT match.
        {
            radixshmem::RadixClient b(shm_name_for(kTargetRank));
            auto bq = b.query(h.data(), n, radixshmem::COMPONENT_MASK_FULL,
                              /*local_only=*/true);
            if (bq.common_hit != 0) {
                fprintf(stderr, "[2phase] pre-finalize hit=%d (published "
                        "before finalize?)\n", bq.common_hit);
                return 102;
            }
        }
        // (caller's KV transfer happens here) — phase 2. Fresh prefix: start=0, publish all n.
        if (cli.remote_insert(kTargetRank, h.data(), n,
                                0, n, rep.slots.data()).error
                != radixshmem::InsertError::OK)
            return 103;
        {
            radixshmem::RadixClient b(shm_name_for(kTargetRank));
            auto bq = b.query(h.data(), n, radixshmem::COMPONENT_MASK_FULL,
                              /*local_only=*/true);
            if (bq.common_hit != n) return 104;
            if (hit_slots(bq) != rep.slots) {
                fprintf(stderr, "[2phase] slots mismatch after finalize\n");
                return 105;
            }
        }
        fprintf(stderr, "[2phase] OK insert->finalize, slots match golden\n");
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[2phase] exception: %s\n", e.what());
        return 106;
    }
}

// ---- #8: remote_recycle as abort — allocation returned to the pool ----
int run_release_abort() {
    try {
        const int n = 16;
        auto h = make_hashes(6000, n);
        radixshmem::RadixClient cli(shm_name_for(kInitiatorRank));

        radixshmem::RadixClient b(shm_name_for(kTargetRank));
        uint32_t used0 = b.mempool_used();

        auto rep = cli.remote_allocate(kTargetRank, n);
        if (rep.status != radixshmem::RemoteStatus::OK ||
            (int)rep.slots.size() != n) return 110;

        // Allocation outside tree: no match, but slots checked out of B's pool.
        if (b_local_hit(h) != 0 || b.mempool_used() != used0 + (uint32_t)n) {
            fprintf(stderr, "[abort] hit=%d used=%u (expected 0 / %u)\n",
                    b_local_hit(h), b.mempool_used(), used0 + n);
            return 111;
        }
        // Abort: recycle the slots — B returns them to the pool.
        if (cli.remote_recycle(kTargetRank, rep.slots.data(), n)
                != radixshmem::RemoteStatus::OK)
            return 112;
        if (b_local_hit(h) != 0 || b.mempool_used() != used0) {
            fprintf(stderr, "[abort] aborted allocation not recycled "
                    "(hit=%d used=%u)\n", b_local_hit(h), b.mempool_used());
            return 113;
        }
        fprintf(stderr, "[abort] OK allocation reclaimed after recycle\n");
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[abort] exception: %s\n", e.what());
        return 114;
    }
}

// ---- #9: phase-2 dedup — re-insert of a published prefix consumes 0, no leak ----
int run_full_dedup() {
    try {
        const int n = 16;
        auto h = make_hashes(7000, n);
        radixshmem::RadixClient cli(shm_name_for(kInitiatorRank));

        // First publish: allocate n, insert all n.
        auto rep = cli.remote_allocate(kTargetRank, n);
        if (rep.status != radixshmem::RemoteStatus::OK ||
            (int)rep.slots.size() != n) return 120;
        auto first = cli.remote_insert(kTargetRank, h.data(), n, 0, n,
                                       rep.slots.data());
        if (first.error != radixshmem::InsertError::OK) return 121;

        // Second round: allocate again, re-insert the SAME prefix. Dedup is now
        // in phase-2 INSERT — local insert matches the whole prefix and consumes
        // 0 slots; auto_recycle returns the surplus reservation.
        auto rep2 = cli.remote_allocate(kTargetRank, n);
        if (rep2.status != radixshmem::RemoteStatus::OK ||
            (int)rep2.slots.size() != n) return 122;
        auto dup = cli.remote_insert(kTargetRank, h.data(), n, 0, n,
                                     rep2.slots.data(), /*auto_recycle=*/true);
        if (dup.error != radixshmem::InsertError::OK) return 123;
        if ((int)dup.unused_slots.size() != n) {
            fprintf(stderr, "[dedup] re-insert landed %d slots (expected 0)\n",
                    n - (int)dup.unused_slots.size());
            return 124;
        }
        // No leaked pin: dedup'd prefix must stay evictable.
        evict_pressure_on_B();
        if (b_local_hit(h) != 0) {
            fprintf(stderr, "[dedup] prefix survived eviction (leaked ref?)\n");
            return 125;
        }
        fprintf(stderr, "[dedup] OK re-insert consumed 0, surplus recycled\n");
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[dedup] exception: %s\n", e.what());
        return 126;
    }
}

// ---- #2 + #3: target dead → TIMEOUT or UNAVAILABLE; cached session stays safe ----
// The two sync fds reach the re-exec'd child as argv numbers; they survive execv (not CLOEXEC).
int g_to_parent_fd = -1, g_from_parent_fd = -1;
// A dead target can return a deadline expiry or a refused gRPC connection.
bool target_down(radixshmem::RemoteStatus st) {
    return st == radixshmem::RemoteStatus::TIMEOUT ||
           st == radixshmem::RemoteStatus::UNAVAILABLE;
}
int run_timeout_and_rebuild() {
    try {
        radixshmem::RadixClient cli(shm_name_for(kInitiatorRank));
        auto h = make_hashes(50000, 4);
        auto ok = cli.remote_query(kTargetRank, h.data(), 4);   // B still up
        if (ok.status != radixshmem::RemoteStatus::OK) return 180;

        char b = 1;
        if (::write(g_to_parent_fd, &b, 1) != 1) return 181;
        if (::read(g_from_parent_fd, &b, 1) != 1) return 182;   // B is dead now

        // #2: the failure is reported in status, no hang, no throw.
        auto r = cli.remote_query(kTargetRank, h.data(), 4);
        if (!target_down(r.status)) {
            fprintf(stderr, "[timeout] status=%d expected TIMEOUT/UNAVAILABLE\n",
                    (int)r.status);
            return 183;
        }
        // #3: the cached session survives; repeated ops keep reporting the failure, never crash (no per-call handle to invalidate).
        auto q = cli.remote_query(kTargetRank, h.data(), 4);
        if (!target_down(q.status)) return 184;
        auto rep = cli.remote_allocate(kTargetRank, 4);
        if (!target_down(rep.status)) return 185;
        fprintf(stderr, "[timeout] OK status=%d + cached session safe\n",
                (int)r.status);

        // #10: B restarts on a new endpoint; the same client reopens its session from the new /peers entry.
        if (::write(g_to_parent_fd, &b, 1) != 1) return 187;
        if (::read(g_from_parent_fd, &b, 1) != 1 || b != 1) return 188;   // B is back
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
        radixshmem::RemoteStatus st;
        do {
            st = cli.remote_query(kTargetRank, h.data(), 4).status;
            if (st == radixshmem::RemoteStatus::OK) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } while (std::chrono::steady_clock::now() < deadline);
        if (st != radixshmem::RemoteStatus::OK) {
            fprintf(stderr, "[rebuild] restarted B unreachable, status=%d\n", (int)st);
            return 189;
        }
        fprintf(stderr, "[rebuild] OK old client reached restarted B\n");
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[timeout] exception: %s\n", e.what());
        return 186;
    }
}

// Client-case dispatch for the re-exec'd child (no etcd argv — the name-attach
// ctor reads cluster coordinates from the shm header the server stamped).
int client_main(int argc, char** argv) {
    const std::string cs = argv[2];
    if (cs == "session_reuse")     return run_session_reuse();
    if (cs == "control_golden")    return run_control_vs_golden();
    if (cs == "bad_target")        return run_bad_target();
    if (cs == "lock_closure")      return run_lock_closure();
    if (cs == "insert_finalize")   return run_insert_finalize();
    if (cs == "release_abort")     return run_release_abort();
    if (cs == "full_dedup")        return run_full_dedup();
    if (cs == "timeout_rebuild") {
        // argv[3]/argv[4]: fd numbers for parent sync (survive execv).
        g_to_parent_fd   = std::atoi(argv[3]);
        g_from_parent_fd = std::atoi(argv[4]);
        return run_timeout_and_rebuild();
    }
    fprintf(stderr, "[client] unknown case '%s'\n", cs.c_str());
    return 90;
}

}  // namespace

int main(int argc, char** argv) {
    // Re-exec'd server: --server <node> <registry> <cluster_id>.
    if (argc >= 5 && std::strcmp(argv[1], "--server") == 0) {
        kRdmaDev = resolve_index_dev_or_skip(kRdmaDev.c_str());
        return run_server_process(argv[2], argv[3], argv[4]);
    }
    // Re-exec'd client: --client <case> [<fd_to_parent> <fd_from_parent>].
    if (argc >= 3 && std::strcmp(argv[1], "--client") == 0) {
        kRdmaDev = resolve_index_dev_or_skip(kRdmaDev.c_str());
        return client_main(argc, argv);
    }

    kRdmaDev = resolve_index_dev_or_skip(kRdmaDev.c_str());

    // Private etcd — sole cluster bootstrap backend; no etcd binary → SKIP.
    radixshmem_test::EtcdHarness etcd;
    if (!etcd.usable()) {
        fprintf(stderr, "[parent] no etcd binary (set RADIXSHMEM_ETCD_BIN or "
                        "put etcd on PATH) — skipping\n");
        return radixshmem_test::kSkipExitCode;
    }
    const std::string registry   = etcd.registry();
    const std::string cluster_id = etcd.cluster_id();

    for (int r = 0; r < kWorldSize; r++) ::shm_unlink(shm_name_for(r).c_str());

    std::vector<pid_t> server_pids(kWorldSize);
    for (int r = 0; r < kWorldSize; r++) {
        pid_t p = spawn({argv[0], "--server", node_name_for(r), registry, cluster_id});
        if (p < 0) { fprintf(stderr, "spawn(server %d) failed\n", r); return 30; }
        server_pids[r] = p;
    }

    auto kill_servers = [&]() {
        for (pid_t p : server_pids) if (p > 0) ::kill(p, SIGTERM);
        for (pid_t p : server_pids) if (p > 0) ::waitpid(p, nullptr, 0);
        for (int r = 0; r < kWorldSize; r++) ::shm_unlink(shm_name_for(r).c_str());
    };

    for (int r = 0; r < kWorldSize; r++) {
        if (!wait_shm_ready(shm_name_for(r), kBootstrapTO)) {
            fprintf(stderr, "FAIL: rank %d never reached cluster-ready\n", r);
            kill_servers();
            return 31;
        }
    }

    int final_rc = 0;
    auto run_case = [&](const char* name) {
        pid_t p = spawn({argv[0], "--client", name});
        if (p < 0) { final_rc = final_rc ? final_rc : 32; return; }
        int rc = wait_one(p, name);
        fprintf(stderr, "[case] %-18s rc=%d\n", name, rc);
        if (rc != 0 && final_rc == 0) final_rc = rc ? rc : 40;
    };

    run_case("session_reuse");     // #1
    run_case("control_golden");    // #4
    run_case("bad_target");        // #5
    run_case("lock_closure");      // #6
    run_case("insert_finalize");   // #7
    run_case("release_abort");     // #8
    run_case("full_dedup");        // #9

    // #2/#3/#10 last: pipe-synchronized kill and restart of B (nothing runs after it). The child
    // is re-exec'd; the two sync-pipe fd numbers ride on its argv (not CLOEXEC).
    {
        int to_parent[2], from_parent[2];
        if (::pipe(to_parent) != 0 || ::pipe(from_parent) != 0) {
            final_rc = final_rc ? final_rc : 33;
        } else {
            pid_t p = spawn({argv[0], "--client", "timeout_rebuild",
                             std::to_string(to_parent[1]),
                             std::to_string(from_parent[0])});
            if (p < 0) {
                final_rc = final_rc ? final_rc : 33;
            } else {
                ::close(to_parent[1]); ::close(from_parent[0]);
                char b = 0;
                if (::read(to_parent[0], &b, 1) == 1) {
                    ::kill(server_pids[kTargetRank], SIGTERM);
                    ::waitpid(server_pids[kTargetRank], nullptr, 0);
                    server_pids[kTargetRank] = -1;
                    char ack = 1;
                    if (::write(from_parent[1], &ack, 1) != 1) { /* child sees EOF */ }
                }
                // #10: restart B (port 0 and a fresh DCT give it a new endpoint), then release the child.
                if (::read(to_parent[0], &b, 1) == 1) {
                    ::shm_unlink(shm_name_for(kTargetRank).c_str());
                    server_pids[kTargetRank] = spawn({argv[0], "--server",
                        node_name_for(kTargetRank), registry, cluster_id});
                    char up = server_pids[kTargetRank] > 0 &&
                              wait_shm_ready(shm_name_for(kTargetRank), kBootstrapTO);
                    if (::write(from_parent[1], &up, 1) != 1) { /* child sees EOF */ }
                }
                int rc = wait_one(p, "timeout_rebuild");
                fprintf(stderr, "[case] %-18s rc=%d\n", "timeout_rebuild", rc);
                if (rc != 0 && final_rc == 0) final_rc = rc ? rc : 40;
                ::close(to_parent[0]); ::close(from_parent[1]);
            }
        }
    }

    kill_servers();

    if (final_rc != 0) {
        fprintf(stderr, "FAIL: test_remote_api rc=%d\n", final_rc);
        return final_rc;
    }
    fprintf(stderr, "PASS: test_remote_api all cases\n");
    return 0;
}
