// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radix_client.h"
#include "radix_server.h"
#include "cluster_protocol.h"
#include "types.h"

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

#ifdef RADIXSHMEM_NO_RDMA
int main() {
    fprintf(stderr, "remote_op_smoke: built without RDMA (no remote-op) — skipped\n");
    return 0;
}
#else

using namespace radixshmem;

namespace {

constexpr const char* kShmPrefix = "/ro_smoke";
constexpr int kWorldSize = 2;
constexpr int kPlen      = 32;

volatile sig_atomic_t g_term = 0;
void on_term(int) { g_term = 1; }

std::string shm_name_for(int rank) {
    return std::string(kShmPrefix) + "_" + std::to_string(rank);
}

std::vector<HashType> make_hashes(int64_t base, int n) {
    std::vector<int64_t> tok(n);
    std::iota(tok.begin(), tok.end(), base);
    return RadixClient::compute_hashes(tok.data(), n, /*block_size=*/1);
}

bool wait_for_cluster_ready(const std::string& shm_name, int timeout_sec) {
    auto deadline = std::chrono::steady_clock::now()
                  + std::chrono::seconds(timeout_sec);
    int fd = -1;
    while (true) {
        fd = ::shm_open(shm_name.c_str(), O_RDONLY, 0);
        if (fd >= 0) break;
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    void* base = ::mmap(nullptr, sizeof(ShmHeader), PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (base == MAP_FAILED) return false;
    auto* hdr = static_cast<ShmHeader*>(base);
    bool ok = false;
    while (!ok) {
        ok = hdr->cluster_ready_magic.load(std::memory_order_acquire)
             == CLUSTER_MANIFEST_MAGIC;
        if (!ok && std::chrono::steady_clock::now() > deadline) break;
        if (!ok) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    ::munmap(base, sizeof(ShmHeader));
    return ok;
}

int run_server(int rank, int master_port) {
    ::signal(SIGTERM, on_term);
    RadixServerConfig cfg;
    cfg.name        = kShmPrefix;
    cfg.shm.max_blocks = 100'000;
    cfg.shm.block_size = 1;
    cfg.rank        = rank;
    cfg.world_size  = kWorldSize;
    cfg.registry    = "etcd://127.0.0.1:2379";
    cfg.cluster_id  = "remote_op_smoke";
    cfg.node_name   = "node" + std::to_string(rank);
    (void)master_port;
    cfg.bootstrap_timeout_sec = 30;
    try {
        RadixServer srv(cfg);
        if (!srv.bootstrap()) return 10;
        while (!g_term) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[smoke server %d] %s\n", rank, e.what());
        return 11;
    }
}

#define CHECK(cond, what)                                                  \
    do {                                                                   \
        if (!(cond)) { fprintf(stderr, "  FAIL: %s\n", what); return 1; }  \
        printf("  ok: %s\n", what);                                        \
    } while (0)

// The actual checks, from a client attached to rank 1 targeting rank 0 (rank-first).
int run_checks(int64_t seed) {
    RadixClient cli(shm_name_for(1));
    if (!cli.is_distributed()) { fprintf(stderr, "  FAIL: not distributed\n"); return 1; }
    const int T = 0;   // target rank

    auto h = make_hashes(seed, kPlen);

    // allocate slots + insert (publish). Fresh prefix: start=0, publish all.
    auto rep = cli.remote_allocate(T, kPlen);
    CHECK(rep.status == RemoteStatus::OK && (int)rep.slots.size() == kPlen,
          "allocate: OK, kPlen slots reserved");
    CHECK(cli.remote_insert(T, h.data(), kPlen, 0, kPlen,
                            rep.slots.data()).unused_slots.empty(),
          "insert: publish all kPlen slots");

    // duplicate insert → phase-2 dedup: local insert matches whole prefix,
    // consumes 0, surplus reservation auto-recycled.
    auto rep2 = cli.remote_allocate(T, kPlen);
    CHECK(rep2.status == RemoteStatus::OK && (int)rep2.slots.size() == kPlen,
          "re-insert: allocate second batch");
    CHECK((int)cli.remote_insert(T, h.data(), kPlen, 0, kPlen,
                            rep2.slots.data(), /*auto_recycle=*/true).unused_slots.size() == kPlen,
          "re-insert: dedup (consumed 0, surplus recycled)");

    // plain query sees the full published prefix
    auto q = cli.remote_query(T, h.data(), kPlen, /*lock=*/false);
    CHECK(q.status == RemoteStatus::OK && q.total_hit_length == kPlen &&
          (int)q.remote_slots.size() == q.total_hit_length,
          "query(plain): full prefix hit, slots == hit");

    // query(lock) + unlock round-trip
    q = cli.remote_query(T, h.data(), kPlen, /*lock=*/true);
    CHECK(q.status == RemoteStatus::OK && q.total_hit_length == kPlen,
          "query(lock): full prefix pinned");
    CHECK(q.finalize, "query(lock): finalize armed");
    q.finalize();

    // unknown prefix → OK with no hit
    auto miss = make_hashes(seed + 1'000'000, kPlen);
    q = cli.remote_query(T, miss.data(), kPlen, /*lock=*/false);
    CHECK(q.status == RemoteStatus::OK && q.total_hit_length == 0,
          "query(miss): OK, hit=0");

    // Abort: allocate then recycle without publishing — slots returned to pool.
    auto h2 = make_hashes(seed + 2'000'000, kPlen);
    (void)h2;
    auto r1 = cli.remote_allocate(T, kPlen);
    CHECK(r1.status == RemoteStatus::OK && (int)r1.slots.size() == kPlen,
          "abort: allocate a batch");
    CHECK(cli.remote_recycle(T, r1.slots.data(), (int)r1.slots.size())
              == RemoteStatus::OK,
          "abort: recycle unpublished slots");

    return 0;
}

int run_grpc() {
    printf("=== remote_op_smoke: transport=grpc ===\n");
    fflush(stdout);   // forked children inherit the stdio buffer — drain it
    int master_port = 27950 + (int)(::getpid() % 40);
    for (int r = 0; r < kWorldSize; r++) ::shm_unlink(shm_name_for(r).c_str());

    pid_t pids[kWorldSize];
    for (int r = 0; r < kWorldSize; r++) {
        pid_t p = ::fork();
        if (p < 0) { perror("fork"); return 2; }
        if (p == 0) _exit(run_server(r, master_port));
        pids[r] = p;
    }
    auto teardown = [&]() {
        for (pid_t p : pids) ::kill(p, SIGTERM);
        for (pid_t p : pids) ::waitpid(p, nullptr, 0);
        for (int r = 0; r < kWorldSize; r++) ::shm_unlink(shm_name_for(r).c_str());
    };

    int rc = 0;
    for (int r = 0; r < kWorldSize && rc == 0; r++)
        if (!wait_for_cluster_ready(shm_name_for(r), 30)) {
            fprintf(stderr, "  FAIL: rank %d never cluster-ready\n", r);
            rc = 3;
        }

    if (rc == 0) {
        // Checks run in a child: a crash there must not skip teardown.
        pid_t c = ::fork();
        if (c < 0) { perror("fork"); rc = 2; }
        else if (c == 0) {
            int crc = run_checks(7'000'000);
            fflush(stdout); fflush(stderr);   // _exit skips stdio flush
            _exit(crc);
        }
        else {
            int st = 0;
            ::waitpid(c, &st, 0);
            rc = WIFEXITED(st) ? WEXITSTATUS(st) : 4;
        }
    }
    teardown();
    printf("%s: transport=grpc\n\n", rc == 0 ? "PASS" : "FAIL");
    fflush(stdout);
    return rc;
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "grpc")) continue;
        fprintf(stderr, "usage: remote_op_smoke [grpc]   (an RDMA RHT's device: "
                        "RADIXSHMEM_INDEX_DEV / RADIXSHMEM_INDEX_GID_IDX)\n");
        return 2;
    }
    return run_grpc();
}

#endif  // RADIXSHMEM_NO_RDMA
