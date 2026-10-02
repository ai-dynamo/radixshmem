// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// test_client_concurrency — two orthogonal concerns for multi-client shm:
//   (1) sharing correctness: clients attaching the same shm (in-process and
//       across a fork) see each other's writes with identical slot values;
//   (2) SeqLock concurrency: multi-reader threads observe consistent snapshots
//       while a single writer hammers inserts/evictions.
// Merged from the former test_client_share + test_concurrent.

#include "test_utils.h"
#include "shmradix/radix_client.h"

#include <atomic>
#include <cassert>
#include <cstdio>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace shmradix;

// ==================================================================
// Sharing #1: in-process clients share writes, and a forked child on the
// same shm reads identical slots + writes back visibly to the parent.
// (Covers the in-process two-client case as a subset of the fork case.)
// ==================================================================
static void test_share_and_fork() {
    TEST_BEGIN("clients + forked child share writes on same shm");

    const char* shm_name = "/test_cc_01";
    ServerLauncher launcher(shm_name, default_cfg());
    RadixClient a(shm_name);
    RadixClient b(shm_name);

    // In-process: b sees a's write, a sees b's write.
    int64_t t1[] = {1, 2, 3};
    int64_t t2[] = {4, 5, 6};
    auto h1 = RadixClient::compute_hashes(t1, 3, 1);
    auto h2 = RadixClient::compute_hashes(t2, 3, 1);
    insert_auto(a, h1.data(), 3);
    assert(b.query(h1.data(), 3, COMPONENT_MASK_FULL).common_hit == 3);
    insert_auto(b, h2.data(), 3);
    assert(a.query(h2.data(), 3, COMPONENT_MASK_FULL).common_hit == 3);
    assert(a.total_blocks() == 6);
    assert(a.name() == shm_name);

    // Cross-process: child attaches, reads a's prefix with identical slots,
    // then writes its own prefix that the parent must see.
    auto aq = a.query(h1.data(), 3, COMPONENT_MASK_FULL);   // parent's view of h1's slots
    pid_t pid = fork();
    if (pid == 0) {
        RadixClient child(shm_name);
        auto r = child.query(h1.data(), 3, COMPONENT_MASK_FULL);
        if (r.common_hit != 3) _exit(1);
        for (int i = 0; i < 3; i++)
            if (hit_slots(r)[i] != hit_slots(aq)[i]) _exit(2);

        int64_t t3[] = {50, 60};
        auto h3 = RadixClient::compute_hashes(t3, 2, 1);
        insert_auto(child, h3.data(), 2);
        if (child.total_blocks() != 8) _exit(3);
        _exit(0);
    }
    int status;
    waitpid(pid, &status, 0);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    assert(a.total_blocks() == 8);   // parent sees child's write

    expect_healthy(a);
    TEST_PASS();
}

// ==================================================================
// Sharing #2: a client constructed AFTER the writes still reads everything,
// with slot values matching the writer's own view.
// ==================================================================
static void test_attach_after_write() {
    TEST_BEGIN("client attached after writes reads identical slots");

    ServerLauncher launcher("/test_cc_02", default_cfg());
    RadixClient writer("/test_cc_02");

    int64_t tokens[] = {1, 2, 3, 4};
    auto h = RadixClient::compute_hashes(tokens, 4, 1);
    insert_auto(writer, h.data(), 4);
    auto wq = writer.query(h.data(), 4, COMPONENT_MASK_FULL);

    RadixClient reader("/test_cc_02");
    auto r = reader.query(h.data(), 4, COMPONENT_MASK_FULL);
    assert(r.common_hit == 4);
    for (int i = 0; i < 4; i++)
        assert(hit_slots(r)[i] == hit_slots(wq)[i]);
    assert(reader.total_radix_nodes() == 2);

    expect_healthy(writer);
    TEST_PASS();
}

// ==================================================================
// Concurrency #1: readers never see a torn query result while the writer
// hammers inserts and periodic full evictions (drives split/merge churn).
// A query must report common_hit in [0, N] — never garbage, never crash.
// ==================================================================
static void test_seqlock_no_torn_under_churn() {
    TEST_BEGIN("multi-reader query stays consistent under writer churn");

    ServerLauncher launcher("/test_cc_03", default_cfg());
    RadixClient client("/test_cc_03");

    constexpr int N = 8;
    constexpr int NUM_KEYS = 32;
    constexpr int READERS = 4;
    constexpr int DURATION_MS = 600;

    std::vector<std::vector<uint64_t>> keys(NUM_KEYS);
    for (int k = 0; k < NUM_KEYS; k++) {
        std::vector<int64_t> tok(N);
        for (int i = 0; i < N; i++) tok[i] = k * 1000 + i;
        keys[k] = RadixClient::compute_hashes(tok.data(), N, 1);
    }

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> bad_results{0};
    std::atomic<uint64_t> reader_iters{0};

    auto reader_fn = [&]() {
        RadixClient r("/test_cc_03");
        uint64_t local_iters = 0, local_bad = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            for (int k = 0; k < NUM_KEYS && !stop.load(); k++) {
                auto qr = r.query(keys[k].data(), N, COMPONENT_MASK_FULL,
                                  /*local_only=*/true, /*lock=*/false,
                                  /*update_meta=*/false);
                if (qr.common_hit < 0 || qr.common_hit > N)
                    local_bad++;
                local_iters++;
            }
        }
        reader_iters.fetch_add(local_iters, std::memory_order_relaxed);
        bad_results.fetch_add(local_bad, std::memory_order_relaxed);
    };

    std::vector<std::thread> readers;
    for (int i = 0; i < READERS; i++) readers.emplace_back(reader_fn);

    auto t0 = std::chrono::steady_clock::now();
    int k = 0;
    while (std::chrono::steady_clock::now() - t0 <
           std::chrono::milliseconds(DURATION_MS)) {
        insert_auto(client, keys[k].data(), N);
        k = (k + 1) % NUM_KEYS;
        if (k == 0)
            force_evict(client, (int)client.mempool_total());
    }
    stop.store(true, std::memory_order_relaxed);
    for (auto& t : readers) t.join();

    fprintf(stderr, "  [info] readers=%d iters=%lu bad=%lu\n",
            READERS, (unsigned long)reader_iters.load(),
            (unsigned long)bad_results.load());
    assert(bad_results.load() == 0);
    assert(reader_iters.load() > 0);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
// Concurrency #2: concurrent inc_ref/dec_ref on the same prefix balance out.
// After all (inc,dec) pairs, ref_count is back to 0 → force-evict clears it.
// ==================================================================
static void test_concurrent_inc_dec_ref() {
    TEST_BEGIN("concurrent inc_ref/dec_ref balance");

    ShmConfig cfg{1000, 16};
    ServerLauncher launcher("/test_cc_04", cfg);
    RadixClient writer("/test_cc_04");

    int64_t tok[] = {1, 2, 3, 4};
    auto h = RadixClient::compute_hashes(tok, 4, 1);
    insert_auto(writer, h.data(), 4);

    constexpr int THREADS = 6;
    constexpr int ITERS = 500;

    auto worker = [&]() {
        RadixClient r("/test_cc_04");
        for (int i = 0; i < ITERS; i++) {
            auto pin = pin_path(r, h.data(), 4);
            pin();
        }
    };

    std::vector<std::thread> ts;
    for (int i = 0; i < THREADS; i++) ts.emplace_back(worker);
    for (auto& t : ts) t.join();

    force_evict(writer, (int)writer.mempool_total());
    assert(writer.query(h.data(), 4, COMPONENT_MASK_FULL, false, false, false).common_hit == 0);

    expect_healthy(writer);
    TEST_PASS();
}

// ==================================================================
int main() {
    printf("=== test_client_concurrency ===\n\n");
    test_share_and_fork();
    test_attach_after_write();
    test_seqlock_no_torn_under_churn();
    test_concurrent_inc_dec_ref();
    return tests_report();
}
