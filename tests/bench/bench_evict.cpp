// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "test_utils.h"
#include "shmradix/radix_client.h"

using namespace shmradix;

struct SeqData {
    std::vector<int64_t> tokens;
    std::vector<uint64_t> hashes;
    std::vector<int32_t> slots;
};

static SeqData make_seq(std::mt19937_64& rng, int len, int slot_base) {
    SeqData s;
    s.tokens.resize(len);
    s.slots.resize(len);
    for (int i = 0; i < len; i++) {
        s.tokens[i] = (int64_t)(rng() & 0x7FFFFFFFLL);
        s.slots[i] = slot_base + i;
    }
    s.hashes = RadixClient::compute_hashes(s.tokens.data(), len, 1);
    return s;
}

using Clock = std::chrono::high_resolution_clock;

static double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static double elapsed_us(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - start).count();
}

// ======================================================================
// Bench 1: Bulk eviction throughput — many disjoint leaves
// ======================================================================
static void bench_bulk_eviction() {
    printf("  %-55s", "bulk reserve (1000 disjoint leaves, 10 tok each)");
    fflush(stdout);

    constexpr int NUM_SEQS = 1000;
    constexpr int SEQ_LEN = 10;
    constexpr int TOTAL = NUM_SEQS * SEQ_LEN;

    ShmConfig cfg{200000,(uint32_t)TOTAL};
    ServerLauncher launcher("/bench_evict_01", cfg);
    RadixClient tree("/bench_evict_01");

    std::mt19937_64 rng(42);
    for (int i = 0; i < NUM_SEQS; i++) {
        auto s = make_seq(rng, SEQ_LEN, i * SEQ_LEN);
        insert_auto(tree, s.hashes.data(), SEQ_LEN);
    }

    assert(tree.total_blocks() == TOTAL);

    auto t0 = Clock::now();
    int free_after = force_evict(tree, TOTAL);
    auto t1 = Clock::now();

    assert(free_after == TOTAL);
    assert(tree.total_blocks() == 0);

    printf("%8.2f ms  (%d tokens)\n", elapsed_ms(t0, t1), TOTAL);
}

// ======================================================================
// Bench 2: Incremental eviction — reserve in small increments
// ======================================================================
static void bench_incremental_eviction() {
    printf("  %-55s", "incremental reserve (100 rounds x 100 tokens)");
    fflush(stdout);

    constexpr int NUM_SEQS = 1000;
    constexpr int SEQ_LEN = 10;
    constexpr int TOTAL = NUM_SEQS * SEQ_LEN;
    constexpr int EVICT_PER_ROUND = 100;
    constexpr int ROUNDS = TOTAL / EVICT_PER_ROUND;

    ShmConfig cfg{200000,(uint32_t)TOTAL};
    ServerLauncher launcher("/bench_evict_02", cfg);
    RadixClient tree("/bench_evict_02");

    std::mt19937_64 rng(123);
    for (int i = 0; i < NUM_SEQS; i++) {
        auto s = make_seq(rng, SEQ_LEN, i * SEQ_LEN);
        insert_auto(tree, s.hashes.data(), SEQ_LEN);
    }

    int target = EVICT_PER_ROUND;
    auto t0 = Clock::now();
    for (int round = 0; round < ROUNDS; round++) {
        force_evict(tree, target);
        target += EVICT_PER_ROUND;
    }
    auto t1 = Clock::now();

    double total_ms = elapsed_ms(t0, t1);
    printf("%8.2f ms  (avg %.1f us/round)\n", total_ms,
           (total_ms * 1000.0) / ROUNDS);
}

// ======================================================================
// Bench 3: LRU sort overhead — large candidate set
// ======================================================================
static void bench_lru_sort_overhead() {
    printf("  %-55s", "LRU sort overhead (5000 leaves, reserve 1 token)");
    fflush(stdout);

    constexpr int NUM_SEQS = 5000;
    constexpr int SEQ_LEN = 4;
    constexpr int TOTAL = NUM_SEQS * SEQ_LEN;

    ShmConfig cfg{200000,(uint32_t)TOTAL};
    ServerLauncher launcher("/bench_evict_03", cfg);
    RadixClient tree("/bench_evict_03");

    std::mt19937_64 rng(777);
    for (int i = 0; i < NUM_SEQS; i++) {
        auto s = make_seq(rng, SEQ_LEN, i * SEQ_LEN);
        insert_auto(tree, s.hashes.data(), SEQ_LEN);
    }

    int target = 1;
    auto t0 = Clock::now();
    constexpr int ITERS = 100;
    for (int i = 0; i < ITERS; i++) {
        force_evict(tree, target);
        target += SEQ_LEN;
    }
    auto t1 = Clock::now();

    double total_us = elapsed_us(t0, t1);
    printf("%8.2f us/call  (%d calls)\n", total_us / ITERS, ITERS);
}

// ======================================================================
// Bench 4: Mixed insert/reserve workload (steady-state simulation)
// ======================================================================
static void bench_mixed_workload() {
    printf("  %-55s", "mixed insert+reserve workload (2000 cycles)");
    fflush(stdout);

    constexpr int CAPACITY_TOKENS = 5000;
    constexpr int SEQ_LEN = 8;
    constexpr int CYCLES = 2000;

    ShmConfig cfg{200000,500000};
    ServerLauncher launcher("/bench_evict_06", cfg);
    RadixClient tree("/bench_evict_06");

    std::mt19937_64 rng(314159);
    int slot_counter = 0;
    int total_inserted = 0;
    int total_evicted = 0;

    auto t0 = Clock::now();
    for (int c = 0; c < CYCLES; c++) {
        auto s = make_seq(rng, SEQ_LEN, slot_counter);
        insert_auto(tree, s.hashes.data(), SEQ_LEN);
        slot_counter += SEQ_LEN;
        total_inserted += SEQ_LEN;

        if ((int)tree.total_blocks() > CAPACITY_TOKENS) {
            int excess = (int)tree.total_blocks() - CAPACITY_TOKENS;
            uint32_t free_before = tree.mempool_free();
            force_evict(tree, (int)free_before + excess);
            total_evicted += (int)(tree.mempool_free() - free_before);
        }
    }
    auto t1 = Clock::now();

    double total_ms = elapsed_ms(t0, t1);
    printf("%8.2f ms  (ins %d, evict %d, avg %.1f us/cycle)\n",
           total_ms, total_inserted, total_evicted,
           (total_ms * 1000.0) / CYCLES);
}

// ======================================================================
// Bench 7: ref_count filtering overhead — half locked
// ======================================================================
static void bench_ref_count_filtering() {
    printf("  %-55s", "ref_count filtering (2000 leaves, 50%% locked)");
    fflush(stdout);

    constexpr int NUM_SEQS = 2000;
    constexpr int SEQ_LEN = 4;
    constexpr int TOTAL = NUM_SEQS * SEQ_LEN;

    ShmConfig cfg{200000,(uint32_t)TOTAL};
    ServerLauncher launcher("/bench_evict_07", cfg);
    RadixClient tree("/bench_evict_07");

    std::mt19937_64 rng(65536);
    std::vector<SeqData> seqs;
    std::vector<shmradix::FinalizeFn> locks;

    for (int i = 0; i < NUM_SEQS; i++) {
        auto s = make_seq(rng, SEQ_LEN, i * SEQ_LEN);
        insert_auto(tree, s.hashes.data(), SEQ_LEN);
        if (i % 2 == 0) {
            locks.push_back(pin_path(tree, s.hashes.data(), SEQ_LEN));
        }
        seqs.push_back(std::move(s));
    }

    auto t0 = Clock::now();
    int free_after = force_evict(tree, TOTAL);
    auto t1 = Clock::now();

    int evicted = TOTAL - (int)tree.total_blocks();
    int expected_max = (NUM_SEQS / 2) * SEQ_LEN;
    assert(evicted <= expected_max);

    printf("%8.2f ms  (%d/%d tokens evicted)\n",
           elapsed_ms(t0, t1), evicted, TOTAL);

    for (auto& l : locks) l();
}

// ======================================================================
// Helpers for large-scale tests
// ======================================================================

static ShmConfig config_for_leaves(int num_leaves) {
    uint32_t max_nodes = (uint32_t)num_leaves + 1024;
    return ShmConfig{max_nodes, (uint32_t)num_leaves + 1024};
}

static void populate_disjoint(RadixClient& tree, int num_leaves,
                              std::mt19937_64& rng) {
    for (int i = 0; i < num_leaves; i++) {
        int64_t tok = (int64_t)(rng());
        auto hash = RadixClient::compute_hashes(&tok, 1, 1);
        insert_auto(tree, hash.data(), 1);
    }
}

static const char* fmt_num(int n, char* buf, size_t bufsz) {
    if (n >= 10000000)     snprintf(buf, bufsz, "%dM", n / 1000000);
    else if (n >= 1000000) snprintf(buf, bufsz, "%.1fM", n / 1000000.0);
    else if (n >= 1000)    snprintf(buf, bufsz, "%dK", n / 1000);
    else                   snprintf(buf, bufsz, "%d", n);
    return buf;
}

static double elapsed_sec(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

// ======================================================================
// Bench 8: Reserve scalability — reserve-all at varying tree sizes
// ======================================================================
static void bench_scalability_evict_all() {
    printf("  [Scalability] reserve-all throughput (1-token disjoint leaves)\n");
    printf("    %-12s %10s %10s %12s %12s\n",
           "Leaves", "Evicted", "Setup", "Reserve", "us/token");
    printf("    %-12s %10s %10s %12s %12s\n",
           "------", "-------", "-----", "-------", "--------");

    int sizes[] = {100, 1000, 10000};

    for (int sz : sizes) {
        auto cfg = config_for_leaves(sz);
        char name[64];
        snprintf(name, sizeof(name), "/bench_sc_all_%d", sz);

        ServerLauncher launcher(name, cfg);
        RadixClient tree(name);
        std::mt19937_64 rng(sz);

        auto ts0 = Clock::now();
        populate_disjoint(tree, sz, rng);
        auto ts1 = Clock::now();

        int total = (int)tree.total_blocks();

        auto te0 = Clock::now();
        force_evict(tree, (int)tree.mempool_total());
        auto te1 = Clock::now();

        int evicted = total - (int)tree.total_blocks();
        char nbuf[32];
        printf("    %-12s %10d %8.2fs %10.2fms %10.2f\n",
               fmt_num(sz, nbuf, sizeof(nbuf)),
               evicted,
               elapsed_sec(ts0, ts1),
               elapsed_ms(te0, te1),
               evicted > 0 ? elapsed_us(te0, te1) / evicted : 0.0);
        fflush(stdout);
    }
}

// ======================================================================
// Bench 9: LRU sort overhead scalability — reserve 1 token at each scale
// ======================================================================
static void bench_scalability_sort_overhead() {
    printf("  [Scalability] LRU sort overhead (reserve 1 token)\n");
    printf("    %-12s %10s %12s\n", "Leaves", "Calls", "us/call");
    printf("    %-12s %10s %12s\n", "------", "-----", "-------");

    struct ScaleSpec { int leaves; int calls; };
    ScaleSpec specs[] = {
        {100,      1000},
        {1000,     500},
        {10000,    200},
        {100000,   50},
    };

    for (auto& sp : specs) {
        auto cfg = config_for_leaves(sp.leaves);
        char name[64];
        snprintf(name, sizeof(name), "/bench_sc_sort_%d", sp.leaves);

        ServerLauncher launcher(name, cfg);
        RadixClient tree(name);
        std::mt19937_64 rng(sp.leaves + 7);
        populate_disjoint(tree, sp.leaves, rng);

        int target = 1;
        auto t0 = Clock::now();
        for (int i = 0; i < sp.calls; i++) {
            force_evict(tree, target);
            target++;
        }
        auto t1 = Clock::now();

        char nbuf[32];
        printf("    %-12s %10d %10.1f\n",
               fmt_num(sp.leaves, nbuf, sizeof(nbuf)),
               sp.calls,
               elapsed_us(t0, t1) / sp.calls);
        fflush(stdout);
    }
}

// ======================================================================
// Bench 10: Incremental reserve scalability — steady-state at each scale
// ======================================================================
static void bench_scalability_incremental() {
    printf("  [Scalability] incremental reserve (evict 1%% per call)\n");
    printf("    %-12s %10s %10s %12s %12s\n",
           "Leaves", "Rounds", "EvictTok", "Total", "us/round");
    printf("    %-12s %10s %10s %12s %12s\n",
           "------", "------", "--------", "-----", "--------");

    int sizes[] = {1000, 10000, 100000};

    for (int sz : sizes) {
        auto cfg = config_for_leaves(sz);
        char name[64];
        snprintf(name, sizeof(name), "/bench_sc_inc_%d", sz);

        ServerLauncher launcher(name, cfg);
        RadixClient tree(name);
        std::mt19937_64 rng(sz + 13);
        populate_disjoint(tree, sz, rng);

        int total = (int)tree.total_blocks();
        int per_round = std::max(1, total / 100);
        int rounds = 10;

        int target = per_round;
        uint32_t blocks_before = tree.total_blocks();
        auto t0 = Clock::now();
        for (int i = 0; i < rounds; i++) {
            force_evict(tree, target);
            target += per_round;
        }
        auto t1 = Clock::now();

        int evicted = (int)blocks_before - (int)tree.total_blocks();
        char nbuf[32];
        printf("    %-12s %10d %10d %9.2fms %10.1f\n",
               fmt_num(sz, nbuf, sizeof(nbuf)),
               rounds, evicted,
               elapsed_ms(t0, t1),
               elapsed_us(t0, t1) / rounds);
        fflush(stdout);
    }
}

// ======================================================================

int main() {
    printf("=== Eviction Performance Benchmarks ===\n\n");

    bench_bulk_eviction();
    bench_incremental_eviction();
    bench_lru_sort_overhead();
    bench_mixed_workload();
    bench_ref_count_filtering();
    printf("\n--- Large-Scale Scalability (100 ~ 10M leaves) ---\n\n");
    bench_scalability_evict_all();
    printf("\n");
    bench_scalability_sort_overhead();
    printf("\n");
    bench_scalability_incremental();

    printf("\nDone.\n");
    return 0;
}
