// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// bench_query — sustained query throughput across hit-rate and lock modes.
//
// Pipeline:
//   1. Warm up by inserting `warmup_factor * mempool_total / prefix_len`
//      ops so the cache reaches steady-state churn (some eviction).
//   2. Measure phase: cycle through the SAME stream rotation as warmup —
//      hits or misses depend on whether the stream's last hash sequence
//      is still resident in the tree.
//
// Each measure-loop op: query(... lock=use_lock, update_meta=use_meta).
// If lock=true and finalize is armed, finalize() is also timed (single
// "user-visible op" semantics, mirroring bench_insert).

#include "bench_util.h"
#include "test_utils.h"
#include "shmradix/radix_client.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

using namespace shmradix;
using bench::HashGen;
using bench::LatencyCollector;
using bench::WorkloadConfig;
using bench::now;
using bench::seconds_since;

struct QueryCfg {
    const char* label;
    uint32_t    max_blocks;
    WorkloadConfig wl;
    bool        lock;
    bool        update_meta;
    // working_set_ratio: how many op-sized inserts to do during warmup,
    // expressed as a multiplier on (mempool_total / prefix_len).
    //   0.5x: WS << mempool → near-100% hit
    //   1.5x: WS > mempool → partial hits (older streams already evicted)
    //   10x : WS >> mempool → mostly miss
    double      working_set_ratio;
    double      duration_sec;
};

struct QueryStats {
    int64_t ops          = 0;
    int64_t full_hits    = 0;
    int64_t partial      = 0;
    int64_t misses       = 0;
    double  duration_sec = 0;
    double  p50_us       = 0;
    double  p90_us       = 0;
    double  p99_us       = 0;
    double  max_us       = 0;
};

static QueryStats run_one(const QueryCfg& c) {
    std::string shm_name = std::string("/bench_q_") + c.label;
    ShmConfig shm;
    shm.max_blocks = c.max_blocks;

    ServerLauncher launcher(shm_name, shm);
    RadixClient client(shm_name);

    HashGen gen(c.wl);

    // ---- Warmup phase: insert enough ops to fill / overflow the pool.
    int prefix_len = c.wl.prefix_len_blocks;
    int64_t warmup_ops =
        (int64_t)((double)c.max_blocks * c.working_set_ratio / prefix_len);
    if (warmup_ops < 1) warmup_ops = 1;
    for (int64_t i = 0; i < warmup_ops; i++) {
        const auto& h = gen.next();
        auto ins = insert_auto(client, h.data(), (int)h.size());
        (void)ins;
    }

    // ---- Measure phase: re-query streams in rotation. Each stream's
    // last-emitted prefix is what currently lives in the tree (modulo
    // eviction). replay_stream(s) regenerates the hashes deterministically.
    LatencyCollector lat;
    QueryStats st;

    int N = c.wl.num_unique_streams;
    int s = 0;
    auto t_start = now();
    while (seconds_since(t_start) < c.duration_sec) {
        const auto& h = gen.replay_stream(s);
        s = (s + 1) % N;

        auto t0 = now();
        auto qr = client.query(h.data(), (int)h.size(),
                                COMPONENT_MASK_FULL, /*local_only=*/true,
                                c.lock, c.update_meta);
        if (qr.finalize) qr.finalize();
        auto t1 = now();
        lat.record(t0, t1);

        st.ops++;
        if (qr.common_hit == prefix_len)         st.full_hits++;
        else if (qr.common_hit > 0)              st.partial++;
        else                                            st.misses++;
    }
    st.duration_sec = seconds_since(t_start);
    lat.summarize(st.p50_us, st.p90_us, st.p99_us, st.max_us);
    return st;
}

static void print_header() {
    printf("%-22s %8s %10s %12s %8s %8s %8s %10s %10s %10s\n",
           "config", "dur(s)", "ops", "ops/s",
           "hit%", "partial%", "miss%",
           "p50_us", "p90_us", "p99_us");
    bench::print_separator();
}

static void print_row(const char* label, const QueryStats& st) {
    double ops_per_sec = st.duration_sec > 0
        ? (double)st.ops / st.duration_sec : 0;
    double total = (double)std::max<int64_t>(1, st.ops);
    double hit_pct     = 100.0 * (double)st.full_hits / total;
    double partial_pct = 100.0 * (double)st.partial   / total;
    double miss_pct    = 100.0 * (double)st.misses    / total;
    printf("%-22s %8.2f %10lld %12.0f %8.1f %8.1f %8.1f "
           "%10.1f %10.1f %10.1f\n",
           label, st.duration_sec, (long long)st.ops, ops_per_sec,
           hit_pct, partial_pct, miss_pct,
           st.p50_us, st.p90_us, st.p99_us);
}

int main() {
    printf("=== bench_query (warmup → measure; each op = query[+finalize if lock]) ===\n\n");

    std::vector<QueryCfg> cfgs;
    auto base_wl = [](){
        WorkloadConfig w;
        w.prefix_len_blocks    = 128;
        w.shared_prefix_blocks = 64;
        w.num_unique_streams   = 200;
        return w;
    };

    // 1. Full hit, pure read: lock=false, update_meta=false (no LRU writes).
    {
        QueryCfg c{"full-hit-readonly", 100'000, base_wl(),
                    false, false, 0.5, 5.0};
        cfgs.push_back(c);
    }
    // 2. Full hit + LRU updates (default user behavior).
    {
        QueryCfg c{"full-hit-update-lru", 100'000, base_wl(),
                    false, true, 0.5, 5.0};
        cfgs.push_back(c);
    }
    // 3. Full hit + lock + finalize (inc_ref/dec_ref overhead).
    {
        QueryCfg c{"full-hit-lock", 100'000, base_wl(),
                    true, true, 0.5, 5.0};
        cfgs.push_back(c);
    }
    // 4. Partial hits — WS > mempool, oldest streams evicted.
    //    num_streams=800 * prefix_len=128 = 102400 blocks > mempool 50000.
    {
        WorkloadConfig wl = base_wl();
        wl.num_unique_streams = 800;
        QueryCfg c{"partial-hit", 50'000, wl, false, true, 2.0, 5.0};
        cfgs.push_back(c);
    }
    // 5. Mostly miss — WS >> mempool.
    //    num_streams=2000 * prefix_len=128 = 256000 blocks >> mempool 5000.
    {
        WorkloadConfig wl = base_wl();
        wl.num_unique_streams = 2000;
        QueryCfg c{"mostly-miss", 5'000, wl, false, true, 50.0, 5.0};
        cfgs.push_back(c);
    }
    // 6. Long prefix hits (deeper walks).
    {
        WorkloadConfig wl;
        wl.prefix_len_blocks    = 1024;
        wl.shared_prefix_blocks = 256;
        wl.num_unique_streams   = 50;
        QueryCfg c{"long-prefix-hit", 200'000, wl, false, true, 0.5, 5.0};
        cfgs.push_back(c);
    }

    print_header();
    for (auto& cfg : cfgs) {
        auto st = run_one(cfg);
        print_row(cfg.label, st);
        fflush(stdout);
    }
    printf("\nDone.\n");
    return 0;
}
