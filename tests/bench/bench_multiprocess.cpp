// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "test_utils.h"
#include "shmradix/radix_client.h"

using namespace shmradix;
using Clock = std::chrono::high_resolution_clock;


// ======================================================================
// Configuration
// ======================================================================

struct BenchConfig {
    int requests_per_process = 1000;

    int system_prompt_len = 64;
    int isl_min = 10000;
    int isl_max = 20000;
    int osl_min = 32;
    int osl_max = 512;
    int max_turn = 3;

    int capacity_tokens = 5000000;
    double shared_prefix_ratio = 0.3;

    uint64_t base_seed = 42;
};

// ======================================================================
// Per-process result (stored in shared memory)
// ======================================================================

struct LatencySummary {
    double mean;
    double p50;
    double p90;
    double p99;
    double max;
    int count;
};

struct ProcessResult {
    int process_id;
    int total_turns;
    int total_turns_skipped;
    int64_t tokens_matched;
    int64_t tokens_inserted;
    int64_t tokens_evicted;
    int64_t input_tokens;
    double wall_time_s;
    LatencySummary match_lat;
    LatencySummary insert_lat;
    LatencySummary evict_lat;
    LatencySummary turn_lat;
};

// ======================================================================
// Turn / Session (pre-generated per process)
// ======================================================================

struct Turn {
    std::vector<int64_t> input_tokens;
    std::vector<uint64_t> input_hashes;
    std::vector<int64_t> full_tokens;
    std::vector<uint64_t> full_hashes;
    std::vector<int32_t> full_slots;
};

struct Session {
    int num_turns;
    int next_turn;
    std::vector<Turn> turns;
    bool done() const { return next_turn >= num_turns; }
};

// ======================================================================
// Latency collector (process-local)
// ======================================================================

struct LatencyCollector {
    std::vector<double> samples_us;

    void record(Clock::time_point t0, Clock::time_point t1) {
        samples_us.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    LatencySummary summarize() {
        LatencySummary s{};
        s.count = (int)samples_us.size();
        if (s.count == 0) return s;
        std::sort(samples_us.begin(), samples_us.end());
        double sum = 0;
        for (double v : samples_us) sum += v;
        s.mean = sum / s.count;
        auto pct = [&](double p) {
            return samples_us[(size_t)(p / 100.0 * (s.count - 1))];
        };
        s.p50 = pct(50);
        s.p90 = pct(90);
        s.p99 = pct(99);
        s.max = samples_us.back();
        return s;
    }
};

// ======================================================================
// Generate sessions for one process
// ======================================================================

static std::vector<Session> generate_sessions(
    const BenchConfig& cfg, int process_id, int num_sessions) {
    std::mt19937_64 rng(cfg.base_seed + (uint64_t)process_id * 1000003);
    int32_t slot_counter = process_id * 10000000;

    std::vector<int64_t> shared_sp(cfg.system_prompt_len);
    {
        std::mt19937_64 sp_rng(cfg.base_seed);
        for (int i = 0; i < cfg.system_prompt_len; i++)
            shared_sp[i] = (int64_t)(sp_rng() & 0x7FFFFFFFLL);
    }

    std::uniform_int_distribution<int> turn_dist(1, cfg.max_turn);
    std::uniform_int_distribution<int> isl_dist(cfg.isl_min, cfg.isl_max);
    std::uniform_int_distribution<int> osl_dist(cfg.osl_min, cfg.osl_max);
    std::uniform_real_distribution<double> ratio_dist(0.0, 1.0);

    std::vector<Session> sessions;
    sessions.reserve(num_sessions);

    for (int s = 0; s < num_sessions; s++) {
        Session session;
        session.num_turns = turn_dist(rng);
        session.next_turn = 0;

        std::vector<int64_t> system_prompt;
        if (ratio_dist(rng) < cfg.shared_prefix_ratio) {
            system_prompt = shared_sp;
        } else {
            system_prompt.resize(cfg.system_prompt_len);
            for (int i = 0; i < cfg.system_prompt_len; i++)
                system_prompt[i] = (int64_t)(rng() & 0x7FFFFFFFLL);
        }

        std::vector<int64_t> history = system_prompt;

        for (int t = 0; t < session.num_turns; t++) {
            Turn turn;
            int isl = isl_dist(rng);
            int osl = osl_dist(rng);

            std::vector<int64_t> new_input(isl);
            for (int i = 0; i < isl; i++)
                new_input[i] = (int64_t)(rng() & 0x7FFFFFFFLL);

            turn.input_tokens = history;
            turn.input_tokens.insert(turn.input_tokens.end(),
                                     new_input.begin(), new_input.end());

            std::vector<int64_t> output(osl);
            for (int i = 0; i < osl; i++)
                output[i] = (int64_t)(rng() & 0x7FFFFFFFLL);

            turn.full_tokens = turn.input_tokens;
            turn.full_tokens.insert(turn.full_tokens.end(),
                                    output.begin(), output.end());

            int input_len = (int)turn.input_tokens.size();
            int full_len = (int)turn.full_tokens.size();

            turn.input_hashes = RadixClient::compute_hashes(
                turn.input_tokens.data(), input_len, 1);

            turn.full_hashes = RadixClient::compute_hashes(
                turn.full_tokens.data(), full_len, 1);

            turn.full_slots.resize(full_len);
            for (int i = 0; i < full_len; i++)
                turn.full_slots[i] = slot_counter++;

            history = turn.full_tokens;
            session.turns.push_back(std::move(turn));
        }
        sessions.push_back(std::move(session));
    }
    return sessions;
}

// ======================================================================
// Worker process main loop
// ======================================================================

static void worker_main(const BenchConfig& cfg, int process_id,
                        const char* shm_name, ProcessResult* result) {
    RadixClient tree(shm_name);

    int concurrent = 20;
    int total_sessions = cfg.requests_per_process + concurrent;
    auto sessions = generate_sessions(cfg, process_id, total_sessions);

    std::vector<int> active;
    int next_session = 0;
    for (int i = 0; i < concurrent && next_session < total_sessions; i++)
        active.push_back(next_session++);

    std::mt19937_64 pick_rng(cfg.base_seed + process_id * 7 + 13);

    LatencyCollector match_col, insert_col, evict_col, turn_col;
    int turns_done = 0;
    int turns_skipped = 0;
    int64_t tokens_matched = 0, tokens_inserted = 0;
    int64_t tokens_evicted = 0, input_tokens = 0;

    auto wall_start = Clock::now();

    while (turns_done < cfg.requests_per_process && !active.empty()) {
        std::uniform_int_distribution<int> dist(0, (int)active.size() - 1);
        int pick = dist(pick_rng);
        int sid = active[pick];
        Session& session = sessions[sid];
        Turn& turn = session.turns[session.next_turn];

        auto turn_start = Clock::now();

        // 1. query (read lock)
        auto t0 = Clock::now();
        auto mr = tree.query(turn.input_hashes.data(),
                             (int)turn.input_hashes.size(),
                             COMPONENT_MASK_FULL);
        auto t1 = Clock::now();
        match_col.record(t0, t1);
        tokens_matched += mr.common_hit;
        input_tokens += (int64_t)turn.input_hashes.size();

        // 2. insert (write lock)
        t0 = Clock::now();
        insert_auto(tree, turn.full_hashes.data(), (int)turn.full_hashes.size());
        t1 = Clock::now();
        insert_col.record(t0, t1);
        tokens_inserted += (int64_t)turn.full_hashes.size();

        // 3. force eviction if over capacity (allocate_slots + immediate recycle)
        uint64_t cur = tree.total_blocks();
        if ((int64_t)cur > cfg.capacity_tokens) {
            int excess = (int)(cur - (uint64_t)cfg.capacity_tokens);
            uint32_t free_before = tree.mempool_free();
            std::vector<int32_t> tmp;
            t0 = Clock::now();
            tree.allocate_slots((int)free_before + excess, tmp);
            if (!tmp.empty())
                tree.recycle_slots(tmp.data(), (int)tmp.size());
            t1 = Clock::now();
            evict_col.record(t0, t1);
            tokens_evicted += (int64_t)(tree.mempool_free() - free_before);
        }

        auto turn_end = Clock::now();
        turn_col.record(turn_start, turn_end);

        session.next_turn++;
        turns_done++;

        if (session.done()) {
            active[pick] = active.back();
            active.pop_back();
            if (next_session < total_sessions)
                active.push_back(next_session++);
        }
    }

    auto wall_end = Clock::now();

    result->process_id = process_id;
    result->total_turns = turns_done;
    result->total_turns_skipped = turns_skipped;
    result->tokens_matched = tokens_matched;
    result->tokens_inserted = tokens_inserted;
    result->tokens_evicted = tokens_evicted;
    result->input_tokens = input_tokens;
    result->wall_time_s =
        std::chrono::duration<double>(wall_end - wall_start).count();
    result->match_lat = match_col.summarize();
    result->insert_lat = insert_col.summarize();
    result->evict_lat = evict_col.summarize();
    result->turn_lat = turn_col.summarize();
}

// ======================================================================
// ShmConfig helper
// ======================================================================

static ShmConfig config_for_bench(const BenchConfig& cfg, int max_procs) {
    uint64_t peak = (uint64_t)cfg.capacity_tokens * 2;
    uint32_t max_nodes = std::max((uint32_t)(peak / 4), (uint32_t)200000);
    uint32_t max_blocks = (uint32_t)peak;
    return ShmConfig{max_nodes, max_blocks};
}

// ======================================================================
// Run one round with N processes
// ======================================================================

static void print_lat_header() {
    printf("    %-6s %6s %6s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n",
           "Proc", "Turns", "Skip",
           "match_p50", "match_p99",
           "ins_p50", "ins_p99",
           "evict_p50", "evict_p99",
           "turn_p50", "turn_p99",
           "Reuse%", "Tput(r/s)");
}

static void print_lat_row(const ProcessResult& r) {
    double reuse = r.input_tokens > 0
        ? 100.0 * (double)r.tokens_matched / (double)r.input_tokens : 0.0;
    double tput = r.wall_time_s > 0 ? r.total_turns / r.wall_time_s : 0.0;
    printf("    %-6d %6d %6d %10.1f %10.1f %10.1f %10.1f %10.1f %10.1f %10.1f %10.1f %9.1f%% %10.1f\n",
           r.process_id, r.total_turns, r.total_turns_skipped,
           r.match_lat.p50, r.match_lat.p99,
           r.insert_lat.p50, r.insert_lat.p99,
           r.evict_lat.p50, r.evict_lat.p99,
           r.turn_lat.p50, r.turn_lat.p99,
           reuse, tput);
}

static void run_round(const BenchConfig& cfg, int num_procs,
                      const char* shm_name) {
    printf("\n=== N=%d process%s, %d requests/process ===\n\n",
           num_procs, num_procs > 1 ? "es" : "", cfg.requests_per_process);

    auto shm_cfg = config_for_bench(cfg, num_procs);
    ServerLauncher launcher(shm_name, shm_cfg);
    RadixClient tree(shm_name);

    // Shared memory for results
    size_t result_size = sizeof(ProcessResult) * num_procs;
    ProcessResult* results = (ProcessResult*)mmap(
        nullptr, result_size, PROT_READ | PROT_WRITE,
        MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    memset(results, 0, result_size);

    // Fork workers
    std::vector<pid_t> pids;
    auto fork_start = Clock::now();

    for (int i = 0; i < num_procs; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            worker_main(cfg, i, shm_name, &results[i]);
            _exit(0);
        }
        assert(pid > 0);
        pids.push_back(pid);
    }

    // Wait for all
    for (pid_t pid : pids) {
        int status;
        waitpid(pid, &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr, "  WARNING: child %d exited with status %d\n",
                    pid, WEXITSTATUS(status));
        }
    }

    auto fork_end = Clock::now();
    double total_wall = std::chrono::duration<double>(fork_end - fork_start).count();

    // Print per-process results
    printf("  Latency (us):\n");
    print_lat_header();
    printf("    %-6s %6s %6s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n",
           "──────", "──────", "──────",
           "──────────", "──────────",
           "──────────", "──────────",
           "──────────", "──────────",
           "──────────", "──────────",
           "──────────", "──────────");
    for (int i = 0; i < num_procs; i++)
        print_lat_row(results[i]);

    // Aggregate
    int64_t total_turns = 0, total_skipped = 0;
    int64_t total_matched = 0, total_inserted = 0;
    int64_t total_evicted = 0, total_input = 0;
    double max_wall = 0;
    for (int i = 0; i < num_procs; i++) {
        total_turns += results[i].total_turns;
        total_skipped += results[i].total_turns_skipped;
        total_matched += results[i].tokens_matched;
        total_inserted += results[i].tokens_inserted;
        total_evicted += results[i].tokens_evicted;
        total_input += results[i].input_tokens;
        if (results[i].wall_time_s > max_wall)
            max_wall = results[i].wall_time_s;
    }

    double agg_reuse = total_input > 0
        ? 100.0 * (double)total_matched / (double)total_input : 0.0;

    printf("\n  Aggregate:\n");
    printf("    Total wall time:      %.2f s\n", total_wall);
    printf("    Total turns:          %lld\n", (long long)total_turns);
    if (total_skipped > 0)
        printf("    Turns skipped:        %lld\n", (long long)total_skipped);
    printf("    Aggregate throughput: %.1f req/s\n",
           (double)total_turns / total_wall);
    printf("    Tokens inserted:      %lld\n", (long long)total_inserted);
    printf("    Tokens evicted:       %lld\n", (long long)total_evicted);
    printf("    Prefix reuse:         %.1f%%\n", agg_reuse);
    printf("    Final tree tokens:    %llu\n",
           (unsigned long long)tree.total_blocks());
    printf("    Final tree nodes:     %u\n", tree.total_radix_nodes());

    munmap(results, result_size);
}

// ======================================================================
// CLI
// ======================================================================

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --requests <int>        Requests per process (default: 1000)\n");
    printf("  --system-prompt <int>   System prompt length (default: 64)\n");
    printf("  --isl-min <int>         (default: 32)\n");
    printf("  --isl-max <int>         (default: 256)\n");
    printf("  --osl-min <int>         (default: 32)\n");
    printf("  --osl-max <int>         (default: 512)\n");
    printf("  --max-turn <int>        (default: 3)\n");
    printf("  --capacity <int>        Token capacity (default: 500000)\n");
    printf("  --shared-prefix <float> (default: 0.3)\n");
    printf("  --procs <list>          Comma-separated process counts (default: 1,2,4,8)\n");
    printf("  --seed <int>            (default: 42)\n");
    printf("  --help\n");
}

static std::vector<int> parse_int_list(const char* s) {
    std::vector<int> v;
    const char* p = s;
    while (*p) {
        v.push_back(atoi(p));
        while (*p && *p != ',') p++;
        if (*p == ',') p++;
    }
    return v;
}

int main(int argc, char** argv) {
    BenchConfig cfg;
    std::vector<int> proc_counts = {1, 2, 4, 8};

    for (int i = 1; i < argc; i++) {
        auto arg = [&](const char* name) { return strcmp(argv[i], name) == 0; };
        auto next_int = [&]() { return atoi(argv[++i]); };
        auto next_float = [&]() { return atof(argv[++i]); };
        auto next_str = [&]() { return argv[++i]; };

        if (arg("--help") || arg("-h")) { print_usage(argv[0]); return 0; }
        else if (arg("--requests"))       cfg.requests_per_process = next_int();
        else if (arg("--system-prompt"))  cfg.system_prompt_len = next_int();
        else if (arg("--isl-min"))        cfg.isl_min = next_int();
        else if (arg("--isl-max"))        cfg.isl_max = next_int();
        else if (arg("--osl-min"))        cfg.osl_min = next_int();
        else if (arg("--osl-max"))        cfg.osl_max = next_int();
        else if (arg("--max-turn"))       cfg.max_turn = next_int();
        else if (arg("--capacity"))       cfg.capacity_tokens = next_int();
        else if (arg("--shared-prefix"))  cfg.shared_prefix_ratio = next_float();
        else if (arg("--procs"))          proc_counts = parse_int_list(next_str());
        else if (arg("--seed"))           cfg.base_seed = (uint64_t)next_int();
        else {
            fprintf(stderr, "Unknown: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    printf("=== Multi-Process RadixTree Benchmark ===\n");
    printf("Config: requests/proc=%d ISL=[%d,%d] OSL=[%d,%d] max_turn=%d "
           "capacity=%d shared_prefix=%.0f%%\n",
           cfg.requests_per_process,
           cfg.isl_min, cfg.isl_max, cfg.osl_min, cfg.osl_max,
           cfg.max_turn, cfg.capacity_tokens,
           cfg.shared_prefix_ratio * 100);

    for (int n : proc_counts) {
        char name[64];
        snprintf(name, sizeof(name), "/bench_mp_%d", n);
        run_round(cfg, n, name);
    }

    printf("\nDone.\n");
    return 0;
}
