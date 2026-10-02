// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// ======================================================================
// Long-Running LLM Workload Stress Test for RadixTree
//
// Simulates multi-turn LLM inference across single or multiple nodes.
// Each "client" thread acts as an inference engine: query → transfer →
// prefill/decode → insert → transfer.  Monitors actual vs theoretical
// hit rates, query latency, and detects crashes/corruption.
// ======================================================================

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include "test_utils.h"
#include "shmradix/radix_client.h"

using namespace shmradix;
using Clock = std::chrono::high_resolution_clock;

// ======================================================================
// Signal handling
// ======================================================================

static volatile sig_atomic_t g_running = 1;
static void signal_handler(int) { g_running = 0; }

// ======================================================================
// Configuration
// ======================================================================

static constexpr int MAX_NODES = 32;
static constexpr int MAX_TEMPLATES = 64;

struct StressConfig {
    int num_nodes             = 1;
    int clients_per_node      = 8;
    int max_concurrency       = 0;      // 0 = num_nodes * clients_per_node
    double target_qps         = 100.0;
    int isl_min               = 32;     // new user-input tokens per turn (min)
    int isl_max               = 512;    // new user-input tokens per turn (max)
    int osl_min               = 16;     // model output tokens per turn (min)
    int osl_max               = 256;    // model output tokens per turn (max)
    int system_prompt_length  = 256;    // tokens
    int max_turns             = 10;
    int block_size            = 16;     // tokens per hash block
    int duration_seconds      = 0;      // 0 = forever
    uint64_t seed             = 42;
    int report_interval_s     = 10;
    uint32_t max_nodes_shm    = 200000;
    uint32_t max_blocks_shm   = 500000;
    // data_pool_ratio removed — DataPool auto-sized from max_nodes
    int num_templates         = 5;      // distinct system prompt templates
    int conv_pool_size        = 200;    // active conversation pool

    // Simulation timing (ms per unit)
    double transfer_ms_per_block = 0.01;  // KV cache transfer
    double prefill_ms_per_block  = 0.05;  // prefill compute
    double decode_ms_per_token   = 0.02;  // decode per output token
};

// ======================================================================
// Data structures
// ======================================================================

struct TurnData {
    std::vector<int64_t> user_input;
    std::vector<int64_t> model_output;
};

struct ConvState {
    int id;
    int template_id;
    int num_turns;            // total turns for this conversation
    int current_turn;
    bool in_flight;
    bool complete;

    std::vector<TurnData> turns;
    std::vector<int64_t> system_prompt;

    // Tracks how many hashes of this conv exist on each node
    int hashes_on_node[MAX_NODES];

    ConvState() : id(0), template_id(0), num_turns(0), current_turn(0),
                  in_flight(false), complete(false) {
        memset(hashes_on_node, 0, sizeof(hashes_on_node));
    }
};

struct Request {
    int conv_id;
    int conv_pool_idx;
    int turn;
    int assigned_node;
    int template_id;
    std::vector<int64_t> input_tokens;    // full input sequence for this turn
    int new_token_count;                   // number of NEW tokens this turn
    int osl;                               // output tokens to generate
    int expected_hit_hashes;               // theoretical hit (hash blocks)
    std::vector<int64_t> output_tokens;    // pre-generated output tokens
};

// ======================================================================
// Per-node work queue
// ======================================================================

struct NodeQueue {
    std::mutex mtx;
    std::condition_variable cv;
    std::deque<Request> requests;
    bool shutdown = false;

    void push(Request&& req) {
        std::lock_guard<std::mutex> lock(mtx);
        requests.push_back(std::move(req));
        cv.notify_one();
    }

    bool pop(Request& out) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] { return !requests.empty() || shutdown; });
        if (shutdown && requests.empty()) return false;
        if (requests.empty()) return false;
        out = std::move(requests.front());
        requests.pop_front();
        return true;
    }

    void signal_shutdown() {
        std::lock_guard<std::mutex> lock(mtx);
        shutdown = true;
        cv.notify_all();
    }
};

// ======================================================================
// Concurrency limiter
// ======================================================================

struct ConcurrencyLimiter {
    std::mutex mtx;
    std::condition_variable cv;
    int slots;

    explicit ConcurrencyLimiter(int max) : slots(max) {}

    void acquire() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] { return slots > 0 || !g_running; });
        if (slots > 0) slots--;
    }

    void release() {
        std::lock_guard<std::mutex> lock(mtx);
        slots++;
        cv.notify_one();
    }
};

// ======================================================================
// Per-client metrics (cache-line aligned, no false sharing)
// ======================================================================

struct alignas(64) ClientMetrics {
    int64_t queries         = 0;
    int64_t actual_hit      = 0;
    int64_t expected_hit    = 0;
    int64_t total_blocks    = 0;  // total hashes queried
    double  query_time_us   = 0;
    double  max_query_us    = 0;
    int64_t inserts         = 0;
    int64_t insert_errors   = 0;
    int64_t requests_done   = 0;

    // Double-buffered per-interval latency collection.
    // Client writes to buf[active]; reporter swaps active then reads the other.
    std::vector<double> latency_buf[2];
    std::atomic<int> active_buf{0};
};

// ======================================================================
// Global template tracking (for theoretical hit computation)
// ======================================================================

static std::atomic<bool> g_template_on_node[MAX_NODES][MAX_TEMPLATES];

// ======================================================================
// Conversation generation
// ======================================================================

static void generate_conversation(ConvState& conv, int id,
                                  const std::vector<std::vector<int64_t>>& templates,
                                  const StressConfig& cfg,
                                  std::mt19937_64& rng) {
    conv.id = id;
    conv.current_turn = 0;
    conv.in_flight = false;
    conv.complete = false;
    memset(conv.hashes_on_node, 0, sizeof(conv.hashes_on_node));

    std::uniform_int_distribution<int> tmpl_dist(0, (int)templates.size() - 1);
    conv.template_id = tmpl_dist(rng);
    conv.system_prompt = templates[conv.template_id];

    std::uniform_int_distribution<int> turn_dist(1, cfg.max_turns);
    conv.num_turns = turn_dist(rng);

    int isl_blk_min = std::max(1, cfg.isl_min / cfg.block_size);
    int isl_blk_max = std::max(isl_blk_min, cfg.isl_max / cfg.block_size);
    int osl_blk_min = std::max(1, cfg.osl_min / cfg.block_size);
    int osl_blk_max = std::max(osl_blk_min, cfg.osl_max / cfg.block_size);

    std::uniform_int_distribution<int> isl_dist(isl_blk_min, isl_blk_max);
    std::uniform_int_distribution<int> osl_dist(osl_blk_min, osl_blk_max);

    conv.turns.resize(conv.num_turns);
    for (int t = 0; t < conv.num_turns; t++) {
        int isl_tokens = isl_dist(rng) * cfg.block_size;
        int osl_tokens = osl_dist(rng) * cfg.block_size;

        conv.turns[t].user_input.resize(isl_tokens);
        for (int i = 0; i < isl_tokens; i++)
            conv.turns[t].user_input[i] = (int64_t)(rng() & 0x7FFFFFFFLL);

        conv.turns[t].model_output.resize(osl_tokens);
        for (int i = 0; i < osl_tokens; i++)
            conv.turns[t].model_output[i] = (int64_t)(rng() & 0x7FFFFFFFLL);
    }
}

// Build full input token sequence for a given turn
static std::vector<int64_t> build_input_tokens(const ConvState& conv, int turn) {
    std::vector<int64_t> tokens;
    int est_size = (int)conv.system_prompt.size();
    for (int t = 0; t <= turn; t++)
        est_size += (int)conv.turns[t].user_input.size()
                  + (t < turn ? (int)conv.turns[t].model_output.size() : 0);
    tokens.reserve(est_size);

    tokens.insert(tokens.end(), conv.system_prompt.begin(),
                  conv.system_prompt.end());
    for (int t = 0; t < turn; t++) {
        tokens.insert(tokens.end(), conv.turns[t].user_input.begin(),
                      conv.turns[t].user_input.end());
        tokens.insert(tokens.end(), conv.turns[t].model_output.begin(),
                      conv.turns[t].model_output.end());
    }
    tokens.insert(tokens.end(), conv.turns[turn].user_input.begin(),
                  conv.turns[turn].user_input.end());
    return tokens;
}

// ======================================================================
// Client thread
// ======================================================================

static void client_thread_fn(int thread_id, int node_id,
                              const std::string& shm_name,
                              NodeQueue& queue,
                              ConcurrencyLimiter& limiter,
                              ClientMetrics& metrics,
                              std::vector<ConvState>& convs,
                              std::mutex& conv_mtx,
                              const StressConfig& cfg) {
    RadixClient tree(shm_name);
    metrics.latency_buf[0].reserve(4096);
    metrics.latency_buf[1].reserve(4096);

    while (g_running) {
        Request req;
        if (!queue.pop(req)) break;

        int num_tokens = (int)req.input_tokens.size();
        int num_hashes = num_tokens / cfg.block_size;

        if (num_hashes <= 0) {
            limiter.release();
            continue;
        }

        auto input_hashes = RadixClient::compute_hashes(
            req.input_tokens.data(), num_tokens, cfg.block_size);

        // ---- Step 1: Query + lock (inc_ref on matched path) ----
        auto t0 = Clock::now();
        auto qr = tree.query(input_hashes.data(), num_hashes,
                             COMPONENT_MASK_FULL,
                             /*local_only=*/false, /*lock=*/true);
        auto t1 = Clock::now();
        double query_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

        int actual_hit = qr.common_hit;

        // ---- Step 2: Simulate KV cache transfer (proportional to hit) ----
        if (actual_hit > 0) {
            int transfer_us = (int)(actual_hit * cfg.transfer_ms_per_block * 1000);
            if (transfer_us > 0)
                std::this_thread::sleep_for(std::chrono::microseconds(transfer_us));
        }

        // ---- Step 3: Unlock matched path ----
        // query(lock=true) pinned the hit; the one-shot handle releases it.
        qr.finalize();

        // ---- Step 4: Simulate prefill + decode ----
        {
            int unmatched = num_hashes - actual_hit;
            int prefill_us = (int)(unmatched * cfg.prefill_ms_per_block * 1000);
            int decode_us  = (int)(req.osl * cfg.decode_ms_per_token * 1000);
            int compute_us = prefill_us + decode_us;
            if (compute_us > 0)
                std::this_thread::sleep_for(std::chrono::microseconds(compute_us));
        }

        // ---- Step 5: Build full output sequence and insert ----
        std::vector<int64_t> full_tokens;
        full_tokens.reserve(req.input_tokens.size() + req.output_tokens.size());
        full_tokens.insert(full_tokens.end(),
                           req.input_tokens.begin(), req.input_tokens.end());
        full_tokens.insert(full_tokens.end(),
                           req.output_tokens.begin(), req.output_tokens.end());

        int full_num_tokens = (int)full_tokens.size();
        int full_num_hashes = full_num_tokens / cfg.block_size;

        shmradix::InsertResult ins_result;
        if (full_num_hashes > 0) {
            auto full_hashes = RadixClient::compute_hashes(
                full_tokens.data(), full_num_tokens, cfg.block_size);
            ins_result = insert_auto(tree, full_hashes.data(), full_num_hashes);

            // ---- Step 6: Lock newly inserted nodes, simulate transfer ----
            int new_blocks = full_num_hashes - actual_hit;
            if (new_blocks > 0) {
                auto pin = pin_path(tree, full_hashes.data(), full_num_hashes);
                int xfer_us = (int)(new_blocks * cfg.transfer_ms_per_block * 1000);
                if (xfer_us > 0)
                    std::this_thread::sleep_for(std::chrono::microseconds(xfer_us));
                pin();
            }
        }

        // ---- Step 7: Update metrics ----
        metrics.queries++;
        metrics.actual_hit    += actual_hit;
        metrics.expected_hit  += req.expected_hit_hashes;
        metrics.total_blocks  += num_hashes;
        metrics.query_time_us += query_us;
        if (query_us > metrics.max_query_us) metrics.max_query_us = query_us;
        metrics.inserts++;
        metrics.requests_done++;
        if (ins_result.error != shmradix::InsertError::OK)
            metrics.insert_errors++;

        metrics.latency_buf[metrics.active_buf.load(std::memory_order_relaxed)]
            .push_back(query_us);

        // ---- Step 8: Update conversation state ----
        {
            std::lock_guard<std::mutex> lock(conv_mtx);
            auto& conv = convs[req.conv_pool_idx];
            conv.hashes_on_node[node_id] = full_num_hashes;
            conv.current_turn++;
            conv.in_flight = false;
            if (conv.current_turn >= conv.num_turns)
                conv.complete = true;
            g_template_on_node[node_id][conv.template_id].store(
                true, std::memory_order_relaxed);
        }

        limiter.release();
    }
}

// ======================================================================
// Percentile helper
// ======================================================================

static double percentile(std::vector<double>& data, double p) {
    if (data.empty()) return 0;
    std::sort(data.begin(), data.end());
    size_t idx = (size_t)(p / 100.0 * (double)(data.size() - 1));
    return data[std::min(idx, data.size() - 1)];
}

// ======================================================================
// Reporter thread
// ======================================================================

struct Snapshot {
    int64_t queries      = 0;
    int64_t actual_hit   = 0;
    int64_t expected_hit = 0;
    int64_t total_blocks = 0;
    double  query_us     = 0;
    double  max_query_us = 0;
    int64_t inserts      = 0;
    int64_t errors       = 0;
    int64_t done         = 0;
};

static void reporter_thread_fn(const std::vector<ClientMetrics*>& all_metrics,
                                 const std::vector<ServerLauncher*>& servers,
                                 const std::vector<std::string>& shm_names,
                                 const StressConfig& cfg) {
    auto start = Clock::now();
    Snapshot prev;

    while (g_running) {
        for (int i = 0; i < cfg.report_interval_s * 10 && g_running; i++)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (!g_running) break;

        auto now = Clock::now();
        double elapsed = std::chrono::duration<double>(now - start).count();

        Snapshot cur;
        // Swap active buffer on each client, then drain the inactive one
        std::vector<double> all_latencies;
        for (auto* m : all_metrics) {
            cur.queries      += m->queries;
            cur.actual_hit   += m->actual_hit;
            cur.expected_hit += m->expected_hit;
            cur.total_blocks += m->total_blocks;
            cur.query_us     += m->query_time_us;
            if (m->max_query_us > cur.max_query_us)
                cur.max_query_us = m->max_query_us;
            cur.inserts += m->inserts;
            cur.errors  += m->insert_errors;
            cur.done    += m->requests_done;

            int old = m->active_buf.load(std::memory_order_relaxed);
            m->active_buf.store(1 - old, std::memory_order_release);
            // Brief pause so in-flight push_back on old buffer completes
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            all_latencies.insert(all_latencies.end(),
                                  m->latency_buf[old].begin(),
                                  m->latency_buf[old].end());
            m->latency_buf[old].clear();
        }

        // Interval metrics
        int64_t dq = cur.queries - prev.queries;
        double interval_qps = dq / (double)cfg.report_interval_s;
        double overall_hit = cur.total_blocks > 0
            ? 100.0 * cur.actual_hit / cur.total_blocks : 0;
        double theoretical_hit = cur.total_blocks > 0
            ? 100.0 * cur.expected_hit / cur.total_blocks : 0;
        double hit_ratio = cur.expected_hit > 0
            ? (double)cur.actual_hit / cur.expected_hit : 0;

        // Per-interval average latency
        double interval_query_us = cur.query_us - prev.query_us;
        double avg_query_us = dq > 0 ? interval_query_us / dq : 0;
        // Per-interval max
        double interval_max_us = 0;
        if (!all_latencies.empty()) {
            for (double v : all_latencies)
                if (v > interval_max_us) interval_max_us = v;
        }

        double p50 = 0, p90 = 0, p99 = 0;
        if (!all_latencies.empty()) {
            p50 = percentile(all_latencies, 50);
            p90 = percentile(all_latencies, 90);
            p99 = percentile(all_latencies, 99);
        }

        int elapsed_m = (int)(elapsed / 60);
        int elapsed_s = (int)elapsed % 60;

        printf("[%02d:%02d] QPS=%.1f  Total=%lld  "
               "Hit=%.1f%%  TheoHit=%.1f%%  Ratio=%.3f  "
               "Avg=%.1fus  P50=%.1fus  P90=%.1fus  P99=%.1fus  Max=%.0fus  "
               "Errors=%lld",
               elapsed_m, elapsed_s, interval_qps,
               (long long)cur.done,
               overall_hit, theoretical_hit, hit_ratio,
               avg_query_us, p50, p90, p99, interval_max_us,
               (long long)cur.errors);

        // Per-node tree stats + crash detection
        for (int n = 0; n < cfg.num_nodes; n++) {
            int exit_status = 0, signal_num = 0;
            bool alive = servers[n]->check_alive(exit_status, signal_num);
            if (!alive) {
                if (signal_num > 0)
                    printf("\n  [!] CRASH: node %d server killed by signal %d (%s)",
                           n, signal_num, strsignal(signal_num));
                else
                    printf("\n  [!] CRASH: node %d server exited with code %d",
                           n, exit_status);
                printf("  [N%d: DEAD]", n);
                continue;
            }
            try {
                RadixClient probe(shm_names[n]);
                uint32_t np_used = probe.total_radix_nodes();
                uint32_t np_total = cfg.max_nodes_shm;
                uint32_t mp_used = probe.mempool_used();
                uint32_t mp_total = probe.mempool_total();
                printf("  [N%d: node=%u/%u(%.0f%%) data=%.0f%% mem=%u/%u(%.0f%%)]",
                       n,
                       np_used, np_total,
                       100.0 * np_used / np_total,
                       probe.data_pool_utilization() * 100.0f,
                       mp_used, mp_total,
                       mp_total > 0 ? 100.0 * mp_used / mp_total : 0.0);
            } catch (...) {
                printf("  [N%d: DISCONNECTED]", n);
            }
        }
        printf("\n");
        fflush(stdout);

        prev = cur;
    }
}

// ======================================================================
// CLI
// ======================================================================

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n\n", prog);
    printf("LLM workload options:\n");
    printf("  --nodes N            Number of nodes (default: 1)\n");
    printf("  --clients N          Clients per node (default: 8)\n");
    printf("  --concurrency N      Max concurrent requests (default: nodes*clients)\n");
    printf("  --qps N              Target QPS (default: 100)\n");
    printf("  --isl-min N          Min new input tokens per turn (default: 32)\n");
    printf("  --isl-max N          Max new input tokens per turn (default: 512)\n");
    printf("  --osl-min N          Min output tokens per turn (default: 16)\n");
    printf("  --osl-max N          Max output tokens per turn (default: 256)\n");
    printf("  --sys-prompt N       System prompt length in tokens (default: 256)\n");
    printf("  --max-turns N        Max turns per conversation (default: 10)\n");
    printf("  --num-templates N    Distinct system prompt templates (default: 5)\n");
    printf("\nTree options:\n");
    printf("  --block-size N       Tokens per hash block (default: 16)\n");
    printf("  --max-nodes N        NodePool capacity per tree (default: 200000)\n");
    printf("  --max-blocks N       Mempool blocks per tree (default: 500000)\n");
    printf("  --data-pool-ratio N  DataPool ratio (default: 5)\n");
    printf("\nTiming options:\n");
    printf("  --transfer-ms N      Transfer ms per block (default: 0.01)\n");
    printf("  --prefill-ms N       Prefill ms per block (default: 0.05)\n");
    printf("  --decode-ms N        Decode ms per output token (default: 0.02)\n");
    printf("\nTest control:\n");
    printf("  --duration N         Test duration in seconds, 0=forever (default: 0)\n");
    printf("  --seed N             Random seed (default: 42)\n");
    printf("  --report-interval N  Stats interval in seconds (default: 10)\n");
    printf("  --conv-pool N        Conversation pool size (default: 200)\n");
    printf("  --help               Show this help\n");
}

// ======================================================================
// Main
// ======================================================================

int main(int argc, char** argv) {
    StressConfig cfg;

    for (int i = 1; i < argc; i++) {
        auto arg = [&](const char* name) { return strcmp(argv[i], name) == 0; };
        auto next_int = [&]() { return atoi(argv[++i]); };
        auto next_dbl = [&]() { return atof(argv[++i]); };

        if (arg("--help") || arg("-h")) { print_usage(argv[0]); return 0; }
        else if (arg("--nodes"))            cfg.num_nodes = next_int();
        else if (arg("--clients"))          cfg.clients_per_node = next_int();
        else if (arg("--concurrency"))      cfg.max_concurrency = next_int();
        else if (arg("--qps"))              cfg.target_qps = next_dbl();
        else if (arg("--isl-min"))          cfg.isl_min = next_int();
        else if (arg("--isl-max"))          cfg.isl_max = next_int();
        else if (arg("--osl-min"))          cfg.osl_min = next_int();
        else if (arg("--osl-max"))          cfg.osl_max = next_int();
        else if (arg("--sys-prompt"))       cfg.system_prompt_length = next_int();
        else if (arg("--max-turns"))        cfg.max_turns = next_int();
        else if (arg("--num-templates"))    cfg.num_templates = next_int();
        else if (arg("--block-size"))       cfg.block_size = next_int();
        else if (arg("--max-nodes"))        cfg.max_nodes_shm = (uint32_t)next_int();
        else if (arg("--max-blocks"))       cfg.max_blocks_shm = (uint32_t)next_int();
        // --data-pool-ratio removed
        else if (arg("--transfer-ms"))      cfg.transfer_ms_per_block = next_dbl();
        else if (arg("--prefill-ms"))       cfg.prefill_ms_per_block = next_dbl();
        else if (arg("--decode-ms"))        cfg.decode_ms_per_token = next_dbl();
        else if (arg("--duration"))         cfg.duration_seconds = next_int();
        else if (arg("--seed"))             cfg.seed = (uint64_t)next_int();
        else if (arg("--report-interval"))  cfg.report_interval_s = next_int();
        else if (arg("--conv-pool"))        cfg.conv_pool_size = next_int();
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); return 1; }
    }

    if (cfg.num_nodes < 1 || cfg.num_nodes > MAX_NODES) {
        fprintf(stderr, "num_nodes must be in [1, %d]\n", MAX_NODES);
        return 1;
    }
    if (cfg.num_templates > MAX_TEMPLATES) {
        fprintf(stderr, "num_templates must be <= %d\n", MAX_TEMPLATES);
        return 1;
    }
    if (cfg.max_concurrency <= 0)
        cfg.max_concurrency = cfg.num_nodes * cfg.clients_per_node;

    // Round system_prompt_length to block_size multiple
    cfg.system_prompt_length = (cfg.system_prompt_length / cfg.block_size)
                                * cfg.block_size;
    if (cfg.system_prompt_length < cfg.block_size)
        cfg.system_prompt_length = cfg.block_size;

    int total_clients = cfg.num_nodes * cfg.clients_per_node;

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // ---- Print configuration ----
    printf("=== LLM Workload Stress Test ===\n\n");
    printf("  Nodes:            %d\n", cfg.num_nodes);
    printf("  Clients/node:     %d  (total: %d)\n",
           cfg.clients_per_node, total_clients);
    printf("  Max concurrency:  %d\n", cfg.max_concurrency);
    printf("  Target QPS:       %.1f\n", cfg.target_qps);
    printf("  ISL range:        %d-%d tokens (%d-%d blocks)\n",
           cfg.isl_min, cfg.isl_max,
           cfg.isl_min / cfg.block_size, cfg.isl_max / cfg.block_size);
    printf("  OSL range:        %d-%d tokens (%d-%d blocks)\n",
           cfg.osl_min, cfg.osl_max,
           cfg.osl_min / cfg.block_size, cfg.osl_max / cfg.block_size);
    printf("  System prompt:    %d tokens (%d blocks)\n",
           cfg.system_prompt_length, cfg.system_prompt_length / cfg.block_size);
    printf("  Max turns:        %d\n", cfg.max_turns);
    printf("  Templates:        %d\n", cfg.num_templates);
    printf("  Block size:       %d tokens\n", cfg.block_size);
    printf("  Conv pool:        %d\n", cfg.conv_pool_size);
    printf("  Duration:         %s\n",
           cfg.duration_seconds > 0
               ? (std::to_string(cfg.duration_seconds) + "s").c_str()
               : "infinite (Ctrl+C to stop)");
    printf("  Seed:             %llu\n", (unsigned long long)cfg.seed);
    printf("\n");

    // ---- Initialize template tracking ----
    for (int n = 0; n < MAX_NODES; n++)
        for (int t = 0; t < MAX_TEMPLATES; t++)
            g_template_on_node[n][t].store(false, std::memory_order_relaxed);

    // ---- Create server processes (one per node) ----
    printf("Starting %d tree server(s)...\n", cfg.num_nodes);
    fflush(stdout);

    shmradix::ShmConfig shm_cfg;
    shm_cfg.max_nodes = cfg.max_nodes_shm;
    shm_cfg.max_blocks = cfg.max_blocks_shm;

    std::vector<std::string> shm_names;
    std::vector<std::unique_ptr<ServerLauncher>> servers;
    for (int n = 0; n < cfg.num_nodes; n++) {
        std::string name = "/stress_node_" + std::to_string(n);
        shm_names.push_back(name);
        servers.push_back(std::make_unique<ServerLauncher>(name, shm_cfg));
    }
    printf("All servers ready.\n\n");

    // ---- Generate system prompt templates ----
    std::mt19937_64 rng(cfg.seed);
    std::vector<std::vector<int64_t>> templates(cfg.num_templates);
    for (int t = 0; t < cfg.num_templates; t++) {
        templates[t].resize(cfg.system_prompt_length);
        for (int i = 0; i < cfg.system_prompt_length; i++)
            templates[t][i] = (int64_t)(rng() & 0x7FFFFFFFLL);
    }

    // ---- Create conversation pool ----
    std::vector<ConvState> convs(cfg.conv_pool_size);
    std::mutex conv_mtx;
    int next_conv_id = 0;
    for (int i = 0; i < cfg.conv_pool_size; i++)
        generate_conversation(convs[i], next_conv_id++, templates, cfg, rng);

    // ---- Create per-node queues ----
    std::vector<NodeQueue> node_queues(cfg.num_nodes);

    // ---- Create concurrency limiter ----
    ConcurrencyLimiter limiter(cfg.max_concurrency);

    // ---- Create client metrics ----
    std::vector<ClientMetrics> metrics(total_clients);
    std::vector<ClientMetrics*> all_metrics;
    for (int i = 0; i < total_clients; i++)
        all_metrics.push_back(&metrics[i]);

    // ---- Start client threads ----
    printf("Starting %d client threads...\n", total_clients);
    std::vector<std::thread> client_threads;
    for (int i = 0; i < total_clients; i++) {
        int node_id = i / cfg.clients_per_node;
        client_threads.emplace_back(client_thread_fn,
            i, node_id, std::cref(shm_names[node_id]),
            std::ref(node_queues[node_id]),
            std::ref(limiter),
            std::ref(metrics[i]),
            std::ref(convs),
            std::ref(conv_mtx),
            std::cref(cfg));
    }

    // ---- Start reporter thread ----
    std::vector<ServerLauncher*> server_ptrs;
    for (auto& s : servers) server_ptrs.push_back(s.get());
    std::thread reporter(reporter_thread_fn,
                          std::cref(all_metrics),
                          std::cref(server_ptrs),
                          std::cref(shm_names),
                          std::cref(cfg));

    // ---- Generator loop (main thread) ----
    printf("Generating requests at %.1f QPS...\n\n", cfg.target_qps);

    auto test_start = Clock::now();
    double request_interval_us = 1e6 / cfg.target_qps;
    auto next_dispatch = Clock::now();
    std::uniform_int_distribution<int> node_dist(0, cfg.num_nodes - 1);
    int64_t total_dispatched = 0;
    int64_t convs_completed = 0;

    while (g_running) {
        // Check duration limit
        if (cfg.duration_seconds > 0) {
            auto now = Clock::now();
            double elapsed = std::chrono::duration<double>(now - test_start).count();
            if (elapsed >= cfg.duration_seconds) break;
        }

        // Rate limiting: wait until next dispatch time
        auto now = Clock::now();
        if (now < next_dispatch) {
            auto wait = std::chrono::duration_cast<std::chrono::microseconds>(
                next_dispatch - now);
            if (wait.count() > 0)
                std::this_thread::sleep_for(wait);
        }
        next_dispatch += std::chrono::microseconds(
            (int64_t)request_interval_us);

        // Prevent clock drift: don't schedule too far in the past
        now = Clock::now();
        if (next_dispatch < now)
            next_dispatch = now;

        // Find a ready conversation
        int pool_idx = -1;
        {
            std::lock_guard<std::mutex> lock(conv_mtx);

            // First pass: find a conv ready for its next turn
            for (int i = 0; i < cfg.conv_pool_size; i++) {
                if (!convs[i].in_flight && !convs[i].complete) {
                    pool_idx = i;
                    break;
                }
            }

            // Second pass: replace a completed conversation
            if (pool_idx < 0) {
                for (int i = 0; i < cfg.conv_pool_size; i++) {
                    if (convs[i].complete) {
                        convs_completed++;
                        generate_conversation(convs[i], next_conv_id++,
                                              templates, cfg, rng);
                        pool_idx = i;
                        break;
                    }
                }
            }

            if (pool_idx >= 0) {
                convs[pool_idx].in_flight = true;
            }
        }

        if (pool_idx < 0) continue;  // all convs in flight, skip this tick

        // Build request
        ConvState& conv = convs[pool_idx];
        int turn = conv.current_turn;
        int node_id = node_dist(rng);

        Request req;
        req.conv_id = conv.id;
        req.conv_pool_idx = pool_idx;
        req.turn = turn;
        req.assigned_node = node_id;
        req.template_id = conv.template_id;
        req.input_tokens = build_input_tokens(conv, turn);
        req.new_token_count = (int)conv.turns[turn].user_input.size();
        req.osl = (int)conv.turns[turn].model_output.size();
        req.output_tokens = conv.turns[turn].model_output;

        // Compute theoretical hit
        int total_input_hashes = (int)req.input_tokens.size() / cfg.block_size;
        int sys_hashes = cfg.system_prompt_length / cfg.block_size;
        int from_conv = conv.hashes_on_node[node_id];
        if (from_conv > 0) {
            req.expected_hit_hashes = std::min(from_conv, total_input_hashes);
        } else if (g_template_on_node[node_id][conv.template_id]
                       .load(std::memory_order_relaxed)) {
            req.expected_hit_hashes = std::min(sys_hashes, total_input_hashes);
        } else {
            req.expected_hit_hashes = 0;
        }

        // Acquire concurrency slot (blocks if at limit)
        limiter.acquire();
        if (!g_running) {
            // Mark conv as not in flight so shutdown is clean
            std::lock_guard<std::mutex> lock(conv_mtx);
            convs[pool_idx].in_flight = false;
            limiter.release();
            break;
        }

        // Dispatch to node queue
        node_queues[node_id].push(std::move(req));
        total_dispatched++;
    }

    // ---- Shutdown ----
    printf("\nShutting down...\n");
    g_running = 0;

    // Wake up all waiting threads
    for (auto& q : node_queues)
        q.signal_shutdown();
    limiter.release();  // unblock any waiting acquire

    for (auto& t : client_threads)
        t.join();
    reporter.join();

    // ---- Final summary ----
    auto test_end = Clock::now();
    double total_time = std::chrono::duration<double>(test_end - test_start).count();

    Snapshot final_snap;
    for (auto* m : all_metrics) {
        final_snap.queries      += m->queries;
        final_snap.actual_hit   += m->actual_hit;
        final_snap.expected_hit += m->expected_hit;
        final_snap.total_blocks += m->total_blocks;
        final_snap.query_us     += m->query_time_us;
        if (m->max_query_us > final_snap.max_query_us)
            final_snap.max_query_us = m->max_query_us;
        final_snap.inserts += m->inserts;
        final_snap.errors  += m->insert_errors;
        final_snap.done    += m->requests_done;
    }

    printf("\n=== Final Summary ===\n\n");
    printf("  Duration:           %.1f s\n", total_time);
    printf("  Total dispatched:   %lld\n", (long long)total_dispatched);
    printf("  Total completed:    %lld\n", (long long)final_snap.done);
    printf("  Conversations done: %lld\n", (long long)convs_completed);
    printf("  Effective QPS:      %.1f\n",
           final_snap.done / total_time);
    printf("\n");

    double overall_hit = final_snap.total_blocks > 0
        ? 100.0 * final_snap.actual_hit / final_snap.total_blocks : 0;
    double theo_hit = final_snap.total_blocks > 0
        ? 100.0 * final_snap.expected_hit / final_snap.total_blocks : 0;
    double ratio = final_snap.expected_hit > 0
        ? (double)final_snap.actual_hit / final_snap.expected_hit : 0;

    printf("  Actual hit rate:    %.2f%%\n", overall_hit);
    printf("  Theoretical hit:    %.2f%%\n", theo_hit);
    printf("  Hit ratio:          %.3f\n", ratio);
    printf("\n");

    double avg_us = final_snap.queries > 0
        ? final_snap.query_us / final_snap.queries : 0;
    printf("  Query latency (cumulative):\n");
    printf("    avg:   %10.1f us\n", avg_us);
    printf("    max:   %10.1f us\n", final_snap.max_query_us);
    printf("\n");

    printf("  Insert errors:      %lld\n", (long long)final_snap.errors);

    // Per-node final stats
    printf("\n  Per-node tree stats:\n");
    for (int n = 0; n < cfg.num_nodes; n++) {
        int exit_status = 0, signal_num = 0;
        bool alive = servers[n]->check_alive(exit_status, signal_num);
        if (!alive) {
            if (signal_num > 0)
                printf("    Node %d: CRASHED (signal %d: %s)\n",
                       n, signal_num, strsignal(signal_num));
            else
                printf("    Node %d: EXITED (code %d)\n", n, exit_status);
            continue;
        }
        try {
            RadixClient probe(shm_names[n]);
            uint32_t np_used = probe.total_radix_nodes();
            uint32_t np_total = cfg.max_nodes_shm;
            uint32_t mp_used = probe.mempool_used();
            uint32_t mp_total = probe.mempool_total();
            printf("    Node %d: nodepool=%u/%u(%.1f%%)  data_pool=%.1f%%  mempool=%u/%u(%.1f%%)\n",
                   n,
                   np_used, np_total,
                   100.0 * np_used / np_total,
                   probe.data_pool_utilization() * 100.0f,
                   mp_used, mp_total,
                   mp_total > 0 ? 100.0 * mp_used / mp_total : 0.0);
        } catch (...) {
            printf("    Node %d: DISCONNECTED\n", n);
        }
    }

    printf("\n%s\n",
           final_snap.errors == 0 ? "PASSED (no errors detected)"
                                  : "FAILED (errors detected)");
    return final_snap.errors > 0 ? 1 : 0;
}
