// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Unified launcher for the RadixServer.
//
// Single-node:
//     radix_server --name /my_kv_cache --max-blocks 100000
//
// Distributed (run on each node; rank is assigned by the registry):
//     radix_server --num-rht-shards M --registry etcd://10.0.0.1:2379 \
//                  --max-blocks 100000

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <exception>
#include <thread>

#include "shmradix/radix_server.h"
#include "shmradix/radix_tree.h"

static volatile sig_atomic_t g_running = 1;

static void signal_handler(int) { g_running = 0; }

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("\nTree options:\n");
    printf("  --name <string>          Shared memory name (default: /shmradix)\n");
    printf("  --max-blocks <int>       Mempool capacity in blocks (default: 0 = disabled)\n");
    printf("  --block-size <int>       Tokens per block (default: 1)\n");
    printf("  --max-nodes <int>        NodePool capacity (default: auto = max_blocks*min(block_size/128,1))\n");
    printf("  --register-chunk-tokens <int> RHT registration granularity in tokens, converted to blocks with --block-size (default: 4096)\n");
    printf("  --background-evict-ratio <float> Background evictor free-fraction target (default: 0.05, 0=disabled)\n");
    printf("  --data-pool-ratio <float>  DataPool size = max_blocks * ratio * block_bytes (default: 4.0)\n");
    printf("  --evict-policy <str>     Eviction policy: lru (default: lru)\n");
    printf("  --interval <int>         Stats print interval in seconds (default: 5)\n");
    printf("\nDistributed options (num-rht-shards > 1 enables distributed mode):\n");
    printf("  --rpc-address <ip>       Bootstrap IP peers connect to (required; 0.0.0.0 for single node)\n");
    printf("  --rpc-interface <nic>    NIC name to resolve the bootstrap IP from; wins over --rpc-address\n");
    printf("  --rpc-port <int>         Bootstrap TCP port (default: 0 = OS-assigned)\n");
    printf("  --settle-ms <int>        Membership stability window in ms (default: 500)\n");
    printf("  --registry <string>      Cluster bootstrap registry etcd://host:port (default: etcd://127.0.0.1:2379)\n");
    printf("  --cluster-id <string>    etcd key namespace radix/<cluster-id>/... (default: default)\n");
    printf("  --node-name <string>     etcd node identity; empty derives from the resolved IP (default: empty)\n");
    printf("  --index-dev <string>      RDMA device name (default: first available)\n");
    printf("  --gid-idx <int>          RDMA GID index (default: 3)\n");
    printf("  --bootstrap-timeout <int> Bootstrap timeout in seconds (default: 60)\n");
    printf("  --lease-ttl <int>         etcd lease TTL in seconds (default: 30)\n");
    printf("  --rht-transport <str>    Client -> RHT holder transport: xrc (default) | dc\n");
    printf("  --peer-index-transport <str>  Remote-walk transport into peer trees: xrc (default) | dc\n");
    printf("  --num-rht-shards <int>   RHT shard count M (default: 0 = M==world-size)\n");
    printf("  --rht-shard-holders <list> Comma-separated holder ranks, e.g. 0,2,4,6\n");
    printf("                           (authoritative only on rank 0; default: ranks [0,M))\n");
    printf("  --rht-slots <int>        RHT slots per bucket W (1|2|4|8, default 1)\n");
    printf("\n  --help                   Show this help\n");
}

int main(int argc, char** argv) {
    shmradix::RadixServerConfig cfg;
    const char* evict_policy_str = "lru";
    int interval = 5;

    for (int i = 1; i < argc; i++) {
        auto arg = [&](const char* flag) { return strcmp(argv[i], flag) == 0; };
        auto next_str = [&]() -> const char* { return argv[++i]; };
        auto next_int = [&]() { return atoi(argv[++i]); };
        auto next_flt = [&]() { return (float)atof(argv[++i]); };

        if (arg("--help") || arg("-h")) { print_usage(argv[0]); return 0; }
        else if (arg("--name"))              cfg.name = next_str();
        else if (arg("--max-blocks"))        cfg.shm.max_blocks = (uint32_t)next_int();
        else if (arg("--block-size"))        cfg.shm.block_size = next_int();
        else if (arg("--max-nodes"))         cfg.shm.max_nodes = (uint32_t)next_int();
        else if (arg("--register-chunk-tokens")) cfg.register_chunk_tokens = next_int();
        else if (arg("--background-evict-ratio")) cfg.shm.background_evict_ratio = next_flt();
        else if (arg("--data-pool-ratio"))   cfg.shm.data_pool_ratio = next_flt();
        else if (arg("--evict-policy"))      evict_policy_str = next_str();
        else if (arg("--interval"))          interval = next_int();
        else if (arg("--settle-ms"))         cfg.settle_ms = next_int();
        else if (arg("--registry"))          cfg.registry = next_str();
        else if (arg("--cluster-id"))        cfg.cluster_id = next_str();
        else if (arg("--node-name"))         cfg.node_name = next_str();
        else if (arg("--rpc-address"))       cfg.rpc_address = next_str();
        else if (arg("--rpc-interface"))     cfg.rpc_interface = next_str();
        else if (arg("--rpc-port"))          cfg.rpc_port = (uint16_t)next_int();
        else if (arg("--index-dev"))          cfg.index_dev = next_str();
        else if (arg("--gid-idx"))           cfg.gid_idx = next_int();
        else if (arg("--bootstrap-timeout")) cfg.bootstrap_timeout_sec = next_int();
        else if (arg("--lease-ttl"))         cfg.lease_ttl_sec = next_int();
        else if (arg("--rht-transport"))     cfg.rht_transport = next_str();
        else if (arg("--peer-index-transport")) cfg.peer_index_transport = next_str();
        else if (arg("--rht-slots"))         cfg.rht_slots_per_bucket = next_int();
        else if (arg("--num-rht-shards"))        cfg.num_rht_shards = next_int();
        else if (arg("--rht-shard-holders")) {
            const char* s = next_str();
            cfg.rht_shard_holders.clear();
            // Parse comma-separated rank list, e.g. "0,2,4,6".
            const char* p = s;
            while (*p) {
                char* end = nullptr;
                long v = strtol(p, &end, 10);
                if (end == p) break;
                cfg.rht_shard_holders.push_back((int)v);
                p = end;
                while (*p == ',' || *p == ' ') p++;
            }
        }
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (strcmp(evict_policy_str, "lru") == 0)
        cfg.shm.evict_policy = shmradix::EvictPolicy::LRU;
    else {
        fprintf(stderr, "Unknown evict policy: %s (supported: lru)\n",
                evict_policy_str);
        return 1;
    }
    cfg.shm.register_chunk_tokens = cfg.register_chunk_tokens;
    uint32_t resolved_max_nodes = cfg.shm.effective_max_nodes();

    auto fmt_size = [](uint64_t bytes) -> std::pair<double, const char*> {
        if (bytes >= 1ULL << 40) return {bytes / (double)(1ULL << 40), "TB"};
        if (bytes >= 1ULL << 30) return {bytes / (double)(1ULL << 30), "GB"};
        return {bytes / (double)(1ULL << 20), "MB"};
    };

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    bool distributed = cfg.num_rht_shards > 1;
    if (distributed) {
        printf("Starting radix_server (distributed, node %s)...\n",
               cfg.node_name.empty() ? "(auto node<ip>)" : cfg.node_name.c_str());
        printf("  Registry:       %s\n", cfg.registry.c_str());
        printf("  Num shards:     %d\n", cfg.num_rht_shards);
        printf("  Cluster ID:     %s\n", cfg.cluster_id.c_str());
        printf("  Max nodes:      %u\n", resolved_max_nodes);
        printf("  Max blocks:     %u%s\n", cfg.shm.max_blocks,
               cfg.shm.max_blocks ? "" : " (mempool disabled)");
        printf("  RDMA device:    %s\n",
               cfg.index_dev.empty() ? "(auto)" : cfg.index_dev.c_str());
        fflush(stdout);
    } else {
        printf("Starting radix_server (standalone)...\n");
        printf("  Name:           %s\n", cfg.name.c_str());
        printf("  Max blocks:     %u%s\n", cfg.shm.max_blocks,
               cfg.shm.max_blocks ? "" : " (mempool disabled)");
        printf("  Block size:     %d tokens\n", cfg.shm.block_size);
        printf("  Max nodes:      %u%s\n", resolved_max_nodes,
               cfg.shm.max_nodes == 0 ? " (auto)" : "");
        printf("  Register chunk: %d tokens (%d blocks)\n", cfg.register_chunk_tokens,
               cfg.shm.effective_register_chunk_size());
        auto layout = shmradix::ShmRadixTree::compute_layout(cfg.shm);
        auto [dp_v, dp_u] = fmt_size(layout.data_pool_size);
        printf("  Data pool:      %.2f %s\n", dp_v, dp_u);
        printf("  BG evict ratio: %.2f%s\n", cfg.shm.background_evict_ratio,
               cfg.shm.background_evict_ratio <= 0 ? " (disabled)" : "");
        printf("  Evict policy:   %s\n", evict_policy_str);
        printf("  Stats interval: %d s\n\n", interval);
    }

    shmradix::RadixServer srv(cfg);
    try {
        if (!srv.bootstrap()) {
            fprintf(stderr, "RadixServer bootstrap failed\n");
            return 1;
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "RadixServer bootstrap failed: %s\n", e.what());
        return 1;
    }

    if (distributed) {
        printf("Cluster ready. rank=%d/%d  shm=%s\n",
               srv.rank(), srv.world_size(), srv.shm_name().c_str());
    } else {
        printf("Server ready. PID=%d  shm=%s\n",
               (int)getpid(), srv.shm_name().c_str());
    }
    printf("Press Ctrl+C to shutdown.\n\n");
    fflush(stdout);

    while (g_running) {
        if (interval <= 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        for (int i = 0; i < interval * 10 && g_running; i++)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (!g_running) break;

        if (distributed) {
            printf("\r[rank %d] nodes=%u  blocks=%llu  data_pool=%.1f%%",
                   srv.rank(), srv.total_radix_nodes(),
                   (unsigned long long)srv.total_blocks(),
                   srv.data_pool_utilization() * 100.0f);
            if (cfg.shm.max_blocks > 0)
                printf("  mempool=%u/%u",
                       srv.mempool_used(), srv.mempool_total());
            printf("  ins_ok=%llu",
                   (unsigned long long)srv.insert_success_count());
        } else {
            uint64_t err_np = srv.insert_error_node_pool_count();
            uint64_t err_dp = srv.insert_error_data_pool_count();
            uint64_t err_ht = srv.insert_error_ht_pool_count();
            uint64_t err_total = err_np + err_dp + err_ht;
            printf("\r[stats] nodes=%u  blocks=%llu  data_pool=%.1f%%",
                   srv.total_radix_nodes(),
                   (unsigned long long)srv.total_blocks(),
                   srv.data_pool_utilization() * 100.0f);
            if (cfg.shm.max_blocks > 0)
                printf("  mempool=%u/%u",
                       srv.mempool_used(), srv.mempool_total());
            printf("  ins_ok=%llu",
                   (unsigned long long)srv.insert_success_count());
            if (err_total > 0)
                printf("  ins_err=%llu(np=%llu,dp=%llu,ht=%llu)",
                       (unsigned long long)err_total,
                       (unsigned long long)err_np,
                       (unsigned long long)err_dp,
                       (unsigned long long)err_ht);
        }
        printf("        ");
        fflush(stdout);
    }

    printf("\nShutting down %s...\n",
           distributed ? "cluster node" : "server");
    return 0;
}
