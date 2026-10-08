// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// etcd-backed cluster view for a client: reads /config and /peers, resolves this
// node's rank, and pushes late joiners via a background /peers watcher. Holds no
// RDMA or shm state — topology only.

#pragma once

#include "radixshmem_config.h"

#ifdef RADIXSHMEM_HAS_ETCD

#include "cluster_manifest.h"
#include "cluster_protocol.h"

#include "etcd_client.hpp"

#include <atomic>
#include <functional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace radixshmem {

struct RegistryClientConfig {
    std::string registry;
    std::string cluster_id;
    std::string node_name;
    int         expected_min_nodes = 0;
    int         bootstrap_timeout_sec = 60;
};

// Decoded cluster config from /config, plus the resolved peer set / identity.
struct ClusterTopology {
    ShardMapMsg            sm{};
    std::vector<PeerEntry> peers;       // dense, rank-indexed
    int                    world_size = 0;
    int                    my_rank = -1;  // -1 when node_name is not a /peers member
    int64_t                peers_rev = 0; // /peers snapshot revision (anchors watch)
};

class RegistryClient {
public:
    // Called with every published peer entry (a new node, or a node that
    // restarted) and with the rank of every node whose entry went away. Both
    // run on the watcher thread.
    using JoinFn  = std::function<void(const PeerEntry&)>;
    using LeaveFn = std::function<void(int rank)>;

    explicit RegistryClient(const RegistryClientConfig& cfg) : cfg_(cfg) {
        keys_.cluster_id = cfg_.cluster_id;
    }
    ~RegistryClient() { stop_watch(); }

    RegistryClient(const RegistryClient&) = delete;
    RegistryClient& operator=(const RegistryClient&) = delete;

    void connect() { cli_.init(etcd::strip_scheme(cfg_.registry)); }

    // Block until the cluster is formed (/config published) and every
    // registered node is ready; false on timeout. A node that is down has
    // neither key and is not waited for.
    bool wait_for_ready() {
        std::string why;
        bool ok = etcd::poll_until(cfg_.bootstrap_timeout_sec, [&] {
            std::string cfg_val;
            if (!cli_.get(keys_.config(), cfg_val)) {
                why = "no /config";
                return false;
            }
            int64_t peers = cli_.count(keys_.peers_prefix());
            int64_t ready = cli_.count(keys_.ready_prefix());
            why = std::to_string(ready) + "/" + std::to_string(peers) +
                  " registered nodes ready";
            return peers > 0 && ready >= peers;
        });
        if (!ok)
            fprintf(stderr, "[RegistryClient] timed out waiting for ready: %s\n",
                    why.c_str());
        return ok;
    }

    // Read /config + /peers into a topology snapshot. my_rank is -1 when this
    // node has no /peers entry (no local server). World size is the rank-slot
    // count; the rank of a node that is down keeps a peer entry with rank -1.
    ClusterTopology read_topology() {
        ClusterTopology t;

        std::string cfg_val;
        if (!cli_.get(keys_.config(), cfg_val))
            throw std::runtime_error("RegistryClient: /config absent");
        if (!etcd::decode_shardmap(cfg_val, t.sm))
            throw std::runtime_error("RegistryClient: bad /config blob");

        int ws = (int)cli_.count(keys_.rank_slot_prefix());
        auto kvs = cli_.get_prefix(keys_.peers_prefix(), &t.peers_rev);
        if (kvs.empty())
            throw std::runtime_error("RegistryClient: no peers registered");
        for (auto& kv : kvs) {   // a joiner may have claimed a slot in between
            PeerEntry e{};
            if (etcd::decode_peer(kv.value, e)) ws = std::max(ws, e.rank + 1);
        }
        t.world_size = ws;
        t.peers.assign(ws, PeerEntry{});
        for (auto& p : t.peers) p.rank = -1;
        for (auto& kv : kvs) {
            PeerEntry e{};
            if (!etcd::decode_peer(kv.value, e) || e.rank < 0) continue;
            t.peers[e.rank] = e;
            std::string node = leaf(kv.key);
            rank_of_node_[node] = e.rank;
            if (node == cfg_.node_name) t.my_rank = e.rank;
        }
        return t;
    }

    // Resolve /shm/<node> to a local shm name; polls until published.
    bool resolve_shm(const std::string& node, std::string& out) {
        return etcd::poll_until(cfg_.bootstrap_timeout_sec,
                                [&] { return cli_.get(keys_.shm(node), out); });
    }

    // Anchor a /peers watcher at start_rev (after read_topology).
    void start_watch(int64_t start_rev, JoinFn on_join, LeaveFn on_leave) {
        on_join_ = std::move(on_join);
        on_leave_ = std::move(on_leave);
        watch_.start(keys_.peers_prefix(), start_rev);
        stop_.store(false, std::memory_order_release);
        thread_ = std::thread(&RegistryClient::watch_loop, this);
    }

    void stop_watch() {
        stop_.store(true, std::memory_order_release);
        if (thread_.joinable()) thread_.join();
        watch_.stop();
    }

    etcd::Client& client() { return cli_; }
    etcd::Keys&   keys()   { return keys_; }

private:
    static std::string leaf(const std::string& key) {
        auto pos = key.rfind('/');
        return pos == std::string::npos ? key : key.substr(pos + 1);
    }

    void watch_loop() {
        while (!stop_.load(std::memory_order_acquire)) {
            etcd::WatchEvent ev;
            if (!watch_.next(500, ev)) continue;
            if (stop_.load(std::memory_order_acquire)) break;
            if (ev.type == etcd::WatchEvent::kDelete) {
                auto it = rank_of_node_.find(leaf(ev.key));
                if (it == rank_of_node_.end()) continue;
                int rank = it->second;
                rank_of_node_.erase(it);
                if (on_leave_) on_leave_(rank);
                continue;
            }
            PeerEntry e{};
            if (!etcd::decode_peer(ev.value, e)) continue;
            if (e.rank < 0) continue;   // round 1: rank not stamped yet
            rank_of_node_[leaf(ev.key)] = e.rank;
            if (on_join_) on_join_(e);
        }
    }

    RegistryClientConfig cfg_;
    etcd::Client         cli_;
    etcd::Keys           keys_;
    etcd::Watch          watch_;
    std::thread          thread_;
    std::atomic<bool>    stop_{false};
    JoinFn               on_join_;
    LeaveFn              on_leave_;
    std::unordered_map<std::string, int> rank_of_node_;   // watcher thread after start_watch
};

} // namespace radixshmem

#endif // RADIXSHMEM_HAS_ETCD
