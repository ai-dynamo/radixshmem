// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "etcd_registry.hpp"

#ifdef RADIXSHMEM_HAS_ETCD

#include "radix_server.h"
#include "cluster_manifest.h"
#include "cluster_protocol.h"

#include "etcd_client.hpp"
#include "xrc_connect.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

namespace radixshmem {

namespace {

// Node name of the last path segment of an etcd key ("radix/x/peers/nodeA" → "nodeA").
std::string leaf(const std::string& key) {
    auto pos = key.rfind('/');
    return pos == std::string::npos ? key : key.substr(pos + 1);
}

} // namespace

// ==================== Impl ====================

struct EtcdRegistry::Impl {
    etcd::Client cli;
    etcd::Keys   keys;
};

EtcdRegistry::EtcdRegistry(RadixServer& rs, std::string endpoint,
                           std::string cluster_id, std::string node_name,
                           std::string shm_name)
    : rs_(rs), endpoint_(std::move(endpoint)),
      cluster_id_(std::move(cluster_id)), node_name_(std::move(node_name)),
      shm_name_(std::move(shm_name)),
      impl_(std::make_unique<Impl>()) {
    impl_->keys.cluster_id = cluster_id_;
}

EtcdRegistry::~EtcdRegistry() {
    if (cluster_lease_ > 0 && config_published_) {
        impl_->cli.keep_alive_stop(cluster_lease_);
    } else if (cluster_lease_ > 0) {   // leader failed before /config: drop its rank slots
        try { impl_->cli.revoke(cluster_lease_); } catch (const std::exception&) {}
    }
    if (lease_id_ <= 0) return;
    try { impl_->cli.revoke(lease_id_); } catch (const std::exception&) {}
}

int EtcdRegistry::await_membership(int gate) {
    const int T = rs_.cfg_.bootstrap_timeout_sec;
    const auto settle = std::chrono::milliseconds(rs_.cfg_.settle_ms);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(T);

    int64_t last = -1;
    auto stable_since = std::chrono::steady_clock::now();
    for (;;) {
        int64_t c = impl_->cli.count(impl_->keys.peers_prefix());
        auto now = std::chrono::steady_clock::now();
        if (c != last) { last = c; stable_since = now; }
        if (c >= gate && (now - stable_since) >= settle) return (int)c;
        if (now >= deadline) return -1;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

int EtcdRegistry::read_my_rank() {
    for (auto& kv : impl_->cli.get_prefix(impl_->keys.rank_slot_prefix())) {
        if (kv.value == node_name_) return std::atoi(leaf(kv.key).c_str());
    }
    return -1;
}

int EtcdRegistry::world_size() {
    return (int)impl_->cli.count(impl_->keys.rank_slot_prefix());
}

// A run killed before revoking its lease leaves its keys until the TTL. Revoke
// that lease so this run starts as after a clean exit, unless the entry still
// belongs to a live server.
bool EtcdRegistry::reclaim_previous_run() {
    auto& K = impl_->keys;
    std::string val;
    int64_t lease = 0;
    if (!impl_->cli.get_with_lease(K.peer(node_name_), val, lease)) return true;

    PeerEntry old{}, mine{};
    rs_.fill_local_entry(mine);
    if (etcd::decode_peer(val, old) && old.xrc_listen_port != 0 &&
        !(old.xrc_listen_port == mine.xrc_listen_port &&
          std::strcmp(old.host, mine.host) == 0)) {
        int fd = xrc_tcp_connect(old.host, old.xrc_listen_port, 1);
        if (fd >= 0) {
            ::close(fd);
            fprintf(stderr, "[EtcdRegistry %s] node name is held by a live "
                    "radix-server at %s:%u\n", node_name_.c_str(), old.host,
                    old.xrc_listen_port);
            return false;
        }
    }
    if (lease > 0) {
        try { impl_->cli.revoke(lease); } catch (const std::exception&) {}   // expired meanwhile
    } else {
        impl_->cli.del(K.peer(node_name_));
        impl_->cli.del(K.shm(node_name_));
        impl_->cli.del(K.ready(node_name_));
    }
    fprintf(stderr, "[EtcdRegistry %s] removed the keys of a previous run "
            "(lease %llx)\n", node_name_.c_str(), (unsigned long long)lease);
    return true;
}

bool EtcdRegistry::attach_cluster_lease() {
    auto& K = impl_->keys;
    std::string v;
    if (!impl_->cli.get(K.cluster_lease(), v)) {
        fprintf(stderr, "[EtcdRegistry %s] /cluster_lease absent\n",
                node_name_.c_str());
        return false;
    }
    cluster_lease_ = std::strtoll(v.c_str(), nullptr, 10);
    config_published_ = true;
    impl_->cli.keep_alive(cluster_lease_);
    if (!impl_->cli.get(K.config(), v)) {   // the lease expired before we renewed it
        fprintf(stderr, "[EtcdRegistry %s] /config vanished\n", node_name_.c_str());
        return false;
    }
    return true;
}

bool EtcdRegistry::adopt_config(const std::string& cfg_val) {
    ShardMapMsg sm{};
    if (!etcd::decode_shardmap(cfg_val, sm)) {
        fprintf(stderr, "[EtcdRegistry %s] bad shard-map blob: corrupt, or "
                "written by a build of another cluster protocol version "
                "(this build: v%u)\n",
                node_name_.c_str(), CLUSTER_PROTOCOL_VERSION);
        return false;
    }
    if (!attach_cluster_lease()) return false;
    if (config_preexisting_) {
        std::string diff = rs_.shard_map_mismatch(sm);
        if (!diff.empty()) {
            fprintf(stderr, "[EtcdRegistry %s] /config of cluster %s disagrees "
                    "with this node: %s; start with the cluster's settings, or "
                    "use another --cluster-id\n", node_name_.c_str(),
                    cluster_id_.c_str(), diff.c_str());
            return false;
        }
    }
    rs_.adopt_shard_map(sm);
    return true;
}

bool EtcdRegistry::run_rendezvous(std::vector<PeerEntry>& out) {
    ttl_ = std::max(rs_.cfg_.lease_ttl_sec, 5);
    try {
        impl_->cli.init(endpoint_);
        lease_id_ = impl_->cli.grant_lease(ttl_);
        impl_->cli.keep_alive(lease_id_);
        if (!reclaim_previous_run()) return false;

        // /config lives on the cluster lease: present means the cluster is
        // formed, and this node rejoins with its rank slot or joins as a data peer.
        std::string cfg_probe;
        config_preexisting_ = impl_->cli.get(impl_->keys.config(), cfg_probe);
        bool ok;
        if (config_preexisting_ && read_my_rank() < 0) {
            fprintf(stderr, "[EtcdRegistry %s] role=joiner (data peer)\n",
                    node_name_.c_str());
            return join_existing(out);
        } else if (config_preexisting_) {
            fprintf(stderr, "[EtcdRegistry %s] role=rejoin\n", node_name_.c_str());
            ok = follower_flow(out, false);
        } else {
            bool is_leader =
                impl_->cli.create(impl_->keys.leader(), node_name_, lease_id_);
            fprintf(stderr, "[EtcdRegistry %s] role=%s\n",
                    node_name_.c_str(), is_leader ? "leader" : "follower");
            ok = is_leader ? leader_flow(out) : follower_flow(out, false);
        }
        if (!ok) return false;

        // etcd barrier: mark ready, wait for every registered node.
        int world = world_size();
        rs_.cfg_.world_size = world;
        rs_.stamp_shm_ready(rs_.cfg_.rank, world);
        impl_->cli.put(impl_->keys.ready(node_name_), "1", lease_id_);
        if (!etcd::poll_until(rs_.cfg_.bootstrap_timeout_sec, [&] {
                return impl_->cli.count(impl_->keys.ready_prefix()) >=
                       impl_->cli.count(impl_->keys.peers_prefix());
            })) {
            fprintf(stderr, "[EtcdRegistry %s] barrier timeout (%ld/%ld ready)\n",
                    node_name_.c_str(),
                    (long)impl_->cli.count(impl_->keys.ready_prefix()),
                    (long)impl_->cli.count(impl_->keys.peers_prefix()));
            return false;
        }
        fprintf(stderr, "[EtcdRegistry %s] cluster ready (%d nodes)\n",
                node_name_.c_str(), world);
        return true;
    } catch (const std::exception& ex) {
        fprintf(stderr, "[EtcdRegistry %s] rendezvous failed: %s\n",
                node_name_.c_str(), ex.what());
        return false;
    }
}

bool EtcdRegistry::leader_flow(std::vector<PeerEntry>& out) {
    auto& K = impl_->keys;

    // Round 1: publish base entry (rank unassigned) + shm name.
    PeerEntry mine{};
    rs_.fill_local_entry(mine);
    mine.rank = -1;
    impl_->cli.put(K.peer(node_name_), etcd::encode_peer(mine), lease_id_);
    impl_->cli.put(K.shm(node_name_), shm_name_, lease_id_);

    // Membership gate: at least max(num_rht_shards, expected_min_nodes) nodes, stable.
    int gate = std::max(rs_.cfg_.num_rht_shards, rs_.cfg_.expected_min_nodes);
    if (gate < 1) gate = 1;
    int n_obs = await_membership(gate);
    if (n_obs < 0) {
        fprintf(stderr, "[EtcdRegistry leader] membership gate timeout "
                "(%ld/%d peers)\n",
                (long)impl_->cli.count(K.peers_prefix()), gate);
        return false;
    }

    // Ranks and /config belong to the cluster, not to this node: they go on a
    // lease every member renews, so they outlive any single node.
    cluster_lease_ = impl_->cli.grant_lease(ttl_);
    impl_->cli.keep_alive(cluster_lease_);
    impl_->cli.put(K.cluster_lease(), std::to_string(cluster_lease_), cluster_lease_);

    // Assign dense ranks by sorted /peers key order (deterministic).
    auto peers = impl_->cli.get_prefix(K.peers_prefix());
    for (size_t i = 0; i < peers.size(); i++)
        impl_->cli.put(K.rank_slot((int)i), leaf(peers[i].key), cluster_lease_);

    // Learn our own rank, then resolve the authoritative shard map.
    int my_rank = read_my_rank();
    if (my_rank < 0) {
        fprintf(stderr, "[EtcdRegistry leader] self rank unresolved\n");
        return false;
    }
    rs_.cfg_.rank = my_rank;

    // RHT min-node barrier: never build a partial shard set.
    if (rs_.cfg_.num_rht_shards > 0 && n_obs < rs_.cfg_.num_rht_shards) {
        fprintf(stderr, "[EtcdRegistry leader] node count %d < num_rht_shards %d — "
                "refusing RHT bring-up\n", n_obs, rs_.cfg_.num_rht_shards);
        return false;
    }

    rs_.resolve_shard_map(n_obs);

    ShardMapMsg sm{};
    sm.magic         = CLUSTER_SHARDMAP_MAGIC;
    sm.version       = CLUSTER_PROTOCOL_VERSION;
    sm.num_shards    = (uint32_t)rs_.num_shards_;
    sm.rht_capacity  = rs_.rht_capacity_;
    sm.rdma_sl       = rs_.rdma_sl_;
    sm.rdma_tc       = rs_.rdma_tc_;
    sm.rht_transport        = rs_.rht_transport_;
    sm.peer_index_transport = rs_.peer_index_transport_;
    sm.rht_slots     = rs_.rht_slots_;
    sm.dc_access_key = rs_.dc_access_key_;
    sm.remote_op_transport = rs_.remote_op_transport_;
    sm.register_chunk_size = (uint32_t)rs_.register_chunk_size();
    for (int s = 0; s < rs_.num_shards_; s++)
        sm.holder_rank[s] = (uint32_t)rs_.holder_of_shard_[s];
    impl_->cli.put(K.config(), etcd::encode_shardmap(sm), cluster_lease_);
    config_published_ = true;

    // Round 2: leader's own RHT shard + remote-op + DCT; re-publish with rank set.
    mine.rank = my_rank;
    rs_.setup_local_rht_shard(mine);
    rs_.setup_remote_op(mine);
    rs_.setup_dct(mine);
    impl_->cli.put(K.peer(node_name_), etcd::encode_peer(mine), lease_id_);

    // stamp_shm_ready() happens in run_rendezvous() once world_size is known.

    // Round 3: assemble the full peer set once every holder is RHT-ready.
    return follower_flow(out, true);
}

bool EtcdRegistry::follower_flow(std::vector<PeerEntry>& out, bool published) {
    auto& K = impl_->keys;
    const int T = rs_.cfg_.bootstrap_timeout_sec;

    if (!published) {
        // Round 1: publish base entry (rank unassigned) + shm name.
        PeerEntry mine{};
        rs_.fill_local_entry(mine);
        mine.rank = -1;
        impl_->cli.put(K.peer(node_name_), etcd::encode_peer(mine), lease_id_);
        impl_->cli.put(K.shm(node_name_), shm_name_, lease_id_);

        // Wait for our leader-assigned rank.
        int my_rank = -1;
        if (!etcd::poll_until(T, [&] { my_rank = read_my_rank(); return my_rank >= 0; })) {
            fprintf(stderr, "[EtcdRegistry %s] rank not assigned\n",
                    node_name_.c_str());
            return false;
        }
        rs_.cfg_.rank = my_rank;

        // Adopt the leader's authoritative shard map.
        std::string cfg_val;
        if (!etcd::poll_until(T, [&] { return impl_->cli.get(K.config(), cfg_val); })) {
            fprintf(stderr, "[EtcdRegistry %s] config not published\n",
                    node_name_.c_str());
            return false;
        }
        if (!adopt_config(cfg_val)) return false;

        // Round 2: holders build RHT + remote-op + DCT; re-publish with rank set.
        mine.rank = my_rank;
        rs_.setup_local_rht_shard(mine);
        rs_.setup_remote_op(mine);
        rs_.setup_dct(mine);
        impl_->cli.put(K.peer(node_name_), etcd::encode_peer(mine), lease_id_);
    }

    // Round 3: wait until every registered node carries its rank (and holders
    // their RHT shard id), then assemble out[rank] = entry. A rank without a
    // /peers entry is a node that is down (rank -1); clients skip its shard.
    auto all_ready = [&]() -> bool {
        int world = world_size();
        out.assign(world, PeerEntry{});
        for (auto& e : out) e.rank = -1;
        bool self = false;
        for (auto& v : impl_->cli.get_prefix_values(K.peers_prefix())) {
            PeerEntry e{};
            if (!etcd::decode_peer(v, e)) continue;
            if (e.rank < 0 || e.rank >= world) return false;   // rank not yet stamped
            bool holder = false;
            for (int s = 0; s < rs_.num_shards_; s++)
                if (rs_.holder_of_shard_[s] == e.rank) { holder = true; break; }
            if (holder && e.rht_shard_id < 0) return false;
            out[e.rank] = e;
            self = self || e.rank == rs_.cfg_.rank;
        }
        return self;
    };
    if (!etcd::poll_until(T, all_ready)) {
        fprintf(stderr, "[EtcdRegistry %s] round-3 assembly timeout\n",
                node_name_.c_str());
        return false;
    }
    return true;
}

bool EtcdRegistry::join_existing(std::vector<PeerEntry>& out) {
    auto& K = impl_->keys;
    const int T = rs_.cfg_.bootstrap_timeout_sec;

    // 1. Adopt the existing shard map (holder set, M, capacity).
    std::string cfg_val;
    if (!impl_->cli.get(K.config(), cfg_val)) {
        fprintf(stderr, "[EtcdRegistry %s] join: /config vanished\n",
                node_name_.c_str());
        return false;
    }
    if (!adopt_config(cfg_val)) return false;

    // 2. Self-CAS a dense rank: from the rank-slot count upward, claim the
    //    first free rank slot. Concurrent joiners fan out to distinct ranks.
    //    The slot is the cluster's, so a restart of this node gets it back.
    int my_rank = -1;
    int base = world_size();
    for (int i = base; i < base + 1024; i++) {
        if (impl_->cli.create(K.rank_slot(i), node_name_, cluster_lease_)) {
            my_rank = i;
            break;
        }
    }
    if (my_rank < 0) {
        fprintf(stderr, "[EtcdRegistry %s] join: no free rank slot\n",
                node_name_.c_str());
        return false;
    }

    // 3. Guard: the claimed rank must not be an RHT holder rank.
    for (int s = 0; s < rs_.num_shards_; s++) {
        if (rs_.holder_of_shard_[s] == my_rank) {
            fprintf(stderr, "[EtcdRegistry %s] join: rank %d collides with "
                    "holder of shard %d — refusing\n",
                    node_name_.c_str(), my_rank, s);
            impl_->cli.del(K.rank_slot(my_rank));
            return false;
        }
    }
    rs_.cfg_.rank = my_rank;

    // 4. Set up remote-op + DCT so existing nodes can reach us; no local RHT shard.
    PeerEntry mine{};
    rs_.fill_local_entry(mine);
    mine.rank = my_rank;
    rs_.setup_remote_op(mine);
    rs_.setup_dct(mine);

    int world = world_size();
    impl_->cli.put(K.peer(node_name_), etcd::encode_peer(mine), lease_id_);
    impl_->cli.put(K.shm(node_name_), shm_name_, lease_id_);
    impl_->cli.put(K.ready(node_name_), "1", lease_id_);
    rs_.cfg_.world_size = world;
    rs_.stamp_shm_ready(my_rank, world);

    fprintf(stderr, "[EtcdRegistry %s] joined as rank %d (data peer, %d nodes)\n",
            node_name_.c_str(), my_rank, world);

    // 5. Assemble the current peer set, indexed by dense rank; ranks of nodes
    //    that are down keep rank -1.
    out.assign(world, PeerEntry{});
    for (auto& e : out) e.rank = -1;
    for (auto& v : impl_->cli.get_prefix_values(K.peers_prefix())) {
        PeerEntry e{};
        if (etcd::decode_peer(v, e) && e.rank >= 0 && e.rank < world)
            out[e.rank] = e;
    }
    (void)T;
    return true;
}

} // namespace radixshmem

#endif // RADIXSHMEM_HAS_ETCD
