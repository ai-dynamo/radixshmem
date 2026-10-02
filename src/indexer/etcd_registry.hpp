// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// EtcdRegistry — the sole cluster bootstrap path (RDMA builds). Discovers peers
// and orders bringup through etcd (lease / create-if-absent / prefix-get): the
// leader assigns dense ranks by sorted /peers order, resolves the shard map from
// the observed node count, and every node publishes its final PeerEntry + shm name.
//
// Guarded on SHMRADIX_HAS_ETCD: when etcd is not built in, this TU is empty and
// there is no cluster bootstrap (CMake FATAL_ERRORs before that in RDMA builds).

#pragma once

#include "shmradix/config.h"

#ifdef SHMRADIX_HAS_ETCD

#include "shmradix/cluster_registry.h"

#include <memory>
#include <string>

namespace shmradix {

class RadixServer;
struct ShardMapMsg;

class EtcdRegistry final : public IClusterRegistry {
public:
    // endpoint: "host:port" (scheme stripped by caller); cluster_id/node_name
    // namespace + identify this peer under radix/<cluster_id>/. shm_name is
    // published to /shm/<node_name> for clients to discover.
    EtcdRegistry(RadixServer& rs, std::string endpoint,
                 std::string cluster_id, std::string node_name,
                 std::string shm_name);
    ~EtcdRegistry() override;

    bool run_rendezvous(std::vector<PeerEntry>& out) override;

    // Kept alive for the object's lifetime; the destructor revokes it.
    int64_t lease_id() const { return lease_id_; }

private:
    // Leader (elected) vs follower flows; both return the fully-assembled peer set.
    // published: this node already ran rounds 1-2 (the leader), only assemble.
    bool leader_flow(std::vector<PeerEntry>& out);
    bool follower_flow(std::vector<PeerEntry>& out, bool published);
    // Adopt the existing shard map, self-CAS a dense rank as a data-only peer,
    // and assemble the current peer set.
    bool join_existing(std::vector<PeerEntry>& out);

    // Wait until count(/peers) >= gate and stays stable for settle_ms; returns
    // the settled count, or -1 on timeout.
    int  await_membership(int gate);
    // Read this node's assigned rank from /rank/slot/*, -1 if not yet published.
    int  read_my_rank();
    // Number of rank slots; ranks are dense in [0, world_size()).
    int  world_size();
    // Remove what a killed earlier run of this node left in etcd; false if a
    // live server still holds this node name.
    bool reclaim_previous_run();
    // Renew the cluster lease that holds /config and the rank slots.
    bool attach_cluster_lease();
    // Decode /config, renew the cluster lease, check it against this node's
    // settings when it predates this run, and install it.
    bool adopt_config(const std::string& cfg_val);

    RadixServer& rs_;
    std::string endpoint_;
    std::string cluster_id_;
    std::string node_name_;
    std::string shm_name_;
    int64_t     lease_id_ = 0;
    int64_t     cluster_lease_ = 0;       // shared by all members
    bool        config_published_ = false;
    int         ttl_ = 30;
    bool        config_preexisting_ = false;

    struct Impl;                 // hides the Go-wrapper C surface from this header
    std::unique_ptr<Impl> impl_;
};

} // namespace shmradix

#endif // SHMRADIX_HAS_ETCD
