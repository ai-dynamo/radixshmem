// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cluster_manifest.h"   // PeerEntry
#include "component_types.h"    // RemoteQuery, QueryReply, PinRecord

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace radixshmem {

class PeerRegistry;
struct PeerFaultCounters;
struct ShardMapMsg;

// A node of the peer the RHT picked, starting at block `position` of the query.
struct RemoteHit {
    int position;
    uint32_t tree_node_id;
};

// Step two of a distributed query: once the RHT named the peer, match the hashes
// on that peer's tree and, with a lock, pin what extends the caller's own hit.
// The cluster runs one backend (/config): one-sided RDMA reads of the peer's shm,
// or the peer's Query and Unlock remote ops over gRPC. A peer that fails is
// skipped for a while; its queries then return UNAVAILABLE.
class PeerIndexBackend {
public:
    virtual ~PeerIndexBackend() = default;

    // `hints` are the RHT routes of `rank` along the hashes. reply.pin names the
    // rank and the run of it that holds the pin.
    virtual QueryReply query(int rank, const uint64_t* hashes, int n, const RemoteQuery& q,
                             const std::vector<RemoteHit>& hints) = 0;
    // Drops a pin of query(lock). Not idempotent, so a release that may have
    // reached the peer is never sent again: a peer that does not answer keeps it.
    virtual void unlock(const uint64_t* hashes, int n, PinRecord&& pin) = 0;

    // Opens the way to `rank` before its first query; false if it failed.
    virtual bool preconnect(int rank) = 0;
    // /peers watcher: the rank published a new entry / its entry went away.
    virtual void peer_changed(const PeerEntry& e) = 0;
    virtual void peer_left(int rank) = 0;
};

struct PeerIndexConfig {
    std::string dev;                  // rdma: device of the one-sided QPs, "" = first
    int gid_idx = 3;
    int bootstrap_timeout_sec = 60;
    PeerFaultCounters* faults = nullptr;
    // Layout a peer's tree must have for its slots to be usable; without one
    // every query fails closed.
    bool has_expected = false;
    ComponentConfig expected;
};

// The backend /config names (ShardMapMsg::peer_index_backend).
std::unique_ptr<PeerIndexBackend> make_peer_index(const ShardMapMsg& sm, PeerRegistry& peers,
                                                  const PeerIndexConfig& cfg);

} // namespace radixshmem
