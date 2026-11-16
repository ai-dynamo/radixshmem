// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Peer index over the peer's Query and Unlock remote ops: the peer matches and pins
// on its own tree. Owns one gRPC channel per peer and needs no RDMA.

#pragma once

#include "bilateral_transport.h"
#include "peer_index.h"

#include "peer_breaker.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace radixshmem {

class PeerRegistry;

// Ops time out after RADIXSHMEM_PEER_INDEX_TIMEOUT_MS (default 500, about when an
// RDMA QP reports a dead peer). UNAVAILABLE or TIMEOUT marks the peer down.
class GrpcPeerIndex final : public PeerIndexBackend {
  public:
    GrpcPeerIndex(PeerRegistry& peers, const PeerIndexConfig& cfg);

    QueryReply query(int rank, const uint64_t* hashes, int n, const RemoteQuery& q,
                     const std::vector<RemoteHit>& hints) override;
    void unlock(const uint64_t* hashes, int n, PinRecord&& pin) override;
    bool preconnect(int rank) override;
    void peer_changed(const PeerEntry& e) override;
    void peer_left(int rank) override;

  private:
    struct Session {
        std::unique_ptr<IRemoteOpClientTransport> ops;
        uint64_t instance = 0;   // the run of the peer the channel reaches
        bool     usable = false; // the peer's tree has the expected layout
        std::atomic<bool> stale{false};   // that run is gone; wait for the new entry
    };
    // Opened from the rank's current entry; null for a rank that is down.
    std::shared_ptr<Session> session(int rank);
    void drop(int rank);

    PeerRegistry&   peers_;
    PeerIndexConfig cfg_;
    int             timeout_ms_;
    PeerBreaker     breaker_;
    std::mutex      mu_;
    std::unordered_map<int, std::shared_ptr<Session>> sessions_;
};

} // namespace radixshmem
