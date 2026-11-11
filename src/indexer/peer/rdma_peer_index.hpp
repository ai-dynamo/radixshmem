// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Peer index over one-sided RDMA: walks a peer's tree with READs and pins it with
// FAAs. Owns the RDMA context and the per-peer endpoints; borrows the peer table.

#pragma once

#include "radixshmem_config.h"
#include "cluster_manifest.h"
#include "peer_index.h"
#include "rdma_remote_walker.h"
#include "types.h"

#include "peer_breaker.hpp"

#include <chrono>
#include <exception>
#include <memory>
#include <mutex>
#include <vector>

namespace radixshmem {

class PeerRegistry;
class RdmaTransportClient;

class RdmaPeerIndex final : public PeerIndexBackend {
  public:
    // Opens cfg.dev with the cluster's QoS and peer-index transport (XRC or DC).
    RdmaPeerIndex(const ShardMapMsg& sm, PeerRegistry& peers, const PeerIndexConfig& cfg);
    ~RdmaPeerIndex() override;

    RdmaPeerIndex(const RdmaPeerIndex&) = delete;
    RdmaPeerIndex& operator=(const RdmaPeerIndex&) = delete;

    QueryReply query(int rank, const uint64_t* hashes, int n, const RemoteQuery& q,
                     const std::vector<RemoteHit>& hints) override;
    void unlock(const uint64_t* hashes, int n, PinRecord&& pin) override;
    // Opens the XRC endpoint now instead of on the first walk; DC needs none.
    bool preconnect(int rank) override;
    void peer_changed(const PeerEntry& e) override;
    void peer_left(int rank) override;

  private:
    // Resize the endpoint slots to match the peer table (grow hook target).
    void resize(int n);
    bool ensure_connected(int rank);
    bool connect_peer(const PeerEntry& peer, int peer_idx);
    // A walk failed: drop the endpoint, the next allowed walk reconnects.
    void fault(int rank, const char* op, const std::exception& e,
               std::chrono::steady_clock::time_point t0);

    PeerRegistry&        peers_;
    PeerIndexConfig      cfg_;
    uint32_t             transport_;
    uint64_t             dc_access_key_;
    std::unique_ptr<RdmaTransportClient> tx_;   // outlives the endpoints in slots_

    struct Slot {
        std::unique_ptr<IRdmaEndpoint> ep;
        RemoteTreeInfo               info;
        uint64_t                     instance = 0;   // run of the peer the endpoint reaches
        bool                         failed = false;   // reconnect with a short timeout
    };
    std::vector<Slot> slots_;   // rank-indexed, guarded by peers_.mutex()
    // walk and release share tx_->buf(); interleaved threads read torn entries.
    std::mutex walk_mu_;
    std::unique_ptr<PeerBreaker> breaker_;
};

} // namespace radixshmem
