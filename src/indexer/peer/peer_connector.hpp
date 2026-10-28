// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Peer-facing plane: per-peer one-sided data-plane endpoints (lazy) for remote
// walks, and the remote-op session factory. Borrows the shared peer table and
// RDMA base; owns only the per-peer endpoint slots and the DC control stack.

#pragma once

#include "radixshmem_config.h"
#include "cluster_manifest.h"
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
#ifndef RADIXSHMEM_NO_RDMA
class RemotePeer;
class DcInitiator;
class DcTarget;
#endif

struct PeerConnectorConfig {
    uint32_t transport = 0;            // peer-index-plane ClusterTransport (XRC vs DC)
    uint32_t remote_op_transport = 0;  // REMOTE_OP_TRANSPORT_*
    uint64_t dc_access_key = 0;
    int      bootstrap_timeout_sec = 60;
    PeerFaultCounters* faults = nullptr;   // node-wide fault counters, may be null
};

class PeerConnector {
  public:
    PeerConnector(PeerRegistry& peers, RdmaTransportClient& tx,
                  const PeerConnectorConfig& cfg);
    ~PeerConnector();

    PeerConnector(const PeerConnector&) = delete;
    PeerConnector& operator=(const PeerConnector&) = delete;

    // Resize the endpoint slots to match the peer table (grow hook target).
    void resize(int n);

    // Expected ComponentConfig checked at connect (client: local config;
    // observer: bootstrap consensus). Unset => component walks fail closed.
    void set_expected_component_config(const ComponentConfig& config) {
        expected_component_config_ = config;
        has_expected_component_config_ = true;
    }

    // Legacy walk shape: folds the remote continuation into the local
    // public result on extension; returns it unchanged otherwise, also when
    // the peer is unreachable (it is then skipped for a while).
    ComponentQueryResult walk_components(
        int remote, const uint64_t* hashes, int n, uint32_t mask, bool lock,
        int incumbent_common_hit, int local_full_hit,
        const std::vector<RemoteHit>& hits, ComponentQueryResult result,
        std::vector<PinRecord>& pins);

    // One-sided release for a prior walk_components(lock=true) pin.
    bool dec_ref_components_remote(int remote, const uint64_t* hashes, int n,
                                   PinRecord&& pin);

    // Open the XRC data-plane endpoint to `rank` now instead of on first walk.
    bool preconnect(int rank);

    // etcd watcher: `rank` published a new entry / its entry went away.
    void peer_changed(int rank);
    void peer_left(int rank);

#ifndef RADIXSHMEM_NO_RDMA
    // Fresh remote-op session to `rank` (ZMQ or DC); caller owns the handle.
    std::shared_ptr<RemotePeer> open_remote(int rank);
#endif

  private:
    bool ensure_connected(int rank);
    bool connect_peer(const PeerEntry& peer, int peer_idx);
    // A walk failed: drop the endpoint, the next allowed walk reconnects.
    void fault(int rank, const char* op, const std::exception& e,
               std::chrono::steady_clock::time_point t0);
#ifndef RADIXSHMEM_NO_RDMA
    std::shared_ptr<RemotePeer> build_rc_session(
        int rank, std::shared_ptr<std::mutex> stack_mu);
#endif

    PeerRegistry&        peers_;
    RdmaTransportClient& tx_;
    PeerConnectorConfig  cfg_;
    ComponentConfig expected_component_config_{};
    bool has_expected_component_config_ = false;

    struct Slot {
        std::unique_ptr<IRdmaEndpoint> ep;
        RemoteTreeInfo               info;
        bool                         failed = false;   // reconnect with a short timeout
    };
    std::vector<Slot> slots_;   // rank-indexed, guarded by peers_.mutex()
    // walk and release share tx_.buf(); interleaved threads read torn entries.
    std::mutex walk_mu_;
    std::unique_ptr<PeerBreaker> breaker_;

#ifndef RADIXSHMEM_NO_RDMA
    std::unique_ptr<DcInitiator> dc_rc_ini_;
    std::unique_ptr<DcTarget>    dc_rc_tgt_;
    std::mutex rc_open_mu_;
    std::shared_ptr<std::mutex> rc_stack_op_mu_;
    int rc_client_rank_ = -1;
#endif
};

} // namespace radixshmem
