// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "peer_connector.hpp"

#include "peer_registry.hpp"
#include "xrc_connect.hpp"
#include "rdma_transport_client.hpp"
#include "cluster_protocol.h"
#ifndef RADIXSHMEM_NO_RDMA
#include "radix_client.h"
#endif
#ifdef RADIXSHMEM_REMOTE_OP_RDMA
#include "dc_bilateral.hpp"
#endif
#ifdef RADIXSHMEM_HAS_GRPC
#include "grpc_remote_op.hpp"
#endif

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <shared_mutex>

namespace radixshmem {

PeerConnector::PeerConnector(PeerRegistry& peers, RdmaTransportClient& tx,
                             const PeerConnectorConfig& cfg)
    : peers_(peers), tx_(tx), cfg_(cfg),
      breaker_(std::make_unique<PeerBreaker>(
          "[RadixClient rank " + std::to_string(peers.my_rank()) + "] peer index",
          cfg.faults)) {}

static double secs_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

PeerConnector::~PeerConnector() {
#ifdef RADIXSHMEM_REMOTE_OP_RDMA
    if (rc_stack_op_mu_ && rc_stack_op_mu_.use_count() > 1) {
        fprintf(stderr,
                "[PeerConnector] WARNING: %ld dc remote-op session(s) (or "
                "retained lock=true results) outlive this client — their "
                "destruction is use-after-free. Drop all RemotePeer handles AND "
                "locked QueryResults before the RadixClient.\n",
                (long)(rc_stack_op_mu_.use_count() - 1));
    }
    dc_rc_ini_.reset();
    dc_rc_tgt_.reset();
#endif
    slots_.clear();
}

void PeerConnector::resize(int n) {
    if ((int)slots_.size() < n) slots_.resize(n);
}

bool PeerConnector::connect_peer(const PeerEntry& peer, int peer_idx) {
    auto& slot = slots_[peer_idx];
    int my_rank = peers_.my_rank();
    breaker_->name(peer_idx, peer.host);

    if (cfg_.transport == CLUSTER_TRANSPORT_DC) {
        ibv_gid dgid;
        memcpy(&dgid, peer.gid_raw, 16);
        int ws = peers_.world_size();
        ibv_ah* ah = tx_.ah_for(peer_idx, dgid, peer.lid, peer.link_layer, ws);
        if (!ah) {
            fprintf(stderr,
                    "[PeerConnector rank %d] ibv_create_ah to peer %d failed\n",
                    my_rank, peer.rank);
            return false;
        }
        DcTargetInfo dst{ah, peer.dct_num, cfg_.dc_access_key};
        slot.ep = std::make_unique<DcReader>(tx_.shared_dci(), dst);
    } else {
        slot.ep = xrc_connect(tx_.ctx(), tx_.cq(), peer.host, peer.xrc_listen_port,
                              my_rank, slot.failed ? std::min(cfg_.bootstrap_timeout_sec, 2)
                                                   : cfg_.bootstrap_timeout_sec);
        if (!slot.ep) return false;   // reported by the breaker
    }

    slot.failed = false;
    auto& ri = slot.info;
    ri.mr.rkey   = peer.shm_rkey;
    ri.mr.addr   = peer.shm_base_addr;
    ri.mr.length = peer.shm_size;
    ri.rkey      = peer.shm_rkey;
    ri.base_addr = peer.shm_base_addr;
    ri.node_pool_offset         = peer.node_pool_offset;
    ri.node_pool_capacity       = peer.node_pool_capacity;
    ri.data_pool_offset         = peer.data_pool_offset;
    ri.global_ht_offset         = peer.global_ht_offset;
    ri.global_ht_bucket_count   = peer.ht_bucket_count;
    ri.global_ht_entry_capacity = peer.ht_entry_capacity;
    ri.root_node_id             = peer.root_node_id;
    ri.rdma_rwlock_addr         = peer.shm_base_addr + peer.rdma_rwlock_offset;
    ri.rdma_reader_count_addr   = ri.rdma_rwlock_addr + RDMA_READER_COUNT_DELTA;

    ri.component_config           = peer.component_config;
    ri.component_state_offset     = peer.component_state_offset;
    ri.component_state_entry_size = (uint32_t)sizeof(NodeComponentState);
    ri.component_info_valid =
        has_expected_component_config_ &&
        peer_component_abi_valid(peer, expected_component_config_);
    if (!ri.component_info_valid)
        fprintf(stderr,
                "[PeerConnector rank %d] remote walks to peer rank %d "
                "disabled (%s): no remote hits from this peer\n",
                my_rank, peer.rank,
                has_expected_component_config_
                    ? "peer config/ABI mismatch"
                    : "no expected config (observer without consensus)");
    return true;
}

// Replaced manifest entry (restart) moves MR identity: rebuild the endpoint.
static bool slot_matches_entry(const RemoteTreeInfo& info, const PeerEntry& e) {
    return info.base_addr == e.shm_base_addr && info.rkey == e.shm_rkey &&
           info.mr.length == e.shm_size;
}

bool PeerConnector::preconnect(int rank) {
    if (cfg_.transport != CLUSTER_TRANSPORT_XRC) return true;
    return ensure_connected(rank);
}

bool PeerConnector::ensure_connected(int rank) {
    {
        std::shared_lock<std::shared_mutex> lk(peers_.mutex());
        if (rank < 0 || rank >= (int)slots_.size()) return false;
        if (slots_[rank].ep &&
            slot_matches_entry(slots_[rank].info,
                               peers_.peers_locked()[rank]))
            return true;
    }
    std::unique_lock<std::shared_mutex> lk(peers_.mutex());
    if (rank < 0 || rank >= (int)slots_.size()) return false;
    if (slots_[rank].ep) {
        if (slot_matches_entry(slots_[rank].info, peers_.peers_locked()[rank]))
            return true;
        fprintf(stderr,
                "[PeerConnector rank %d] peer rank %d manifest entry replaced; "
                "invalidating the old endpoint\n", peers_.my_rank(), rank);
        slots_[rank].ep.reset();
        slots_[rank].info = RemoteTreeInfo{};
    }
    if (rank == peers_.my_rank()) return false;
    if (peers_.peers_locked()[rank].rank != rank) return false;   // node down
    auto t0 = std::chrono::steady_clock::now();
    if (!connect_peer(peers_.peers_locked()[rank], rank)) {
        slots_[rank].failed = true;
        breaker_->failed(rank, "connect", "no QP", secs_since(t0));
        return false;
    }
    return true;
}

void PeerConnector::peer_changed(int rank) { breaker_->retry_now(rank); }

void PeerConnector::peer_left(int rank) {
    {
        std::unique_lock<std::shared_mutex> lk(peers_.mutex());
        if (rank < 0 || rank >= (int)slots_.size()) return;
        slots_[rank].ep.reset();
        slots_[rank].info = RemoteTreeInfo{};
    }
    breaker_->left(rank, "left the cluster (etcd entry removed)");
}

void PeerConnector::fault(int rank, const char* op, const std::exception& e,
                          std::chrono::steady_clock::time_point t0) {
    {
        std::unique_lock<std::shared_mutex> lk(peers_.mutex());
        // A healthy DC endpoint only saw another target's error on the shared DCI.
        if (slots_[rank].ep && !slots_[rank].ep->broken()) return;
        slots_[rank].ep.reset();
        slots_[rank].info = RemoteTreeInfo{};
        slots_[rank].failed = true;
    }
    breaker_->failed(rank, op, e.what(), secs_since(t0));
}

ComponentQueryResult PeerConnector::walk_components(
        int remote, const uint64_t* hashes, int n, uint32_t mask, bool lock,
        int incumbent_common_hit, int local_full_hit,
        const std::vector<RemoteHit>& hits, ComponentQueryResult result,
        std::vector<PinRecord>& pins) {
    if (breaker_->gate(remote) == PeerBreaker::Gate::SKIP ||
        !ensure_connected(remote)) {
        if (breaker_->is_down(remote)) breaker_->skipped(remote, PeerBreaker::Op::WALK);
        return result;
    }

    std::lock_guard<std::mutex> wg(walk_mu_);
    std::shared_lock<std::shared_mutex> lk(peers_.mutex());
    auto& slot = slots_[remote];
    if (!slot.ep) return result;   // invalidated by a concurrent replacement
    RdmaRemoteWalker walker(*slot.ep, tx_.buf(), tx_.buf_lkey(),
                            RdmaTransportClient::BUF_SIZE);

    RemoteComponentWalkResult walk_res;
    auto t0 = std::chrono::steady_clock::now();
    try {
        if (!hits.empty()) {
            walk_res = walker.walk_components_batched(slot.info, hashes, n, mask,
                                                      lock, incumbent_common_hit,
                                                      local_full_hit, hits.data(),
                                                      (int)hits.size());
        } else {
            walk_res = walker.walk_components(slot.info, hashes, n, mask, lock,
                                              incumbent_common_hit, local_full_hit);
        }
    } catch (const std::exception& e) {
        lk.unlock();
        fault(remote, "remote walk", e, t0);
        breaker_->skipped(remote, PeerBreaker::Op::WALK);
        return result;
    }
    breaker_->succeeded(remote);

    result.rdma_read_count   = walk_res.rdma_read_count;
    result.rdma_atomic_count = walk_res.rdma_atomic_count;

    const int matched_blocks = (int)walk_res.tree.common_hit;
    if (walk_res.tree.status == RemoteStatus::OK &&
        matched_blocks > incumbent_common_hit) {
        // Fold: Full = local [0,c) + remote [c,matched); SWA/Mamba belong
        // whole to the remote tail source.
        const int c = std::min(local_full_hit, matched_blocks);
        result.common_hit = matched_blocks;
        if ((mask & COMPONENT_MASK_FULL) != 0) {
            if (c > 0 && !result.full_fragments.empty())
                result.full_fragments[0].slot_ids.resize((size_t)c);
            else
                result.full_fragments.clear();
            if (matched_blocks > c)
                result.full_fragments.push_back(ComponentSlotFragment{
                    (int32_t)remote, c,
                    std::vector<int32_t>(
                        walk_res.tree.full_slots.begin() + c,
                        walk_res.tree.full_slots.begin() + matched_blocks)});
        }
        result.swa = ComponentSlotFragment{
            walk_res.tree.swa_slots.empty() ? -1 : (int32_t)remote,
            (int)walk_res.tree.swa_start,
            std::move(walk_res.tree.swa_slots)};
        if (walk_res.tree.mamba_checkpoint_slot != INVALID_COMPONENT_SLOT)
            result.mamba = ComponentCheckpoint{
                (int32_t)remote, walk_res.tree.mamba_checkpoint_slot};
        else
            result.mamba.reset();
        if (!walk_res.remote_pin.empty()) {
            walk_res.remote_pin.source_rank = (int32_t)remote;
            pins.push_back(std::move(walk_res.remote_pin));
        }
    }
    return result;
}

bool PeerConnector::dec_ref_components_remote(int remote,
                                              const uint64_t* hashes, int n,
                                              PinRecord&& pin) {
    if (remote < 0 || remote >= peers_.world_size()) return false;
    std::lock_guard<std::mutex> wg(walk_mu_);
    std::shared_lock<std::shared_mutex> lk(peers_.mutex());
    if (remote >= (int)slots_.size()) return false;
    auto& slot = slots_[remote];
    if (!slot.ep || breaker_->gate(remote) == PeerBreaker::Gate::SKIP) return false;
    RdmaRemoteWalker walker(*slot.ep, tx_.buf(), tx_.buf_lkey(),
                            RdmaTransportClient::BUF_SIZE);
    auto t0 = std::chrono::steady_clock::now();
    try {
        return walker.dec_ref_components_remote(slot.info, hashes, n,
                                                std::move(pin));
    } catch (const std::exception& e) {
        lk.unlock();
        fault(remote, "remote release", e, t0);
        return false;
    }
}

// ==================== Remote-op session factory ====================

#ifndef RADIXSHMEM_NO_RDMA

static int resolve_rc_client_rank(int default_rank) {
    int client_rank = default_rank;
    if (const char* e = ::getenv("RADIXSHMEM_RC_CLIENT_RANK")) {
        char* end = nullptr;
        long v = std::strtol(e, &end, 10);
        if (end != e && *end == '\0' && v >= 0 && v <= INT_MAX)
            client_rank = (int)v;
        else
            fprintf(stderr, "[PeerConnector] ignoring invalid "
                    "RADIXSHMEM_RC_CLIENT_RANK='%s'\n", e);
    }
    return client_rank;
}

std::shared_ptr<RemotePeer> PeerConnector::build_rc_session(
        int rank, std::shared_ptr<std::mutex> stack_mu) {
    PeerEntry p;
    {
        std::shared_lock<std::shared_mutex> lk(peers_.mutex());
        if (rank < 0 || rank >= peers_.size_locked())
            throw std::out_of_range("open_remote: rank out of range");
        p = peers_.peers_locked()[rank];
    }
    if (p.rank != rank)
        throw std::runtime_error("open_remote: peer rank " + std::to_string(rank) +
                                 " is not registered (node down)");
    int client_rank = rc_client_rank_;
    if (cfg_.remote_op_transport == REMOTE_OP_TRANSPORT_DC) {
#ifdef RADIXSHMEM_REMOTE_OP_RDMA
        if (p.rc_dct_num == 0)
            throw std::runtime_error(
                "open_remote: peer rank " + std::to_string(rank) +
                " has remote ops disabled (rc_dct_num == 0)");
        if (!dc_rc_tgt_) {
            auto tgt = std::make_unique<DcTarget>(
                tx_.ctx(), cfg_.dc_access_key, /*srq_max_wr=*/16);
            auto ini = std::make_unique<DcInitiator>(tx_.ctx(), cfg_.dc_access_key);
            dc_rc_tgt_ = std::move(tgt);
            dc_rc_ini_ = std::move(ini);
        }
        int p_eff = 1;
        for (int i = 0; i < 3 && p.rc_dct_num_extra[i] != 0; ++i) ++p_eff;
        int lane = (int)((unsigned)client_rank % (unsigned)p_eff);
        uint32_t dctn = lane == 0 ? p.rc_dct_num : p.rc_dct_num_extra[lane - 1];
        ibv_gid dgid;
        std::memcpy(&dgid, p.gid_raw, 16);
        ibv_ah* ah = dc_create_ah(tx_.ctx(), dgid, p.lid, p.link_layer);
        if (!ah)
            throw std::runtime_error(
                "open_remote: ibv_create_ah to rank " +
                std::to_string(rank) + " failed");
        auto t = std::make_unique<DcClientTransport>(
            *dc_rc_ini_, *dc_rc_tgt_, ah, dctn, client_rank,
            std::move(stack_mu));
        std::string dep = "dc://" + std::string(p.host) + "#" +
                          std::to_string(dctn);
        auto rc = std::make_shared<RemotePeer>(std::move(t), std::move(dep));
        rc->set_peer_rank(rank);
        return rc;
#else
        throw std::runtime_error(
            "open_remote: cluster remote-op transport is dc but "
            "this build has no RDMA control-plane backend");
#endif
    }

    if (p.remote_op_listen_port == 0)
        throw std::runtime_error(
            "open_remote: peer rank " + std::to_string(rank) +
            " has remote ops disabled (remote-op listen port == 0)");
    std::string hostport = std::string(p.host) + ":" +
                           std::to_string(p.remote_op_listen_port);

    if (cfg_.remote_op_transport == REMOTE_OP_TRANSPORT_GRPC) {
#ifdef RADIXSHMEM_HAS_GRPC
        auto t = std::make_unique<GrpcRemoteOpClientTransport>(
            hostport, grpc_remote_op_timeout_ms());
        auto rc = std::make_shared<RemotePeer>(std::move(t), "grpc://" + hostport);
        rc->set_peer_rank(rank);
        return rc;
#else
        throw std::runtime_error(
            "open_remote: cluster remote-op transport is grpc but "
            "gRPC is not compiled into this build");
#endif
    }
    if (cfg_.remote_op_transport != REMOTE_OP_TRANSPORT_ZMQ) {
        throw std::runtime_error(
            "open_remote: cluster advertises remote-op transport enum " +
            std::to_string(cfg_.remote_op_transport) + ", which this build "
            "does not serve (removed xrc/rdma backend?) — redeploy rank 0 "
            "with remote_op_transport = grpc, dc or zmq");
    }
    std::string ep = "tcp://" + hostport;
#ifdef RADIXSHMEM_HAS_ZMQ
    auto rc = std::make_shared<RemotePeer>(ep);
    rc->set_peer_rank(rank);
    return rc;
#else
    throw std::runtime_error(
        "open_remote: cluster remote-op transport is zmq but "
        "ZMQ is not compiled into this build");
#endif
}

std::shared_ptr<RemotePeer> PeerConnector::open_remote(int rank) {
    if (rank == peers_.my_rank())
        throw std::runtime_error(
            "open_remote: target is this client's own rank — "
            "talking to yourself is query()'s job");
    std::lock_guard<std::mutex> g(rc_open_mu_);
    if (rc_client_rank_ < 0)
        rc_client_rank_ = resolve_rc_client_rank(peers_.my_rank());
    // DC sessions share one initiator+target stack; ZMQ sessions own theirs.
    std::shared_ptr<std::mutex> mu;
    std::unique_lock<std::mutex> opg;
    if (cfg_.remote_op_transport == REMOTE_OP_TRANSPORT_DC) {
        if (!rc_stack_op_mu_)
            rc_stack_op_mu_ = std::make_shared<std::mutex>();
        mu = rc_stack_op_mu_;
        opg = std::unique_lock<std::mutex>(*mu);
    }
    return build_rc_session(rank, std::move(mu));
}

#endif  // !RADIXSHMEM_NO_RDMA

} // namespace radixshmem
