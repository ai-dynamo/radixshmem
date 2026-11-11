// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "rdma_peer_index.hpp"

#include "peer_registry.hpp"
#include "xrc_connect.hpp"
#include "rdma_transport_client.hpp"
#include "cluster_protocol.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <shared_mutex>

namespace radixshmem {

RdmaPeerIndex::RdmaPeerIndex(const ShardMapMsg& sm, PeerRegistry& peers,
                             const PeerIndexConfig& cfg)
    : peers_(peers), cfg_(cfg), transport_(sm.peer_index_transport),
      dc_access_key_(sm.dc_access_key),
      breaker_(std::make_unique<PeerBreaker>(
          "[RadixClient rank " + std::to_string(peers.my_rank()) + "] peer index",
          cfg.faults)) {
    // QoS (rank 0's SL/TC) is adopted inside the transport before any QP connect.
    tx_ = std::make_unique<RdmaTransportClient>(cfg_.dev.empty() ? nullptr : cfg_.dev.c_str(),
                                                cfg_.gid_idx, sm.rdma_sl, sm.rdma_tc,
                                                peers.world_size());
    if (transport_ == CLUSTER_TRANSPORT_DC) {
        tx_->shared_dci();
        fprintf(stderr, "[RadixClient rank %d] DC initiator up\n", peers.my_rank());
    }
    resize(peers.world_size());
    peers_.set_grow_hook([this](int n) { resize(n); });
}

static double secs_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

RdmaPeerIndex::~RdmaPeerIndex() {
    peers_.set_grow_hook(nullptr);
    slots_.clear();
}

void RdmaPeerIndex::resize(int n) {
    if ((int)slots_.size() < n) slots_.resize(n);
}

bool RdmaPeerIndex::connect_peer(const PeerEntry& peer, int peer_idx) {
    auto& slot = slots_[peer_idx];
    int my_rank = peers_.my_rank();
    breaker_->name(peer_idx, peer.host);

    if (transport_ == CLUSTER_TRANSPORT_DC) {
        ibv_gid dgid;
        memcpy(&dgid, peer.gid_raw, 16);
        int ws = peers_.world_size();
        ibv_ah* ah = tx_->ah_for(peer_idx, dgid, peer.lid, peer.link_layer, ws);
        if (!ah) {
            fprintf(stderr,
                    "[RdmaPeerIndex rank %d] ibv_create_ah to peer %d failed\n",
                    my_rank, peer.rank);
            return false;
        }
        DcTargetInfo dst{ah, peer.dct_num, dc_access_key_};
        slot.ep = std::make_unique<DcReader>(tx_->shared_dci(), dst);
    } else {
        slot.ep = xrc_connect(tx_->ctx(), tx_->cq(), peer.host, peer.xrc_listen_port,
                              my_rank, slot.failed ? std::min(cfg_.bootstrap_timeout_sec, 2)
                                                   : cfg_.bootstrap_timeout_sec);
        if (!slot.ep) return false;   // reported by the breaker
    }

    slot.failed = false;
    slot.instance = peer.instance_id;
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
        cfg_.has_expected && peer_component_abi_valid(peer, cfg_.expected);
    if (!ri.component_info_valid)
        fprintf(stderr,
                "[RdmaPeerIndex rank %d] remote walks to peer rank %d "
                "disabled (%s): no remote hits from this peer\n",
                my_rank, peer.rank,
                cfg_.has_expected
                    ? "peer config/ABI mismatch"
                    : "no expected config (observer without consensus)");
    return true;
}

// Replaced manifest entry (restart) moves MR identity: rebuild the endpoint.
static bool slot_matches_entry(const RemoteTreeInfo& info, const PeerEntry& e) {
    return info.base_addr == e.shm_base_addr && info.rkey == e.shm_rkey &&
           info.mr.length == e.shm_size;
}

bool RdmaPeerIndex::preconnect(int rank) {
    if (transport_ != CLUSTER_TRANSPORT_XRC) return true;
    return ensure_connected(rank);
}

bool RdmaPeerIndex::ensure_connected(int rank) {
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
                "[RdmaPeerIndex rank %d] peer rank %d manifest entry replaced; "
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

void RdmaPeerIndex::peer_changed(const PeerEntry& e) {
    breaker_->retry_now(e.rank);
}

void RdmaPeerIndex::peer_left(int rank) {
    {
        std::unique_lock<std::shared_mutex> lk(peers_.mutex());
        if (rank < 0 || rank >= (int)slots_.size()) return;
        slots_[rank].ep.reset();
        slots_[rank].info = RemoteTreeInfo{};
    }
    breaker_->left(rank, "left the cluster (etcd entry removed)");
}

void RdmaPeerIndex::fault(int rank, const char* op, const std::exception& e,
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

QueryReply RdmaPeerIndex::query(int remote, const uint64_t* hashes, int n,
                                const RemoteQuery& q, const std::vector<RemoteHit>& hints) {
    QueryReply out;
    out.status = RemoteStatus::UNAVAILABLE;
    if (breaker_->gate(remote) == PeerBreaker::Gate::SKIP ||
        !ensure_connected(remote)) {
        if (breaker_->is_down(remote)) breaker_->skipped(remote, PeerBreaker::Op::WALK);
        return out;
    }

    std::lock_guard<std::mutex> wg(walk_mu_);
    std::shared_lock<std::shared_mutex> lk(peers_.mutex());
    auto& slot = slots_[remote];
    if (!slot.ep) return out;   // invalidated by a concurrent replacement
    RdmaRemoteWalker walker(*slot.ep, tx_->buf(), tx_->buf_lkey(),
                            RdmaTransportClient::BUF_SIZE);

    const int incumbent = (int)q.window.incumbent;
    const int full_from = (int)q.window.full_from;
    RemoteComponentWalkResult w;
    auto t0 = std::chrono::steady_clock::now();
    try {
        if (!hints.empty()) {
            w = walker.walk_components_batched(slot.info, hashes, n, q.mask, q.lock,
                                               incumbent, full_from, hints.data(),
                                               (int)hints.size());
        } else {
            w = walker.walk_components(slot.info, hashes, n, q.mask, q.lock,
                                       incumbent, full_from);
        }
    } catch (const std::exception& e) {
        lk.unlock();
        fault(remote, "remote walk", e, t0);
        breaker_->skipped(remote, PeerBreaker::Op::WALK);
        return out;
    }
    breaker_->succeeded(remote);

    out.status = w.tree.status;
    out.rdma_read_count = w.rdma_read_count;
    out.rdma_atomic_count = w.rdma_atomic_count;
    if (out.status != RemoteStatus::OK) return out;
    const uint32_t hit = w.tree.common_hit;
    out.hit_length = (int)hit;
    if (!w.tree.full_slots.empty())
        out.slots.assign(w.tree.full_slots.begin() + std::min(q.window.full_from, hit),
                         w.tree.full_slots.begin() + hit);
    out.swa_start = w.tree.swa_start;
    out.swa_slots = std::move(w.tree.swa_slots);
    out.mamba_slot = w.tree.mamba_checkpoint_slot;
    out.pin = std::move(w.remote_pin);
    out.pin.source_rank = (int32_t)remote;
    out.pin.source_instance = slot.instance;
    return out;
}

void RdmaPeerIndex::unlock(const uint64_t* hashes, int n, PinRecord&& pin) {
    const int remote = pin.source_rank;
    if (remote < 0 || remote >= peers_.world_size()) return;
    std::lock_guard<std::mutex> wg(walk_mu_);
    std::shared_lock<std::shared_mutex> lk(peers_.mutex());
    if (remote >= (int)slots_.size()) return;
    auto& slot = slots_[remote];
    // The run that took the pin is gone (with its pins) once the slot reaches another.
    if (!slot.ep || slot.instance != pin.source_instance) return;
    if (breaker_->gate(remote) == PeerBreaker::Gate::SKIP) {
        breaker_->skipped(remote, PeerBreaker::Op::RELEASE);
        return;
    }
    RdmaRemoteWalker walker(*slot.ep, tx_->buf(), tx_->buf_lkey(),
                            RdmaTransportClient::BUF_SIZE);
    auto t0 = std::chrono::steady_clock::now();
    try {
        (void)walker.dec_ref_components_remote(slot.info, hashes, n, std::move(pin));
    } catch (const std::exception& e) {
        lk.unlock();
        fault(remote, "remote release", e, t0);
    }
}

} // namespace radixshmem
