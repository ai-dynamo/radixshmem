// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared client-side RDMA base: device context, CQ, a registered scratch buffer
// for one-sided walk replies, and one process-shared DC initiator. Peer, RHT, and
// remote-op planes borrow these; none of them owns the context.

#pragma once

#include "rdma_transport.hpp"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace radixshmem {

class RdmaTransportClient {
  public:
    static constexpr size_t BUF_SIZE = 1 << 20;

    // Open the device, size the CQ for `world_size` peers, and register the
    // scratch buffer. QoS is adopted before any QP connect.
    RdmaTransportClient(const char* dev, int gid_idx, int sl, int tc,
                        int world_size) {
        ctx_ = std::make_unique<RdmaContext>(dev, 1, gid_idx);
        ctx_->set_qos(sl, tc);
        int cq_depth = std::max(256, world_size * 128);
        cq_ = ibv_create_cq(ctx_->ctx(), cq_depth, nullptr, nullptr, 0);
        if (!cq_)
            throw std::runtime_error("RdmaTransportClient: ibv_create_cq failed");
        buf_ = aligned_alloc(4096, BUF_SIZE);
        memset(buf_, 0, BUF_SIZE);
        buf_mr_ = std::make_unique<RdmaMemoryRegion>(
            ctx_->pd(), buf_, BUF_SIZE,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
            IBV_ACCESS_REMOTE_READ);
    }

    ~RdmaTransportClient() {
        // AHs after the endpoints (dropped by the owner) but before the DCI/ctx.
        for (ibv_ah* ah : ah_) if (ah) ibv_destroy_ah(ah);
        for (ibv_ah* ah : retired_ah_) ibv_destroy_ah(ah);
        ah_.clear();
        dci_.reset();
        buf_mr_.reset();
        if (buf_) { free(buf_); buf_ = nullptr; }
        if (cq_) { ibv_destroy_cq(cq_); cq_ = nullptr; }
        ctx_.reset();
    }

    RdmaTransportClient(const RdmaTransportClient&) = delete;
    RdmaTransportClient& operator=(const RdmaTransportClient&) = delete;

    RdmaContext& ctx()      { return *ctx_; }
    ibv_cq*      cq()       { return cq_; }
    void*        buf()      { return buf_; }
    uint32_t     buf_lkey() { return buf_mr_->lkey(); }

    // Lazily built shared DC initiator (one DCI, its own CQ). Callers that
    // need it before any endpoint connects trigger the build here.
    DcInitiator* shared_dci() {
        if (!dci_) dci_ = std::make_unique<DcInitiator>(*ctx_);
        return dci_.get();
    }
    DcInitiator* shared_dci_if_present() { return dci_.get(); }

    // Drop the shared DCI; only valid once every DcReader borrowing it is gone.
    void reset_dci() { dci_.reset(); }

    // Per-rank DC address handle, cached and shared across data + RHT planes.
    // Built once from the peer's gid/lid; nullptr on failure. `n` sizes the cache.
    ibv_ah* ah_for(int rank, const ibv_gid& dgid, uint16_t dlid,
                   uint8_t link_layer, int n) {
        if ((int)ah_.size() < n) { ah_.resize(n, nullptr); ah_gid_.resize(n); }
        if (rank < 0 || rank >= (int)ah_.size()) return nullptr;
        if (ah_[rank] && std::memcmp(&ah_gid_[rank], &dgid, sizeof(dgid)) != 0) {
            retired_ah_.push_back(ah_[rank]);   // a peer moved; old endpoints may still hold it
            ah_[rank] = nullptr;
        }
        if (!ah_[rank]) {
            ah_[rank] = dc_create_ah(*ctx_, dgid, dlid, link_layer);
            ah_gid_[rank] = dgid;
        }
        return ah_[rank];
    }

  private:
    std::unique_ptr<RdmaContext>      ctx_;
    ibv_cq*                            cq_ = nullptr;
    void*                              buf_ = nullptr;
    std::unique_ptr<RdmaMemoryRegion>  buf_mr_;
    std::unique_ptr<DcInitiator>       dci_;
    std::vector<ibv_ah*>               ah_;   // per-rank, shared data + RHT planes
    std::vector<ibv_gid>               ah_gid_;
    std::vector<ibv_ah*>               retired_ah_;
};

} // namespace radixshmem
