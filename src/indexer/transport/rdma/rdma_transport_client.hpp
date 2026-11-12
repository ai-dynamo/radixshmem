// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Client-side RDMA base of one plane (peer index, RHT): device context, CQ, an
// optional registered scratch buffer for one-sided walk replies, and a DC initiator.

#pragma once

#include "cluster_manifest.h"   // PeerEntry
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

    // Open the device, size the CQ for `world_size` peers, and register a
    // scratch buffer of `buf_size` bytes (0 = none). QoS is adopted before any
    // QP connect. The device and GID index: RADIXSHMEM_INDEX_DEV / _GID_IDX, else
    // what `self` (this node's /peers entry, null for an observer) published,
    // else the first device and GID index 3.
    RdmaTransportClient(const PeerEntry* self, int sl, int tc, int world_size,
                        size_t buf_size = BUF_SIZE) {
        IndexDevice d = index_device_env();
        const bool published = self && self->index_dev[0] != '\0';
        if (d.dev.empty() && published)
            d.dev.assign(self->index_dev, strnlen(self->index_dev, sizeof(self->index_dev)));
        if (d.gid_idx < 0) d.gid_idx = published ? self->gid_index : 3;
        ctx_ = std::make_unique<RdmaContext>(d.dev.empty() ? nullptr : d.dev.c_str(), 1,
                                             d.gid_idx);
        ctx_->set_qos(sl, tc);
        int cq_depth = std::max(256, world_size * 128);
        cq_ = ibv_create_cq(ctx_->ctx(), cq_depth, nullptr, nullptr, 0);
        if (!cq_)
            throw std::runtime_error("RdmaTransportClient: ibv_create_cq failed");
        if (buf_size == 0) return;
        buf_ = aligned_alloc(4096, buf_size);
        memset(buf_, 0, buf_size);
        buf_mr_ = std::make_unique<RdmaMemoryRegion>(
            ctx_->pd(), buf_, buf_size,
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

    // Per-rank DC address handle, cached.
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
    std::vector<ibv_ah*>               ah_;   // per-rank
    std::vector<ibv_gid>               ah_gid_;
    std::vector<ibv_ah*>               retired_ah_;
};

} // namespace radixshmem
