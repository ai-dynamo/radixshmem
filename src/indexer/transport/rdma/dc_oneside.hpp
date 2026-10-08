// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// DC (Dynamically Connected) — one shared, mutex-guarded DCI reaches all DCTs
// (O(1) QPs vs O(N) XRC); peer selected per-WR via (AH, dct_num, dc_key).

#pragma once

#include "context.hpp"
#include "endpoint.hpp"

#include <atomic>
#include <cstdint>
#include <deque>
#include <infiniband/verbs.h>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace radixshmem {

// One received message pulled off a DcTarget CQ by poll(); dctn = receiving DCT (demux key).
struct DcRecvCompletion {
    bool     ok       = false;   // false = timeout (no completion within window)
    uint32_t dctn     = 0;       // receiving DCT (from wc.qp_num)
    uint32_t byte_len = 0;       // wc.byte_len — bytes received (RECV)
    uint64_t wr_id    = 0;       // wc.wr_id — which pooled RECV buffer got it
    const char* data  = nullptr; // → pooled RECV buffer (valid until next poll())
};

// Fault flag of one target's endpoint. Set when an op to that target fails,
// also when the error only surfaces while the shared DCI serves another target.
struct DcOwner {
    std::atomic<bool> failed{false};
};

// What an initiator needs to address one target (from the manifest).
struct DcTargetInfo {
    ibv_ah*  ah       = nullptr;   // owned by DcInitiator's AH cache
    uint32_t dct_num  = 0;
    uint64_t dc_key   = 0;
    DcOwner* owner    = nullptr;   // set by DcReader
};

class DcInitiator {
public:
    // Data-plane initiator (READ|FAA|WRITE) on its own CQ.
    explicit DcInitiator(RdmaContext& ctx);
    // Control-plane initiator (SEND). Owns a private send CQ.
    DcInitiator(RdmaContext& ctx, uint64_t dc_key);
    ~DcInitiator();

    DcInitiator(const DcInitiator&) = delete;
    DcInitiator& operator=(const DcInitiator&) = delete;

    // Blocking, fully-signaled one-sided ops against `dst`. Serialized by mu_.
    void rdma_read(const DcTargetInfo& dst,
                   void* local_buf, uint32_t local_lkey,
                   uint64_t remote_addr, uint32_t remote_rkey, uint32_t length);
    uint64_t rdma_fetch_add(const DcTargetInfo& dst,
                            uint64_t remote_addr, uint32_t remote_rkey,
                            uint64_t add_val,
                            void* local_result_buf, uint32_t local_lkey);
    void rdma_read_multi(const DcTargetInfo& dst,
                         const RdmaReadOp* ops, int count, uint32_t local_lkey);

    // One-sided inline write, unsignaled; `local_buf` need only stay valid
    // until the call returns.
    void rdma_write_async_inline(const DcTargetInfo& dst,
                                 uint64_t remote_addr, uint32_t remote_rkey,
                                 const void* local_buf, uint32_t length);
    // Signaled re-write of a cell already written (idempotent): the DCI runs
    // WRs in order, so its completion proves every earlier write landed.
    void fence(const DcTargetInfo& dst, uint64_t remote_addr, uint32_t remote_rkey,
               const void* value, uint32_t length);

    // Blocking SEND to (ah, dctn) with dc_key (ctrl only); resets DCI to RTS on failure, then throws.
    void send(ibv_ah* ah, uint32_t dctn,
              const void* buf, uint32_t lkey, uint32_t len);

    // Endpoints whose failures the initiator may flag; see DcOwner.
    void add_owner(DcOwner* o);
    void remove_owner(DcOwner* o);

    bool valid() const { return qp_ != nullptr; }

private:
    void reset_to_rts();   // bring DCI to RTS; also recovers from ERR state
    // Poll until `wrid` completes; throws on its error or on an earlier
    // unsignaled write's error, flagging the owner that failed.
    void await(uint64_t wrid, const DcTargetInfo& dst, const char* what);
    void flag(uint64_t wr_id);
    uint64_t next_wrid() { return kSignaled | ++wr_seq_; }

    // wr_id of a signaled op; unsignaled writes carry their DcOwner*.
    static constexpr uint64_t kSignaled = 1ull << 63;
    // Unsignaled WQEs are reclaimed only by a later signaled completion.
    static constexpr int kSignalEvery = 32;

    RdmaContext* ctx_ = nullptr;
    ibv_qp*      qp_  = nullptr;
    ibv_qp_ex*   qpx_ = nullptr;
    void*        mqpx_ = nullptr;   // mlx5dv_qp_ex* (opaque here)
    ibv_cq*      cq_  = nullptr;
    uint64_t     dc_key_ = 0;       // stamped on send() (ctrl flavor)
    // reset_to_rts budget, fixed per flavor at construction.
    uint8_t      timeout_ = 0, retry_cnt_ = 0, rnr_retry_ = 0;
    uint64_t     wr_seq_ = 0;
    int          unsignaled_ = 0;
    const char*  who_ = "DcInitiator";
    std::mutex   mu_;
    std::unordered_set<DcOwner*> owners_;
};

// Per-peer IRdmaEndpoint adapter: shared DcInitiator + this peer's DcTargetInfo.
// Keeps the last written cell for the flush fence.
class DcReader : public IRdmaEndpoint {
public:
    DcReader(DcInitiator* dci, DcTargetInfo dst) : dci_(dci), dst_(dst) {
        dst_.owner = &owner_;
        dci_->add_owner(&owner_);
    }
    ~DcReader() override { dci_->remove_owner(&owner_); }

    DcReader(const DcReader&) = delete;
    DcReader& operator=(const DcReader&) = delete;

    void rdma_read(void* local_buf, uint32_t local_lkey,
                   uint64_t remote_addr, uint32_t remote_rkey,
                   uint32_t length) override {
        dci_->rdma_read(dst_, local_buf, local_lkey,
                        remote_addr, remote_rkey, length);
    }
    uint64_t rdma_fetch_add(uint64_t remote_addr, uint32_t remote_rkey,
                            uint64_t add_val, void* res, uint32_t res_lkey) override {
        return dci_->rdma_fetch_add(dst_, remote_addr, remote_rkey,
                                    add_val, res, res_lkey);
    }
    void rdma_read_multi(const RdmaReadOp* ops, int count,
                         uint32_t local_lkey) override {
        dci_->rdma_read_multi(dst_, ops, count, local_lkey);
    }
    void rdma_write_async_inline(uint64_t remote_addr, uint32_t remote_rkey,
                                 const void* local_buf, uint32_t length) override;
    void flush() override;
    bool broken() override { return owner_.failed.load(std::memory_order_acquire); }

private:
    DcInitiator* dci_;
    DcTargetInfo dst_;
    DcOwner      owner_;
    struct {
        uint64_t addr = 0;
        uint32_t rkey = 0, len = 0;
        uint8_t  value[64];   // <= max_inline_data
        bool     dirty = false;
    } last_;
};

// Build an AH toward a peer port (RoCE GRH / IB dlid); caller owns it (ibv_destroy_ah).
ibv_ah* dc_create_ah(RdmaContext& ctx, const ibv_gid& dgid, uint16_t dlid,
                     uint8_t link_layer);

class DcTarget {
public:
    DcTarget(RdmaContext& ctx, uint64_t dc_key, int srq_max_wr = 64);
    ~DcTarget();

    DcTarget(const DcTarget&) = delete;
    DcTarget& operator=(const DcTarget&) = delete;

    uint32_t open_dct();
    void     close_dct(uint32_t dctn);

    // Target-owned RECV pool; consumer MUST repost_recv the wr_id. Caps at SRQ capacity.
    void provision_recv_pool(int count, uint32_t buf_bytes);
    void repost_recv(uint64_t wr_id);
    int  recv_pool_size() const { return (int)pool_.size(); }

    // Wait one RECV completion (want_dctn 0 = any); single drainer only.
    DcRecvCompletion poll(int timeout_ms, uint32_t want_dctn = 0);

    RdmaContext& ctx() const { return *ctx_; }

private:
    int  drain_all();          // recv-only: SENDs complete on initiators' CQs
    bool pump(int timeout_ms);
    void post_recv(void* buf, uint32_t lkey, uint32_t len, uint64_t wr_id);

    RdmaContext* ctx_ = nullptr;
    uint64_t dc_key_ = 0;
    ibv_cq*  cq_  = nullptr;
    ibv_srq* srq_ = nullptr;
    std::vector<std::pair<uint32_t, ibv_qp*>> dcts_;   // (dctn, DCT)

    std::deque<DcRecvCompletion> recv_ready_;
    int wc_err_logged_ = 0;

    // High base so a pool wr_id is recognizable in WC-error logs.
    static constexpr uint64_t POOL_WR_BASE = 0x60000000ULL;
    struct PoolBuf { std::vector<char> mem; RdmaMemoryRegion mr; };
    std::vector<PoolBuf> pool_;
};

} // namespace radixshmem
