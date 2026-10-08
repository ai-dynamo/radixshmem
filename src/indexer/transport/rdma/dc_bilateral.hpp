// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cluster_protocol.h"     // RemoteStatus, RemoteOp* headers
#include "rdma_transport.hpp"        // DcInitiator, DcTarget, dc_create_ah
#include "bilateral_transport.h"   // IRemoteOp{Client,Server}Transport

#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

namespace radixshmem {

// ---- wire ctrl header (48B, 8B-aligned, all fields naturally aligned) ----
constexpr uint32_t DC_CTRL_MAGIC      = 0xDCC0DE01;
constexpr uint16_t DC_CTRL_VERSION    = 1;
constexpr uint16_t DC_CTRL_FLAG_REPLY = 0x1;   // flags bit0: reply direction

struct DcCtrlHeader {
    uint32_t magic;           // DC_CTRL_MAGIC
    uint16_t version;         // DC_CTRL_VERSION
    uint16_t flags;           // bit0 = is_reply
    uint64_t incarnation;     // request: session incarnation; reply: echoed
    uint8_t  reply_gid[16];   // request: requester port GID; reply: 0
    uint32_t reply_dctn;      // request: requester reply DCT; reply: 0
    int32_t  client_rank;     // diagnostics only (server routes by ctrl header)
    uint16_t reply_lid;       // request: requester LID (IB only); reply: 0
    uint8_t  link_layer;      // IBV_LINK_LAYER_* of the requester's port
    uint8_t  _pad[5];
};
static_assert(sizeof(DcCtrlHeader) == 48,
              "DcCtrlHeader layout changed — bump DC_CTRL_VERSION");
static_assert(alignof(DcCtrlHeader) == 8, "DcCtrlHeader alignment changed");

// Max wire sizes: ctrl header + op protocol max req/reply.
constexpr uint32_t kDcCtrlMaxReqWire   = sizeof(DcCtrlHeader) + REMOTE_OP_MAX_REQ_BYTES;
constexpr uint32_t kDcCtrlMaxReplyWire = sizeof(DcCtrlHeader) + REMOTE_OP_MAX_REPLY_BYTES;

// ---- client side: shared initiator+target, per-session reply DCT ----

class DcClientTransport : public IRemoteOpClientTransport {
public:
    DcClientTransport(DcInitiator& ini, DcTarget& tgt, ibv_ah* server_ah,
                      uint32_t server_dctn, int client_rank, int max_retry = 3);
    ~DcClientTransport() override;

    DcClientTransport(const DcClientTransport&) = delete;
    DcClientTransport& operator=(const DcClientTransport&) = delete;

    RemoteStatus send_request(const void* req, size_t len, uint64_t req_id,
                              std::vector<char>& reply_out) override;

private:
    DcInitiator& ini_;          // borrowed (per-process)
    DcTarget&    tgt_;          // borrowed (per-process)
    ibv_ah*      server_ah_;    // owned
    uint32_t     server_dctn_;
    int          client_rank_;
    int          max_retry_;

    uint64_t     incarnation_;  // random nonzero; must init BEFORE the DCT
    uint32_t     reply_dctn_; // this session's reply DCT (demux key)

    // Local port addressing stamped into every request ctrl header.
    ibv_gid  local_gid_{};
    uint16_t local_lid_ = 0;
    uint8_t  local_link_layer_ = 0;

    // Send scratch (one request at a time — strict request/reply).
    std::vector<char> send_buf_;
    RdmaMemoryRegion  send_mr_;
};

// ---- server side: stateless; owns its lane pairs outright ----

class DcServerTransport : public IRemoteOpServerTransport {
public:
    DcServerTransport(RdmaContext& ctx, uint64_t dc_access_key, int lanes,
                      int k_pool = 64);
    ~DcServerTransport() override;

    DcServerTransport(const DcServerTransport&) = delete;
    DcServerTransport& operator=(const DcServerTransport&) = delete;

    int lanes() const override { return (int)lanes_.size(); }
    // Lane's DCT dctn (lane 0 → rc_dct_num, lanes 1..3 → rc_dct_num_extra).
    uint32_t dct_num(int lane) const;

    int  recv_request(std::vector<char>& buf, ReplyContext& ctx_out,
                      int timeout_ms, int lane = 0) override;

    void send_reply(const ReplyContext& ctx, const void* data,
                    size_t len) override;

private:
    struct Lane {
        DcTarget*    tgt  = nullptr;    // → owned_targets_
        DcInitiator* ini  = nullptr;    // → owned_initiators_
        uint32_t     dctn = 0;          // the published request DCT
        std::vector<char> reply_buf;
        RdmaMemoryRegion  reply_mr;
        // gid(16) + lid(2) + link_layer(1) → AH.
        std::map<std::array<uint8_t, 19>, ibv_ah*> ah_cache;
        int hdr_err_logged = 0;
    };

    // Must declare before lanes_ so lanes_ destroys first.
    std::vector<std::unique_ptr<DcTarget>>    owned_targets_;
    std::vector<std::unique_ptr<DcInitiator>> owned_initiators_;
    std::vector<std::unique_ptr<Lane>> lanes_;   // never resized after construction (lanes() unlocked)

    std::atomic<uint64_t> replies_dropped_{0};    // no AH / SEND failed
    std::atomic<uint64_t> hdrs_dropped_{0};  // malformed request ctrl header
};

} // namespace radixshmem
