// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// DC (DCI/DCT) bilateral backend for the remote-op control plane.
// See dc_bilateral.hpp and docs/remote_op_dc_transport_design.md.

#include "dc_bilateral.hpp"

#include "cluster_manifest.h"   // PeerEntry

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>

namespace radixshmem {

namespace {

// Per-handle incarnation: random nonzero 64-bit guarding dctn reuse via echo check.
uint64_t make_incarnation() {
    std::random_device rd;
    uint64_t v;
    do {
        v = ((uint64_t)rd() << 32) ^ (uint64_t)rd();
    } while (v == 0);
    return v;
}

}  // namespace

// ==================== DcClientTransport ====================

DcClientTransport::DcClientTransport(DcInitiator& ini, DcTarget& tgt,
                                     ibv_ah* server_ah, uint32_t server_dctn,
                                     int client_rank,
                                     std::shared_ptr<std::mutex> stack_mu,
                                     int max_retry)
    : WireRemoteOpClientTransport(std::move(stack_mu)), ini_(ini), tgt_(tgt), server_ah_(server_ah), server_dctn_(server_dctn),
      client_rank_(client_rank), max_retry_(max_retry),
      incarnation_(make_incarnation()), reply_dctn_(tgt.open_dct()) {
    try {
        // Local port addressing for the request ctrl header (the reply route).
        RdmaContext& ctx = tgt_.ctx();
        ibv_port_attr pa{};
        if (ibv_query_port(ctx.ctx(), ctx.port(), &pa) != 0)
            throw std::runtime_error("DcClientTransport: ibv_query_port failed");
        local_gid_        = ctx.gid();
        local_lid_        = pa.lid;
        local_link_layer_ = (uint8_t)pa.link_layer;

        send_buf_.resize(kDcCtrlMaxReqWire);
        send_mr_ = RdmaMemoryRegion(ctx.pd(), send_buf_.data(),
                                    send_buf_.size(), IBV_ACCESS_LOCAL_WRITE);
        if (tgt_.recv_pool_size() == 0)
            tgt_.provision_recv_pool(/*count=*/8, kDcCtrlMaxReplyWire);
    } catch (...) {
        tgt_.close_dct(reply_dctn_);
        if (server_ah_) ibv_destroy_ah(server_ah_);
        throw;
    }
}

DcClientTransport::~DcClientTransport() {
    auto g = guard();
    if (server_ah_) ibv_destroy_ah(server_ah_);
    tgt_.close_dct(reply_dctn_);
}

RemoteStatus DcClientTransport::send_request(const void* req, size_t len,
                                             uint64_t req_id,
                                             std::vector<char>& reply_out) {
    if (sizeof(DcCtrlHeader) + len > send_buf_.size())
        return RemoteStatus::BAD_REPLY;   // caller bug

    // Request ctrl header: the reply route + this session's incarnation.
    DcCtrlHeader hdr{};
    hdr.magic       = DC_CTRL_MAGIC;
    hdr.version     = DC_CTRL_VERSION;
    hdr.flags       = 0;
    hdr.incarnation = incarnation_;
    std::memcpy(hdr.reply_gid, &local_gid_, 16);
    hdr.reply_dctn  = reply_dctn_;
    hdr.client_rank = client_rank_;
    hdr.reply_lid   = local_lid_;
    hdr.link_layer  = local_link_layer_;

    const uint32_t wire_len = (uint32_t)(sizeof(hdr) + len);

    for (int attempt = 0; attempt < max_retry_; ++attempt) {
        std::memcpy(send_buf_.data(), &hdr, sizeof(hdr));
        std::memcpy(send_buf_.data() + sizeof(hdr), req, len);
        try {
            ini_.send(server_ah_, server_dctn_,
                      send_buf_.data(), send_mr_.lkey(), wire_len);
        } catch (const std::exception&) {
            continue;
        }

        constexpr int kReplyTimeoutMs = 2000;
        auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(kReplyTimeoutMs);
        for (;;) {
            int left = (int)std::chrono::duration_cast<std::chrono::milliseconds>(
                           deadline - std::chrono::steady_clock::now()).count();
            if (left <= 0) break;   // timeout → outer retry
            DcRecvCompletion c = tgt_.poll(left, reply_dctn_);
            if (!c.ok) break;       // timeout → outer retry

            bool valid = false;
            if (c.data &&
                c.byte_len >= sizeof(DcCtrlHeader) + sizeof(RemoteOpReplyHeader)) {
                auto* re = reinterpret_cast<const DcCtrlHeader*>(c.data);
                auto* rh = reinterpret_cast<const RemoteOpReplyHeader*>(
                               c.data + sizeof(DcCtrlHeader));
                if (re->magic == DC_CTRL_MAGIC &&
                    re->version == DC_CTRL_VERSION &&
                    (re->flags & DC_CTRL_FLAG_REPLY) &&
                    re->incarnation == incarnation_ &&
                    rh->magic == REMOTE_OP_REPLY_MAGIC &&
                    rh->version == REMOTE_OP_PROTOCOL_VERSION &&
                    rh->req_id == req_id) {
                    reply_out.assign(c.data + sizeof(DcCtrlHeader),
                                     c.data + c.byte_len);
                    valid = true;
                }
            }
            tgt_.repost_recv(c.wr_id);   // consumed → repost, regardless
            if (valid) return RemoteStatus::OK;
        }
        // timeout on this attempt → outer loop retries (re-send)
    }
    return RemoteStatus::TIMEOUT;
}

// ==================== DcServerTransport ====================

DcServerTransport::DcServerTransport(RdmaContext& ctx, uint64_t dc_access_key,
                                     int lanes, int k_pool) {
    // Per lane: own DcTarget + published DCT + reply DcInitiator; lane-0 DCT
    // doubles as the DC data-plane read target.
    if (lanes < 1) lanes = 1;
    if (lanes > 4) {
        fprintf(stderr, "[DcServerTransport] lanes=%d exceeds the manifest's "
                "4-lane cap — clamping\n", lanes);
        lanes = 4;
    }
    for (int i = 0; i < lanes; ++i) {
        auto tgt = std::make_unique<DcTarget>(ctx, dc_access_key, k_pool);
        auto ini = std::make_unique<DcInitiator>(ctx, dc_access_key);
        auto L   = std::make_unique<Lane>();
        L->tgt  = tgt.get();
        L->ini  = ini.get();
        L->dctn = tgt->open_dct();
        tgt->provision_recv_pool(k_pool, kDcCtrlMaxReqWire);
        L->reply_buf.resize(kDcCtrlMaxReplyWire);
        L->reply_mr = RdmaMemoryRegion(ctx.pd(), L->reply_buf.data(),
                                       L->reply_buf.size(),
                                       IBV_ACCESS_LOCAL_WRITE);
        lanes_.push_back(std::move(L));
        owned_targets_.push_back(std::move(tgt));
        owned_initiators_.push_back(std::move(ini));
    }
    fprintf(stderr, "[DcServerTransport] control-plane target up: lanes=%d "
            "pool=%d/lane\n", lanes, k_pool);
}

uint32_t DcServerTransport::dct_num(int lane) const {
    return (lane >= 0 && lane < (int)lanes_.size()) ? lanes_[lane]->dctn : 0;
}

DcServerTransport::~DcServerTransport() {
    for (auto& L : lanes_)
        for (auto& kv : L->ah_cache)
            if (kv.second) ibv_destroy_ah(kv.second);
    uint64_t rd = replies_dropped_.load(std::memory_order_relaxed);
    uint64_t ed = hdrs_dropped_.load(std::memory_order_relaxed);
    if (rd || ed)
        fprintf(stderr, "[DcServerTransport] dropped: replies=%llu (dead "
                "target / no AH) bad_ctrl_hdrs=%llu\n",
                (unsigned long long)rd, (unsigned long long)ed);
}

int DcServerTransport::recv_request(std::vector<char>& buf,
                                    ReplyContext& ctx_out, int timeout_ms,
                                    int lane) {
    Lane& L = *lanes_[lane];
    DcRecvCompletion c = L.tgt->poll(timeout_ms);
    if (!c.ok) return -1;

    if (!c.data) {
        fprintf(stderr, "[DcServerTransport] dropping non-pool RECV "
                "(dctn=%u wr_id=%llu len=%u)\n",
                c.dctn, (unsigned long long)c.wr_id, c.byte_len);
        return -1;
    }

    // Transport-level ctrl-header validation (op-header validation is the poller's).
    const auto* hdr = reinterpret_cast<const DcCtrlHeader*>(c.data);
    bool env_ok = c.byte_len >= sizeof(DcCtrlHeader) &&
                  hdr->magic == DC_CTRL_MAGIC &&
                  hdr->version == DC_CTRL_VERSION &&
                  (hdr->flags & DC_CTRL_FLAG_REPLY) == 0 &&
                  hdr->incarnation != 0 &&
                  hdr->reply_dctn != 0;
    if (!env_ok) {
        hdrs_dropped_.fetch_add(1, std::memory_order_relaxed);
        if (L.hdr_err_logged < 8) {
            fprintf(stderr, "[DcServerTransport] dropping bad ctrl header "
                    "(len=%u)%s\n", c.byte_len,
                    ++L.hdr_err_logged == 8 ? " (suppressing further)" : "");
        }
        L.tgt->repost_recv(c.wr_id);
        return -1;
    }

    // Copy op payload without the ctrl header, then repost to repay credit.
    const char* payload = c.data + sizeof(DcCtrlHeader);
    uint32_t payload_len = c.byte_len - (uint32_t)sizeof(DcCtrlHeader);
    buf.assign(payload, payload + payload_len);

    ctx_out = ReplyContext{};
    ctx_out.lane = (uint16_t)lane;   // send_reply routes via this lane's DCI
    std::memcpy(ctx_out.gid, hdr->reply_gid, 16);
    ctx_out.dct_num     = hdr->reply_dctn;
    ctx_out.incarnation = hdr->incarnation;
    ctx_out.lid         = hdr->reply_lid;
    ctx_out.link_layer  = hdr->link_layer;

    L.tgt->repost_recv(c.wr_id);
    return (int)payload_len;
}

void DcServerTransport::send_reply(const ReplyContext& ctx,
                                   const void* data, size_t len) {
    if (ctx.lane >= lanes_.size()) return;   // corrupt ctx → drop, never OOB
    Lane& L = *lanes_[ctx.lane];
    if (sizeof(DcCtrlHeader) + len > L.reply_buf.size())
        throw std::runtime_error("DcServerTransport: reply exceeds buffer");

    // AH cache: gid + lid + link_layer → ibv_ah. Poller-j-only, no lock.
    std::array<uint8_t, 19> key{};
    std::memcpy(key.data(), ctx.gid, 16);
    key[16] = (uint8_t)(ctx.lid & 0xff);
    key[17] = (uint8_t)(ctx.lid >> 8);
    key[18] = ctx.link_layer;
    ibv_ah* ah = nullptr;
    auto it = L.ah_cache.find(key);
    if (it != L.ah_cache.end()) {
        ah = it->second;
    } else {
        ibv_gid dgid;
        std::memcpy(&dgid, ctx.gid, 16);
        ah = dc_create_ah(L.tgt->ctx(), dgid, ctx.lid, ctx.link_layer);
        if (!ah) {
            replies_dropped_.fetch_add(1, std::memory_order_relaxed);
            return;   // unroutable — peer times out and retries
        }
        L.ah_cache.emplace(key, ah);
    }

    auto* hdr = reinterpret_cast<DcCtrlHeader*>(L.reply_buf.data());
    *hdr = {};
    hdr->magic       = DC_CTRL_MAGIC;
    hdr->version     = DC_CTRL_VERSION;
    hdr->flags       = DC_CTRL_FLAG_REPLY;
    hdr->incarnation = ctx.incarnation;   // echoed — the client-side guard
    hdr->client_rank = -1;
    std::memcpy(L.reply_buf.data() + sizeof(DcCtrlHeader), data, len);

    try {
        L.ini->send(ah, ctx.dct_num, L.reply_buf.data(),
                    L.reply_mr.lkey(),
                    (uint32_t)(sizeof(DcCtrlHeader) + len));
    } catch (const std::exception&) {
        replies_dropped_.fetch_add(1, std::memory_order_relaxed);
    }
}

// ==================== construction ====================

namespace {

int rc_resolve_lanes() {
    int lanes = 1;
    if (const char* e = ::getenv("RADIXSHMEM_RC_LANES")) {
        int v = std::atoi(e);
        if (v >= 1 && v <= 8) lanes = v;
    }
    return lanes;
}

}  // namespace

DcClientStack::DcClientStack(RdmaContext& ctx, uint64_t dc_access_key)
    : ctx_(ctx), dc_access_key_(dc_access_key) {}

DcClientStack::~DcClientStack() {
    if (stack_mu_.use_count() > 1) {
        fprintf(stderr,
                "[PeerConnector] WARNING: %ld dc remote-op session(s) (or "
                "retained lock=true results) outlive this client — their "
                "destruction is use-after-free. Drop all RemotePeer handles AND "
                "locked QueryResults before the RadixClient.\n",
                (long)(stack_mu_.use_count() - 1));
    }
    ini_.reset();
    tgt_.reset();
}

std::unique_ptr<DcClientTransport> DcClientStack::open(const PeerEntry& p,
                                                     int client_rank,
                                                     std::string& endpoint) {
    if (p.rc_dct_num == 0)
        throw std::runtime_error(
            "open_remote: peer rank " + std::to_string(p.rank) +
            " has remote ops disabled (rc_dct_num == 0)");
    std::lock_guard<std::mutex> g(*stack_mu_);
    if (!tgt_) {
        auto tgt = std::make_unique<DcTarget>(
            ctx_, dc_access_key_, /*srq_max_wr=*/16);
        auto ini = std::make_unique<DcInitiator>(ctx_, dc_access_key_);
        tgt_ = std::move(tgt);
        ini_ = std::move(ini);
    }
    int p_eff = 1;
    for (int i = 0; i < 3 && p.rc_dct_num_extra[i] != 0; ++i) ++p_eff;
    int lane = (int)((unsigned)client_rank % (unsigned)p_eff);
    uint32_t dctn = lane == 0 ? p.rc_dct_num : p.rc_dct_num_extra[lane - 1];
    ibv_gid dgid;
    std::memcpy(&dgid, p.gid_raw, 16);
    ibv_ah* ah = dc_create_ah(ctx_, dgid, p.lid, p.link_layer);
    if (!ah)
        throw std::runtime_error(
            "open_remote: ibv_create_ah to rank " +
            std::to_string(p.rank) + " failed");
    endpoint = "dc://" + std::string(p.host) + "#" + std::to_string(dctn);
    return std::make_unique<DcClientTransport>(
        *ini_, *tgt_, ah, dctn, client_rank, stack_mu_);
}

std::unique_ptr<DcServerTransport> make_dc_server_transport(RdmaContext& ctx,
                                                            uint64_t dc_access_key) {
    int k_pool = 64;
    if (const char* e = ::getenv("RADIXSHMEM_RC_KPOOL")) {
        int v = std::atoi(e);
        if (v >= 1 && v <= 4096) k_pool = v;
    }
    return std::make_unique<DcServerTransport>(
        ctx, dc_access_key, std::min(rc_resolve_lanes(), 4), k_pool);
}

} // namespace radixshmem
