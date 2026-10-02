// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// nixl_backend — ITransferBackend over nixlAgent (backend="UCX"), compiled as
// a C++20 island (see nixl_backend.hpp). Same validation rules as
// mooncake_backend.cpp; the two are interchangeable behind SlotTransfer.
//
// NIXL differences that shape the design:
//   - Remote metadata does not expose the peer's data-pool base/length, so
//     open_segment() asks for it over NIXL notifications: "BUFREQ" is answered
//     with "BUFRESP:<base>:<len>".
//   - fetchRemoteMD/sendLocalMD are asynchronous with no completion signal, so
//     open_segment() resends them on a timer, sends BUFREQ once checkRemoteMD
//     succeeds, and repeats until BUFRESP arrives or it times out.

#include "nixl_backend.hpp"

#include "../range_merge.hpp"

#include "nixl.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <sched.h>
#include <unistd.h>

namespace shmradix {

namespace {

// How long open_segment() waits for the peer's data-pool geometry before
// giving up. Generous: agent/connection setup on a loaded host can itself
// take a while (UCX worker creation), and this runs once per peer, not once
// per transfer.
constexpr auto kBufGeomTimeout = std::chrono::seconds(30);
constexpr auto kBufReqRetry    = std::chrono::milliseconds(200);
constexpr auto kMdRetry        = std::chrono::seconds(1);
constexpr auto kNotifPollDelay = std::chrono::milliseconds(2);

constexpr const char* kBufReqMsg    = "BUFREQ";
constexpr const char* kBufRespPfx   = "BUFRESP:";

std::string
make_bufresp(uint64_t addr, uint64_t len) {
    return std::string(kBufRespPfx) + std::to_string(addr) + ":" + std::to_string(len);
}

// Returns false if `msg` is not a well-formed "BUFRESP:<addr>:<len>".
bool
parse_bufresp(const std::string& msg, uint64_t& addr, uint64_t& len) {
    const size_t pfx = std::strlen(kBufRespPfx);
    if (msg.size() <= pfx || msg.compare(0, pfx, kBufRespPfx) != 0) return false;
    const size_t sep = msg.find(':', pfx);
    if (sep == std::string::npos) return false;
    try {
        addr = std::stoull(msg.substr(pfx, sep - pfx));
        len  = std::stoull(msg.substr(sep + 1));
    } catch (...) {
        return false;
    }
    return true;
}

class NixlBackend final : public ITransferBackend {
public:
    NixlBackend() = default;
    ~NixlBackend() override;

    bool init(const TransferConfig& cfg);

    int  register_memory(void* addr, uint64_t len, const std::string& location) override;
    int  unregister_memory(void* addr) override;
    int  open_segment(const std::string& seg_name, SegmentInfo& out) override;
    void close_segment(SegmentId id) override;
    const std::string& local_segment() const override { return local_segment_; }

    BatchHandle submit(const TransferDesc* descs, int n) override;
    XferState   poll(BatchHandle h, uint64_t* done_bytes) override;
    XferState   wait(BatchHandle h, int timeout_ms) override;
    int         release(BatchHandle h) override;

    const char* backend_name() const override { return "nixl"; }

private:
    struct Segment {
        SegmentInfo info;
        bool is_self = false;
    };
    struct Batch {
        bool           is_local = false;
        nixlXferReqH*  req      = nullptr;
        uint64_t       bytes    = 0;
        bool           completed_at_post = false;
    };

    bool covers_local(const void* addr, uint64_t len) const;   // mu_ held
    bool own_region(AddrRange& out) const;                     // mu_ held
    void remerge();                                            // mu_ held

    // Split "host:port" (as published via local_segment()). False on a
    // malformed name (no ':', or a non-numeric port).
    static bool split_host_port(const std::string& seg, std::string& host, int& port);

    // Fires sendLocalMD + fetchRemoteMD towards `seg`/`host:port`; does not
    // block for completion.
    void kick_md_exchange(const std::string& seg, const std::string& host, int port);

    // Background thread: the sole caller of agent_->getNotifs(). Answers
    // incoming BUFREQ with our own geometry and files incoming BUFRESP into
    // peer_buf_ for open_segment()'s waiters.
    void notif_loop();

    std::unique_ptr<nixlAgent> agent_;
    std::string local_segment_;
    bool        publish_ = true;

    mutable std::mutex mu_;
    std::vector<AddrRange> registered_;
    std::vector<AddrRange> merged_;
    std::unordered_map<SegmentId, Segment>     open_;
    std::unordered_map<std::string, SegmentId> open_by_name_;
    std::unordered_map<BatchHandle, Batch>     batches_;
    SegmentId   next_segment_ = 1;
    BatchHandle next_batch_   = 1;

    std::thread             notif_thread_;
    std::atomic<bool>       stop_notif_{false};
    std::mutex              notif_mu_;
    std::condition_variable notif_cv_;
    std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> peer_buf_;  // peer -> (base,len)
};

NixlBackend::~NixlBackend() {
    stop_notif_.store(true, std::memory_order_relaxed);
    if (notif_thread_.joinable()) notif_thread_.join();

    if (!agent_) return;
    for (const auto& kv : batches_) {
        if (kv.second.is_local || kv.second.req == nullptr) continue;
        if (agent_->getXferStatus(kv.second.req) == NIXL_IN_PROG) {
            fprintf(stderr, "[shmradix] NixlBackend: batch %lu still in flight at "
                            "teardown; leaking its request handle\n",
                    (unsigned long)kv.first);
            continue;
        }
        agent_->releaseXferReq(kv.second.req);
    }
    for (const auto& r : registered_) {
        nixl_reg_dlist_t descs(DRAM_SEG);
        descs.addDesc(nixlBlobDesc(r.addr, r.len, 0, ""));
        agent_->deregisterMem(descs);
    }
}

bool NixlBackend::split_host_port(const std::string& seg, std::string& host, int& port) {
    const size_t pos = seg.rfind(':');
    if (pos == std::string::npos || pos == 0 || pos + 1 >= seg.size()) return false;
    host = seg.substr(0, pos);
    try {
        port = std::stoi(seg.substr(pos + 1));
    } catch (...) {
        return false;
    }
    return port > 0 && port <= 65535;
}

bool NixlBackend::init(const TransferConfig& cfg) {
    publish_ = cfg.publish;

    std::string want = cfg.local_segment;
    if (want.empty()) {
        if (cfg.ip_or_host.empty()) {
            fprintf(stderr, "[shmradix] NixlBackend: no local_segment and no "
                            "ip_or_host — cannot pick an address to publish\n");
            return false;
        }
        want = cfg.ip_or_host + ":" + std::to_string(cfg.rpc_port);
    }
    std::string self_host;
    int self_port = 0;
    if (!split_host_port(want, self_host, self_port)) {
        fprintf(stderr, "[shmradix] NixlBackend: local segment '%s' is not a "
                        "'host:port' address\n", want.c_str());
        return false;
    }
    if (self_port == 0) {
        // Unlike Mooncake's engine, NIXL's P2P listener gives no way to read
        // back an OS-assigned ephemeral port after binding — whatever we
        // publish as local_segment() must be the literal port we bound, or
        // peers connect to the wrong place. rpc_port=0 ("engine picks") is a
        // Mooncake-only convenience; the nixl backend needs an explicit port.
        fprintf(stderr, "[shmradix] NixlBackend: rpc_port=0 ('engine picks') is "
                        "not supported — pass an explicit --transfer-port for "
                        "the nixl backend\n");
        return false;
    }
    local_segment_ = want;

    const nixlAgentConfig agent_cfg(/*use_prog_thread=*/false,
                                    /*use_listen_thread=*/true,
                                    /*port=*/static_cast<uint16_t>(self_port),
                                    /*sync_mode=*/nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT);
    try {
        agent_.reset(new nixlAgent(local_segment_, agent_cfg));
    } catch (const std::exception& e) {
        fprintf(stderr, "[shmradix] NixlBackend: nixlAgent('%s') failed: %s\n",
                local_segment_.c_str(), e.what());
        agent_.reset();
        return false;
    }

    nixl_b_params_t params;
    if (!cfg.device_filter.empty()) {
        std::string joined;
        for (size_t i = 0; i < cfg.device_filter.size(); ++i) {
            if (i) joined += ", ";
            joined += cfg.device_filter[i];
        }
        params["device_list"] = joined;
    }
    nixlBackendH* backend = nullptr;
    const nixl_status_t st = agent_->createBackend("UCX", params, backend);
    if (st != NIXL_SUCCESS || backend == nullptr) {
        fprintf(stderr, "[shmradix] NixlBackend: createBackend(UCX) failed: %d\n", (int)st);
        agent_.reset();
        return false;
    }

    notif_thread_ = std::thread([this] { notif_loop(); });
    return true;
}

void NixlBackend::notif_loop() {
    while (!stop_notif_.load(std::memory_order_relaxed)) {
        nixl_notifs_t notifs;
        const nixl_status_t st = agent_->getNotifs(notifs);
        if (st == NIXL_SUCCESS) {
            for (auto& kv : notifs) {
                const std::string& peer = kv.first;
                for (auto& msg : kv.second) {
                    if (msg == kBufReqMsg) {
                        AddrRange r{};
                        bool have;
                        {
                            std::lock_guard<std::mutex> lk(mu_);
                            have = own_region(r);
                        }
                        if (have && publish_) agent_->genNotif(peer, make_bufresp(r.addr, r.len));
                        continue;
                    }
                    uint64_t addr = 0, len = 0;
                    if (parse_bufresp(msg, addr, len)) {
                        {
                            std::lock_guard<std::mutex> lk(notif_mu_);
                            peer_buf_[peer] = {addr, len};
                        }
                        notif_cv_.notify_all();
                    }
                }
            }
        }
        std::this_thread::sleep_for(kNotifPollDelay);
    }
}

int NixlBackend::register_memory(void* addr, uint64_t len, const std::string& /*location*/) {
    if (addr == nullptr || len == 0) return XFER_EINVAL;
    if (!agent_) return XFER_EBACKEND;

    const AddrRange r{reinterpret_cast<uint64_t>(addr), len};
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (const auto& e : registered_)
            if (e.addr == r.addr) return XFER_EINVAL;
    }

    nixl_reg_dlist_t descs(DRAM_SEG);
    descs.addDesc(nixlBlobDesc(r.addr, r.len, 0, ""));
    const nixl_status_t st = agent_->registerMem(descs);
    if (st != NIXL_SUCCESS) {
        fprintf(stderr, "[shmradix] NixlBackend: registerMem(%p, %lu) failed: %d\n",
                addr, (unsigned long)len, (int)st);
        return XFER_EBACKEND;
    }

    std::lock_guard<std::mutex> lk(mu_);
    registered_.push_back(r);
    remerge();
    return XFER_OK;
}

int NixlBackend::unregister_memory(void* addr) {
    if (!agent_) return XFER_EBACKEND;
    const uint64_t a = reinterpret_cast<uint64_t>(addr);

    std::lock_guard<std::mutex> lk(mu_);
    auto i = registered_.begin();
    for (; i != registered_.end(); ++i) if (i->addr == a) break;
    if (i == registered_.end()) return XFER_ENOENT;

    nixl_reg_dlist_t descs(DRAM_SEG);
    descs.addDesc(nixlBlobDesc(i->addr, i->len, 0, ""));
    if (agent_->deregisterMem(descs) != NIXL_SUCCESS) return XFER_EBACKEND;
    registered_.erase(i);
    remerge();
    return XFER_OK;
}

void NixlBackend::remerge() {
    merged_ = merge_contiguous(registered_);
}

bool NixlBackend::own_region(AddrRange& out) const {
    if (merged_.size() != 1) return false;
    out = merged_[0];
    return true;
}

bool NixlBackend::covers_local(const void* addr, uint64_t len) const {
    const uint64_t a = reinterpret_cast<uint64_t>(addr);
    uint64_t end = 0;
    if (__builtin_add_overflow(a, len, &end)) return false;
    for (const auto& r : merged_)
        if (a >= r.addr && end <= r.end()) return true;
    return false;
}

void NixlBackend::kick_md_exchange(const std::string& seg, const std::string& host, int port) {
    nixl_opt_args_t xa;
    xa.ipAddr = host;
    xa.port   = port;
    agent_->sendLocalMD(&xa);
    agent_->fetchRemoteMD(seg, &xa);
}

int NixlBackend::open_segment(const std::string& seg_name, SegmentInfo& out) {
    if (seg_name.empty()) return XFER_EINVAL;
    if (!agent_) return XFER_EBACKEND;

    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = open_by_name_.find(seg_name);
        if (it != open_by_name_.end()) { out = open_[it->second].info; return XFER_OK; }
    }

    Segment seg;

    if (seg_name == local_segment_) {
        // Our own segment: same-process loopback, no NIXL round trip needed
        // (and NIXL P2P would fail trying to connect a listener to itself).
        std::lock_guard<std::mutex> lk(mu_);
        AddrRange r{};
        if (!own_region(r)) {
            fprintf(stderr, "[shmradix] NixlBackend: own segment is not one "
                            "contiguous registered region\n");
            return XFER_EGEOMETRY;
        }
        seg.is_self     = true;
        seg.info.base   = r.addr;
        seg.info.length = r.len;
    } else {
        std::string host;
        int port = 0;
        if (!split_host_port(seg_name, host, port)) {
            fprintf(stderr, "[shmradix] NixlBackend: peer segment '%s' is not a "
                            "'host:port' address\n", seg_name.c_str());
            return XFER_EINVAL;
        }

        {
            std::lock_guard<std::mutex> lk(notif_mu_);
            peer_buf_.erase(seg_name);
        }
        const auto deadline = std::chrono::steady_clock::now() + kBufGeomTimeout;
        auto next_kick = std::chrono::steady_clock::now();
        auto next_md   = next_kick;
        uint64_t base = 0, len = 0;
        bool got = false;
        while (std::chrono::steady_clock::now() < deadline) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= next_md) {
                kick_md_exchange(seg_name, host, port);
                next_md = now + kMdRetry;
            }
            if (now >= next_kick) {
                if (agent_->checkRemoteMD(seg_name, nixl_xfer_dlist_t(DRAM_SEG)) == NIXL_SUCCESS)
                    agent_->genNotif(seg_name, kBufReqMsg);
                next_kick = now + kBufReqRetry;
            }
            std::unique_lock<std::mutex> lk(notif_mu_);
            got = notif_cv_.wait_until(lk, next_kick, [&] {
                return peer_buf_.count(seg_name) != 0;
            });
            if (got) {
                std::tie(base, len) = peer_buf_[seg_name];
                peer_buf_.erase(seg_name);
                break;
            }
        }
        if (!got) {
            fprintf(stderr, "[shmradix] NixlBackend: open_segment('%s') timed out "
                            "waiting for peer buffer geometry after %lds\n",
                    seg_name.c_str(), (long)kBufGeomTimeout.count());
            return XFER_ENOENT;
        }
        seg.is_self     = false;
        seg.info.base   = base;
        seg.info.length = len;
    }

    std::lock_guard<std::mutex> lk(mu_);
    seg.info.id             = next_segment_++;
    open_[seg.info.id]      = seg;
    open_by_name_[seg_name] = seg.info.id;
    out = seg.info;
    return XFER_OK;
}

void NixlBackend::close_segment(SegmentId id) {
    std::string seg_name;
    bool remote = false;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = open_.find(id);
        if (it == open_.end()) return;
        remote = !it->second.is_self;
        for (auto& kv : open_by_name_)
            if (kv.second == id) { seg_name = kv.first; break; }
        open_.erase(it);
        if (!seg_name.empty()) open_by_name_.erase(seg_name);
    }
    if (remote && !seg_name.empty() && agent_) agent_->invalidateRemoteMD(seg_name);
}

BatchHandle NixlBackend::submit(const TransferDesc* descs, int n) {
    if (descs == nullptr || n <= 0 || !agent_) return INVALID_BATCH;

    std::unique_lock<std::mutex> lk(mu_);

    uint64_t total = 0;
    bool is_self = false;
    SegmentId target = INVALID_SEGMENT;
    XferOp op = descs[0].op;
    for (int i = 0; i < n; ++i) {
        const TransferDesc& d = descs[i];
        if (d.length == 0 || d.local_addr == nullptr) {
            fprintf(stderr, "[shmradix] NixlBackend: desc %d has null addr or zero length\n", i);
            return INVALID_BATCH;
        }
        if (!covers_local(d.local_addr, d.length)) {
            fprintf(stderr, "[shmradix] NixlBackend: desc %d local range [%p,+%lu) "
                            "is not inside any registered region\n",
                    i, d.local_addr, (unsigned long)d.length);
            return INVALID_BATCH;
        }
        auto it = open_.find(d.target);
        if (it == open_.end()) {
            fprintf(stderr, "[shmradix] NixlBackend: desc %d targets unknown segment %d\n",
                    i, (int)d.target);
            return INVALID_BATCH;
        }
        const Segment& seg = it->second;
        uint64_t rend = 0;
        if (__builtin_add_overflow(d.target_offset, d.length, &rend) ||
            d.target_offset < seg.info.base || rend > seg.info.base + seg.info.length) {
            fprintf(stderr, "[shmradix] NixlBackend: desc %d remote range "
                            "[0x%lx,+%lu) escapes segment %d [0x%lx,+%lu)\n",
                    i, (unsigned long)d.target_offset, (unsigned long)d.length,
                    (int)d.target, (unsigned long)seg.info.base,
                    (unsigned long)seg.info.length);
            return INVALID_BATCH;
        }
        if (i == 0) { is_self = seg.is_self; target = d.target; }
        else if (is_self != seg.is_self || target != d.target) {
            fprintf(stderr, "[shmradix] NixlBackend: batch targets more than one "
                            "segment, or mixes the local segment with a remote one; "
                            "split it\n");
            return INVALID_BATCH;
        }
        if (d.op != op) {
            fprintf(stderr, "[shmradix] NixlBackend: batch mixes READ and WRITE; "
                            "split it (NIXL's createXferReq takes one op per batch)\n");
            return INVALID_BATCH;
        }
        total += d.length;
    }

    if (is_self) {
        for (int i = 0; i < n; ++i) {
            const TransferDesc& d = descs[i];
            void* remote = reinterpret_cast<void*>(d.target_offset);
            if (d.op == XferOp::READ) std::memcpy(d.local_addr, remote, d.length);
            else                      std::memcpy(remote, d.local_addr, d.length);
        }
        const BatchHandle h = next_batch_++;
        batches_[h] = Batch{true, nullptr, total, true};
        return h;
    }

    // Reverse-lookup the target's segment name: createXferReq wants the peer's
    // agent name, which is exactly what we published it under.
    std::string remote_agent;
    for (const auto& kv : open_by_name_) {
        if (kv.second == target) { remote_agent = kv.first; break; }
    }
    if (remote_agent.empty()) {
        fprintf(stderr, "[shmradix] NixlBackend: segment %d has no name on record\n",
                (int)target);
        return INVALID_BATCH;
    }

    nixl_xfer_dlist_t local_descs(DRAM_SEG);
    nixl_xfer_dlist_t remote_descs(DRAM_SEG);
    for (int i = 0; i < n; ++i) {
        local_descs.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(descs[i].local_addr),
                                          descs[i].length, 0));
        remote_descs.addDesc(nixlBasicDesc(descs[i].target_offset, descs[i].length, 0));
    }
    const nixl_xfer_op_t nixl_op = (op == XferOp::READ) ? NIXL_READ : NIXL_WRITE;

    lk.unlock();
    nixlXferReqH* req = nullptr;
    nixl_status_t st = agent_->createXferReq(nixl_op, local_descs, remote_descs,
                                             remote_agent, req);
    if (st != NIXL_SUCCESS || req == nullptr) {
        fprintf(stderr, "[shmradix] NixlBackend: createXferReq to '%s' failed: %d\n",
                remote_agent.c_str(), (int)st);
        return INVALID_BATCH;
    }
    st = agent_->postXferReq(req);
    if (st != NIXL_SUCCESS && st != NIXL_IN_PROG) {
        fprintf(stderr, "[shmradix] NixlBackend: postXferReq to '%s' failed: %d\n",
                remote_agent.c_str(), (int)st);
        agent_->releaseXferReq(req);
        return INVALID_BATCH;
    }
    lk.lock();

    const BatchHandle h = next_batch_++;
    batches_[h] = Batch{false, req, total, st == NIXL_SUCCESS};
    return h;
}

XferState NixlBackend::poll(BatchHandle h, uint64_t* done_bytes) {
    Batch b;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = batches_.find(h);
        if (it == batches_.end()) return XferState::INVALID;
        b = it->second;
    }
    if (b.is_local) {
        if (done_bytes) *done_bytes = b.bytes;
        return XferState::COMPLETED;
    }
    if (b.completed_at_post) {
        if (done_bytes) *done_bytes = b.bytes;
        return XferState::COMPLETED;
    }

    const nixl_status_t st = agent_->getXferStatus(b.req);
    if (st == NIXL_SUCCESS) {
        if (done_bytes) *done_bytes = b.bytes;
        return XferState::COMPLETED;
    }
    if (st == NIXL_IN_PROG) {
        if (done_bytes) *done_bytes = 0;
        return XferState::PENDING;
    }
    if (done_bytes) *done_bytes = 0;
    return XferState::FAILED;
}

XferState NixlBackend::wait(BatchHandle h, int timeout_ms) {
    const auto start = std::chrono::steady_clock::now();
    for (int spin = 0;; ++spin) {
        const XferState s = poll(h, nullptr);
        if (s != XferState::PENDING) return s;
        if (timeout_ms >= 0) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();
            if (elapsed >= timeout_ms) return XferState::TIMEOUT;
        }
        if (spin < 100) sched_yield();
        else            ::usleep(50);
    }
}

int NixlBackend::release(BatchHandle h) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = batches_.find(h);
    if (it == batches_.end()) return XFER_ENOENT;
    if (it->second.is_local || it->second.req == nullptr) { batches_.erase(it); return XFER_OK; }
    if (agent_->getXferStatus(it->second.req) == NIXL_IN_PROG) return XFER_EBUSY;

    const nixl_status_t st = agent_->releaseXferReq(it->second.req);
    if (st != NIXL_SUCCESS) {
        fprintf(stderr, "[shmradix] NixlBackend: releaseXferReq failed: %d\n", (int)st);
        return XFER_EBACKEND;
    }
    batches_.erase(it);
    return XFER_OK;
}

} // namespace

std::unique_ptr<ITransferBackend> create_nixl_backend(const TransferConfig& cfg) {
    auto be = std::unique_ptr<NixlBackend>(new NixlBackend());
    if (!be->init(cfg)) return nullptr;
    return be;
}

} // namespace shmradix
