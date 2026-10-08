// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// mooncake_backend — ITransferBackend over mooncake::TransferEngine.
//
// Compiled as a C++20 island (see mooncake_backend.hpp). It enforces the
// registration, geometry and batch-validation rules the abstraction promises;
// test_mooncake_loopback is where those rules are pinned down.
//
// Two Mooncake specifics drive the design:
//
//   - TransferRequest.target_offset is the peer's *absolute* virtual address,
//     not an offset into its segment. The abstraction keeps that meaning, so
//     addresses are computed once, by whoever knows the peer's base.
//   - A registration longer than max_mr_size is published as several adjacent
//     BufferDescs (rdma_transport.cpp:303). Reading only buffers[0] would
//     silently address a fraction of a large store, so they are fused.

#include "mooncake_backend.hpp"

#include "../range_merge.hpp"

#include "transfer_engine.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <sched.h>
#include <unistd.h>

namespace radixshmem {

namespace {

XferState from_mooncake(mooncake::TransferStatusEnum s) {
    using E = mooncake::TransferStatusEnum;
    switch (s) {
        case E::COMPLETED: return XferState::COMPLETED;
        case E::FAILED:    return XferState::FAILED;
        case E::TIMEOUT:   return XferState::TIMEOUT;
        // A cancelled batch will never complete; surfacing it as anything other
        // than a failure would hang the caller's wait loop.
        case E::CANCELED:  return XferState::FAILED;
        case E::INVALID:   return XferState::INVALID;
        case E::WAITING:
        case E::PENDING:   return XferState::PENDING;
    }
    return XferState::INVALID;
}

class MooncakeBackend final : public ITransferBackend {
public:
    MooncakeBackend() = default;
    ~MooncakeBackend() override;

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

    const char* backend_name() const override { return "mooncake"; }

private:
    struct Segment {
        SegmentInfo info;
        mooncake::SegmentHandle handle = 0;
        bool is_self = false;   // same process: memcpy, no QP roundtrip
    };
    struct Batch {
        bool              is_local = false;
        mooncake::BatchID id       = mooncake::INVALID_BATCH_ID;
        uint64_t          bytes    = 0;
    };

    // All three want mu_ held.
    bool covers_local(const void* addr, uint64_t len) const;
    // Merged view of everything this process registered. Used both to validate
    // local addresses and to describe our own segment to ourselves.
    bool own_region(AddrRange& out) const;
    // Ends every mutation of registered_. Fusing here rather than per
    // descriptor keeps submit()'s validation loop allocation-free.
    void remerge();

    std::unique_ptr<mooncake::TransferEngine> engine_;
    std::string local_segment_;
    bool        publish_ = true;

    mutable std::mutex mu_;
    std::vector<AddrRange> registered_;   // as registered, for unregistering
    std::vector<AddrRange> merged_;       // sorted, disjoint, fused
    std::unordered_map<SegmentId, Segment>     open_;
    std::unordered_map<std::string, SegmentId> open_by_name_;
    std::unordered_map<BatchHandle, Batch>     batches_;
    SegmentId   next_segment_ = 1;
    BatchHandle next_batch_   = 1;
};

MooncakeBackend::~MooncakeBackend() {
    if (!engine_) return;
    // Batches the caller never released; the finished ones are still freeable.
    for (const auto& kv : batches_) {
        if (kv.second.is_local) continue;
        if (!engine_->freeBatchID(kv.second.id).ok()) {
            fprintf(stderr, "[radixshmem] MooncakeBackend: batch %lu still in flight at "
                            "teardown; leaking its descriptor\n",
                    (unsigned long)kv.first);
        }
    }
    for (const auto& r : registered_)
        engine_->unregisterLocalMemory(reinterpret_cast<void*>(r.addr));
    for (const auto& kv : open_) {
        if (!kv.second.is_self) engine_->closeSegment(kv.second.handle);
    }
}

bool MooncakeBackend::init(const TransferConfig& cfg) {
    publish_ = cfg.publish;

    const bool auto_discover = cfg.protocol.empty() || cfg.protocol == "rdma";
    // The engine intersects this allowlist with its own MC_TE_FILTERS env
    // allowlist, so an empty filter here is "whatever the environment says",
    // not "everything".
    if (!cfg.device_filter.empty()) {
        engine_.reset(new mooncake::TransferEngine(auto_discover, cfg.device_filter));
    } else {
        engine_.reset(new mooncake::TransferEngine(auto_discover));
    }

    // Under P2PHANDSHAKE the engine takes the *host* from this name and ignores
    // its port, binding whatever it can get instead — so an empty name yields a
    // segment called ":<port>", which no peer can resolve. Always hand it a
    // host, and read the bound name back from getLocalIpAndPort().
    std::string want = cfg.local_segment;
    if (want.empty() && !cfg.ip_or_host.empty()) {
        want = cfg.ip_or_host + ":" + std::to_string(cfg.rpc_port);
    }
    if (engine_->init(cfg.metadata, want, cfg.ip_or_host, cfg.rpc_port) != 0) {
        fprintf(stderr, "[radixshmem] MooncakeBackend: engine init failed "
                        "(metadata='%s', name='%s')\n",
                cfg.metadata.c_str(), want.c_str());
        engine_.reset();
        return false;
    }

    if (!auto_discover) {
        if (engine_->installTransport(cfg.protocol, nullptr) == nullptr) {
            fprintf(stderr, "[radixshmem] MooncakeBackend: installTransport('%s') failed\n",
                    cfg.protocol.c_str());
            engine_.reset();
            return false;
        }
    }

    local_segment_ = engine_->getLocalIpAndPort();
    if (local_segment_.empty() || local_segment_[0] == ':') {
        fprintf(stderr, "[radixshmem] MooncakeBackend: engine bound an unusable segment "
                        "name '%s' (no host) — set TransferConfig::ip_or_host\n",
                local_segment_.c_str());
        engine_.reset();
        return false;
    }
    // Only the port is expected to move; a different host means the request was
    // not honoured and peers would be told the wrong address.
    const std::string want_host = want.substr(0, want.rfind(':'));
    const std::string got_host  = local_segment_.substr(0, local_segment_.rfind(':'));
    if (!want_host.empty() && want_host != got_host) {
        fprintf(stderr, "[radixshmem] MooncakeBackend: requested host '%s' but bound "
                        "'%s'; peers must use the latter\n",
                want_host.c_str(), local_segment_.c_str());
    }
    return true;
}

int MooncakeBackend::register_memory(void* addr, uint64_t len, const std::string& location) {
    if (addr == nullptr || len == 0) return XFER_EINVAL;
    if (!engine_) return XFER_EBACKEND;

    const AddrRange r{reinterpret_cast<uint64_t>(addr), len};
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (const auto& e : registered_)
            if (e.addr == r.addr) return XFER_EINVAL;
    }

    // publish=false still creates the MR (LOCAL_WRITE only), so this process can
    // initiate transfers without publishing a second segment for memory a peer
    // process already publishes (plan D3).
    const std::string loc = location.empty() ? mooncake::kWildcardLocation : location;
    if (engine_->registerLocalMemory(addr, len, loc, publish_, publish_) != 0) {
        fprintf(stderr, "[radixshmem] MooncakeBackend: registerLocalMemory(%p, %lu) failed\n",
                addr, (unsigned long)len);
        return XFER_EBACKEND;
    }

    std::lock_guard<std::mutex> lk(mu_);
    registered_.push_back(r);
    remerge();
    return XFER_OK;
}

int MooncakeBackend::unregister_memory(void* addr) {
    if (!engine_) return XFER_EBACKEND;
    const uint64_t a = reinterpret_cast<uint64_t>(addr);

    std::lock_guard<std::mutex> lk(mu_);
    auto i = registered_.begin();
    for (; i != registered_.end(); ++i) if (i->addr == a) break;
    if (i == registered_.end()) return XFER_ENOENT;

    if (engine_->unregisterLocalMemory(addr) != 0) return XFER_EBACKEND;
    registered_.erase(i);
    remerge();
    return XFER_OK;
}

void MooncakeBackend::remerge() {
    merged_ = merge_contiguous(registered_);
}

bool MooncakeBackend::own_region(AddrRange& out) const {
    if (merged_.size() != 1) return false;
    out = merged_[0];
    return true;
}

int MooncakeBackend::open_segment(const std::string& seg_name, SegmentInfo& out) {
    if (seg_name.empty()) return XFER_EINVAL;
    if (!engine_) return XFER_EBACKEND;

    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = open_by_name_.find(seg_name);
        if (it != open_by_name_.end()) { out = open_[it->second].info; return XFER_OK; }
    }

    Segment seg;

    if (seg_name == local_segment_) {
        // Our own segment. Mooncake would try to hand-shake with itself and
        // fail, and even if it worked a same-process transfer has no business
        // going near a QP -- this is the single-node multi-DP path, where the
        // peer's memory is simply ours.
        std::lock_guard<std::mutex> lk(mu_);
        AddrRange r{};
        if (!own_region(r)) {
            fprintf(stderr, "[radixshmem] MooncakeBackend: own segment is not one "
                            "contiguous registered region\n");
            return XFER_EGEOMETRY;
        }
        seg.is_self     = true;
        seg.info.base   = r.addr;
        seg.info.length = r.len;
    } else {
        const mooncake::SegmentHandle h = engine_->openSegment(seg_name);
        if (static_cast<int64_t>(h) < 0) {
            fprintf(stderr, "[radixshmem] MooncakeBackend: openSegment('%s') failed\n",
                    seg_name.c_str());
            return XFER_ENOENT;
        }
        auto desc = engine_->getMetadata()->getSegmentDescByID(h);
        if (!desc) {
            fprintf(stderr, "[radixshmem] MooncakeBackend: no descriptor for segment '%s'\n",
                    seg_name.c_str());
            engine_->closeSegment(h);
            return XFER_ENOENT;
        }

        std::vector<AddrRange> raw;
        raw.reserve(desc->buffers.size());
        for (const auto& b : desc->buffers) raw.push_back(AddrRange{b.addr, b.length});
        std::vector<AddrRange> merged = merge_contiguous(std::move(raw));

        if (merged.empty()) {
            fprintf(stderr, "[radixshmem] MooncakeBackend: segment '%s' publishes no buffers\n",
                    seg_name.c_str());
            engine_->closeSegment(h);
            return XFER_ENOENT;
        }
        if (merged.size() != 1) {
            fprintf(stderr, "[radixshmem] MooncakeBackend: segment '%s' publishes %zu "
                            "disjoint regions; a data segment must be one contiguous "
                            "range\n", seg_name.c_str(), merged.size());
            engine_->closeSegment(h);
            return XFER_EGEOMETRY;
        }
        seg.handle      = h;
        seg.info.base   = merged[0].addr;
        seg.info.length = merged[0].len;
    }

    std::lock_guard<std::mutex> lk(mu_);
    seg.info.id             = next_segment_++;
    open_[seg.info.id]      = seg;
    open_by_name_[seg_name] = seg.info.id;
    out = seg.info;
    return XFER_OK;
}

void MooncakeBackend::close_segment(SegmentId id) {
    mooncake::SegmentHandle h = 0;
    bool close_it = false;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = open_.find(id);
        if (it == open_.end()) return;
        close_it = !it->second.is_self;
        h = it->second.handle;
        for (auto n = open_by_name_.begin(); n != open_by_name_.end(); ++n) {
            if (n->second == id) { open_by_name_.erase(n); break; }
        }
        open_.erase(it);
    }
    if (close_it && engine_) engine_->closeSegment(h);
}

bool MooncakeBackend::covers_local(const void* addr, uint64_t len) const {
    const uint64_t a = reinterpret_cast<uint64_t>(addr);
    uint64_t end = 0;
    if (__builtin_add_overflow(a, len, &end)) return false;
    for (const auto& r : merged_)
        if (a >= r.addr && end <= r.end()) return true;
    return false;
}

BatchHandle MooncakeBackend::submit(const TransferDesc* descs, int n) {
    if (descs == nullptr || n <= 0 || !engine_) return INVALID_BATCH;

    std::unique_lock<std::mutex> lk(mu_);

    // Validate the whole batch first: a rejected batch must leave every
    // destination untouched, local or remote.
    uint64_t total = 0;
    bool is_self = false;
    for (int i = 0; i < n; ++i) {
        const TransferDesc& d = descs[i];
        if (d.length == 0 || d.local_addr == nullptr) {
            fprintf(stderr, "[radixshmem] MooncakeBackend: desc %d has null addr or zero length\n", i);
            return INVALID_BATCH;
        }
        if (!covers_local(d.local_addr, d.length)) {
            fprintf(stderr, "[radixshmem] MooncakeBackend: desc %d local range [%p,+%lu) "
                            "is not inside any registered region\n",
                    i, d.local_addr, (unsigned long)d.length);
            return INVALID_BATCH;
        }
        auto it = open_.find(d.target);
        if (it == open_.end()) {
            fprintf(stderr, "[radixshmem] MooncakeBackend: desc %d targets unknown segment %d\n",
                    i, (int)d.target);
            return INVALID_BATCH;
        }
        const Segment& seg = it->second;
        uint64_t rend = 0;
        if (__builtin_add_overflow(d.target_offset, d.length, &rend) ||
            d.target_offset < seg.info.base || rend > seg.info.base + seg.info.length) {
            fprintf(stderr, "[radixshmem] MooncakeBackend: desc %d remote range "
                            "[0x%lx,+%lu) escapes segment %d [0x%lx,+%lu)\n",
                    i, (unsigned long)d.target_offset, (unsigned long)d.length,
                    (int)d.target, (unsigned long)seg.info.base,
                    (unsigned long)seg.info.length);
            return INVALID_BATCH;
        }
        if (i == 0) is_self = seg.is_self;
        else if (is_self != seg.is_self) {
            fprintf(stderr, "[radixshmem] MooncakeBackend: batch mixes the local segment "
                            "with a remote one; split it\n");
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
        batches_[h] = Batch{true, mooncake::INVALID_BATCH_ID, total};
        return h;
    }

    std::vector<mooncake::TransferRequest> reqs;
    reqs.reserve(n);
    for (int i = 0; i < n; ++i) {
        const TransferDesc& d = descs[i];
        mooncake::TransferRequest r;
        r.opcode        = (d.op == XferOp::READ) ? mooncake::TransferRequest::READ
                                                 : mooncake::TransferRequest::WRITE;
        r.source        = d.local_addr;
        r.target_id     = open_[d.target].handle;
        r.target_offset = d.target_offset;
        r.length        = d.length;
        reqs.push_back(r);
    }

    const mooncake::BatchID bid = engine_->allocateBatchID(static_cast<size_t>(n));
    if (bid == mooncake::INVALID_BATCH_ID) {
        fprintf(stderr, "[radixshmem] MooncakeBackend: allocateBatchID(%d) failed\n", n);
        return INVALID_BATCH;
    }

    // Drop the lock across submitTransfer: it can block, and holding it would
    // serialise every other thread's poll().
    lk.unlock();
    const auto st = engine_->submitTransfer(bid, reqs);
    if (!st.ok()) {
        fprintf(stderr, "[radixshmem] MooncakeBackend: submitTransfer failed: %.*s\n",
                (int)st.message().size(), st.message().data());
        engine_->freeBatchID(bid);
        return INVALID_BATCH;
    }
    lk.lock();

    const BatchHandle h = next_batch_++;
    batches_[h] = Batch{false, bid, total};
    return h;
}

XferState MooncakeBackend::poll(BatchHandle h, uint64_t* done_bytes) {
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

    mooncake::TransferStatus st{};
    const auto rc = engine_->getBatchTransferStatus(b.id, st);
    if (!rc.ok()) return XferState::FAILED;
    if (done_bytes) *done_bytes = st.transferred_bytes;
    return from_mooncake(st.s);
}

XferState MooncakeBackend::wait(BatchHandle h, int timeout_ms) {
    const auto start = std::chrono::steady_clock::now();
    // Spin briefly first: a small RDMA batch usually lands within microseconds,
    // and sleeping straight away would dominate its latency.
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

int MooncakeBackend::release(BatchHandle h) {
    // Held across freeBatchID, unlike submit(): freeing only walks the task list
    // and deletes, and dropping the lock would let a second release() of the
    // same handle free it twice.
    std::lock_guard<std::mutex> lk(mu_);
    auto it = batches_.find(h);
    if (it == batches_.end()) return XFER_ENOENT;
    if (it->second.is_local) { batches_.erase(it); return XFER_OK; }

    // Mooncake decides: it refuses a batch with an unfinished transfer
    // (multi_transport.cpp:101). Erasing regardless stranded the descriptor.
    const auto st = engine_->freeBatchID(it->second.id);
    if (!st.ok()) {
        if (st.IsBatchBusy()) return XFER_EBUSY;
        fprintf(stderr, "[radixshmem] MooncakeBackend: freeBatchID failed: %.*s\n",
                (int)st.message().size(), st.message().data());
    }
    batches_.erase(it);
    return st.ok() ? XFER_OK : XFER_EBACKEND;
}

} // namespace

std::unique_ptr<ITransferBackend> create_mooncake_backend(const TransferConfig& cfg) {
    auto be = std::unique_ptr<MooncakeBackend>(new MooncakeBackend());
    if (!be->init(cfg)) return nullptr;
    return be;
}

} // namespace radixshmem
