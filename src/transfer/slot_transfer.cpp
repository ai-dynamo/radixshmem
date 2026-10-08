// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "slot_transfer.h"

#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace radixshmem {

SlotTransfer::SlotTransfer(SlotStore& store, ITransferBackend& backend)
    : store_(store), backend_(backend) {}

int SlotTransfer::register_store(const std::string& location) {
    if (!store_.valid()) return XFER_EINVAL;
    return backend_.register_memory(store_.data_base(), store_.data_bytes(), location);
}

// Published layout (or its absence) -> absolute peer geometry, or a refusal.
bool SlotTransfer::resolve_peer(const StoreGeometry* published, const SegmentInfo& info,
                                StoreGeometry& out, std::string& why) const {
    const StoreGeometry mine = store_.layout();

    StoreGeometry layout;
    if (published == nullptr) {
        if (mine.pool_mask == COMPONENT_MASK_FULL) {
            // Single-pool contract: count = whatever the segment holds.
            const uint64_t sb = mine[ComponentType::FULL].slot_bytes;
            if (info.length < sb || info.length % sb != 0) {
                why = "publishes " + std::to_string(info.length) +
                      " bytes, which is not a whole number of " +
                      std::to_string(sb) + "-byte slots";
                return false;
            }
            layout.pool_mask = COMPONENT_MASK_FULL;
            layout[ComponentType::FULL] = PoolGeometry{0, info.length / sb, sb};
        } else {
            layout = mine;   // assumed identical; still checked to fit
        }
    } else {
        layout = *published;
        if (layout.pool_mask != mine.pool_mask) {
            why = "has pool mask " + std::to_string(layout.pool_mask) +
                  ", we have " + std::to_string(mine.pool_mask) +
                  " — the two stores hold different KV components";
            return false;
        }
        for (int i = 0; i < NUM_COMPONENT_TYPES; ++i) {
            const ComponentType t = static_cast<ComponentType>(i);
            if (!layout.has(t)) continue;
            if (layout[t].slot_bytes != mine[t].slot_bytes) {
                why = std::string("has a ") + std::to_string(layout[t].slot_bytes) +
                      "-byte " + component_type_name(t) + " slot stride, we have " +
                      std::to_string(mine[t].slot_bytes) +
                      " — slot ids would not mean the same bytes on both sides";
                return false;
            }
            if (layout[t].num_slots == 0) {
                why = std::string("publishes an empty ") + component_type_name(t) + " pool";
                return false;
            }
        }
    }

    // Pools in order, disjoint, inside the segment.
    uint64_t prev_end = 0;
    for (int i = 0; i < NUM_COMPONENT_TYPES; ++i) {
        const ComponentType t = static_cast<ComponentType>(i);
        if (!layout.has(t)) continue;
        const PoolGeometry& p = layout[t];
        uint64_t bytes = 0, end = 0;
        if (__builtin_mul_overflow(p.num_slots, p.slot_bytes, &bytes) ||
            __builtin_add_overflow(p.base, bytes, &end) ||
            p.base < prev_end || end > info.length) {
            why = std::string("its ") + component_type_name(t) + " pool [" +
                  std::to_string(p.base) + ", +" + std::to_string(p.num_slots) + " x " +
                  std::to_string(p.slot_bytes) + ") does not fit its " +
                  std::to_string(info.length) + "-byte segment";
            return false;
        }
        prev_end = end;
    }

    out = layout.rebased(info.base);
    return true;
}

int SlotTransfer::attach_peer(const std::string& peer_name, const StoreGeometry* published,
                              StoreGeometry* out) {
    if (peer_name.empty()) return XFER_EINVAL;
    if (!store_.valid()) return XFER_EINVAL;
    if (published != nullptr &&
        ((published->pool_mask & COMPONENT_MASK_FULL) == 0 ||
         (published->pool_mask & ~COMPONENT_MASK_ALL) != 0))
        return XFER_EINVAL;

    // Held across open_segment: control path, and it keeps lookup-then-insert atomic.
    std::lock_guard<std::mutex> lk(mu_);

    auto it = peers_.find(peer_name);
    if (it != peers_.end()) {
        // The cache never outvotes the registry.
        if (published != nullptr && !published->same_strides(it->second.geo)) {
            fprintf(stderr, "[radixshmem] SlotTransfer: peer '%s' was attached with a "
                            "different pool layout than the one now published\n",
                    peer_name.c_str());
            return XFER_EGEOMETRY;
        }
        if (out) *out = it->second.geo;
        return XFER_OK;
    }

    // Strides are checked before opening; the segment fit only after.
    if (published != nullptr && !published->same_strides(store_.layout())) {
        SegmentInfo none;
        StoreGeometry unused;
        std::string why;
        resolve_peer(published, none, unused, why);
        fprintf(stderr, "[radixshmem] SlotTransfer: peer '%s' %s\n", peer_name.c_str(), why.c_str());
        return XFER_EGEOMETRY;
    }

    SegmentInfo info;
    const int rc = backend_.open_segment(peer_name, info);
    if (rc != XFER_OK) return rc;

    Peer p;
    std::string why;
    if (!resolve_peer(published, info, p.geo, why)) {
        fprintf(stderr, "[radixshmem] SlotTransfer: peer '%s' %s\n", peer_name.c_str(), why.c_str());
        backend_.close_segment(info.id);
        return XFER_EGEOMETRY;
    }
    p.id = info.id;
    peers_[peer_name] = p;
    if (out) *out = p.geo;
    return XFER_OK;
}

int SlotTransfer::attach_peer(const std::string& peer_name, uint64_t peer_slot_bytes,
                              SlotGeometry* out) {
    if (!store_.valid()) return XFER_EINVAL;
    if (peer_slot_bytes == 0) return XFER_EINVAL;
    if (store_.pool_mask() != COMPONENT_MASK_FULL) {
        fprintf(stderr, "[radixshmem] SlotTransfer: attach_peer(stride) is for FULL-only "
                        "stores; this one has more pools — pass the peer's published "
                        "StoreGeometry\n");
        return XFER_EINVAL;
    }
    const uint64_t sb = store_.slot_bytes();
    if (peer_slot_bytes != PEER_SLOT_BYTES_UNCHECKED && peer_slot_bytes != sb) {
        fprintf(stderr, "[radixshmem] SlotTransfer: peer '%s' has a %lu-byte slot "
                        "stride, we have %lu — slot ids would not mean the same "
                        "bytes on both sides\n",
                peer_name.c_str(), (unsigned long)peer_slot_bytes, (unsigned long)sb);
        return XFER_EGEOMETRY;
    }
    // Count comes off the segment: published == nullptr.
    StoreGeometry g;
    const int rc = attach_peer(peer_name, nullptr, &g);
    if (rc == XFER_OK && out) *out = g[ComponentType::FULL];
    return rc;
}

int SlotTransfer::detach_peer(const std::string& peer_name) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = peers_.find(peer_name);
    if (it == peers_.end()) return XFER_OK;   // idempotent
    if (it->second.inflight > 0) {
        fprintf(stderr, "[radixshmem] SlotTransfer: peer '%s' has %d batch(es) in "
                        "flight; wait() and release() them before detaching\n",
                peer_name.c_str(), it->second.inflight);
        return XFER_EBUSY;
    }
    backend_.close_segment(it->second.id);
    peers_.erase(it);
    return XFER_OK;
}

void SlotTransfer::drop_inflight_locked(const std::string& peer_name) {
    auto it = peers_.find(peer_name);
    if (it != peers_.end() && it->second.inflight > 0) --it->second.inflight;
}

bool SlotTransfer::peer_geometry(const std::string& peer_name, StoreGeometry& out) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = peers_.find(peer_name);
    if (it == peers_.end()) return false;
    out = it->second.geo;
    return true;
}

bool SlotTransfer::peer_geometry(const std::string& peer_name, SlotGeometry& out) const {
    StoreGeometry g;
    if (!peer_geometry(peer_name, g)) return false;
    out = g[ComponentType::FULL];
    return true;
}

BatchHandle SlotTransfer::read(const std::string& peer, ComponentType c,
                               const int32_t* remote_slots, const int32_t* local_slots,
                               int n, int* descriptors) {
    return submit_batch(XferOp::READ, peer, c, local_slots, remote_slots, n, descriptors);
}

BatchHandle SlotTransfer::write(const std::string& peer, ComponentType c,
                                const int32_t* local_slots, const int32_t* remote_slots,
                                int n, int* descriptors) {
    return submit_batch(XferOp::WRITE, peer, c, local_slots, remote_slots, n, descriptors);
}

BatchHandle SlotTransfer::submit_batch(XferOp op, const std::string& peer, ComponentType c,
                                       const int32_t* local_slots,
                                       const int32_t* remote_slots, int n,
                                       int* descriptors) {
    if (descriptors) *descriptors = 0;
    if (n <= 0 || local_slots == nullptr || remote_slots == nullptr) return INVALID_BATCH;
    if (!component_type_valid(c)) return INVALID_BATCH;
    if (!store_.has_pool(c)) {
        fprintf(stderr, "[radixshmem] SlotTransfer: this store has no %s pool\n",
                component_type_name(c));
        return INVALID_BATCH;
    }

    // Copied, not referenced: the lock drops before submit.
    Peer p;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = peers_.find(peer);
        if (it == peers_.end()) {
            fprintf(stderr, "[radixshmem] SlotTransfer: peer '%s' is not attached\n", peer.c_str());
            return INVALID_BATCH;
        }
        p = it->second;
    }
    if (!p.geo.has(c)) {
        fprintf(stderr, "[radixshmem] SlotTransfer: peer '%s' has no %s pool\n",
                peer.c_str(), component_type_name(c));
        return INVALID_BATCH;
    }
    const PoolGeometry& rp = p.geo[c];
    const uint64_t sb = rp.slot_bytes;   // == store_.slot_bytes(c), by attach_peer
    const uint64_t local_n = store_.num_slots(c);

    // All-or-nothing: validate everything before building anything.
    for (int i = 0; i < n; ++i) {
        if (local_slots[i] < 0 || (uint64_t)local_slots[i] >= local_n) {
            fprintf(stderr, "[radixshmem] SlotTransfer: local %s slot %d out of range "
                            "[0,%lu)\n", component_type_name(c), local_slots[i],
                    (unsigned long)local_n);
            return INVALID_BATCH;
        }
        if (remote_slots[i] < 0 || (uint64_t)remote_slots[i] >= rp.num_slots) {
            fprintf(stderr, "[radixshmem] SlotTransfer: remote %s slot %d out of range "
                            "[0,%lu) on peer '%s'\n", component_type_name(c),
                    remote_slots[i], (unsigned long)rp.num_slots, peer.c_str());
            return INVALID_BATCH;
        }
    }

    // Fuse runs contiguous on both sides.
    std::vector<TransferDesc> descs;
    descs.reserve((size_t)n);
    for (int i = 0; i < n;) {
        int run = 1;
        while (i + run < n &&
               local_slots[i + run]  == local_slots[i + run - 1] + 1 &&
               remote_slots[i + run] == remote_slots[i + run - 1] + 1) {
            ++run;
        }

        TransferDesc d;
        d.op            = op;
        d.local_addr    = store_.slot_addr(c, (uint64_t)local_slots[i]);
        d.target        = p.id;
        d.target_offset = rp.base + (uint64_t)remote_slots[i] * sb;
        d.length        = (uint64_t)run * sb;
        descs.push_back(d);

        i += run;
    }

    // Re-check the segment is still the one the descriptors were built from,
    // then hold the peer until release().
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = peers_.find(peer);
        if (it == peers_.end() || it->second.id != p.id) {
            fprintf(stderr, "[radixshmem] SlotTransfer: peer '%s' was detached while "
                            "the batch was being built\n", peer.c_str());
            return INVALID_BATCH;
        }
        ++it->second.inflight;
    }

    const BatchHandle h = backend_.submit(descs.data(), (int)descs.size());

    std::lock_guard<std::mutex> lk(mu_);
    if (h == INVALID_BATCH) {
        drop_inflight_locked(peer);
        return INVALID_BATCH;
    }
    batch_peer_[h] = peer;
    if (descriptors) *descriptors = (int)descs.size();
    return h;
}

XferState SlotTransfer::poll(BatchHandle h, uint64_t* done_bytes) {
    return backend_.poll(h, done_bytes);
}

XferState SlotTransfer::wait(BatchHandle h, int timeout_ms) {
    return backend_.wait(h, timeout_ms);
}

int SlotTransfer::release(BatchHandle h) {
    // A refused release keeps the hold.
    const int rc = backend_.release(h);
    if (rc == XFER_EBUSY) return rc;

    std::lock_guard<std::mutex> lk(mu_);
    auto it = batch_peer_.find(h);
    if (it == batch_peer_.end()) return rc;   // already released, or not ours
    drop_inflight_locked(it->second);
    batch_peer_.erase(it);
    return rc;
}

} // namespace radixshmem
