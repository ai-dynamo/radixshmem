// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "component_kind.h"
#include "slot_geometry.h"
#include "slot_store.h"
#include "transfer.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace radixshmem {

inline constexpr uint64_t PEER_SLOT_BYTES_UNCHECKED = UINT64_MAX;

class SlotTransfer {
public:
    SlotTransfer(SlotStore& store, ITransferBackend& backend);

    SlotTransfer(const SlotTransfer&) = delete;
    SlotTransfer& operator=(const SlotTransfer&) = delete;

    // ==================== Control path ====================

    int register_store(const std::string& location = "");

    int attach_peer(const std::string& peer_name, const StoreGeometry* published,
                    StoreGeometry* out = nullptr);

    int attach_peer(const std::string& peer_name, uint64_t peer_slot_bytes,
                    SlotGeometry* out = nullptr);

    int detach_peer(const std::string& peer_name);

    bool peer_geometry(const std::string& peer_name, StoreGeometry& out) const;
    bool peer_geometry(const std::string& peer_name, SlotGeometry& out) const;

    const std::string& local_segment() const { return backend_.local_segment(); }

    // ==================== Data path ====================

    BatchHandle read(const std::string& peer, ComponentType c,
                     const int32_t* remote_slots, const int32_t* local_slots, int n,
                     int* descriptors = nullptr);
    BatchHandle write(const std::string& peer, ComponentType c,
                      const int32_t* local_slots, const int32_t* remote_slots, int n,
                      int* descriptors = nullptr);

    BatchHandle read(const std::string& peer, const int32_t* remote_slots,
                     const int32_t* local_slots, int n, int* descriptors = nullptr) {
        return read(peer, ComponentType::FULL, remote_slots, local_slots, n, descriptors);
    }
    BatchHandle write(const std::string& peer, const int32_t* local_slots,
                      const int32_t* remote_slots, int n, int* descriptors = nullptr) {
        return write(peer, ComponentType::FULL, local_slots, remote_slots, n, descriptors);
    }

    XferState poll(BatchHandle h, uint64_t* done_bytes = nullptr);
    XferState wait(BatchHandle h, int timeout_ms);
    int       release(BatchHandle h);

    const SlotStore&        store()   const { return store_; }
    const ITransferBackend& backend() const { return backend_; }

private:
    BatchHandle submit_batch(XferOp op, const std::string& peer, ComponentType c,
                             const int32_t* local_slots,
                             const int32_t* remote_slots, int n,
                             int* descriptors);

    bool resolve_peer(const StoreGeometry* published, const SegmentInfo& info,
                      StoreGeometry& out, std::string& why) const;

    SlotStore&        store_;
    ITransferBackend& backend_;

    void drop_inflight_locked(const std::string& peer_name);

    struct Peer {
        SegmentId     id = INVALID_SEGMENT;
        StoreGeometry geo;           // absolute bases
        int           inflight = 0;  // submitted, not yet released
    };
    mutable std::mutex mu_;
    std::unordered_map<std::string, Peer> peers_;
    std::unordered_map<BatchHandle, std::string> batch_peer_;
};

} // namespace radixshmem
