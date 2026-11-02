// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Local-tree remote ops and endpoint lifecycle; transport creation stays in the factory.
#include "radix_server.h"

#ifndef RADIXSHMEM_NO_RDMA

#include "remote_op_service.hpp"
#include "local.h"                 // LocalTreeConnector (svc_tree_)
#include "radix_client.h"          // RadixClient (svc_client_)
#include "cluster_protocol.h"      // MAX_REMOTE_OP_BLOCKS
#include "bilateral_transport.h"   // IRemoteOpServerTransport
#include "remote_op_factory.hpp"   // make_remote_op_server

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace radixshmem {

namespace {

bool valid_count(int n) { return n >= 0 && n <= MAX_REMOTE_OP_BLOCKS; }

// [start, start + length) inside [0, n).
bool valid_range(int start, int length, int n) {
    return start >= 0 && length >= 0 && start <= n - length;
}

bool expired(RemoteOpService::Deadline deadline) {
    return std::chrono::system_clock::now() + std::chrono::milliseconds(10) >= deadline;
}

}  // namespace

// ==================== RemoteOpService ====================

AllocateReply RemoteOpService::allocate(int n, Deadline deadline) {
    // Allocate without hashes or dedup; insert consumes only the unmatched tail.
    AllocateReply out;
    if (!valid_count(n)) { out.status = RemoteStatus::BAD_REQUEST; return out; }
    if (n == 0) return out;
    std::lock_guard<std::mutex> g(insert_mu_);
    if (expired(deadline)) { out.status = RemoteStatus::TIMEOUT; return out; }
    int got = tree_.allocate_slots(n, out.slots);
    if (got < n) {
        tree_.recycle_slots(out.slots.data(), got);
        out.slots.clear();
        out.error = InsertError::MEMPOOL_FULL;
    }
    return out;
}

QueryReply RemoteOpService::query(const uint64_t* hashes, int n, bool lock,
                                  bool update_meta, Deadline deadline) {
    QueryReply out;
    if (!valid_count(n) || (n > 0 && !hashes)) {
        out.status = RemoteStatus::BAD_REQUEST;
        return out;
    }
    if (expired(deadline)) { out.status = RemoteStatus::TIMEOUT; return out; }
    TreeComponentQueryResult r = tree_.tree().query_components(
        hashes, n, COMPONENT_MASK_FULL, lock, update_meta);
    out.hit_length = (int)r.common_hit;
    out.slots      = std::move(r.full_slots);   // len == hit_length
    if (lock) {
        out.pin_end  = (int)r.pin.full.end;
        out.pin_tail = std::move(r.pin.tail);
    }
    return out;
}

InsertResult RemoteOpService::insert(const uint64_t* hashes, int n, int start,
                                     int length, const int32_t* slots,
                                     bool auto_recycle) {
    InsertResult out;
    if (!valid_count(n) || !valid_range(start, length, n) ||
        (n > 0 && !hashes) || (length > 0 && !slots)) {
        out.error = InsertError::REMOTE_FAILED;
        out.remote_status = RemoteStatus::BAD_REQUEST;
        return out;
    }
    if (length == 0) return out;
    std::lock_guard<std::mutex> g(insert_mu_);
    // auto_recycle: the target recycles the surplus; else the initiator does.
    return client_.insert(hashes, n, slots, length, start, auto_recycle);
}

RemoteStatus RemoteOpService::recycle(const int32_t* slots, int num_slots) {
    if (!valid_count(num_slots) || (num_slots > 0 && !slots))
        return RemoteStatus::BAD_REQUEST;
    if (num_slots > 0) tree_.recycle_slots(slots, num_slots);
    return RemoteStatus::OK;
}

RemoteStatus RemoteOpService::unlock(const uint64_t* hashes, int n, int start,
                                     int length, int pin_end,
                                     const uint64_t* pin_tail, int pin_tail_n) {
    if (!valid_count(n) || !valid_range(start, length, n) ||
        (n > 0 && !hashes) || !valid_count(pin_tail_n) ||
        (pin_tail_n > 0 && !pin_tail) || pin_end < start + length)
        return RemoteStatus::BAD_REQUEST;
    if (length == 0) return RemoteStatus::OK;
    PinRecord pin;
    pin.full = {(uint32_t)start, (uint32_t)pin_end};
    pin.tail.assign(pin_tail, pin_tail + pin_tail_n);
    return tree_.tree().dec_ref(hashes, n, std::move(pin)) ? RemoteStatus::OK
                                                           : RemoteStatus::BAD_REQUEST;
}

// ==================== RadixServer remote-op endpoint ====================

RemoteOpAddr RadixServer::rc_build_endpoint() {
    // Body of the build_remote_op callback.
    RemoteOpAddr addr{};
    rc_server_ = make_remote_op_server(cfg_.remote_op_listen_port, instance_id_, addr);
    return addr;
}

void RadixServer::rc_spawn() {
    if (!rc_server_) return;
    if (!svc_client_ || !svc_tree_) {
        throw std::runtime_error(
            "rc_spawn: remote-op server is up but svc_* not built "
            "(need world_size>1 + successful svc RadixClient bootstrap)");
    }
    rc_service_ = std::make_unique<RemoteOpService>(*svc_tree_, *svc_client_,
                                                    svc_client_mu_);
    rc_server_->start(*rc_service_);
}

void RadixServer::rc_stop() {
    if (rc_server_) rc_server_->stop();
    rc_server_.reset();
    rc_service_.reset();
}

}  // namespace radixshmem

#endif  // !RADIXSHMEM_NO_RDMA
