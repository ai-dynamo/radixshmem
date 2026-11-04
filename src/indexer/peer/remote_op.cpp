// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RemotePeer: a remote-op session to a peer's local tree over one backend.
#include "radix_client.h"          // RemotePeer decl + reply structs

#ifndef RADIXSHMEM_NO_RDMA

#include "bilateral_transport.h"    // IRemoteOpClientTransport
#include "peer_breaker.hpp"

#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace radixshmem {

RemotePeer::RemotePeer(
    std::unique_ptr<IRemoteOpClientTransport> transport, std::string endpoint)
    : transport_(std::move(transport)), endpoint_(std::move(endpoint)) {}

RemotePeer::~RemotePeer() = default;
RemotePeer::RemotePeer(RemotePeer&&) noexcept = default;
RemotePeer& RemotePeer::operator=(RemotePeer&&) noexcept = default;

namespace {

using Clock = std::chrono::steady_clock;

double since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

InsertResult unreachable_insert() {
    InsertResult r;
    r.error = InsertError::REMOTE_FAILED;
    r.remote_status = RemoteStatus::UNAVAILABLE;
    return r;
}

}  // namespace

bool RemotePeer::skipped() {
    if (!breaker_ || breaker_->gate(peer_rank_) != PeerBreaker::Gate::SKIP) return false;
    breaker_->skipped(peer_rank_, PeerBreaker::Op::REMOTE_OP);
    return true;
}

void RemotePeer::observe(const char* op, RemoteStatus st, double secs) {
    if (!breaker_) return;
    if (st == RemoteStatus::OK)
        breaker_->succeeded(peer_rank_);
    else if (st == RemoteStatus::UNAVAILABLE)
        breaker_->failed(peer_rank_, op, "unavailable", secs);
    else if (st == RemoteStatus::TIMEOUT && !transport_->connected())
        breaker_->failed(peer_rank_, op, "timed out while connecting", secs);
}

AllocateReply RemotePeer::allocate(int n, ComponentType component) {
    AllocateReply r;
    if (skipped()) {
        r.status = RemoteStatus::UNAVAILABLE;
        return r;
    }
    const auto t0 = Clock::now();
    r = transport_->allocate(n, component);
    observe("allocate", r.status, since(t0));
    return r;
}

AllocateComponentsReply RemotePeer::allocate_components(
        const std::vector<ComponentCount>& components) {
    AllocateComponentsReply r;
    if (skipped()) {
        r.status = RemoteStatus::UNAVAILABLE;
        return r;
    }
    const auto t0 = Clock::now();
    r = transport_->allocate_components(components);
    observe("allocate components", r.status, since(t0));
    return r;
}

InsertResult RemotePeer::insert(const uint64_t* hashes, int n,
                                int start, int length, const int32_t* slots,
                                bool auto_recycle, ComponentType component) {
    if (skipped()) return unreachable_insert();
    const auto t0 = Clock::now();
    InsertResult r = transport_->insert(hashes, n, start, length, slots, auto_recycle, component);
    observe("insert", r.remote_status, since(t0));
    return r;
}

InsertPathResult RemotePeer::insert_path(const uint64_t* hashes, int n, int start, int length,
                                         const int32_t* full_slots,
                                         const std::vector<ComponentRun>& runs,
                                         bool auto_recycle) {
    InsertPathResult r;
    if (skipped()) {
        r.remote_status = RemoteStatus::UNAVAILABLE;
        r.full = unreachable_insert();
        r.runs.assign(runs.size(), r.full);
        return r;
    }
    const auto t0 = Clock::now();
    r = transport_->insert_path(hashes, n, start, length, full_slots, runs, auto_recycle);
    observe("insert path", r.remote_status, since(t0));
    return r;
}

QueryReply RemotePeer::query(const uint64_t* hashes, int n, bool lock,
                             bool update_meta) {
    QueryReply r;
    if (skipped()) {
        r.status = RemoteStatus::UNAVAILABLE;
        return r;
    }
    const auto t0 = Clock::now();
    r = transport_->query(hashes, n, lock, update_meta);
    observe("query", r.status, since(t0));
    return r;
}

RemoteStatus RemotePeer::recycle(const int32_t* slots, int num_slots,
                                 ComponentType component) {
    if (skipped()) return RemoteStatus::UNAVAILABLE;
    const auto t0 = Clock::now();
    const RemoteStatus st = transport_->recycle(slots, num_slots, component);
    observe("recycle", st, since(t0));
    return st;
}

RemoteStatus RemotePeer::unlock(const uint64_t* hashes, int n, int start,
                                int length, int pin_end,
                                const std::vector<uint64_t>& pin_tail) {
    if (skipped()) return RemoteStatus::UNAVAILABLE;
    const auto t0 = Clock::now();
    const RemoteStatus st = transport_->unlock(hashes, n, start, length, pin_end, pin_tail);
    observe("unlock", st, since(t0));
    return st;
}

}  // namespace radixshmem

#endif  // !RADIXSHMEM_NO_RDMA
