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

bool valid_component(ComponentType c) { return (int)c < NUM_COMPONENT_TYPES; }

// [start, start + length) inside [0, n).
bool valid_range(int start, int length, int n) {
    return start >= 0 && length >= 0 && start <= n - length;
}

constexpr int64_t DEDUP_WRITE_TIMEOUT_MS = 3000;
constexpr int64_t DEDUP_WRITE_MARGIN_MS = 1500;
constexpr int64_t DEDUP_REPLY_MARGIN_MS = 200;

bool expired(RemoteOpService::Deadline deadline) {
    return std::chrono::system_clock::now() + std::chrono::milliseconds(10) >= deadline;
}

}  // namespace

// ==================== RemoteOpService ====================

RemoteOpService::RemoteOpService(LocalTreeConnector& tree, RadixClient& client,
                                 std::mutex& insert_mu)
    : tree_(tree), client_(client), insert_mu_(insert_mu) {
    const char* e = ::getenv("RADIXSHMEM_DISABLE_AUTO_DEDUP");
    dedup_off_ = e != nullptr && std::string(e) == "1";
}

AllocateReply RemoteOpService::allocate(int n, Deadline deadline,
                                        ComponentType component) {
    // Allocate without hashes or dedup; insert consumes only the unmatched tail.
    AllocateReply out;
    if (!valid_count(n) || !valid_component(component)) {
        out.status = RemoteStatus::BAD_REQUEST;
        return out;
    }
    if (n == 0) return out;
    std::lock_guard<std::mutex> g(insert_mu_);
    if (expired(deadline)) { out.status = RemoteStatus::TIMEOUT; return out; }
    int got = tree_.allocate_slots(n, out.slots, component);
    if (got < n) {
        tree_.recycle_slots(out.slots.data(), got, component);
        out.slots.clear();
        out.error = InsertError::MEMPOOL_FULL;
    }
    return out;
}

AllocateComponentsReply RemoteOpService::allocate_components(
        const std::vector<ComponentCount>& components, Deadline deadline) {
    AllocateComponentsReply out;
    bool seen[NUM_COMPONENT_TYPES] = {};
    int64_t total = 0;
    for (const ComponentCount& c : components) {
        if (!valid_component(c.component) || seen[(int)c.component] || !valid_count(c.n)) {
            out.status = RemoteStatus::BAD_REQUEST;
            return out;
        }
        seen[(int)c.component] = true;
        total += c.n;
    }
    if (total > MAX_REMOTE_OP_BLOCKS) {
        out.status = RemoteStatus::BAD_REQUEST;
        return out;
    }
    if (components.empty()) return out;
    std::lock_guard<std::mutex> g(insert_mu_);
    if (expired(deadline)) { out.status = RemoteStatus::TIMEOUT; return out; }
    out.slots.resize(components.size());
    for (size_t i = 0; i < components.size(); i++) {
        const ComponentCount& c = components[i];
        if (c.n == 0 || tree_.allocate_slots(c.n, out.slots[i], c.component) == c.n) continue;
        for (size_t j = 0; j <= i; j++)
            tree_.recycle_slots(out.slots[j].data(), (int)out.slots[j].size(),
                                components[j].component);
        out.slots.clear();
        out.error = InsertError::MEMPOOL_FULL;
        return out;
    }
    return out;
}

QueryReply RemoteOpService::query(const uint64_t* hashes, int n, const RemoteQuery& q,
                                  Deadline deadline) {
    QueryReply out;
    if (!valid_count(n) || (n > 0 && !hashes)) {
        out.status = RemoteStatus::BAD_REQUEST;
        return out;
    }
    if (expired(deadline)) { out.status = RemoteStatus::TIMEOUT; return out; }
    TreeComponentQueryResult r = tree_.tree().query_components(
        hashes, n, q.mask, q.lock, q.update_meta, &q.window);
    if (r.status != RemoteStatus::OK) { out.status = r.status; return out; }
    const uint32_t hit = r.common_hit;
    out.hit_length = (int)hit;
    if (!r.full_slots.empty())
        out.slots.assign(r.full_slots.begin() + std::min(q.window.full_from, hit),
                         r.full_slots.begin() + hit);
    out.swa_start  = r.swa_start;
    out.swa_slots  = std::move(r.swa_slots);
    out.mamba_slot = r.mamba_checkpoint_slot;
    out.pin        = std::move(r.pin);
    return out;
}

InsertResult RemoteOpService::insert(const uint64_t* hashes, int n, int start,
                                     int length, const int32_t* slots,
                                     bool auto_recycle, ComponentType component) {
    InsertResult out;
    if (!valid_count(n) || !valid_range(start, length, n) || !valid_component(component) ||
        (n > 0 && !hashes) || (length > 0 && !slots)) {
        out.error = InsertError::REMOTE_FAILED;
        out.remote_status = RemoteStatus::BAD_REQUEST;
        return out;
    }
    if (length == 0) return out;
    std::lock_guard<std::mutex> g(insert_mu_);
    // auto_recycle: the target recycles the surplus; else the initiator does.
    return client_.insert(hashes, n, slots, length, start, auto_recycle, component);
}

InsertPathResult RemoteOpService::insert_path(const uint64_t* hashes, int n, int start,
                                              int length, const int32_t* full_slots,
                                              const std::vector<ComponentRun>& runs,
                                              bool auto_recycle) {
    InsertPathResult out;
    out.runs.resize(runs.size());
    bool sane = valid_count(n) && valid_range(start, length, n) && (n == 0 || hashes) &&
                (length == 0 || full_slots) && runs.size() <= (size_t)MAX_REMOTE_OP_BLOCKS;
    for (const ComponentRun& r : runs)
        sane = sane && (r.component == ComponentType::SWA || r.component == ComponentType::MAMBA) &&
               r.end > 0 && r.end <= n && !r.slots.empty() && (int)r.slots.size() <= r.end &&
               (r.component == ComponentType::SWA || r.slots.size() == 1);
    if (!sane) {
        out.remote_status = RemoteStatus::BAD_REQUEST;
        return out;
    }
    std::lock_guard<std::mutex> g(insert_mu_);
    if (length > 0)
        out.full = client_.insert(hashes, n, full_slots, length, start, auto_recycle,
                                  ComponentType::FULL);
    for (size_t i = 0; i < runs.size(); i++)
        out.runs[i] = client_.insert(hashes, runs[i].end, runs[i].slots.data(),
                                     (int)runs[i].slots.size(), 0, auto_recycle,
                                     runs[i].component);
    return out;
}

RemoteStatus RemoteOpService::recycle(const int32_t* slots, int num_slots,
                                      ComponentType component) {
    if (!valid_count(num_slots) || (num_slots > 0 && !slots) || !valid_component(component))
        return RemoteStatus::BAD_REQUEST;
    if (num_slots > 0) tree_.recycle_slots(slots, num_slots, component);
    return RemoteStatus::OK;
}

RemoteStatus RemoteOpService::unlock(const uint64_t* hashes, int n, PinRecord&& pin) {
    if (!valid_count(n) || (n > 0 && !hashes) || !valid_count((int)pin.tail.size()) ||
        pin.full.start > pin.full.end || pin.swa.start > pin.swa.end)
        return RemoteStatus::BAD_REQUEST;
    if (pin.empty()) return RemoteStatus::OK;
    return tree_.tree().dec_ref(hashes, n, std::move(pin)) ? RemoteStatus::OK
                                                           : RemoteStatus::BAD_REQUEST;
}

DedupReply RemoteOpService::dedup_prefix(const uint64_t* hashes, int n,
                                         const DedupRequest& req, Deadline deadline) {
    DedupReply out;
    size_t landing = 0;
    for (const std::vector<int32_t>& l : req.landing) landing += l.size();
    if (!valid_count(n) || (n > 0 && !hashes) || landing > (size_t)MAX_REMOTE_OP_BLOCKS) {
        out.status = RemoteStatus::BAD_REQUEST;
        return out;
    }
    for (int k = 0; k < NUM_COMPONENT_TYPES; k++) out.unused[k] = req.landing[k];
    if (dedup_off_ || n == 0) return out;
    if (landing == 0 || writer_ == nullptr || !writer_->active() || req.rank < 0) {
        std::lock_guard<std::mutex> g(insert_mu_);
        out.freed = tree_.tree().dedup_prefix(hashes, n);
        return out;
    }
    uint32_t budget[NUM_COMPONENT_TYPES];
    for (int k = 0; k < NUM_COMPONENT_TYPES; k++) budget[k] = (uint32_t)req.landing[k].size();
    DedupPlan plan;
    {
        std::lock_guard<std::mutex> g(insert_mu_);
        if (!tree_.tree().plan_dedup(hashes, n, req.tail_cap, req.branch_cap, budget, plan))
            return out;
    }
    std::vector<int32_t> local[NUM_COMPONENT_TYPES], remote[NUM_COMPONENT_TYPES];
    auto land = [&](int k, int32_t& slot) {
        local[k].push_back(slot);
        slot = req.landing[k][remote[k].size()];
        remote[k].push_back(slot);
    };
    for (MovedPath& p : plan.moved) {
        for (int32_t& s : p.full) land((int)ComponentType::FULL, s);
        for (MovedCell& c : p.cells) land((int)c.component, c.slot);
    }
    for (int k = 0; k < NUM_COMPONENT_TYPES; k++)
        out.unused[k].assign(req.landing[k].begin() + remote[k].size(), req.landing[k].end());
    DedupWriter::Result r = DedupWriter::OK;
    if (!plan.moved.empty()) {
        // The writer may wait past its timeout; the caller's deadline must still hold.
        int64_t ms = DEDUP_WRITE_TIMEOUT_MS;
        if (deadline != Deadline::max())
            ms = std::min<int64_t>(ms, std::chrono::duration_cast<std::chrono::milliseconds>(
                                           deadline - std::chrono::system_clock::now()).count() -
                                           DEDUP_WRITE_MARGIN_MS);
        r = ms > 0 ? writer_->write(req.rank, local, remote, (int)ms) : DedupWriter::FAILED;
    }
    std::lock_guard<std::mutex> g(insert_mu_);
    // A caller that gives up adopts nothing: past its deadline the copy stays here.
    const bool in_time = deadline == Deadline::max() ||
                         std::chrono::system_clock::now() +
                                 std::chrono::milliseconds(DEDUP_REPLY_MARGIN_MS) < deadline;
    const int freed = tree_.tree().finish_dedup(plan, r == DedupWriter::OK && in_time);
    out.freed = std::max(freed, 0);
    if (r == DedupWriter::OK && freed >= 0)
        out.moved = std::move(plan.moved);
    else if (r != DedupWriter::LATE)
        for (int k = 0; k < NUM_COMPONENT_TYPES; k++)
            out.unused[k].insert(out.unused[k].end(), remote[k].begin(), remote[k].end());
    return out;
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
    rc_service_->set_dedup_writer(dedup_writes_.get());
    rc_server_->start(*rc_service_);
}

void RadixServer::rc_stop() {
    if (rc_server_) rc_server_->stop();
    rc_server_.reset();
    rc_service_.reset();
}

}  // namespace radixshmem

#endif  // !RADIXSHMEM_NO_RDMA
