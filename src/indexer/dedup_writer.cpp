// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dedup_writer.h"

#include <algorithm>
#include <chrono>

namespace radixshmem {

namespace {

constexpr int GRACE_MS = 1000;   // a popped copy finishes this long after its own timeout

}  // namespace

bool DedupWriteQueue::active() {
    std::lock_guard<std::mutex> g(mu_);
    return active_;
}

DedupWriter::Result DedupWriteQueue::write(int rank,
                                           const std::vector<int32_t> (&local)[NUM_COMPONENT_TYPES],
                                           const std::vector<int32_t> (&remote)[NUM_COMPONENT_TYPES],
                                           int timeout_ms) {
    std::unique_lock<std::mutex> lk(mu_);
    if (!active_) return FAILED;
    const uint64_t id = next_id_++;
    Pending& p = pending_[id];
    p.w.id = id;
    p.w.rank = rank;
    p.w.timeout_ms = timeout_ms;
    for (int k = 0; k < NUM_COMPONENT_TYPES; k++) {
        p.w.local[k] = local[k];
        p.w.remote[k] = remote[k];
    }
    queued_.push_back(id);
    cv_.notify_all();
    cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms + GRACE_MS),
                 [&] { return p.done || !active_; });
    const Result r = p.done ? p.result : p.popped ? LATE : FAILED;
    pending_.erase(id);
    return r;
}

std::vector<DedupWrite> DedupWriteQueue::pop(size_t max_writes, int timeout_ms) {
    std::vector<DedupWrite> out;
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait_for(lk, std::chrono::milliseconds(std::max(timeout_ms, 0)),
                 [&] { return !queued_.empty() || !active_; });
    while (active_ && !queued_.empty() && out.size() < max_writes) {
        auto it = pending_.find(queued_.front());
        queued_.pop_front();
        if (it == pending_.end()) continue;
        it->second.popped = true;
        out.push_back(it->second.w);
    }
    return out;
}

void DedupWriteQueue::complete(uint64_t id, int result) {
    std::lock_guard<std::mutex> g(mu_);
    auto it = pending_.find(id);
    if (it == pending_.end()) return;
    it->second.done = true;
    it->second.result = result == OK ? OK : result == FAILED ? FAILED : LATE;
    cv_.notify_all();
}

void DedupWriteQueue::set_active(bool on) {
    std::lock_guard<std::mutex> g(mu_);
    active_ = on;
    if (!on) queued_.clear();
    cv_.notify_all();
}

} // namespace radixshmem
