// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "writeback.h"

#include <chrono>

namespace radixshmem {

void WritebackQueue::account(const HeldEviction& h, bool add) {
    uint64_t c[NUM_COMPONENT_TYPES];
    h.covered(c);
    for (int k = 0; k < NUM_COMPONENT_TYPES; k++)
        st_.inflight_slots[k] = add ? st_.inflight_slots[k] + c[k] : st_.inflight_slots[k] - c[k];
}

void WritebackQueue::release(const std::vector<HeldEviction>& hs, bool completed,
                             uint64_t from_popped) {
    // Slots before the count: a lower in-flight count with the slots still held
    // would let the evictor start another batch.
    tree_.release_held(hs);
    std::lock_guard<std::mutex> g(mu_);
    for (const HeldEviction& h : hs) account(h, false);
    releasing_ -= from_popped;
    (completed ? st_.completed : st_.dropped) += hs.size();
}

void WritebackQueue::push(std::vector<HeldEviction>&& hs) {
    if (hs.empty()) return;
    {
        std::lock_guard<std::mutex> g(mu_);
        for (HeldEviction& h : hs) {
            account(h, true);
            queued_.push_back({next_id_++, std::move(h)});
            st_.produced++;
        }
    }
    cv_.notify_all();
}

std::vector<WritebackTask> WritebackQueue::pop(size_t max_tasks, int timeout_ms) {
    std::vector<WritebackTask> out;
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms < 0 ? 0 : timeout_ms),
                 [&] { return !queued_.empty(); });
    const uint64_t gen = tree_.reset_generation();
    while (!queued_.empty() && out.size() < max_tasks) {
        WritebackTask t = std::move(queued_.front());
        queued_.pop_front();
        if (t.h.generation != gen) {   // reset() already freed its slots
            account(t.h, false);
            st_.dropped++;
            continue;
        }
        out.push_back(t);
        popped_.emplace(t.id, std::move(t));
    }
    return out;
}

size_t WritebackQueue::complete(const std::vector<uint64_t>& ids) {
    std::vector<HeldEviction> done;
    {
        std::lock_guard<std::mutex> g(mu_);
        for (uint64_t id : ids) {
            auto it = popped_.find(id);
            if (it == popped_.end()) continue;
            done.push_back(std::move(it->second.h));
            popped_.erase(it);
        }
        releasing_ += done.size();
    }
    release(done, true, done.size());
    return done.size();
}

void WritebackQueue::drop_all() {
    std::vector<HeldEviction> all;
    uint64_t from_popped;
    {
        std::lock_guard<std::mutex> g(mu_);
        from_popped = popped_.size();
        for (auto& kv : popped_) all.push_back(std::move(kv.second.h));
        for (WritebackTask& t : queued_) all.push_back(std::move(t.h));
        queued_.clear();
        popped_.clear();
        releasing_ += from_popped;
    }
    release(all, false, from_popped);
}

void WritebackQueue::inflight_slots(uint64_t out[NUM_COMPONENT_TYPES]) const {
    std::lock_guard<std::mutex> g(mu_);
    for (int k = 0; k < NUM_COMPONENT_TYPES; k++) out[k] = st_.inflight_slots[k];
}

WritebackStats WritebackQueue::stats() const {
    std::lock_guard<std::mutex> g(mu_);
    WritebackStats s = st_;
    s.queued = queued_.size();
    s.popped = popped_.size() + releasing_;
    return s;
}

} // namespace radixshmem
