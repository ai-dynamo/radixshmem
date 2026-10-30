// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// WritebackQueue: background-evicted leaves waiting for write-back.

#pragma once

#include "radix_tree.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace radixshmem {

struct WritebackTask {
    uint64_t id = 0;
    HeldEviction h;
};

struct WritebackStats {
    uint64_t queued = 0;        // pushed, not popped yet
    uint64_t popped = 0;        // popped, not completed yet
    uint64_t produced = 0;
    uint64_t completed = 0;
    uint64_t dropped = 0;       // released without a write-back
    uint64_t inflight_slots[NUM_COMPONENT_TYPES] = {};   // covered by queued + popped tasks
};

class WritebackQueue {
public:
    explicit WritebackQueue(RadixShmemTree& tree) : tree_(tree) {}
    ~WritebackQueue() { drop_all(); }

    WritebackQueue(const WritebackQueue&) = delete;
    WritebackQueue& operator=(const WritebackQueue&) = delete;

    void push(std::vector<HeldEviction>&& hs);
    std::vector<WritebackTask> pop(size_t max_tasks, int timeout_ms);
    size_t complete(const std::vector<uint64_t>& ids);
    void drop_all();
    void inflight_slots(uint64_t out[NUM_COMPONENT_TYPES]) const;
    WritebackStats stats() const;

private:
    RadixShmemTree& tree_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<WritebackTask> queued_;
    std::unordered_map<uint64_t, WritebackTask> popped_;
    uint64_t next_id_ = 1;
    WritebackStats st_;

    void account(const HeldEviction& h, bool add);   // under mu_
};

} // namespace radixshmem
