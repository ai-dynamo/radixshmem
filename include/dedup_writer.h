// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "component_kind.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace radixshmem {

// Copies a dedup source's slots into the landing slots of the puller `rank`.
class DedupWriter {
public:
    enum Result { OK = 0, FAILED = 1, LATE = 2 };   // LATE: writes may still land
    virtual ~DedupWriter() = default;
    virtual bool active() = 0;
    virtual Result write(int rank, const std::vector<int32_t> (&local)[NUM_COMPONENT_TYPES],
                         const std::vector<int32_t> (&remote)[NUM_COMPONENT_TYPES],
                         int timeout_ms) = 0;
};

struct DedupWrite {
    uint64_t id = 0;
    int rank = -1;
    int timeout_ms = 0;
    std::vector<int32_t> local[NUM_COMPONENT_TYPES];
    std::vector<int32_t> remote[NUM_COMPONENT_TYPES];
};

// write() queues the copy for the server's transfer loop, which pop()s and complete()s
// it, and waits for that; while inactive it fails at once.
class DedupWriteQueue : public DedupWriter {
public:
    bool active() override;
    Result write(int rank, const std::vector<int32_t> (&local)[NUM_COMPONENT_TYPES],
                 const std::vector<int32_t> (&remote)[NUM_COMPONENT_TYPES],
                 int timeout_ms) override;
    std::vector<DedupWrite> pop(size_t max_writes, int timeout_ms);
    void complete(uint64_t id, int result);
    void set_active(bool on);

private:
    struct Pending {
        DedupWrite w;
        bool popped = false;
        bool done = false;
        Result result = FAILED;
    };
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<uint64_t> queued_;
    std::unordered_map<uint64_t, Pending> pending_;
    uint64_t next_id_ = 1;
    bool active_ = false;
};

} // namespace radixshmem
