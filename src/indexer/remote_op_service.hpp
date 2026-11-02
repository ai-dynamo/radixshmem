// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "radix_client.h"   // RadixClient

#include <chrono>
#include <cstdint>
#include <mutex>

namespace radixshmem {

class LocalTreeConnector;

// Local-tree ops return BAD_REQUEST for invalid input; gRPC maps exceptions to INTERNAL.
class RemoteOpService {
public:
    using Deadline = std::chrono::system_clock::time_point;

    // `insert_mu` serializes allocate and insert (the svc client's RHT publish).
    RemoteOpService(LocalTreeConnector& tree, RadixClient& client,
                    std::mutex& insert_mu)
        : tree_(tree), client_(client), insert_mu_(insert_mu) {}

    // All-or-nothing: on a short pool, error = MEMPOOL_FULL and no slots.
    AllocateReply allocate(int n, Deadline deadline = Deadline::max());
    QueryReply    query(const uint64_t* hashes, int n, bool lock,
                        bool update_meta, Deadline deadline = Deadline::max());
    // slots[i] holds hashes[start + i]; invalid requests yield REMOTE_FAILED with remote_status = BAD_REQUEST.
    InsertResult  insert(const uint64_t* hashes, int n, int start, int length,
                         const int32_t* slots, bool auto_recycle);
    RemoteStatus  recycle(const int32_t* slots, int num_slots);
    // Drop a query(lock=true) pin over hashes[start, start + length).
    RemoteStatus  unlock(const uint64_t* hashes, int n, int start, int length,
                         int pin_end, const uint64_t* pin_tail, int pin_tail_n);

private:
    LocalTreeConnector& tree_;
    RadixClient&        client_;
    std::mutex&         insert_mu_;
};

}  // namespace radixshmem
