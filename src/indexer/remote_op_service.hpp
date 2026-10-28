// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "radix_client.h"   // AllocateReply, QueryReply, InsertResult, RemoteStatus

#include <cstdint>
#include <mutex>

namespace radixshmem {

class LocalTreeConnector;

// Server-side business of the remote-op plane, shared by every backend.
// Validates the op's arguments (BAD_REQUEST on failure) and runs it against
// the local tree; knows nothing about wire formats. Methods may throw; the
// backend maps a throw to INTERNAL.
class RemoteOpService {
public:
    // `insert_mu` serializes allocate and insert (the svc client's RHT publish).
    RemoteOpService(LocalTreeConnector& tree, RadixClient& client,
                    std::mutex& insert_mu)
        : tree_(tree), client_(client), insert_mu_(insert_mu) {}

    // All-or-nothing: on a short pool, error = MEMPOOL_FULL and no slots.
    AllocateReply allocate(int n);
    QueryReply    query(const uint64_t* hashes, int n, bool lock,
                        bool update_meta);
    // slots[i] holds hashes[start + i]. A rejected request reports
    // error = REMOTE_FAILED with remote_status = BAD_REQUEST.
    InsertResult  insert(const uint64_t* hashes, int n, int start,
                         const int32_t* slots, int num_slots,
                         bool auto_recycle);
    RemoteStatus  recycle(const int32_t* slots, int num_slots);
    // Drop a query(lock=true) pin over hashes[start, start + length).
    RemoteStatus  unlock(const uint64_t* hashes, int n, int start, int length);

private:
    LocalTreeConnector& tree_;
    RadixClient&        client_;
    std::mutex&         insert_mu_;
};

}  // namespace radixshmem
