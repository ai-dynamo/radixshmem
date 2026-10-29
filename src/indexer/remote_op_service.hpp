// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "radix_client.h"   // RadixClient

#include <cstdint>
#include <mutex>

namespace radixshmem {

class LocalTreeConnector;

// Wire-agnostic local-tree ops: invalid arguments yield BAD_REQUEST; backends map exceptions to INTERNAL.
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
    // slots[i] holds hashes[start + i]; invalid requests yield REMOTE_FAILED with remote_status = BAD_REQUEST.
    InsertResult  insert(const uint64_t* hashes, int n, int start, int length,
                         const int32_t* slots, bool auto_recycle);
    RemoteStatus  recycle(const int32_t* slots, int num_slots);
    // Drop a query(lock=true) pin over hashes[start, start + length).
    RemoteStatus  unlock(const uint64_t* hashes, int n, int start, int length);

private:
    LocalTreeConnector& tree_;
    RadixClient&        client_;
    std::mutex&         insert_mu_;
};

}  // namespace radixshmem
