// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dedup_writer.h"
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
                    std::mutex& insert_mu);

    // All-or-nothing: on a short pool, error = MEMPOOL_FULL and no slots.
    AllocateReply allocate(int n, Deadline deadline = Deadline::max(),
                           ComponentType component = ComponentType::FULL);
    AllocateComponentsReply allocate_components(const std::vector<ComponentCount>& components,
                                                Deadline deadline = Deadline::max());
    QueryReply    query(const uint64_t* hashes, int n, bool lock,
                        bool update_meta, Deadline deadline = Deadline::max());
    // slots[i] holds hashes[start + i]; invalid requests yield REMOTE_FAILED with remote_status = BAD_REQUEST.
    InsertResult  insert(const uint64_t* hashes, int n, int start, int length,
                         const int32_t* slots, bool auto_recycle,
                         ComponentType component = ComponentType::FULL);
    InsertPathResult insert_path(const uint64_t* hashes, int n, int start, int length,
                                 const int32_t* full_slots, const std::vector<ComponentRun>& runs,
                                 bool auto_recycle);
    RemoteStatus  recycle(const int32_t* slots, int num_slots,
                          ComponentType component = ComponentType::FULL);
    // Drop a query(lock=true) pin over hashes[start, start + length).
    RemoteStatus  unlock(const uint64_t* hashes, int n, int start, int length,
                         int pin_end, const uint64_t* pin_tail, int pin_tail_n);
    // With landing slots in `req`, also writes what blocks the drop into them through
    // the dedup writer; without an active writer it only drops.
    DedupReply    dedup_prefix(const uint64_t* hashes, int n, const DedupRequest& req = {},
                               Deadline deadline = Deadline::max());
    void set_dedup_writer(DedupWriter* w) { writer_ = w; }

private:
    LocalTreeConnector& tree_;
    RadixClient&        client_;
    std::mutex&         insert_mu_;
    bool                dedup_off_;
    DedupWriter*        writer_ = nullptr;
};

}  // namespace radixshmem
