// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RemotePeer: a remote-op session to a peer's local tree over one backend.
#include "radix_client.h"          // RemotePeer decl + reply structs

#ifndef RADIXSHMEM_NO_RDMA

#include "bilateral_transport.h"    // IRemoteOpClientTransport

#include <memory>
#include <string>
#include <utility>

namespace radixshmem {

RemotePeer::RemotePeer(
    std::unique_ptr<IRemoteOpClientTransport> transport, std::string endpoint)
    : transport_(std::move(transport)), endpoint_(std::move(endpoint)) {}

RemotePeer::~RemotePeer() = default;
RemotePeer::RemotePeer(RemotePeer&&) noexcept = default;
RemotePeer& RemotePeer::operator=(RemotePeer&&) noexcept = default;

AllocateReply RemotePeer::allocate(int n) { return transport_->allocate(n); }

InsertResult RemotePeer::insert(const uint64_t* hashes, int n,
                                int start, int length, const int32_t* slots,
                                bool auto_recycle) {
    return transport_->insert(hashes, n, start, length, slots, auto_recycle);
}

QueryReply RemotePeer::query(const uint64_t* hashes, int n, bool lock,
                             bool update_meta) {
    return transport_->query(hashes, n, lock, update_meta);
}

RemoteStatus RemotePeer::recycle(const int32_t* slots, int num_slots) {
    return transport_->recycle(slots, num_slots);
}

RemoteStatus RemotePeer::unlock(const uint64_t* hashes, int n, int start,
                                int length) {
    return transport_->unlock(hashes, n, start, length);
}

}  // namespace radixshmem

#endif  // !RADIXSHMEM_NO_RDMA
