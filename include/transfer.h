// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace radixshmem {

using SegmentId   = int32_t;
using BatchHandle = uint64_t;

inline constexpr SegmentId   INVALID_SEGMENT = -1;
inline constexpr BatchHandle INVALID_BATCH   = UINT64_MAX;

// Control-path return codes. Anything < 0 is a failure.
inline constexpr int XFER_OK           =  0;
inline constexpr int XFER_EINVAL       = -1;   // malformed argument
inline constexpr int XFER_ENOENT       = -2;   // no such segment / registration
inline constexpr int XFER_EGEOMETRY    = -3;   // segment is not one contiguous region
inline constexpr int XFER_EBACKEND     = -4;   // the underlying engine refused
inline constexpr int XFER_EBUSY        = -5;   // still in use (batches in flight)

// READ pulls bytes from the peer into local memory; WRITE pushes the other way.
enum class XferOp { READ, WRITE };

enum class XferState { PENDING, COMPLETED, FAILED, TIMEOUT, INVALID };

struct TransferDesc {
    XferOp    op            = XferOp::READ;
    void*     local_addr    = nullptr;   // must lie inside a registered range
    SegmentId target        = INVALID_SEGMENT;
    uint64_t  target_offset = 0;
    uint64_t  length        = 0;
};

struct SegmentInfo {
    SegmentId id     = INVALID_SEGMENT;
    uint64_t  base   = 0;   // absolute address of the first published byte
    uint64_t  length = 0;
};

struct TransferConfig {
    std::string backend       = "mooncake";      // the only backend
    std::string metadata      = "P2PHANDSHAKE";  // or "etcd://host:2379"
    std::string local_segment;                   // this node's segment name
    std::string ip_or_host;                      // empty = auto-detect
    uint64_t    rpc_port      = 12345;
    std::string protocol      = "rdma";          // "rdma" | "tcp"
    std::vector<std::string> device_filter;
    bool        publish       = true;
};

class ITransferBackend {
public:
    virtual ~ITransferBackend() = default;

    // --- registration face ---

    virtual int  register_memory(void* addr, uint64_t len,
                                 const std::string& location = "") = 0;
    virtual int  unregister_memory(void* addr) = 0;

    virtual int  open_segment(const std::string& seg_name, SegmentInfo& out) = 0;
    virtual void close_segment(SegmentId id) = 0;

    virtual const std::string& local_segment() const = 0;

    // --- data face ---

    virtual BatchHandle submit(const TransferDesc* descs, int n) = 0;

    virtual XferState poll(BatchHandle h, uint64_t* done_bytes = nullptr) = 0;

    virtual XferState wait(BatchHandle h, int timeout_ms) = 0;

    virtual int  release(BatchHandle h) = 0;

    virtual const char* backend_name() const = 0;
};

// Returns nullptr when the backend name is unknown or when the requested
std::unique_ptr<ITransferBackend> create_transfer_backend(const TransferConfig& cfg);

} // namespace radixshmem
