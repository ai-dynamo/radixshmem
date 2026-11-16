// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Test-only engine surface for compiling the real MooncakeBackend without an
// installed Mooncake library. It injects freeBatchID failures; it does not
// simulate transport progress or validate Mooncake's implementation.

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mooncake {

using BatchID = uint64_t;
using SegmentHandle = uint64_t;
inline constexpr BatchID INVALID_BATCH_ID = UINT64_MAX;
inline constexpr const char* kWildcardLocation = "*";

enum class TransferStatusEnum { WAITING, PENDING, COMPLETED, FAILED, TIMEOUT, CANCELED, INVALID };
struct TransferStatus {
    TransferStatusEnum s = TransferStatusEnum::PENDING;
    uint64_t transferred_bytes = 0;
};
struct TransferRequest {
    enum Opcode { READ, WRITE } opcode;
    void* source;
    SegmentHandle target_id;
    uint64_t target_offset, length;
};

struct Status {
    enum Code { OK, BUSY, ERROR } code = OK;
    bool ok() const { return code == OK; }
    bool IsBatchBusy() const { return code == BUSY; }
    std::string message() const { return "injected freeBatchID failure"; }
};

namespace test {
inline Status::Code release_code = Status::OK;
inline TransferStatusEnum transfer_state = TransferStatusEnum::PENDING;
inline std::vector<BatchID> release_calls;
inline int retired_batches = 0;
} // namespace test

struct SegmentDesc {
    struct Buffer { uint64_t addr, length; };
    std::vector<Buffer> buffers{{0x100000, 4096}};
};
struct Metadata {
    std::shared_ptr<SegmentDesc> getSegmentDescByID(SegmentHandle) {
        return std::make_shared<SegmentDesc>();
    }
};

class TransferEngine {
public:
    explicit TransferEngine(bool, const std::vector<std::string>& = {}) {}
    int init(const std::string&, const std::string& name, const std::string&, uint64_t) {
        name_ = name;
        return 0;
    }
    void* installTransport(const std::string&, void*) { return this; }
    const std::string& getLocalIpAndPort() const { return name_; }
    int registerLocalMemory(void*, uint64_t, const std::string&, bool, bool) { return 0; }
    int unregisterLocalMemory(void*) { return 0; }
    SegmentHandle openSegment(const std::string&) { return 1; }
    void closeSegment(SegmentHandle) {}
    Metadata* getMetadata() { return &metadata_; }

    BatchID allocateBatchID(size_t) {
        const BatchID id = next_++;
        batches_[id] = 0;
        return id;
    }
    Status submitTransfer(BatchID id, const std::vector<TransferRequest>& requests) {
        auto it = batches_.find(id);
        if (it == batches_.end()) return {Status::ERROR};
        for (const auto& r : requests) it->second += r.length;
        return {};
    }
    Status getBatchTransferStatus(BatchID id, TransferStatus& out) {
        auto it = batches_.find(id);
        if (it == batches_.end()) return {Status::ERROR};
        out.s = test::transfer_state;
        out.transferred_bytes = out.s == TransferStatusEnum::COMPLETED ? it->second : 0;
        return {};
    }
    Status freeBatchID(BatchID id) {
        test::release_calls.push_back(id);
        if (!batches_.count(id)) return {Status::ERROR};
        if (test::release_code != Status::OK) return {test::release_code};
        batches_.erase(id);
        ++test::retired_batches;
        return {};
    }

private:
    std::string name_;
    Metadata metadata_;
    BatchID next_ = 100;
    std::unordered_map<BatchID, uint64_t> batches_;
};

} // namespace mooncake
