// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RHT core: the shared layer under both endpoints (holder and consumer).
// PRIVATE header (src/indexer/rht): pulls in the verbs transport surface.
// Callers outside the module use shmradix/rht_endpoint.h + rht_types.h only.
//
//   RhtShard   — one shard's standalone POSIX shm region (header + flat uint64
//                bucket array). Created on holder ranks, RDMA-registered, and
//                addressed by remote peers via (base_addr, rkey).
//   RhtEngine  — the routing + read/write engine over the shard map: shard/
//                bucket/tag routing, W-slot chaining, local-atomic vs one-sided
//                RDMA per shard. Endpoints assemble one and forward to it.
//
// Public value types (RHTValue, RhtInsertEntry, RhtRemoveEntry, RhtShardAddr)
// live in shmradix/rht_types.h.

#pragma once

#include "shmradix/rht_types.h"
#include "peer_breaker.hpp"
#include "rdma_transport.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <string>
#include <vector>

namespace shmradix {

// ==================== RhtShard — one shard's shm base ====================

static constexpr uint64_t RHT_SHARD_MAGIC = 0x52485348'41524400ULL; // "RHSHARD\0"

static constexpr uint32_t RHT_MIN_CAPACITY = 16;
uint32_t rht_normalize_capacity(uint32_t requested);

struct RhtSlot {
    uint64_t head;
    uint64_t ext;
};
static_assert(sizeof(RhtSlot) == 16, "RhtSlot must be 16 bytes");
static constexpr uint32_t RHT_SLOT_BYTES = sizeof(RhtSlot);

struct RhtShardHeader {
    uint64_t magic;         // RHT_SHARD_MAGIC once initialized
    uint32_t capacity;      // bucket count (power of 2)
    uint32_t slots_per_bucket;  // W: 8-byte slots per bucket (chaining depth)
    uint64_t data_offset;   // byte offset from region base to slot[0]
    // Identity — lets the holder's client process CONFIRM it mapped the
    // intended object (right shard id and owning rank) when it attaches.
    int32_t  shard_id;      // which shard S this region holds
    int32_t  holder_rank;   // owning rank
    uint32_t slot_bytes;    // RHT_SLOT_BYTES; attach rejects a mismatched layout
    uint32_t _pad;
    uint64_t _reserved[3];
};
static_assert(sizeof(RhtShardHeader) == 64, "RhtShardHeader must be 64 bytes");

class RhtShard {
public:
    RhtShard() = default;
    ~RhtShard();

    RhtShard(const RhtShard&) = delete;
    RhtShard& operator=(const RhtShard&) = delete;
    RhtShard(RhtShard&& o) noexcept;
    RhtShard& operator=(RhtShard&& o) noexcept;

    static RhtShard create(const std::string& name, uint32_t capacity,
                           uint32_t slots_per_bucket,
                           int shard_id, int holder_rank);

    static RhtShard attach(const std::string& name, int timeout_sec = 60);

    // Zero every slot (empty entry == 0); the header/identity is preserved.
    void reset();

    int shard_id()    const { return shard_id_; }
    int holder_rank() const { return holder_rank_; }

    // Pointer to slot[0] (RhtSlot array, capacity*W cells). Used for
    // same-process local reads/writes on the holder.
    RhtSlot*  buckets() const { return buckets_; }
    uint32_t  capacity() const { return capacity_; }
    uint32_t  slots_per_bucket() const { return slots_per_bucket_; }

    // Mapping base + total size — what the caller registers as the RHT-MR.
    void*  base()  const { return base_; }
    size_t size()  const { return size_; }

    // Byte offset from `base()` to slot[0].
    uint64_t data_offset() const;

    bool valid() const { return base_ != nullptr; }

private:
    int         fd_   = -1;
    void*       base_ = nullptr;
    size_t      size_ = 0;
    RhtSlot*    buckets_ = nullptr;
    uint32_t    capacity_ = 0;
    uint32_t    slots_per_bucket_ = 1;
    int         shard_id_ = -1;
    int         holder_rank_ = -1;
    bool        is_owner_ = false;
    std::string name_;
};

// ==================== RhtEngine — routing + read/write ====================

class RhtEngine {
public:
    RhtEngine(int num_shards, uint32_t shard_capacity,
              uint32_t slots_per_bucket, std::vector<void*> shard_ptrs);
    ~RhtEngine() = default;

    RhtEngine(const RhtEngine&) = delete;
    RhtEngine& operator=(const RhtEngine&) = delete;

    void enable_rdma(std::vector<IRdmaEndpoint*> endpoints,
                     std::vector<RdmaMrInfo> shard_mrs,
                     void* local_buf, uint32_t local_lkey,
                     size_t local_buf_bytes);

    void set_fault_handling(std::vector<int> holder_of_shard, PeerBreaker* breaker,
                            std::function<IRdmaEndpoint*(int)> connect,
                            std::function<void(int)> release);
    void holder_changed(int holder, const RdmaMrInfo* mr);

    void lookup_batch(const uint64_t* keys, int count,
                      std::vector<RHTValue>* results);
    void dump_bucket(uint64_t key, std::vector<RhtSlot>& out);

    void remove_batch(const RhtRemoveEntry* entries, int count);

    void insert_batch(const RhtInsertEntry* entries, int count);

    // Drain all pending async writes across every remote endpoint.
    void flush();

    int num_shards() const { return num_shards_; }

private:
    bool remove(uint64_t key, int32_t expected_node_id,
                uint32_t expected_tree_node_id);

    // Pack head: [key_tag(16) | node_id+1(16) | tree_node_id(32)] = 8 bytes
    // Empty = 0 (node_id+1 == 0 means no entry).
    static uint64_t pack(uint16_t tag, int32_t node_id, uint32_t tree_node_id);
    static bool unpack(uint64_t head, uint16_t expected_tag,
                       int32_t& node_id, uint32_t& tree_node_id);
    // Pack ext: [chk(16) | 0(16) | num_blocks(32)]; chk seals (head, num_blocks).
    static uint64_t pack_ext(uint64_t head, uint32_t num_blocks);
    static bool unpack_ext(uint64_t head, uint64_t ext, uint32_t& num_blocks);
    static RhtSlot make_slot(uint64_t head, uint32_t num_blocks) {
        return RhtSlot{head, head ? pack_ext(head, num_blocks) : 0};
    }

    int shard_of(uint64_t key) const;
    uint32_t bucket_of(uint64_t key) const;
    uint16_t tag_of(uint64_t key) const;

    // Read all W cells of a bucket into out[0..W). Local = 2 relaxed loads per
    // cell; remote = one RDMA read of W*16 bytes. false: holder unreachable.
    bool read_bucket(int shard, uint32_t bucket, RhtSlot* out);
    // Write a single cell (two relaxed stores local / 16B inline RDMA write remote).
    bool write_slot(int shard, uint32_t bucket, uint32_t slot, RhtSlot value);

    using Clock = std::chrono::steady_clock;
    // Endpoint of a remote shard, or nullptr while its holder is skipped.
    IRdmaEndpoint* remote_ep(int shard);
    void release_holder(int holder);
    void fault(int shard, const char* op, const std::exception& e, Clock::time_point t0);
    void dropped(int shard, PeerBreaker::Op op, uint64_t n = 1);
    void ok(int shard);

    int num_shards_;
    uint32_t shard_capacity_;
    uint32_t shard_mask_;
    uint32_t slots_per_bucket_;   // W
    std::vector<void*> shards_;

    std::vector<IRdmaEndpoint*> endpoints_;
    std::vector<RdmaMrInfo> shard_mrs_;
    void* rdma_buf_ = nullptr;
    uint32_t rdma_lkey_ = 0;
    size_t rdma_buf_bytes_ = 0;

    std::vector<int> holder_;
    PeerBreaker* breaker_ = nullptr;
    std::function<IRdmaEndpoint*(int)> connect_;
    std::function<void(int)> release_;
};

} // namespace shmradix
