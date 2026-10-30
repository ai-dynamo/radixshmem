// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

namespace radixshmem {

class RadixShmemTree;

// Bit flags returned by check_health(); 0 == healthy.
enum HealthFlag : uint32_t {
    HEALTH_OK                = 0,
    HEALTH_RWLOCK_READ_HELD  = 1u << 0,   // process-shared rwlock: readers inside
    HEALTH_RWLOCK_WRITE_HELD = 1u << 1,   // process-shared rwlock: writer inside / waiting
    HEALTH_LRU_LOCK_HELD     = 1u << 2,   // lru_mutex locked
    HEALTH_RDMA_LOCK_HELD    = 1u << 3,   // rdma write flag set or remote readers registered
    HEALTH_FULL_SLOT_LEAK    = 1u << 4,   // FULL mempool used != slots referenced by the tree
    HEALTH_SWA_SLOT_LEAK     = 1u << 5,   // SWA pool used != SWA slots in the tree
    HEALTH_MAMBA_SLOT_LEAK   = 1u << 6,   // Mamba pool used != checkpoints in the tree
    HEALTH_REF_LEAK          = 1u << 7,   // some node has full/swa/mamba ref_count != 0
    HEALTH_NODE_LEAK         = 1u << 8,   // nodes reachable from root != node_pool_size
    HEALTH_NOT_ATTACHED      = 1u << 31,  // no local tree to audit (observer / pre-bootstrap)
};

struct HealthReport {
    uint32_t flags = HEALTH_OK;

    // ---- locks (snapshot, nothing acquired) ----
    bool     locks_inspected        = false;  // false on a non-glibc libc: lock checks skipped
    uint32_t rwlock_readers         = 0;
    uint32_t rwlock_writers_waiting = 0;
    bool     rwlock_writer_held     = false;
    bool     lru_locked             = false;
    int64_t  rdma_rwlock            = 0;      // write_flag | reader_count; bit 63 = writer, low bits = RDMA readers

    // ---- slot accounting, one entry per component pool ----
    struct Pool {
        bool     enabled   = false;  // pool exists in this shm
        uint32_t total     = 0;
        uint32_t used      = 0;      // mempool: total - free
        uint32_t in_tree   = 0;      // slot references found walking the tree
        uint32_t leaked    = 0;      // marked used in the pool, not referenced by any node
        uint32_t dangling  = 0;      // referenced by a node but free / out of range in the pool
        uint32_t duplicate = 0;      // referenced by more than one node position
        uint32_t held      = 0;      // held for write-back; also counted as leaked
        bool ok() const { return !leaked && !dangling && !duplicate && used == in_tree; }
    };
    Pool full, swa, mamba;

    // ---- pins ----
    uint32_t nodes_with_refs = 0;
    int64_t  ref_total       = 0;    // sum of full + swa + mamba refs over all nodes

    // ---- nodes ----
    uint32_t nodes_reachable = 0;    // root walk
    uint32_t nodes_in_pool   = 0;    // ShmHeader::node_pool_size

    bool ok() const { return flags == HEALTH_OK; }
    // One line per failed check; "healthy" when ok().
    std::string describe() const;
};

// Audit `tree` without taking any lock. Returns the report; report.flags is
// the failure code (0 == success).
HealthReport check_health(const RadixShmemTree& tree);

} // namespace radixshmem
