// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "component_kind.h"
#include "finalize_fn.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <pthread.h>
#include <string>
#include <vector>

namespace radixshmem {

using HashType = uint64_t;

static constexpr uint32_t INVALID_NODE_ID = UINT32_MAX;
static constexpr uint64_t INVALID_OFFSET = UINT64_MAX;
static constexpr uint64_t SHM_MAGIC = 0x534852444958ULL;
static constexpr uint32_t SHM_VERSION = 26;   // 26: peer fault counters
// cluster_ready_magic sentinel — same-host clients spin on it as a local ready flag.
static constexpr uint64_t CLUSTER_MANIFEST_MAGIC = 0x434c5354524d4644ULL; // "CLSTRMFD"
static constexpr size_t HUGEPAGE_SIZE = 2ULL * 1024 * 1024;

static constexpr uint32_t HT_END = UINT32_MAX;  // end-of-chain sentinel
static constexpr int DEFAULT_REGISTER_CHUNK_TOKENS = 4096;

inline uint64_t align_up(uint64_t val, uint64_t alignment) {
    return (val + alignment - 1) & ~(alignment - 1);
}

// Round up to a multiple of `m`. Works for any m (align_up only for powers of 2).
inline uint64_t round_up_multiple(uint64_t val, uint64_t m) {
    return ((val + m - 1) / m) * m;
}

inline bool is_component_config_valid(uint32_t mask,
                                      uint32_t swa_window_blocks) noexcept {
    return (mask & ~COMPONENT_MASK_ALL) == 0 &&
           ((mask & COMPONENT_MASK_SWA) == 0 || swa_window_blocks > 0);
}

static constexpr uint32_t COMPONENT_ABI_VERSION = 7;   // 7: slot LRU

// Written into ShmHeader on create; attach fails closed on any mismatch.
struct ComponentConfig {
    uint32_t abi_version         = COMPONENT_ABI_VERSION;
    uint32_t component_mask      = COMPONENT_MASK_FULL;
    uint32_t logical_block_tokens = 1;   // == ShmConfig::block_size
    uint32_t swa_window_blocks   = 0;
    uint64_t mamba_state_bytes   = 0;
};
static_assert(sizeof(ComponentConfig) == 24, "ComponentConfig must be 24 bytes");

struct alignas(64) NodeComponentState {
    std::atomic<int64_t>  swa_ref_count;
    std::atomic<int64_t>  mamba_ref_count;
    uint32_t swa_live_blocks;
    uint8_t  reserved[44];
};

static constexpr uint32_t INVALID_SLOT_LINK = UINT32_MAX;
struct SlotLruEntry {
    uint32_t prev;
    uint32_t next;
    uint32_t node_id;
    uint32_t idx;      // lane index (Mamba: 0)
    uint64_t ts;
};
static_assert(sizeof(SlotLruEntry) == 24, "SlotLruEntry must be 24 bytes");
static_assert(sizeof(NodeComponentState) == 64,
              "NodeComponentState must be 64 bytes");

struct AuxListAnchors {
    uint32_t head;
    uint32_t tail;
};
static_assert(sizeof(AuxListAnchors) == 8, "AuxListAnchors must be 8 bytes");

inline constexpr size_t data_block_bytes(int num_blocks, bool swa_lane) {
    return (size_t)num_blocks * (swa_lane ? 16 : 12);
}

enum class RemoteStatus : uint16_t {
    OK            = 0,
    BAD_REQUEST   = 1,  // target validation failed: magic/version/length mismatch
    UNKNOWN_OP    = 2,  // op the target doesn't recognize
    TIMEOUT       = 4,  // request() retries exhausted with no reply (target down / net)
    BAD_REPLY     = 5,  // reply received but magic/version/req_id check failed
    BAD_TARGET    = 6,  // remote_* target out of range / == own rank (client synthesized)
    INTERNAL      = 7,
    UNAVAILABLE   = 8,  // target unreachable or not serving (may still have executed the op)
    RESOURCE_EXHAUSTED = 9,  // target refused the op: concurrent admission limit reached
};

struct QueryResult {
    RemoteStatus status = RemoteStatus::OK;

    // Totals
    int total_hit_length = 0;   // local + remote matched blocks
    int local_hit_length = 0;   // matched on the local tree alone

    // Local-tree matched-prefix slots
    std::vector<int32_t> prefix_slots;

    // Remote-tree match details (single-node: remote_node_id == -1)
    int32_t  remote_node_id      = -1;
    int      remote_hit_length   = 0;
    std::vector<int32_t> remote_slots;

    // RDMA stats (only populated when a remote walk actually ran)
    int rdma_read_count   = 0;
    int rdma_atomic_count = 0;

    // One-shot cleanup (dec_ref local+remote for lock=true; no-op for lock=false).
    FinalizeFn finalize;
};

// ==================== Buddy Allocator (DataPool) ====================

// Doubly-linked free block header, stored in-place in free blocks.
struct FreeBlockHeader {
    uint64_t next; // absolute offset (INVALID_OFFSET = end)
    uint64_t prev; // absolute offset (INVALID_OFFSET = start)
};

struct BuddyHeader {
    static constexpr int MIN_DATA_BLOCKS = 4;
    static constexpr int MAX_DATA_BLOCKS = 4096;
    // Buddy geometry is fixed at the widest payload so every shm shares it.
    static constexpr uint32_t MIN_BLOCK_SIZE =
        (uint32_t)data_block_bytes(MIN_DATA_BLOCKS, /*swa_lane=*/true); // 64
    static constexpr int MAX_ORDER = 10;                            // 2^10 = 4096/4
    static constexpr int NUM_ORDERS = MAX_ORDER + 1;
    static constexpr uint8_t ORDER_NONE = 0xFF;

    // Per-order doubly-linked free list heads (absolute offsets)
    uint64_t free_list_head[NUM_ORDERS];

    uint64_t blocks_base;      // absolute offset of data blocks region
    uint64_t blocks_size;      // total data blocks bytes (multiple of max-order block)
    uint64_t bitmap_offset;    // absolute offset of allocation bitmap
    uint64_t order_map_offset; // absolute offset of order map
    uint32_t num_min_blocks;   // blocks_size / MIN_BLOCK_SIZE
    uint32_t _pad;

    std::atomic<uint64_t> allocated_bytes;

    static constexpr uint64_t order_size(int order) {
        return (uint64_t)MIN_BLOCK_SIZE << order;
    }
};
static_assert(BuddyHeader::MIN_BLOCK_SIZE >= sizeof(FreeBlockHeader),
              "MIN_BLOCK_SIZE must fit FreeBlockHeader");
static_assert((BuddyHeader::MAX_DATA_BLOCKS / BuddyHeader::MIN_DATA_BLOCKS) ==
              (1 << BuddyHeader::MAX_ORDER),
              "MAX_ORDER must equal log2(MAX_DATA_BLOCKS / MIN_DATA_BLOCKS)");

// ==================== Mempool ====================

struct alignas(32) MempoolHeader {
    uint32_t total_blocks;
    uint32_t stack_top;      // free_stack[0..stack_top-1] are available
    uint32_t _pad[6];
};

// ==================== Global Child Hash Table (Chaining) ====================

// ==================== ShmHeader ====================

// One-sided index-plane ops lost to unreachable peers, summed over every
// process attached to the node's tree.
struct PeerFaultCounters {
    std::atomic<uint64_t> down_events;   // peers marked unreachable
    std::atomic<uint64_t> rht_writes;    // RHT publishes dropped
    std::atomic<uint64_t> rht_removes;   // RHT removals dropped
    std::atomic<uint64_t> rht_lookups;   // RHT probes answered as misses
    std::atomic<uint64_t> walks;         // remote walks skipped

    void clear() {
        for (auto* c : {&down_events, &rht_writes, &rht_removes, &rht_lookups, &walks})
            c->store(0, std::memory_order_relaxed);
    }
};

struct ShmHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t root_node_id;

    uint32_t node_pool_capacity;
    std::atomic<uint32_t> node_pool_size;
    uint32_t node_free_head;
    uint32_t leaf_list_head;
    uint32_t leaf_list_tail;

    std::atomic<uint64_t> access_counter;

    uint64_t node_pool_offset;

    // DataPool: starts with BuddyHeader + bitmap + order_map + data blocks
    uint64_t data_pool_offset;
    uint64_t data_pool_total_size;

    // Global child hash table (chaining)
    uint64_t global_ht_offset;
    uint32_t global_ht_bucket_count;   // power of 2
    uint32_t global_ht_entry_capacity; // max entries in pool
    uint32_t global_ht_free_head;      // free list head index
    uint32_t _ht_pad;

    std::atomic<uint64_t> total_blocks;

    uint64_t mempool_offset;
    uint32_t mempool_total_blocks;
    uint32_t _mempool_pad;

    std::atomic<uint64_t> insert_success;
    std::atomic<uint64_t> insert_error_node_pool;
    std::atomic<uint64_t> insert_error_data_pool;
    std::atomic<uint64_t> insert_error_ht_pool;
    std::atomic<uint64_t> insert_error_mempool;

    PeerFaultCounters peer_faults;

    // Eviction config (written by shm owner on create, read by every attach).
    float    background_evict_ratio;
    uint32_t evict_policy;

    // Block size (tokens per block); written on create, read by every attach.
    int32_t  block_size;
    uint32_t _block_size_pad;

    // New-node boundary alignment stride (cluster RHT probe granularity); create-time.
    int32_t  register_chunk_size;
    uint32_t _rcs_pad;

    // ---- Component ABI (create-time; attach fails closed on mismatch) ----
    ComponentConfig component_config;
    uint64_t component_state_offset;       // NodeComponentState[node_pool_capacity]
    uint32_t component_state_entry_size;   // == sizeof(NodeComponentState)
    uint32_t _cst_pad;
    uint64_t swa_mempool_offset;           // 0 = absent
    uint32_t swa_mempool_total_blocks;
    uint32_t _swa_mp_pad;
    uint64_t mamba_mempool_offset;         // 0 = absent
    uint32_t mamba_mempool_total_blocks;
    uint32_t _mamba_mp_pad;
    uint64_t swa_lru_offset;               // SlotLruEntry[swa_mempool_total_blocks]; 0 = absent
    uint64_t mamba_lru_offset;             // SlotLruEntry[mamba_mempool_total_blocks]
    AuxListAnchors swa_list;               // slot ids
    AuxListAnchors mamba_list;

    alignas(64) std::atomic<int64_t> rdma_write_flag{0};
    char _rwf_pad[56];
    alignas(64) std::atomic<int64_t> rdma_reader_count{0};
    char _rrc_pad[56];

    uint32_t is_distributed;                      // 1 = joined a cluster, 0 = standalone
    uint32_t my_rank;                             // this shm owner's rank
    std::atomic<uint64_t> cluster_ready_magic{0}; // CLUSTER_MANIFEST_MAGIC once ready
    char     registry[64];                        // "etcd://host:port"
    char     cluster_id[64];                      // etcd key namespace
    char     node_name[64];                       // this shm owner's etcd identity

    pthread_rwlock_t rwlock;
    pthread_mutex_t  lru_mutex;
};
static constexpr uint64_t RDMA_READER_COUNT_DELTA = 64;
static_assert(offsetof(ShmHeader, rdma_reader_count) - offsetof(ShmHeader, rdma_write_flag)
              == RDMA_READER_COUNT_DELTA, "reader count must sit one line after the write flag");

// ==================== TreeNode ====================

struct alignas(64) TreeNode {
    uint32_t parent_id;
    uint32_t num_blocks;
    uint32_t children_count;
    // Node-END Mamba checkpoint; writes require the tree write lock.
    // Reuses _rsv_ready to keep TreeNode 64 bytes.
    int32_t  mamba_checkpoint_slot;

    uint64_t data_offset;                   // into DataPool data blocks
    std::atomic<uint64_t> last_access_time;

    std::atomic<int64_t> ref_count;         // 8B aligned for RDMA Fetch-and-Add

    uint32_t next_free;
    uint32_t leaf_prev;
    uint32_t leaf_next;
    uint32_t first_child_entry;   // head of per-parent sibling chain in HT entry pool
    uint64_t _pad_version;
};
static_assert(sizeof(TreeNode) == 64, "TreeNode must be 64 bytes");

// ==================== ChainHTEntry ====================

struct ChainHTEntry {
    uint64_t token_hash;
    uint32_t parent_id;
    uint32_t child_node_id;
    uint32_t hash_next;       // next in hash bucket chain (or free list next when free)
    uint32_t sibling_next;    // next child of same parent
};
static_assert(sizeof(ChainHTEntry) == 24, "ChainHTEntry must be 24 bytes");

// ==================== Config / Result ====================

enum class InsertError : uint32_t {
    OK             = 0,
    NODE_POOL_FULL = 1,
    DATA_POOL_FULL = 2,
    HT_POOL_FULL   = 3,
    MEMPOOL_FULL   = 4,   // out of slots; insert may be partial (unlanded slots in unused_slots)
    REMOTE_FAILED  = 5,   // remote_insert only: transport error / bad target / request rejected
    BAD_REQUEST       = 6,   // component publish: component disabled / bad window shape
    FULL_PATH_MISSING = 7,   // component publish: [0, n) Full path incomplete
};

enum class EvictPolicy : uint32_t {
    LRU = 0,
};

struct ShmConfig {
    uint32_t max_nodes = 0;     // 0 = auto: max_blocks * min(block_size / 128, 1)
    uint32_t max_blocks = 0;
    int block_size = 1;         // tokens per block (for compute_hashes)
    int register_chunk_tokens = DEFAULT_REGISTER_CHUNK_TOKENS;   // 0 = unaligned nodes
    float background_evict_ratio = 0.05f;
    EvictPolicy evict_policy = EvictPolicy::LRU;
    float data_pool_ratio = 4.0f;
    std::string hugepage_path = "/dev/hugepages";

    // Component ABI (defaults = Full-only, no auxiliary pools).
    uint32_t component_mask   = COMPONENT_MASK_FULL;
    uint32_t swa_window_blocks = 0;
    uint64_t mamba_state_bytes = 0;
    uint32_t swa_max_blocks   = 0;   // SWA slot-id space (SwaMempool capacity)
    uint32_t mamba_max_slots  = 0;   // Mamba checkpoint slot-id space

    // register_chunk_tokens in blocks (cs): node-boundary alignment and RHT
    // registration stride, at least one block; 0 stays 0 (unaligned).
    int effective_register_chunk_size() const {
        if (register_chunk_tokens <= 0) return 0;
        return std::max(1, register_chunk_tokens / std::max(block_size, 1));
    }

    uint32_t effective_max_nodes() const {
        if (max_nodes > 0) return max_nodes;
        if (max_blocks > 0) {
            double ratio = std::min((double)block_size / 128.0, 1.0);
            return std::max(1024u, (uint32_t)((uint64_t)max_blocks * ratio));
        }
        return 200000;
    }

    uint64_t effective_data_pool_bytes() const {
        const size_t per_block =
            data_block_bytes(1, component_has_swa(component_mask));
        if (max_blocks > 0)
            return (uint64_t)(max_blocks * per_block * data_pool_ratio);
        return (uint64_t)effective_max_nodes() * 4 * per_block;
    }
};

// Internal matcher state from query_internal() (not public): hit length plus
// the terminal node, from which path_spans() rebuilds the matched path.
struct QueryInternalDetail {
    int matched_blocks = 0;
    uint32_t last_node_id = INVALID_NODE_ID;
    int last_node_matched_blocks = 0;
};

struct AllocateReply {
    RemoteStatus status = RemoteStatus::OK;
    InsertError  error  = InsertError::OK;   // MEMPOOL_FULL when status is OK but nothing was allocated
    std::vector<int32_t> slots;   // target-local reserved slot ids (empty on failure)
};

struct QueryReply {
    RemoteStatus status = RemoteStatus::OK;
    int hit_length       = 0;   // B local-tree total matched blocks
    std::vector<int32_t> slots; // target-local physical slot ids (len == hit_length)
};

struct InsertResult {
    InsertError error = InsertError::OK;
    // remote_insert only: the comms status behind error == REMOTE_FAILED.
    RemoteStatus remote_status = RemoteStatus::OK;
    // Slots not consumed (leading redundant prefix + unlanded suffix, merged).
    // Always populated; auto-recycled inside insert when auto_recycle=true.
    std::vector<int32_t> unused_slots;
    // Component publish only: a boundary split changed the topology, so the
    // owner should republish the [0, n) RHT chain.
    bool topology_changed = false;
    // Full insert that split a node off the inserted path: the suffix keeps
    // the node id under a new first hash and needs its own route.
    uint32_t split_node_id = INVALID_NODE_ID;
    uint32_t split_num_blocks = 0;
    uint64_t split_first_hash = 0;
};

struct NodeBoundary {
    int offset;
    int num_blocks;          // blocks of this node matched by the query
    uint64_t first_hash;
    uint32_t tree_node_id;
    uint32_t node_blocks;    // the node's full length (RHT publishes this)
};

} // namespace radixshmem
