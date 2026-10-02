// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "data_pool.hpp"
#include "child_table.hpp"
#include "component_types.h"
#include "mempool.hpp"
#include "slot_lru.hpp"
#include "node_pool.hpp"
#include "types.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace shmradix {

struct MatchedNodeSpanView;
class HealthAuditor;   // src/indexer/health_check.cpp (read-only audit)

// Evicted node reported to the RHT-cleanup callback. first_hash = RHT key
// (block-0 hash); tree_node_id = registered slot id.
struct EvictedNode {
    uint64_t first_hash;
    uint32_t tree_node_id;
};

// Result of one merge_leaf_into_parent() call.
struct MergeOutcome {
    bool     merged        = false;
    bool     full          = false;             // true: leaf absorbed & freed
    uint32_t parent_id     = INVALID_NODE_ID;
    uint64_t old_leaf_hash = 0;
    uint64_t new_leaf_hash = 0;                 // partial only: leaf's new first_hash
};

struct ShmLayoutInfo {
    uint64_t total_size;
    uint64_t header_size;
    uint64_t node_pool_size;
    uint64_t component_state_size;   // NodeComponentState[max_nodes]
    uint64_t data_pool_size;
    uint64_t ht_size;
    uint64_t mempool_size;
    uint64_t swa_mempool_size;
    uint64_t mamba_mempool_size;
    uint64_t swa_lru_size;           // SlotLruEntry[swa_max_blocks]
    uint64_t mamba_lru_size;         // SlotLruEntry[mamba_max_slots]
    uint32_t ht_bucket_count;
    uint32_t ht_entry_capacity;
};

class ShmRadixTree {
public:
    ~ShmRadixTree();

    ShmRadixTree(ShmRadixTree&& other) noexcept;
    ShmRadixTree& operator=(ShmRadixTree&& other) noexcept;
    ShmRadixTree(const ShmRadixTree&) = delete;
    ShmRadixTree& operator=(const ShmRadixTree&) = delete;

    static ShmLayoutInfo compute_layout(const ShmConfig& config);
    static ShmRadixTree create(const std::string& name, const ShmConfig& config);
    static ShmRadixTree attach(const std::string& name,
                               const std::string& hugepage_path = "/dev/hugepages");

    // Release every live PinRecord before detach(), destroy(), or reset().
    void detach();
    void destroy();

    InsertResult insert(const uint64_t* hashes, int n,
                        const int32_t* slots, int num_slots,
                        int start = 0, bool auto_recycle = true,
                        ComponentType component = ComponentType::FULL);
    void flush_lru(const uint32_t* node_ids, int count);

    // Clear all tree data in place under the write lock (empty tree, single root,
    // slots free); layout/magic/locks untouched. Invalidates prior slot/node IDs.
    void reset();

    // Background eviction: non-blocking write_lock, evict up to target_blocks.
    // Returns blocks freed (0 if lock was busy).
    int try_evict_batch(int target_blocks);

    int try_evict_component_batch(ComponentType component, int target_slots);

    std::vector<NodeBoundary> get_node_boundaries(const uint64_t* hashes, int n);

    // Component-aware peer of query(): mask selects enabled components, lock
    // pins whole-node ranges, and update_meta controls Full LRU metadata.
    [[nodiscard]] TreeComponentQueryResult query_components(
        const uint64_t* hashes, int n, uint32_t mask, bool lock = false,
        bool update_meta = true);

    // Positional SWA publish: the slots cover the trailing positions
    // [n - num_slots, n), 1 <= num_slots <= n. The window W only gates queries.
    InsertResult insert_swa(const uint64_t* hashes, int n,
                            const int32_t* slots, int num_slots,
                            bool auto_recycle = true);
    InsertResult insert_mamba(const uint64_t* hashes, int n,
                              const int32_t* slots, int num_slots,
                              bool auto_recycle = true);

    // Component ABI stamped at create time (from ShmConfig).
    ComponentConfig component_config() const;

    static uint64_t now_ts();

    void read_lock();
    void read_unlock();

    // ---- Component ref pair: keys in, release data out; caller-balanced.
    // Match-and-pin the CURRENT tree; fused callers use query_components(lock=true).
    [[nodiscard]] PinRecord inc_ref(const uint64_t* hashes, int n,
                                    uint32_t mask);
    // Legacy-trust release: re-walks the caller's hashes,
    // decrements unconditionally inside the ranges (clamped at zero).
    [[nodiscard]] bool dec_ref(const uint64_t* hashes, int n,
                               PinRecord&& pin);

    // Allocate slot IDs from the component's pool (FULL auto-evicts LRU and may
    // return fewer; SWA/Mamba are all-or-none — insufficient capacity returns 0).
    int allocate_slots(int num_slots, std::vector<int32_t>& out,
                       ComponentType component = ComponentType::FULL);

    // Return allocated slot IDs to the component's pool; safe on never-inserted slots.
    void recycle_slots(const int32_t* slots, int num,
                       ComponentType component = ComponentType::FULL);

    void set_background_evict_ratio(float ratio) { background_evict_ratio_ = ratio; }
    float background_evict_ratio() const { return background_evict_ratio_; }

    void set_evict_policy(EvictPolicy p) { evict_policy_ = p; }
    EvictPolicy evict_policy() const { return evict_policy_; }

    // Optional eviction callback (RHT cleanup), fired AFTER the write lock drops.
    using EvictCallback = std::function<void(const std::vector<EvictedNode>&)>;
    void set_evict_callback(EvictCallback cb) { evict_cb_ = std::move(cb); }
    // True iff node_id is live again with this first hash: the evicted id was
    // recycled (LIFO pool) for the same prefix before its RHT removal ran.
    bool node_head_matches(uint32_t node_id, uint64_t first_hash);

    // Stats
    uint32_t total_radix_nodes() const;
    uint64_t total_blocks() const;
    float data_pool_utilization() const;
    uint32_t node_pool_capacity() const;
    uint32_t node_pool_free() const;
    uint32_t mempool_total() const;
    uint32_t mempool_free() const;
    uint32_t mempool_used() const;
    uint32_t swa_mempool_total() const;
    uint32_t swa_mempool_free() const;
    uint32_t mamba_mempool_total() const;
    uint32_t mamba_mempool_free() const;

    uint64_t insert_success_count() const;
    PeerFaultCounters* peer_faults() const;
    uint64_t insert_error_node_pool_count() const;
    uint64_t insert_error_data_pool_count() const;
    uint64_t insert_error_ht_pool_count() const;
    uint64_t insert_error_mempool_count() const;

    void print_tree() const;

    // RDMA support: expose shared memory region
    void* shm_base() const { return base_; }
    size_t shm_size() const { return size_; }

    // New-node alignment stride == cluster RHT probe granularity (from ShmHeader).
    int register_chunk_size() const { return cs_; }

    static void set_write_op(const char* op);
    // Write locking (public for ClusterNode distributed locking)
    bool try_write_lock();
    void rdma_writer_enter(bool trace);
    void write_lock();
    void write_unlock();

private:
    friend class HealthAuditor;

    ShmRadixTree() = default;

    void init_pools();

    // Allocate the empty root and publish it into the header (create/reset).
    void alloc_root();

    // Core insert; ext_slots positionally bound from `start`, new nodes align to cs_.
    InsertResult insert_impl(const uint64_t* hashes, int n,
                             const int32_t* ext_slots, int ext_num_slots,
                             int start, bool auto_recycle);

    void* base_ = nullptr;
    size_t size_ = 0;
    std::string name_;
    bool is_owner_ = false;
    bool is_hugepage_ = false;
    int fd_ = -1;

    NodePool nodes_;
    DataPool data_;
    ChildTable child_table_;
    Mempool mempool_;
    Mempool swa_mempool_;    // SWA slot-id space (absent unless configured)
    Mempool mamba_mempool_;  // Mamba checkpoint slot-id space
    SlotLru swa_lru_;        // per-slot LRU over the SWA pool
    SlotLru mamba_lru_;
    Mempool* try_component_pool(ComponentType component) noexcept;
    float background_evict_ratio_ = 0.05f;
    EvictPolicy evict_policy_ = EvictPolicy::LRU;
    int cs_ = 0;  // register_chunk_size, read from ShmHeader on create/attach
    // Mask-derived layout (ShmHeader::component_config): the SWA lane exists
    // only with SWA, the NodeComponentState table only with SWA or Mamba.
    bool has_swa_ = false;
    bool has_mamba_ = false;
    bool has_aux_ = false;

    // Eviction-cleanup plumbing (RHT); pending buffer is thread_local.
    EvictCallback evict_cb_;
    static std::vector<EvictedNode>& evicted_pending();
    void flush_evicted();

    // ---- Shared memory accessors ----
    ShmHeader* header() const;

    // ---- Child management ----
    bool add_child(uint32_t parent_id, uint64_t token_hash, uint32_t child_id);
    void remove_child(uint32_t parent_id, uint64_t token_hash, uint32_t child_id);

    // ---- Structural operations ----
    uint32_t split_node(uint32_t node_id, int prefix_length);

    // Auto-merge only-child leaf into parent (inverse of split_node); caller holds
    // write_lock. abs_leaf_start = leaf's block offset from root (cs_ alignment).
    MergeOutcome merge_leaf_into_parent(uint32_t leaf_id, int abs_leaf_start);

    // ---- Leaf list (LRU ordered: head=LRU, tail=MRU) ----
    void leaf_list_add(uint32_t node_id);
    void leaf_list_remove(uint32_t node_id);
    void leaf_list_move_to_tail(uint32_t node_id);
    void leaf_list_insert_by_access_time(uint32_t node_id);
    bool is_in_leaf_list(uint32_t node_id) const;

    // ---- Component slot LRUs (SWA per block, Mamba per checkpoint) ----
    void retire_swa(uint32_t id, std::vector<int32_t>& out);
    void retire_mamba(uint32_t id, std::vector<int32_t>& out);
    void relink_swa(uint32_t id);   // entries := this node's lane; recount
    void aux_on_merge(uint32_t parent_id, uint32_t leaf_id);
    // Query-touch: the hit window's SWA blocks / the Mamba END checkpoint move
    // to the tail of their list.
    void aux_touch(const std::vector<MatchedNodeSpanView>& spans, uint32_t mask,
                   uint32_t common_hit, uint32_t swa_start, uint64_t ts);
    // Pop from the list head, skipping slots on pinned nodes, until need is met.
    int evict_swa_walk(int need_slots);
    int evict_mamba_walk(int need_slots);

    // ---- Eviction internals ----
    // Pin-eligibility: full/swa/mamba ref classes on the node are all zero.
    bool component_refs_zero(uint32_t node_id) const;
    int  evict_node(uint32_t node_id, std::vector<int32_t>& out);
    // protect: node to pin for the walk (INVALID_NODE_ID = none).
    int  evict(int need, uint32_t protect = INVALID_NODE_ID);
    int  evict_walk(int need);
    void post_evict_fixup(uint32_t parent_id);
    void touch_node(uint32_t node_id);

    void print_node(uint32_t node_id, int indent) const;

    // ---- Concurrency helpers ----
    pthread_rwlock_t* rwlock() const;
    void lru_lock();
    void lru_unlock();

    QueryInternalDetail query_internal(const uint64_t* hashes, int n, uint64_t ts,
                                       bool touch = true);
    void touch_node_atomic(uint32_t node_id, uint64_t ts);

    // Node-ID primitives — INTERNAL ONLY (not split-invariant); use hash-path forms externally.
    void inc_ref(uint32_t node_id);
    void dec_ref(uint32_t node_id);
    // Counter primitive: bounded decrement (floors at 0); true if decremented.
    static bool dec_ref(std::atomic<int64_t>& ref) noexcept;

    // Caller holds the read lock. Whole-node coverage: Full over [0, full_hit),
    // SWA/Mamba at common_hit; `pin` is filled before the first increment.
    void inc_ref(const std::vector<MatchedNodeSpanView>& spans,
                 uint32_t mask, uint32_t common_hit, uint32_t swa_start,
                 uint32_t full_hit, PinRecord& pin);

    // Leaf-list move for a Full hit (node timestamps come from query_internal).
    void touch_full(const std::vector<MatchedNodeSpanView>& spans, uint32_t mask,
                   uint32_t full_hit, uint32_t last_node_id);

    // Matched path root->terminal, rebuilt from the terminal via parent_id;
    // only nodes with matched_blocks >= 1. Borrowed views, read lock held.
    std::vector<MatchedNodeSpanView> path_spans(const QueryInternalDetail& m);

    static uint32_t next_power_of_2(uint32_t v);
};

} // namespace shmradix
