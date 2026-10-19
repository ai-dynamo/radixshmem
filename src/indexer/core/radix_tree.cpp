// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radix_tree.h"
#include "cluster_manifest.h"
#include "component_set.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef __x86_64__
#include <immintrin.h>
#include <cstdlib>
#include <sys/time.h>

namespace {
long lock_trace_threshold_us() {
    static long t = [] { const char* e = ::getenv("RADIXSHMEM_LOCK_TRACE"); return e ? atol(e) : -1L; }();
    return t;
}
uint64_t lt_now_us() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
thread_local const char* g_write_op = "?";
thread_local uint64_t g_lt_req = 0, g_lt_wr = 0, g_lt_drain = 0;
thread_local long long g_lt_readers = 0;
void lt_check_stuck(uint64_t prev, const char* where) {
    if (lock_trace_threshold_us() < 0 || !(prev & (1ULL << 63))) return;
    struct timeval tv; gettimeofday(&tv, nullptr);
    fprintf(stderr, "[locktrace %ld.%03ld pid=%d] STUCK_BIT at %s op=%s prev=0x%llx readers=%lld\n",
            (long)tv.tv_sec, (long)tv.tv_usec / 1000, (int)getpid(), where, g_write_op,
            (unsigned long long)prev, (long long)(prev & ~(1ULL << 63)));
}
}
#endif

namespace radixshmem {

static void validate_component_config(uint32_t mask, uint32_t swa_window_blocks,
                                      const char* operation) {
    if (!is_component_config_valid(mask, swa_window_blocks))
        throw std::runtime_error(
            std::string(operation) +
            ": invalid ComponentConfig (unknown component bits, or SWA "
            "enabled with swa_window_blocks == 0)");
}

// ==================== Lifecycle ====================

RadixShmemTree::~RadixShmemTree() { detach(); }

RadixShmemTree::RadixShmemTree(RadixShmemTree&& o) noexcept
    : base_(o.base_), size_(o.size_), name_(std::move(o.name_)),
      is_owner_(o.is_owner_), is_hugepage_(o.is_hugepage_), fd_(o.fd_),
      nodes_(o.nodes_), data_(o.data_), child_table_(o.child_table_),
      mempool_(o.mempool_), swa_mempool_(o.swa_mempool_),
      mamba_mempool_(o.mamba_mempool_), swa_lru_(o.swa_lru_),
      mamba_lru_(o.mamba_lru_), background_evict_ratio_(o.background_evict_ratio_),
      evict_policy_(o.evict_policy_), cs_(o.cs_), has_swa_(o.has_swa_),
      has_mamba_(o.has_mamba_), has_aux_(o.has_aux_),
      evict_cb_(std::move(o.evict_cb_)),
      evict_observers_(std::move(o.evict_observers_)),
      kvevent_writer_(o.kvevent_writer_) {
    o.base_ = nullptr;
    o.fd_ = -1;
    o.kvevent_writer_ = KvEventRingWriter();
}

RadixShmemTree& RadixShmemTree::operator=(RadixShmemTree&& o) noexcept {
    if (this != &o) {
        detach();
        base_ = o.base_;
        size_ = o.size_;
        name_ = std::move(o.name_);
        is_owner_ = o.is_owner_;
        is_hugepage_ = o.is_hugepage_;
        fd_ = o.fd_;
        nodes_ = o.nodes_;
        data_ = o.data_;
        child_table_ = o.child_table_;
        mempool_ = o.mempool_;
        swa_mempool_ = o.swa_mempool_;
        mamba_mempool_ = o.mamba_mempool_;
        swa_lru_ = o.swa_lru_;
        mamba_lru_ = o.mamba_lru_;
        background_evict_ratio_ = o.background_evict_ratio_;
        evict_policy_ = o.evict_policy_;
        cs_ = o.cs_;
        has_swa_ = o.has_swa_;
        has_mamba_ = o.has_mamba_;
        has_aux_ = o.has_aux_;
        evict_cb_ = std::move(o.evict_cb_);
        evict_observers_ = std::move(o.evict_observers_);
        kvevent_writer_ = o.kvevent_writer_;
        o.base_ = nullptr;
        o.fd_ = -1;
        o.kvevent_writer_ = KvEventRingWriter();
    }
    return *this;
}

uint32_t RadixShmemTree::next_power_of_2(uint32_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

void RadixShmemTree::init_pools() {
    ShmHeader* hdr = header();
    const uint32_t cmask = hdr->component_config.component_mask;
    has_swa_   = component_has_swa(cmask);
    has_mamba_ = component_has_mamba(cmask);
    has_aux_   = component_has_aux(cmask);

    auto* node_base = reinterpret_cast<TreeNode*>(
        (char*)base_ + hdr->node_pool_offset);
    auto* cst_base = hdr->component_state_offset
        ? reinterpret_cast<NodeComponentState*>(
              (char*)base_ + hdr->component_state_offset)
        : nullptr;
    nodes_ = NodePool(node_base, hdr->node_pool_capacity,
                      &hdr->node_free_head, &hdr->node_pool_size, cst_base);

    auto* bh = reinterpret_cast<BuddyHeader*>(
        (char*)base_ + hdr->data_pool_offset);
    data_ = DataPool(base_, bh, has_swa_);

    auto* ht_base = (char*)base_ + hdr->global_ht_offset;
    auto* ht_buckets = reinterpret_cast<uint32_t*>(ht_base);
    auto* ht_entries = reinterpret_cast<ChainHTEntry*>(
        ht_base + (uint64_t)hdr->global_ht_bucket_count * sizeof(uint32_t));
    child_table_ = ChildTable(ht_buckets, ht_entries,
                  hdr->global_ht_bucket_count, hdr->global_ht_entry_capacity,
                  &hdr->global_ht_free_head);

    const struct { Mempool* pool; uint64_t offset; uint32_t total; } pools[] = {
        {&mempool_, hdr->mempool_offset, hdr->mempool_total_blocks},
        {&swa_mempool_, hdr->swa_mempool_offset, hdr->swa_mempool_total_blocks},
        {&mamba_mempool_, hdr->mamba_mempool_offset,
         hdr->mamba_mempool_total_blocks},
    };
    for (const auto& p : pools)
        if (p.total > 0)
            *p.pool = Mempool::attach_region((char*)base_ + p.offset, p.total);
    if (hdr->swa_lru_offset != 0)
        swa_lru_ = SlotLru(reinterpret_cast<SlotLruEntry*>(
                               (char*)base_ + hdr->swa_lru_offset),
                           hdr->swa_mempool_total_blocks, &hdr->swa_list);
    if (hdr->mamba_lru_offset != 0)
        mamba_lru_ = SlotLru(reinterpret_cast<SlotLruEntry*>(
                                 (char*)base_ + hdr->mamba_lru_offset),
                             hdr->mamba_mempool_total_blocks, &hdr->mamba_list);

    // KV-event ring. Binding it on attach too is harmless: only the process
    // running the evictor ever publishes, and readers bind separately.
    if (hdr->kvevent_ring_offset != 0)
        kvevent_writer_ = KvEventRingWriter((char*)base_ + hdr->kvevent_ring_offset);
}

uint64_t RadixShmemTree::kvevent_ring_offset() const {
    if (base_ == nullptr) return 0;
    return header()->kvevent_ring_offset;
}

void RadixShmemTree::alloc_root() {
    ShmHeader* hdr = header();
    uint32_t root_id = nodes_.alloc();
    hdr->root_node_id = root_id;
    TreeNode* root = nodes_.get(root_id);
    root->num_blocks = 0;
    root->data_offset = INVALID_OFFSET;
    root->parent_id = INVALID_NODE_ID;
}

void RadixShmemTree::reset() {
    set_write_op("reset");
    write_lock();

    nodes_.reset();   // also resets every ComponentStateTable entry
    data_.reset();
    child_table_.reset();
    for (Mempool* p : {&mempool_, &swa_mempool_, &mamba_mempool_})
        if (p->enabled())
            p->reset();

    for (SlotLru* l : {&swa_lru_, &mamba_lru_})
        if (l->enabled())
            l->reset();

    // Clear per-run header state; layout / magic / locks stay put.
    ShmHeader* hdr = header();
    hdr->leaf_list_head = INVALID_NODE_ID;
    hdr->leaf_list_tail = INVALID_NODE_ID;
    hdr->total_blocks.store(0, std::memory_order_relaxed);
    hdr->access_counter.store(0, std::memory_order_relaxed);
    hdr->insert_success.store(0, std::memory_order_relaxed);
    hdr->insert_error_node_pool.store(0, std::memory_order_relaxed);
    hdr->insert_error_data_pool.store(0, std::memory_order_relaxed);
    hdr->insert_error_ht_pool.store(0, std::memory_order_relaxed);
    hdr->insert_error_mempool.store(0, std::memory_order_relaxed);
    hdr->peer_faults.clear();

    alloc_root();

    write_unlock();
}

ShmLayoutInfo RadixShmemTree::compute_layout(const ShmConfig& config) {
    ShmLayoutInfo info{};
    info.header_size = align_up(sizeof(ShmHeader), 64);
    info.node_pool_size = (uint64_t)config.max_nodes * sizeof(TreeNode);
    info.component_state_size =
        component_has_aux(config.component_mask)
            ? (uint64_t)config.max_nodes * sizeof(NodeComponentState)
            : 0;

    constexpr uint64_t max_order_block =
        BuddyHeader::order_size(BuddyHeader::MAX_ORDER);
    uint64_t blocks_size = round_up_multiple(config.effective_data_pool_bytes(), max_order_block);
    uint32_t num_min_blocks = (uint32_t)(blocks_size / BuddyHeader::MIN_BLOCK_SIZE);
    uint64_t buddy_hdr_size = align_up(sizeof(BuddyHeader), 16);
    uint64_t bitmap_bytes   = align_up((uint64_t)(num_min_blocks + 7) / 8, 8);
    uint64_t omap_bytes     = align_up((uint64_t)num_min_blocks, 8);
    uint64_t data_metadata  = align_up(buddy_hdr_size + bitmap_bytes + omap_bytes, 16);
    info.data_pool_size = data_metadata + blocks_size;

    info.ht_bucket_count = next_power_of_2(
        config.max_nodes / 2 > 0 ? config.max_nodes / 2 : 1);
    info.ht_entry_capacity = config.max_nodes;
    info.ht_size = (uint64_t)info.ht_bucket_count * sizeof(uint32_t)
                 + (uint64_t)info.ht_entry_capacity * sizeof(ChainHTEntry);

    info.mempool_size = Mempool::region_bytes(config.max_blocks);
    info.swa_mempool_size = Mempool::region_bytes(config.swa_max_blocks);
    info.mamba_mempool_size = Mempool::region_bytes(config.mamba_max_slots);
    info.swa_lru_size = SlotLru::region_bytes(config.swa_max_blocks);
    info.mamba_lru_size = SlotLru::region_bytes(config.mamba_max_slots);

    info.kvevent_ring_capacity =
        kvevent_ring_normalize_capacity(config.kvevent_ring_capacity);
    info.kvevent_ring_size = kvevent_ring_bytes(info.kvevent_ring_capacity);

    info.total_size = info.header_size + info.node_pool_size
                    + info.component_state_size
                    + info.data_pool_size + info.ht_size
                    + info.mempool_size
                    + info.swa_mempool_size + info.mamba_mempool_size
                    + info.swa_lru_size + info.mamba_lru_size
                    + info.kvevent_ring_size;
    return info;
}

RadixShmemTree RadixShmemTree::create(const std::string& name,
                                  const ShmConfig& config) {
    validate_component_config(config.component_mask,
                              config.swa_window_blocks, "create");
    const int cs = config.effective_register_chunk_size();
    if (cs > BuddyHeader::MAX_DATA_BLOCKS)
        throw std::runtime_error(
            "register_chunk_tokens (" + std::to_string(config.register_chunk_tokens) +
            " tokens = " + std::to_string(cs) + " blocks) exceeds buddy allocator max capacity (" +
            std::to_string(BuddyHeader::MAX_DATA_BLOCKS) + " blocks)");

    RadixShmemTree tree;
    tree.name_ = name;
    tree.is_owner_ = true;
    tree.background_evict_ratio_ = config.background_evict_ratio;
    tree.evict_policy_ = config.evict_policy;
    tree.cs_ = cs;

    ShmLayoutInfo layout = compute_layout(config);
    uint64_t header_size     = layout.header_size;
    uint64_t node_pool_size  = layout.node_pool_size;
    uint64_t data_pool_total = layout.data_pool_size;
    uint32_t ht_bucket_count   = layout.ht_bucket_count;
    uint32_t ht_entry_capacity = layout.ht_entry_capacity;
    uint64_t ht_size_bytes     = layout.ht_size;

    tree.size_ = layout.total_size;

    // ---------- Create shared memory ----------
    bool hugepage_ok = false;
    if (!config.hugepage_path.empty()) {
        tree.size_ = align_up(tree.size_, HUGEPAGE_SIZE);
        std::string base_name = (!name.empty() && name[0] == '/') ? name.substr(1) : name;
        std::string hp_path = config.hugepage_path + "/" + base_name;
        ::unlink(hp_path.c_str());
        int fd = ::open(hp_path.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd >= 0) {
            if (ftruncate(fd, tree.size_) == 0) {
                void* p = mmap(nullptr, tree.size_, PROT_READ | PROT_WRITE,
                               MAP_SHARED | MAP_POPULATE, fd, 0);
                if (p != MAP_FAILED) {
                    tree.fd_ = fd;
                    tree.base_ = p;
                    tree.name_ = hp_path;
                    tree.is_hugepage_ = true;
                    hugepage_ok = true;
                    fprintf(stderr, "[radixshmem] Using hugepages: %s (%.1f MB)\n",
                            hp_path.c_str(), tree.size_ / (1024.0 * 1024.0));
                } else {
                    close(fd);
                    ::unlink(hp_path.c_str());
                }
            } else {
                close(fd);
                ::unlink(hp_path.c_str());
            }
        }
        if (!hugepage_ok)
            fprintf(stderr, "[radixshmem] Hugepage alloc failed, falling back to shm_open\n");
    }

    if (!hugepage_ok) {
        shm_unlink(name.c_str());
        tree.fd_ = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
        if (tree.fd_ < 0)
            throw std::runtime_error("shm_open create failed: " + name);
        if (ftruncate(tree.fd_, tree.size_) < 0) {
            close(tree.fd_);
            shm_unlink(name.c_str());
            throw std::runtime_error("ftruncate failed");
        }
        tree.base_ = mmap(nullptr, tree.size_, PROT_READ | PROT_WRITE,
                           MAP_SHARED, tree.fd_, 0);
        if (tree.base_ == MAP_FAILED) {
            close(tree.fd_);
            shm_unlink(name.c_str());
            throw std::runtime_error("mmap failed");
        }
    }
    memset(tree.base_, 0, tree.size_);

    // ---------- ShmHeader ----------
    // magic is written last so attach() won't succeed until init is complete
    ShmHeader* hdr = tree.header();
    hdr->version = SHM_VERSION;
    hdr->node_pool_capacity = config.max_nodes;
    hdr->node_pool_offset   = header_size;

    // 0 = no table (Full-only); peers key their CST expectations off the mask.
    hdr->component_state_offset =
        layout.component_state_size ? header_size + node_pool_size : 0;
    hdr->component_state_entry_size = (uint32_t)sizeof(NodeComponentState);

    hdr->data_pool_offset     = header_size + node_pool_size
                              + layout.component_state_size;
    hdr->data_pool_total_size = data_pool_total;

    hdr->global_ht_offset        = hdr->data_pool_offset + data_pool_total;
    hdr->global_ht_bucket_count   = ht_bucket_count;
    hdr->global_ht_entry_capacity = ht_entry_capacity;

    // Mempool regions (full / swa / mamba slot-id spaces): wire header offsets
    // and initialize each present region (region_bytes(0) == 0, so absent = no-op).
    const struct { uint64_t* offset; uint32_t* total; uint32_t count;
                   uint64_t bytes; } pools[] = {
        {&hdr->mempool_offset, &hdr->mempool_total_blocks,
         config.max_blocks, layout.mempool_size},
        {&hdr->swa_mempool_offset, &hdr->swa_mempool_total_blocks,
         config.swa_max_blocks, layout.swa_mempool_size},
        {&hdr->mamba_mempool_offset, &hdr->mamba_mempool_total_blocks,
         config.mamba_max_slots, layout.mamba_mempool_size},
    };
    uint64_t pool_cursor = hdr->global_ht_offset + ht_size_bytes;
    for (const auto& p : pools) {
        *p.offset = (p.count > 0) ? pool_cursor : 0;
        *p.total  = p.count;
        pool_cursor += p.bytes;
        if (p.count > 0)
            Mempool::init_region((char*)tree.base_ + *p.offset, p.count);
    }
    const struct { uint64_t* offset; uint32_t count; uint64_t bytes; } lrus[] = {
        {&hdr->swa_lru_offset, config.swa_max_blocks, layout.swa_lru_size},
        {&hdr->mamba_lru_offset, config.mamba_max_slots, layout.mamba_lru_size},
    };
    for (const auto& l : lrus) {
        *l.offset = (l.count > 0) ? pool_cursor : 0;
        pool_cursor += l.bytes;
        if (l.count > 0)
            SlotLru::init_region((char*)tree.base_ + *l.offset, l.count);
    }

    // KV-event eviction ring (last region; 0 capacity = absent, offset 0).
    hdr->kvevent_ring_capacity = layout.kvevent_ring_capacity;
    hdr->kvevent_ring_offset = layout.kvevent_ring_size ? pool_cursor : 0;
    if (layout.kvevent_ring_size) {
        KvEventRingWriter::init((char*)tree.base_ + hdr->kvevent_ring_offset,
                                layout.kvevent_ring_capacity);
        pool_cursor += layout.kvevent_ring_size;
    }

    hdr->component_config.abi_version          = COMPONENT_ABI_VERSION;
    hdr->component_config.component_mask       = config.component_mask;
    hdr->component_config.logical_block_tokens = (uint32_t)config.block_size;
    hdr->component_config.swa_window_blocks    = config.swa_window_blocks;
    hdr->component_config.mamba_state_bytes    = config.mamba_state_bytes;
    hdr->swa_list = AuxListAnchors{INVALID_SLOT_LINK, INVALID_SLOT_LINK};
    hdr->mamba_list = AuxListAnchors{INVALID_SLOT_LINK, INVALID_SLOT_LINK};

    // Cluster coordinates: standalone defaults; stamped by RadixServer on bootstrap.
    hdr->is_distributed = 0;
    hdr->my_rank        = 0;
    hdr->registry[0]    = '\0';
    hdr->cluster_id[0]  = '\0';
    hdr->node_name[0]   = '\0';
    hdr->cluster_ready_magic.store(0, std::memory_order_release);

    hdr->background_evict_ratio = config.background_evict_ratio;
    hdr->evict_policy = static_cast<uint32_t>(config.evict_policy);
    hdr->block_size   = config.block_size;
    hdr->register_chunk_size = cs;

    hdr->total_blocks.store(0, std::memory_order_relaxed);
    hdr->leaf_list_head = INVALID_NODE_ID;
    hdr->leaf_list_tail = INVALID_NODE_ID;
    hdr->access_counter.store(0, std::memory_order_relaxed);
    hdr->node_pool_size.store(0, std::memory_order_relaxed);
    hdr->insert_success.store(0, std::memory_order_relaxed);
    hdr->insert_error_node_pool.store(0, std::memory_order_relaxed);
    hdr->insert_error_data_pool.store(0, std::memory_order_relaxed);
    hdr->insert_error_ht_pool.store(0, std::memory_order_relaxed);
    hdr->insert_error_mempool.store(0, std::memory_order_relaxed);
    hdr->peer_faults.clear();

    hdr->rdma_write_flag.store(0, std::memory_order_relaxed);
    hdr->rdma_reader_count.store(0, std::memory_order_relaxed);

    // Initialize process-shared rwlock
    pthread_rwlockattr_t attr;
    pthread_rwlockattr_init(&attr);
    pthread_rwlockattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
    pthread_rwlock_init(&hdr->rwlock, &attr);
    pthread_rwlockattr_destroy(&attr);

    // Initialize process-shared LRU mutex
    pthread_mutexattr_t mattr;
    pthread_mutexattr_init(&mattr);
    pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&hdr->lru_mutex, &mattr);
    pthread_mutexattr_destroy(&mattr);

    // ---------- NodePool ----------
    auto* node_base = reinterpret_cast<TreeNode*>(
        (char*)tree.base_ + hdr->node_pool_offset);
    NodePool::init_free_list(node_base, config.max_nodes, hdr->node_free_head);

    // ---------- ComponentStateTable (aux components only) ----------
    // memset(0) above is not enough: LRU links must start INVALID, not node 0.
    if (hdr->component_state_offset != 0) {
        auto* cst_base = reinterpret_cast<NodeComponentState*>(
            (char*)tree.base_ + hdr->component_state_offset);
        for (uint32_t i = 0; i < config.max_nodes; i++)
            NodePool::reset_component_state(cst_base[i]);
    }

    // ---------- DataPool (buddy allocator) ----------
    constexpr uint64_t max_order_block =
        BuddyHeader::order_size(BuddyHeader::MAX_ORDER);
    uint64_t blocks_size = round_up_multiple(config.effective_data_pool_bytes(), max_order_block);
    uint32_t num_min_blocks = (uint32_t)(blocks_size / BuddyHeader::MIN_BLOCK_SIZE);
    uint64_t buddy_hdr_size = align_up(sizeof(BuddyHeader), 16);
    uint64_t bitmap_bytes   = align_up((uint64_t)(num_min_blocks + 7) / 8, 8);
    uint64_t omap_bytes     = align_up((uint64_t)num_min_blocks, 8);

    auto* bh = reinterpret_cast<BuddyHeader*>(
        (char*)tree.base_ + hdr->data_pool_offset);
    DataPool::init(tree.base_, bh, hdr->data_pool_offset,
                   buddy_hdr_size, bitmap_bytes, omap_bytes, blocks_size);

    // Initialize HT: buckets to HT_END, entries into free list
    auto* ht_base = (char*)tree.base_ + hdr->global_ht_offset;
    auto* ht_buckets = reinterpret_cast<uint32_t*>(ht_base);
    for (uint32_t i = 0; i < ht_bucket_count; i++)
        ht_buckets[i] = HT_END;
    auto* ht_entries = reinterpret_cast<ChainHTEntry*>(
        ht_base + (uint64_t)ht_bucket_count * sizeof(uint32_t));
    ChildTable::init_free_list(ht_entries, ht_entry_capacity,
                           hdr->global_ht_free_head);

    // ---------- Init pool wrappers ----------
    // (Mempool regions were initialized alongside their header offsets above.)
    tree.init_pools();

    // ---------- Root node ----------
    tree.alloc_root();

    // Write magic last so concurrent attach() won't succeed until init is done
    std::atomic_thread_fence(std::memory_order_release);
    hdr->magic = SHM_MAGIC;

    return tree;
}

RadixShmemTree RadixShmemTree::attach(const std::string& name,
                                  const std::string& hugepage_path) {
    RadixShmemTree tree;
    tree.is_owner_ = false;

    // Try hugepage path first
    if (!hugepage_path.empty()) {
        std::string base_name = (!name.empty() && name[0] == '/') ? name.substr(1) : name;
        std::string hp_path = hugepage_path + "/" + base_name;
        int fd = ::open(hp_path.c_str(), O_RDWR);
        if (fd >= 0) {
            struct stat st;
            if (fstat(fd, &st) == 0 && st.st_size > 0) {
                tree.size_ = st.st_size;
                void* p = mmap(nullptr, tree.size_, PROT_READ | PROT_WRITE,
                               MAP_SHARED | MAP_POPULATE, fd, 0);
                if (p != MAP_FAILED) {
                    tree.fd_ = fd;
                    tree.base_ = p;
                    tree.name_ = hp_path;
                    tree.is_hugepage_ = true;
                } else {
                    close(fd);
                }
            } else {
                close(fd);
            }
        }
    }

    // Fallback to POSIX shm
    if (!tree.base_) {
        tree.name_ = name;
        tree.fd_ = shm_open(name.c_str(), O_RDWR, 0666);
        if (tree.fd_ < 0)
            throw std::runtime_error("shm_open attach failed: " + name);

        struct stat st;
        if (fstat(tree.fd_, &st) < 0) {
            close(tree.fd_);
            throw std::runtime_error("fstat failed");
        }
        tree.size_ = st.st_size;

        tree.base_ = mmap(nullptr, tree.size_, PROT_READ | PROT_WRITE,
                           MAP_SHARED, tree.fd_, 0);
        if (tree.base_ == MAP_FAILED) {
            close(tree.fd_);
            throw std::runtime_error("mmap attach failed");
        }
    }

    ShmHeader* hdr = tree.header();
    if (hdr->magic != SHM_MAGIC)
        throw std::runtime_error("Invalid magic number");
    if (hdr->version != SHM_VERSION)
        throw std::runtime_error("Version mismatch");
    // Component ABI cross-checks (fail closed on a corrupt/foreign header).
    if (hdr->component_config.abi_version != COMPONENT_ABI_VERSION)
        throw std::runtime_error(
            "Component ABI version mismatch: region has " +
            std::to_string(hdr->component_config.abi_version) + ", expected " +
            std::to_string(COMPONENT_ABI_VERSION));
    if (hdr->component_state_entry_size != sizeof(NodeComponentState))
        throw std::runtime_error(
            "ComponentStateTable entry size mismatch: region has " +
            std::to_string(hdr->component_state_entry_size) + ", expected " +
            std::to_string(sizeof(NodeComponentState)));
    if (hdr->component_config.logical_block_tokens != (uint32_t)hdr->block_size)
        throw std::runtime_error("ComponentConfig.logical_block_tokens "
                                 "disagrees with ShmHeader.block_size");
    validate_component_config(hdr->component_config.component_mask,
                              hdr->component_config.swa_window_blocks,
                              "attach");

    tree.background_evict_ratio_ = hdr->background_evict_ratio;
    tree.evict_policy_ = static_cast<EvictPolicy>(hdr->evict_policy);
    tree.cs_           = hdr->register_chunk_size;

    tree.init_pools();
    return tree;
}

void RadixShmemTree::detach() {
    if (base_ && base_ != MAP_FAILED) {
        munmap(base_, size_);
        base_ = nullptr;
    }
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}

void RadixShmemTree::destroy() {
    if (base_ && base_ != MAP_FAILED && is_owner_) {
        pthread_rwlock_destroy(&header()->rwlock);
        pthread_mutex_destroy(&header()->lru_mutex);
    }
    detach();
    if (!name_.empty()) {
        if (is_hugepage_)
            ::unlink(name_.c_str());
        else
            shm_unlink(name_.c_str());
        name_.clear();
    }
}

// ==================== Accessors ====================

ShmHeader* RadixShmemTree::header() const {
    return reinterpret_cast<ShmHeader*>(base_);
}

// ==================== Concurrency helpers ====================

pthread_rwlock_t* RadixShmemTree::rwlock() const {
    return &header()->rwlock;
}

void RadixShmemTree::read_lock() { pthread_rwlock_rdlock(rwlock()); }
void RadixShmemTree::read_unlock() { pthread_rwlock_unlock(rwlock()); }

static constexpr int64_t RDMA_WRITE_BIT = 1LL << 63;

// Flush cache line to DRAM so NIC PCIe atomics and CPU see the same value.
static inline void rdma_rwlock_flush(const volatile void* addr) {
#ifdef __x86_64__
    _mm_clflush(const_cast<const void*>(addr));
    _mm_mfence();
#else
    __sync_synchronize();
#endif
}

bool RadixShmemTree::try_write_lock() {
    if (pthread_rwlock_trywrlock(rwlock()) != 0)
        return false;
    const bool tr = lock_trace_threshold_us() >= 0;
    if (tr) { g_lt_req = lt_now_us(); g_lt_wr = g_lt_req; }
    rdma_writer_enter(tr);
    return true;
}

void RadixShmemTree::set_write_op(const char* op) { g_write_op = op; }

void RadixShmemTree::rdma_writer_enter(bool tr) {
    auto* flag = &header()->rdma_write_flag;
    auto* cnt  = &header()->rdma_reader_count;
    lt_check_stuck(flag->exchange(RDMA_WRITE_BIT, std::memory_order_acq_rel), "write_lock");
    rdma_rwlock_flush(flag);
    rdma_rwlock_flush(cnt);
    if (tr) g_lt_readers = (long long)cnt->load(std::memory_order_relaxed);
    uint64_t stall_spins = 0;
    while (cnt->load(std::memory_order_acquire) != 0) {
        rdma_rwlock_flush(cnt);
#ifdef __x86_64__
        _mm_pause();
#endif
        if (++stall_spins == (1ull << 24)) {
            fprintf(stderr,
                    "[RadixShmemTree] write_lock stalled for seconds: "
                    "rdma_reader_count=%lld not draining — leaked/stuck "
                    "remote readers?\n",
                    (long long)cnt->load(std::memory_order_relaxed));
        }
    }
    if (tr) g_lt_drain = lt_now_us();
}

void RadixShmemTree::write_lock() {
    const bool tr = lock_trace_threshold_us() >= 0;
    if (tr) g_lt_req = lt_now_us();
    pthread_rwlock_wrlock(rwlock());
    if (tr) g_lt_wr = lt_now_us();
    rdma_writer_enter(tr);
}

void RadixShmemTree::write_unlock() {
    auto* flag = &header()->rdma_write_flag;
    flag->store(0, std::memory_order_release);
    rdma_rwlock_flush(flag);
    pthread_rwlock_unlock(rwlock());
    const long thr = lock_trace_threshold_us();
    if (thr >= 0 && g_lt_req) {
        uint64_t end = lt_now_us();
        if ((long)(end - g_lt_req) > thr) {
            struct timeval tv; gettimeofday(&tv, nullptr);
            fprintf(stderr, "[locktrace %ld.%03ld pid=%d] op=%s wr_wait=%luus drain=%luus hold=%luus readers=%lld\n",
                    (long)tv.tv_sec, (long)tv.tv_usec / 1000, (int)getpid(), g_write_op,
                    (unsigned long)(g_lt_wr - g_lt_req), (unsigned long)(g_lt_drain - g_lt_wr),
                    (unsigned long)(end - g_lt_drain), g_lt_readers);
        }
        g_lt_req = 0;
    }
}
void RadixShmemTree::lru_lock() { pthread_mutex_lock(&header()->lru_mutex); }
void RadixShmemTree::lru_unlock() { pthread_mutex_unlock(&header()->lru_mutex); }


void RadixShmemTree::touch_node_atomic(uint32_t node_id, uint64_t ts) {
    nodes_.get(node_id)->last_access_time.store(ts, std::memory_order_relaxed);
}

void RadixShmemTree::touch_node(uint32_t node_id) {
    uint64_t t = now_ts();
    nodes_.get(node_id)->last_access_time.store(t, std::memory_order_relaxed);
    if (is_in_leaf_list(node_id))
        leaf_list_move_to_tail(node_id);
}

uint64_t RadixShmemTree::now_ts() {
    return (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
}

// ==================== Child management ====================

bool RadixShmemTree::add_child(uint32_t parent_id, uint64_t token_hash,
                             uint32_t child_id) {
    TreeNode* parent = nodes_.get(parent_id);
    if (!child_table_.insert(parent_id, token_hash, child_id, parent))
        return false;
    leaf_list_remove(parent_id);   // gaining a child → no longer a leaf
    parent->children_count++;
    return true;
}

void RadixShmemTree::remove_child(uint32_t parent_id, uint64_t token_hash,
                                uint32_t child_id) {
    TreeNode* parent = nodes_.get(parent_id);
    child_table_.remove(parent_id, token_hash, parent);
    parent->children_count--;
}

// ==================== Split & Merge ====================


// Copy `count` blocks of every lane this shm carries (hash / full slot /
// swa slot) from src payload position src_pos to dst payload position dst_pos.
static void copy_lanes(DataPool& d, uint64_t dst, int dst_n, int dst_pos,
                       uint64_t src, int src_n, int src_pos, int count) {
    memcpy(d.get_hashes(dst) + dst_pos, d.get_hashes(src) + src_pos,
           count * sizeof(uint64_t));
    memcpy(d.get_slots(dst, dst_n) + dst_pos,
           d.get_slots(src, src_n) + src_pos, count * sizeof(int32_t));
    if (d.has_swa_lane())
        memcpy(d.get_swa_slots(dst, dst_n) + dst_pos,
               d.get_swa_slots(src, src_n) + src_pos, count * sizeof(int32_t));
}

uint32_t RadixShmemTree::split_node(uint32_t node_id, int prefix_length) {

    TreeNode* node = nodes_.get(node_id);
    int n = (int)node->num_blocks;
    assert(prefix_length > 0 && prefix_length < n);

    uint64_t orig_data = node->data_offset;
    uint32_t parent_id = node->parent_id;
    uint64_t orig_head_hash = data_.get_hashes(orig_data)[0];

    int suffix_length = n - prefix_length;

    // Acquire all resources up front; roll back and return INVALID on any
    // alloc failure, leaving the original node untouched.
    uint64_t prefix_data = data_.alloc(prefix_length);
    if (prefix_data == INVALID_OFFSET)
        return INVALID_NODE_ID;
    uint64_t suffix_data = data_.alloc(suffix_length);
    if (suffix_data == INVALID_OFFSET) {
        data_.free(prefix_data, prefix_length);
        return INVALID_NODE_ID;
    }
    uint32_t prefix_id = nodes_.alloc();
    if (prefix_id == INVALID_NODE_ID) {
        data_.free(prefix_data, prefix_length);
        data_.free(suffix_data, suffix_length);
        return INVALID_NODE_ID;
    }

    // All allocations succeeded — now it is safe to mutate state.
    copy_lanes(data_, prefix_data, prefix_length, 0,
               orig_data, n, 0, prefix_length);
    copy_lanes(data_, suffix_data, suffix_length, 0,
               orig_data, n, prefix_length, suffix_length);

    data_.free(orig_data, n);

    uint64_t suffix_head_hash = data_.get_hashes(suffix_data)[0];

    TreeNode* prefix_node = nodes_.get(prefix_id);
    node = nodes_.get(node_id);

    int64_t orig_rc = node->ref_count.load(std::memory_order_relaxed);
    uint64_t orig_at = node->last_access_time.load(std::memory_order_relaxed);

    prefix_node->num_blocks = prefix_length;
    prefix_node->data_offset = prefix_data;
    prefix_node->parent_id = parent_id;
    prefix_node->ref_count.fetch_add(orig_rc, std::memory_order_relaxed);
    // Whole-node component pins follow the split: Full/SWA refs copy to the
    // prefix, Mamba stays with the suffix END; live counts follow the lanes.
    if (has_aux_) {
        NodeComponentState* pre_cs = nodes_.component_state(prefix_id);
        NodeComponentState* suf_cs = nodes_.component_state(node_id);
        pre_cs->swa_ref_count.store(
            suf_cs->swa_ref_count.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
    }
    // Inherit access_time from split node to preserve strict LRU ordering.
    prefix_node->last_access_time.store(orig_at, std::memory_order_relaxed);
    // The suffix keeps the original END checkpoint; the prefix starts INVALID.

    node->data_offset = suffix_data;
    node->num_blocks = suffix_length;
    node->parent_id = prefix_id;

    if (has_swa_) {
        relink_swa(prefix_id);
        relink_swa(node_id);
    }

    if (parent_id != INVALID_NODE_ID)
        child_table_.update_child(parent_id, orig_head_hash, prefix_id);

    add_child(prefix_id, suffix_head_hash, node_id);


    return prefix_id;
}

// Pin-eligibility for restructuring/eviction: all three component ref
// classes (full/swa/mamba) on the node must be zero.
bool RadixShmemTree::component_refs_zero(uint32_t node_id) const {
    if (nodes_.get(node_id)->ref_count.load(std::memory_order_relaxed) != 0)
        return false;
    if (!has_aux_) return true;
    NodeComponentState* cs = nodes_.component_state(node_id);
    return cs->swa_ref_count.load(std::memory_order_relaxed) == 0 &&
           cs->mamba_ref_count.load(std::memory_order_relaxed) == 0;
}

// Fold an only-child ready leaf up into its ready parent (inverse of split_node,
// same alloc-first discipline). Caller holds write_lock and checked the trigger.
MergeOutcome RadixShmemTree::merge_leaf_into_parent(uint32_t leaf_id,
                                                  int abs_leaf_start) {
    const int cs = cs_;
    MergeOutcome out;
    static constexpr int MAX_BLOCKS_PER_NODE = BuddyHeader::MAX_DATA_BLOCKS;

    TreeNode* leaf = nodes_.get(leaf_id);
    uint32_t parent_id = leaf->parent_id;
    TreeNode* parent = nodes_.get(parent_id);

    // Pinned fragments must not be restructured: any component ref on either
    // node defers the merge (retryable after release).
    if (!component_refs_zero(leaf_id) || !component_refs_zero(parent_id))
        return out;
    // A checkpoint lives on its node END; folding the leaf in would make that
    // END internal and destroy it. Sequential extension must keep it, so the
    // merge waits until Mamba eviction retires the checkpoint.
    if (parent->mamba_checkpoint_slot != INVALID_COMPONENT_SLOT)
        return out;
    // A leaf starting on a cs multiple is an RHT probe anchor: folding it in
    // would drop this peer's route at that position.
    if (cs > 0 && abs_leaf_start % cs == 0)
        return out;

    int pn = (int)parent->num_blocks;
    int ln = (int)leaf->num_blocks;
    int p_start = abs_leaf_start - pn;
    uint64_t old_leaf_hash = data_.get_hashes(leaf->data_offset)[0];

    // Absorb the whole leaf if it fits under the 4096 cap, else stop on the
    // largest cs multiple <= cap so the remainder starts on a probe boundary.
    int cap_end  = p_start + MAX_BLOCKS_PER_NODE;
    int full_end = abs_leaf_start + ln;
    int absorb;
    if (full_end <= cap_end) {
        absorb = ln;
    } else {
        int rem_start = (cs > 0) ? (cap_end / cs) * cs : cap_end;
        absorb = rem_start - abs_leaf_start;
        if (absorb <= 0)
            return out;
    }

    if (absorb == ln) {
        // FULL: leaf's blocks move into parent; leaf is freed.
        int total = pn + ln;
        uint64_t new_data = data_.alloc(total);
        if (new_data == INVALID_OFFSET)
            return out;

        parent = nodes_.get(parent_id);
        leaf   = nodes_.get(leaf_id);
        uint64_t p_old = parent->data_offset;
        uint64_t l_old = leaf->data_offset;

        copy_lanes(data_, new_data, total, 0, p_old, pn, 0, pn);
        copy_lanes(data_, new_data, total, pn, l_old, ln, 0, ln);

        data_.free(p_old, pn);
        data_.free(l_old, ln);

        parent->num_blocks  = total;
        parent->data_offset = new_data;
        if (has_swa_)
            relink_swa(parent_id);
        // The leaf END survives as the merged END (the parent END carries no
        // checkpoint: gated above).
        aux_on_merge(parent_id, leaf_id);
        // Do NOT sum leaf ref onto parent (merge collapses two nodes to one).

        remove_child(parent_id, old_leaf_hash, leaf_id);
        leaf_list_remove(leaf_id);
        nodes_.free(leaf_id);
        parent->last_access_time.store(now_ts(), std::memory_order_relaxed);
        leaf_list_add(parent_id);

        out.merged = true;
        out.full = true;
        out.parent_id = parent_id;
        out.old_leaf_hash = old_leaf_hash;
        return out;
    }

    // PARTIAL: rebalance the parent/leaf boundary onto a cs multiple.
    int rem = ln - absorb;
    uint64_t p_data = data_.alloc(pn + absorb);
    if (p_data == INVALID_OFFSET)
        return out;
    uint64_t l_data = data_.alloc(rem);
    if (l_data == INVALID_OFFSET) {
        data_.free(p_data, pn + absorb);
        return out;
    }

    parent = nodes_.get(parent_id);
    leaf   = nodes_.get(leaf_id);
    uint64_t p_old = parent->data_offset;
    uint64_t l_old = leaf->data_offset;

    // parent := parent.blocks + leaf.blocks[0..absorb)
    copy_lanes(data_, p_data, pn + absorb, 0, p_old, pn, 0, pn);
    copy_lanes(data_, p_data, pn + absorb, pn, l_old, ln, 0, absorb);

    // leaf := leaf.blocks[absorb..ln)
    copy_lanes(data_, l_data, rem, 0, l_old, ln, absorb, rem);

    data_.free(p_old, pn);
    data_.free(l_old, ln);

    parent->num_blocks  = pn + absorb;
    parent->data_offset = p_data;
    // The parent END moves deeper; it carried no checkpoint (gated above).
    // The leaf keeps its END checkpoint (its absolute END is unchanged).
    assert(parent->mamba_checkpoint_slot == INVALID_COMPONENT_SLOT);
    leaf->num_blocks  = rem;
    leaf->data_offset = l_data;
    if (has_swa_) {
        relink_swa(parent_id);
        relink_swa(leaf_id);
    }
    uint64_t new_leaf_hash = data_.get_hashes(l_data)[0];

    // Re-key the leaf in the parent's child HT (its first_hash changed).
    remove_child(parent_id, old_leaf_hash, leaf_id);
    add_child(parent_id, new_leaf_hash, leaf_id);

    out.merged = true;
    out.full = false;
    out.parent_id = parent_id;
    out.old_leaf_hash = old_leaf_hash;
    out.new_leaf_hash = new_leaf_hash;
    return out;
}

// ==================== Leaf list ====================

void RadixShmemTree::leaf_list_add(uint32_t node_id) {
    ShmHeader* hdr = header();
    if (is_in_leaf_list(node_id))   // duplicate add would corrupt prev/next links
        return;
    TreeNode* node = nodes_.get(node_id);
    node->leaf_next = INVALID_NODE_ID;
    node->leaf_prev = hdr->leaf_list_tail;
    if (hdr->leaf_list_tail != INVALID_NODE_ID)
        nodes_.get(hdr->leaf_list_tail)->leaf_next = node_id;
    else
        hdr->leaf_list_head = node_id;
    hdr->leaf_list_tail = node_id;
}

void RadixShmemTree::leaf_list_remove(uint32_t node_id) {
    if (!is_in_leaf_list(node_id)) return;
    ShmHeader* hdr = header();
    TreeNode* node = nodes_.get(node_id);
    if (node->leaf_prev != INVALID_NODE_ID)
        nodes_.get(node->leaf_prev)->leaf_next = node->leaf_next;
    else
        hdr->leaf_list_head = node->leaf_next;
    if (node->leaf_next != INVALID_NODE_ID)
        nodes_.get(node->leaf_next)->leaf_prev = node->leaf_prev;
    else
        hdr->leaf_list_tail = node->leaf_prev;
    node->leaf_prev = INVALID_NODE_ID;
    node->leaf_next = INVALID_NODE_ID;
}

void RadixShmemTree::leaf_list_move_to_tail(uint32_t node_id) {
    if (header()->leaf_list_tail == node_id) return;
    leaf_list_remove(node_id);
    leaf_list_add(node_id);
}

void RadixShmemTree::leaf_list_insert_by_access_time(uint32_t node_id) {
    ShmHeader* hdr = header();
    TreeNode* node = nodes_.get(node_id);
    uint64_t t = node->last_access_time.load(std::memory_order_relaxed);

    uint32_t cur = hdr->leaf_list_tail;
    while (cur != INVALID_NODE_ID) {
        TreeNode* cn = nodes_.get(cur);
        if (cn->last_access_time.load(std::memory_order_relaxed) <= t) {
            node->leaf_prev = cur;
            node->leaf_next = cn->leaf_next;
            if (cn->leaf_next != INVALID_NODE_ID)
                nodes_.get(cn->leaf_next)->leaf_prev = node_id;
            else
                hdr->leaf_list_tail = node_id;
            cn->leaf_next = node_id;
            return;
        }
        cur = cn->leaf_prev;
    }

    node->leaf_prev = INVALID_NODE_ID;
    node->leaf_next = hdr->leaf_list_head;
    if (hdr->leaf_list_head != INVALID_NODE_ID)
        nodes_.get(hdr->leaf_list_head)->leaf_prev = node_id;
    else
        hdr->leaf_list_tail = node_id;
    hdr->leaf_list_head = node_id;
}

bool RadixShmemTree::is_in_leaf_list(uint32_t node_id) const {
    return header()->leaf_list_head == node_id ||
           nodes_.get(node_id)->leaf_prev != INVALID_NODE_ID;
}

// ==================== Eviction internals ====================

int RadixShmemTree::evict_node(uint32_t node_id, std::vector<int32_t>& out) {
    TreeNode* node = nodes_.get(node_id);
    int n = (int)node->num_blocks;
    uint32_t parent_id = node->parent_id;
    uint64_t head_hash = data_.get_hashes(node->data_offset)[0];

    // Record (first_hash, node_id) for distributed RHT cleanup, plus every
    // block hash for KV-event publication (flushed after unlock). The full
    // hash list is only materialised when someone is listening: the hashes
    // live in the data pool and are gone once the node is freed below.
    if (has_evict_consumers()) {
        EvictedNode rec;
        rec.first_hash = head_hash;
        rec.tree_node_id = node_id;
        if (wants_evicted_hashes()) {
            const uint64_t* hashes = data_.get_hashes(node->data_offset);
            rec.block_hashes.assign(hashes, hashes + n);
        }
        evicted_pending().push_back(std::move(rec));
    }

    int32_t* slots = data_.get_slots(node->data_offset, n);
    for (int i = 0; i < n; i++)
        out.push_back(slots[i]);

    std::vector<int32_t> retired;
    retire_swa(node_id, retired);
    const size_t swa_retired = retired.size();
    retire_mamba(node_id, retired);
    assert(node->mamba_checkpoint_slot == INVALID_COMPONENT_SLOT);
    assert(!has_aux_ || nodes_.component_state(node_id)->swa_live_blocks == 0);

    leaf_list_remove(node_id);

    size_t recycle_start = out.size() - n;

    data_.free(node->data_offset, n);
    node = nodes_.get(node_id);
    remove_child(parent_id, head_hash, node_id);
    if (swa_retired > 0)
        swa_mempool_.recycle(retired.data(), (int)swa_retired);
    if (retired.size() > swa_retired)
        mamba_mempool_.recycle(retired.data() + swa_retired,
                               (int)(retired.size() - swa_retired));
    nodes_.free(node_id);
    header()->total_blocks.fetch_sub(n, std::memory_order_relaxed);

    if (mempool_.enabled())
        mempool_.recycle(out.data() + recycle_start, n);

    post_evict_fixup(parent_id);
    return n;
}

void RadixShmemTree::post_evict_fixup(uint32_t check_id) {
    if (check_id != header()->root_node_id &&
        nodes_.get(check_id)->children_count == 0)
        leaf_list_insert_by_access_time(check_id);
}

// ==================== Core operations ====================

TreeComponentQueryResult RadixShmemTree::query_components(
        const uint64_t* hashes, int n, uint32_t mask, bool lock,
        bool update_meta) {
    const ComponentConfig& config = header()->component_config;

    // mask must be a non-empty subset of the enabled components; invalid
    // calls fail closed with BAD_REQUEST and no LRU touch. n == 0 is OK-miss.
    if (mask == 0 || (mask & ~COMPONENT_MASK_ALL) != 0 ||
        !is_component_config_valid(config.component_mask,
                                   config.swa_window_blocks) ||
        (mask & config.component_mask) != mask ||
        n < 0 || (n > 0 && hashes == nullptr)) {
        TreeComponentQueryResult result;
        result.status = RemoteStatus::BAD_REQUEST;
        return result;
    }
    if (n == 0)
        return TreeComponentQueryResult{};

    const bool want_full = (mask & COMPONENT_MASK_FULL) != 0;
    read_lock();
    const uint64_t ts = now_ts();
    QueryInternalDetail m = query_internal(hashes, n, ts,
                                           update_meta && want_full);
    std::vector<MatchedNodeSpanView> spans = path_spans(m);
    ComponentQueryContext ctx;
    ctx.config = config;
    ctx.mask = mask;
    ctx.path = MatchedPathView{spans.data(), (uint32_t)spans.size(),
                               (uint32_t)m.matched_blocks};
    auto detail = ComponentSet().query(ctx);
    // Full recency and Full pin both span the topology hit [0, f), not
    // common_hit: a remote fold may hand the caller local Full blocks in
    // [common_hit, f).
    const uint32_t full_hit = want_full ? detail.topology_match_length : 0;
    if (update_meta) {
        touch_full(spans, mask, full_hit, m.last_node_id);
        aux_touch(spans, mask, detail.common_hit, detail.swa_start, ts);
    }
    if (lock) {
        if (detail.common_hit > 0 || full_hit > 0)
            inc_ref(spans, mask, detail.common_hit, detail.swa_start, full_hit,
                    detail.pin);
    }
    read_unlock();
    return detail;
}

// Runs under the read lock; eviction needs the write lock, so no gap opens.
// Full covers [0, full_hit); SWA/Mamba cover the aux boundary common_hit.
void RadixShmemTree::inc_ref(const std::vector<MatchedNodeSpanView>& spans,
                           uint32_t mask, uint32_t common_hit,
                           uint32_t swa_start, uint32_t full_hit,
                           PinRecord& pin) {
    const uint32_t L = common_hit;
    // Index math plus, for a partial terminal, a copy of its tail hashes.
    PinCoverage cov = compute_pin_coverage(spans.data(), spans.size(), mask, L,
                                           swa_start, 0, full_hit);
    record_pin_tail(spans.data(), spans.size(), cov, full_hit, L, pin);
    if (cov.swa_from < spans.size())
        pin.swa = {spans[cov.swa_from].absolute_start, cov.aux_cover_end};
    // With MAMBA in the mask, acceptance at L implies L is spans[aux_k]'s
    // node END and its checkpoint is live.
    if (cov.pin_mamba)
        pin.mamba_end = (int32_t)L;
    if (cov.has_full)
        for (size_t i = cov.full_from; i <= cov.full_k; i++)
            nodes_.get(spans[i].node_id)
                ->ref_count.fetch_add(1, std::memory_order_relaxed);
    for (size_t i = cov.swa_from; i <= cov.aux_k && i < spans.size(); i++)
        nodes_.component_state(spans[i].node_id)
            ->swa_ref_count.fetch_add(1, std::memory_order_relaxed);
    if (cov.pin_mamba)
        nodes_.component_state(spans[cov.aux_k].node_id)
            ->mamba_ref_count.fetch_add(1, std::memory_order_relaxed);
}

void RadixShmemTree::touch_full(const std::vector<MatchedNodeSpanView>& spans,
                             uint32_t mask, uint32_t full_hit,
                             uint32_t last_node_id) {
    if ((mask & COMPONENT_MASK_FULL) == 0 || full_hit == 0)
        return;
    // Full-only keeps legacy behavior: the walk terminal moves even on an
    // ml==0 split; otherwise the deepest matched node moves.
    uint32_t move_id = mask == COMPONENT_MASK_FULL
                           ? last_node_id
                           : (spans.empty() ? INVALID_NODE_ID
                                            : spans.back().node_id);
    lru_lock();
    if (move_id != INVALID_NODE_ID && is_in_leaf_list(move_id))
        leaf_list_move_to_tail(move_id);
    lru_unlock();
}

// Match-and-pin against the current tree; unlike void inc_ref, partial pins
// are visible in the record. Fused callers use query_components(lock=true).
PinRecord RadixShmemTree::inc_ref(const uint64_t* hashes, int n, uint32_t mask) {
    return std::move(query_components(hashes, n, mask,
                                      /*lock=*/true,
                                      /*update_meta=*/false).pin);
}

void RadixShmemTree::flush_lru(const uint32_t* node_ids, int count) {
    lru_lock();
    for (int i = 0; i < count; i++) {
        uint32_t id = node_ids[i];
        if (id != INVALID_NODE_ID && is_in_leaf_list(id))
            leaf_list_move_to_tail(id);
    }
    lru_unlock();
}

bool RadixShmemTree::dec_ref(const uint64_t* hashes, int n,
                            PinRecord&& pin) {
    const bool full_present = pin.full.start < pin.full.end;
    // Lanes this shm does not carry cannot hold a pin: ignore those ranges.
    const bool swa_present = has_swa_ && pin.swa.start < pin.swa.end;
    const bool mamba_present = has_mamba_ && pin.mamba_end > 0;
    if (!full_present && !swa_present && !mamba_present)
        return true;   // empty record: release is a no-op

    // Legacy trust model: re-walk by the caller's hashes and
    // decrement unconditionally inside the recorded lane ranges. The walk
    // must reach the deepest lane END; the record's tail supplies the
    // terminal's blocks past the caller's keys, whichever lane pinned it.
    uint32_t end = 0;
    if (full_present) end = pin.full.end;
    if (swa_present && pin.swa.end > end) end = pin.swa.end;
    if (mamba_present && (uint32_t)pin.mamba_end > end)
        end = (uint32_t)pin.mamba_end;
    const uint32_t tail = (uint32_t)pin.tail.size();
    if (tail > end) return false;
    const uint32_t need = end - tail;   // caller keys the prefix [0, need)
    if (hashes == nullptr || n <= 0 || (uint32_t)n < need)
        return false;
    std::vector<uint64_t> keys;
    const uint64_t* k = hashes;
    if (tail > 0) {
        keys.assign(hashes, hashes + need);
        keys.insert(keys.end(), pin.tail.begin(), pin.tail.end());
        k = keys.data();                // exactly `end` keys
    }

    read_lock();
    QueryInternalDetail m = query_internal(k, (int)end, 0, false);
    for (const MatchedNodeSpanView& v : path_spans(m)) {
        const uint32_t s = v.absolute_start;
        const uint32_t e = s + v.node_blocks;
        if (e > end) break;   // only nodes fully inside [0, end)
        if (full_present && s >= pin.full.start && s < pin.full.end)
            (void)dec_ref(nodes_.get(v.node_id)->ref_count);
        if (swa_present && s >= pin.swa.start && s < pin.swa.end)
            (void)dec_ref(nodes_.component_state(v.node_id)->swa_ref_count);
        if (mamba_present && e == (uint32_t)pin.mamba_end)
            (void)dec_ref(nodes_.component_state(v.node_id)->mamba_ref_count);
    }
    read_unlock();
    return true;
}

QueryInternalDetail RadixShmemTree::query_internal(const uint64_t* hashes, int n,
                                                 uint64_t ts, bool touch) {
    QueryInternalDetail result;

    ShmHeader* hdr = header();
    uint32_t current_id = hdr->root_node_id;
    int matched = 0;
    result.last_node_id = current_id;

    // Terminal node: compare block by block from the node's start.
    auto match_tail = [&](TreeNode* cur) {
        int cmp = std::min((int)cur->num_blocks, n - matched);
        uint64_t* nh = data_.get_hashes(cur->data_offset);
        int ml = 0;
        while (ml < cmp && nh[ml] == hashes[matched + ml]) ml++;
        if (ml > 0 && touch) touch_node_atomic(current_id, ts);
        matched += ml;
        result.last_node_id = current_id;
        result.last_node_matched_blocks = ml;
    };

    while (matched < n) {
        TreeNode* cur = nodes_.get(current_id);
        bool is_root = (current_id == hdr->root_node_id);

        if (!is_root && matched + (int)cur->num_blocks >= n) {
            match_tail(cur);
            break;
        }

        int next_pos = matched + (is_root ? 0 : (int)cur->num_blocks);
        if (next_pos < n && cur->children_count > 0) {
            uint32_t child_id = child_table_.lookup(current_id, hashes[next_pos]);
            if (child_id != INVALID_NODE_ID) {
                if (!is_root) {
                    if (touch) touch_node_atomic(current_id, ts);
                    matched += (int)cur->num_blocks;
                }
                current_id = child_id;
                continue;
            }
        }

        if (!is_root) {
            match_tail(cur);
        } else {
            result.last_node_id = current_id;
            result.last_node_matched_blocks = 0;
        }
        break;
    }

    result.matched_blocks = matched;
    return result;
}

std::vector<MatchedNodeSpanView> RadixShmemTree::path_spans(
        const QueryInternalDetail& m) {
    std::vector<MatchedNodeSpanView> spans;
    const uint32_t root = header()->root_node_id;
    uint32_t id = m.last_node_id;
    uint32_t ml = (uint32_t)m.last_node_matched_blocks;
    uint32_t end = (uint32_t)m.matched_blocks;
    if (id == INVALID_NODE_ID || id == root) return spans;
    if (ml == 0) {
        id = nodes_.get(id)->parent_id;
        if (id == root || id == INVALID_NODE_ID) return spans;
        ml = nodes_.get(id)->num_blocks;
    }
    while (id != root && id != INVALID_NODE_ID) {
        TreeNode* node = nodes_.get(id);
        const int nb = (int)node->num_blocks;
        MatchedNodeSpanView v;
        v.node_id = id;
        v.absolute_start = end - ml;
        v.node_blocks = (uint32_t)nb;
        v.matched_blocks = ml;
        v.hashes = data_.get_hashes(node->data_offset);
        v.full_slots = data_.get_slots(node->data_offset, nb);
        v.swa_slots = data_.get_swa_slots(node->data_offset, nb);
        v.mamba_checkpoint_slot = node->mamba_checkpoint_slot;
        spans.push_back(v);
        end -= ml;
        id = node->parent_id;
        if (id != root && id != INVALID_NODE_ID)
            ml = nodes_.get(id)->num_blocks;
    }
    std::reverse(spans.begin(), spans.end());
    return spans;
}


void RadixShmemTree::inc_ref(uint32_t node_id) {
    nodes_.get(node_id)->ref_count.fetch_add(1, std::memory_order_relaxed);
}

bool RadixShmemTree::dec_ref(std::atomic<int64_t>& ref) noexcept {
    int64_t cur = ref.load(std::memory_order_relaxed);
    while (cur > 0) {
        if (ref.compare_exchange_weak(cur, cur - 1, std::memory_order_relaxed))
            return true;
    }
    return false;
}

void RadixShmemTree::dec_ref(uint32_t node_id) {
    // Legacy semantics: silently clamp at zero.
    (void)dec_ref(nodes_.get(node_id)->ref_count);
}

std::vector<NodeBoundary> RadixShmemTree::get_node_boundaries(
    const uint64_t* hashes, int n) {
    std::vector<NodeBoundary> boundaries;
    if (n <= 0) return boundaries;

    read_lock();
    uint32_t cur_id = header()->root_node_id;
    int matched = 0;

    while (matched < n) {
        TreeNode* cur = nodes_.get(cur_id);
        bool is_root = (cur_id == header()->root_node_id);

        if (!is_root) {
            int nb = (int)cur->num_blocks;
            uint64_t* nh = data_.get_hashes(cur->data_offset);
            int cmp = std::min(nb, n - matched);
            int ml = 0;
            for (int i = 0; i < cmp; i++) {
                if (nh[i] == hashes[matched + i]) ml++;
                else break;
            }
            if (ml > 0)
                boundaries.push_back({matched, ml, nh[0], cur_id, (uint32_t)nb});
            matched += ml;
            if (ml < nb) break;
        }

        if (matched < n && cur->children_count > 0) {
            uint32_t child = child_table_.lookup(cur_id, hashes[matched]);
            if (child != INVALID_NODE_ID) {
                cur_id = child;
                continue;
            }
        }
        break;
    }
    read_unlock();
    return boundaries;
}

InsertResult RadixShmemTree::insert(const uint64_t* hashes, int n,
                                  const int32_t* slots, int num_slots,
                                  int start, bool auto_recycle,
                                  ComponentType component) {
    // `start` positions FULL slots only; SWA covers the trailing num_slots
    // positions and MAMBA the END n, so both derive their own anchor.
    if (component != ComponentType::FULL && start != 0) {
        InsertResult result;
        result.error = InsertError::BAD_REQUEST;
        if (slots != nullptr && num_slots > 0) {
            result.unused_slots.assign(slots, slots + num_slots);
            if (auto_recycle) recycle_slots(slots, num_slots, component);
        }
        return result;
    }
    switch (component) {
    case ComponentType::FULL:
        return insert_impl(hashes, n, slots, num_slots, start, auto_recycle);
    case ComponentType::SWA:
        return insert_swa(hashes, n, slots, num_slots, auto_recycle);
    case ComponentType::MAMBA:
        return insert_mamba(hashes, n, slots, num_slots, auto_recycle);
    }
    InsertResult result;
    result.error = InsertError::BAD_REQUEST;
    if (slots != nullptr && num_slots > 0)
        result.unused_slots.assign(slots, slots + num_slots);
    return result;
}

// Core insert: ext_slots positionally bound from `start` (ext_slots[i] holds data
// for hashes[start+i]); leading redundant + unlanded suffix slots go to unused_slots.
InsertResult RadixShmemTree::insert_impl(const uint64_t* hashes, int n,
                                       const int32_t* ext_slots,
                                       int ext_num_slots,
                                       int start, bool auto_recycle) {
    set_write_op("insert");
    InsertResult result;

    // Recycle collected unused slots after unlock; returns result (used at exits).
    auto finish = [&]() -> InsertResult {
        if (auto_recycle && !result.unused_slots.empty())
            recycle_slots(result.unused_slots.data(),
                          (int)result.unused_slots.size());
        return result;
    };

    // Negative n is an invalid arg (empty return). n==0 is a legal no-op: nothing
    // to insert, so every supplied slot is unused (returned/recycled per contract).
    if (n < 0) return result;
    if (n == 0) {
        if (ext_num_slots > 0)
            result.unused_slots.assign(ext_slots, ext_slots + ext_num_slots);
        return finish();
    }

    write_lock();
    QueryInternalDetail match = query_internal(hashes, n, now_ts());

    // Strict-LRU: promote terminal leaf to tail (safe under write_lock).
    if (match.last_node_id != INVALID_NODE_ID &&
        is_in_leaf_list(match.last_node_id))
        leaf_list_move_to_tail(match.last_node_id);

    // Prefix broke (parent evicted/shrunk): real matched < start, can't form a
    // contiguous prefix from `start`. Reject entirely; all slots are unused.
    if (match.matched_blocks < start) {
        result.unused_slots.assign(ext_slots, ext_slots + ext_num_slots);
        write_unlock();
        return finish();
    }

    // no need to insert (matched covers the whole sequence)
    if (match.matched_blocks >= n) {
        result.unused_slots.assign(ext_slots, ext_slots + ext_num_slots);
        header()->insert_success.fetch_add(1, std::memory_order_relaxed);
        write_unlock();
        return finish();
    }

    int insert_start = match.matched_blocks;
    uint32_t parent_id = match.last_node_id;

    // Someone inserted [start, matched) ahead of us: skip those leading slots.
    int skip = insert_start - start;
    if (skip > ext_num_slots) skip = ext_num_slots;
    if (skip > 0)
        result.unused_slots.assign(ext_slots, ext_slots + skip);
    const int32_t* cons_slots = ext_slots + skip;
    int cons_num = ext_num_slots - skip;

    if (match.last_node_matched_blocks > 0 &&
        match.last_node_matched_blocks < (int)nodes_.get(parent_id)->num_blocks) {
        uint32_t prefix_id = split_node(parent_id, match.last_node_matched_blocks);
        if (prefix_id == INVALID_NODE_ID) {
            // Pool exhausted during split; abort gracefully, original untouched.
            result.error = InsertError::DATA_POOL_FULL;
            header()->insert_error_data_pool.fetch_add(1, std::memory_order_relaxed);
            // Append (not assign): keep the leading skip slots already recorded.
            result.unused_slots.insert(result.unused_slots.end(),
                                       cons_slots, cons_slots + cons_num);
            write_unlock();
            return finish();
        }
        TreeNode* suffix = nodes_.get(parent_id);
        result.split_node_id = parent_id;
        result.split_num_blocks = suffix->num_blocks;
        result.split_first_hash = data_.get_hashes(suffix->data_offset)[0];
        parent_id = prefix_id;
    }

    // Consumable region: cons_slots[0:new_slots] map to hashes[insert_start:].
    int new_slots = n - insert_start;
    if (new_slots > cons_num)
        new_slots = cons_num;
    std::vector<int32_t> cons(cons_slots, cons_slots + new_slots);

    if (new_slots <= 0) {
        if (parent_id != header()->root_node_id &&
            nodes_.get(parent_id)->children_count == 0 &&
            !is_in_leaf_list(parent_id))
            leaf_list_add(parent_id);
        if (result.error == InsertError::OK)
            header()->insert_success.fetch_add(1, std::memory_order_relaxed);
        write_unlock();
        flush_evicted();
        return finish();
    }

    static constexpr int MAX_BLOCKS_PER_NODE = BuddyHeader::MAX_DATA_BLOCKS;
    // cap = max blocks per node; cs_>0 aligns every boundary to a cs multiple.
    int cap = (cs_ > 0)
              ? std::min(cs_, MAX_BLOCKS_PER_NODE)
              : MAX_BLOCKS_PER_NODE;

    int remaining = new_slots;
    int pos = insert_start;
    int slot_idx = 0;
    // new_leaf_ids: only leaves built in this loop (split prefixes excluded).
    std::vector<uint32_t> new_leaf_ids;

    while (remaining > 0) {
        int boundary = ((pos / cap) + 1) * cap;
        int chunk = std::min({remaining, cap, boundary - pos});

        uint32_t new_id = nodes_.alloc();
        if (new_id == INVALID_NODE_ID) {
            evict(chunk, parent_id);
            new_id = nodes_.alloc();
            if (new_id == INVALID_NODE_ID) {
                result.error = InsertError::NODE_POOL_FULL;
                header()->insert_error_node_pool.fetch_add(1, std::memory_order_relaxed);
                fprintf(stderr, "[radixshmem] WARNING: node pool full (%u/%u), "
                        "increase max_nodes\n",
                        nodes_.size(), header()->node_pool_capacity);
                break;
            }
        }

        uint64_t data_off = data_.alloc(chunk);
        if (data_off == INVALID_OFFSET) {
            evict(chunk, parent_id);
            data_off = data_.alloc(chunk);
            if (data_off == INVALID_OFFSET) {
                nodes_.free(new_id);
                result.error = InsertError::DATA_POOL_FULL;
                header()->insert_error_data_pool.fetch_add(1, std::memory_order_relaxed);
                fprintf(stderr, "[radixshmem] WARNING: data pool full (%.1f%%), "
                        "increase max_blocks or data_pool_ratio\n",
                        data_.utilization() * 100.0f);
                break;
            }
        }

        TreeNode* new_node = nodes_.get(new_id);
    
        new_node->num_blocks  = chunk;
        new_node->data_offset = data_off;
        new_node->parent_id   = parent_id;

        memcpy(data_.get_hashes(data_off), hashes + pos,
               chunk * sizeof(uint64_t));
        memcpy(data_.get_slots(data_off, chunk),
               cons.data() + slot_idx,
               chunk * sizeof(int32_t));
        // SWA lane starts absent; a later SWA publish backfills it in place.
        if (has_swa_)
            std::fill_n(data_.get_swa_slots(data_off, chunk), chunk,
                        INVALID_COMPONENT_SLOT);

        if (!add_child(parent_id, hashes[pos], new_id)) {
            data_.free(data_off, chunk);
            nodes_.free(new_id);
            result.error = InsertError::HT_POOL_FULL;
            header()->insert_error_ht_pool.fetch_add(1, std::memory_order_relaxed);
            break;
        }

        header()->total_blocks.fetch_add(chunk, std::memory_order_relaxed);

        // Only the last chunk is a real leaf.
        new_node->last_access_time.store(now_ts(), std::memory_order_relaxed);
        if (chunk == remaining)
            leaf_list_add(new_id);

        new_leaf_ids.push_back(new_id);

        parent_id = new_id;
        pos += chunk;
        slot_idx += chunk;
        remaining -= chunk;
    }

    int inserted = slot_idx;

    // Return every unconsumed supplied slot: the unlanded suffix and any surplus beyond n.
    if (inserted < cons_num)
        result.unused_slots.insert(result.unused_slots.end(),
                                   cons_slots + inserted, cons_slots + cons_num);

    if (inserted > 0) {
        // Re-add last leaf only if not already listed (avoid double-free).
        if (!new_leaf_ids.empty() && result.error != InsertError::OK &&
            !is_in_leaf_list(new_leaf_ids.back())) {
            leaf_list_add(new_leaf_ids.back());
        }
    } else {
        if (parent_id != header()->root_node_id &&
            nodes_.get(parent_id)->children_count == 0 &&
            !is_in_leaf_list(parent_id))
            leaf_list_add(parent_id);
    }

    // Auto-merge (single-chunk extension): fold only-child leaf into its parent.
    if (result.error == InsertError::OK && new_leaf_ids.size() == 1) {
        uint32_t leaf_id = new_leaf_ids.back();
        uint32_t p_id = nodes_.get(leaf_id)->parent_id;
        if (p_id != header()->root_node_id &&
            nodes_.get(p_id)->children_count == 1) {
            int ln = (int)nodes_.get(leaf_id)->num_blocks;
            merge_leaf_into_parent(leaf_id, n - ln);
        }
    }

    if (result.error == InsertError::OK)
        header()->insert_success.fetch_add(1, std::memory_order_relaxed);

    write_unlock();
    flush_evicted();
    return finish();
}

// ==================== Evict (called inside write_lock) ====================

// Strict-LRU eviction; leaf_list ordered head=LRU, tail=MRU.
static inline uint64_t access_time_of(const TreeNode* n) {
    return n->last_access_time.load(std::memory_order_relaxed);
}

int RadixShmemTree::evict_walk(int need) {
    int freed = 0;
    std::vector<int32_t> tmp;
    // Restart from head after each eviction (evict_node mutates leaf_list);
    // skip pinned/root, stop when a full pass evicts nothing.
    while (freed < need) {
        uint32_t cur = header()->leaf_list_head;
        bool evicted_one = false;
        while (cur != INVALID_NODE_ID) {
            TreeNode* node = nodes_.get(cur);
            if (cur != header()->root_node_id && component_refs_zero(cur)) {
                freed += evict_node(cur, tmp);
                evicted_one = true;
                break;
            }
            cur = node->leaf_next;
        }
        if (!evicted_one) break;
    }
    return freed;
}

int RadixShmemTree::evict(int need, uint32_t protect) {
    if (protect != INVALID_NODE_ID) inc_ref(protect);
    int freed = evict_walk(need);
    if (protect != INVALID_NODE_ID) dec_ref(protect);
    return freed;
}

bool RadixShmemTree::node_head_matches(uint32_t node_id, uint64_t first_hash) {
    if (node_id >= header()->node_pool_capacity) return false;
    read_lock();
    const TreeNode* nd = nodes_.get(node_id);
    bool live = nd->parent_id != INVALID_NODE_ID && nd->num_blocks > 0 &&
                nd->data_offset != INVALID_OFFSET &&
                data_.get_hashes(nd->data_offset)[0] == first_hash;
    read_unlock();
    return live;
}

std::vector<EvictedNode>& RadixShmemTree::evicted_pending() {
    // Per-thread buffer: foreground inserter and background evictor never share.
    static thread_local std::vector<EvictedNode> buf;
    return buf;
}

void RadixShmemTree::flush_evicted() {
    // Call after the write lock is dropped. The buffer is thread_local, so the
    // foreground and evictor threads each drain their own.
    auto& batch = evicted_pending();
    if (!batch.empty()) {
        if (evict_cb_) evict_cb_(batch);
        for (auto& obs : evict_observers_) obs(batch);
        if (kvevent_writer_.enabled()) {
            // One publish per node: entries never straddle two nodes, so a
            // consumer that loses a slot loses whole blocks, never a
            // half-decoded hash list.
            for (const auto& e : batch)
                if (!e.block_hashes.empty())
                    kvevent_writer_.publish(e.block_hashes.data(),
                                            (uint32_t)e.block_hashes.size());
        }
    }
    batch.clear();
}

int RadixShmemTree::try_evict_batch(int target_blocks) {
    set_write_op("evict_bg");
    if (!try_write_lock()) return 0;
    int freed = evict_walk(target_blocks);
    write_unlock();
    flush_evicted();
    return freed;
}

int RadixShmemTree::try_evict_component_batch(ComponentType component,
                                            int target_slots) {
    set_write_op("evict_comp_bg");
    if (component != ComponentType::SWA && component != ComponentType::MAMBA)
        return -1;
    Mempool* pool = try_component_pool(component);
    if (!pool->enabled() || target_slots <= 0) return 0;
    if (!try_write_lock()) return 0;
    int freed = component == ComponentType::SWA
                    ? evict_swa_walk(target_slots)
                    : evict_mamba_walk(target_slots);
    write_unlock();
    return freed;
}

// ==================== Slot allocation (for external slot mgmt)

Mempool* RadixShmemTree::try_component_pool(ComponentType component) noexcept {
    switch (component) {
    case ComponentType::FULL:
        return &mempool_;
    case ComponentType::SWA:
        return &swa_mempool_;
    case ComponentType::MAMBA:
        return &mamba_mempool_;
    }
    return nullptr;
}

int RadixShmemTree::allocate_slots(int num_slots, std::vector<int32_t>& out,
                                 ComponentType component) {
    set_write_op("alloc");
    out.clear();
    Mempool* pool = try_component_pool(component);
    if (pool == nullptr) return -1;
    if (!pool->enabled()) return 0;
    // Bad args: non-positive, or larger than the pool can ever hold. Signal -1
    // rather than silently returning 0 / a truncated set.
    if (num_slots <= 0 || (uint32_t)num_slots > pool->total()) return -1;

    write_lock();
    int free_now = (int)pool->num_free();
    if (free_now < num_slots) {
        if (component != ComponentType::FULL) {
            // All-or-none: evict from this component's own chain, then retry
            // the same request exactly once.
            int target = num_slots - free_now;
            if (component == ComponentType::SWA)
                evict_swa_walk(target);
            else
                evict_mamba_walk(target);
            if ((int)pool->num_free() < num_slots) {
                write_unlock();
                return 0;
            }
        } else {
            evict(num_slots - free_now);
        }
    }
    out.resize(num_slots);
    int got = pool->allocate(num_slots, out.data());
    out.resize(got);
    assert(component == ComponentType::FULL || got == num_slots);
    write_unlock();
    flush_evicted();
    return got;
}

void RadixShmemTree::recycle_slots(const int32_t* slots, int num,
                                 ComponentType component) {
    set_write_op("recycle");
    Mempool* pool = try_component_pool(component);
    if (pool == nullptr || !pool->enabled() || num <= 0 || slots == nullptr)
        return;
    write_lock();
    pool->recycle(slots, num);   // idempotent: double-recycle is a no-op
    write_unlock();
}

// ==================== Stats & debug ====================

uint32_t RadixShmemTree::total_radix_nodes() const {
    return nodes_.size();
}

uint64_t RadixShmemTree::total_blocks() const {
    return header()->total_blocks.load(std::memory_order_relaxed);
}

float RadixShmemTree::data_pool_utilization() const {
    return data_.utilization();
}

uint32_t RadixShmemTree::node_pool_capacity() const {
    return header()->node_pool_capacity;
}

uint32_t RadixShmemTree::node_pool_free() const {
    return header()->node_pool_capacity - nodes_.size();
}

uint32_t RadixShmemTree::mempool_total() const { return mempool_.total(); }
uint32_t RadixShmemTree::mempool_free() const { return mempool_.num_free(); }
uint32_t RadixShmemTree::mempool_used() const { return mempool_.num_used(); }
uint32_t RadixShmemTree::swa_mempool_total() const { return swa_mempool_.total(); }
uint32_t RadixShmemTree::swa_mempool_free() const { return swa_mempool_.num_free(); }
uint32_t RadixShmemTree::mamba_mempool_total() const { return mamba_mempool_.total(); }
uint32_t RadixShmemTree::mamba_mempool_free() const { return mamba_mempool_.num_free(); }

ComponentConfig RadixShmemTree::component_config() const {
    return header()->component_config;
}

PeerFaultCounters* RadixShmemTree::peer_faults() const {
    return &header()->peer_faults;
}

uint64_t RadixShmemTree::insert_success_count() const {
    return header()->insert_success.load(std::memory_order_relaxed);
}
uint64_t RadixShmemTree::insert_error_node_pool_count() const {
    return header()->insert_error_node_pool.load(std::memory_order_relaxed);
}
uint64_t RadixShmemTree::insert_error_data_pool_count() const {
    return header()->insert_error_data_pool.load(std::memory_order_relaxed);
}
uint64_t RadixShmemTree::insert_error_ht_pool_count() const {
    return header()->insert_error_ht_pool.load(std::memory_order_relaxed);
}
uint64_t RadixShmemTree::insert_error_mempool_count() const {
    return header()->insert_error_mempool.load(std::memory_order_relaxed);
}

void RadixShmemTree::print_node(uint32_t node_id, int indent) const {
    TreeNode* node = nodes_.get(node_id);
    for (int i = 0; i < indent; i++) printf("  ");

    if (node_id == header()->root_node_id) {
        printf("[root] children=%u\n", node->children_count);
    } else {
        printf("[node %u] blocks=%u children=%u hashes=[", node_id,
               node->num_blocks, node->children_count);
        uint64_t* h = data_.get_hashes(node->data_offset);
        int32_t* s  = data_.get_slots(node->data_offset, node->num_blocks);
        for (int i = 0; i < (int)node->num_blocks; i++) {
            if (i) printf(",");
            printf("%llx", (unsigned long long)h[i]);
        }
        printf("] slots=[");
        for (int i = 0; i < (int)node->num_blocks; i++) {
            if (i) printf(",");
            printf("%d", s[i]);
        }
        if (has_swa_) {
            printf("] swa=[");
            int32_t* sw = data_.get_swa_slots(node->data_offset,
                                              node->num_blocks);
            for (int i = 0; i < (int)node->num_blocks; i++) {
                if (i) printf(",");
                printf("%d", sw[i]);
            }
        }
        printf("] ckpt=%d\n", node->mamba_checkpoint_slot);
    }

    if (node->children_count > 0) {
        std::vector<uint32_t> children;
        child_table_.collect_children(node, children);
        for (uint32_t cid : children)
            print_node(cid, indent + 1);
    }
}

void RadixShmemTree::print_tree() const {
    printf("=== RadixShmemTree (nodes=%u blocks=%llu) ===\n",
           total_radix_nodes(), (unsigned long long)total_blocks());
    print_node(header()->root_node_id, 0);
    printf("===\n");
}

} // namespace radixshmem
