// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "shmradix/component_kind.h"
#include "shmradix/slot_geometry.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

namespace shmradix {

class ShmRegion;

inline constexpr uint64_t SLOT_STORE_MAGIC   = 0x524F5453544F4C53ULL;
inline constexpr uint32_t SLOT_STORE_VERSION = 2;   // 2: per-component pools

struct SlotPoolDesc {
    uint64_t num_slots;     // 0 = absent (all fields 0)
    uint64_t slot_bytes;    // stride, rounded up to slot_align
    uint64_t data_offset;   // slot 0 of this pool
    uint64_t _rsv;
};
static_assert(sizeof(SlotPoolDesc) == 32, "SlotPoolDesc must be 32 bytes");

struct SlotStoreHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t pool_mask;     // COMPONENT_MASK_* of present pools
    uint64_t data_offset;   // start of the registerable range (first pool)
    uint64_t data_bytes;    // registerable range length
    uint64_t total_bytes;   // whole mapping
    uint64_t slot_align;
    int32_t  owner_pid;
    uint32_t _pad0;
    SlotPoolDesc pools[NUM_COMPONENT_TYPES];   // indexed by ComponentType
    std::atomic<uint32_t> ready;   // written last; attach() spins on it
    uint32_t _pad1;
    uint64_t _rsv[6];
};

static_assert(std::atomic<uint32_t>::is_always_lock_free,
              "SlotStoreHeader::ready must be lock-free to live in shared memory");
static_assert(sizeof(SlotStoreHeader) <= 4096,
              "SlotStoreHeader must fit below the first aligned pool");

struct SlotPoolConfig {
    uint64_t num_slots  = 0;   // == the indexer's capacity for this component
    uint64_t slot_bytes = 0;   // payload bytes per slot (unrounded)

    bool enabled() const noexcept { return num_slots != 0 || slot_bytes != 0; }
};

struct SlotStoreConfig {
    std::string    name;                 // POSIX shm name
    SlotPoolConfig full;                 // required
    SlotPoolConfig swa;
    SlotPoolConfig mamba;
    uint64_t       slot_align = 4096;    // stride alignment (power of two)
    std::string    hugepage_path = "/dev/hugepages";  // empty = plain shm_open
    bool           prefault   = true;    // MAP_POPULATE

    const SlotPoolConfig& pool(ComponentType t) const noexcept {
        switch (t) {
            case ComponentType::SWA:   return swa;
            case ComponentType::MAMBA: return mamba;
            default:                   return full;
        }
    }
    uint32_t pool_mask() const noexcept {
        uint32_t m = 0;
        for (int i = 0; i < NUM_COMPONENT_TYPES; ++i)
            if (pool(static_cast<ComponentType>(i)).enabled()) m |= 1u << i;
        return m;
    }
};

class SlotStore {
public:
    SlotStore();
    ~SlotStore();

    SlotStore(SlotStore&&) noexcept;
    SlotStore& operator=(SlotStore&&) noexcept;
    SlotStore(const SlotStore&) = delete;
    SlotStore& operator=(const SlotStore&) = delete;

    // ==================== Control path ====================

    static SlotStore create(const SlotStoreConfig& cfg);

    static SlotStore attach(const std::string& name,
                            const std::string& hugepage_path = "/dev/hugepages",
                            int timeout_ms = 5000);

    // Remove a leftover region by name (crash cleanup). Never throws.
    static void destroy(const std::string& name,
                        const std::string& hugepage_path = "/dev/hugepages");

    // ==================== Data path (no throw, no allocation) ====================

    void* slot_addr(ComponentType c, uint64_t slot_id) const {
        const Pool* p = pool_ptr(c);
        if (!p || slot_id >= p->num_slots) return nullptr;
        return p->base + slot_id * p->stride;
    }
    uint64_t slot_offset(ComponentType c, uint64_t slot_id) const {
        const Pool* p = pool_ptr(c);
        if (!p || slot_id >= p->num_slots) return UINT64_MAX;
        return (uint64_t)(p->base - data_base_) + slot_id * p->stride;
    }

    // FULL-pool shorthands.
    void*    slot_addr(uint64_t slot_id) const { return slot_addr(ComponentType::FULL, slot_id); }
    uint64_t slot_offset(uint64_t slot_id) const { return slot_offset(ComponentType::FULL, slot_id); }
    uint64_t num_slots()  const { return pools_[0].num_slots; }
    uint64_t slot_bytes() const { return pools_[0].stride; }

    // Per-pool geometry; 0 / nullptr for an absent pool.
    bool     has_pool(ComponentType c) const { return pool_ptr(c) != nullptr; }
    uint32_t pool_mask() const { return pool_mask_; }
    uint64_t num_slots(ComponentType c) const  { const Pool* p = pool_ptr(c); return p ? p->num_slots : 0; }
    uint64_t slot_bytes(ComponentType c) const { const Pool* p = pool_ptr(c); return p ? p->stride : 0; }
    uint64_t pool_offset(ComponentType c) const {   // relative to data_base()
        const Pool* p = pool_ptr(c); return p ? (uint64_t)(p->base - data_base_) : 0;
    }
    uint64_t pool_bytes(ComponentType c) const {    // num_slots x stride
        const Pool* p = pool_ptr(c); return p ? p->num_slots * p->stride : 0;
    }
    void* pool_base(ComponentType c) const { const Pool* p = pool_ptr(c); return p ? p->base : nullptr; }

    // Pool offsets relative to data_base(): what a node publishes.
    StoreGeometry layout() const;

    // The one range the transfer plane registers (header excluded).
    void*    data_base()  const { return data_base_; }
    uint64_t data_bytes() const { return data_bytes_; }

    bool     valid()      const { return data_base_ != nullptr; }

    bool is_owner() const;
    bool is_hugepage() const;
    const std::string& name() const { return name_; }

private:
    struct Pool {
        char*    base      = nullptr;
        uint64_t num_slots = 0;
        uint64_t stride    = 0;
    };
    const Pool* pool_ptr(ComponentType c) const {
        const uint32_t i = static_cast<uint32_t>(c);
        if (i >= (uint32_t)NUM_COMPONENT_TYPES || !(pool_mask_ & (1u << i))) return nullptr;
        return &pools_[i];
    }

    void cache_geometry();   // header -> hot-path members
    void steal(SlotStore& o) noexcept;

    std::unique_ptr<ShmRegion> region_;
    std::string name_;
    char*    data_base_  = nullptr;
    uint64_t data_bytes_ = 0;
    uint32_t pool_mask_  = 0;
    Pool     pools_[NUM_COMPONENT_TYPES];
};

} // namespace shmradix
