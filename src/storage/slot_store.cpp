// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "slot_store.h"

#include "shm_region.hpp"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <unistd.h>

namespace radixshmem {

namespace {

bool is_pow2(uint64_t v) { return v != 0 && (v & (v - 1)) == 0; }

// Pools start on their own aligned page: header and pools never share one.
uint64_t pool_align(uint64_t slot_align) {
    return slot_align > 4096 ? slot_align : 4096;
}

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error(what); }

} // namespace

SlotStore::SlotStore()  = default;
SlotStore::~SlotStore() = default;

void SlotStore::steal(SlotStore& o) noexcept {
    region_     = std::move(o.region_);
    name_       = std::move(o.name_);
    data_base_  = o.data_base_;
    data_bytes_ = o.data_bytes_;
    pool_mask_  = o.pool_mask_;
    for (int i = 0; i < NUM_COMPONENT_TYPES; ++i) pools_[i] = o.pools_[i];
    o.data_base_  = nullptr;
    o.data_bytes_ = 0;
    o.pool_mask_  = 0;
    for (int i = 0; i < NUM_COMPONENT_TYPES; ++i) o.pools_[i] = Pool{};
}

SlotStore::SlotStore(SlotStore&& o) noexcept { steal(o); }

SlotStore& SlotStore::operator=(SlotStore&& o) noexcept {
    if (this != &o) steal(o);
    return *this;
}

bool SlotStore::is_owner() const { return region_ && region_->is_owner(); }
bool SlotStore::is_hugepage() const { return region_ && region_->is_hugepage(); }

// Header -> hot-path members, read once.
void SlotStore::cache_geometry() {
    auto* hdr = static_cast<SlotStoreHeader*>(region_->base());
    char* base  = static_cast<char*>(region_->base());
    data_base_  = base + hdr->data_offset;
    data_bytes_ = hdr->data_bytes;
    pool_mask_  = hdr->pool_mask;
    for (int i = 0; i < NUM_COMPONENT_TYPES; ++i) {
        const SlotPoolDesc& d = hdr->pools[i];
        if (pool_mask_ & (1u << i)) {
            pools_[i].base      = base + d.data_offset;
            pools_[i].num_slots = d.num_slots;
            pools_[i].stride    = d.slot_bytes;
        } else {
            pools_[i] = Pool{};
        }
    }
}

StoreGeometry SlotStore::layout() const {
    StoreGeometry g;
    g.pool_mask = pool_mask_;
    for (int i = 0; i < NUM_COMPONENT_TYPES; ++i) {
        if (!(pool_mask_ & (1u << i))) continue;
        g.pools[i].base       = (uint64_t)(pools_[i].base - data_base_);
        g.pools[i].num_slots  = pools_[i].num_slots;
        g.pools[i].slot_bytes = pools_[i].stride;
    }
    return g;
}

SlotStore SlotStore::create(const SlotStoreConfig& cfg) {
    if (cfg.name.empty())
        fail("SlotStore::create: empty name");
    if (cfg.full.num_slots == 0 || cfg.full.slot_bytes == 0)
        fail("SlotStore::create: the FULL pool needs num_slots and slot_bytes > 0");
    for (int i = 1; i < NUM_COMPONENT_TYPES; ++i) {
        const ComponentType t = static_cast<ComponentType>(i);
        const SlotPoolConfig& p = cfg.pool(t);
        // Half-specified pool: refuse rather than guess.
        if (p.enabled() && (p.num_slots == 0 || p.slot_bytes == 0))
            fail(std::string("SlotStore::create: ") + component_type_name(t) +
                 " pool needs both num_slots and slot_bytes > 0 (or both 0)");
    }
    if (!is_pow2(cfg.slot_align))
        fail("SlotStore::create: slot_align must be a power of two");

    const uint64_t A = pool_align(cfg.slot_align);
    const uint64_t data_offset = region_align_up(sizeof(SlotStoreHeader), A);

    // Pools in ComponentType order; stride rounded to slot_align, start to A.
    // No padding after the last pool, so FULL-only data_bytes == n x stride.
    SlotPoolDesc pools[NUM_COMPONENT_TYPES] = {};
    uint32_t mask = 0;
    uint64_t cursor = data_offset;
    for (int i = 0; i < NUM_COMPONENT_TYPES; ++i) {
        const ComponentType t = static_cast<ComponentType>(i);
        const SlotPoolConfig& p = cfg.pool(t);
        if (!p.enabled()) continue;

        // A wrapped round-up would give stride 0 and pass every size check.
        uint64_t stride = 0;
        if (!region_align_up_checked(p.slot_bytes, cfg.slot_align, &stride))
            fail(std::string("SlotStore::create: ") + component_type_name(t) +
                 " slot_bytes overflows the alignment round-up");
        uint64_t start = 0, bytes = 0, end = 0;
        if (!region_align_up_checked(cursor, A, &start) ||
            __builtin_mul_overflow(p.num_slots, stride, &bytes) ||
            __builtin_add_overflow(start, bytes, &end))
            fail(std::string("SlotStore::create: ") + component_type_name(t) +
                 " pool size overflow");

        pools[i].num_slots   = p.num_slots;
        pools[i].slot_bytes  = stride;
        pools[i].data_offset = start;
        mask |= 1u << i;
        cursor = end;
    }
    const uint64_t total_bytes = cursor;
    const uint64_t data_bytes  = total_bytes - data_offset;

    SlotStore s;
    s.name_ = cfg.name;
    s.region_ = std::make_unique<ShmRegion>(
        ShmRegion::create(cfg.name, total_bytes, cfg.hugepage_path, cfg.prefault));

    // Only the header is zeroed; a fresh shm object is already zero-filled.
    auto* hdr = static_cast<SlotStoreHeader*>(s.region_->base());
    std::memset(hdr, 0, sizeof(SlotStoreHeader));
    hdr->magic       = SLOT_STORE_MAGIC;
    hdr->version     = SLOT_STORE_VERSION;
    hdr->pool_mask   = mask;
    hdr->data_offset = data_offset;
    hdr->data_bytes  = data_bytes;
    hdr->total_bytes = total_bytes;
    hdr->slot_align  = cfg.slot_align;
    hdr->owner_pid   = (int32_t)::getpid();
    for (int i = 0; i < NUM_COMPONENT_TYPES; ++i) hdr->pools[i] = pools[i];
    hdr->ready.store(1, std::memory_order_release);   // publish last

    s.cache_geometry();
    return s;
}

SlotStore SlotStore::attach(const std::string& name,
                            const std::string& hugepage_path, int timeout_ms) {
    SlotStore s;
    s.name_ = name;
    s.region_ = std::make_unique<ShmRegion>(
        ShmRegion::attach(name, sizeof(SlotStoreHeader), hugepage_path));

    auto* hdr = static_cast<SlotStoreHeader*>(s.region_->base());

    // Owner publishes `ready` last; spin until it lands.
    bool ready = false;
    for (int waited = 0; waited <= timeout_ms; waited += 1) {
        if (hdr->ready.load(std::memory_order_acquire) == 1) { ready = true; break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!ready)
        fail("SlotStore::attach: region not ready: " + name);

    if (hdr->magic != SLOT_STORE_MAGIC)
        fail("SlotStore::attach: bad magic: " + name);
    if (hdr->version != SLOT_STORE_VERSION)
        fail("SlotStore::attach: version mismatch (want " +
             std::to_string(SLOT_STORE_VERSION) + ", got " +
             std::to_string(hdr->version) + "): " + name);

    // Re-derive and bound everything the hot path uses, mirroring create().
    const auto corrupt = [&](const char* why) {
        fail(std::string("SlotStore::attach: corrupt geometry (") + why + "): " + name);
    };
    const uint32_t mask = hdr->pool_mask;
    if ((mask & COMPONENT_MASK_FULL) == 0 || (mask & ~COMPONENT_MASK_ALL) != 0)
        corrupt("pool mask");
    if (!is_pow2(hdr->slot_align)) corrupt("slot_align");
    uint64_t data_end = 0;
    if (hdr->data_offset < sizeof(SlotStoreHeader) ||
        hdr->total_bytes > s.region_->bytes() ||
        __builtin_add_overflow(hdr->data_offset, hdr->data_bytes, &data_end) ||
        data_end > hdr->total_bytes)
        corrupt("data range");

    uint64_t prev_end = hdr->data_offset;
    bool first = true;
    for (int i = 0; i < NUM_COMPONENT_TYPES; ++i) {
        const SlotPoolDesc& d = hdr->pools[i];
        if (!(mask & (1u << i))) {
            if (d.num_slots != 0 || d.slot_bytes != 0 || d.data_offset != 0)
                corrupt("absent pool not zeroed");
            continue;
        }
        uint64_t bytes = 0, end = 0;
        if (d.num_slots == 0 || d.slot_bytes == 0 ||
            d.data_offset % hdr->slot_align != 0 ||
            d.data_offset < prev_end ||                       // ordered, disjoint
            (first && d.data_offset != hdr->data_offset) ||   // range starts at pool 0
            __builtin_mul_overflow(d.num_slots, d.slot_bytes, &bytes) ||
            __builtin_add_overflow(d.data_offset, bytes, &end) ||
            end > data_end)
            corrupt(component_type_name(static_cast<ComponentType>(i)));
        prev_end = end;
        first = false;
    }
    if (prev_end != data_end) corrupt("range ends past the last pool");

    s.cache_geometry();
    return s;
}

void SlotStore::destroy(const std::string& name, const std::string& hugepage_path) {
    ShmRegion::unlink(name, hugepage_path);
}

} // namespace radixshmem
