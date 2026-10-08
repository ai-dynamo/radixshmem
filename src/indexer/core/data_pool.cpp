// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "data_pool.hpp"

#include <cstring>
#include <stdexcept>

namespace radixshmem {

DataPool::DataPool(void* shm_base, BuddyHeader* header, bool swa_lane)
    : base_(shm_base), header_(header), swa_lane_(swa_lane) {}

void DataPool::init(void* shm_base, BuddyHeader* header,
                    uint64_t data_pool_offset,
                    uint64_t buddy_hdr_size, uint64_t bitmap_bytes,
                    uint64_t omap_bytes, uint64_t blocks_size) {
    constexpr uint64_t max_order_block =
        BuddyHeader::order_size(BuddyHeader::MAX_ORDER);

    for (int i = 0; i < BuddyHeader::NUM_ORDERS; i++)
        header->free_list_head[i] = INVALID_OFFSET;

    header->bitmap_offset    = data_pool_offset + buddy_hdr_size;
    header->order_map_offset = header->bitmap_offset + bitmap_bytes;
    header->blocks_base      = data_pool_offset +
                               align_up(buddy_hdr_size + bitmap_bytes + omap_bytes, 16);
    header->blocks_size      = blocks_size;
    header->num_min_blocks   = (uint32_t)(blocks_size / BuddyHeader::MIN_BLOCK_SIZE);

    DataPool pool(shm_base, header, /*swa_lane=*/false);
    pool.rebuild_free_lists();
}

void DataPool::reset() {
    rebuild_free_lists();
}

void DataPool::rebuild_free_lists() {
    constexpr uint64_t max_order_block =
        BuddyHeader::order_size(BuddyHeader::MAX_ORDER);

    for (int i = 0; i < BuddyHeader::NUM_ORDERS; i++)
        header_->free_list_head[i] = INVALID_OFFSET;
    header_->allocated_bytes.store(0, std::memory_order_relaxed);

    uint64_t bitmap_bytes = align_up((uint64_t)(header_->num_min_blocks + 7) / 8, 8);
    memset(bitmap(), 0, bitmap_bytes);
    memset(order_map(), BuddyHeader::ORDER_NONE, header_->num_min_blocks);

    uint32_t num_max_blocks = (uint32_t)(header_->blocks_size / max_order_block);
    for (uint32_t i = 0; i < num_max_blocks; i++) {
        uint64_t off = header_->blocks_base + (uint64_t)i * max_order_block;
        uint32_t idx = i * (1u << BuddyHeader::MAX_ORDER);
        order_map()[idx] = (uint8_t)BuddyHeader::MAX_ORDER;
        free_list_push(BuddyHeader::MAX_ORDER, off);
    }
}

int DataPool::blocks_to_order(int num_blocks) const {
    size_t need = data_block_bytes(num_blocks, swa_lane_);
    if (need < BuddyHeader::MIN_BLOCK_SIZE)
        need = BuddyHeader::MIN_BLOCK_SIZE;
    int order = 0;
    size_t sz = BuddyHeader::MIN_BLOCK_SIZE;
    while (sz < need) { sz <<= 1; order++; }
    if (order > BuddyHeader::MAX_ORDER)
        throw std::runtime_error("Requested block exceeds MAX_ORDER");
    return order;
}

uint64_t* DataPool::get_hashes(uint64_t offset) const {
    return reinterpret_cast<uint64_t*>((char*)base_ + offset);
}

int32_t* DataPool::get_slots(uint64_t offset, int num_blocks) const {
    return reinterpret_cast<int32_t*>(
        (char*)base_ + offset + num_blocks * sizeof(uint64_t));
}

int32_t* DataPool::get_swa_slots(uint64_t offset, int num_blocks) const {
    if (!swa_lane_) return nullptr;
    return reinterpret_cast<int32_t*>(
        (char*)base_ + offset + num_blocks * (sizeof(uint64_t) + sizeof(int32_t)));
}

// ---- Bitmap ----

uint8_t* DataPool::bitmap() const {
    return reinterpret_cast<uint8_t*>((char*)base_ + header_->bitmap_offset);
}

uint8_t* DataPool::order_map() const {
    return reinterpret_cast<uint8_t*>((char*)base_ + header_->order_map_offset);
}

uint32_t DataPool::offset_to_index(uint64_t offset) const {
    return (uint32_t)((offset - header_->blocks_base) /
                      BuddyHeader::MIN_BLOCK_SIZE);
}

uint64_t DataPool::index_to_offset(uint32_t index) const {
    return header_->blocks_base +
           (uint64_t)index * BuddyHeader::MIN_BLOCK_SIZE;
}

void DataPool::bitmap_set(uint32_t start, int order) {
    uint8_t* bm = bitmap();
    uint32_t count = 1u << order;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t bit = start + i;
        bm[bit >> 3] |= (1u << (bit & 7));
    }
}

void DataPool::bitmap_clear(uint32_t start, int order) {
    uint8_t* bm = bitmap();
    uint32_t count = 1u << order;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t bit = start + i;
        bm[bit >> 3] &= ~(1u << (bit & 7));
    }
}

bool DataPool::bitmap_all_free(uint32_t start, int order) const {
    uint8_t* bm = bitmap();
    uint32_t count = 1u << order;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t bit = start + i;
        if (bm[bit >> 3] & (1u << (bit & 7))) return false;
    }
    return true;
}

// ---- Free list ----

void DataPool::free_list_push(int order, uint64_t offset) {
    auto* blk = reinterpret_cast<FreeBlockHeader*>((char*)base_ + offset);
    blk->prev = INVALID_OFFSET;
    blk->next = header_->free_list_head[order];
    if (header_->free_list_head[order] != INVALID_OFFSET) {
        auto* old_head = reinterpret_cast<FreeBlockHeader*>(
            (char*)base_ + header_->free_list_head[order]);
        old_head->prev = offset;
    }
    header_->free_list_head[order] = offset;
}

void DataPool::free_list_remove(int order, uint64_t offset) {
    auto* blk = reinterpret_cast<FreeBlockHeader*>((char*)base_ + offset);
    if (blk->prev != INVALID_OFFSET) {
        auto* p = reinterpret_cast<FreeBlockHeader*>(
            (char*)base_ + blk->prev);
        p->next = blk->next;
    } else {
        header_->free_list_head[order] = blk->next;
    }
    if (blk->next != INVALID_OFFSET) {
        auto* n = reinterpret_cast<FreeBlockHeader*>(
            (char*)base_ + blk->next);
        n->prev = blk->prev;
    }
}

uint64_t DataPool::free_list_pop(int order) {
    uint64_t off = header_->free_list_head[order];
    if (off == INVALID_OFFSET) return INVALID_OFFSET;
    free_list_remove(order, off);
    return off;
}

// ---- Alloc / Free ----

uint64_t DataPool::alloc(int num_blocks) {
    int target = blocks_to_order(num_blocks);

    int found = -1;
    for (int o = target; o < BuddyHeader::NUM_ORDERS; o++) {
        if (header_->free_list_head[o] != INVALID_OFFSET) {
            found = o;
            break;
        }
    }
    if (found < 0)
        return INVALID_OFFSET;

    uint64_t block = free_list_pop(found);
    uint32_t idx = offset_to_index(block);
    order_map()[idx] = BuddyHeader::ORDER_NONE;

    for (int o = found; o > target; o--) {
        uint64_t right_off = block + BuddyHeader::order_size(o - 1);
        uint32_t right_idx = offset_to_index(right_off);
        order_map()[right_idx] = (uint8_t)(o - 1);
        free_list_push(o - 1, right_off);
    }

    bitmap_set(idx, target);
    header_->allocated_bytes.fetch_add(BuddyHeader::order_size(target),
                                       std::memory_order_relaxed);
    return block;
}

void DataPool::free(uint64_t offset, int num_blocks) {
    if (offset == INVALID_OFFSET) return;

    int order = blocks_to_order(num_blocks);
    header_->allocated_bytes.fetch_sub(BuddyHeader::order_size(order),
                                       std::memory_order_relaxed);
    uint32_t idx = offset_to_index(offset);

    bitmap_clear(idx, order);

    while (order < BuddyHeader::MAX_ORDER) {
        uint32_t buddy_idx = idx ^ (1u << order);
        if (buddy_idx >= header_->num_min_blocks) break;
        if (!bitmap_all_free(buddy_idx, order)) break;
        if (order_map()[buddy_idx] != (uint8_t)order) break;

        uint64_t buddy_off = index_to_offset(buddy_idx);
        free_list_remove(order, buddy_off);
        order_map()[buddy_idx] = BuddyHeader::ORDER_NONE;

        if (buddy_idx < idx) idx = buddy_idx;
        order++;
    }

    uint64_t merged_off = index_to_offset(idx);
    order_map()[idx] = (uint8_t)order;
    free_list_push(order, merged_off);
}

float DataPool::utilization() const {
    if (header_->blocks_size == 0) return 0.0f;
    uint64_t used = header_->allocated_bytes.load(std::memory_order_relaxed);
    return (float)used / (float)header_->blocks_size;
}

} // namespace radixshmem
