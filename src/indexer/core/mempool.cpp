// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mempool.hpp"

#include <algorithm>
#include <cstring>

namespace shmradix {

Mempool::Mempool(MempoolHeader* header, uint32_t* free_stack,
                 uint64_t* used_bitmap)
    : header_(header), free_stack_(free_stack), used_bitmap_(used_bitmap) {}

void Mempool::init(MempoolHeader* header, uint32_t* free_stack,
                   uint64_t* used_bitmap, uint32_t total_blocks) {
    header->total_blocks = total_blocks;
    header->stack_top = total_blocks;
    for (uint32_t i = 0; i < total_blocks; i++)
        free_stack[i] = i;
    // 0 = free: every slot starts free.
    std::memset(used_bitmap, 0, bitmap_words(total_blocks) * sizeof(uint64_t));
}

static void mempool_region_ptrs(void* region_base, uint32_t total_blocks,
                                MempoolHeader** hdr, uint32_t** stack,
                                uint64_t** bitmap) {
    *hdr = reinterpret_cast<MempoolHeader*>(region_base);
    uint64_t stack_off = align_up(sizeof(MempoolHeader), 32);
    *stack = reinterpret_cast<uint32_t*>((char*)region_base + stack_off);
    uint64_t bitmap_off = align_up(
        stack_off + (uint64_t)total_blocks * sizeof(uint32_t), 8);
    *bitmap = reinterpret_cast<uint64_t*>((char*)region_base + bitmap_off);
}

Mempool Mempool::attach_region(void* region_base, uint32_t total_blocks) {
    if (!region_base || total_blocks == 0) return Mempool();
    MempoolHeader* hdr; uint32_t* stack; uint64_t* bitmap;
    mempool_region_ptrs(region_base, total_blocks, &hdr, &stack, &bitmap);
    return Mempool(hdr, stack, bitmap);
}

void Mempool::init_region(void* region_base, uint32_t total_blocks) {
    if (!region_base || total_blocks == 0) return;
    MempoolHeader* hdr; uint32_t* stack; uint64_t* bitmap;
    mempool_region_ptrs(region_base, total_blocks, &hdr, &stack, &bitmap);
    init(hdr, stack, bitmap, total_blocks);
}

int Mempool::allocate(int num, int32_t* out_ids) {
    if (!header_ || num <= 0) return 0;
    int avail = std::min(num, (int)header_->stack_top);
    header_->stack_top -= avail;
    for (int i = 0; i < avail; i++) {
        uint32_t id = free_stack_[header_->stack_top + i];
        out_ids[i] = (int32_t)id;
        mark_used(id);
    }
    return avail;
}

int Mempool::recycle(const int32_t* ids, int num) {
    if (!header_ || num <= 0 || !ids) return 0;
    const uint32_t total = header_->total_blocks;
    int freed = 0;
    for (int i = 0; i < num; i++) {
        int32_t id = ids[i];
        if (id < 0 || (uint32_t)id >= total) continue;   // out of range
        if (!is_used(id)) continue;                       // already free / dup
        mark_free(id);
        free_stack_[header_->stack_top++] = (uint32_t)id;
        freed++;
    }
    return freed;
}

void Mempool::reset() {
    if (!header_) return;
    init(header_, free_stack_, used_bitmap_, header_->total_blocks);
}

uint32_t Mempool::num_free() const {
    return header_ ? header_->stack_top : 0;
}

uint32_t Mempool::num_used() const {
    return header_ ? header_->total_blocks - header_->stack_top : 0;
}

uint32_t Mempool::total() const {
    return header_ ? header_->total_blocks : 0;
}

bool Mempool::enabled() const {
    return header_ != nullptr;
}

} // namespace shmradix
