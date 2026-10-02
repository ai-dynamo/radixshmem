// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "slot_lru.hpp"

namespace shmradix {

SlotLru::SlotLru(SlotLruEntry* entries, uint32_t capacity,
                 AuxListAnchors* anchors)
    : entries_(entries), capacity_(capacity), anchors_(anchors) {}

void SlotLru::init_region(void* base, uint32_t capacity) {
    auto* e = reinterpret_cast<SlotLruEntry*>(base);
    for (uint32_t i = 0; i < capacity; i++)
        e[i] = SlotLruEntry{INVALID_SLOT_LINK, INVALID_SLOT_LINK,
                            INVALID_NODE_ID, 0, 0};
}

void SlotLru::reset() {
    init_region(entries_, capacity_);
    *anchors_ = AuxListAnchors{INVALID_SLOT_LINK, INVALID_SLOT_LINK};
}

void SlotLru::link_tail(int32_t slot) {
    SlotLruEntry& e = entries_[slot];
    e.prev = anchors_->tail;
    e.next = INVALID_SLOT_LINK;
    if (anchors_->tail != INVALID_SLOT_LINK)
        entries_[anchors_->tail].next = (uint32_t)slot;
    else
        anchors_->head = (uint32_t)slot;
    anchors_->tail = (uint32_t)slot;
}

void SlotLru::push_tail(int32_t slot, uint32_t node_id, uint32_t idx,
                        uint64_t ts) {
    if (linked(slot)) unlink(slot);
    SlotLruEntry& e = entries_[slot];
    e.node_id = node_id;
    e.idx = idx;
    e.ts = ts;
    link_tail(slot);
}

void SlotLru::move_tail(int32_t slot, uint64_t ts) {
    if (!linked(slot)) return;
    SlotLruEntry& e = entries_[slot];
    e.ts = ts;
    if (anchors_->tail == (uint32_t)slot) return;
    if (e.prev != INVALID_SLOT_LINK) entries_[e.prev].next = e.next;
    else anchors_->head = e.next;
    entries_[e.next].prev = e.prev;   // not the tail, so next exists
    link_tail(slot);
}

void SlotLru::unlink(int32_t slot) {
    if (!linked(slot)) return;
    SlotLruEntry& e = entries_[slot];
    if (e.prev != INVALID_SLOT_LINK) entries_[e.prev].next = e.next;
    else anchors_->head = e.next;
    if (e.next != INVALID_SLOT_LINK) entries_[e.next].prev = e.prev;
    else anchors_->tail = e.prev;
    e = SlotLruEntry{INVALID_SLOT_LINK, INVALID_SLOT_LINK, INVALID_NODE_ID, 0, 0};
}

} // namespace shmradix
