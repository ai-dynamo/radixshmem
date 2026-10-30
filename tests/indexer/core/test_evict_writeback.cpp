// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Background eviction with write-back: held slots, pinned prefixes, WritebackQueue.

#include "radix_tree.h"
#include "test_utils.h"
#include "writeback.h"

#include <cassert>
#include <cstdint>
#include <sys/mman.h>
#include <vector>

using namespace radixshmem;

static std::vector<uint64_t> seq(uint64_t base, int n) {
    std::vector<uint64_t> h(n);
    for (int i = 0; i < n; i++) h[i] = base + (uint64_t)i;
    return h;
}

static ShmConfig full_cfg(uint32_t max_blocks, int cs) {
    ShmConfig c;
    c.max_nodes = 64;
    c.max_blocks = max_blocks;
    c.register_chunk_tokens = cs;   // a leaf starting on a cs multiple is never merged
    c.hugepage_path = "";
    return c;
}

static ShmConfig aux_cfg() {
    ShmConfig c;
    c.max_nodes = 64;
    c.max_blocks = 64;
    c.register_chunk_tokens = 4;
    c.hugepage_path = "";
    c.component_mask = COMPONENT_MASK_ALL;
    c.swa_window_blocks = 4;
    c.mamba_state_bytes = 256;
    c.swa_max_blocks = 64;
    c.mamba_max_slots = 16;
    return c;
}

static std::vector<int32_t> insert_fresh(RadixShmemTree& t, const std::vector<uint64_t>& h,
                                         int n, int start = 0) {
    std::vector<int32_t> slots;
    int got = t.allocate_slots(n - start, slots);
    assert(got == n - start);
    auto r = t.insert(h.data(), n, slots.data(), (int)slots.size(), start, true);
    assert(r.error == InsertError::OK);
    return slots;
}

static void publish(RadixShmemTree& t, const std::vector<uint64_t>& h, int n,
                    ComponentType kind, int count) {
    std::vector<int32_t> s;
    int got = t.allocate_slots(count, s, kind);
    assert(got == count);
    auto r = t.insert(h.data(), n, s.data(), count, 0, true, kind);
    assert(r.error == InsertError::OK);
}

static void test_hold_returns_full_prefix() {
    TEST_BEGIN("hold: full prefix, held tail, pinned ancestors");
    const char* name = "/tewb_prefix";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, full_cfg(256, 10));
    auto h = seq(0x1000, 100);
    auto head = insert_fresh(t, h, 90);
    auto tail = insert_fresh(t, h, 100, 90);
    const uint32_t free0 = t.mempool_free();

    std::vector<HeldEviction> held;
    assert(t.try_evict_batch(10, &held) == 10);
    assert(held.size() == 1);
    const HeldEviction& e = held[0];
    assert(e.end() == 100 && e.evicted_from == 90);
    assert(e.hashes == h);
    assert(std::equal(head.begin(), head.end(), e.full_slots.begin()));
    assert(std::equal(tail.begin(), tail.end(), e.full_slots.begin() + 90));
    assert(e.swa_slots.empty() && e.mamba_ends.empty());
    assert(t.mempool_free() == free0);
    assert(t.held_slots(ComponentType::FULL) == 10);
    assert((int)t.query_components(h.data(), 100, COMPONENT_MASK_FULL, false, false)
               .topology_match_length == 90);

    // The pinned prefix is evicted neither in the background nor in the foreground.
    assert(t.try_evict_batch(100) == 0);
    std::vector<int32_t> s;
    assert(t.allocate_slots((int)t.mempool_free() + 1, s) == (int)free0);
    t.recycle_slots(s.data(), (int)s.size());

    t.release_held(held);
    assert(t.mempool_free() == free0 + 10);
    assert(t.held_slots(ComponentType::FULL) == 0);
    assert(t.try_evict_batch(100) == 90);
    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_foreground_never_holds() {
    TEST_BEGIN("hold: foreground eviction recycles at once");
    const char* name = "/tewb_fg";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, full_cfg(32, 16));
    auto a = seq(0x2000, 16), b = seq(0x3000, 16);
    insert_fresh(t, a, 16);
    insert_fresh(t, b, 16);
    assert(t.mempool_free() == 0);
    std::vector<int32_t> s;
    assert(t.allocate_slots(8, s) == 8);   // evicts one leaf in the foreground
    assert(t.held_slots(ComponentType::FULL) == 0);
    t.recycle_slots(s.data(), (int)s.size());
    assert(t.try_evict_batch(32) == 16);   // the other one, background without hold
    assert(t.mempool_free() == 32);
    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_split_of_pinned_ancestor() {
    TEST_BEGIN("hold: a split ancestor is unpinned on both halves");
    const char* name = "/tewb_split";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, full_cfg(256, 10));
    auto h = seq(0x4000, 100);
    insert_fresh(t, h, 90);
    insert_fresh(t, h, 100, 90);
    std::vector<HeldEviction> held;
    assert(t.try_evict_batch(10, &held) == 10 && held.size() == 1);

    auto d = h;
    d.resize(55);
    for (int i = 45; i < 55; i++) d[i] = 0x9000 + (uint64_t)i;
    insert_fresh(t, d, 55, 45);                    // splits [40, 50) at 45; new [45, 50), [50, 55)
    assert(t.try_evict_batch(100) == 10);          // only the new nodes: the halves are pinned

    t.release_held(held);
    expect_healthy(t);
    assert(t.try_evict_batch(100) == 90);
    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_budget_and_shared_prefix() {
    TEST_BEGIN("hold: budget stops a batch; siblings share prefix");
    const char* name = "/tewb_budget";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, full_cfg(256, 8));
    auto a = seq(0x5000, 12);
    auto b = a;
    b.resize(16);
    for (int i = 8; i < 16; i++) b[i] = 0xB000 + (uint64_t)i;
    insert_fresh(t, a, 8);
    insert_fresh(t, a, 12, 8);
    insert_fresh(t, b, 16, 8);

    std::vector<HeldEviction> held;
    assert(t.try_evict_batch(100, &held, /*budget=*/1) > 0);
    assert(held.size() == 1);                      // the first leaf already exceeded the budget
    assert(t.try_evict_batch(100, &held) > 0);
    assert(held.size() == 2);                      // the sibling; the pinned parent stays
    assert(held[0].evicted_from == 8 && held[1].evicted_from == 8);
    assert(std::equal(a.begin(), a.begin() + 8, held[1].hashes.begin()));
    assert(t.held_slots(ComponentType::FULL) == 12);
    assert(t.try_evict_batch(100) == 0);

    t.release_held(held);
    assert(t.held_slots(ComponentType::FULL) == 0);
    assert(t.try_evict_batch(100) == 8);
    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_aux_held_and_pinned() {
    TEST_BEGIN("hold: SWA lane + Mamba END held, aux pinned");
    const char* name = "/tewb_aux";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, aux_cfg());
    auto h = seq(0x6000, 8);
    insert_fresh(t, h, 8);                         // nodes [0, 4) and [4, 8)
    publish(t, h, 4, ComponentType::SWA, 4);
    publish(t, h, 4, ComponentType::MAMBA, 1);
    publish(t, h, 8, ComponentType::SWA, 4);
    publish(t, h, 8, ComponentType::MAMBA, 1);
    const uint32_t full0 = t.mempool_free(), swa0 = t.swa_mempool_free(),
                   mamba0 = t.mamba_mempool_free();

    std::vector<HeldEviction> held;
    assert(t.try_evict_batch(4, &held) == 4 && held.size() == 1);
    const HeldEviction& e = held[0];
    assert(e.end() == 8 && e.evicted_from == 4);
    assert(e.swa_slots.size() == 8);
    for (int32_t s : e.swa_slots) assert(s != INVALID_COMPONENT_SLOT);
    assert((e.mamba_ends == std::vector<uint32_t>{4, 8}) && e.mamba_slots.size() == 2);
    uint64_t c[NUM_COMPONENT_TYPES];
    e.covered(c);
    assert(c[0] == 8 && c[1] == 8 && c[2] == 2);
    assert(t.held_slots(ComponentType::FULL) == 4 && t.held_slots(ComponentType::SWA) == 4 &&
           t.held_slots(ComponentType::MAMBA) == 1);
    assert(t.mempool_free() == full0 && t.swa_mempool_free() == swa0 &&
           t.mamba_mempool_free() == mamba0);

    // The ancestor's SWA / Mamba slots are pinned too.
    assert(t.try_evict_component_batch(ComponentType::SWA, 100) == 0);
    assert(t.try_evict_component_batch(ComponentType::MAMBA, 100) == 0);
    assert(t.try_evict_batch(100) == 0);

    t.release_held(held);
    assert(t.mempool_free() == full0 + 4 && t.swa_mempool_free() == swa0 + 4 &&
           t.mamba_mempool_free() == mamba0 + 1);
    expect_healthy(t);
    assert(t.try_evict_component_batch(ComponentType::SWA, 100) == 4);
    assert(t.try_evict_component_batch(ComponentType::MAMBA, 100) == 1);
    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_release_after_reset() {
    TEST_BEGIN("hold: release from before reset() is a no-op");
    const char* name = "/tewb_reset";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, full_cfg(32, 8));
    auto h = seq(0x7000, 16);
    insert_fresh(t, h, 8);
    insert_fresh(t, h, 16, 8);
    std::vector<HeldEviction> held;
    assert(t.try_evict_batch(8, &held) == 8 && held.size() == 1);
    t.reset();
    assert(t.held_slots(ComponentType::FULL) == 0 && t.mempool_free() == 32);
    std::vector<int32_t> s;
    assert(t.allocate_slots(32, s) == 32);         // includes the old held ids
    t.release_held(held);
    assert(t.mempool_free() == 0);
    t.recycle_slots(s.data(), (int)s.size());
    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_queue() {
    TEST_BEGIN("queue: pop / complete / drop / stale / inflight");
    const char* name = "/tewb_queue";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, full_cfg(256, 8));
    auto a = seq(0x8000, 12);
    auto b = a;
    b.resize(16);
    for (int i = 8; i < 16; i++) b[i] = 0xC000 + (uint64_t)i;
    insert_fresh(t, a, 8);
    insert_fresh(t, a, 12, 8);
    insert_fresh(t, b, 16, 8);
    const uint32_t free0 = t.mempool_free();

    WritebackQueue q(t);
    assert(q.pop(4, 0).empty());
    std::vector<HeldEviction> held;
    assert(t.try_evict_batch(100, &held) == 12 && held.size() == 2);
    q.push(std::move(held));
    uint64_t in[NUM_COMPONENT_TYPES];
    q.inflight_slots(in);
    assert(in[0] == 12 + 16);                      // prefixes included

    auto first = q.pop(1, 0);
    assert(first.size() == 1);
    auto rest = q.pop(8, 0);
    assert(rest.size() == 1 && rest[0].id != first[0].id);
    assert(q.complete({first[0].id, 999}) == 1);
    assert(q.complete({first[0].id}) == 0);
    q.inflight_slots(in);
    assert(in[0] == rest[0].h.end());
    assert(t.mempool_free() == free0 + first[0].h.end() - first[0].h.evicted_from);
    q.drop_all();
    assert(t.mempool_free() == free0 + 12 && t.held_slots(ComponentType::FULL) == 0);
    WritebackStats st = q.stats();
    assert(st.produced == 2 && st.completed == 1 && st.dropped == 1 && st.inflight_slots[0] == 0);

    // A task from before reset() is released, not returned.
    insert_fresh(t, a, 8);
    insert_fresh(t, a, 12, 8);
    held.clear();
    assert(t.try_evict_batch(4, &held) == 4);
    q.push(std::move(held));
    t.reset();
    assert(q.pop(8, 0).empty());
    assert(q.stats().dropped == 2 && t.mempool_free() == 256);
    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

int main() {
    printf("test_evict_writeback\n");
    test_hold_returns_full_prefix();
    test_foreground_never_holds();
    test_split_of_pinned_ancestor();
    test_budget_and_shared_prefix();
    test_aux_held_and_pinned();
    test_release_after_reset();
    test_queue();
    return tests_report();
}
