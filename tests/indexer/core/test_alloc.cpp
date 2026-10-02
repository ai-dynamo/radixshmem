// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// test_alloc — slot allocator contract: allocate_slots / recycle_slots.
//
// Pure allocator view (no tree structure) — allocate is the sole eviction
// point. Asserts the CURRENT behavior:
//   - basic allocate: distinct slots, mempool used/free move by N
//   - recycle restores free count and slots are reusable
//   - num_slots <= 0 returns 0 with empty out (no error signalled)
//   - request beyond capacity truncates to what's free (return == capacity),
//     no error signalled
//   - allocate past a full pool evicts LRU and still returns fresh slots
//
//   - bad args (num<=0 or num>capacity) return -1, pool untouched
//   - repeat recycle of an already-free slot is a silent no-op (dedup)

#include "test_utils.h"
#include "shmradix/radix_client.h"

#include <cassert>
#include <set>
#include <vector>

using namespace shmradix;

// ==================================================================
static void test_allocate_basic() {
    TEST_BEGIN("allocate_slots: N distinct slots, pool counters move by N");

    ServerLauncher launcher("/test_al_01", default_cfg());
    RadixClient client("/test_al_01");

    const uint32_t free0 = client.mempool_free();
    assert(client.mempool_used() == 0);

    std::vector<int32_t> slots;
    int got = client.allocate_slots(8, slots);
    assert(got == 8);
    assert((int)slots.size() == 8);

    // Slots are distinct physical block ids.
    std::set<int32_t> uniq(slots.begin(), slots.end());
    assert((int)uniq.size() == 8);

    assert(client.mempool_used() == 8);
    assert(client.mempool_free() == free0 - 8);

    client.recycle_slots(slots.data(), got);   // nothing left behind
    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
static void test_recycle_restores() {
    TEST_BEGIN("recycle_slots: free count restored, slots reusable");

    ServerLauncher launcher("/test_al_02", default_cfg());
    RadixClient client("/test_al_02");

    const uint32_t free0 = client.mempool_free();

    std::vector<int32_t> slots;
    int got = client.allocate_slots(5, slots);
    assert(got == 5);
    assert(client.mempool_free() == free0 - 5);

    client.recycle_slots(slots.data(), got);
    assert(client.mempool_free() == free0);
    assert(client.mempool_used() == 0);

    // The pool is whole again — a fresh allocation of the same size succeeds.
    std::vector<int32_t> slots2;
    assert(client.allocate_slots(5, slots2) == 5);
    client.recycle_slots(slots2.data(), 5);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
static void test_allocate_bad_args() {
    TEST_BEGIN("allocate_slots: num<=0 or num>capacity returns -1, pool untouched");

    ShmConfig cfg{1000, 4};
    ServerLauncher launcher("/test_al_03", cfg);
    RadixClient client("/test_al_03");

    const uint32_t free0 = client.mempool_free();
    const uint32_t total = client.mempool_total();
    assert(total == 4);

    std::vector<int32_t> slots;
    assert(client.allocate_slots(0, slots) == -1);
    assert(slots.empty());
    assert(client.allocate_slots(-3, slots) == -1);
    assert(slots.empty());

    // Beyond pool capacity is an error, not a truncated allocation.
    assert(client.allocate_slots((int)total + 1, slots) == -1);
    assert(slots.empty());

    // A request of exactly capacity is still valid.
    assert(client.allocate_slots((int)total, slots) == (int)total);
    client.recycle_slots(slots.data(), (int)slots.size());

    // Pool untouched after the failed calls.
    assert(client.mempool_free() == free0);
    assert(client.mempool_used() == 0);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
static void test_recycle_dedup() {
    TEST_BEGIN("recycle_slots: repeat recycle of a free slot is a no-op");

    ServerLauncher launcher("/test_al_04", default_cfg());
    RadixClient client("/test_al_04");

    const uint32_t free0 = client.mempool_free();

    std::vector<int32_t> slots;
    int got = client.allocate_slots(5, slots);
    assert(got == 5);

    // First recycle returns all 5 to the pool.
    client.recycle_slots(slots.data(), got);
    assert(client.mempool_free() == free0);

    // Recycling the same slots again must NOT grow the free count past total
    // (would overflow the free-stack without dedup).
    client.recycle_slots(slots.data(), got);
    assert(client.mempool_free() == free0);
    assert(client.mempool_free() <= client.mempool_total());

    // Pool still consistent — a full-capacity allocation succeeds afterwards.
    std::vector<int32_t> all;
    assert(client.allocate_slots((int)client.mempool_total(), all)
           == (int)client.mempool_total());
    client.recycle_slots(all.data(), (int)all.size());

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
static void test_allocate_evicts_lru() {
    TEST_BEGIN("allocate_slots: full pool evicts LRU, returns fresh slots");

    // 4-block pool. Fill it via an insert (leaves evictable, ref==0), then
    // allocate again — the sole eviction point must reclaim LRU and succeed.
    ShmConfig cfg{1000, 4};
    cfg.background_evict_ratio = 0.0f;
    cfg.register_chunk_tokens = 1;
    ServerLauncher launcher("/test_al_05", cfg);
    RadixClient client("/test_al_05");

    int64_t t[] = {1, 2, 3, 4};
    auto h = RadixClient::compute_hashes(t, 4, 1);
    auto ins = insert_auto(client, h.data(), 4);
    assert(ins.unused_slots.empty());   // all 4 landed
    assert(client.mempool_free() == 0);   // pool full, nothing pinned

    // No free blocks left; allocate must evict LRU to satisfy the request.
    std::vector<int32_t> slots;
    int got = client.allocate_slots(2, slots);
    assert(got == 2);
    assert((int)slots.size() == 2);

    // Return the evicted-and-reissued slots; the 2 surviving blocks stay in the tree.
    client.recycle_slots(slots.data(), got);
    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
int main() {
    printf("=== test_alloc ===\n\n");
    test_allocate_basic();
    test_recycle_restores();
    test_allocate_bad_args();
    test_recycle_dedup();
    test_allocate_evicts_lru();
    return tests_report();
}
