// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// test_tree_state — ref / FinalizeFn semantics, including split-invariance
// (ref_count survives splits caused by later inserts).

#include "test_utils.h"
#include "radix_client.h"

#include <cassert>
#include <vector>

using namespace radixshmem;

// ==================================================================
static void test_inc_dec_ref_path() {
    TEST_BEGIN("inc_ref / dec_ref hash-path semantics");

    ShmConfig cfg{1000, 4};
    ServerLauncher launcher("/test_ts_04", cfg);
    RadixClient client("/test_ts_04");

    int64_t t1[] = {10, 20};
    int64_t t2[] = {30, 40};
    auto h1 = RadixClient::compute_hashes(t1, 2, 1);
    auto h2 = RadixClient::compute_hashes(t2, 2, 1);
    insert_auto(client, h1.data(), 2);
    insert_auto(client, h2.data(), 2);

    // Lock h1, force-evict capacity → h1 must stay, h2 must go.
    auto pin_h1 = pin_path(client, h1.data(), 2);
    force_evict(client, (int)client.mempool_total());
    assert(client.query(h1.data(), 2, COMPONENT_MASK_FULL).common_hit == 2);
    assert(client.query(h2.data(), 2, COMPONENT_MASK_FULL).common_hit == 0);

    // Unlock — now h1 is also evictable.
    pin_h1();
    force_evict(client, (int)client.mempool_total());
    assert(client.query(h1.data(), 2, COMPONENT_MASK_FULL).common_hit == 0);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
static void test_finalize_is_one_shot() {
    TEST_BEGIN("finalize is one-shot: a repeat call is a no-op (ref never negative)");

    ServerLauncher launcher("/test_ts_05", default_cfg());
    RadixClient client("/test_ts_05");

    int64_t t[] = {1, 2, 3};
    auto h = RadixClient::compute_hashes(t, 3, 1);
    insert_auto(client, h.data(), 3);

    // Pin once, release twice — the second call must not touch the ref.
    auto pin = pin_path(client, h.data(), 3);
    pin();
    pin();
    assert(!pin);
    // Back at ref==0: the path is evictable.
    force_evict(client, (int)client.mempool_total());
    assert(client.total_blocks() == 0);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
static void test_inc_ref_survives_split() {
    TEST_BEGIN("inc_ref then later split: dec_ref(hashes) clears all parts");

    ShmConfig cfg{1000, 7};
    ServerLauncher launcher("/test_ts_06", cfg);
    RadixClient client("/test_ts_06");

    int64_t ta[] = {1, 2, 3, 4, 5};
    auto ha = RadixClient::compute_hashes(ta, 5, 1);
    insert_auto(client, ha.data(), 5);
    auto pin_ha = pin_path(client, ha.data(), 5);

    int64_t tb[] = {1, 2, 3, 6, 7};
    auto hb = RadixClient::compute_hashes(tb, 5, 1);
    insert_auto(client, hb.data(), 5);  // triggers split

    force_evict(client, (int)client.mempool_total());
    assert(client.query(ha.data(), 5, COMPONENT_MASK_FULL).common_hit == 5);

    pin_ha();
    force_evict(client, (int)client.mempool_total());
    assert(client.query(ha.data(), 5, COMPONENT_MASK_FULL).common_hit == 0);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
static void test_split_inherits_ref() {
    TEST_BEGIN("split copies ref_count to both parts");

    ShmConfig cfg{1000, 7};
    ServerLauncher launcher("/test_ts_07", cfg);
    RadixClient client("/test_ts_07");

    int64_t ta[] = {1, 2, 3, 4, 5};
    auto ha = RadixClient::compute_hashes(ta, 5, 1);
    insert_pinned(client, ha.data(), 5);  // pin the whole prefix

    int64_t tb[] = {1, 2, 3, 6, 7};
    auto hb = RadixClient::compute_hashes(tb, 5, 1);
    insert_auto(client, hb.data(), 5);  // split at pos 3

    // After split both halves of [1..5] should remain ref-locked.
    auto r = client.query(ha.data(), 5, COMPONENT_MASK_FULL);
    assert(r.common_hit == 5);

    force_evict(client, (int)client.mempool_total());
    // [6,7] (ref=0) evictable; [1..3] + [4..5] still pinned.
    assert(client.query(ha.data(), 5, COMPONENT_MASK_FULL).common_hit == 5);

    expect_healthy(client, HEALTH_REF_LEAK);   // insert_pinned holds its pins for the test's lifetime
    TEST_PASS();
}

// ==================================================================
static void test_insert_pinned_protects_leaf() {
    TEST_BEGIN("insert + inc_ref protects new leaf from eviction");

    ShmConfig cfg{1000, 6};
    ServerLauncher launcher("/test_ts_08", cfg);
    RadixClient client("/test_ts_08");

    int64_t t1[] = {1, 2, 3};
    int64_t t2[] = {4, 5, 6};
    auto h1 = RadixClient::compute_hashes(t1, 3, 1);
    auto h2 = RadixClient::compute_hashes(t2, 3, 1);
    insert_auto(client, h1.data(), 3);
    auto pin_h1 = pin_path(client, h1.data(), 3);
    insert_auto(client, h2.data(), 3);

    force_evict(client, 3);  // capacity for 3 more — must take h2, not h1
    assert(client.query(h1.data(), 3, COMPONENT_MASK_FULL).common_hit == 3);
    assert(client.query(h2.data(), 3, COMPONENT_MASK_FULL).common_hit == 0);

    pin_h1();
    force_evict(client, 6);
    assert(client.query(h1.data(), 3, COMPONENT_MASK_FULL).common_hit == 0);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
static void test_finalize_query_lock_and_noop() {
    TEST_BEGIN("FinalizeFn covers query(lock=true) and is no-op for lock=false");

    ShmConfig cfg{1000, 6};
    ServerLauncher launcher("/test_ts_11", cfg);
    RadixClient client("/test_ts_11");

    int64_t t1[] = {1, 2, 3};
    int64_t t2[] = {4, 5, 6};
    auto h1 = RadixClient::compute_hashes(t1, 3, 1);
    auto h2 = RadixClient::compute_hashes(t2, 3, 1);
    insert_auto(client, h1.data(), 3);
    insert_auto(client, h2.data(), 3);

    // lock=false → finalize is empty.
    auto qr_plain = client.query(h1.data(), 3, COMPONENT_MASK_FULL);
    assert(!(bool)qr_plain.finalize);
    qr_plain.finalize();  // safe no-op

    // lock=true → finalize releases the ref.
    auto qr = client.query(h1.data(), 3, COMPONENT_MASK_FULL,
                            /*local_only=*/false, /*lock=*/true);
    assert((bool)qr.finalize);

    // h1 is locked; evict 3 must hit h2.
    force_evict(client, 3);
    assert(client.query(h1.data(), 3, COMPONENT_MASK_FULL).common_hit == 3);
    assert(client.query(h2.data(), 3, COMPONENT_MASK_FULL).common_hit == 0);

    qr.finalize();
    force_evict(client, (int)client.mempool_total());
    assert(client.query(h1.data(), 3, COMPONENT_MASK_FULL).common_hit == 0);

    // Idempotent.
    qr.finalize();
    assert(!(bool)qr.finalize);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
int main() {
    printf("=== test_tree_state ===\n\n");
    test_inc_dec_ref_path();
    test_finalize_is_one_shot();
    test_inc_ref_survives_split();
    test_split_inherits_ref();
    test_insert_pinned_protects_leaf();
    test_finalize_query_lock_and_noop();
    return tests_report();
}
