// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// test_tree_evict — capacity management & strict LRU.

#include "test_utils.h"
#include "radix_client.h"

#include <cassert>
#include <vector>

using namespace radixshmem;

// ==================================================================
static void test_ref_count_protects_from_eviction() {
    TEST_BEGIN("ref_count > 0 blocks eviction");

    ShmConfig cfg{1000, 4};
    ServerLauncher launcher("/test_te_01", cfg);
    RadixClient client("/test_te_01");

    int64_t t1[] = {10, 20};
    int64_t t2[] = {30, 40};
    auto h1 = RadixClient::compute_hashes(t1, 2, 1);
    auto h2 = RadixClient::compute_hashes(t2, 2, 1);
    insert_auto(client, h1.data(), 2);
    insert_auto(client, h2.data(), 2);
    auto pin_h1 = pin_path(client, h1.data(), 2);

    int free_after = force_evict(client, 2);
    assert(free_after >= 2);
    assert(client.query(h1.data(), 2, COMPONENT_MASK_FULL).common_hit == 2);
    assert(client.query(h2.data(), 2, COMPONENT_MASK_FULL).common_hit == 0);

    pin_h1();
    free_after = force_evict(client, 4);
    assert(free_after >= 4);
    assert(client.query(h1.data(), 2, COMPONENT_MASK_FULL).common_hit == 0);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
static void test_data_pool_utilization_reported() {
    TEST_BEGIN("data_pool_utilization tracks fill level");

    ServerLauncher launcher("/test_te_02", default_cfg());
    RadixClient client("/test_te_02");

    float u0 = client.data_pool_utilization();
    assert(u0 < 0.001f);

    int64_t t[] = {1, 2, 3};
    auto h = RadixClient::compute_hashes(t, 3, 1);
    insert_auto(client, h.data(), 3);
    assert(client.data_pool_utilization() > 0.0f);

    force_evict(client, (int)client.mempool_total());
    assert(client.data_pool_utilization() < 0.001f);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
static void test_strict_lru_query_updates() {
    TEST_BEGIN("strict LRU: query (default update_meta) reorders leaf");

    ShmConfig cfg{1000, 4};
    ServerLauncher launcher("/test_te_03", cfg);
    RadixClient client("/test_te_03");

    int64_t ta = 10, tb = 20, tc = 30, td = 40;
    auto ha = RadixClient::compute_hashes(&ta, 1, 1);
    auto hb = RadixClient::compute_hashes(&tb, 1, 1);
    auto hc = RadixClient::compute_hashes(&tc, 1, 1);
    auto hd = RadixClient::compute_hashes(&td, 1, 1);
    insert_auto(client, ha.data(), 1);
    insert_auto(client, hb.data(), 1);
    insert_auto(client, hc.data(), 1);
    insert_auto(client, hd.data(), 1);
    assert(client.mempool_free() == 0);

    // Touch a → leaf order [b, c, d, a]
    client.query(ha.data(), 1, COMPONENT_MASK_FULL);
    force_evict(client, 1);
    // b should have been evicted (now LRU head).
    assert(client.query(ha.data(), 1, COMPONENT_MASK_FULL, false, false, false).common_hit == 1);
    assert(client.query(hb.data(), 1, COMPONENT_MASK_FULL, false, false, false).common_hit == 0);
    assert(client.query(hc.data(), 1, COMPONENT_MASK_FULL, false, false, false).common_hit == 1);
    assert(client.query(hd.data(), 1, COMPONENT_MASK_FULL, false, false, false).common_hit == 1);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
static void test_strict_lru_insert_updates() {
    TEST_BEGIN("strict LRU: insert always reorders leaf");

    ShmConfig cfg{1000, 3};
    ServerLauncher launcher("/test_te_04", cfg);
    RadixClient client("/test_te_04");

    int64_t ta = 5, tb = 6, tc = 7;
    auto ha = RadixClient::compute_hashes(&ta, 1, 1);
    auto hb = RadixClient::compute_hashes(&tb, 1, 1);
    auto hc = RadixClient::compute_hashes(&tc, 1, 1);
    insert_auto(client, ha.data(), 1);
    insert_auto(client, hb.data(), 1);
    insert_auto(client, hc.data(), 1);

    // Re-insert a (full-match path) — promotes a to MRU.
    insert_auto(client, ha.data(), 1);
    force_evict(client, 1);
    assert(client.query(ha.data(), 1, COMPONENT_MASK_FULL, false, false, false).common_hit == 1);
    assert(client.query(hb.data(), 1, COMPONENT_MASK_FULL, false, false, false).common_hit == 0);
    assert(client.query(hc.data(), 1, COMPONENT_MASK_FULL, false, false, false).common_hit == 1);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
static void test_query_update_meta_false_skips_lru() {
    TEST_BEGIN("query(update_meta=false) leaves LRU untouched");

    ShmConfig cfg{1000, 3};
    ServerLauncher launcher("/test_te_05", cfg);
    RadixClient client("/test_te_05");

    int64_t ta = 1, tb = 2, tc = 3;
    auto ha = RadixClient::compute_hashes(&ta, 1, 1);
    auto hb = RadixClient::compute_hashes(&tb, 1, 1);
    auto hc = RadixClient::compute_hashes(&tc, 1, 1);
    insert_auto(client, ha.data(), 1);
    insert_auto(client, hb.data(), 1);
    insert_auto(client, hc.data(), 1);

    // Peek a (does NOT touch LRU) → a still LRU head, gets evicted.
    client.query(ha.data(), 1, COMPONENT_MASK_FULL, false, false,
                 /*update_meta=*/false);
    force_evict(client, 1);
    assert(client.query(ha.data(), 1, COMPONENT_MASK_FULL, false, false, false).common_hit == 0);
    assert(client.query(hb.data(), 1, COMPONENT_MASK_FULL, false, false, false).common_hit == 1);
    assert(client.query(hc.data(), 1, COMPONENT_MASK_FULL, false, false, false).common_hit == 1);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
static void test_evict_long_chain_single_pass() {
    TEST_BEGIN("evict_walk drains a multi-node leaf chain node-by-node");

    // cs=1 makes every block its own radix node, so [1..N] is a real N-node
    // chain (+root), not one fat node. This actually exercises evict_walk's
    // node-by-node link removal — with the default cs=128 the whole sequence
    // would collapse into a single node and only test one evict_node() call.
    constexpr int N = 64;
    ShmConfig cfg;
    cfg.max_nodes = 1000;
    cfg.max_blocks = N;
    cfg.register_chunk_tokens = 1;
    ServerLauncher launcher("/test_te_06", cfg);
    RadixClient client("/test_te_06");

    std::vector<int64_t> tokens(N);
    for (int i = 0; i < N; i++) tokens[i] = i + 1;
    auto h = RadixClient::compute_hashes(tokens.data(), N, 1);
    auto ins = insert_auto(client, h.data(), N);
    assert(ins.error == InsertError::OK);
    assert((int)client.mempool_used() == N);
    // N data nodes + root.
    assert((int)client.total_radix_nodes() == N + 1);

    // A single force_evict of N drains the whole chain: evict_walk removes the
    // deepest leaf, its parent becomes a leaf, and so on up the chain.
    std::vector<int32_t> tmp;
    client.allocate_slots(N, tmp);
    assert((int)tmp.size() == N);
    client.recycle_slots(tmp.data(), (int)tmp.size());

    assert(client.query(h.data(), N, COMPONENT_MASK_FULL, false, false, false).common_hit == 0);
    // Every chain node gone; only root remains.
    assert((int)client.total_radix_nodes() == 1);
    assert(client.total_blocks() == 0);
    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
static void test_auto_evict_on_insert() {
    TEST_BEGIN("insert into full mempool triggers auto-evict");

    ShmConfig cfg{1000, 10};
    ServerLauncher launcher("/test_te_07", cfg);
    RadixClient client("/test_te_07");

    int64_t t1[] = {1, 2, 3, 4, 5};
    int64_t t2[] = {6, 7, 8, 9, 10};
    auto h1 = RadixClient::compute_hashes(t1, 5, 1);
    auto h2 = RadixClient::compute_hashes(t2, 5, 1);
    insert_auto(client, h1.data(), 5);
    insert_auto(client, h2.data(), 5);
    assert(client.mempool_free() == 0);

    // Touch h2 (MRU) — h1 is LRU.
    client.query(h2.data(), 5, COMPONENT_MASK_FULL);

    int64_t t3[] = {11, 12, 13};
    auto h3 = RadixClient::compute_hashes(t3, 3, 1);
    auto ins3 = insert_auto(client, h3.data(), 3);  // forces eviction of h1
    assert(ins3.error == InsertError::OK);
    assert(ins3.unused_slots.empty());   // all 3 shortfall slots landed

    assert(client.query(h1.data(), 5, COMPONENT_MASK_FULL, false, false, false).common_hit == 0);
    assert(client.query(h2.data(), 5, COMPONENT_MASK_FULL, false, false, false).common_hit == 5);
    assert(client.query(h3.data(), 3, COMPONENT_MASK_FULL, false, false, false).common_hit == 3);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
static void test_mempool_full_allocate_shortfall() {
    TEST_BEGIN("mempool full → allocate_slots returns shortfall; insert fits what it gets");

    // 4 slots total. Pin two prefixes (4 blocks) so eviction can't help.
    // Slot allocation and tree registration are decoupled now: "full" surfaces
    // at allocate_slots (fewer slots than asked), not as an insert error.
    ShmConfig cfg{1000, 4};
    ServerLauncher launcher("/test_te_08", cfg);
    RadixClient client("/test_te_08");

    int64_t t1[] = {1, 2};
    int64_t t2[] = {3, 4};
    auto h1 = RadixClient::compute_hashes(t1, 2, 1);
    auto h2 = RadixClient::compute_hashes(t2, 2, 1);
    insert_pinned(client, h1.data(), 2);  // pin against eviction
    insert_pinned(client, h2.data(), 2);
    assert(client.mempool_free() == 0);

    // Request 4 new slots — none free, evict can't reclaim the pinned ones.
    std::vector<int32_t> slots;
    int got = client.allocate_slots(4, slots);
    assert(got == 0);                 // nothing could be allocated
    assert(slots.empty());

    // Inserting with the (empty) slot pool registers nothing, no error.
    int64_t t3[] = {100, 101, 102, 103};
    auto h3 = RadixClient::compute_hashes(t3, 4, 1);
    auto ins3 = client.insert(h3.data(), 4, slots.data(), (int)slots.size());
    assert(ins3.error == InsertError::OK);
    assert(ins3.unused_slots.empty());   // no slots given → nothing to recycle
    expect_healthy(client, HEALTH_REF_LEAK);   // insert_pinned holds its pins for the test's lifetime
    TEST_PASS();
}

// ==================================================================
// Parent relay on eviction — the REAL mechanism.
//
// evict_walk() (radix_tree.cpp) always restarts from leaf_list_head and
// evicts the first unpinned leaf. There is no cur/next comparison. Relay
// happens indirectly: when evict_node() frees a leaf whose parent then has
// children_count==0, post_evict_fixup() calls leaf_list_insert_by_access_time()
// to splice the now-leaf parent back into leaf_list at its access_time slot.
// A split parent P inherits the pre-split node's access_time (split_node:
// prefix_node.last_access_time = orig_at), so P is typically OLD.
//
// leaf_list_insert_by_access_time() has two insertion paths, and these two
// tests drive one each:
//   (a) HEAD insert (radix_tree.cpp:856): P older than every current leaf →
//       spliced at head → next evict_walk pass relay-evicts P immediately.
//   (b) MID/TAIL insert (radix_tree.cpp:843): an older leaf still in the list →
//       P spliced AFTER it → P survives this round as a leaf.
//
// We use cs=1 (one block per node) and max_blocks pinned to the exact working
// set so allocate_slots evicts a controlled number of blocks.
// ==================================================================
static void test_evict_relay_parent_head_insert() {
    TEST_BEGIN("relay: only-children evicted → parent head-inserted → relay-evicted");

    // Topology: root → A[10] → {B[11], C[12]}, plus independent Z[99].
    // A inherits the oldest access_time (created first, before Z). 4 blocks
    // total (99,10,11,12); pin the pool to 4 so eviction is exact.
    ShmConfig cfg;
    cfg.max_nodes = 1000;
    cfg.max_blocks = 4;
    cfg.register_chunk_tokens = 1;
    cfg.background_evict_ratio = 0.0f;
    ServerLauncher launcher("/test_te_lru_a", cfg);
    RadixClient client("/test_te_lru_a");

    int64_t t_ab[] = {10, 11};
    int64_t t_ac[] = {10, 12};
    int64_t t_z[]  = {99};
    auto h_ab = RadixClient::compute_hashes(t_ab, 2, 1);
    auto h_ac = RadixClient::compute_hashes(t_ac, 2, 1);
    auto h_z  = RadixClient::compute_hashes(t_z, 1, 1);
    insert_auto(client, h_ab.data(), 2);   // 10 → 11
    insert_auto(client, h_ac.data(), 2);   // split at 10: A[10] → {B[11], C[12]}
    insert_auto(client, h_z.data(), 1);    // Z newer than A

    // Touch B, C, Z (Z last = MRU). A stays internal and keeps its old
    // inherited access_time — older than Z.
    client.query(h_ab.data(), 2, COMPONENT_MASK_FULL);
    client.query(h_ac.data(), 2, COMPONENT_MASK_FULL);
    client.query(h_z.data(), 1, COMPONENT_MASK_FULL);

    // Evict 3 blocks: B(1) + C(1) → A becomes leaf, head-inserted (oldest) →
    // 3rd pass relay-evicts A[10]. Z (MRU) survives.
    std::vector<int32_t> tmp;
    client.allocate_slots(3, tmp);
    if (!tmp.empty()) client.recycle_slots(tmp.data(), (int)tmp.size());

    // Whole [10] path gone — A was relay-evicted, not just its children.
    assert(client.query(h_ab.data(), 2, COMPONENT_MASK_FULL, false, false, false).common_hit == 0);
    assert(client.query(h_ac.data(), 2, COMPONENT_MASK_FULL, false, false, false).common_hit == 0);
    // Z survives.
    assert(client.query(h_z.data(), 1, COMPONENT_MASK_FULL, false, false, false).common_hit == 1);
    // Only root + Z remain.
    assert((int)client.total_radix_nodes() == 2);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
static void test_evict_relay_parent_mid_insert() {
    TEST_BEGIN("relay: parent mid-inserted after an older pinned leaf, survives");

    // Same A[10] → {B[11], C[12]} split, but now an OLDER leaf Z[99] is
    // pinned so it stays in leaf_list without being evicted. When A becomes
    // a leaf it is spliced AFTER Z (leaf_list_insert_by_access_time hits the
    // cn->at <= t branch), so once B+C are gone A survives this round.
    ShmConfig cfg;
    cfg.max_nodes = 1000;
    cfg.max_blocks = 4;
    cfg.register_chunk_tokens = 1;
    cfg.background_evict_ratio = 0.0f;
    ServerLauncher launcher("/test_te_lru_b", cfg);
    RadixClient client("/test_te_lru_b");

    int64_t t_z[]  = {99};
    int64_t t_ab[] = {10, 11};
    int64_t t_ac[] = {10, 12};
    auto h_z  = RadixClient::compute_hashes(t_z, 1, 1);
    auto h_ab = RadixClient::compute_hashes(t_ab, 2, 1);
    auto h_ac = RadixClient::compute_hashes(t_ac, 2, 1);
    insert_auto(client, h_z.data(), 1);    // Z oldest
    insert_auto(client, h_ab.data(), 2);   // 10 → 11
    insert_auto(client, h_ac.data(), 2);   // split: A[10] → {B[11], C[12]}

    auto pin_z = pin_path(client, h_z.data(), 1);   // Z stays in list, never evicted
    client.query(h_z.data(), 1, COMPONENT_MASK_FULL);
    client.query(h_ab.data(), 2, COMPONENT_MASK_FULL);
    client.query(h_ac.data(), 2, COMPONENT_MASK_FULL);

    // Evict exactly 2 blocks: B(1) + C(1). Z is pinned/skipped. After C, A
    // becomes a leaf and is spliced after (older) Z → not evicted this round.
    std::vector<int32_t> tmp;
    client.allocate_slots(2, tmp);
    if (!tmp.empty()) client.recycle_slots(tmp.data(), (int)tmp.size());

    // B and C gone; the shared parent A[10] survives as a leaf.
    assert(client.query(h_ab.data(), 2, COMPONENT_MASK_FULL, false, false, false).common_hit == 1);
    assert(client.query(h_ac.data(), 2, COMPONENT_MASK_FULL, false, false, false).common_hit == 1);
    // Pinned Z survives.
    assert(client.query(h_z.data(), 1, COMPONENT_MASK_FULL, false, false, false).common_hit == 1);
    // root + Z + A remain.
    assert((int)client.total_radix_nodes() == 3);

    pin_z();
    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
// evict_walk early-outs: an empty tree and a need<=0 request both free
// nothing and must not crash (evict_walk's first pass finds no victim).
// ==================================================================
static void test_evict_walk_noop_paths() {
    TEST_BEGIN("evict on empty tree / need<=0 is a safe no-op");

    ShmConfig cfg;
    cfg.max_nodes = 1000;
    cfg.max_blocks = 10;
    cfg.register_chunk_tokens = 1;
    ServerLauncher launcher("/test_te_noop", cfg);
    RadixClient client("/test_te_noop");

    // Empty tree — nothing to evict.
    int f = force_evict(client, 5);
    assert(f == (int)client.mempool_total());   // all still free
    assert((int)client.total_radix_nodes() == 1);  // just root

    // Populate one leaf, then a zero-slot allocation: allocate_slots(0) is an
    // invalid arg (returns -1, evicts nothing) — the populated leaf is untouched.
    int64_t t = 7;
    auto h = RadixClient::compute_hashes(&t, 1, 1);
    insert_auto(client, h.data(), 1);
    int before = (int)client.mempool_free();
    std::vector<int32_t> tmp;
    assert(client.allocate_slots(0, tmp) == -1);
    assert(tmp.empty());
    assert((int)client.mempool_free() == before);   // nothing evicted
    assert(client.query(h.data(), 1, COMPONENT_MASK_FULL, false, false, false).common_hit == 1);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
// Mid-insert eviction protects the parent being built. Inserting a chain
// longer than the free pool triggers evict() partway through; the in-progress
// parent is inc_ref'd (radix_tree.cpp:1292/1306 evict(chunk, parent_id)) so it
// can't evict the very prefix it is extending. The whole chain must land.
// ==================================================================
static void test_insert_evict_protects_building_parent() {
    TEST_BEGIN("insert longer than pool: mid-insert evict spares the building parent");

    ShmConfig cfg;
    cfg.max_nodes = 1000;
    cfg.max_blocks = 6;
    cfg.register_chunk_tokens = 1;
    cfg.background_evict_ratio = 0.0f;
    ServerLauncher launcher("/test_te_protect", cfg);
    RadixClient client("/test_te_protect");

    // Fill the pool with 6 unrelated single-block leaves.
    for (int i = 0; i < 6; i++) {
        int64_t t = 500 + i;
        auto h = RadixClient::compute_hashes(&t, 1, 1);
        insert_auto(client, h.data(), 1);
    }
    assert(client.mempool_free() == 0);

    // Insert a fresh 6-block chain: needs all 6 slots, so eviction fires mid-
    // insert. If the building parent weren't protected the chain would eat its
    // own prefix and land short. It must land fully.
    std::vector<int64_t> chain = {1, 2, 3, 4, 5, 6};
    auto hc = RadixClient::compute_hashes(chain.data(), 6, 1);
    auto r = insert_auto(client, hc.data(), 6);
    assert(r.error == InsertError::OK);
    assert(client.query(hc.data(), 6, COMPONENT_MASK_FULL, false, false, false).common_hit == 6);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
int main() {
    printf("=== test_tree_evict ===\n\n");
    test_ref_count_protects_from_eviction();
    test_data_pool_utilization_reported();
    test_strict_lru_query_updates();
    test_strict_lru_insert_updates();
    test_query_update_meta_false_skips_lru();
    test_evict_long_chain_single_pass();
    test_auto_evict_on_insert();
    test_mempool_full_allocate_shortfall();
    test_evict_relay_parent_head_insert();
    test_evict_relay_parent_mid_insert();
    test_evict_walk_noop_paths();
    test_insert_evict_protects_building_parent();
    return tests_report();
}
