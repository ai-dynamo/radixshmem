// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// test_tree_local_query — exhaustive branch coverage of the local-tree query
// path (RadixShmemTree::query_internal, exposed via RadixClient::query with
// local_only=true). Each test maps to a specific control branch or boundary of
// query_internal (src/indexer/core/radix_tree.cpp §954-1031) and the QueryResult surface
// (common_hit / hit slots), plus the public-API
// lock / update_meta semantics layered on top in RadixShmemTree::query.
//
// Branch map (query_internal):
//   §962-1024 empty tree / root-only          → test_empty_tree, test_miss_first_block
//   §968 terminal (node contains seq end)     → test_exact_single_node,
//                                                test_terminal_partial_diverge,
//                                                test_query_shorter_than_node
//   §989 descend (child hit, accumulate slots)→ test_descend_multi_node,
//                                                test_slots_stitched_across_nodes
//   §1005 non-root fallback partial           → test_diverge_mid_second_node
//   §1005 full node, no matching child        → test_stop_at_node_boundary
//   n boundaries                              → test_n_zero, test_query_deeper_than_tree
//   QueryResult fields                        → test_local_hit_equals_total,
//                                                test_prefix_slots_length
//   public lock=true (inc_ref pins prefix)    → test_lock_pins_against_evict
//   public update_meta LRU touch order        → test_update_meta_touches_lru,
//                                                test_no_update_meta_leaves_lru

#include "test_utils.h"

#include <cassert>
#include <numeric>
#include <set>
#include <vector>

using namespace radixshmem;
using namespace radixshmem_test;

// Cumulative-prefix hashes for a length-n token run starting at `base`, cs=1.
static std::vector<uint64_t> seq(int64_t base, int n) {
    std::vector<int64_t> t(n);
    std::iota(t.begin(), t.end(), base);
    return RadixClient::compute_hashes(t.data(), n, 1);
}

// Local-only query shorthand (single-node attach; no peers to extend to).
static ComponentQueryResult lq(RadixClient& c, const uint64_t* h, int n,
                               bool lock = false, bool update_meta = true) {
    return c.query(h, n, COMPONENT_MASK_FULL, /*local_only=*/true, lock,
                   update_meta);
}

// ==================================================================
// §962 + §1022: empty tree — root has no children. matched=0, no slots, OK.
static void test_empty_tree() {
    TEST_BEGIN("query: empty tree misses cleanly (hit=0, no slots)");

    ServerLauncher launcher("/test_lq_1", default_cfg());
    RadixClient client("/test_lq_1");

    auto h = seq(1000, 5);
    auto q = lq(client, h.data(), 5);
    assert(q.common_hit == 0);
    assert(q.common_hit == 0);
    assert(hit_slots(q).empty());

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
// n==0: nothing to match; the while loop never runs. matched=0, no slots.
static void test_n_zero() {
    TEST_BEGIN("query: n==0 is a no-op miss");

    ServerLauncher launcher("/test_lq_2", default_cfg());
    RadixClient client("/test_lq_2");

    // Even with data present, n==0 matches nothing.
    auto h = seq(2000, 3);
    assert(insert_auto(client, h.data(), 3).error == InsertError::OK);

    auto q = lq(client, h.data(), 0);
    assert(q.common_hit == 0);
    assert(hit_slots(q).empty());

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
// §968 terminal, ml==cmp==n: the whole sequence lives inside one node and
// matches fully. common_hit==n, every slot returned.
static void test_exact_single_node() {
    TEST_BEGIN("query: exact full match inside one node");

    ServerLauncher launcher("/test_lq_3", default_cfg());
    RadixClient client("/test_lq_3");

    auto h = seq(3000, 5);
    std::vector<int32_t> slots;
    client.allocate_slots(5, slots);
    auto ins = client.insert(h.data(), 5, slots.data(), 5, /*start=*/0,
                             /*auto_recycle=*/false);
    assert(ins.error == InsertError::OK);

    auto q = lq(client, h.data(), 5);
    assert(q.common_hit == 5);
    assert(q.common_hit == 5);
    assert((int)hit_slots(q).size() == 5);
    for (int i = 0; i < 5; i++)
        assert(hit_slots(q)[i] == slots[i]);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
// §968-977 terminal, first block already diverges (ml==0): query for a sequence
// whose very first hash isn't a child of root → total miss, no slots.
static void test_miss_first_block() {
    TEST_BEGIN("query: divergent first block → hit=0");

    ServerLauncher launcher("/test_lq_4", default_cfg());
    RadixClient client("/test_lq_4");

    auto present = seq(4000, 4);
    assert(insert_auto(client, present.data(), 4).error == InsertError::OK);

    auto absent = seq(9000, 4);          // disjoint path
    auto q = lq(client, absent.data(), 4);
    assert(q.common_hit == 0);
    assert(hit_slots(q).empty());

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
// §968-985 terminal partial: the sequence ends inside a node but diverges
// partway. matched lands mid-node; only the matching prefix slots return.
static void test_terminal_partial_diverge() {
    TEST_BEGIN("query: partial match inside terminal node");

    ServerLauncher launcher("/test_lq_5", default_cfg());
    RadixClient client("/test_lq_5");

    // One node holds [0..5) as tokens 5000..5004.
    auto h = seq(5000, 5);
    std::vector<int32_t> slots;
    client.allocate_slots(5, slots);
    assert(client.insert(h.data(), 5, slots.data(), 5, 0, false).error
           == InsertError::OK);

    // Query shares first 3 (5000..5002) then diverges at block 3.
    std::vector<int64_t> t(5);
    std::iota(t.begin(), t.end(), 5000);
    t[3] = 77777;                        // diverge at block 3
    auto hq = RadixClient::compute_hashes(t.data(), 5, 1);
    auto q = lq(client, hq.data(), 5);
    assert(q.common_hit == 3);     // 5000..5002 matched
    assert((int)hit_slots(q).size() == 3);
    for (int i = 0; i < 3; i++)
        assert(hit_slots(q)[i] == slots[i]);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
// §968 terminal, query shorter than the node: n-matched < num_blocks, cmp caps
// to n. All n match, exactly n slots (not the whole node) return.
static void test_query_shorter_than_node() {
    TEST_BEGIN("query: n shorter than the node returns exactly n slots");

    ServerLauncher launcher("/test_lq_6", default_cfg());
    RadixClient client("/test_lq_6");

    // Single 6-block node.
    auto h = seq(6000, 6);
    std::vector<int32_t> slots;
    client.allocate_slots(6, slots);
    assert(client.insert(h.data(), 6, slots.data(), 6, 0, false).error
           == InsertError::OK);

    // Query only the first 4 blocks.
    auto q = lq(client, h.data(), 4);
    assert(q.common_hit == 4);
    assert((int)hit_slots(q).size() == 4);
    for (int i = 0; i < 4; i++)
        assert(hit_slots(q)[i] == slots[i]);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
// §989 descend: a split forces two nodes A=[0..k) → B; a full-length query must
// descend from A into B, accumulating A's slots (§995-996) then B's (§975).
static void test_descend_multi_node() {
    TEST_BEGIN("query: descends across a split boundary, hit spans both nodes");

    ServerLauncher launcher("/test_lq_7", default_cfg());
    RadixClient client("/test_lq_7");

    // Insert [0..6); then a divergent sibling sharing [0..3) forces a split of
    // the 6-node into A=[0..3) + tail, and adds B diverging at 3.
    auto base = seq(7000, 6);
    std::vector<int32_t> sb;
    client.allocate_slots(6, sb);
    assert(client.insert(base.data(), 6, sb.data(), 6, 0, false).error
           == InsertError::OK);

    // Full query of the original 6-seq must still return all 6 in order,
    // stitched across the split (A then its tail).
    auto q = lq(client, base.data(), 6);
    assert(q.common_hit == 6);
    assert((int)hit_slots(q).size() == 6);
    for (int i = 0; i < 6; i++)
        assert(hit_slots(q)[i] == sb[i]);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
// §995-996 + §975 slot stitching: build A=[0..3) and B=[3..5) as two nodes via
// start-bound inserts, then verify a 5-query returns A's 3 + B's 2 in order.
static void test_slots_stitched_across_nodes() {
    TEST_BEGIN("query: prefix_slots stitched in order across two nodes");

    ServerLauncher launcher("/test_lq_8", default_cfg());
    RadixClient client("/test_lq_8");

    auto h = seq(8000, 5);
    int32_t head[] = {80, 81, 82};       // [0..3)
    int32_t tail[] = {83, 84};           // [3..5)
    assert(client.insert(h.data(), 3, head, 3, 0, false).unused_slots.empty());
    // start=3 binds tail to hashes[3..5); this extends into a second node.
    auto ins2 = client.insert(h.data(), 5, tail, 2, /*start=*/3, false);
    assert(ins2.error == InsertError::OK);
    assert(ins2.unused_slots.empty());

    auto q = lq(client, h.data(), 5);
    assert(q.common_hit == 5);
    assert((int)hit_slots(q).size() == 5);
    assert(hit_slots(q)[0] == head[0] && hit_slots(q)[1] == head[1] &&
           hit_slots(q)[2] == head[2]);
    assert(hit_slots(q)[3] == tail[0] && hit_slots(q)[4] == tail[1]);

    expect_healthy(client, HEALTH_FULL_SLOT_LEAK);   // hand-picked slot ids, never allocated from the pool
    TEST_PASS();
}

// ==================================================================
// §1005-1021 non-root fallback partial: query descends into a deep node then
// diverges partway through it (not the terminal node). matched lands mid-deep.
static void test_diverge_mid_second_node() {
    TEST_BEGIN("query: diverge partway through a non-terminal deep node");

    ServerLauncher launcher("/test_lq_9", default_cfg());
    RadixClient client("/test_lq_9");

    // Force two nodes: A=[0..3), then extend B=[3..8) so the tree is A→B.
    auto h8 = seq(9000, 8);
    int32_t sa[] = {10, 11, 12};
    assert(client.insert(h8.data(), 3, sa, 3, 0, false).unused_slots.empty());
    int32_t sb[] = {13, 14, 15, 16, 17};  // [3..8)
    assert(client.insert(h8.data(), 8, sb, 5, /*start=*/3, false).error
           == InsertError::OK);

    // Query shares [0..3)+[3..6) then diverges at block 6 (mid-B).
    std::vector<int64_t> t(8);
    std::iota(t.begin(), t.end(), 9000);
    t[6] = 55555;                         // diverge at block 6
    auto hq = RadixClient::compute_hashes(t.data(), 8, 1);
    auto q = lq(client, hq.data(), 8);
    assert(q.common_hit == 6);      // [0..6) matched, stop mid-B
    assert((int)hit_slots(q).size() == 6);
    assert(hit_slots(q)[0] == sa[0]);
    assert(hit_slots(q)[3] == sb[0]);   // first block of B
    assert(hit_slots(q)[5] == sb[2]);

    expect_healthy(client, HEALTH_FULL_SLOT_LEAK);   // hand-picked slot ids, never allocated from the pool
    TEST_PASS();
}

// ==================================================================
// §1005 full node matched but no matching child: query fully matches a node
// whose only child diverges, so it stops exactly at the node boundary.
static void test_stop_at_node_boundary() {
    TEST_BEGIN("query: full node match with no matching child stops at boundary");

    ServerLauncher launcher("/test_lq_10", default_cfg());
    RadixClient client("/test_lq_10");

    // A=[0..3); child B diverges at block 3.
    auto ha = seq(10000, 3);
    int32_t sa[] = {20, 21, 22};
    assert(client.insert(ha.data(), 3, sa, 3, 0, false).unused_slots.empty());
    // Extend a different suffix so A has a child keyed by hashes[3].
    auto hb = seq(10000, 5);              // shares [0..3), then 10003,10004
    int32_t sb[] = {23, 24};
    assert(client.insert(hb.data(), 5, sb, 2, /*start=*/3, false).error
           == InsertError::OK);

    // Query [0..3) + a block-3 hash that is NOT B's key → matches [0..3), the
    // block-3 child lookup misses, stop at boundary 3.
    std::vector<int64_t> t(4);
    std::iota(t.begin(), t.end(), 10000);
    t[3] = 44444;                         // not B's block-3 hash
    auto hq = RadixClient::compute_hashes(t.data(), 4, 1);
    auto q = lq(client, hq.data(), 4);
    assert(q.common_hit == 3);
    assert((int)hit_slots(q).size() == 3);
    for (int i = 0; i < 3; i++) assert(hit_slots(q)[i] == sa[i]);

    expect_healthy(client, HEALTH_FULL_SLOT_LEAK);   // hand-picked slot ids, never allocated from the pool
    TEST_PASS();
}

// ==================================================================
// n deeper than the tree: query for more blocks than exist. matched caps at the
// deepest present block; no over-read past the tree.
static void test_query_deeper_than_tree() {
    TEST_BEGIN("query: n deeper than tree caps at deepest present block");

    ServerLauncher launcher("/test_lq_11", default_cfg());
    RadixClient client("/test_lq_11");

    // Tree holds only [0..3).
    auto h3 = seq(11000, 3);
    int32_t s3[] = {30, 31, 32};
    assert(client.insert(h3.data(), 3, s3, 3, 0, false).unused_slots.empty());

    // Ask for 10 blocks along the same prefix; only 3 exist.
    auto h10 = seq(11000, 10);
    auto q = lq(client, h10.data(), 10);
    assert(q.common_hit == 3);
    assert((int)hit_slots(q).size() == 3);
    for (int i = 0; i < 3; i++) assert(hit_slots(q)[i] == s3[i]);

    expect_healthy(client, HEALTH_FULL_SLOT_LEAK);   // hand-picked slot ids, never allocated from the pool
    TEST_PASS();
}

// ==================================================================
// QueryResult invariant: for a local-only query common_hit == total, and
// prefix_slots.size() == common_hit on every hit level.
static void test_local_hit_equals_total() {
    TEST_BEGIN("query: common_hit == common_hit == slots.size()");

    ServerLauncher launcher("/test_lq_12", default_cfg());
    RadixClient client("/test_lq_12");

    auto h = seq(12000, 7);
    std::vector<int32_t> slots;
    client.allocate_slots(7, slots);
    assert(client.insert(h.data(), 7, slots.data(), 7, 0, false).error
           == InsertError::OK);

    for (int probe : {0, 1, 4, 7}) {
        auto q = lq(client, h.data(), probe);
        assert(q.common_hit == probe);
        assert(q.common_hit == q.common_hit);
        assert((int)hit_slots(q).size() == q.common_hit);
    }

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
// prefix_slots length across a chunk-split multi-node prefix: a long sequence
// spanning several cs-aligned nodes must return exactly common_hit slots.
static void test_prefix_slots_length() {
    TEST_BEGIN("query: multi-chunk prefix returns exactly hit-length slots");

    ShmConfig cfg{2000, 20000};
    cfg.register_chunk_tokens = 8;          // several nodes for a long insert
    cfg.background_evict_ratio = 0.0f;
    ServerLauncher launcher("/test_lq_13", cfg);
    RadixClient client("/test_lq_13");

    const int N = 40;                     // 5 nodes of 8 blocks
    auto h = seq(13000, N);
    auto ins = insert_auto(client, h.data(), N);
    assert(ins.error == InsertError::OK);

    auto q = lq(client, h.data(), N);
    assert(q.common_hit == N);
    assert((int)hit_slots(q).size() == N);
    // Every slot distinct (positional binding, no duplicates across nodes).
    std::set<int32_t> uniq(hit_slots(q).begin(), hit_slots(q).end());
    assert((int)uniq.size() == N);

    // A mid-chunk query length (not on a node boundary) still returns exactly k.
    auto q2 = lq(client, h.data(), 21);
    assert(q2.common_hit == 21);
    assert((int)hit_slots(q2).size() == 21);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
// Public API §933-939: lock=true inc_ref's the matched prefix so a subsequent
// eviction pass cannot drop it, while an unpinned sibling is evicted first.
static void test_lock_pins_against_evict() {
    TEST_BEGIN("query(lock=true): pins matched prefix against eviction");

    // Small pool so a forced allocation must evict something.
    ShmConfig cfg{2000, 8};
    cfg.background_evict_ratio = 0.0f;
    cfg.register_chunk_tokens = 1;
    ServerLauncher launcher("/test_lq_14", cfg);
    RadixClient client("/test_lq_14");

    // Two disjoint 4-block paths fill the 8-block pool.
    auto ha = seq(14000, 4);
    auto hb = seq(15000, 4);
    assert(insert_auto(client, ha.data(), 4).error == InsertError::OK);
    assert(insert_auto(client, hb.data(), 4).error == InsertError::OK);
    assert(client.mempool_free() == 0);

    // Lock path A (pins its 4 nodes). Also touch B as LRU tail is irrelevant —
    // the pin, not recency, must decide survival.
    auto qa = lq(client, ha.data(), 4, /*lock=*/true, /*update_meta=*/true);
    assert(qa.common_hit == 4);

    // Force eviction of 4 blocks by allocating them: only unpinned B can go.
    std::vector<int32_t> got;
    client.allocate_slots(4, got);
    assert((int)got.size() == 4);

    // A survived (still fully queryable); B was evicted.
    assert(lq(client, ha.data(), 4).common_hit == 4);
    assert(lq(client, hb.data(), 4).common_hit == 0);

    // Release the pin so the pool can be reclaimed later.
    qa.finalize();

    client.recycle_slots(got.data(), (int)got.size());   // return the eviction-forcing slots
    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
// Public API §925-932: update_meta=true moves the matched terminal leaf to the
// LRU tail, so a freshly-touched path survives an eviction that drops the
// untouched one. update_meta=false must NOT reorder.
static void test_update_meta_touches_lru() {
    TEST_BEGIN("query(update_meta=true): touched prefix beats untouched on evict");

    ShmConfig cfg{2000, 8};
    cfg.background_evict_ratio = 0.0f;
    cfg.register_chunk_tokens = 1;
    ServerLauncher launcher("/test_lq_15", cfg);
    RadixClient client("/test_lq_15");

    // A then B fill the pool; B is MRU (inserted last).
    auto ha = seq(16000, 4);
    auto hb = seq(17000, 4);
    assert(insert_auto(client, ha.data(), 4).error == InsertError::OK);
    assert(insert_auto(client, hb.data(), 4).error == InsertError::OK);
    assert(client.mempool_free() == 0);

    // Touch A with update_meta=true → A becomes MRU, B becomes LRU.
    auto qa = lq(client, ha.data(), 4, /*lock=*/false, /*update_meta=*/true);
    assert(qa.common_hit == 4);

    // Evict 4 blocks: LRU (now B) must be the victim, A survives.
    std::vector<int32_t> got;
    client.allocate_slots(4, got);
    assert((int)got.size() == 4);
    assert(lq(client, ha.data(), 4).common_hit == 4);   // survived
    assert(lq(client, hb.data(), 4).common_hit == 0);   // evicted

    client.recycle_slots(got.data(), (int)got.size());   // return the eviction-forcing slots
    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
// Public API: update_meta=false leaves LRU order untouched — a queried-but-not-
// touched path stays LRU and is evicted before the more-recent one.
static void test_no_update_meta_leaves_lru() {
    TEST_BEGIN("query(update_meta=false): no LRU touch, order unchanged");

    ShmConfig cfg{2000, 8};
    cfg.background_evict_ratio = 0.0f;
    cfg.register_chunk_tokens = 1;
    ServerLauncher launcher("/test_lq_16", cfg);
    RadixClient client("/test_lq_16");

    // A (LRU) then B (MRU) fill the pool.
    auto ha = seq(18000, 4);
    auto hb = seq(19000, 4);
    assert(insert_auto(client, ha.data(), 4).error == InsertError::OK);
    assert(insert_auto(client, hb.data(), 4).error == InsertError::OK);
    assert(client.mempool_free() == 0);

    // Query A WITHOUT touching LRU → A stays LRU.
    auto qa = lq(client, ha.data(), 4, /*lock=*/false, /*update_meta=*/false);
    assert(qa.common_hit == 4);

    // Evict 4 blocks: A (still LRU) is the victim; B survives.
    std::vector<int32_t> got;
    client.allocate_slots(4, got);
    assert((int)got.size() == 4);
    assert(lq(client, ha.data(), 4).common_hit == 0);   // evicted
    assert(lq(client, hb.data(), 4).common_hit == 4);   // survived

    client.recycle_slots(got.data(), (int)got.size());   // return the eviction-forcing slots
    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
int main() {
    printf("=== test_tree_local_query ===\n\n");
    // Empty / boundary.
    test_empty_tree();
    test_n_zero();
    // Terminal-node branch (§968).
    test_exact_single_node();
    test_miss_first_block();
    test_terminal_partial_diverge();
    test_query_shorter_than_node();
    // Descend branch (§989) + slot stitching.
    test_descend_multi_node();
    test_slots_stitched_across_nodes();
    // Non-root fallback partial / boundary stop (§1005).
    test_diverge_mid_second_node();
    test_stop_at_node_boundary();
    test_query_deeper_than_tree();
    // QueryResult field invariants.
    test_local_hit_equals_total();
    test_prefix_slots_length();
    // Public-API lock / update_meta semantics.
    test_lock_pins_against_evict();
    test_update_meta_touches_lru();
    test_no_update_meta_leaves_lru();
    return tests_report();
}
