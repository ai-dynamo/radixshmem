// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// test_tree_insert — insert data flow: basic insert/match, prefix split,
// chunk-aligned boundaries, pool-full error paths, and caller-supplied slot
// inserts (exact/short/surplus supply, prefix interaction). Pure query/hash
// coverage lives in test_tree_basic; eviction in test_tree_evict.

#include "test_utils.h"
#include "hash.h"
#include "radix_client.h"

#include <cassert>
#include <cstring>
#include <numeric>
#include <set>
#include <vector>

using namespace radixshmem;

// Build a hash sequence [base+0 .. base+n) so distinct tests don't collide.
static std::vector<uint64_t> seq(int64_t base, int n) {
    std::vector<int64_t> t(n);
    std::iota(t.begin(), t.end(), base);
    return RadixClient::compute_hashes(t.data(), n, 1);
}

// ==================================================================
static void test_basic_insert_match() {
    TEST_BEGIN("basic insert + full match");

    ServerLauncher launcher("/test_ti_01", default_cfg());
    RadixClient client("/test_ti_01");

    int64_t tokens[] = {10, 20, 30, 40, 50};
    auto h = RadixClient::compute_hashes(tokens, 5, 1);
    auto ins = insert_auto(client, h.data(), 5);
    assert(ins.error == InsertError::OK);
    assert(ins.unused_slots.empty());          // all 5 landed
    assert(client.total_radix_nodes() == 2);

    auto r = client.query(h.data(), 5, COMPONENT_MASK_FULL);
    assert(r.common_hit == 5);
    assert((int)hit_slots(r).size() == 5);
    // Slots are distinct physical blocks.
    std::set<int32_t> uniq(hit_slots(r).begin(), hit_slots(r).end());
    assert(uniq.size() == 5);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
static void test_prefix_sharing_split() {
    TEST_BEGIN("prefix sharing triggers split");

    ServerLauncher launcher("/test_ti_02", default_cfg());
    RadixClient client("/test_ti_02");

    int64_t t1[] = {1, 2, 3, 4, 5};
    int64_t t2[] = {1, 2, 3, 6, 7};
    auto h1 = RadixClient::compute_hashes(t1, 5, 1);
    auto h2 = RadixClient::compute_hashes(t2, 5, 1);
    int32_t s1[] = {10, 11, 12, 13, 14};
    int32_t s2[] = {20, 21};                   // suffix for h2, bound at start=3
    client.insert(h1.data(), 5, s1, 5);
    auto ins2 = client.insert(h2.data(), 5, s2, 2, /*start=*/3);
    assert(ins2.error == InsertError::OK);
    assert(ins2.unused_slots.empty());         // both suffix slots landed
    assert(client.total_radix_nodes() == 4);   // root + prefix + 2 suffix

    auto r2 = client.query(h2.data(), 5, COMPONENT_MASK_FULL);
    assert(r2.common_hit == 5);
    // First 3 slots come from h1's original allocation; last 2 from s2.
    for (int i = 0; i < 3; i++)
        assert(hit_slots(r2)[i] == s1[i]);
    assert(hit_slots(r2)[3] == s2[0]);
    assert(hit_slots(r2)[4] == s2[1]);

    expect_healthy(client, HEALTH_FULL_SLOT_LEAK);   // hand-picked slot ids, never allocated from the pool
    TEST_PASS();
}

// ==================================================================
static void test_multiple_inserts_multiple_branches() {
    TEST_BEGIN("multi-branch tree all queryable");

    ServerLauncher launcher("/test_ti_03", default_cfg());
    RadixClient client("/test_ti_03");

    int64_t ta[] = {1, 2, 3, 4};
    int64_t tb[] = {1, 2, 5, 6};
    int64_t tc[] = {1, 2, 3, 7};
    int64_t td[] = {8, 9};
    auto ha = RadixClient::compute_hashes(ta, 4, 1);
    auto hb = RadixClient::compute_hashes(tb, 4, 1);
    auto hc = RadixClient::compute_hashes(tc, 4, 1);
    auto hd = RadixClient::compute_hashes(td, 2, 1);
    insert_auto(client, ha.data(), 4);
    insert_auto(client, hb.data(), 4);
    insert_auto(client, hc.data(), 4);
    insert_auto(client, hd.data(), 2);

    assert(client.query(ha.data(), 4, COMPONENT_MASK_FULL).common_hit == 4);
    assert(client.query(hb.data(), 4, COMPONENT_MASK_FULL).common_hit == 4);
    assert(client.query(hc.data(), 4, COMPONENT_MASK_FULL).common_hit == 4);
    assert(client.query(hd.data(), 2, COMPONENT_MASK_FULL).common_hit == 2);
    assert(client.query(ha.data(), 2, COMPONENT_MASK_FULL).common_hit == 2);  // shorter probe

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
static void test_duplicate_insert_idempotent() {
    TEST_BEGIN("duplicate insert is idempotent");

    ServerLauncher launcher("/test_ti_04", default_cfg());
    RadixClient client("/test_ti_04");

    int64_t t[] = {1, 2, 3};
    auto h = RadixClient::compute_hashes(t, 3, 1);
    insert_auto(client, h.data(), 3);
    uint32_t nodes_before = client.total_radix_nodes();

    auto ins2 = insert_auto(client, h.data(), 3);
    assert(ins2.error == InsertError::OK);
    assert(ins2.unused_slots.empty());   // nothing allocated, nothing to recycle
    assert(client.total_radix_nodes() == nodes_before);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
static void test_insert_chunked_local() {
    TEST_BEGIN("single insert spans multiple node-sized chunks");

    // No register_chunk_tokens — tree cap is MAX_BLOCKS_PER_NODE (4096),
    // but with 8 max_blocks we can still verify the chunk-loop runs
    // intact across multiple insert iterations.
    ServerLauncher launcher("/test_ti_05", default_cfg());
    RadixClient client("/test_ti_05");

    constexpr int N = 5000;  // > MAX_BLOCKS_PER_NODE (4096)
    std::vector<int64_t> tokens(N);
    for (int i = 0; i < N; i++) tokens[i] = i + 7000;
    auto h = RadixClient::compute_hashes(tokens.data(), N, 1);

    auto ins = insert_auto(client, h.data(), N);
    assert(ins.error == InsertError::OK);
    assert(ins.unused_slots.empty());

    auto r = client.query(h.data(), N, COMPONENT_MASK_FULL);
    assert(r.common_hit == N);
    assert((int)hit_slots(r).size() == N);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
// Chunk-aligned insert (register_chunk_tokens): cs splits the append into
// cs-sized nodes. InsertResult no longer carries node metadata, so we verify
// the split structurally via node count + query round-trip.
// ==================================================================

static void test_chunk_aligned_boundaries() {
    TEST_BEGIN("register_chunk_tokens aligns new-node boundaries");

    // cs now lives in the shm header (owner sets it on create); every attach
    // reads it. Build the server with register_chunk_tokens=8.
    ShmConfig cfg{1000, 10000};
    cfg.register_chunk_tokens = 8;
    ServerLauncher launcher("/test_ti_06", cfg);
    RadixClient client("/test_ti_06");

    constexpr int N = 24;
    std::vector<int64_t> tokens(N);
    for (int i = 0; i < N; i++) tokens[i] = i + 100;
    auto h = RadixClient::compute_hashes(tokens.data(), N, 1);

    auto ins = insert_auto(client, h.data(), N);
    assert(ins.error == InsertError::OK);
    assert(ins.unused_slots.empty());
    // [0,8) [8,16) [16,24) — three cs-aligned nodes off the root = 4 total.
    assert(client.total_radix_nodes() == 4);

    auto r = client.query(h.data(), N, COMPONENT_MASK_FULL);
    assert(r.common_hit == N);
    assert((int)hit_slots(r).size() == N);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
static void test_chunk_align_non_cs_start() {
    TEST_BEGIN("cs alignment with insert_start not on cs boundary");

    // Server-wide cs=8 (header-borne). Every insert aligns to it.
    ShmConfig cfg{1000, 10000};
    cfg.register_chunk_tokens = 8;
    ServerLauncher launcher("/test_ti_07", cfg);
    RadixClient client("/test_ti_07");

    // First insert 12 blocks — cs=8 splits into [0,8) + [8,12).
    std::vector<int64_t> tok1(12);
    for (int i = 0; i < 12; i++) tok1[i] = i + 200;
    auto h1 = RadixClient::compute_hashes(tok1.data(), 12, 1);
    insert_auto(client, h1.data(), 12);

    // Second insert: shares [0,12), diverges at 12, total 24 blocks.
    std::vector<int64_t> tok2(24);
    for (int i = 0; i < 12; i++) tok2[i] = i + 200;
    for (int i = 12; i < 24; i++) tok2[i] = i + 500;
    auto h2 = RadixClient::compute_hashes(tok2.data(), 24, 1);

    auto ins = insert_auto(client, h2.data(), 24);
    assert(ins.error == InsertError::OK);
    assert(ins.unused_slots.empty());   // 12 new suffix blocks all landed

    // Both prefixes remain queryable end-to-end; the divergent suffix starting
    // at 12 (not a cs multiple) still lands intact.
    assert(client.query(h1.data(), 12, COMPONENT_MASK_FULL).common_hit == 12);
    auto r = client.query(h2.data(), 24, COMPONENT_MASK_FULL);
    assert(r.common_hit == 24);
    assert((int)hit_slots(r).size() == 24);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
static void test_split_prefix_is_routable() {
    TEST_BEGIN("split-created prefix node stays queryable");

    ServerLauncher launcher("/test_ti_08", default_cfg());
    RadixClient client("/test_ti_08");

    // Insert [1..7] as one big node.
    int64_t ta[] = {1, 2, 3, 4, 5, 6, 7};
    auto ha = RadixClient::compute_hashes(ta, 7, 1);
    insert_auto(client, ha.data(), 7);

    // Insert [1,2,3,99,100] diverges at index 3 → splits the [1..7] node
    // into prefix [1..3] + suffix [4..7], then appends a new leaf [99,100].
    int64_t tb[] = {1, 2, 3, 99, 100};
    auto hb = RadixClient::compute_hashes(tb, 5, 1);
    auto ins = insert_auto(client, hb.data(), 5);
    assert(ins.error == InsertError::OK);
    assert(ins.unused_slots.empty());   // 2 new suffix blocks landed

    // root + prefix[1..3) + suffix[4..7) + leaf[99,100) = 4 nodes.
    assert(client.total_radix_nodes() == 4);
    // Both branches queryable end-to-end after the split.
    assert(client.query(ha.data(), 7, COMPONENT_MASK_FULL).common_hit == 7);
    assert(client.query(hb.data(), 5, COMPONENT_MASK_FULL).common_hit == 5);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
static void test_node_pool_full_error() {
    TEST_BEGIN("NodePool full → NODE_POOL_FULL");

    // Tiny node pool: only 4 nodes total. After enough disjoint inserts
    // the pool exhausts.
    ShmConfig cfg;
    cfg.max_nodes = 4;
    cfg.max_blocks = 1000;
    ServerLauncher launcher("/test_ti_09", cfg);
    RadixClient client("/test_ti_09");

    // insert_pinned pins every prior leaf (ref_count>0), so the mid-insert
    // node-pool eviction can't reclaim nodes to make room — otherwise evict
    // would recycle an unpinned leaf on every shortfall and NODE_POOL_FULL
    // is never hit.
    int err_hit = 0;
    for (int i = 0; i < 16; i++) {
        int64_t t = 1000 + i;
        auto h = RadixClient::compute_hashes(&t, 1, 1);
        auto ins = insert_pinned(client, h.data(), 1);
        if (ins.error == InsertError::NODE_POOL_FULL) { err_hit = 1; break; }
    }
    assert(err_hit == 1);

    expect_healthy(client, HEALTH_REF_LEAK);   // insert_pinned holds its pins for the test's lifetime
    TEST_PASS();
}

// ==================================================================
static void test_ht_pool_full_error() {
    TEST_BEGIN("HT pool full → HT_POOL_FULL (or NODE_POOL_FULL)");

    // ht_entry_capacity == max_nodes. With small max_nodes either pool can
    // fill first; both are acceptable — we just need to exercise the path.
    ShmConfig cfg;
    cfg.max_nodes = 8;
    cfg.max_blocks = 1000;
    ServerLauncher launcher("/test_ti_10", cfg);
    RadixClient client("/test_ti_10");

    // insert_pinned pins each leaf so eviction can't recycle nodes/entries to
    // make room (same reasoning as test_node_pool_full_error).
    int err_hit = 0;
    for (int i = 0; i < 16; i++) {
        int64_t t = 5000 + i;
        auto h = RadixClient::compute_hashes(&t, 1, 1);
        auto ins = insert_pinned(client, h.data(), 1);
        if (ins.error == InsertError::HT_POOL_FULL ||
            ins.error == InsertError::NODE_POOL_FULL) {
            err_hit = 1; break;
        }
    }
    assert(err_hit == 1);

    expect_healthy(client, HEALTH_REF_LEAK);   // insert_pinned holds its pins for the test's lifetime
    TEST_PASS();
}

// ==================================================================
// Regression: after eviction removes a node's LAST child, the next insert that
// re-parents under that node must take it back off the leaf_list.
//
// cs=1 builds root -> n1[1] -> n2[2] -> ... -> n5[5]. allocate_slots for the
// [1,2,3,6,7] insert evicts the tail (n5, n4), so n3 loses its only child and
// lands on the leaf_list (children==0). The insert then add_child(n3, [6]) —
// which must remove n3 from the leaf_list. If it doesn't, n3 sits there as an
// internal node and the next eviction drops it, wiping [1,2,3].
// ==================================================================
static void test_insert_evict_last_child_keeps_leaf_list_sane() {
    TEST_BEGIN("insert: evict of a node's last child keeps leaf_list invariant");

    // 5 slots hold [1..5]. cs=1 → one block per node. background_evict_ratio=0 disables
    // amplification + the bg evictor for a deterministic sequence.
    ShmConfig cfg{1000, 5};
    cfg.background_evict_ratio = 0.0f;
    cfg.register_chunk_tokens = 1;
    ServerLauncher launcher("/test_ti_11", cfg);
    RadixClient client("/test_ti_11");

    int64_t t_la[]  = {1, 2, 3, 4, 5};
    int64_t t_p[]   = {1, 2, 3};
    int64_t t_c[]   = {1, 2, 3, 6, 7};
    auto h_la = RadixClient::compute_hashes(t_la, 5, 1);
    auto h_p  = RadixClient::compute_hashes(t_p, 3, 1);
    auto h_c  = RadixClient::compute_hashes(t_c, 5, 1);

    // Build root -> n1[1] -> ... -> n5[5]; [1,2,3] is a full-match no-op.
    auto ins1 = insert_auto(client, h_la.data(), 5);
    assert(ins1.error == InsertError::OK);
    auto ins_split = insert_auto(client, h_p.data(), 3);
    assert(ins_split.error == InsertError::OK);
    assert(ins_split.unused_slots.empty());   // already present, nothing allocated
    assert(client.mempool_free() == 0);

    // Insert [1,2,3,6,7] pinned: allocate_slots evicts n5,n4 (freeing 2), so n3
    // becomes childless and hits the leaf_list; the insert re-parents [6] under
    // n3 (add_child must pull n3 back off). The pin protects the new tail.
    auto ins2 = insert_pinned(client, h_c.data(), 5);
    assert(ins2.error == InsertError::OK);
    assert(ins2.unused_slots.empty());   // 2 shortfall slots both landed

    // Tree is now root -> n1[1] -> n2[2] -> n3[3] -> [6] -> [7]; [4],[5] gone.
    assert(client.query(h_c.data(), 5, COMPONENT_MASK_FULL, false, false, false).common_hit == 5);
    assert(client.query(h_la.data(), 5, COMPONENT_MASK_FULL, false, false, false).common_hit == 3);
    assert(client.mempool_used() == 5);   // [1][2][3][6][7]

    // Force an eviction round. The tail leaf is pinned -> nothing evictable and
    // n3 is untouched. The buggy code left n3 on the leaf_list as an internal
    // node, so it evicts n3 directly and collapses [1,2,3].
    force_evict(client, 2);
    assert(client.query(h_p.data(), 3, COMPONENT_MASK_FULL, false, false, false).common_hit == 3);

    expect_healthy(client, HEALTH_REF_LEAK);   // insert_pinned holds its pins for the test's lifetime
    TEST_PASS();
}

// ==================================================================
// Caller-supplied slot inserts (external slots, consumed front-first).
// ==================================================================

static void test_exact_supply() {
    TEST_BEGIN("insert (external slots): exact supply lands verbatim");

    ServerLauncher launcher("/test_ti_12", default_cfg());
    RadixClient client("/test_ti_12");

    int64_t tokens[] = {10, 20, 30, 40, 50};
    auto h = RadixClient::compute_hashes(tokens, 5, 1);
    int32_t slots[] = {100, 101, 102, 103, 104};

    auto ins = client.insert(h.data(), 5, slots, 5);
    assert(ins.error == InsertError::OK);
    assert(ins.unused_slots.empty());   // all 5 supplied slots landed

    // The exact slots we supplied must be what the tree hands back on query.
    auto r = client.query(h.data(), 5, COMPONENT_MASK_FULL);
    assert(r.common_hit == 5);
    assert((int)hit_slots(r).size() == 5);
    for (int i = 0; i < 5; i++)
        assert(hit_slots(r)[i] == slots[i]);

    expect_healthy(client, HEALTH_FULL_SLOT_LEAK);   // hand-picked slot ids, never allocated from the pool
    TEST_PASS();
}

// ==================================================================
static void test_supply_too_few() {
    TEST_BEGIN("insert (external slots): short supply drops tail blocks");

    ServerLauncher launcher("/test_ti_13", default_cfg());
    RadixClient client("/test_ti_13");

    // 10 hashes but only 4 slots → only 4 blocks inserted, no MEMPOOL_FULL
    // (slots are external).
    int64_t tokens[10];
    for (int i = 0; i < 10; i++) tokens[i] = 1000 + i;
    auto h = RadixClient::compute_hashes(tokens, 10, 1);
    int32_t slots[] = {200, 201, 202, 203};

    auto ins = client.insert(h.data(), 10, slots, 4);
    assert(ins.error == InsertError::OK);
    assert(ins.unused_slots.empty());   // all 4 supplied slots consumed

    // Only the first 4 blocks are present; querying all 10 hits a 4-prefix.
    auto r = client.query(h.data(), 10, COMPONENT_MASK_FULL);
    assert(r.common_hit == 4);
    for (int i = 0; i < 4; i++)
        assert(hit_slots(r)[i] == slots[i]);

    expect_healthy(client, HEALTH_FULL_SLOT_LEAK);   // hand-picked slot ids, never allocated from the pool
    TEST_PASS();
}

// ==================================================================
static void test_supply_too_many() {
    TEST_BEGIN("insert (external slots): surplus slots left untouched");

    ServerLauncher launcher("/test_ti_14", default_cfg());
    RadixClient client("/test_ti_14");

    int64_t tokens[] = {1, 2, 3};
    auto h = RadixClient::compute_hashes(tokens, 3, 1);
    // Supply 6 slots for a 3-block insert; only the first 3 are consumed.
    // Literal (non-pool) slot IDs → auto_recycle=false so the surplus isn't
    // fed back into the mempool free list.
    int32_t slots[] = {300, 301, 302, 303, 304, 305};

    auto ins = client.insert(h.data(), 3, slots, 6, /*start=*/0,
                             /*auto_recycle=*/false);
    assert(ins.error == InsertError::OK);
    // 3 consumed, trailing 3 surplus returned as unused.
    assert(ins.unused_slots.size() == 3);
    assert(ins.unused_slots[0] == 303);
    assert(ins.unused_slots[1] == 304);
    assert(ins.unused_slots[2] == 305);

    auto r = client.query(h.data(), 3, COMPONENT_MASK_FULL);
    assert(r.common_hit == 3);
    for (int i = 0; i < 3; i++)
        assert(hit_slots(r)[i] == slots[i]);

    expect_healthy(client, HEALTH_FULL_SLOT_LEAK);   // hand-picked slot ids, never allocated from the pool
    TEST_PASS();
}

// ==================================================================
static void test_prefix_match_consumes_suffix_only() {
    TEST_BEGIN("insert (external slots, start=0): redundant prefix slots recycled");

    ServerLauncher launcher("/test_ti_15", default_cfg());
    RadixClient client("/test_ti_15");

    // First insert establishes a 3-block prefix.
    int64_t t1[] = {1, 2, 3, 4, 5};
    auto h1 = RadixClient::compute_hashes(t1, 5, 1);
    int32_t s1[] = {10, 11, 12, 13, 14};
    auto ins1 = client.insert(h1.data(), 5, s1, 5, /*start=*/0,
                              /*auto_recycle=*/false);
    assert(ins1.unused_slots.empty());

    // Second shares the first 3 blocks. Slots are POSITIONALLY bound to
    // hashes[start:] — with start=0, s2[i] binds hashes[i], so the 3 already-
    // matched blocks make s2[0..3) redundant (returned as unused) and only
    // s2[3],s2[4] land on the two new suffix blocks.
    int64_t t2[] = {1, 2, 3, 6, 7};
    auto h2 = RadixClient::compute_hashes(t2, 5, 1);
    int32_t s2[] = {20, 21, 22, 23, 24};
    auto ins2 = client.insert(h2.data(), 5, s2, 5, /*start=*/0,
                              /*auto_recycle=*/false);
    assert(ins2.error == InsertError::OK);
    // Leading 3 redundant (matched > start), returned as unused.
    assert(ins2.unused_slots.size() == 3);
    assert(ins2.unused_slots[0] == 20);
    assert(ins2.unused_slots[1] == 21);
    assert(ins2.unused_slots[2] == 22);

    auto r = client.query(h2.data(), 5, COMPONENT_MASK_FULL);
    assert(r.common_hit == 5);
    // First 3 slots inherited from h1's allocation; last 2 are s2[3],s2[4]
    // (positionally bound to hashes[3],hashes[4]).
    for (int i = 0; i < 3; i++)
        assert(hit_slots(r)[i] == s1[i]);
    assert(hit_slots(r)[3] == s2[3]);
    assert(hit_slots(r)[4] == s2[4]);

    expect_healthy(client, HEALTH_FULL_SLOT_LEAK);   // hand-picked slot ids, never allocated from the pool
    TEST_PASS();
}

// ==================================================================
static void test_full_match_consumes_nothing() {
    TEST_BEGIN("insert (external slots): full match inserts nothing");

    ServerLauncher launcher("/test_ti_16", default_cfg());
    RadixClient client("/test_ti_16");

    int64_t tokens[] = {7, 8, 9};
    auto h = RadixClient::compute_hashes(tokens, 3, 1);
    int32_t s1[] = {30, 31, 32};
    auto ins1 = client.insert(h.data(), 3, s1, 3, /*start=*/0,
                              /*auto_recycle=*/false);
    assert(ins1.unused_slots.empty());

    // Re-insert identical hashes: whole prefix already present (matched==n),
    // start=0 makes all 3 supplied slots redundant → nothing lands.
    int32_t s2[] = {40, 41, 42};
    auto ins2 = client.insert(h.data(), 3, s2, 3, /*start=*/0,
                              /*auto_recycle=*/false);
    assert(ins2.error == InsertError::OK);
    assert(ins2.unused_slots.size() == 3);   // all redundant

    // Original slots remain bound.
    auto r = client.query(h.data(), 3, COMPONENT_MASK_FULL);
    for (int i = 0; i < 3; i++)
        assert(hit_slots(r)[i] == s1[i]);

    expect_healthy(client, HEALTH_FULL_SLOT_LEAK);   // hand-picked slot ids, never allocated from the pool
    TEST_PASS();
}

// ==================================================================
static void test_prefix_slots_full_match() {
    TEST_BEGIN("query: prefix_slots covers the full matched prefix");

    ServerLauncher launcher("/test_ti_17", default_cfg());
    RadixClient client("/test_ti_17");

    // Build a two-node prefix: first 3 blocks, then extend by 2 — yielding
    // node A=[0,3) + node B=[3,5). prefix_slots must stitch both nodes.
    int64_t tokens[] = {71, 72, 73, 74, 75};
    auto h = RadixClient::compute_hashes(tokens, 5, 1);
    int32_t head_slots[] = {80, 81, 82};   // first 3
    int32_t tail_slots[] = {83, 84};       // suffix 2, bound at start=3

    auto ins1 = client.insert(h.data(), 3, head_slots, 3, /*start=*/0,
                              /*auto_recycle=*/false);
    assert(ins1.unused_slots.empty());

    // Shortfall-only supply: tail_slots binds hashes[3:5], so start=3.
    auto ins2 = client.insert(h.data(), 5, tail_slots, 2, /*start=*/3,
                              /*auto_recycle=*/false);
    assert(ins2.error == InsertError::OK);
    assert(ins2.unused_slots.empty());   // both suffix slots landed

    auto r = client.query(h.data(), 5, COMPONENT_MASK_FULL);
    assert(r.common_hit == 5);

    // prefix_slots: the full matched prefix (all 5), stitched across A and B.
    assert((int)hit_slots(r).size() == 5);
    for (int i = 0; i < 3; i++)
        assert(hit_slots(r)[i] == head_slots[i]);
    assert(hit_slots(r)[3] == tail_slots[0]);
    assert(hit_slots(r)[4] == tail_slots[1]);

    expect_healthy(client, HEALTH_FULL_SLOT_LEAK);   // hand-picked slot ids, never allocated from the pool
    TEST_PASS();
}

// ==================================================================
static void test_mix_with_regular_insert() {
    TEST_BEGIN("insert (external slots): coexists with mempool insert");

    ServerLauncher launcher("/test_ti_18", default_cfg());
    RadixClient client("/test_ti_18");

    // Regular insert allocates from mempool.
    int64_t t1[] = {1, 2, 3};
    auto h1 = RadixClient::compute_hashes(t1, 3, 1);
    auto ins1 = insert_auto(client, h1.data(), 3);
    assert(ins1.unused_slots.empty());
    auto q1 = client.query(h1.data(), 3, COMPONENT_MASK_FULL);   // capture the mempool slots

    // External-slot insert on a disjoint path, same tree.
    int64_t t2[] = {91, 92, 93};
    auto h2 = RadixClient::compute_hashes(t2, 3, 1);
    int32_t s2[] = {500, 501, 502};
    auto ins2 = client.insert(h2.data(), 3, s2, 3, /*start=*/0,
                              /*auto_recycle=*/false);
    assert(ins2.unused_slots.empty());

    // Both paths are independently queryable.
    auto r1 = client.query(h1.data(), 3, COMPONENT_MASK_FULL);
    assert(r1.common_hit == 3);
    for (int i = 0; i < 3; i++)
        assert(hit_slots(r1)[i] == hit_slots(q1)[i]);

    auto r2 = client.query(h2.data(), 3, COMPONENT_MASK_FULL);
    assert(r2.common_hit == 3);
    for (int i = 0; i < 3; i++)
        assert(hit_slots(r2)[i] == s2[i]);

    expect_healthy(client, HEALTH_FULL_SLOT_LEAK);   // hand-picked slot ids, never allocated from the pool
    TEST_PASS();
}

// ==================================================================
// Empty / non-positive input. n==0 is a legal no-op: nothing inserts, so every
// supplied slot is returned unused (and recycled when auto_recycle). n<0 is an
// invalid arg: empty return, slots untouched.
static void test_empty_input() {
    TEST_BEGIN("insert: n==0 returns/recycles all slots; n<0 is a no-op");

    ServerLauncher launcher("/test_ti_19", default_cfg());
    RadixClient client("/test_ti_19");

    uint64_t h[] = {1, 2, 3};
    int32_t s[] = {10, 11, 12};

    // n==0 with literal slots + auto_recycle=false: all 3 returned, none consumed.
    auto r0 = client.insert(h, 0, s, 3, /*start=*/0, /*auto_recycle=*/false);
    assert(r0.error == InsertError::OK);
    assert((int)r0.unused_slots.size() == 3);
    std::set<int32_t> u0(r0.unused_slots.begin(), r0.unused_slots.end());
    for (int i = 0; i < 3; i++) assert(u0.count(s[i]) == 1);
    assert(client.total_radix_nodes() == 1);   // root only

    // n==0 with real mempool slots + auto_recycle=true: pool returns to full.
    std::vector<int32_t> ps;
    client.allocate_slots(3, ps);
    uint32_t free_before = client.mempool_free();
    auto rp = client.insert(h, 0, ps.data(), (int)ps.size(),
                            /*start=*/0, /*auto_recycle=*/true);
    assert(rp.error == InsertError::OK);
    assert((int)rp.unused_slots.size() == 3);
    assert(client.mempool_free() == free_before + 3);   // all 3 recycled
    assert(client.total_radix_nodes() == 1);

    // n<0: invalid arg, empty return, nothing recycled.
    auto rneg = client.insert(h, -5, s, 3);
    assert(rneg.error == InsertError::OK);
    assert(rneg.unused_slots.empty());
    assert(client.total_radix_nodes() == 1);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
// Auto-merge FULL path: appending one block at a time to an only-child tail
// folds each new leaf up into its parent, so the tail stays one growing node
// (root+1) instead of a chain.
static void test_automerge_full_progressive_collapse() {
    TEST_BEGIN("insert auto-merge: progressive appends collapse to one node");

    ServerLauncher launcher("/test_ti_20", default_cfg());
    RadixClient client("/test_ti_20");

    auto h = seq(1000, 64);
    insert_auto(client, h.data(), 3);                    // seed [0,3)
    for (int k = 4; k <= 64; ++k) {
        auto ins = insert_auto(client, h.data(), k);     // extends by one block
        assert(ins.error == InsertError::OK);
        assert(ins.unused_slots.empty());                // the 1 shortfall slot landed
        assert(client.total_radix_nodes() == 2);         // root + one merged tail
    }

    auto r = client.query(h.data(), 64, COMPONENT_MASK_FULL);
    assert(r.common_hit == 64);
    assert((int)hit_slots(r).size() == 64);
    // Slots stay distinct physical blocks (merge double-maps nothing).
    std::vector<int32_t> s = hit_slots(r);
    std::sort(s.begin(), s.end());
    assert(std::unique(s.begin(), s.end()) == s.end());

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
// Auto-merge multi-child guard (insert_impl §1350 children_count==1): a parent
// with two children must never fold either leaf up, even when the appended
// leaf would otherwise be foldable.
static void test_automerge_multichild_guard() {
    TEST_BEGIN("insert auto-merge: parent with two children is never merged into");

    ServerLauncher launcher("/test_ti_21", default_cfg());
    RadixClient client("/test_ti_21");

    // A=[0..5) as one node.
    auto ha = seq(3000, 5);
    insert_auto(client, ha.data(), 5);

    // B diverges at block 4 → splits A into P=[0..4) + [4] and adds [4'].
    // P now has two children; the freshly-inserted leaf must not fold up.
    std::vector<int64_t> tb(5);
    std::iota(tb.begin(), tb.end(), 3000);
    tb[4] = 9999;                                        // divergent last block
    auto hb = RadixClient::compute_hashes(tb.data(), 5, 1);
    auto ins = insert_auto(client, hb.data(), 5);
    assert(ins.error == InsertError::OK);
    assert(ins.unused_slots.empty());   // 1 divergent block landed

    // root + P[0..4) + leaf[4] + leaf[4'] = 4 nodes — nothing collapsed.
    assert(client.total_radix_nodes() == 4);
    assert(client.query(ha.data(), 5, COMPONENT_MASK_FULL, false, false, false).common_hit == 5);
    assert(client.query(hb.data(), 5, COMPONENT_MASK_FULL, false, false, false).common_hit == 5);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
// Auto-merge PARTIAL / cap edge (merge_leaf_into_parent §694 full_end>cap_end):
// progressive collapse fills one node to exactly 4096, then the next append
// can't be fully absorbed and starts a fresh node.
static void test_automerge_cap_4096_boundary() {
    TEST_BEGIN("insert auto-merge: node fills to 4096 then a new node starts");

    ShmConfig cfg{0, 20000};                            // headroom for a 4096 node
    cfg.register_chunk_tokens = 4096;   // cs multiples are merge anchors; keep them off [1,4096)
    ServerLauncher launcher("/test_ti_22", cfg);
    RadixClient client("/test_ti_22");

    const int N = 4097;
    auto h = seq(4000, N);
    insert_auto(client, h.data(), 1);
    for (int k = 2; k <= N; ++k) {
        auto ins = insert_auto(client, h.data(), k);
        assert(ins.error == InsertError::OK);
    }

    // The tail capped at 4096: block 4096 could not fold in, so it became a
    // second node. root + [0,4096) + [4096,4097).
    assert(client.total_radix_nodes() == 3);
    assert(client.total_blocks() == (uint64_t)N);

    auto r = client.query(h.data(), N, COMPONENT_MASK_FULL);
    assert(r.common_hit == N);
    assert((int)hit_slots(r).size() == N);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
// Auto-merge anchor: a leaf that starts on a register_chunk_tokens multiple is
// an RHT probe anchor and never folds into its parent; leaves starting off
// the grid still fold.
static void test_automerge_keeps_cs_anchor() {
    TEST_BEGIN("insert auto-merge: leaf starting on a cs multiple keeps its node");

    ShmConfig cfg{0, 2000};
    cfg.register_chunk_tokens = 128;
    ServerLauncher launcher("/test_ti_23", cfg);
    RadixClient client("/test_ti_23");

    auto h = seq(5000, 300);
    insert_auto(client, h.data(), 128);                  // [0,128)
    insert_auto(client, h.data(), 129);                  // [128,129) stays: anchor
    assert(client.total_radix_nodes() == 3);
    insert_auto(client, h.data(), 200);                  // [129,200) folds into [128,129)
    assert(client.total_radix_nodes() == 3);
    insert_auto(client, h.data(), 256);                  // [200,256) folds: [128,256)
    assert(client.total_radix_nodes() == 3);
    insert_auto(client, h.data(), 300);                  // [256,300) stays: anchor
    assert(client.total_radix_nodes() == 4);
    assert(client.total_blocks() == 300);

    auto r = client.query(h.data(), 300, COMPONENT_MASK_FULL);
    assert(r.common_hit == 300 && (int)hit_slots(r).size() == 300);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
// DataPool full inside the append loop (insert_impl §1276): data_.alloc fails,
// the mid-loop evict can't free enough (tail pinned), DATA_POOL_FULL is set and
// the loop breaks. A partial insert may land; existing pinned data survives.
static void test_data_pool_full_loop() {
    TEST_BEGIN("insert: DataPool exhausted mid-loop → DATA_POOL_FULL, prior data intact");

    // Tiny data pool: ratio 0.05 → DataPool holds ~1k blocks, far fewer than the
    // 8192-block mempool and 100k node pool, so data_.alloc is the sole bottleneck
    // and starves first. background evictor off + everything pinned so nothing can
    // be reclaimed mid-loop.
    ShmConfig cfg;
    cfg.max_nodes = 100000;
    cfg.max_blocks = 8192;
    cfg.data_pool_ratio = 0.05f;
    cfg.background_evict_ratio = 0.0f;
    cfg.register_chunk_tokens = 1;
    ServerLauncher launcher("/test_ti_23", cfg);
    RadixClient client("/test_ti_23");

    // Pin disjoint single-block inserts until the data pool refuses one.
    int err_hit = 0;
    int64_t survivor = -1;
    std::vector<uint64_t> h_survivor;
    for (int i = 0; i < 8192; i++) {
        int64_t t = 20000 + i;
        auto h = RadixClient::compute_hashes(&t, 1, 1);
        auto ins = insert_pinned(client, h.data(), 1);
        if (ins.error == InsertError::DATA_POOL_FULL) { err_hit = 1; break; }
        assert(ins.error == InsertError::OK);
        survivor = t; h_survivor = h;    // last one that made it in
    }
    assert(err_hit == 1);

    // A block that was accepted before the pool filled is still queryable —
    // the failed insert did not corrupt committed data.
    assert(survivor >= 0);
    assert(client.query(h_survivor.data(), 1, COMPONENT_MASK_FULL, false, false, false)
               .common_hit == 1);

    expect_healthy(client, HEALTH_REF_LEAK);   // insert_pinned holds its pins for the test's lifetime
    TEST_PASS();
}

// ==================================================================
// DataPool full inside split (insert_impl §1210): a divergent insert needs to
// split an existing node, but split_node's data_.alloc fails. The insert aborts
// with DATA_POOL_FULL and the original node is left fully intact (query still
// returns the whole original prefix).
static void test_data_pool_full_split() {
    TEST_BEGIN("insert: DataPool exhausted during split → DATA_POOL_FULL, original intact");

    // One node holding [0..8) then fill the rest of a tiny data pool with pinned
    // singletons so no free data blocks remain for the split's prefix/suffix.
    // Node/mempool sized well above data-pool capacity so DataPool starves first.
    ShmConfig cfg;
    cfg.max_nodes = 100000;
    cfg.max_blocks = 8192;
    cfg.data_pool_ratio = 0.05f;
    cfg.background_evict_ratio = 0.0f;
    cfg.register_chunk_tokens = 128;      // [0..8) stays one node (no cs split)
    ServerLauncher launcher("/test_ti_24", cfg);
    RadixClient client("/test_ti_24");

    // The node we will try to split — pin it so eviction can't drop it.
    auto base = seq(30000, 8);
    auto ins_base = insert_pinned(client, base.data(), 8);
    assert(ins_base.error == InsertError::OK);
    assert(ins_base.unused_slots.empty());   // all 8 landed

    // Starve the data pool with pinned singletons until alloc refuses.
    for (int i = 0; i < 8192; i++) {
        int64_t t = 40000 + i;
        auto h = RadixClient::compute_hashes(&t, 1, 1);
        auto ins = insert_pinned(client, h.data(), 1);
        if (ins.error != InsertError::OK) break;   // pool full — good, stop here
    }

    // Now insert a sibling that shares [0..4) of `base` and diverges at 4:
    // this must split the base node, but no data blocks are free → the split
    // fails with DATA_POOL_FULL.
    std::vector<int64_t> div(5);
    std::iota(div.begin(), div.end(), 30000);   // shares 30000..30003
    div[4] = 88888;                              // diverge at block 4
    auto hdiv = RadixClient::compute_hashes(div.data(), 5, 1);
    auto ins_div = client.insert(hdiv.data(), 5, nullptr, 0);
    // slots=0 anyway, but the split happens before slot consumption, so the
    // DATA_POOL_FULL from split_node is what we must observe.
    assert(ins_div.error == InsertError::DATA_POOL_FULL);

    // The original 8-block prefix is untouched — split rolled back cleanly.
    assert(client.query(base.data(), 8, COMPONENT_MASK_FULL, false, false, false)
               .common_hit == 8);

    expect_healthy(client, HEALTH_REF_LEAK);   // insert_pinned holds its pins for the test's lifetime
    TEST_PASS();
}

// ==================================================================
// Split leaves new_slots<=0 (insert_impl §1230): a divergent insert that splits
// a node but supplies zero slots for the new suffix. The split-created prefix
// becomes a childless non-root node and must be re-added to the leaf_list, and
// the tree stays queryable to the split boundary.
static void test_split_zero_new_slots_reparents() {
    TEST_BEGIN("insert: split with zero supplied slots re-lists prefix node");

    ServerLauncher launcher("/test_ti_25", default_cfg());
    RadixClient client("/test_ti_25");

    // A=[0..5) as one node.
    auto ha = seq(6000, 5);
    auto ins_a = insert_auto(client, ha.data(), 5);
    assert(ins_a.error == InsertError::OK);
    assert(ins_a.unused_slots.empty());
    uint32_t nodes_after_a = client.total_radix_nodes();  // root + A = 2

    // Insert [0..3) of A + one divergent block, but supply ZERO slots. The
    // matched prefix is 3 (mid-node), forcing a split of A into P=[0..3)+[3..5);
    // new_slots = 4-3 capped to 0 → nothing appended, but P was created.
    std::vector<int64_t> tb(4);
    std::iota(tb.begin(), tb.end(), 6000);       // shares 6000..6002
    tb[3] = 77777;                                // divergent 4th block
    auto hb = RadixClient::compute_hashes(tb.data(), 4, 1);
    auto ins_b = client.insert(hb.data(), 4, nullptr, 0);
    assert(ins_b.error == InsertError::OK);
    assert(ins_b.unused_slots.empty());   // no slots supplied
    // The split created the prefix node P: root + P[0..3) + suffix[3..5) = 3.
    assert(client.total_radix_nodes() == nodes_after_a + 1);

    // Both the split boundary and the original tail remain queryable.
    assert(client.query(ha.data(), 3, COMPONENT_MASK_FULL, false, false, false).common_hit == 3);
    assert(client.query(ha.data(), 5, COMPONENT_MASK_FULL, false, false, false).common_hit == 5);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
// (a) 100 sequential 1-block inserts through insert_auto — every one lands a
// fresh single node, none recycles a slot, tree grows monotonically.
static void test_start_100x_single_block() {
    TEST_BEGIN("start: 100x 1-block insert_auto, each lands, no recycle");

    ServerLauncher launcher("/test_ti_26", default_cfg());
    RadixClient client("/test_ti_26");

    std::vector<std::vector<uint64_t>> hs;
    for (int i = 0; i < 100; i++) {
        int64_t tok = 100000 + i;                 // distinct 1-token sequence
        auto h = RadixClient::compute_hashes(&tok, 1, 1);
        auto ins = insert_auto(client, h.data(), 1);
        assert(ins.error == InsertError::OK);
        assert(ins.unused_slots.empty());         // one fresh slot, all consumed
        hs.push_back(std::move(h));
    }
    assert(client.total_blocks() == 100);

    // Every slot round-trips: query each hash back, hit length 1.
    for (auto& h : hs)
        assert(client.query(h.data(), 1, COMPONENT_MASK_FULL).common_hit == 1);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
// (b) 10x insert of one long (multi-node / multi-chunk) sequence. First lands
// all; the next 9 are full matches → publish nothing, allocate nothing, tree
// stays a fixed size, no corruption.
static void test_start_10x_long_sequence_idempotent() {
    TEST_BEGIN("start: 10x long-sequence insert, first lands then idempotent");

    // cs*k spans multiple nodes/chunks. cs=8, k=40 → 320 blocks.
    ShmConfig cfg{20000, 4000};
    cfg.register_chunk_tokens = 8;
    ServerLauncher launcher("/test_ti_27", cfg);
    RadixClient client("/test_ti_27");

    constexpr int N = 8 * 40;
    std::vector<int64_t> tok(N);
    for (int i = 0; i < N; i++) tok[i] = 200000 + i;
    auto h = RadixClient::compute_hashes(tok.data(), N, 1);

    auto ins0 = insert_auto(client, h.data(), N);
    assert(ins0.error == InsertError::OK);
    assert(ins0.unused_slots.empty());            // fresh: all N landed
    uint32_t nodes0 = client.total_radix_nodes();
    int64_t blocks0 = (int64_t)client.total_blocks();
    assert(blocks0 == N);

    for (int i = 0; i < 9; i++) {
        auto ins = insert_auto(client, h.data(), N);
        assert(ins.error == InsertError::OK);
        // Full match → insert_auto allocates 0, so unused is empty (nothing to
        // give back) and the tree is unchanged.
        assert(ins.unused_slots.empty());
        assert(client.total_radix_nodes() == nodes0);
        assert((int64_t)client.total_blocks() == blocks0);
    }
    assert(client.query(h.data(), N, COMPONENT_MASK_FULL).common_hit == N);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
// (c) Reject (#3): peek matched=X, allocate the shortfall, then the matched
// prefix gets evicted before insert → real matched < start → whole insert
// rejected, every allocated slot returned unused (and auto-recycled), the child
// path never lands.
static void test_start_reject_evicted_prefix() {
    TEST_BEGIN("start: matched<start rejects, all slots unused, path misses");

    // 4-block pool, cs=1, deterministic (no bg evict). A=[0..4) FILLS the pool;
    // extend B=A+[4..8) needs 4 more — allocating them evicts all of A (LRU,
    // unpinned), so at insert time matched drops to 0 < start=4.
    ShmConfig cfg{2000, 4};
    cfg.background_evict_ratio = 0.0f;
    cfg.register_chunk_tokens = 1;
    ServerLauncher launcher("/test_ti_28", cfg);
    RadixClient client("/test_ti_28");

    // Prefix A = [0..4). Land it (unpinned → evictable). Fills the pool.
    std::vector<int64_t> ta(4);
    for (int i = 0; i < 4; i++) ta[i] = 300000 + i;
    auto ha = RadixClient::compute_hashes(ta.data(), 4, 1);
    auto insA = insert_auto(client, ha.data(), 4);
    assert(insA.error == InsertError::OK);
    assert(insA.unused_slots.empty());
    assert(client.mempool_free() == 0);   // pool full, A occupies all 4

    // B = A + [4..8), 8 blocks. Peek matched (== 4, A present).
    std::vector<int64_t> tb(8);
    for (int i = 0; i < 8; i++) tb[i] = 300000 + i;   // shares A's first 4
    auto hb = RadixClient::compute_hashes(tb.data(), 8, 1);
    int matched = (int)client.query(hb.data(), 8, COMPONENT_MASK_FULL, false, false, false)
                      .common_hit;
    assert(matched == 4);

    // Allocate the shortfall (4 slots). Pool is full → this evicts all of A's
    // blocks (LRU). KV data is (conceptually) written into these slots now.
    std::vector<int32_t> slots;
    client.allocate_slots(4, slots);
    assert((int)slots.size() == 4);
    // A is gone: the prefix no longer matches.
    assert(client.query(hb.data(), 8, COMPONENT_MASK_FULL, false, false, false).common_hit == 0);

    // Real matched is now 0 < start=4 (A evicted). Insert must reject wholesale.
    auto ins = client.insert(hb.data(), 8, slots.data(), (int)slots.size(),
                             /*start=*/matched, /*auto_recycle=*/true);
    assert(ins.error == InsertError::OK);                 // reject is not an error
    assert((int)ins.unused_slots.size() == (int)slots.size());  // all returned
    // auto_recycle returned our 4 slots to a now-empty pool.
    assert(client.mempool_free() == (uint32_t)slots.size());
    assert(client.total_blocks() == 0);   // nothing in the tree

    // The child path never landed: querying B still misses entirely.
    assert(client.query(hb.data(), 8, COMPONENT_MASK_FULL, false, false, false).common_hit == 0);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
// (d) Leading redundant (#4): peek matched=4, allocate for [4..9), but before
// insert a competing writer lands [4..7) (and a divergent [9]) so real
// matched=7. skip=3 → only slots for [7..9) consume, the 3 leading slots are
// returned unused (redundant) and recycled; mempool accounting stays exact.
static void test_start_skip_leading_redundant() {
    TEST_BEGIN("start: matched>start skips leading redundant slots");

    ShmConfig cfg{4000, 64};
    cfg.background_evict_ratio = 0.0f;
    cfg.register_chunk_tokens = 1;
    ServerLauncher launcher("/test_ti_29", cfg);
    RadixClient client("/test_ti_29");

    // Prefix P = [0..4). matched for the 9-seq below will start at 4.
    std::vector<int64_t> tp(4);
    for (int i = 0; i < 4; i++) tp[i] = 400000 + i;
    auto hp = RadixClient::compute_hashes(tp.data(), 4, 1);
    assert(insert_auto(client, hp.data(), 4).error == InsertError::OK);

    // Target sequence S = [0..9). Peek matched=4, plan to fill [4..9) (5 slots).
    std::vector<int64_t> ts(9);
    for (int i = 0; i < 9; i++) ts[i] = 400000 + i;
    auto hs = RadixClient::compute_hashes(ts.data(), 9, 1);
    int matched = (int)client.query(hs.data(), 9, COMPONENT_MASK_FULL, false, false, false)
                      .common_hit;
    assert(matched == 4);

    std::vector<int32_t> slots;
    client.allocate_slots(9 - matched, slots);           // 5 slots for [4..9)
    assert((int)slots.size() == 5);

    // Competing writer lands [4..7) so the real prefix now reaches 7. (Uses its
    // own allocation; independent of our 5 pre-allocated slots.)
    std::vector<int64_t> tc(7);
    for (int i = 0; i < 7; i++) tc[i] = 400000 + i;       // [0..7)
    auto hc = RadixClient::compute_hashes(tc.data(), 7, 1);
    assert(insert_auto(client, hc.data(), 7).error == InsertError::OK);
    assert(client.query(hs.data(), 9, COMPONENT_MASK_FULL, false, false, false).common_hit == 7);

    // Now insert S with the STALE start=4. Real matched=7 → skip=3: the 3
    // leading slots (for [4..7), already present) go unused; only slots[3],
    // slots[4] land for [7..9).
    uint32_t free_before = client.mempool_free();
    auto ins = client.insert(hs.data(), 9, slots.data(), (int)slots.size(),
                             /*start=*/matched, /*auto_recycle=*/true);
    assert(ins.error == InsertError::OK);
    assert((int)ins.unused_slots.size() == 3);           // the 3 leading redundant
    // The 3 recycled leading slots are exactly slots[0..3).
    std::set<int32_t> unused(ins.unused_slots.begin(), ins.unused_slots.end());
    for (int i = 0; i < 3; i++) assert(unused.count(slots[i]) == 1);
    // 3 slots recycled, 2 consumed → net free grows by 3.
    assert(client.mempool_free() == free_before + 3);

    // Full sequence now lands to 9; the two new blocks used slots[3],slots[4].
    auto q = client.query(hs.data(), 9, COMPONENT_MASK_FULL);
    assert(q.common_hit == 9);
    assert(hit_slots(q)[7] == slots[3]);
    assert(hit_slots(q)[8] == slots[4]);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
// (e) P1 regression: leading-skip slots must survive a split that then fails.
// stale start (start < real matched) records leading redundant slots, then a
// partial-node split hits DATA_POOL_FULL. The abort path must APPEND the
// consumable slots to unused_slots, not ASSIGN over the leading ones — else the
// skip slots vanish from unused_slots and auto_recycle leaks them.
static void test_start_skip_split_datapool_full() {
    TEST_BEGIN("start: skip + partial-split DATA_POOL_FULL returns ALL slots");

    ShmConfig cfg;
    cfg.max_nodes = 100000;
    cfg.max_blocks = 8192;
    cfg.data_pool_ratio = 0.05f;
    cfg.background_evict_ratio = 0.0f;
    cfg.register_chunk_tokens = 128;      // [0..8) stays one node (no cs split)
    ServerLauncher launcher("/test_ti_30", cfg);
    RadixClient client("/test_ti_30");

    // Splittable pinned base A=[0..8).
    auto base = seq(30000, 8);
    auto ins_base = insert_pinned(client, base.data(), 8);
    assert(ins_base.error == InsertError::OK);
    assert(ins_base.unused_slots.empty());

    // Starve the data pool so the upcoming split's data_.alloc refuses.
    for (int i = 0; i < 8192; i++) {
        int64_t t = 40000 + i;
        auto h = RadixClient::compute_hashes(&t, 1, 1);
        auto ins = insert_pinned(client, h.data(), 1);
        if (ins.error != InsertError::OK) break;
    }

    // Divergent sibling sharing [0..4), diverging at 4. Real matched=4; pass a
    // STALE start=2 → skip=2. Supply 5 literal slots (2 leading + 3 consumable).
    // The mid-node split (matched=4 < node's 8 blocks) fires and fails.
    std::vector<int64_t> div(5);
    std::iota(div.begin(), div.end(), 30000);   // 30000..30003 shared
    div[4] = 88888;                              // diverge at block 4
    auto hdiv = RadixClient::compute_hashes(div.data(), 5, 1);
    int32_t slots[] = {700, 701, 702, 703, 704};

    auto ins = client.insert(hdiv.data(), 5, slots, 5, /*start=*/2,
                             /*auto_recycle=*/false);
    assert(ins.error == InsertError::DATA_POOL_FULL);
    // ALL 5 supplied slots must come back: 2 leading (skip) + 3 consumable.
    // Before the fix the leading 2 were clobbered by assign → only 3 returned.
    assert((int)ins.unused_slots.size() == 5);
    std::set<int32_t> got(ins.unused_slots.begin(), ins.unused_slots.end());
    for (int i = 0; i < 5; i++) assert(got.count(slots[i]) == 1);

    // Original prefix untouched — split rolled back cleanly.
    assert(client.query(base.data(), 8, COMPONENT_MASK_FULL, false, false, false)
               .common_hit == 8);

    expect_healthy(client, HEALTH_REF_LEAK);   // insert_pinned holds its pins for the test's lifetime
    TEST_PASS();
}

// ==================================================================
// (f) P2: multi-chunk partial success. cs forces several nodes; the node pool
// runs out mid-loop so the leading chunks land but the trailing chunk fails.
// The unlanded suffix must be returned in order and recycled by auto_recycle.
static void test_pool_full_partial_suffix() {
    TEST_BEGIN("start: partial multi-chunk success returns unlanded suffix in order");

    // cs=1 → one node per block. Size the node pool so only a few blocks fit,
    // forcing a mid-sequence NODE_POOL_FULL after some chunks have landed.
    ShmConfig cfg;
    cfg.max_nodes = 8;                  // root + a handful of leaves
    cfg.max_blocks = 4096;
    cfg.data_pool_ratio = 5;
    cfg.background_evict_ratio = 0.0f;
    cfg.register_chunk_tokens = 1;
    ServerLauncher launcher("/test_ti_31", cfg);
    RadixClient client("/test_ti_31");

    // Pin what lands so eviction can't recycle nodes to satisfy the tail.
    // A long sequence: only (max_nodes-1) single-block nodes can ever exist.
    const int N = 20;
    auto seqh = seq(50000, N);
    std::vector<int32_t> slots;
    client.allocate_slots(N, slots);
    assert((int)slots.size() == N);
    uint32_t free_after_alloc = client.mempool_free();

    auto ins = client.insert(seqh.data(), N, slots.data(), N, /*start=*/0,
                             /*auto_recycle=*/true);
    assert(ins.error == InsertError::NODE_POOL_FULL);

    // Some prefix landed; the rest is unlanded suffix. landed = N - unused.
    int landed = N - (int)ins.unused_slots.size();
    assert(landed > 0 && landed < N);
    // The unlanded suffix is exactly slots[landed:N], in order.
    for (int i = 0; i < (int)ins.unused_slots.size(); i++)
        assert(ins.unused_slots[i] == slots[landed + i]);
    // auto_recycle returned the unlanded suffix to the pool.
    assert(client.mempool_free() == free_after_alloc + (int)ins.unused_slots.size());

    // The landed prefix is queryable up to `landed`.
    assert(client.query(seqh.data(), landed, COMPONENT_MASK_FULL, false, false, false)
               .common_hit == landed);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
// (g) P2: auto_recycle actually recycles across surplus / full-match / leading-
// redundant, using REAL mempool slots (not literals) so mempool_free() is the
// ground truth for each path.
static void test_auto_recycle_real_slots() {
    TEST_BEGIN("auto_recycle: surplus, full-match, leading-redundant all recycle");

    ShmConfig cfg{4000, 64};
    cfg.background_evict_ratio = 0.0f;
    cfg.register_chunk_tokens = 1;
    ServerLauncher launcher("/test_ti_32", cfg);
    RadixClient client("/test_ti_32");

    // --- surplus: supply 6 real slots for a 3-block fresh insert ---
    auto h = seq(60000, 3);
    std::vector<int32_t> s6;
    client.allocate_slots(6, s6);
    uint32_t free_pre = client.mempool_free();
    auto rs = client.insert(h.data(), 3, s6.data(), 6, /*start=*/0,
                            /*auto_recycle=*/true);
    assert(rs.error == InsertError::OK);
    assert((int)rs.unused_slots.size() == 3);            // trailing 3 surplus
    assert(client.mempool_free() == free_pre + 3);       // recycled the surplus

    // --- full-match: re-insert same prefix; all supplied slots redundant ---
    std::vector<int32_t> s3;
    client.allocate_slots(3, s3);
    free_pre = client.mempool_free();
    auto rf = client.insert(h.data(), 3, s3.data(), 3, /*start=*/0,
                            /*auto_recycle=*/true);
    assert(rf.error == InsertError::OK);
    assert((int)rf.unused_slots.size() == 3);            // all redundant
    assert(client.mempool_free() == free_pre + 3);       // all recycled back

    // --- shortfall + surplus: extend [3..5), supply 3 at start=3 (1 surplus) ---
    auto h5 = seq(60000, 5);          // shares [0..3), extends to 5
    std::vector<int32_t> s3b;
    client.allocate_slots(3, s3b);
    free_pre = client.mempool_free();
    // matched==3==start → no skip; 2 land ([3..5)), 1 surplus recycled.
    auto rl = client.insert(h5.data(), 5, s3b.data(), 3, /*start=*/3,
                            /*auto_recycle=*/true);
    assert(rl.error == InsertError::OK);
    assert((int)rl.unused_slots.size() == 1);
    assert(client.mempool_free() == free_pre + 1);
    auto q = client.query(h5.data(), 5, COMPONENT_MASK_FULL);
    assert(q.common_hit == 5);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
// (h) Combined: leading redundant AND trailing surplus in one insert. skip>0
// contributes leading slots; supply exceeds the shortfall so trailing slots are
// surplus. unused_slots must carry BOTH segments (leading first, then surplus).
static void test_skip_and_surplus_combined() {
    TEST_BEGIN("start: leading redundant + trailing surplus both in unused_slots");

    ShmConfig cfg{4000, 64};
    cfg.background_evict_ratio = 0.0f;
    cfg.register_chunk_tokens = 1;
    ServerLauncher launcher("/test_ti_33", cfg);
    RadixClient client("/test_ti_33");

    // Prefix P=[0..5). A later 7-block sequence shares [0..5).
    auto hp = seq(70000, 5);
    assert(insert_auto(client, hp.data(), 5).error == InsertError::OK);

    auto hs = seq(70000, 7);          // [0..7): matched will be 5
    int matched = (int)client.query(hs.data(), 7, COMPONENT_MASK_FULL, false, false, false)
                      .common_hit;
    assert(matched == 5);

    // Supply 6 REAL slots but pass STALE start=2 → skip = matched-start = 3
    // leading. Consumable = 6-3 = 3 for the 2-block shortfall [5..7) → 1 surplus.
    std::vector<int32_t> slots;
    client.allocate_slots(6, slots);
    assert((int)slots.size() == 6);
    uint32_t free_pre = client.mempool_free();

    auto ins = client.insert(hs.data(), 7, slots.data(), 6, /*start=*/2,
                             /*auto_recycle=*/true);
    assert(ins.error == InsertError::OK);
    // 3 leading (skip) + 1 trailing surplus = 4 unused; 2 landed for [5..7).
    assert((int)ins.unused_slots.size() == 4);
    // Leading segment first: slots[0..3). Trailing surplus: slots[5].
    assert(ins.unused_slots[0] == slots[0]);
    assert(ins.unused_slots[1] == slots[1]);
    assert(ins.unused_slots[2] == slots[2]);
    assert(ins.unused_slots[3] == slots[5]);
    // slots[3],slots[4] landed on [5],[6].
    assert(client.mempool_free() == free_pre + 4);      // 4 recycled

    auto q = client.query(hs.data(), 7, COMPONENT_MASK_FULL);
    assert(q.common_hit == 7);
    assert(hit_slots(q)[5] == slots[3]);
    assert(hit_slots(q)[6] == slots[4]);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
// (i) skip consumes ALL slots (matched>start AND num_slots<=matched-start): the
// sequence is still not fully matched (n>matched) but no slot is left for the
// suffix. Every supplied slot is leading-redundant → all unused + recycled, the
// unmatched suffix does NOT insert (new_slots capped to 0 → §1262 branch). Also
// exercises the skip>ext_num_slots cap at §1234.
static void test_start_skip_consumes_all_slots() {
    TEST_BEGIN("start: skip eats all slots, unmatched suffix not inserted");

    ShmConfig cfg{4000, 64};
    cfg.background_evict_ratio = 0.0f;
    cfg.register_chunk_tokens = 1;
    ServerLauncher launcher("/test_ti_34", cfg);
    RadixClient client("/test_ti_34");

    // Prefix P=[0..5). A 7-block sequence shares [0..5): real matched=5.
    auto hp = seq(70000, 5);
    assert(insert_auto(client, hp.data(), 5).error == InsertError::OK);
    uint32_t blocks_before = client.total_blocks();

    auto hs = seq(70000, 7);          // [0..7)
    assert(client.query(hs.data(), 7, COMPONENT_MASK_FULL, false, false, false).common_hit == 5);

    // STALE start=1 → skip demand = matched-start = 4. Supply only 4 real slots
    // (num_slots == matched-start): skip caps to 4, eats every slot, cons_num=0.
    // hashes[5..7) stay uninserted because no slot remains for them.
    std::vector<int32_t> slots;
    client.allocate_slots(4, slots);
    assert((int)slots.size() == 4);
    uint32_t free_pre = client.mempool_free();

    auto ins = client.insert(hs.data(), 7, slots.data(), 4, /*start=*/1,
                             /*auto_recycle=*/true);
    assert(ins.error == InsertError::OK);
    // All 4 slots are leading-redundant → all unused.
    assert((int)ins.unused_slots.size() == 4);
    std::set<int32_t> got(ins.unused_slots.begin(), ins.unused_slots.end());
    for (int i = 0; i < 4; i++) assert(got.count(slots[i]) == 1);
    // All 4 recycled; nothing consumed.
    assert(client.mempool_free() == free_pre + 4);
    // Suffix [5..7) never landed: tree unchanged, query still stops at 5.
    assert(client.total_blocks() == blocks_before);
    assert(client.query(hs.data(), 7, COMPONENT_MASK_FULL, false, false, false).common_hit == 5);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
// (j) P2: DATA_POOL_FULL in the APPEND loop (§1295) — a cleanup path DISTINCT
// from the NODE-pool path (#31): here nodes_.alloc succeeded, so the abort must
// nodes_.free(new_id) before break. #31 leaves that node allocated; this one
// must not leak it. With the data pool pinned-full the first append iteration's
// data_.alloc fails (evict can't reclaim — all pinned) → break at slot_idx=0, so
// the WHOLE supply is unlanded suffix and must return in order + auto-recycle.
// (Mixed partial-landing is not asserted here: pinning-full is the only way to
// make the append fail deterministically; the unlanded-suffix invariant is
// exercised across all landed values by #31's node-pool partial case.)
static void test_data_pool_full_append_suffix() {
    TEST_BEGIN("start: DATA_POOL_FULL in append frees its node + returns all suffix");

    // Tiny data pool (sole bottleneck), roomy node/mempool. cs=1 → append loop.
    ShmConfig cfg;
    cfg.max_nodes = 100000;
    cfg.max_blocks = 8192;
    cfg.data_pool_ratio = 0.05f;
    cfg.background_evict_ratio = 0.0f;
    cfg.register_chunk_tokens = 1;
    ServerLauncher launcher("/test_ti_35", cfg);
    RadixClient client("/test_ti_35");

    // Pin-fill the data pool so nothing is reclaimable during the next insert.
    uint32_t nodes_full = 0;
    for (int i = 0; i < 8192; i++) {
        int64_t t = 20000 + i;
        auto h = RadixClient::compute_hashes(&t, 1, 1);
        auto ins = insert_pinned(client, h.data(), 1);
        if (ins.error == InsertError::DATA_POOL_FULL) { nodes_full = client.total_radix_nodes(); break; }
        assert(ins.error == InsertError::OK);
    }
    assert(nodes_full > 0);

    const int N = 8;
    auto seqh = seq(90000, N);
    std::vector<int32_t> slots;
    client.allocate_slots(N, slots);         // mempool has room (8192 blocks)
    assert((int)slots.size() == N);
    uint32_t free_after_alloc = client.mempool_free();

    auto ins = client.insert(seqh.data(), N, slots.data(), N, /*start=*/0,
                             /*auto_recycle=*/true);
    assert(ins.error == InsertError::DATA_POOL_FULL);

    // Nothing landed (pool pinned-full): all N slots are unlanded suffix, in order.
    int landed = N - (int)ins.unused_slots.size();
    assert(landed == 0);
    for (int i = 0; i < N; i++)
        assert(ins.unused_slots[i] == slots[i]);
    // auto_recycle returned the whole supply; mempool restored.
    assert(client.mempool_free() == free_after_alloc + N);
    // The failed insert allocated then freed its node — no node leak, tree intact.
    assert(client.total_radix_nodes() == nodes_full);
    assert(client.query(seqh.data(), N, COMPONENT_MASK_FULL, false, false, false).common_hit == 0);

    expect_healthy(client, HEALTH_REF_LEAK);   // insert_pinned holds its pins for the test's lifetime
    TEST_PASS();
}

// ==================================================================
int main() {
    printf("=== test_tree_insert ===\n\n");
    // Mempool-backed inserts + split + chunk-align + pool-full.
    test_basic_insert_match();
    test_prefix_sharing_split();
    test_multiple_inserts_multiple_branches();
    test_duplicate_insert_idempotent();
    test_insert_chunked_local();
    test_chunk_aligned_boundaries();
    test_chunk_align_non_cs_start();
    test_split_prefix_is_routable();
    test_node_pool_full_error();
    test_ht_pool_full_error();
    test_insert_evict_last_child_keeps_leaf_list_sane();
    // Caller-supplied slot inserts.
    test_exact_supply();
    test_supply_too_few();
    test_supply_too_many();
    test_prefix_match_consumes_suffix_only();
    test_full_match_consumes_nothing();
    test_prefix_slots_full_match();
    test_mix_with_regular_insert();
    // Boundary / error-path branches.
    test_empty_input();
    test_data_pool_full_loop();
    test_data_pool_full_split();
    test_split_zero_new_slots_reparents();
    // Auto-merge branches (FULL pop / multi-child guard / 4096 partial cap).
    test_automerge_full_progressive_collapse();
    test_automerge_multichild_guard();
    test_automerge_cap_4096_boundary();
    test_automerge_keeps_cs_anchor();
    // start-param positional binding: sequential, idempotent, reject, skip.
    test_start_100x_single_block();
    test_start_10x_long_sequence_idempotent();
    test_start_reject_evicted_prefix();
    test_start_skip_leading_redundant();
    // unused_slots edge coverage: skip+split-fail (P1 regression), partial
    // multi-chunk suffix, auto_recycle over real mempool slots, skip+surplus.
    test_start_skip_split_datapool_full();
    test_pool_full_partial_suffix();
    test_auto_recycle_real_slots();
    test_skip_and_surplus_combined();
    // skip eats all slots (suffix unfunded); DATA_POOL_FULL append-path suffix.
    test_start_skip_consumes_all_slots();
    test_data_pool_full_append_suffix();
    return tests_report();
}
