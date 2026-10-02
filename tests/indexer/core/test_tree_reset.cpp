// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// test_tree_reset — RadixClient::reset() clears the node-local tree in place
// (empty tree + single root, all slots free, pools rebuilt) and leaves the
// region reusable without any reattach. Plus a standalone RhtShard::reset()
// check that a holder's shard is zeroed (RDMA builds only).

#include "test_utils.h"
#include "shmradix/radix_client.h"
#ifndef SHMRADIX_NO_RDMA
#include "rht_core.hpp"   // private module header: this test pokes RhtShard directly
#endif

#include <cassert>
#include <vector>

using namespace shmradix;

// Insert three overlapping prefixes so the tree has splits, several nodes, and
// a non-trivial mempool/data-pool footprint. Returns the fresh-tree baselines
// captured before any insert, for post-reset comparison.
struct Baseline {
    uint32_t nodes;
    uint64_t blocks;
    uint32_t mp_free;
    uint32_t mp_total;
};

static Baseline snapshot(RadixClient& c) {
    return Baseline{c.total_radix_nodes(), c.total_blocks(),
                    c.mempool_free(), c.mempool_total()};
}

static void populate(RadixClient& c) {
    int64_t a[] = {1, 2, 3, 4, 5};
    int64_t b[] = {1, 2, 3, 6, 7};
    int64_t d[] = {8, 9, 10};
    auto ha = RadixClient::compute_hashes(a, 5, 1);
    auto hb = RadixClient::compute_hashes(b, 5, 1);
    auto hd = RadixClient::compute_hashes(d, 3, 1);
    assert(insert_auto(c, ha.data(), 5).error == InsertError::OK);
    assert(insert_auto(c, hb.data(), 5).error == InsertError::OK);
    assert(insert_auto(c, hd.data(), 3).error == InsertError::OK);
}

// ==================================================================
static void test_reset_restores_empty_state() {
    TEST_BEGIN("reset clears tree back to fresh empty state");

    ServerLauncher launcher("/test_tr_01", default_cfg());
    RadixClient client("/test_tr_01");

    Baseline fresh = snapshot(client);
    assert(fresh.nodes == 1);       // root only
    assert(fresh.blocks == 0);
    assert(fresh.mp_free == fresh.mp_total);

    populate(client);
    assert(client.total_radix_nodes() > 1);
    assert(client.total_blocks() > 0);
    assert(client.mempool_free() < client.mempool_total());

    client.reset();

    // Back to the fresh baseline exactly.
    assert(client.total_radix_nodes() == fresh.nodes);
    assert(client.total_blocks() == fresh.blocks);
    assert(client.mempool_free() == fresh.mp_free);
    assert(client.mempool_total() == fresh.mp_total);
    assert(client.data_pool_utilization() == 0.0f);

    // Everything previously inserted now misses.
    int64_t a[] = {1, 2, 3, 4, 5};
    auto ha = RadixClient::compute_hashes(a, 5, 1);
    assert(client.query(ha.data(), 5, COMPONENT_MASK_FULL).common_hit == 0);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
static void test_reinsert_after_reset() {
    TEST_BEGIN("insert works again after reset (pools rebuilt)");

    ServerLauncher launcher("/test_tr_02", default_cfg());
    RadixClient client("/test_tr_02");

    populate(client);
    client.reset();

    // Re-insert the same batch: must behave like a first insert on an empty
    // tree — full miss prefix, fresh slots, queryable.
    int64_t a[] = {1, 2, 3, 4, 5};
    auto ha = RadixClient::compute_hashes(a, 5, 1);
    auto ins = insert_auto(client, ha.data(), 5);
    assert(ins.error == InsertError::OK);
    assert(ins.unused_slots.empty());          // fresh tree: all 5 landed
    assert(client.total_radix_nodes() == 2);   // root + one leaf

    auto r = client.query(ha.data(), 5, COMPONENT_MASK_FULL);
    assert(r.common_hit == 5);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
static void test_double_reset_idempotent() {
    TEST_BEGIN("reset is idempotent (double reset, and reset on empty)");

    ServerLauncher launcher("/test_tr_03", default_cfg());
    RadixClient client("/test_tr_03");

    Baseline fresh = snapshot(client);

    // Reset an already-empty tree — no-op-equivalent.
    client.reset();
    assert(client.total_radix_nodes() == fresh.nodes);
    assert(client.mempool_free() == fresh.mp_free);

    populate(client);
    client.reset();
    client.reset();   // second reset must not corrupt the rebuilt pools
    assert(client.total_radix_nodes() == fresh.nodes);
    assert(client.total_blocks() == fresh.blocks);
    assert(client.mempool_free() == fresh.mp_free);

    // Still fully usable.
    int64_t d[] = {42, 43};
    auto hd = RadixClient::compute_hashes(d, 2, 1);
    assert(insert_auto(client, hd.data(), 2).error == InsertError::OK);
    assert(client.query(hd.data(), 2, COMPONENT_MASK_FULL).common_hit == 2);

    expect_healthy(client);
    TEST_PASS();
}

// ==================================================================
static void test_reset_then_fill_and_reset_slots_recycled() {
    TEST_BEGIN("slots fully recycled across reset cycles");

    ShmConfig cfg{1000, 64};   // small mempool so exhaustion is observable
    ServerLauncher launcher("/test_tr_04", cfg);
    RadixClient client("/test_tr_04");

    uint32_t total = client.mempool_total();
    assert(total > 0);

    // Two fill/reset cycles: the second must start from a fully-free mempool,
    // proving no slots leaked across reset.
    for (int cycle = 0; cycle < 2; cycle++) {
        assert(client.mempool_free() == total);
        constexpr int N = 40;
        std::vector<int64_t> t(N);
        for (int i = 0; i < N; i++) t[i] = cycle * 1000 + i;
        auto h = RadixClient::compute_hashes(t.data(), N, 1);
        auto ins = insert_auto(client, h.data(), N);
        assert(ins.error == InsertError::OK);
        assert(client.mempool_free() < total);
        client.reset();
    }
    assert(client.mempool_free() == total);

    expect_healthy(client);
    TEST_PASS();
}

#ifndef SHMRADIX_NO_RDMA
// ==================================================================
// Standalone RhtShard::reset() — the holder-side memset that
// ClientCluster::reset_owned_rht_shard() drives. Verifies every slot is zeroed
// and the header/identity survives.
static void test_rht_shard_reset() {
    TEST_BEGIN("RhtShard::reset zeros all slots, keeps identity");

    const std::string name = "/test_tr_rht_shard";
    constexpr uint32_t capacity = 256;   // power of 2
    constexpr uint32_t W = 4;            // slots per bucket
    constexpr int shard_id = 2;
    constexpr int holder_rank = 1;

    RhtShard shard =
        RhtShard::create(name, capacity, W, shard_id, holder_rank);
    assert(shard.valid());
    assert(shard.capacity() == capacity);
    assert(shard.slots_per_bucket() == W);

    RhtSlot* buckets = shard.buckets();
    const size_t nslots = (size_t)capacity * W;
    auto all_zero = [&]() {
        for (size_t i = 0; i < nslots; i++)
            if (buckets[i].head != 0 || buckets[i].ext != 0) return false;
        return true;
    };

    // Freshly created → all zero.
    assert(all_zero());

    // Dirty every cell (both halves) with a non-zero pattern.
    for (size_t i = 0; i < nslots; i++) {
        buckets[i].head = 0xDEADBEEF00000000ULL | i;
        buckets[i].ext  = 0xCAFEBABE00000000ULL | i;
    }

    shard.reset();

    // All cells back to the empty sentinel (0); identity preserved.
    assert(all_zero());
    assert(shard.shard_id() == shard_id);
    assert(shard.holder_rank() == holder_rank);
    assert(shard.capacity() == capacity);
    assert(shard.slots_per_bucket() == W);

    // Idempotent.
    shard.reset();
    assert(all_zero());

    TEST_PASS();   // no radix tree in this case: nothing to audit
}
#endif // SHMRADIX_NO_RDMA

// A holder receives a REQUESTED capacity and must allocate the normalized
// bucket count (power of two, floor 16) so the engine's mask on every peer
// matches its MR extent. This pins the rule at the shard boundary.
static void test_rht_shard_capacity_normalized() {
    TEST_BEGIN("RhtShard::create normalizes requested capacity");

    struct Case { uint32_t requested, expected; };
    const Case cases[] = {
        {0, 16}, {1, 16}, {16, 16}, {17, 32}, {1000, 1024}, {4096, 4096},
    };
    int i = 0;
    for (const Case& c : cases) {
        const std::string name = "/test_tr_rht_cap" + std::to_string(i++);
        RhtShard shard = RhtShard::create(name, c.requested, /*W=*/2, 0, 0);
        assert(shard.valid());
        assert(shard.capacity() == c.expected);
        assert(rht_normalize_capacity(c.requested) == c.expected);
        // The region must actually hold capacity*W cells past slot[0].
        assert(shard.size() >= shard.data_offset() +
                                   (size_t)c.expected * 2 * sizeof(RhtSlot));
        // Attach sees the normalized value in the header, not the request.
        RhtShard peer = RhtShard::attach(name, /*timeout_sec=*/5);
        assert(peer.capacity() == c.expected);
    }

    TEST_PASS();
}

// ==================================================================
int main() {
    printf("=== test_tree_reset ===\n\n");
    test_reset_restores_empty_state();
    test_reinsert_after_reset();
    test_double_reset_idempotent();
    test_reset_then_fill_and_reset_slots_recycled();
#ifndef SHMRADIX_NO_RDMA
    test_rht_shard_reset();
    test_rht_shard_capacity_normalized();
#endif
    return tests_report();
}
