// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dedup_writer.h"
#include "radix_tree.h"
#include "test_utils.h"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <future>
#include <sys/mman.h>
#include <thread>
#include <vector>

using namespace radixshmem;

static const uint32_t BIG[NUM_COMPONENT_TYPES] = {1024, 1024, 64};

static std::vector<uint64_t> seq(uint64_t base, int n) {
    std::vector<uint64_t> h(n);
    for (int i = 0; i < n; i++) h[i] = base + (uint64_t)i;
    return h;
}

static std::vector<uint64_t> cat(const std::vector<uint64_t>& a, int n, const std::vector<uint64_t>& b) {
    std::vector<uint64_t> h(a.begin(), a.begin() + n);
    h.insert(h.end(), b.begin(), b.end());
    return h;
}

static ShmConfig cfg(bool aux) {
    ShmConfig c;
    c.max_nodes = 1024;
    c.max_blocks = 256;
    c.register_chunk_tokens = 4;
    c.hugepage_path = "";
    if (aux) {
        c.component_mask = COMPONENT_MASK_ALL;
        c.swa_window_blocks = 4;
        c.mamba_state_bytes = 256;
        c.swa_max_blocks = 256;
        c.mamba_max_slots = 64;
    }
    return c;
}

static RadixShmemTree fresh(const char* name, bool aux = false) {
    shm_unlink(name);
    return RadixShmemTree::create(name, cfg(aux));
}

static int match(RadixShmemTree& t, const std::vector<uint64_t>& h) {
    return (int)t.query_components(h.data(), (int)h.size(), COMPONENT_MASK_FULL, false, false)
        .topology_match_length;
}

static void put(RadixShmemTree& t, const std::vector<uint64_t>& h, int n = -1) {
    n = n < 0 ? (int)h.size() : n;
    const int m = (int)t.query_components(h.data(), n, COMPONENT_MASK_FULL, false, false)
                      .topology_match_length;
    std::vector<int32_t> s;
    assert(t.allocate_slots(n - m, s) == n - m);
    assert(t.insert(h.data(), n, s.data(), n - m, m).error == InsertError::OK);
}

static void put_aux(RadixShmemTree& t, const std::vector<uint64_t>& h, int n, ComponentType k,
                    int count) {
    std::vector<int32_t> s;
    assert(t.allocate_slots(count, s, k) == count);
    assert(t.insert(h.data(), n, s.data(), count, 0, true, k).error == InsertError::OK);
}

static ShmHeader* hdr_of(RadixShmemTree& t) { return reinterpret_cast<ShmHeader*>(t.shm_base()); }

static TreeNode* node_of(RadixShmemTree& t, uint32_t id) {
    return reinterpret_cast<TreeNode*>((char*)t.shm_base() + hdr_of(t)->node_pool_offset) + id;
}

static uint32_t node_at(RadixShmemTree& t, const std::vector<uint64_t>& h) {
    return t.get_node_boundaries(h.data(), (int)h.size()).back().tree_node_id;
}

static uint64_t leaf_ts(RadixShmemTree& t, const std::vector<uint64_t>& h) {
    return node_of(t, node_at(t, h))->last_access_time.load();
}

static SlotLruEntry* lru_of(RadixShmemTree& t, bool swa) {
    ShmHeader* h = hdr_of(t);
    return reinterpret_cast<SlotLruEntry*>((char*)t.shm_base() +
                                           (swa ? h->swa_lru_offset : h->mamba_lru_offset));
}

static const int32_t* swa_lane(RadixShmemTree& t, uint32_t id) {
    TreeNode* n = node_of(t, id);
    return reinterpret_cast<const int32_t*>((char*)t.shm_base() + n->data_offset +
                                            12u * n->num_blocks);
}

// SWA cell of block `pos` on path h.
static int32_t swa_cell(RadixShmemTree& t, const std::vector<uint64_t>& h, int pos) {
    for (const NodeBoundary& b : t.get_node_boundaries(h.data(), pos + 1))
        if (pos < b.offset + b.num_blocks) return swa_lane(t, b.tree_node_id)[pos - b.offset];
    return INVALID_COMPONENT_SLOT;
}

static void check_lru_order(RadixShmemTree& t, bool swa) {
    ShmHeader* h = hdr_of(t);
    SlotLruEntry* e = lru_of(t, swa);
    const AuxListAnchors& a = swa ? h->swa_list : h->mamba_list;
    uint64_t ts = 0;
    for (uint32_t cur = a.head; cur != INVALID_SLOT_LINK; cur = e[cur].next) {
        assert(e[cur].ts >= ts);
        ts = e[cur].ts;
    }
}

static void check_leaves(RadixShmemTree& t) {
    ShmHeader* h = hdr_of(t);
    uint64_t ts = 0;
    for (uint32_t id = h->leaf_list_head; id != INVALID_NODE_ID; id = node_of(t, id)->leaf_next) {
        assert(node_of(t, id)->children_count == 0);
        assert(node_of(t, id)->last_access_time.load() >= ts);
        ts = node_of(t, id)->last_access_time.load();
    }
}

// Stands in for the RDMA copy: slots of the puller's pools replace the source's.
static void land(RadixShmemTree& a, std::vector<MovedPath>& moved) {
    for (MovedPath& p : moved) {
        std::vector<int32_t> s;
        if (!p.full.empty()) {
            assert(a.allocate_slots((int)p.full.size(), s) == (int)p.full.size());
            p.full = s;
        }
        for (MovedCell& c : p.cells) {
            assert(a.allocate_slots(1, s, c.component) == 1);
            c.slot = s[0];
        }
    }
}

// Adopts like RadixClient::dedup_move, which recycles what is reported unused.
static void adopt_all(RadixShmemTree& a, const std::vector<uint64_t>& req,
                      const std::vector<MovedPath>& moved, std::vector<int32_t> unused[3]) {
    for (const MovedPath& p : moved) {
        std::vector<uint64_t> path = cat(req, p.base, p.hashes);
        a.adopt(path.data(), (int)path.size(), p, unused);
    }
    for (int k = 0; k < NUM_COMPONENT_TYPES; k++)
        if (!unused[k].empty())
            a.recycle_slots(unused[k].data(), (int)unused[k].size(), (ComponentType)k);
}

static void sleep_ms(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

static void test_tail_and_forks() {
    TEST_BEGIN("tail and short side branches move, a long branch stops the climb");
    auto b = fresh("/tdm_b1"), a = fresh("/tdm_a1");
    auto P = seq(0x1000, 16);
    auto tail = cat(P, 16, seq(0x5000, 6));
    auto s1 = cat(P, 8, seq(0x2000, 4));
    auto s2 = cat(P, 12, seq(0x3000, 2));
    auto s3 = cat(P, 4, seq(0x4000, 20));
    put(b, tail);
    put(b, s1);
    put(b, s2);
    put(b, s3);
    const uint64_t t_tail = leaf_ts(b, tail), t_s1 = leaf_ts(b, s1);
    sleep_ms(5);
    put(a, P);

    DedupPlan plan;
    assert(b.plan_dedup(P.data(), 16, 256, 16, BIG, plan));
    assert(plan.moved.size() == 3);
    const MovedPath &mt = plan.moved[0], &m2 = plan.moved[1], &m1 = plan.moved[2];
    assert(mt.base == 16 && mt.hashes == seq(0x5000, 6) && mt.full_from == 16 && mt.full.size() == 6);
    assert(m2.base == 12 && m2.hashes == seq(0x3000, 2) && m2.full_from == 12 && m2.full.size() == 2);
    assert(m1.base == 8 && m1.hashes == seq(0x2000, 4) && m1.full_from == 8 && m1.full.size() == 4);
    assert(mt.cells.empty() && mt.full_age_ns.size() == 6);
    assert(b.dedup_prefix(P.data(), 16) == 0);   // pinned
    assert(b.finish_dedup(plan, true) == 12 + 6 + 2 + 4);
    assert(match(b, P) == 4 && match(b, s3) == 24 && b.total_blocks() == 24);
    check_leaves(b);
    expect_healthy(b);

    std::vector<int32_t> unused[3];
    land(a, plan.moved);
    adopt_all(a, P, plan.moved, unused);
    assert(unused[0].empty() && unused[1].empty() && unused[2].empty());
    assert(match(a, tail) == 22 && match(a, s1) == 12 && match(a, s2) == 14);
    const uint64_t at = leaf_ts(a, tail), a1 = leaf_ts(a, s1);
    assert(at >= t_tail && at - t_tail < 20000000ull && a1 >= t_s1 && a1 - t_s1 < 20000000ull);
    assert(at < leaf_ts(a, P));
    check_leaves(a);
    expect_healthy(a);
    b.destroy();
    a.destroy();
    TEST_PASS();
}

static void test_shapes_and_caps() {
    TEST_BEGIN("forked tail and forked side subtree move whole; caps stop them");
    auto b = fresh("/tdm_b2"), a = fresh("/tdm_a2");
    auto P = seq(0x1000, 8);
    auto t1 = cat(P, 8, seq(0x5000, 4)), t2 = cat(P, 8, seq(0x6000, 3));
    auto x = cat(P, 4, seq(0x7000, 4));
    auto y = cat(x, 6, seq(0x8000, 3));
    for (auto* h : {&t1, &t2, &x, &y}) put(b, *h);
    put(a, P);

    DedupPlan plan;
    assert(!b.plan_dedup(P.data(), 8, 6, 16, BIG, plan));
    assert(b.plan_dedup(P.data(), 8, 7, 6, BIG, plan));
    assert(plan.moved.size() == 2 && plan.moved[0].base == 8 && plan.moved[1].base == 8);
    assert(b.finish_dedup(plan, true) == 4 + 7);
    assert(match(b, P) == 4 && match(b, y) == 9);

    put(b, t1);
    put(b, t2);
    assert(b.plan_dedup(P.data(), 8, 7, 7, BIG, plan));
    assert(plan.moved.size() == 4);
    int full = 0, from_fork = 0;
    for (const MovedPath& p : plan.moved) {
        full += (int)p.full.size();
        if (p.base == 4 && p.full_from == 6) from_fork++;
    }
    assert(full == 7 + 7 && from_fork == 1);
    assert(b.finish_dedup(plan, true) == 8 + 7 + 7 && b.total_blocks() == 0);
    std::vector<int32_t> unused[3];
    land(a, plan.moved);
    adopt_all(a, P, plan.moved, unused);
    assert(match(a, t1) == 12 && match(a, t2) == 11 && match(a, x) == 8 && match(a, y) == 9);
    assert(unused[0].empty());
    check_leaves(a);
    expect_healthy(a);
    expect_healthy(b);
    b.destroy();
    a.destroy();
    TEST_PASS();
}

static void test_pins_and_changes() {
    TEST_BEGIN("pins stop the plan; a change during the copy keeps everything");
    auto b = fresh("/tdm_b3");
    auto P = seq(0x1000, 16);
    auto s1 = cat(P, 8, seq(0x2000, 4));
    auto s2 = cat(P, 12, seq(0x3000, 2));
    put(b, P);
    put(b, s1);
    put(b, s2);
    DedupPlan plan;

    const uint32_t side = node_at(b, s1);
    node_of(b, side)->local_ref_count.fetch_add(1);
    assert(b.plan_dedup(P.data(), 16, 256, 16, BIG, plan));
    assert(plan.moved.size() == 1 && plan.moved[0].base == 12);
    assert(b.finish_dedup(plan, false) == 0);
    node_of(b, side)->local_ref_count.fetch_sub(1);
    expect_healthy(b);

    assert(b.plan_dedup(P.data(), 16, 256, 16, BIG, plan));
    put(b, cat(P, 12, seq(0x9000, 2)));
    assert(b.finish_dedup(plan, true) == 0);
    expect_healthy(b);

    assert(b.plan_dedup(P.data(), 16, 256, 16, BIG, plan));
    put(b, cat(P, 14, seq(0xA000, 2)));   // splits a pinned node
    assert(b.finish_dedup(plan, true) == 0);
    expect_healthy(b);
    assert(b.total_blocks() == 16 + 4 + 2 + 2 + 2);

    assert(b.plan_dedup(P.data(), 16, 256, 16, BIG, plan));
    assert(b.finish_dedup(plan, true) == 26 && b.total_blocks() == 0);
    expect_healthy(b);
    check_leaves(b);
    b.destroy();
    TEST_PASS();
}

static void test_cells() {
    TEST_BEGIN("SWA cells and Mamba checkpoints of the dropped path move; ages kept");
    auto b = fresh("/tdm_b4", true), a = fresh("/tdm_a4", true);
    auto P = seq(0x1000, 12);
    put(b, P);
    put_aux(b, P, 4, ComponentType::SWA, 4);
    put_aux(b, P, 8, ComponentType::SWA, 4);
    put_aux(b, P, 12, ComponentType::SWA, 4);
    put_aux(b, P, 6, ComponentType::MAMBA, 1);
    put_aux(b, P, 12, ComponentType::MAMBA, 1);
    const uint64_t t_cell = lru_of(b, true)[swa_cell(b, P, 2)].ts;
    sleep_ms(5);
    put(a, P, 10);
    put_aux(a, P, 8, ComponentType::SWA, 4);
    const int32_t own = swa_cell(a, P, 5);
    const uint64_t own_ts = lru_of(a, true)[own].ts;

    DedupPlan plan;
    const uint32_t small[3] = {1024, 5, 64};
    assert(b.plan_dedup(P.data(), 10, 256, 16, small, plan));
    assert(b.finish_dedup(plan, false) == 0);
    const uint32_t none[3] = {0, 1024, 64};
    assert(!b.plan_dedup(P.data(), 10, 256, 16, none, plan));

    assert(b.plan_dedup(P.data(), 10, 256, 16, BIG, plan));
    assert(plan.moved.size() == 2);
    const MovedPath &tail = plan.moved[0], &path = plan.moved[1];
    assert(tail.base == 10 && tail.hashes.size() == 2 && tail.full.size() == 2);
    assert(tail.cells.size() == 3 && tail.cells.back().component == ComponentType::MAMBA &&
           tail.cells.back().pos == 11);
    assert(path.base == 10 && path.hashes.empty() && path.full.empty() && path.cells.size() == 11);
    assert(b.finish_dedup(plan, true) == 12 && b.total_blocks() == 0);
    expect_healthy(b);

    std::vector<int32_t> unused[3];
    land(a, plan.moved);
    adopt_all(a, P, plan.moved, unused);
    assert(unused[0].empty() && unused[1].size() == 4 && unused[2].empty());
    assert(swa_cell(a, P, 5) == own && lru_of(a, true)[own].ts == own_ts);
    const uint64_t moved_ts = lru_of(a, true)[swa_cell(a, P, 2)].ts;
    assert(moved_ts >= t_cell && moved_ts - t_cell < 20000000ull);
    TreeComponentQueryResult q = a.query_components(P.data(), 12, COMPONENT_MASK_ALL, false, false);
    assert(q.common_hit == 12);
    q = a.query_components(P.data(), 6, COMPONENT_MASK_ALL, false, false);
    assert(q.common_hit == 6);
    check_lru_order(a, true);
    check_lru_order(a, false);
    check_leaves(a);
    expect_healthy(a);
    b.destroy();
    a.destroy();
    TEST_PASS();
}

static void test_landing_alloc() {
    TEST_BEGIN("landing slots come from free slots only");
    auto t = fresh("/tdm_alloc");
    put(t, seq(0x1000, 250));
    std::vector<int32_t> s;
    assert(t.allocate_slots(64, s, ComponentType::FULL, false) == 6 && s.size() == 6);
    assert(t.total_blocks() == 250);
    assert(t.allocate_slots(4, s, ComponentType::FULL, false) == 0 && s.empty());
    t.destroy();
    TEST_PASS();
}

static void test_plan_edges() {
    TEST_BEGIN("plan: no match, reader pin on the path, pinned tail or bottom, nested forks");
    auto b = fresh("/tdm_b6"), a = fresh("/tdm_a6");
    DedupPlan plan;
    auto P = seq(0x1000, 16);
    assert(!b.plan_dedup(P.data(), 0, 256, 16, BIG, plan));
    assert(!b.plan_dedup(nullptr, 4, 256, 16, BIG, plan));
    assert(!b.plan_dedup(P.data(), 16, 256, 16, BIG, plan));

    put(b, P);
    put(b, cat(P, 8, seq(0x2000, 4)));
    TreeComponentQueryResult q = b.query_components(P.data(), 6, COMPONENT_MASK_FULL, true, false);
    assert(b.plan_dedup(P.data(), 16, 256, 16, BIG, plan) && plan.moved.empty());
    assert(b.finish_dedup(plan, true) == 8 && match(b, P) == 8);
    assert(b.dec_ref(P.data(), 6, std::move(q.pin)));
    expect_healthy(b);

    auto Q = seq(0x3000, 8);
    auto tail = cat(Q, 8, seq(0x4000, 4));
    put(b, tail);
    TreeNode* tn = node_of(b, node_at(b, tail));
    tn->local_ref_count.fetch_add(1);
    assert(!b.plan_dedup(Q.data(), 8, 256, 16, BIG, plan));
    assert(!b.plan_dedup(tail.data(), 12, 256, 16, BIG, plan));
    tn->local_ref_count.fetch_sub(1);
    expect_healthy(b);

    auto R = seq(0x5000, 8);
    auto x = cat(R, 4, seq(0x6000, 4));
    auto y = cat(x, 6, seq(0x7000, 2));
    auto z = cat(y, 7, seq(0x8000, 2));
    for (auto* h : {&R, &x, &y, &z}) put(b, *h);
    put(a, R);
    assert(b.plan_dedup(R.data(), 8, 256, 16, BIG, plan));
    int full = 0;
    for (const MovedPath& p : plan.moved) full += (int)p.full.size();
    assert(plan.moved.size() == 3 && full == 8);
    assert(b.finish_dedup(plan, true) == 8 + 8);
    std::vector<int32_t> unused[3];
    land(a, plan.moved);
    adopt_all(a, R, plan.moved, unused);
    assert(match(a, x) == 8 && match(a, y) == 8 && match(a, z) == 9 && unused[0].empty());
    expect_healthy(a);
    expect_healthy(b);
    b.destroy();
    a.destroy();
    TEST_PASS();
}

static void test_cells_edges() {
    TEST_BEGIN("cells: a checkpoint at the match END, cells that do not fit stop the bottom");
    auto b = fresh("/tdm_b7", true);
    auto P = seq(0x1000, 8);
    put(b, P);
    put_aux(b, P, 8, ComponentType::SWA, 4);
    put_aux(b, P, 8, ComponentType::MAMBA, 1);
    DedupPlan plan;
    const uint32_t three[3] = {1024, 3, 64};
    assert(!b.plan_dedup(P.data(), 8, 256, 16, three, plan));
    assert(b.plan_dedup(P.data(), 8, 256, 16, BIG, plan));
    assert(plan.moved.size() == 1 && plan.moved[0].base == 8 && plan.moved[0].cells.size() == 5);
    const MovedCell& ck = plan.moved[0].cells.back();
    assert(ck.component == ComponentType::MAMBA && ck.pos == 7);
    assert(b.finish_dedup(plan, true) == 8);
    expect_healthy(b);
    b.destroy();
    TEST_PASS();
}

static void test_finish_sees_changes() {
    TEST_BEGIN("finish keeps the subtree on a new cell, checkpoint or reader pin, and after reset()");
    auto b = fresh("/tdm_b8", true);
    auto P = seq(0x1000, 8);
    DedupPlan plan;
    put(b, P);
    put_aux(b, P, 4, ComponentType::SWA, 4);
    assert(b.plan_dedup(P.data(), 8, 256, 16, BIG, plan));
    put_aux(b, P, 8, ComponentType::SWA, 4);
    assert(b.finish_dedup(plan, true) == 0);
    expect_healthy(b);

    assert(b.plan_dedup(P.data(), 8, 256, 16, BIG, plan));
    put_aux(b, P, 8, ComponentType::MAMBA, 1);
    assert(b.finish_dedup(plan, true) == 0);
    expect_healthy(b);

    assert(b.plan_dedup(P.data(), 8, 256, 16, BIG, plan));
    TreeComponentQueryResult q = b.query_components(P.data(), 8, COMPONENT_MASK_FULL, true, false);
    assert(b.finish_dedup(plan, true) == 0);
    assert(b.dec_ref(P.data(), 8, std::move(q.pin)));
    expect_healthy(b);

    assert(b.plan_dedup(P.data(), 8, 256, 16, BIG, plan));
    b.reset();
    assert(b.finish_dedup(plan, true) == -1 && b.total_blocks() == 0);
    expect_healthy(b);
    put(b, P);
    assert(b.plan_dedup(P.data(), 8, 256, 16, BIG, plan) && b.finish_dedup(plan, true) == 8);
    expect_healthy(b);
    b.destroy();
    TEST_PASS();
}

static void test_adopt_edges() {
    TEST_BEGIN("adopt keeps what is there, untouched, and returns slots it cannot place");
    auto b = fresh("/tdm_b9", true), a = fresh("/tdm_a9", true);
    auto P = seq(0x1000, 8);
    auto tail = cat(P, 8, seq(0x5000, 4));
    put(b, tail);
    put_aux(b, tail, 12, ComponentType::MAMBA, 1);
    put(a, tail, 10);
    put_aux(a, tail, 12 - 2, ComponentType::MAMBA, 1);
    const uint64_t before = leaf_ts(a, cat(P, 8, seq(0x5000, 2)));
    sleep_ms(2);
    DedupPlan plan;
    assert(b.plan_dedup(P.data(), 8, 256, 16, BIG, plan));
    assert(b.finish_dedup(plan, true) == 12);
    std::vector<int32_t> unused[3];
    land(a, plan.moved);
    adopt_all(a, P, plan.moved, unused);
    assert(match(a, tail) == 12 && unused[0].size() == 2 && unused[2].empty());
    assert(leaf_ts(a, cat(P, 8, seq(0x5000, 2))) == before);
    expect_healthy(a);

    auto c = fresh("/tdm_c9", true);
    for (auto& u : unused) u.clear();
    land(c, plan.moved);
    adopt_all(c, P, plan.moved, unused);
    assert(c.total_blocks() == 0 && unused[0].size() == 4 && unused[2].size() == 1);
    MovedPath far;
    far.base = 8;
    far.full_from = 8;
    std::vector<int32_t> s;
    assert(a.allocate_slots(1, s, ComponentType::MAMBA) == 1);
    far.cells.push_back({ComponentType::MAMBA, 11, s[0], 0});
    assert(a.allocate_slots(1, s, ComponentType::SWA) == 1);
    far.cells.push_back({ComponentType::SWA, 20, s[0], 0});
    for (auto& u : unused) u.clear();
    adopt_all(a, tail, {far}, unused);
    assert(unused[1].size() == 1 && unused[2].size() == 1);
    expect_healthy(a);
    expect_healthy(c);
    a.destroy();
    b.destroy();
    c.destroy();
    TEST_PASS();
}

static void test_slot_lru_by_ts() {
    TEST_BEGIN("SlotLru::insert_by_ts: by time, into an empty list, past the step limit at the head");
    std::vector<SlotLruEntry> e(16);
    AuxListAnchors anchors{INVALID_SLOT_LINK, INVALID_SLOT_LINK};
    SlotLru::init_region(e.data(), 16);
    SlotLru lru(e.data(), 16, &anchors);
    lru.insert_by_ts(0, 1, 0, 50, 8);
    assert(anchors.head == 0 && anchors.tail == 0);
    for (int i = 1; i <= 9; i++) lru.push_tail(i, 1, 0, 50 + 10 * i);
    lru.insert_by_ts(12, 1, 0, 75, 100);
    assert(e[12].prev == 2 && e[12].next == 3 && e[2].next == 12 && e[3].prev == 12);
    lru.insert_by_ts(13, 1, 0, 10, 3);
    assert(anchors.head == 13 && e[13].prev == INVALID_SLOT_LINK && e[13].next == 0);
    lru.insert_by_ts(14, 1, 0, 200, 0);
    assert(anchors.tail == 14 && e[9].next == 14);
    std::vector<int32_t> order;
    for (uint32_t cur = anchors.head; cur != INVALID_SLOT_LINK; cur = e[cur].next) order.push_back((int32_t)cur);
    assert(order == (std::vector<int32_t>{13, 0, 1, 2, 12, 3, 4, 5, 6, 7, 8, 9, 14}));
    TEST_PASS();
}

static void test_landing_pools() {
    TEST_BEGIN("landing slots: a missing pool gives none, more than the pool gives the free ones");
    auto t = fresh("/tdm_pools");
    std::vector<int32_t> s;
    assert(t.allocate_slots(8, s, ComponentType::SWA, false) == 0 && s.empty());
    assert(t.allocate_slots(1000, s, ComponentType::FULL, false) == 256 && s.size() == 256);
    t.recycle_slots(s.data(), (int)s.size());
    t.destroy();
    TEST_PASS();
}

static void test_write_queue() {
    TEST_BEGIN("DedupWriteQueue: inactive, unclaimed, claimed but unfinished, finished, switched off");
    DedupWriteQueue q;
    std::vector<int32_t> local[3] = {{1, 2}, {}, {}}, remote[3] = {{7, 8}, {}, {}};
    assert(!q.active() && q.write(3, local, remote, 50) == DedupWriter::FAILED);
    q.set_active(true);
    assert(q.write(3, local, remote, 10) == DedupWriter::FAILED);
    assert(q.pop(4, 0).empty());

    auto w = std::async(std::launch::async, [&] { return q.write(3, local, remote, 10); });
    std::vector<DedupWrite> got = q.pop(4, 1000);
    assert(got.size() == 1 && got[0].rank == 3 && got[0].local[0] == local[0] &&
           got[0].remote[0] == remote[0] && got[0].timeout_ms == 10);
    assert(w.get() == DedupWriter::LATE);
    q.complete(got[0].id, DedupWriter::OK);

    for (int r : {0, 1, 2, 7}) {
        auto f = std::async(std::launch::async, [&] { return q.write(1, local, remote, 2000); });
        got = q.pop(4, 1000);
        assert(got.size() == 1);
        q.complete(got[0].id, r);
        assert(f.get() == (r == 0 ? DedupWriter::OK : r == 1 ? DedupWriter::FAILED : DedupWriter::LATE));
    }

    auto f1 = std::async(std::launch::async, [&] { return q.write(1, local, remote, 5000); });
    got = q.pop(4, 1000);
    auto f2 = std::async(std::launch::async, [&] { return q.write(2, local, remote, 5000); });
    sleep_ms(50);
    q.set_active(false);
    assert(f1.get() == DedupWriter::LATE && f2.get() == DedupWriter::FAILED);
    assert(q.pop(4, 0).empty());
    TEST_PASS();
}

int main() {
    test_tail_and_forks();
    test_shapes_and_caps();
    test_pins_and_changes();
    test_cells();
    test_landing_alloc();
    test_plan_edges();
    test_cells_edges();
    test_finish_sees_changes();
    test_adopt_edges();
    test_slot_lru_by_ts();
    test_landing_pools();
    test_write_queue();
    return 0;
}
