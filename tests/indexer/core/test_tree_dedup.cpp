// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radix_tree.h"
#include "test_utils.h"

#include <cassert>
#include <cstdint>
#include <random>
#include <sys/mman.h>
#include <utility>
#include <vector>

using namespace radixshmem;

static std::vector<uint64_t> seq(uint64_t base, int n) {
    std::vector<uint64_t> h(n);
    for (int i = 0; i < n; i++) h[i] = base + (uint64_t)i;
    return h;
}

static ShmConfig cfg(uint32_t max_blocks = 256) {
    ShmConfig c;
    c.max_nodes = 1024;
    c.max_blocks = max_blocks;
    c.register_chunk_tokens = 4;
    c.hugepage_path = "";
    return c;
}

static ShmConfig aux_cfg() {
    ShmConfig c = cfg(64);
    c.component_mask = COMPONENT_MASK_ALL;
    c.swa_window_blocks = 4;
    c.mamba_state_bytes = 256;
    c.swa_max_blocks = 64;
    c.mamba_max_slots = 16;
    return c;
}

static int match(RadixShmemTree& t, const std::vector<uint64_t>& h, int n) {
    return (int)t.query_components(h.data(), n, COMPONENT_MASK_FULL, false, false)
        .topology_match_length;
}

static void insert(RadixShmemTree& t, const std::vector<uint64_t>& h, int n,
                   ComponentType kind = ComponentType::FULL, int count = 0) {
    const int m = kind == ComponentType::FULL ? match(t, h, n) : 0;
    const int k = kind == ComponentType::FULL ? n - m : count;
    std::vector<int32_t> s;
    assert(t.allocate_slots(k, s, kind) == k);
    auto r = t.insert(h.data(), n, s.data(), k, kind == ComponentType::FULL ? m : 0, true, kind);
    assert(r.error == InsertError::OK);
}

static TreeNode* nodes_of(const RadixShmemTree& t, ShmHeader*& hdr) {
    auto* base = static_cast<char*>(t.shm_base());
    hdr = reinterpret_cast<ShmHeader*>(base);
    return reinterpret_cast<TreeNode*>(base + hdr->node_pool_offset);
}

static void check_leaves(const RadixShmemTree& t, bool by_time = true) {
    ShmHeader* hdr;
    TreeNode* nodes = nodes_of(t, hdr);
    uint32_t leaves = 0;
    for (uint32_t i = 0; i < hdr->node_pool_capacity; i++)
        if (i != hdr->root_node_id && nodes[i].num_blocks > 0 && nodes[i].children_count == 0)
            leaves++;
    uint32_t seen = 0, prev = INVALID_NODE_ID;
    uint64_t ts = 0;
    for (uint32_t id = hdr->leaf_list_head; id != INVALID_NODE_ID; id = nodes[id].leaf_next) {
        const TreeNode& nd = nodes[id];
        assert(++seen <= leaves);
        assert(nd.leaf_prev == prev && id != hdr->root_node_id && nd.num_blocks > 0 &&
               nd.children_count == 0);
        assert(!by_time || nd.last_access_time.load() >= ts);
        ts = nd.last_access_time.load();
        prev = id;
    }
    assert(seen == leaves && hdr->leaf_list_tail == prev);
}

static RadixShmemTree fresh(const char* name, const ShmConfig& c) {
    shm_unlink(name);
    return RadixShmemTree::create(name, c);
}

static void test_covered_chain() {
    TEST_BEGIN("a covered leaf chain goes up to the root");
    auto t = fresh("/ttd_chain", cfg());
    auto h = seq(0x1000, 12);
    insert(t, h, 4);
    insert(t, h, 8);
    insert(t, h, 12);
    std::vector<uint64_t> got;
    t.set_evict_callback([&](const std::vector<NodeHead>& v) {
        for (const NodeHead& e : v) got.push_back(e.first_hash);
    });
    auto longer = seq(0x1000, 16);
    assert(t.dedup_prefix(longer.data(), 16) == 12);
    assert(got == (std::vector<uint64_t>{h[8], h[4], h[0]}));
    assert(t.total_blocks() == 0 && t.mempool_free() == 256);
    check_leaves(t);

    insert(t, h, 12);
    assert(t.dedup_prefix(h.data(), 12) == 12 && t.total_blocks() == 0);
    check_leaves(t);
    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_not_covered() {
    TEST_BEGIN("a fork inside or below the match, a longer path, no match: kept");
    auto t = fresh("/ttd_kept", cfg());
    auto h = seq(0x2000, 12);
    auto g = h;
    g[2] = 0x2F02;
    insert(t, g, 4);
    assert(t.dedup_prefix(h.data(), 4) == 0 && match(t, g, 4) == 4);

    auto a = seq(0x3000, 12), b = a;
    b[8] = 0x3F08;
    insert(t, a, 12);
    assert(t.dedup_prefix(a.data(), 8) == 0);
    assert(t.dedup_prefix(b.data(), 12) == 0);
    assert(t.dedup_prefix(a.data(), 6) == 0);

    auto c = seq(0x4000, 6);
    insert(t, c, 6);
    assert(t.dedup_prefix(c.data(), 5) == 0 && match(t, c, 6) == 6);

    auto none = seq(0x5000, 8);
    assert(t.dedup_prefix(none.data(), 8) == 0);
    assert(t.dedup_prefix(nullptr, 0) == 0);
    assert(t.total_blocks() == 4 + 12 + 6);
    check_leaves(t);
    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_stops_at_branch() {
    TEST_BEGIN("the climb stops at a node with another child");
    auto t = fresh("/ttd_branch", cfg());
    auto a = seq(0x6000, 16), b = a;
    for (int i = 8; i < 16; i++) b[i] = 0x6F00 + (uint64_t)i;
    insert(t, a, 16);
    insert(t, b, 16);
    assert(t.dedup_prefix(a.data(), 16) == 8);
    assert(match(t, a, 16) == 8 && match(t, b, 16) == 16);
    check_leaves(t);
    assert(t.dedup_prefix(b.data(), 16) == 16 && t.total_blocks() == 0);
    check_leaves(t);
    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_pinned_leaf() {
    TEST_BEGIN("a pinned leaf is kept");
    auto t = fresh("/ttd_pinleaf", cfg());
    auto h = seq(0x7000, 8);
    insert(t, h, 8);
    auto q = t.query_components(h.data(), 8, COMPONENT_MASK_FULL, true, false);
    assert(t.dedup_prefix(h.data(), 8) == 0 && match(t, h, 8) == 8);
    assert(t.dedup_prefix(h.data(), 6) == 0);
    assert(t.dec_ref(h.data(), 8, std::move(q.pin)));
    assert(t.dedup_prefix(h.data(), 8) == 8);
    check_leaves(t);
    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_pinned_ancestor() {
    TEST_BEGIN("a pinned ancestor stops the climb and rejoins the LRU by time");
    auto t = fresh("/ttd_pinanc", cfg());
    auto x = seq(0x8000, 4), h = seq(0x9000, 12), y = seq(0xA000, 4);
    insert(t, x, 4);
    insert(t, h, 12);
    insert(t, y, 4);
    auto q = t.query_components(h.data(), 8, COMPONENT_MASK_FULL, true, false);
    assert(t.dedup_prefix(h.data(), 12) == 4 && match(t, h, 12) == 8);
    check_leaves(t);
    assert(t.dec_ref(h.data(), 8, std::move(q.pin)));

    assert(t.try_evict_batch(1) == 4 && match(t, x, 4) == 0 && match(t, h, 8) == 8);
    assert(t.try_evict_batch(1) == 4 && match(t, h, 8) == 4 && match(t, y, 4) == 4);
    assert(t.try_evict_batch(1) == 4 && match(t, h, 8) == 0 && match(t, y, 4) == 4);
    check_leaves(t);
    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_leaf_positions() {
    TEST_BEGIN("a leaf at the LRU head, middle, tail or alone");
    auto t = fresh("/ttd_pos", cfg());
    auto a = seq(0xB000, 4), b = seq(0xB100, 4), c = seq(0xB200, 4), d = seq(0xB300, 4);
    insert(t, a, 4);
    insert(t, b, 4);
    insert(t, c, 4);
    assert(t.dedup_prefix(b.data(), 4) == 4);
    check_leaves(t);
    assert(t.dedup_prefix(a.data(), 4) == 4);
    check_leaves(t);
    insert(t, d, 4);
    assert(t.dedup_prefix(d.data(), 4) == 4);
    check_leaves(t);
    assert(t.try_evict_batch(1) == 4 && match(t, c, 4) == 0);
    insert(t, c, 4);
    assert(t.dedup_prefix(c.data(), 4) == 4);
    check_leaves(t);
    ShmHeader* hdr;
    nodes_of(t, hdr);
    assert(hdr->leaf_list_head == INVALID_NODE_ID && hdr->leaf_list_tail == INVALID_NODE_ID);
    insert(t, a, 4);
    insert(t, b, 4);
    assert(t.try_evict_batch(1) == 4 && match(t, a, 4) == 0 && match(t, b, 4) == 4);
    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_aux_lanes() {
    TEST_BEGIN("SWA cells and Mamba checkpoints of the chain are recycled");
    auto t = fresh("/ttd_aux", aux_cfg());
    auto h = seq(0xC000, 8);
    insert(t, h, 4);
    insert(t, h, 8);
    insert(t, h, 4, ComponentType::SWA, 4);
    insert(t, h, 4, ComponentType::MAMBA, 1);
    insert(t, h, 8, ComponentType::SWA, 4);
    insert(t, h, 8, ComponentType::MAMBA, 1);
    assert(t.swa_mempool_free() == 56 && t.mamba_mempool_free() == 14);
    assert(t.dedup_prefix(h.data(), 8) == 8);
    assert(t.mempool_free() == 64 && t.swa_mempool_free() == 64 && t.mamba_mempool_free() == 16);
    check_leaves(t);
    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_random() {
    TEST_BEGIN("random insert / pin / dedup / evict keep the LRU list whole");
    const uint32_t cap = 512;
    auto t = fresh("/ttd_rand", cfg(cap));
    std::mt19937 rng(7);
    auto pick = [&](int n) { return (int)(rng() % (uint32_t)n); };
    std::vector<std::vector<uint64_t>> paths;
    for (int i = 0; i < 12; i++) {
        std::vector<uint64_t> p;
        if (!paths.empty() && pick(3) != 0) {
            const auto& base = paths[pick((int)paths.size())];
            p.assign(base.begin(), base.begin() + 1 + pick((int)base.size() - 1));
        }
        while (p.size() < 32) p.push_back(0x100000 * (uint64_t)(i + 1) + p.size());
        paths.push_back(p);
    }
    struct Pin { std::vector<uint64_t> h; int n; int held; PinRecord pin; };
    std::vector<Pin> pins;
    for (int step = 0; step < 4000; step++) {
        const auto& p = paths[pick((int)paths.size())];
        const int n = 1 + pick((int)p.size());
        switch (pick(5)) {
        case 0: {
            const int m = match(t, p, n);
            if (m == n) break;
            std::vector<int32_t> s;
            const int got = t.allocate_slots(n - m, s);
            if (got < n - m || match(t, p, n) != m) {
                if (got > 0) t.recycle_slots(s.data(), got);
                break;
            }
            assert(t.insert(p.data(), n, s.data(), got, m, true).error == InsertError::OK);
            break;
        }
        case 1:
            if (pins.size() < 6) {
                auto q = t.query_components(p.data(), n, COMPONENT_MASK_FULL, true, false);
                pins.push_back({p, n, (int)q.topology_match_length, std::move(q.pin)});
            }
            break;
        case 2:
            if (!pins.empty()) {
                const int i = pick((int)pins.size());
                assert(t.dec_ref(pins[i].h.data(), pins[i].n, std::move(pins[i].pin)));
                pins.erase(pins.begin() + i);
            }
            break;
        case 3: {
            const uint64_t blocks = t.total_blocks();
            const int before = match(t, p, n);
            const int freed = t.dedup_prefix(p.data(), n);
            assert(match(t, p, n) == before - freed && t.total_blocks() == blocks - freed);
            assert(t.dedup_prefix(p.data(), n) == 0);
            break;
        }
        default:
            t.try_evict_batch(1 + pick(8));
        }
        check_leaves(t, false);
        assert(t.mempool_free() + t.total_blocks() == cap);
        for (const Pin& x : pins) assert(match(t, x.h, x.n) >= x.held);
    }
    for (Pin& x : pins) assert(t.dec_ref(x.h.data(), x.n, std::move(x.pin)));
    t.try_evict_batch((int)cap);
    assert(t.total_blocks() == 0 && t.mempool_free() == cap);
    check_leaves(t);
    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

int main() {
    printf("test_tree_dedup\n");
    test_covered_chain();
    test_not_covered();
    test_stops_at_branch();
    test_pinned_leaf();
    test_pinned_ancestor();
    test_leaf_positions();
    test_aux_lanes();
    test_random();
    return tests_report();
}
