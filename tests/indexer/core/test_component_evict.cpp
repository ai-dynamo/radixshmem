// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// EV1 lifecycle coverage for the aux component chains; they have no public
// read API by design, so the validator reads the shm structures directly.

#include "cluster_manifest.h"
#include "radixshmem_config.h"
#include "radix_tree.h"
#include "test_utils.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <sys/mman.h>
#include <thread>
#include <vector>

using namespace radixshmem;

static constexpr uint32_t FS = COMPONENT_MASK_FULL | COMPONENT_MASK_SWA;
static constexpr uint32_t FM = COMPONENT_MASK_FULL | COMPONENT_MASK_MAMBA;
static constexpr uint32_t FSM = COMPONENT_MASK_ALL;

static std::vector<uint64_t> seq(uint64_t base, int n) {
    std::vector<uint64_t> h(n);
    for (int i = 0; i < n; i++) h[i] = base + (uint64_t)i;
    return h;
}

static ShmConfig cfg(uint32_t w, int cs, uint32_t max_blocks) {
    ShmConfig c;
    c.max_nodes = 512;
    c.max_blocks = max_blocks;
    c.register_chunk_tokens = cs;
    c.hugepage_path = "";
    c.component_mask = FSM;
    c.swa_window_blocks = w;
    c.mamba_state_bytes = 256;
    c.swa_max_blocks = max_blocks;
    c.mamba_max_slots = 64;
    return c;
}

static std::vector<int32_t> insert_fresh(RadixShmemTree& t, const uint64_t* h,
                                         int n, int start = 0) {
    std::vector<int32_t> slots;
    int got = t.allocate_slots(n - start, slots);
    assert(got == n - start);
    auto r = t.insert(h, n, slots.data(), (int)slots.size(), start, true);
    assert(r.error == InsertError::OK);
    return slots;
}

static void publish_swa(RadixShmemTree& t, const uint64_t* h, int n, int W) {
    const int start = n > W ? n - W : 0;
    std::vector<int32_t> s;
    int got = t.allocate_slots(n - start, s, ComponentType::SWA);
    assert(got == n - start);
    auto r = t.insert(h, n, s.data(), (int)s.size(), /*start=*/0,
                      /*auto_recycle=*/true, ComponentType::SWA);
    assert(r.error == InsertError::OK);
}

static int32_t publish_mamba(RadixShmemTree& t, const uint64_t* h, int n) {
    std::vector<int32_t> s;
    int got = t.allocate_slots(1, s, ComponentType::MAMBA);
    assert(got == 1);
    auto r = t.insert(h, n, s.data(), 1, /*start=*/0,
                      /*auto_recycle=*/true, ComponentType::MAMBA);
    assert(r.error == InsertError::OK);
    return s[0];
}

static ShmHeader* hdr_of(RadixShmemTree& t) {
    return reinterpret_cast<ShmHeader*>(t.shm_base());
}
static NodeComponentState* cst_of(RadixShmemTree& t, uint32_t id) {
    ShmHeader* h = hdr_of(t);
    return reinterpret_cast<NodeComponentState*>(
               (char*)t.shm_base() + h->component_state_offset) + id;
}
static TreeNode* node_of(RadixShmemTree& t, uint32_t id) {
    ShmHeader* h = hdr_of(t);
    return reinterpret_cast<TreeNode*>(
               (char*)t.shm_base() + h->node_pool_offset) + id;
}
static SlotLruEntry* lru_of(RadixShmemTree& t, bool swa) {
    ShmHeader* h = hdr_of(t);
    const uint64_t off = swa ? h->swa_lru_offset : h->mamba_lru_offset;
    return off ? reinterpret_cast<SlotLruEntry*>((char*)t.shm_base() + off)
               : nullptr;
}
static const int32_t* swa_lane(RadixShmemTree& t, uint32_t id) {
    TreeNode* n = node_of(t, id);
    return reinterpret_cast<const int32_t*>(
        (char*)t.shm_base() + n->data_offset + 12u * n->num_blocks);
}

// Slots in list order (head = oldest), validating the back links.
static std::vector<int32_t> slots(RadixShmemTree& t, bool swa) {
    std::vector<int32_t> out;
    SlotLruEntry* e = lru_of(t, swa);
    if (e == nullptr) return out;
    ShmHeader* h = hdr_of(t);
    const AuxListAnchors& a = swa ? h->swa_list : h->mamba_list;
    const uint32_t cap = swa ? h->swa_mempool_total_blocks
                             : h->mamba_mempool_total_blocks;
    uint32_t prev = INVALID_SLOT_LINK;
    for (uint32_t cur = a.head; cur != INVALID_SLOT_LINK; cur = e[cur].next) {
        assert(out.size() < cap && cur < cap);
        assert(e[cur].prev == prev && e[cur].node_id != INVALID_NODE_ID);
        out.push_back((int32_t)cur);
        prev = cur;
    }
    assert(a.tail == prev);
    return out;
}
// Node ids in list order, consecutive repeats collapsed.
static std::vector<uint32_t> chain(RadixShmemTree& t, bool swa) {
    std::vector<uint32_t> ids;
    SlotLruEntry* e = lru_of(t, swa);
    for (int32_t s : slots(t, swa)) {
        const uint32_t n = e[s].node_id;
        if (ids.empty() || ids.back() != n) ids.push_back(n);
    }
    return ids;
}
static uint64_t swa_ts(RadixShmemTree& t, uint32_t id) {   // node's oldest block
    uint64_t m = UINT64_MAX;
    SlotLruEntry* e = lru_of(t, true);
    for (int32_t s : slots(t, true))
        if (e[s].node_id == id && e[s].ts < m) m = e[s].ts;
    return m;
}
static uint64_t mamba_ts(RadixShmemTree& t, uint32_t id) {
    return lru_of(t, false)[node_of(t, id)->mamba_checkpoint_slot].ts;
}

// Live blocks actually present in the node's SWA lane (0 without a lane).
static uint32_t lane_live_count(RadixShmemTree& t, uint32_t id) {
    ShmHeader* h = hdr_of(t);
    if (!component_has_swa(h->component_config.component_mask)) return 0;
    TreeNode* n = node_of(t, id);
    const int32_t* lane = swa_lane(t, id);
    uint32_t c = 0;
    for (uint32_t i = 0; i < n->num_blocks; i++)
        if (lane[i] != INVALID_COMPONENT_SLOT) c++;
    return c;
}

// Both lists and the lanes/checkpoints describe the same slots; ts ascends.
static void check_membership(RadixShmemTree& t) {
    auto sw = slots(t, true), mb = slots(t, false);
    SlotLruEntry* es = lru_of(t, true);
    SlotLruEntry* em = lru_of(t, false);
    ShmHeader* h = hdr_of(t);
    std::vector<uint32_t> swa_cnt(h->node_pool_capacity, 0);
    for (int32_t s : sw) {
        const SlotLruEntry& e = es[s];
        assert(e.idx < node_of(t, e.node_id)->num_blocks);
        assert(swa_lane(t, e.node_id)[e.idx] == s);
        swa_cnt[e.node_id]++;
    }
    for (int32_t s : mb)
        assert(node_of(t, em[s].node_id)->mamba_checkpoint_slot == s);
    for (uint32_t id = 0; id < h->node_pool_capacity; id++) {
        TreeNode* n = node_of(t, id);
        if (n->num_blocks == 0) {
            assert(swa_cnt[id] == 0);
            continue;
        }
        NodeComponentState* cs = cst_of(t, id);
        assert(cs->swa_live_blocks == lane_live_count(t, id) &&
               cs->swa_live_blocks == swa_cnt[id]);
        if (n->mamba_checkpoint_slot != INVALID_COMPONENT_SLOT)
            assert(em[n->mamba_checkpoint_slot].node_id == id);
    }
    for (size_t i = 1; i < sw.size(); i++) assert(es[sw[i - 1]].ts <= es[sw[i]].ts);
    for (size_t i = 1; i < mb.size(); i++) assert(em[mb[i - 1]].ts <= em[mb[i]].ts);
}

static void test_publish_and_fww() {
    TEST_BEGIN("publish: joins MRU; republish keeps slots, refreshes the stamp");

    const char* name = "/tce_publish";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 4, 64));
    auto h = seq(0x100, 8);
    insert_fresh(t, h.data(), 8);
    assert(chain(t, true).empty() && chain(t, false).empty());

    publish_swa(t, h.data(), 8, 4);
    auto sw = chain(t, true);
    assert(sw.size() == 1);
    const uint32_t B = sw[0];
    const uint64_t ts0 = swa_ts(t, B);
    assert(ts0 > 0);

    const uint32_t free1 = t.swa_mempool_free();
    publish_swa(t, h.data(), 8, 4);   // FWW: new slots return, stamp moves
    assert(chain(t, true) == sw && swa_ts(t, B) > ts0);
    assert(t.swa_mempool_free() == free1);

    publish_mamba(t, h.data(), 8);
    auto mb = chain(t, false);
    assert(mb.size() == 1 && mb[0] == B);
    const uint64_t mts0 = mamba_ts(t, B);
    assert(mts0 > 0);

    {
        std::vector<int32_t> s;
        assert(t.allocate_slots(1, s, ComponentType::MAMBA) == 1);
        auto r = t.insert(h.data(), 8, s.data(), 1, /*start=*/0,
                          /*auto_recycle=*/true, ComponentType::MAMBA);
        assert(r.error == InsertError::OK && r.unused_slots.size() == 1);
    }
    assert(chain(t, false) == mb && mamba_ts(t, B) == mts0);

    {   // positional start is FULL-only: SWA/MAMBA refuse start != 0 whole.
        std::vector<int32_t> s;
        assert(t.allocate_slots(1, s, ComponentType::MAMBA) == 1);
        const uint32_t free_before = t.mamba_mempool_free();
        auto r = t.insert(h.data(), 8, s.data(), 1, /*start=*/1,
                          /*auto_recycle=*/true, ComponentType::MAMBA);
        assert(r.error == InsertError::BAD_REQUEST &&
               r.unused_slots.size() == 1);
        assert(t.mamba_mempool_free() == free_before + 1);
    }

    check_membership(t);

    t.reset();
    assert(hdr_of(t)->swa_list.head == INVALID_SLOT_LINK &&
           hdr_of(t)->swa_list.tail == INVALID_SLOT_LINK &&
           hdr_of(t)->mamba_list.head == INVALID_SLOT_LINK &&
           hdr_of(t)->mamba_list.tail == INVALID_SLOT_LINK);
    assert(slots(t, true).empty() && slots(t, false).empty());

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_touch_quadrants() {
    TEST_BEGIN("touch: only update_meta=true moves; per-component chains");

    const char* name = "/tce_touch";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 4, 64));
    auto ha = seq(0xA00, 8), hb = seq(0xB00, 8);
    insert_fresh(t, ha.data(), 8);
    insert_fresh(t, hb.data(), 8);
    publish_swa(t, ha.data(), 8, 4);
    publish_swa(t, hb.data(), 8, 4);
    publish_mamba(t, ha.data(), 8);
    publish_mamba(t, hb.data(), 8);
    auto sw = chain(t, true);
    assert(sw.size() == 2);
    const uint32_t A = sw[0], B = sw[1];
    assert((chain(t, false) == std::vector<uint32_t>{A, B}));

    (void)t.query_components(ha.data(), 8, FS, false, false);
    auto rp = t.query_components(ha.data(), 8, FSM, true, false);
    assert((chain(t, true) == std::vector<uint32_t>{A, B}));
    assert((chain(t, false) == std::vector<uint32_t>{A, B}));
    assert(t.dec_ref(ha.data(), 8, std::move(rp.pin)));

    const uint64_t tsA = swa_ts(t, A);
    (void)t.query_components(ha.data(), 8, FS, false, true);
    assert((chain(t, true) == std::vector<uint32_t>{B, A}));
    assert((chain(t, false) == std::vector<uint32_t>{A, B}));
    assert(swa_ts(t, A) >= tsA);

    (void)t.query_components(ha.data(), 8, FM, false, true);
    assert((chain(t, true) == std::vector<uint32_t>{B, A}));
    assert((chain(t, false) == std::vector<uint32_t>{B, A}));

    auto rl = t.query_components(hb.data(), 8, FSM, true, true);
    assert((chain(t, true) == std::vector<uint32_t>{A, B}));
    assert((chain(t, false) == std::vector<uint32_t>{A, B}));
    assert(t.dec_ref(hb.data(), 8, std::move(rl.pin)));

    (void)t.query_components(ha.data(), 8, COMPONENT_MASK_FULL, false, true);
    (void)t.query_components(ha.data(), 8, COMPONENT_MASK_FULL, false, true);
    assert((chain(t, true) == std::vector<uint32_t>{A, B}));
    assert((chain(t, false) == std::vector<uint32_t>{A, B}));

    auto hc = seq(0xC00, 4);
    (void)t.query_components(hc.data(), 4, FSM, false, true);
    assert((chain(t, true) == std::vector<uint32_t>{A, B}));

    check_membership(t);
    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_touch_window_order() {
    TEST_BEGIN("touch: only the hit window is re-stamped; prefix blocks keep their age");

    const char* name = "/tce_touchw";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 4, 64));
    auto h = seq(0xC10, 8), he = seq(0xD10, 4);
    insert_fresh(t, h.data(), 8);
    insert_fresh(t, he.data(), 4);
    publish_swa(t, h.data(), 4, 4);    // N1 [0,4)
    publish_swa(t, h.data(), 8, 4);    // N2 [4,8)
    publish_swa(t, he.data(), 4, 4);   // E
    auto sw = chain(t, true);
    assert(sw.size() == 3);
    const uint32_t N1 = sw[0], N2 = sw[1], E = sw[2];

    const uint64_t ts1 = swa_ts(t, N1), ts2 = swa_ts(t, N2);
    (void)t.query_components(h.data(), 8, FS, false, true);
    assert((chain(t, true) == std::vector<uint32_t>{N1, E, N2}));
    assert(swa_ts(t, N1) == ts1 && swa_ts(t, N2) > ts2);

    check_membership(t);
    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_split_membership() {
    TEST_BEGIN("split: suffix keeps its place, prefix joins beside it");

    const char* name = "/tce_split";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(8, 8, 64));
    auto h = seq(0xD00, 8), he = seq(0xE00, 4);
    insert_fresh(t, h.data(), 8);
    insert_fresh(t, he.data(), 4);
    publish_swa(t, h.data(), 8, 8);
    publish_swa(t, he.data(), 4, 8);
    auto sw = chain(t, true);
    assert(sw.size() == 2);
    const uint32_t X = sw[0], E = sw[1];
    const uint64_t tsX = swa_ts(t, X);

    publish_mamba(t, h.data(), 6);
    auto sw2 = chain(t, true);
    assert(sw2.size() == 3 && sw2[1] == X && sw2[2] == E);
    const uint32_t P = sw2[0];
    assert(node_of(t, P)->num_blocks == 6 && node_of(t, X)->num_blocks == 2);
    assert(swa_ts(t, P) == tsX && swa_ts(t, X) == tsX);
    assert((chain(t, false) == std::vector<uint32_t>{P}));

    auto r = t.query_components(h.data(), 8, FS, false, false);
    assert(r.common_hit == 8);

    check_membership(t);
    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_merge_membership() {
    TEST_BEGIN("merge: a parent END checkpoint defers the fold; extension keeps it");

    const char* name = "/tce_merge";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 16, 64));   // cs off the 4-block grid: folds stay folds
    auto h = seq(0xF00, 8), hb = seq(0x1000, 4);
    insert_fresh(t, h.data(), 4);
    insert_fresh(t, hb.data(), 4);
    const int32_t ck = publish_mamba(t, h.data(), 4);
    publish_mamba(t, hb.data(), 4);
    auto mb = chain(t, false);
    assert(mb.size() == 2);
    const uint32_t A = mb[0];
    const uint32_t mamba_free = t.mamba_mempool_free();
    const uint32_t nodes_before = t.total_radix_nodes();

    // Sequential extension: the only-child leaf would fold into A, but A's
    // END carries a checkpoint, so the merge is deferred and the checkpoint
    // stays reachable.
    insert_fresh(t, h.data(), 8, /*start=*/4);
    assert(chain(t, false) == mb);
    assert(t.mamba_mempool_free() == mamba_free);
    assert(t.total_radix_nodes() == nodes_before + 1);
    assert(node_of(t, A)->num_blocks == 4 &&
           node_of(t, A)->mamba_checkpoint_slot == ck);
    auto q = t.query_components(h.data(), 4, FM);
    assert(q.common_hit == 4 && q.mamba_checkpoint_slot == ck);

    // No checkpoint on the parent END: the fold proceeds as before.
    auto hc = seq(0x2000, 8);
    insert_fresh(t, hc.data(), 4);
    const uint32_t nodes_c = t.total_radix_nodes();
    insert_fresh(t, hc.data(), 8, /*start=*/4);
    assert(t.total_radix_nodes() == nodes_c);
    assert(t.query_components(hc.data(), 8, COMPONENT_MASK_FULL).common_hit == 8);

    check_membership(t);
    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

// ==================================================================
static void test_cascade_and_node_reuse() {
    TEST_BEGIN("cascade: retire before detach; reused node IDs leave no ghost");

    const char* name = "/tce_cascade";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 4, 64));
    auto ha = seq(0x2000, 4), hb = seq(0x3000, 4), hc = seq(0x4000, 4),
         hd = seq(0x5000, 4);
    insert_fresh(t, ha.data(), 4);
    insert_fresh(t, hb.data(), 4);
    insert_fresh(t, hc.data(), 4);
    for (auto* h : {&ha, &hb, &hc}) {
        publish_swa(t, h->data(), 4, 4);
        publish_mamba(t, h->data(), 4);
    }
    auto sw = chain(t, true);
    assert(sw.size() == 3);
    const uint32_t A = sw[0], B = sw[1], C = sw[2];
    const uint32_t swa_free = t.swa_mempool_free();
    const uint32_t mamba_free = t.mamba_mempool_free();

    (void)t.query_components(ha.data(), 4, COMPONENT_MASK_FULL, false, true);
    assert(t.try_evict_batch(4) == 4);
    assert((chain(t, true) == std::vector<uint32_t>{A, C}));
    assert((chain(t, false) == std::vector<uint32_t>{A, C}));
    assert(t.swa_mempool_free() == swa_free + 4);
    assert(t.mamba_mempool_free() == mamba_free + 1);
    assert((int)t.query_components(hb.data(), 4, COMPONENT_MASK_FULL, false, false).common_hit == 0);
    check_membership(t);

    assert(t.try_evict_batch(4) == 4);
    assert((chain(t, true) == std::vector<uint32_t>{A}));
    assert(t.try_evict_batch(4) == 4);
    assert(chain(t, true).empty() && chain(t, false).empty());
    assert(t.swa_mempool_free() == swa_free + 12);
    assert(t.mamba_mempool_free() == mamba_free + 3);

    insert_fresh(t, hd.data(), 4);
    check_membership(t);
    publish_swa(t, hd.data(), 4, 4);
    auto sw2 = chain(t, true);
    assert(sw2.size() == 1 && sw2[0] == A);
    check_membership(t);

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_independent_eviction() {
    TEST_BEGIN("evict: LRU order, node over-evict, other lanes untouched");

    const char* name = "/tce_walk";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 4, 64));
    int cascades = 0;
    t.set_evict_callback([&](const std::vector<EvictedNode>&) { cascades++; });
    auto ha = seq(0x8000, 4), hb = seq(0x9000, 4), hc = seq(0xA000, 4);
    for (auto* h : {&ha, &hb, &hc}) {
        insert_fresh(t, h->data(), 4);
        publish_swa(t, h->data(), 4, 4);
        publish_mamba(t, h->data(), 4);
    }
    auto sw = chain(t, true);
    const uint32_t A = sw[0], B = sw[1], C = sw[2];
    const uint32_t swa_free = t.swa_mempool_free();
    const uint32_t mamba_free = t.mamba_mempool_free();

    assert(t.try_evict_component_batch(ComponentType::SWA, 4) == 4);
    assert((chain(t, true) == std::vector<uint32_t>{B, C}));
    assert((chain(t, false) == std::vector<uint32_t>{A, B, C}));
    assert(t.swa_mempool_free() == swa_free + 4);
    assert((int)t.query_components(ha.data(), 4, COMPONENT_MASK_FULL, false, false).common_hit == 4);
    auto r = t.query_components(ha.data(), 4, FS, false, false);
    assert(r.topology_match_length == 4 && r.common_hit == 0);

    assert(t.try_evict_component_batch(ComponentType::MAMBA, 1) == 1);
    assert((chain(t, false) == std::vector<uint32_t>{B, C}));
    assert((chain(t, true) == std::vector<uint32_t>{B, C}));
    assert(t.mamba_mempool_free() == mamba_free + 1);

    assert(t.try_evict_component_batch(ComponentType::SWA, 100) == 8);
    assert(t.try_evict_component_batch(ComponentType::SWA, 100) == 0);
    assert(t.try_evict_component_batch(ComponentType::MAMBA, 100) == 2);
    assert(t.try_evict_component_batch(ComponentType::MAMBA, 100) == 0);
    assert(t.swa_mempool_free() == swa_free + 12);
    assert(t.mamba_mempool_free() == mamba_free + 3);
    assert(cascades == 0);

    check_membership(t);
    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_evict_swa_by_slot() {
    TEST_BEGIN("evict: pops exactly the shortfall from the oldest slots");

    const char* name = "/tce_swa_slots";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 4, 64));
    auto ha = seq(0x3100, 4), hb = seq(0x3200, 8), hc = seq(0x3300, 8);
    insert_fresh(t, ha.data(), 4);
    insert_fresh(t, hb.data(), 8);
    insert_fresh(t, hc.data(), 8);
    publish_swa(t, ha.data(), 4, 4);   // A
    publish_swa(t, hb.data(), 8, 4);   // B2 [4,8)
    auto sw = chain(t, true);
    assert(sw.size() == 2);
    const uint32_t A = sw[0], B2 = sw[1];
    const uint32_t free0 = t.swa_mempool_free();

    // A hit moves B2's window to the tail; A's blocks are the oldest.
    (void)t.query_components(hb.data(), 8, FS, false, true);
    assert(t.try_evict_component_batch(ComponentType::SWA, 1) == 1);
    assert((chain(t, true) == std::vector<uint32_t>{A, B2}));
    assert(cst_of(t, A)->swa_live_blocks == 3 && lane_live_count(t, A) == 3);
    assert(swa_lane(t, A)[0] == INVALID_COMPONENT_SLOT);
    assert(t.query_components(ha.data(), 4, FS, false, false).common_hit == 0);
    assert(t.query_components(ha.data(), 4, COMPONENT_MASK_FULL, false,
                              false).common_hit == 4);
    assert(t.try_evict_component_batch(ComponentType::SWA, 3) == 3);
    assert((chain(t, true) == std::vector<uint32_t>{B2}));
    assert(t.swa_mempool_free() == free0 + 4);

    // A window spanning two nodes lists in block order; a shortfall crossing
    // publishes takes B2 whole and then C1's oldest block.
    publish_swa(t, hc.data(), 6, 4);   // C1 [2,4) + C2 [4,6)
    auto sw2 = chain(t, true);
    assert(sw2.size() == 3 && sw2[0] == B2);
    const uint32_t C1 = sw2[1], C2 = sw2[2];
    assert(t.try_evict_component_batch(ComponentType::SWA, 5) == 5);
    assert((chain(t, true) == std::vector<uint32_t>{C1, C2}));
    assert(cst_of(t, C1)->swa_live_blocks == 1 &&
           cst_of(t, C2)->swa_live_blocks == 2);
    assert(t.swa_mempool_free() == free0 + 5);
    assert(t.query_components(hc.data(), 6, FS, false, false).common_hit == 0);

    publish_swa(t, hc.data(), 8, 4);   // [4,6) kept and moved, [6,8) added
    assert(cst_of(t, C2)->swa_live_blocks == 4);
    assert(t.query_components(hc.data(), 8, FS, false, false).common_hit == 8);

    check_membership(t);
    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_evict_ref_gate() {
    TEST_BEGIN("evict: own-lane ref protects; the other lane's ref does not");

    const char* name = "/tce_gate";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 4, 64));
    auto ha = seq(0xB000, 4), hb = seq(0xC000, 4), hc = seq(0xD000, 4);
    for (auto* h : {&ha, &hb, &hc}) {
        insert_fresh(t, h->data(), 4);
        publish_swa(t, h->data(), 4, 4);
        publish_mamba(t, h->data(), 4);
    }
    auto sw = chain(t, true);
    const uint32_t A = sw[0], B = sw[1], C = sw[2];

    auto ra = t.query_components(ha.data(), 4, FS, true, false);
    auto rb = t.query_components(hb.data(), 4, FM, true, false);

    assert(t.try_evict_component_batch(ComponentType::SWA, 4) == 4);
    assert((chain(t, true) == std::vector<uint32_t>{A, C}));

    auto rc = t.query_components(hc.data(), 4, FS, true, false);
    assert(t.try_evict_component_batch(ComponentType::SWA, 100) == 0);
    assert((chain(t, true) == std::vector<uint32_t>{A, C}));

    assert(t.dec_ref(ha.data(), 4, std::move(ra.pin)));
    assert(t.dec_ref(hc.data(), 4, std::move(rc.pin)));
    assert(t.try_evict_component_batch(ComponentType::SWA, 100) == 8);

    assert(t.try_evict_component_batch(ComponentType::MAMBA, 100) == 2);
    auto mb = chain(t, false);
    assert(mb.size() == 1 && mb[0] == B);
    assert(t.try_evict_component_batch(ComponentType::MAMBA, 100) == 0);
    assert(t.dec_ref(hb.data(), 4, std::move(rb.pin)));

    check_membership(t);
    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_try_entry_gates() {
    TEST_BEGIN("evict: try entry rejects FULL, no-ops on busy/zero/disabled");

    const char* name = "/tce_entry";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 4, 64));
    auto h = seq(0xE000, 4);
    insert_fresh(t, h.data(), 4);
    publish_swa(t, h.data(), 4, 4);

    assert(t.try_evict_component_batch(ComponentType::FULL, 4) == -1);
    assert(t.try_evict_component_batch((ComponentType)7, 4) == -1);
    assert(t.try_evict_component_batch(ComponentType::SWA, 0) == 0);
    assert(t.try_evict_component_batch(ComponentType::SWA, -3) == 0);

    auto sw0 = chain(t, true);
    t.write_lock();
    int r = -99;
    std::thread th(
        [&] { r = t.try_evict_component_batch(ComponentType::SWA, 4); });
    th.join();
    t.write_unlock();
    assert(r == 0 && chain(t, true) == sw0);

    const char* name2 = "/tce_entry2";
    shm_unlink(name2);
    ShmConfig c2 = cfg(0, 4, 64);
    c2.component_mask = COMPONENT_MASK_FULL;
    c2.swa_window_blocks = 0;
    c2.mamba_state_bytes = 0;
    c2.swa_max_blocks = 0;
    c2.mamba_max_slots = 0;
    auto t2 = RadixShmemTree::create(name2, c2);
    assert(t2.try_evict_component_batch(ComponentType::SWA, 4) == 0);
    assert(t2.try_evict_component_batch(ComponentType::MAMBA, 4) == 0);
    t2.destroy();

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_allocate_retry() {
    TEST_BEGIN("alloc: shortfall evicts own lane, retries once, all-or-none");

    const char* name = "/tce_retry";
    shm_unlink(name);
    ShmConfig c = cfg(4, 4, 64);
    c.swa_max_blocks = 8;
    c.mamba_max_slots = 2;
    auto t = RadixShmemTree::create(name, c);
    auto ha = seq(0x1100, 4), hb = seq(0x1200, 4), hc = seq(0x1300, 4);
    insert_fresh(t, ha.data(), 4);
    insert_fresh(t, hb.data(), 4);
    insert_fresh(t, hc.data(), 4);
    publish_swa(t, ha.data(), 4, 4);
    publish_swa(t, hb.data(), 4, 4);
    assert(t.swa_mempool_free() == 0);

    std::vector<int32_t> out;
    assert(t.allocate_slots(4, out, ComponentType::SWA) == 4);
    assert(chain(t, true).size() == 1);
    auto rr = t.insert(hc.data(), 4, out.data(), 4, /*start=*/0,
                       /*auto_recycle=*/true, ComponentType::SWA);
    assert(rr.error == InsertError::OK);
    assert(chain(t, true).size() == 2 && t.swa_mempool_free() == 0);

    auto rb = t.query_components(hb.data(), 4, FS, true, false);
    auto rc = t.query_components(hc.data(), 4, FS, true, false);
    out.assign(1, 42);
    assert(t.allocate_slots(4, out, ComponentType::SWA) == 0 && out.empty());
    assert(chain(t, true).size() == 2);

    assert(t.dec_ref(hb.data(), 4, std::move(rb.pin)));
    assert(t.dec_ref(hc.data(), 4, std::move(rc.pin)));
    assert(t.allocate_slots(8, out, ComponentType::SWA) == 8);
    assert(chain(t, true).empty());
    t.recycle_slots(out.data(), 8, ComponentType::SWA);

    publish_mamba(t, ha.data(), 4);
    publish_mamba(t, hb.data(), 4);
    assert(t.mamba_mempool_free() == 0);
    std::vector<int32_t> mo;
    assert(t.allocate_slots(1, mo, ComponentType::MAMBA) == 1);
    assert(chain(t, false).size() == 1);
    t.recycle_slots(mo.data(), 1, ComponentType::MAMBA);

    check_membership(t);
    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_merge_live_swa() {
    TEST_BEGIN("merge: live SWA lanes fold; chain membership follows the count");

    const char* name = "/tce_merge_swa";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 16, 64));   // cs off the 4-block grid: folds stay folds
    auto h = seq(0x2F00, 8);
    insert_fresh(t, h.data(), 4);
    publish_swa(t, h.data(), 4, 4);            // A[0,4) live 4
    auto sw = chain(t, true);
    assert(sw.size() == 1);
    const uint32_t A = sw[0];
    const uint32_t nodes_before = t.total_radix_nodes();

    // Extension folds the fresh leaf into A: A keeps its chain slot, the
    // lane over [4,8) stays INVALID, the count is unchanged.
    insert_fresh(t, h.data(), 8, /*start=*/4);
    assert(t.total_radix_nodes() == nodes_before);
    assert(node_of(t, A)->num_blocks == 8 &&
           cst_of(t, A)->swa_live_blocks == 4);
    assert((chain(t, true) == std::vector<uint32_t>{A}));
    assert(t.query_components(h.data(), 8, FS, false, false).common_hit == 4);

    // A tail publish fills [4,8) in place: still one node, now fully live.
    publish_swa(t, h.data(), 8, 4);
    assert(cst_of(t, A)->swa_live_blocks == 8);
    assert(t.query_components(h.data(), 8, FS, false, false).common_hit == 8);

    // Retiring the lane frees all eight and empties the chain.
    assert(t.try_evict_component_batch(ComponentType::SWA, 8) == 8);
    assert(chain(t, true).empty() && cst_of(t, A)->swa_live_blocks == 0);

    check_membership(t);
    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_merge_ref_gate() {
    TEST_BEGIN("merge: a Full or SWA ref on the parent defers the fold");

    const char* name = "/tce_merge_ref";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 16, 64));   // cs off the 4-block grid: folds stay folds
    auto ha = seq(0x4100, 12), hb = seq(0x4200, 12);
    insert_fresh(t, ha.data(), 4);
    insert_fresh(t, hb.data(), 4);
    publish_swa(t, hb.data(), 4, 4);
    auto ra = t.query_components(ha.data(), 4, COMPONENT_MASK_FULL, true, false);
    auto rb = t.query_components(hb.data(), 4, COMPONENT_MASK_SWA, true, false);
    assert(!ra.pin.full.empty() && !rb.pin.swa.empty());
    const uint32_t nodes0 = t.total_radix_nodes();

    insert_fresh(t, ha.data(), 8, /*start=*/4);
    insert_fresh(t, hb.data(), 8, /*start=*/4);
    assert(t.total_radix_nodes() == nodes0 + 2);
    assert(t.get_node_boundaries(ha.data(), 8).size() == 2);
    assert(t.get_node_boundaries(hb.data(), 8).size() == 2);
    assert(t.dec_ref(ha.data(), 4, std::move(ra.pin)));
    assert(t.dec_ref(hb.data(), 4, std::move(rb.pin)));

    // Unpinned leaves fold the next extension as usual.
    insert_fresh(t, ha.data(), 12, /*start=*/8);
    insert_fresh(t, hb.data(), 12, /*start=*/8);
    assert(t.total_radix_nodes() == nodes0 + 2);
    assert(t.get_node_boundaries(ha.data(), 12).size() == 2);
    assert(t.query_components(hb.data(), 12, FS, false, false).common_hit == 4);

    check_membership(t);
    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_partial_component_configs() {
    TEST_BEGIN("config: FS / FM shms refuse the absent component and run without it");

    ShmConfig c_fs = cfg(4, 16, 64);   // cs off the 4-block grid: the fold below stays a fold
    c_fs.component_mask = FS;
    c_fs.mamba_state_bytes = 0;
    c_fs.mamba_max_slots = 0;
    ShmConfig c_fm = cfg(4, 16, 64);
    c_fm.component_mask = FM;
    c_fm.swa_window_blocks = 0;
    c_fm.swa_max_blocks = 0;
    shm_unlink("/tce_cfg_fs");
    shm_unlink("/tce_cfg_fm");
    auto ts = RadixShmemTree::create("/tce_cfg_fs", c_fs);
    auto tm = RadixShmemTree::create("/tce_cfg_fm", c_fm);
    assert(hdr_of(ts)->swa_lru_offset != 0 && hdr_of(ts)->mamba_lru_offset == 0);
    assert(hdr_of(tm)->swa_lru_offset == 0 && hdr_of(tm)->mamba_lru_offset != 0);

    auto h = seq(0x5100, 8);
    insert_fresh(ts, h.data(), 4);
    insert_fresh(tm, h.data(), 4);
    std::vector<int32_t> v;
    int32_t fake[4] = {0, 1, 2, 3};
    assert(ts.allocate_slots(1, v, ComponentType::MAMBA) == 0 && v.empty());
    assert(tm.allocate_slots(4, v, ComponentType::SWA) == 0 && v.empty());
    ts.recycle_slots(fake, 1, ComponentType::MAMBA);
    tm.recycle_slots(fake, 4, ComponentType::SWA);
    auto rm = ts.insert(h.data(), 4, fake, 1, 0, true, ComponentType::MAMBA);
    assert(rm.error == InsertError::BAD_REQUEST && rm.unused_slots.size() == 1);
    auto rs = tm.insert(h.data(), 4, fake, 4, 0, true, ComponentType::SWA);
    assert(rs.error == InsertError::BAD_REQUEST && rs.unused_slots.size() == 4);
    assert(ts.query_components(h.data(), 4, FM).status == RemoteStatus::BAD_REQUEST);
    assert(tm.query_components(h.data(), 4, FS).status == RemoteStatus::BAD_REQUEST);
    assert(ts.try_evict_component_batch(ComponentType::MAMBA, 4) == 0);
    assert(tm.try_evict_component_batch(ComponentType::SWA, 4) == 0);

    publish_swa(ts, h.data(), 4, 4);
    const uint32_t n0 = ts.total_radix_nodes();
    insert_fresh(ts, h.data(), 8, /*start=*/4);
    assert(ts.total_radix_nodes() == n0);
    assert(ts.query_components(h.data(), 8, FS).common_hit == 4);
    check_membership(ts);
    const uint32_t swa_free = ts.swa_mempool_free();
    assert(ts.try_evict_batch(8) == 8);
    assert(ts.swa_mempool_free() == swa_free + 4 && chain(ts, true).empty());

    publish_mamba(tm, h.data(), 4);
    const uint32_t m0 = tm.total_radix_nodes();
    insert_fresh(tm, h.data(), 8, /*start=*/4);
    assert(tm.total_radix_nodes() == m0 + 1);
    assert(tm.query_components(h.data(), 8, FM).common_hit == 4);
    check_membership(tm);
    assert(tm.try_evict_batch(8) == 8);
    assert(tm.mamba_mempool_free() == tm.mamba_mempool_total() &&
           chain(tm, false).empty());

    expect_healthy(ts);
    expect_healthy(tm);
    ts.destroy();
    tm.destroy();
    TEST_PASS();
}

static void test_layout_follows_mask() {
    TEST_BEGIN("layout: SWA lane / CST exist only for enabled components");

    auto c_full = cfg(4, 4, 64);
    c_full.component_mask = COMPONENT_MASK_FULL;
    c_full.swa_window_blocks = 0;
    c_full.swa_max_blocks = 0;
    c_full.mamba_max_slots = 0;
    auto c_fm = cfg(4, 4, 64);
    c_fm.component_mask = FM;
    c_fm.swa_window_blocks = 0;
    c_fm.swa_max_blocks = 0;
    auto c_fsm = cfg(4, 4, 64);
    for (const char* n : {"/tce_lay_f", "/tce_lay_fm", "/tce_lay_fsm"})
        shm_unlink(n);
    auto tf   = RadixShmemTree::create("/tce_lay_f", c_full);
    auto tfm  = RadixShmemTree::create("/tce_lay_fm", c_fm);
    auto tfsm = RadixShmemTree::create("/tce_lay_fsm", c_fsm);

    // CST only with an aux component; the SWA lane only with SWA (12 vs 16
    // B/block). The buddy pool rounds to 64 KB, so compare the configured
    // bytes for the lane and the shm sizes for the table.
    assert(hdr_of(tf)->component_state_offset == 0);
    assert(hdr_of(tfm)->component_state_offset != 0 &&
           hdr_of(tfsm)->component_state_offset != 0);
    assert(c_full.effective_data_pool_bytes() == c_fm.effective_data_pool_bytes());
    assert(c_fm.effective_data_pool_bytes() * 16 ==
           c_fsm.effective_data_pool_bytes() * 12);
    assert(hdr_of(tf)->swa_lru_offset == 0 && hdr_of(tf)->mamba_lru_offset == 0);
    assert(hdr_of(tfm)->swa_lru_offset == 0 && hdr_of(tfm)->mamba_lru_offset != 0);
    assert(hdr_of(tfsm)->swa_lru_offset != 0 && hdr_of(tfsm)->mamba_lru_offset != 0);
    assert(hdr_of(tf)->data_pool_total_size <=
           hdr_of(tfsm)->data_pool_total_size);
    assert(tf.shm_size() < tfm.shm_size() && tfm.shm_size() <= tfsm.shm_size());

    // Full-only still splits, merges and evicts through the CST-less paths.
    auto h = seq(0x3000, 8), hb = seq(0x3000, 4);
    hb[3] = 0x9999;
    insert_fresh(tf, h.data(), 8);
    insert_fresh(tf, hb.data(), 4);   // splits the first node at 3
    assert(tf.query_components(h.data(), 8, COMPONENT_MASK_FULL).common_hit == 8);
    assert(tf.query_components(hb.data(), 4, COMPONENT_MASK_FULL).common_hit == 4);
    auto q = tf.query_components(h.data(), 8, COMPONENT_MASK_FULL, /*lock=*/true);
    assert(q.common_hit == 8 && !q.pin.empty());
    assert(tf.dec_ref(h.data(), 8, std::move(q.pin)));
    std::vector<int32_t> tmp;
    tf.allocate_slots((int)tf.mempool_total(), tmp);
    assert(tf.query_components(h.data(), 8, COMPONENT_MASK_FULL).common_hit == 0);

    // The manifest gate accepts the Full-only shape (CST offset 0) and
    // rejects a stray table.
    PeerEntry e{};
    e.shm_size = tf.shm_size();
    e.node_pool_capacity = hdr_of(tf)->node_pool_capacity;
    e.component_config = hdr_of(tf)->component_config;
    e.component_state_offset = 0;
    assert(peer_component_abi_valid(e, tf.component_config()));
    e.component_state_offset = 64;
    assert(!peer_component_abi_valid(e, tf.component_config()));
    PeerEntry em{};
    em.shm_size = tfm.shm_size();
    em.node_pool_capacity = hdr_of(tfm)->node_pool_capacity;
    em.component_config = hdr_of(tfm)->component_config;
    em.component_state_offset = 0;   // aux config must publish its table
    assert(!peer_component_abi_valid(em, tfm.component_config()));

    tf.recycle_slots(tmp.data(), (int)tmp.size());   // hand the drained pool back before the audit
    check_membership(tfsm);
    expect_healthy(tf);
    expect_healthy(tfm);
    expect_healthy(tfsm);
    tf.destroy();
    tfm.destroy();
    tfsm.destroy();
    TEST_PASS();
}

int main() {
    printf("test_component_evict\n");
    test_publish_and_fww();
    test_touch_quadrants();
    test_touch_window_order();
    test_split_membership();
    test_merge_membership();
    test_merge_live_swa();
    test_merge_ref_gate();
    test_partial_component_configs();
    test_layout_follows_mask();
    test_cascade_and_node_reuse();
    test_independent_eviction();
    test_evict_swa_by_slot();
    test_evict_ref_gate();
    test_try_entry_gates();
    test_allocate_retry();
    printf("ALL PASS\n");
    return 0;
}
