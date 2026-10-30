// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Consolidated component-query coverage: per-block SWA / node-END Mamba
// semantics, joint
// common_hit, trust-model pin/release, the client facade, and the remote walk.

#include "radixshmem_config.h"
#include "radix_client.h"
#include "local.h"
#include "radix_tree.h"
#include "test_utils.h"

#ifndef RADIXSHMEM_NO_RDMA
#include "fake_rdma_endpoint.hpp"
#include "cluster_manifest.h"
#include "rdma_remote_walker.h"
#endif

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace radixshmem;

static_assert(std::is_aggregate_v<PinRecord>);
static_assert(std::is_default_constructible_v<PinRecord>);
static_assert(std::is_copy_constructible_v<PinRecord>);
static_assert(std::is_nothrow_move_constructible_v<PinRecord>);

static constexpr uint32_t FS = COMPONENT_MASK_FULL | COMPONENT_MASK_SWA;
static constexpr uint32_t FM = COMPONENT_MASK_FULL | COMPONENT_MASK_MAMBA;
static constexpr uint32_t FSM =
    COMPONENT_MASK_FULL | COMPONENT_MASK_SWA | COMPONENT_MASK_MAMBA;

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

static void publish_at(RadixShmemTree& t, const uint64_t* h, int n, int W,
                       std::vector<int32_t>* pos = nullptr) {
    const int start = n > W ? n - W : 0;
    std::vector<int32_t> s;
    int got = t.allocate_slots(n - start, s, ComponentType::SWA);
    assert(got == n - start);
    auto r = t.insert(h, n, s.data(), (int)s.size(), /*start=*/0,
                      /*auto_recycle=*/false, ComponentType::SWA);
    assert(r.error == InsertError::OK);
    if (pos) {
        for (int p = start; p < n; p++) {
            int32_t sl = s[(size_t)(p - start)];
            if (std::find(r.unused_slots.begin(), r.unused_slots.end(), sl) ==
                r.unused_slots.end())
                (*pos)[(size_t)p] = sl;
        }
    }
    t.recycle_slots(r.unused_slots.data(), (int)r.unused_slots.size(),
                    ComponentType::SWA);
}

static int32_t alloc_mamba(RadixShmemTree& t) {
    std::vector<int32_t> s;
    int got = t.allocate_slots(1, s, ComponentType::MAMBA);
    assert(got == 1);
    return s[0];
}

static int32_t mamba_ok(RadixShmemTree& t, const uint64_t* h, int n) {
    int32_t s = alloc_mamba(t);
    auto r = t.insert(h, n, &s, 1, /*start=*/0, /*auto_recycle=*/true,
                      ComponentType::MAMBA);
    assert(r.error == InsertError::OK && r.unused_slots.empty());
    return s;
}
static InsertError mamba_fail(RadixShmemTree& t, const uint64_t* h, int n) {
    int32_t s = alloc_mamba(t);
    auto r = t.insert(h, n, &s, 1, /*start=*/0, /*auto_recycle=*/true,
                      ComponentType::MAMBA);
    assert(r.unused_slots.size() == 1 && r.unused_slots[0] == s);
    return r.error;
}

static TreeComponentQueryResult qc(RadixShmemTree& t, const uint64_t* h, int n,
                                   uint32_t mask) {
    auto r = t.query_components(h, n, mask, /*lock=*/false,
                                /*update_meta=*/false);
    assert(r.status == RemoteStatus::OK);
    assert(r.common_hit <= r.topology_match_length);
    return r;
}
static TreeComponentQueryResult qp(RadixShmemTree& t, const uint64_t* h, int n,
                                   uint32_t mask) {
    auto r = t.query_components(h, n, mask, /*lock=*/true,
                                /*update_meta=*/false);
    assert(r.status == RemoteStatus::OK);
    return r;
}

static void pressure(RadixShmemTree& t, int n) {
    std::vector<int32_t> tmp;
    int got = t.allocate_slots(n, tmp);
    assert(got == n);
    t.recycle_slots(tmp.data(), got);
}

static int hit(RadixShmemTree& t, const uint64_t* h, int n) {
    return (int)t.query_components(h, n, COMPONENT_MASK_FULL, false, false).common_hit;
}

static void test_swa_any_boundary_acceptance() {
    TEST_BEGIN("match: SWA accepts any boundary whose window is live");

    const char* name = "/test_cq_swa_end";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(/*w=*/7, /*cs=*/3, 64));
    auto h = seq(1000, 12);
    insert_fresh(t, h.data(), 12);
    std::vector<int32_t> ws(12, INVALID_COMPONENT_SLOT);
    for (int e : {3, 6, 9, 12}) publish_at(t, h.data(), e, 7, &ws);

    auto full = qc(t, h.data(), 12, FS);
    assert(full.common_hit == 12);
    assert(full.swa_start == 5 && full.swa_slots.size() == 7);
    for (int i = 0; i < 7; i++)
        assert(full.swa_slots[(size_t)i] == ws[(size_t)(5 + i)]);

    auto shallow = qc(t, h.data(), 3, FS);
    assert(shallow.common_hit == 3 && shallow.swa_start == 0 &&
           shallow.swa_slots.size() == 3);

    auto mid = qc(t, h.data(), 5, FS);
    assert(mid.common_hit == 5 && mid.topology_match_length == 5);
    assert(mid.swa_start == 0 && mid.swa_slots.size() == 5);
    for (int i = 0; i < 5; i++)
        assert(mid.swa_slots[(size_t)i] == ws[(size_t)i]);

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_swa_gap_fallback_and_reaccept() {
    TEST_BEGIN("match: gaps reject nearby ENDs; deeper END re-accepts");

    const char* name = "/test_cq_swa_gap";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 3, 128));
    auto h = seq(2000, 12);
    insert_fresh(t, h.data(), 12);
    for (int e : {3, 6, 9}) publish_at(t, h.data(), e, 4);

    auto gap = qc(t, h.data(), 12, FS);
    assert(gap.common_hit == 9);
    assert(gap.swa_start == 5 && gap.swa_slots.size() == 4);

    auto h2 = seq(3000, 12);
    insert_fresh(t, h2.data(), 12);
    for (int e : {6, 9, 12}) publish_at(t, h2.data(), e, 4);
    auto reacc = qc(t, h2.data(), 12, FS);
    assert(reacc.common_hit == 12 && reacc.swa_start == 8);
    auto blocked = qc(t, h2.data(), 5, FS);
    assert(blocked.common_hit == 0);

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_mamba_node_end() {
    TEST_BEGIN("match: Mamba on exact node ENDs, first-writer-wins");

    const char* name = "/test_cq_mamba";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 4, 64));
    auto h = seq(4000, 12);
    auto fs = insert_fresh(t, h.data(), 12);

    int32_t m6 = mamba_ok(t, h.data(), 6);
    auto hx = seq(9000, 4);
    assert(mamba_fail(t, hx.data(), 4) == InsertError::FULL_PATH_MISSING);
    int32_t m8 = mamba_ok(t, h.data(), 8);
    assert(mamba_fail(t, h.data(), 8) == InsertError::OK);

    auto r6 = qc(t, h.data(), 7, FM);
    assert(r6.common_hit == 6 && r6.mamba_checkpoint_slot == m6);
    auto r8 = qc(t, h.data(), 12, FM);
    assert(r8.common_hit == 8 && r8.mamba_checkpoint_slot == m8);
    assert(r8.full_slots == fs);

    int32_t m12 = mamba_ok(t, h.data(), 12);
    auto part = qc(t, h.data(), 10, FM);
    assert(part.common_hit == 8 && part.mamba_checkpoint_slot == m8);
    auto r12 = qc(t, h.data(), 12, FM);
    assert(r12.common_hit == 12 && r12.mamba_checkpoint_slot == m12);

    auto monly = qc(t, h.data(), 12, COMPONENT_MASK_MAMBA);
    assert(monly.common_hit == 12 && monly.mamba_checkpoint_slot == m12);
    assert(monly.full_slots.empty() && monly.swa_slots.empty());

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_joint_common_hit() {
    TEST_BEGIN("match: joint common_hit is the deepest all-accepted END");

    const char* name = "/test_cq_joint";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 4, 64));
    auto h = seq(5000, 12);
    insert_fresh(t, h.data(), 12);
    std::vector<int32_t> ws(12, INVALID_COMPONENT_SLOT);
    for (int e : {4, 8}) publish_at(t, h.data(), e, 4, &ws);
    int32_t m8 = mamba_ok(t, h.data(), 8);
    int32_t m12 = mamba_ok(t, h.data(), 12);
    (void)m12;

    auto r = qc(t, h.data(), 12, FSM);
    assert(r.common_hit == 8 && r.mamba_checkpoint_slot == m8);
    assert(r.full_slots.size() == 12);
    assert(r.swa_start == 4 && r.swa_slots.size() == 4);

    auto rs = qc(t, h.data(), 12, FS);
    assert(rs.common_hit == 8);
    for (size_t i = 0; i < rs.swa_slots.size(); i++)
        assert(rs.swa_slots[i] == ws[rs.swa_start + i]);

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

// Bottom-up intersection: Mamba's deepest END fails SWA twice before the
// alternating descent settles, or empties when no END has a live window.
static void test_joint_alternating_descent() {
    TEST_BEGIN("match: joint descent alternates SWA/Mamba until settled");

    const char* name = "/test_cq_alt";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 4, 64));
    auto h = seq(5200, 12);
    insert_fresh(t, h.data(), 12);           // nodes [0,4) [4,8) [8,12)
    int32_t m4 = mamba_ok(t, h.data(), 4);
    int32_t m8 = mamba_ok(t, h.data(), 8);
    int32_t m12 = mamba_ok(t, h.data(), 12);
    (void)m8; (void)m12;
    publish_at(t, h.data(), 9, 4);           // SWA live only on [5,9)

    auto s_only = qc(t, h.data(), 12, COMPONENT_MASK_SWA);
    assert(s_only.common_hit == 9 && s_only.swa_start == 5);
    auto m_only = qc(t, h.data(), 12, COMPONENT_MASK_MAMBA);
    assert(m_only.common_hit == 12);
    auto joint = qc(t, h.data(), 12, FSM);   // 12 -> swa 9 -> mamba 8 -> swa 0
    assert(joint.common_hit == 0 && joint.topology_match_length == 12);
    assert(joint.full_slots.size() == 12 && joint.swa_slots.empty() &&
           joint.mamba_checkpoint_slot == INVALID_COMPONENT_SLOT);

    publish_at(t, h.data(), 4, 4);           // [0,4) live: settles at END 4
    auto j4 = qc(t, h.data(), 12, FSM);
    assert(j4.common_hit == 4 && j4.mamba_checkpoint_slot == m4);
    assert(j4.swa_start == 0 && j4.swa_slots.size() == 4);

    // A short prefix under W: all-live accepts at f, one dead block rejects.
    auto short_ok = qc(t, h.data(), 3, FS);
    assert(short_ok.common_hit == 3 && short_ok.swa_slots.size() == 3);
    auto h2 = seq(5300, 3);
    insert_fresh(t, h2.data(), 3);
    auto short_dead = qc(t, h2.data(), 3, FS);
    assert(short_dead.common_hit == 0);

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_mask_validation_and_node_fww() {
    TEST_BEGIN("match: masks fail closed; republish keeps first-writer slots");

    const char* name = "/test_cq_mask";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(/*w=*/8, /*cs=*/4, 32));
    auto h = seq(6000, 8);
    insert_fresh(t, h.data(), 8);

    assert(t.query_components(h.data(), 8, 0, false, true).status ==
           RemoteStatus::BAD_REQUEST);
    assert(t.query_components(h.data(), 8, 1u << 7, false, true).status ==
           RemoteStatus::BAD_REQUEST);
    auto* hdr = reinterpret_cast<ShmHeader*>(t.shm_base());
    ComponentConfig saved = hdr->component_config;
    hdr->component_config.swa_window_blocks = 0;
    assert(t.query_components(h.data(), 8, FS, false, true).status ==
           RemoteStatus::BAD_REQUEST);
    hdr->component_config = saved;
    auto z = qc(t, nullptr, 0, FSM);
    assert(z.common_hit == 0);

    std::vector<int32_t> pos(8, INVALID_COMPONENT_SLOT);
    publish_at(t, h.data(), 6, 8, &pos);
    auto cutq = qc(t, h.data(), 8, FS);
    assert(cutq.common_hit == 6 && cutq.topology_match_length == 8);

    publish_at(t, h.data(), 8, 8, &pos);
    auto done = qc(t, h.data(), 8, FS);
    assert(done.common_hit == 8 && done.swa_slots.size() == 8);
    for (int i = 0; i < 8; i++)
        assert(done.swa_slots[(size_t)i] == pos[(size_t)i]);

    std::vector<int32_t> b;
    assert(t.allocate_slots(8, b, ComponentType::SWA) == 8);
    auto r = t.insert_swa(h.data(), 8, b.data(), 8, /*auto_recycle=*/false);
    assert(r.error == InsertError::OK && !r.topology_changed);
    assert(r.unused_slots == b);
    t.recycle_slots(b.data(), 8, ComponentType::SWA);
    auto again = qc(t, h.data(), 8, FS);
    for (int i = 0; i < 8; i++)
        assert(again.swa_slots[(size_t)i] == pos[(size_t)i]);

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_swa_publish_window_refresh() {
    TEST_BEGIN("publish: exact window, live blocks keep slots and take the stamp");

    const char* name = "/test_cq_cuts";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(/*w=*/4, /*cs=*/8, 32));
    auto h = seq(7000, 8);
    insert_fresh(t, h.data(), 8);

    auto hx = seq(7900, 8);
    std::vector<int32_t> s;
    assert(t.allocate_slots(4, s, ComponentType::SWA) == 4);
    auto miss = t.insert_swa(hx.data(), 8, s.data(), 4, false);
    assert(miss.error == InsertError::FULL_PATH_MISSING);
    assert(miss.unused_slots == s && !miss.topology_changed);

    // num_slots must be in [1, n]: zero and more than the path both refuse whole.
    auto none = t.insert_swa(h.data(), 8, s.data(), 0, false);
    assert(none.error == InsertError::BAD_REQUEST && none.unused_slots.empty());
    auto over = t.insert_swa(h.data(), 3, s.data(), 4, false);
    assert(over.error == InsertError::BAD_REQUEST && over.unused_slots == s);

    // Window [2,6) at n=6: no cut, the single node stays [0,8).
    auto two = t.insert_swa(h.data(), 6, s.data(), 4, false);
    assert(two.error == InsertError::OK && !two.topology_changed);
    assert(two.unused_slots.empty());
    assert(t.get_node_boundaries(h.data(), 8).size() == 1);
    auto q6 = qc(t, h.data(), 8, FS);
    assert(q6.common_hit == 6 && q6.swa_start == 2);
    for (int i = 0; i < 4; i++)
        assert(q6.swa_slots[(size_t)i] == s[(size_t)i]);

    // Window [4,8) at n=8: [4,6) keeps s[2], s[3] and returns the new slots.
    std::vector<int32_t> s2;
    assert(t.allocate_slots(4, s2, ComponentType::SWA) == 4);
    auto full = t.insert_swa(h.data(), 8, s2.data(), 4, false);
    assert(full.error == InsertError::OK);
    assert((full.unused_slots == std::vector<int32_t>(s2.begin(), s2.begin() + 2)));
    t.recycle_slots(full.unused_slots.data(), 2, ComponentType::SWA);
    auto q8 = qc(t, h.data(), 8, FS);
    assert(q8.common_hit == 8 && q8.swa_slots.size() == 4);
    assert(q8.swa_slots[0] == s[2] && q8.swa_slots[1] == s[3] &&
           q8.swa_slots[2] == s2[2] && q8.swa_slots[3] == s2[3]);

    // Republishing a fully live window returns every slot; the blocks are
    // re-stamped (the node key stays: [2,4) from the first publish is older).
    auto* hdr = reinterpret_cast<ShmHeader*>(t.shm_base());
    auto block_ts = [&](int b) {
        const uint32_t nid = t.get_node_boundaries(h.data(), 8)[0].tree_node_id;
        auto* node = reinterpret_cast<TreeNode*>(
                         (char*)t.shm_base() + hdr->node_pool_offset) + nid;
        const int32_t slot = reinterpret_cast<const int32_t*>(
            (char*)t.shm_base() + node->data_offset + 12u * node->num_blocks)[b];
        return reinterpret_cast<const SlotLruEntry*>(
            (char*)t.shm_base() + hdr->swa_lru_offset)[slot].ts;
    };
    const uint64_t ts7 = block_ts(7), ts3 = block_ts(3);
    std::vector<int32_t> s3;
    assert(t.allocate_slots(4, s3, ComponentType::SWA) == 4);
    auto again = t.insert_swa(h.data(), 8, s3.data(), 4, false);
    assert(again.error == InsertError::OK && again.unused_slots == s3);
    t.recycle_slots(s3.data(), 4, ComponentType::SWA);
    assert(block_ts(7) > ts7 && block_ts(3) == ts3);

    auto h2 = seq(7500, 4);
    std::vector<int32_t> fs;
    assert(t.allocate_slots(4, fs) == 4);
    auto fr = t.insert(h2.data(), 4, fs.data(), 4, /*start=*/0,
                       /*auto_recycle=*/true, ComponentType::FULL);
    assert(fr.error == InsertError::OK && hit(t, h2.data(), 4) == 4);
    int32_t m4 = mamba_ok(t, h2.data(), 4);
    auto rq = qc(t, h2.data(), 4, FM);
    assert(rq.common_hit == 4 && rq.mamba_checkpoint_slot == m4);
    int32_t m2 = alloc_mamba(t);
    auto mc = t.insert(h2.data(), 2, &m2, 1, /*start=*/0,
                       /*auto_recycle=*/true, ComponentType::MAMBA);
    assert(mc.error == InsertError::OK && mc.topology_changed);
    auto rq2 = qc(t, h2.data(), 2, FM);
    assert(rq2.common_hit == 2 && rq2.mamba_checkpoint_slot == m2);
    std::vector<int32_t> ms;
    assert(t.allocate_slots(2, ms, ComponentType::MAMBA) == 2);
    auto nm = t.insert(h2.data(), 4, ms.data(), 2, /*start=*/0,
                       /*auto_recycle=*/true, ComponentType::MAMBA);
    assert(nm.error == InsertError::BAD_REQUEST && nm.unused_slots == ms);

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_swa_positional_publish() {
    TEST_BEGIN("publish: SWA covers the trailing num_slots cells; W only gates the query");

    const char* name = "/test_cq_pos";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(/*w=*/4, /*cs=*/8, 64));

    // One position at a time (decode appends a block per step): every publish
    // covers [n-1, n) and the window at n is live as soon as its cells are.
    auto h = seq(8100, 12);
    insert_fresh(t, h.data(), 12);
    std::vector<int32_t> one(12);
    for (int n = 1; n <= 12; n++) {
        std::vector<int32_t> s;
        assert(t.allocate_slots(1, s, ComponentType::SWA) == 1);
        auto r = t.insert_swa(h.data(), n, s.data(), 1, false);
        assert(r.error == InsertError::OK && r.unused_slots.empty());
        one[(size_t)(n - 1)] = s[0];
        auto q = qc(t, h.data(), n, FS);
        assert((int)q.common_hit == n);
        assert(q.swa_start == (uint32_t)(n > 4 ? n - 4 : 0));
        for (uint32_t p = q.swa_start; p < (uint32_t)n; p++)
            assert(q.swa_slots[p - q.swa_start] == one[p]);
    }

    // A long node in one call: 10 cells from the root, n slots for n positions.
    auto h2 = seq(8200, 10);
    insert_fresh(t, h2.data(), 10);
    std::vector<int32_t> dense;
    assert(t.allocate_slots(10, dense, ComponentType::SWA) == 10);
    auto d = t.insert_swa(h2.data(), 10, dense.data(), 10, false);
    assert(d.error == InsertError::OK && d.unused_slots.empty());
    auto q10 = qc(t, h2.data(), 10, FS);
    assert(q10.common_hit == 10 && q10.swa_start == 6);
    auto q3 = qc(t, h2.data(), 3, FS);   // shorter than W: [0,3) all live
    assert(q3.common_hit == 3 && q3.swa_start == 0 && q3.swa_slots.size() == 3);

    // A tail-only publish leaves the head of the node unpublished: the window
    // at the end is short of W and nothing before it is live, so no hit until
    // the head is filled in by a later positional publish.
    auto h3 = seq(8300, 8);
    insert_fresh(t, h3.data(), 8);
    std::vector<int32_t> tail;
    assert(t.allocate_slots(2, tail, ComponentType::SWA) == 2);
    assert(t.insert_swa(h3.data(), 8, tail.data(), 2, false).error ==
           InsertError::OK);
    assert(qc(t, h3.data(), 8, FS).common_hit == 0);
    std::vector<int32_t> head;
    assert(t.allocate_slots(6, head, ComponentType::SWA) == 6);
    auto hd = t.insert_swa(h3.data(), 6, head.data(), 6, false);
    assert(hd.error == InsertError::OK && hd.unused_slots.empty());
    auto q8 = qc(t, h3.data(), 8, FS);
    assert(q8.common_hit == 8 && q8.swa_start == 4);
    assert(q8.swa_slots[0] == head[4] && q8.swa_slots[1] == head[5] &&
           q8.swa_slots[2] == tail[0] && q8.swa_slots[3] == tail[1]);

    // Re-publishing live cells is block-level FWW: every slot comes back.
    std::vector<int32_t> dup;
    assert(t.allocate_slots(3, dup, ComponentType::SWA) == 3);
    auto f = t.insert_swa(h3.data(), 8, dup.data(), 3, false);
    assert(f.error == InsertError::OK && f.unused_slots == dup);
    t.recycle_slots(dup.data(), 3, ComponentType::SWA);

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_update_meta_component_gating() {
    TEST_BEGIN("match: only a requested FULL protects Full recency");

    for (int use_full = 0; use_full <= 1; use_full++) {
        const char* name = "/test_cq_meta";
        shm_unlink(name);
        auto t = RadixShmemTree::create(name, cfg(4, 4, 8));
        auto ha = seq(7000, 4), hb = seq(7100, 4);
        insert_fresh(t, ha.data(), 4);
        insert_fresh(t, hb.data(), 4);
        publish_at(t, ha.data(), 4, 4);

        uint32_t mask = use_full ? FS : COMPONENT_MASK_SWA;
        auto r = t.query_components(ha.data(), 4, mask, /*lock=*/false,
                                    /*update_meta=*/true);
        assert(r.status == RemoteStatus::OK && r.common_hit == 4);

        pressure(t, 4);
        if (use_full) {
            assert(hit(t, ha.data(), 4) == 4 && hit(t, hb.data(), 4) == 0);
        } else {
            assert(hit(t, ha.data(), 4) == 0 && hit(t, hb.data(), 4) == 4);
        }
        expect_healthy(t);
        t.destroy();
    }
    TEST_PASS();
}

static void test_update_meta_touches_full_extent() {
    TEST_BEGIN("match: a requested FULL refreshes recency to f, not common_hit");

    const char* name = "/test_cq_meta_f";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 4, 12));
    auto ha = seq(7200, 8), hb = seq(7300, 4);
    insert_fresh(t, ha.data(), 8);     // A[0,4) -> B[4,8): B is the older leaf
    insert_fresh(t, hb.data(), 4);     // newer leaf
    publish_at(t, ha.data(), 4, 4);    // SWA LIVE on A only: joint hit stops at 4

    auto r = t.query_components(ha.data(), 8, FS, /*lock=*/false,
                                /*update_meta=*/true);
    assert(r.status == RemoteStatus::OK && r.common_hit == 4 &&
           r.topology_match_length == 8);

    // Pool is full: evicting 4 blocks must take hb, not B — the Full touch
    // reached B even though common_hit stopped at A's END.
    pressure(t, 4);
    assert(hit(t, ha.data(), 8) == 8 && hit(t, hb.data(), 4) == 0);

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

// ==================================================================
static void test_component_slot_allocation() {
    TEST_BEGIN("alloc: component pools are all-or-none; recycle balances");

    const char* name = "/test_cq_alloc";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 4, 16));

    std::vector<int32_t> v;
    const auto invalid_component = static_cast<ComponentType>(999u);
    assert(t.allocate_slots(1, v, invalid_component) == -1 && v.empty());
    assert(t.allocate_slots(0, v, ComponentType::SWA) == -1);
    assert(t.allocate_slots(17, v, ComponentType::SWA) == -1);
    assert(t.allocate_slots(65, v, ComponentType::MAMBA) == -1);

    std::vector<int32_t> a, b;
    assert(t.allocate_slots(12, a, ComponentType::SWA) == 12);
    assert(t.swa_mempool_free() == 4);
    assert(t.allocate_slots(5, b, ComponentType::SWA) == 0);
    assert(b.empty() && t.swa_mempool_free() == 4);
    assert(t.allocate_slots(4, b, ComponentType::SWA) == 4);
    assert(t.swa_mempool_free() == 0);

    t.recycle_slots(a.data(), (int)a.size(), ComponentType::SWA);
    assert(t.swa_mempool_free() == 12);
    t.recycle_slots(a.data(), (int)a.size(), ComponentType::SWA);
    assert(t.swa_mempool_free() == 12);
    t.recycle_slots(b.data(), (int)b.size(), ComponentType::SWA);
    assert(t.swa_mempool_free() == 16);

    auto h = seq(1000, 8);
    auto full_slots = insert_fresh(t, h.data(), 8);
    const uint32_t full_free = t.mempool_free();
    int32_t aliased_full_slot = full_slots[0];
    auto invalid = t.insert(h.data(), 1, &aliased_full_slot, 1,
                            /*start=*/0, /*auto_recycle=*/true,
                            invalid_component);
    assert(invalid.error == InsertError::BAD_REQUEST);
    assert(invalid.unused_slots == std::vector<int32_t>{aliased_full_slot});
    t.recycle_slots(&aliased_full_slot, 1, invalid_component);
    assert(t.mempool_free() == full_free && hit(t, h.data(), 8) == 8);
    for (int e : {4, 8}) publish_at(t, h.data(), e, 4);
    mamba_ok(t, h.data(), 8);
    assert(t.swa_mempool_free() == 8);
    assert(t.mamba_mempool_free() == 63);
    pressure(t, 16);
    assert(hit(t, h.data(), 8) == 0);
    assert(t.swa_mempool_free() == 16);
    assert(t.mamba_mempool_free() == 64);

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_mamba_merge_keeps_checkpoint() {
    TEST_BEGIN("merge: a parent END checkpoint defers the fold and survives");

    const char* name = "/test_cq_merge";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 4, 32));
    auto h = seq(1000, 8);
    insert_fresh(t, h.data(), 4);
    mamba_ok(t, h.data(), 4);
    assert(t.mamba_mempool_free() == 63);

    // Sequential extension: no auto-merge, the END 4 checkpoint stays.
    insert_fresh(t, h.data(), 8, /*start=*/4);
    assert(t.get_node_boundaries(h.data(), 8).size() == 2);
    assert(t.mamba_mempool_free() == 63);
    assert(hit(t, h.data(), 8) == 8);
    assert(qc(t, h.data(), 4, FM).common_hit == 4);
    assert(qc(t, h.data(), 8, FM).common_hit == 4);

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_pin_release_lifecycle() {
    TEST_BEGIN("pin: three refs block eviction; caller-keyed release unblocks");

    const char* name = "/test_cq_pin";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 4, 8));
    auto ha = seq(1000, 4), hb = seq(2000, 4);
    auto sa = insert_fresh(t, ha.data(), 4);
    insert_fresh(t, hb.data(), 4);
    publish_at(t, ha.data(), 4, 4);
    int32_t m4 = mamba_ok(t, ha.data(), 4);

    auto r = qp(t, ha.data(), 4, FSM);
    assert(r.common_hit == 4 && !r.pin.empty());
    assert(r.pin.full.start == 0 && r.pin.full.end == 4);
    assert(r.pin.swa.start == 0 && r.pin.swa.end == 4);
    assert(r.pin.mamba_end == 4);
    assert(r.full_slots == sa && r.mamba_checkpoint_slot == m4);

    pressure(t, 4);
    assert(hit(t, ha.data(), 4) == 4 && hit(t, hb.data(), 4) == 0);

    assert(t.dec_ref(ha.data(), 4, std::move(r.pin)));
    pressure(t, 8);
    assert(hit(t, ha.data(), 4) == 0);

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_pin_per_lane_ends_and_overpin() {
    TEST_BEGIN("pin: Full to f, aux to h; SWA left-extends [S,L) to [A,L)");

    const char* name = "/test_cq_lanes";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 4, 64));
    auto h = seq(9000, 12);
    insert_fresh(t, h.data(), 4);
    std::vector<uint64_t> fork(h.begin(), h.begin() + 4);
    auto d = seq(9900, 4);
    fork.insert(fork.end(), d.begin(), d.end());
    insert_fresh(t, fork.data(), 8, /*start=*/4);
    insert_fresh(t, h.data(), 12, /*start=*/4);
    mamba_ok(t, h.data(), 4);

    auto r = qp(t, h.data(), 12, FM);
    assert(r.common_hit == 4 && r.topology_match_length == 12);
    assert(r.pin.full.start == 0 && r.pin.full.end == 12);
    assert(r.pin.swa.empty() && r.pin.mamba_end == 4);
    pressure(t, 8);
    assert(hit(t, h.data(), 12) == 12);
    assert(t.dec_ref(h.data(), 12, std::move(r.pin)));
    pressure(t, 64);
    assert(hit(t, h.data(), 12) == 0);

    const char* name2 = "/test_cq_overpin";
    shm_unlink(name2);
    auto t2 = RadixShmemTree::create(name2, cfg(7, 3, 64));
    auto h2 = seq(9700, 12);
    insert_fresh(t2, h2.data(), 12);
    for (int e : {3, 6, 9, 12}) publish_at(t2, h2.data(), e, 7);
    auto r2 = qp(t2, h2.data(), 12, COMPONENT_MASK_SWA);
    assert(r2.common_hit == 12);
    assert(r2.swa_start == 5 && r2.swa_slots.size() == 7);
    assert(r2.pin.swa.start == 3 && r2.pin.swa.end == 12);
    assert(r2.pin.full.empty());
    pressure(t2, 52);
    assert(hit(t2, h2.data(), 12) == 12);
    assert(t2.dec_ref(h2.data(), 12, std::move(r2.pin)));
    pressure(t2, 64);
    assert(hit(t2, h2.data(), 12) == 0);

    expect_healthy(t);
    expect_healthy(t2);
    t.destroy();
    t2.destroy();
    TEST_PASS();
}

static void test_pin_partial_terminal_then_split() {
    TEST_BEGIN("pin: partial terminal pinned whole; tail keys release past f");

    const char* name = "/test_cq_mid";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(8, 8, 16));
    auto ha = seq(4000, 8);
    insert_fresh(t, ha.data(), 8);
    insert_fresh(t, seq(4100, 8).data(), 8);

    auto r = qp(t, ha.data(), 5, COMPONENT_MASK_FULL);
    assert(r.common_hit == 5 && r.full_slots.size() == 5);
    assert(r.pin.full.start == 0 && r.pin.full.end == 8);
    assert(r.pin.tail.size() == 3 &&
           std::equal(r.pin.tail.begin(), r.pin.tail.end(),
                      ha.begin() + 5));
    pressure(t, 8);   // only the unpinned neighbour can go
    assert(hit(t, ha.data(), 8) == 8);

    // Split the pinned node past f: [0,6) copies the ref, [6,8) keeps it.
    std::vector<uint64_t> hd(ha.begin(), ha.begin() + 6);
    auto d = seq(4900, 2);
    hd.insert(hd.end(), d.begin(), d.end());
    insert_fresh(t, hd.data(), 8, /*start=*/6);
    assert(t.get_node_boundaries(ha.data(), 8).size() == 2);

    // Five caller keys plus the recorded tail reach [6,8) as well.
    assert(t.dec_ref(ha.data(), 5, std::move(r.pin)));
    pressure(t, 16);
    assert(hit(t, ha.data(), 8) == 0);

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_pin_swa_terminal_past_query() {
    TEST_BEGIN("pin: SWA lane on a terminal longer than the query releases");

    // cs=4 splits a 16-block insert into four nodes; a 10-block query ends
    // inside [8,12), so the pinned terminal reaches past the caller's keys.
    // Full records the two tail hashes; the SWA range must release with the
    // same ten caller keys (its last pinned node starts at 8 < 10).
    const char* name = "/test_cq_swa_tail";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 4, 64));
    auto h = seq(16000, 16);
    insert_fresh(t, h.data(), 16);
    assert(t.get_node_boundaries(h.data(), 16).size() == 4);
    publish_at(t, h.data(), 10, 4);   // SWA live on [6,10)

    auto r = qp(t, h.data(), 10, FS);
    assert(r.common_hit == 10 && r.topology_match_length == 10);
    assert(r.swa_start == 6 && r.swa_slots.size() == 4);
    assert(r.pin.full.start == 0 && r.pin.full.end == 12 &&
           r.pin.tail.size() == 2);
    assert(r.pin.swa.start == 4 && r.pin.swa.end == 12);
    // The release must not be refused because swa.end (12) exceeds n (10).
    assert(t.dec_ref(h.data(), 10, std::move(r.pin)));
    pressure(t, 64);
    assert(hit(t, h.data(), 16) == 0);
    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_pin_swa_only_terminal_past_query() {
    TEST_BEGIN("pin: SWA-only lock on a terminal longer than the query releases");

    // Same tree, SWA alone in the mask: no Full range, so the record has no
    // Full tail. The SWA range still ends at the terminal's node END (12);
    // release with the ten caller keys must reach it (N1 and N2 pinned).
    const char* name = "/test_cq_swa_only_tail";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 4, 64));
    auto h = seq(16100, 16);
    insert_fresh(t, h.data(), 16);
    assert(t.get_node_boundaries(h.data(), 16).size() == 4);
    publish_at(t, h.data(), 10, 4);   // SWA live on [6,10)

    auto r = qp(t, h.data(), 10, COMPONENT_MASK_SWA);
    assert(r.common_hit == 10 && r.topology_match_length == 10);
    assert(r.swa_start == 6 && r.swa_slots.size() == 4);
    assert(r.full_slots.empty() && r.pin.full.empty());
    assert(r.pin.swa.start == 4 && r.pin.swa.end == 12);
    assert(t.dec_ref(h.data(), 10, std::move(r.pin)));
    pressure(t, 64);
    assert(hit(t, h.data(), 16) == 0);

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_pinned_split_then_release() {
    TEST_BEGIN("pin: split copies Full/SWA refs; release re-walks and balances");

    const char* name = "/test_cq_split";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 8, 32));
    auto ha = seq(5000, 8);
    insert_fresh(t, ha.data(), 8);
    publish_at(t, ha.data(), 8, 4);
    int32_t m8s = mamba_ok(t, ha.data(), 8);

    auto r = qp(t, ha.data(), 8, FSM);
    assert(r.common_hit == 8 && r.pin.mamba_end == 8);

    std::vector<uint64_t> hd(ha.begin(), ha.begin() + 5);
    auto dt = seq(5900, 3);
    hd.insert(hd.end(), dt.begin(), dt.end());
    insert_fresh(t, hd.data(), 8, /*start=*/5);
    assert(qc(t, ha.data(), 8, FM).mamba_checkpoint_slot == m8s);

    insert_fresh(t, seq(5100, 8).data(), 8);
    pressure(t, 14);
    assert(hit(t, ha.data(), 8) == 8);
    assert(hit(t, hd.data(), 8) == 5);

    assert(t.dec_ref(ha.data(), 8, std::move(r.pin)));
    pressure(t, 24);
    assert(hit(t, ha.data(), 8) == 0);

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_release_trust_model() {
    TEST_BEGIN("release: unconditional clamp; short/missing keys refused");

    const char* name = "/test_cq_release";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(8, 4, 32));
    auto h = seq(10000, 8);
    insert_fresh(t, h.data(), 8);

    auto r1 = qp(t, h.data(), 8, COMPONENT_MASK_FULL);
    auto r2 = qp(t, h.data(), 8, COMPONENT_MASK_FULL);
    PinRecord extra = r1.pin;
    assert(t.dec_ref(h.data(), 8, std::move(r1.pin)));
    pressure(t, 24);
    assert(hit(t, h.data(), 8) == 8);
    assert(t.dec_ref(h.data(), 8, std::move(r2.pin)));
    assert(t.dec_ref(h.data(), 8, std::move(extra)));
    pressure(t, 32);
    assert(hit(t, h.data(), 8) == 0);

    insert_fresh(t, h.data(), 8);
    auto r3 = qp(t, h.data(), 8, COMPONENT_MASK_FULL);
    PinRecord copy = r3.pin;
    assert(!t.dec_ref(h.data(), 2, std::move(copy)));
    PinRecord copy2 = r3.pin;
    assert(!t.dec_ref(nullptr, 0, std::move(copy2)));
    auto miss = qp(t, seq(20000, 4).data(), 4, COMPONENT_MASK_FULL);
    assert(miss.pin.empty());
    assert(t.dec_ref(h.data(), 8, std::move(miss.pin)));
    assert(t.dec_ref(h.data(), 8, std::move(r3.pin)));

    const char* name2 = "/test_cq_release_long";
    shm_unlink(name2);
    auto t2 = RadixShmemTree::create(name2, cfg(4, 1, 64));
    auto h2 = seq(11000, 20);
    insert_fresh(t2, h2.data(), 20);
    auto r4 = qp(t2, h2.data(), 20, COMPONENT_MASK_FULL);
    assert(t2.dec_ref(h2.data(), 20, std::move(r4.pin)));
    pressure(t2, 64);
    assert(hit(t2, h2.data(), 20) == 0);

    expect_healthy(t);
    expect_healthy(t2);
    t.destroy();
    t2.destroy();
    TEST_PASS();
}

static void test_facade_shape_and_finalize() {
    TEST_BEGIN("facade: cropped source-tagged runs; one-shot finalize");

    const char* name = "/test_cq_facade";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 4, 64));
    auto h = seq(1000, 12);
    auto fs = insert_fresh(t, h.data(), 12);
    for (int e : {4, 8, 12}) publish_at(t, h.data(), e, 4);
    int32_t m8 = mamba_ok(t, h.data(), 8);

    RadixClient c(name);
    auto r = c.query(h.data(), 12, FM, /*local_only=*/true,
                                /*lock=*/false, /*update_meta=*/false);
    assert(r.status == RemoteStatus::OK && r.common_hit == 8);
    assert(r.full_fragments.size() == 1 && r.full_fragments[0].source_rank == 0);
    assert(r.full_fragments[0].offset == 0);
    assert(r.full_fragments[0].slot_ids.size() == 8);
    for (int i = 0; i < 8; i++) assert(r.full_fragments[0].slot_ids[i] == fs[i]);
    assert(r.mamba && r.mamba->source_rank == 0);
    assert(r.mamba->slot_id == m8);
    assert(r.finalize.empty());

    auto rs = c.query(h.data(), 8, FSM, true, false, false);
    assert(rs.common_hit == 8);
    assert(rs.swa.source_rank == 0 && rs.swa.offset == 4 &&
           rs.swa.slot_ids.size() == 4);

    auto hx = seq(9000, 4);
    auto rm = c.query(hx.data(), 4, FSM, true, /*lock=*/true);
    assert(rm.status == RemoteStatus::OK && rm.common_hit == 0);
    assert(rm.full_fragments.empty() && rm.swa.source_rank == -1 && !rm.mamba);
    assert(rm.finalize.empty());
    rm.finalize();

    auto rr = c.query(h.data(), 8, COMPONENT_MASK_FULL,
                                 /*local_only=*/false);
    assert(rr.status == RemoteStatus::OK && rr.common_hit == 8);
    assert(rr.full_fragments.size() == 1 && rr.full_fragments[0].source_rank == 0);
    assert(rr.rdma_read_count == 0 && rr.rdma_atomic_count == 0);

    insert_fresh(t, seq(2000, 4).data(), 4);
    auto rl = c.query(h.data(), 8, FSM, true, /*lock=*/true,
                                 /*update_meta=*/false);
    assert(rl.common_hit == 8 && (bool)rl.finalize);
    pressure(t, 48);
    assert(hit(t, h.data(), 8) == 8);
    rl.finalize();
    assert(!(bool)rl.finalize && rl.finalize.empty());
    rl.finalize();
    pressure(t, 64);
    assert(hit(t, h.data(), 8) == 0);

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_insert_arg_validation() {
    TEST_BEGIN("publish: bad args refuse whole; a corrupted config fails closed");

    const char* name = "/test_cq_args";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 4, 32));
    auto h = seq(12000, 8);
    insert_fresh(t, h.data(), 8);
    std::vector<int32_t> s;
    assert(t.allocate_slots(4, s, ComponentType::SWA) == 4);
    const uint32_t swa_free = t.swa_mempool_free();

    auto r = t.insert_swa(h.data(), 0, s.data(), 0, true);
    assert(r.error == InsertError::BAD_REQUEST && r.unused_slots.empty());
    r = t.insert_swa(nullptr, 8, s.data(), 4, true);
    assert(r.error == InsertError::BAD_REQUEST && r.unused_slots == s);
    assert(t.swa_mempool_free() == swa_free + 4);
    assert(t.allocate_slots(4, s, ComponentType::SWA) == 4);
    r = t.insert_swa(h.data(), 8, nullptr, 4, true);
    assert(r.error == InsertError::BAD_REQUEST && r.unused_slots.empty());
    // more slots than positions refuses whole; a non-zero start is FULL-only
    r = t.insert_swa(h.data(), 3, s.data(), 4, false);
    assert(r.error == InsertError::BAD_REQUEST && r.unused_slots == s);
    r = t.insert(h.data(), 8, s.data(), 4, /*start=*/4, /*auto_recycle=*/false,
                 ComponentType::SWA);
    assert(r.error == InsertError::BAD_REQUEST && r.unused_slots == s);

    int32_t m = alloc_mamba(t);
    const uint32_t mamba_free = t.mamba_mempool_free();
    auto rm = t.insert_mamba(h.data(), 0, &m, 1, false);
    assert(rm.error == InsertError::BAD_REQUEST && rm.unused_slots.size() == 1);
    assert(t.mamba_mempool_free() == mamba_free);
    rm = t.insert_mamba(nullptr, 8, &m, 1, false);
    assert(rm.error == InsertError::BAD_REQUEST && rm.unused_slots.size() == 1);
    rm = t.insert_mamba(h.data(), 8, nullptr, 1, true);
    assert(rm.error == InsertError::BAD_REQUEST && rm.unused_slots.empty());

    auto* hdr = reinterpret_cast<ShmHeader*>(t.shm_base());
    ComponentConfig saved = hdr->component_config;
    hdr->component_config.swa_window_blocks = 0;
    r = t.insert_swa(h.data(), 8, s.data(), 4, false);
    assert(r.error == InsertError::BAD_REQUEST && r.unused_slots == s);
    rm = t.insert_mamba(h.data(), 8, &m, 1, false);
    assert(rm.error == InsertError::BAD_REQUEST && rm.unused_slots.size() == 1);
    hdr->component_config = saved;

    // FWW without auto_recycle: the loser is reported, not recycled.
    assert(t.insert_mamba(h.data(), 8, &m, 1, true).error == InsertError::OK);
    int32_t m2 = alloc_mamba(t);
    const uint32_t before = t.mamba_mempool_free();
    auto fww = t.insert_mamba(h.data(), 8, &m2, 1, false);
    assert(fww.error == InsertError::OK &&
           fww.unused_slots == std::vector<int32_t>{m2});
    assert(t.mamba_mempool_free() == before);
    t.recycle_slots(&m2, 1, ComponentType::MAMBA);
    assert(t.mamba_mempool_free() == before + 1);
    t.recycle_slots(s.data(), 4, ComponentType::SWA);

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_mamba_only_pin_release() {
    TEST_BEGIN("pin: a MAMBA-only lock pins the END node; release keys off mamba_end");

    const char* name = "/test_cq_mpin";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 4, 8));
    auto ha = seq(13000, 4), hb = seq(13100, 4);
    insert_fresh(t, ha.data(), 4);
    insert_fresh(t, hb.data(), 4);
    int32_t m = mamba_ok(t, ha.data(), 4);

    auto r = qp(t, ha.data(), 4, COMPONENT_MASK_MAMBA);
    assert(r.common_hit == 4 && r.mamba_checkpoint_slot == m);
    assert(r.pin.full.empty() && r.pin.swa.empty() && r.pin.mamba_end == 4);
    pressure(t, 4);
    assert(hit(t, ha.data(), 4) == 4 && hit(t, hb.data(), 4) == 0);
    assert(t.dec_ref(ha.data(), 4, std::move(r.pin)));
    pressure(t, 8);
    assert(hit(t, ha.data(), 4) == 0);

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_facade_bad_mask_and_crop_to_zero() {
    TEST_BEGIN("facade: bad masks refuse; a Full-only hit under FS yields no fragments");

    const char* name = "/test_cq_facade2";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 4, 32));
    auto h = seq(2500, 8);
    insert_fresh(t, h.data(), 8);
    RadixClient c(name);
    assert(c.query(h.data(), 8, 0, true).status == RemoteStatus::BAD_REQUEST);
    assert(c.query(h.data(), 8, 1u << 7, true).status == RemoteStatus::BAD_REQUEST);
    assert(c.query(nullptr, 8, FS, true).status == RemoteStatus::BAD_REQUEST);

    // Full matched to 8 but no SWA: common_hit 0 crops the run away; the
    // Full pin over [0,8) is still taken and finalize releases it.
    auto r = c.query(h.data(), 8, FS, true, /*lock=*/true);
    assert(r.status == RemoteStatus::OK && r.common_hit == 0);
    assert(r.full_fragments.empty() && r.swa.source_rank == -1 && !r.mamba);
    assert((bool)r.finalize);
    pressure(t, 24);
    assert(hit(t, h.data(), 8) == 8);
    r.finalize();
    pressure(t, 32);
    assert(hit(t, h.data(), 8) == 0);

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

#ifdef RADIXSHMEM_NO_RDMA
static void test_nordma_client_boundaries() {
    TEST_BEGIN("no RDMA: local access works and distributed entry points reject");
    auto requires_rdma = [](auto&& action) {
        bool rejected = false;
        try {
            action();
        } catch (const std::runtime_error& error) {
            rejected = std::string(error.what()).find("RDMA") != std::string::npos;
        }
        assert(rejected);
    };

    RadixClientConfig client_config;
    requires_rdma([&] { RadixClient client(client_config); });
    const std::string name = "/test_cq_nordma_" + std::to_string(::getpid());
    for (int criterion = 0; criterion < 3; ++criterion) {
        RadixServerConfig server_config;
        server_config.name = name;
        if (criterion == 0) server_config.world_size = 2;
        if (criterion == 1) server_config.num_rht_shards = 2;
        if (criterion == 2) server_config.expected_min_nodes = 2;
        requires_rdma([&] {
            RadixServer server(server_config);
            server.bootstrap();
        });
    }

    auto tree = RadixShmemTree::create(name, cfg(4, 4, 32));
    auto hashes = seq(4000, 4);
    {
        RadixClient client(name);
        assert(!client.is_distributed() && client.world_size() == 1);
        assert(insert_auto(client, hashes.data(), 4).error == InsertError::OK);
    }
    {
        LocalTreeConnector borrowed(name);
        requires_rdma([&] { RadixClient client(client_config, &borrowed); });
        assert(borrowed.check_health() == 0);
    }
    auto* header = static_cast<ShmHeader*>(tree.shm_base());
    header->is_distributed = 1;
    header->my_rank = 3;
    requires_rdma([&] { RadixClient client(name); });
    {
        RadixClient reader(name, true);
        assert(!reader.is_distributed() && reader.rank() == 3);
        auto result = reader.query(hashes.data(), 4, COMPONENT_MASK_FULL);
        assert(result.status == RemoteStatus::OK && result.common_hit == 4);
        bool rejected = false;
        try {
            std::vector<int32_t> slots;
            reader.allocate_slots(1, slots);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        assert(rejected);
    }
    expect_healthy(tree);
    tree.destroy();
    TEST_PASS();
}
#endif

static void test_client_component_insert() {
    TEST_BEGIN("facade: unified insert publishes every component end to end");

    const char* name = "/test_cq_client_pub";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 4, 32));
    RadixClient c(name);
    auto h = seq(3000, 8);

    std::vector<int32_t> fs;
    assert(c.allocate_slots(8, fs) == 8);
    assert(c.insert(h.data(), 8, fs.data(), 8).error == InsertError::OK);

    std::vector<int32_t> ws, ms;
    assert(c.allocate_slots(4, ws, ComponentType::SWA) == 4);
    assert(c.insert(h.data(), 8, ws.data(), 4, /*start=*/0,
                    /*auto_recycle=*/true,
                    ComponentType::SWA).error == InsertError::OK);
    assert(c.allocate_slots(1, ms, ComponentType::MAMBA) == 1);
    assert(c.insert(h.data(), 8, ms.data(), 1, /*start=*/0,
                    /*auto_recycle=*/true,
                    ComponentType::MAMBA).error == InsertError::OK);

    auto q = c.query(h.data(), 8, FSM, /*local_only=*/true,
                     /*lock=*/true, /*update_meta=*/false);
    assert(q.status == RemoteStatus::OK && q.common_hit == 8);
    assert(q.swa.slot_ids.size() == 4 && q.mamba && q.mamba->slot_id == ms[0]);
    q.finalize();

    pressure(t, 32);
    assert(hit(t, h.data(), 8) == 0);
    assert(t.swa_mempool_free() == t.swa_mempool_total());
    assert(t.mamba_mempool_free() == t.mamba_mempool_total());

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

#ifndef RADIXSHMEM_NO_RDMA

static PeerEntry peer_entry_of(RadixShmemTree& t) {
    auto* hdr = reinterpret_cast<const ShmHeader*>(t.shm_base());
    PeerEntry e{};
    e.shm_size               = t.shm_size();
    e.node_pool_capacity     = hdr->node_pool_capacity;
    e.component_config       = hdr->component_config;
    e.component_state_offset = hdr->component_state_offset;
    return e;
}

struct Rig {
    FakeRdmaEndpoint ep;
    void* scratch;
    RdmaRemoteWalker walker;
    RemoteTreeInfo info;

    explicit Rig(RadixShmemTree& t)
        : scratch(aligned_alloc(4096, 1 << 20)),
          walker(ep, scratch, /*lkey=*/0, 1 << 20),
          info(walker.read_remote_info(
              RdmaMrInfo{/*rkey=*/1, (uint64_t)t.shm_base(), t.shm_size()})) {
        PeerEntry e = peer_entry_of(t);
        info.component_config           = e.component_config;
        info.component_state_offset     = e.component_state_offset;
        info.component_state_entry_size = (uint32_t)sizeof(NodeComponentState);
        info.component_info_valid =
            peer_component_abi_valid(e, t.component_config());
    }
    ~Rig() { free(scratch); }
};

static void assert_same(const TreeComponentQueryResult& a,
                        const TreeComponentQueryResult& b) {
    assert(a.status == b.status);
    assert(a.topology_match_length == b.topology_match_length);
    assert(a.common_hit == b.common_hit);
    assert(a.full_slots == b.full_slots);
    assert(a.swa_start == b.swa_start);
    assert(a.swa_slots == b.swa_slots);
    assert(a.mamba_checkpoint_slot == b.mamba_checkpoint_slot);
}

static void test_remote_parity_and_abi_gate() {
    TEST_BEGIN("remote: local == remote per mask/len; ABI fails closed");

    const char* name = "/test_cq_remote_diff";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 3, 128));
    auto h = seq(1000, 12);
    insert_fresh(t, h.data(), 12);
    for (int e : {3, 6, 9}) publish_at(t, h.data(), e, 4);
    mamba_ok(t, h.data(), 9);

    Rig r(t);
    const uint32_t masks[] = {COMPONENT_MASK_FULL, FS, FM, FSM,
                              COMPONENT_MASK_SWA, COMPONENT_MASK_MAMBA};
    const int lens[] = {12, 10, 9, 5, 3};
    for (uint32_t mask : masks) {
        for (int len : lens) {
            auto local = t.query_components(h.data(), len, mask, false, false);
            auto remote =
                r.walker.walk_components(r.info, h.data(), len, mask, false);
            assert_same(local, remote.tree);
            assert(remote.rdma_read_count > 0);
            assert(remote.rdma_atomic_count >= 2);
        }
    }
    auto hx = seq(9000, 4);
    auto rm = r.walker.walk_components(r.info, hx.data(), 4, FSM, false);
    assert(rm.tree.status == RemoteStatus::OK && rm.tree.common_hit == 0);

    const ComponentConfig local = t.component_config();
    PeerEntry good = peer_entry_of(t);
    assert(peer_component_abi_valid(good, local));
    PeerEntry e = good;
    e.component_config.abi_version = 1;
    assert(!peer_component_abi_valid(e, local));
    e = good;
    e.component_config.swa_window_blocks += 1;
    assert(!peer_component_abi_valid(e, local));
    e = good;
    e.component_state_offset = e.shm_size;
    assert(!peer_component_abi_valid(e, local));
    RemoteTreeInfo broken = r.info;
    broken.component_info_valid = false;
    auto res = r.walker.walk_components(broken, nullptr, 0, FSM, false);
    assert(res.tree.status == RemoteStatus::BAD_REQUEST);

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_remote_pin_incumbent_release() {
    TEST_BEGIN("remote: incumbent gates pin; FAA release balances");

    const char* name = "/test_cq_remote_pin";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 4, 8));
    auto ha = seq(2000, 4), hb = seq(2100, 4);
    auto sa = insert_fresh(t, ha.data(), 4);
    insert_fresh(t, hb.data(), 4);
    publish_at(t, ha.data(), 4, 4);
    mamba_ok(t, ha.data(), 4);

    Rig r(t);
    r.ep.reset_counters();
    auto res = r.walker.walk_components(r.info, ha.data(), 4, FSM,
                                        /*lock=*/true);
    assert(res.tree.status == RemoteStatus::OK && res.tree.common_hit == 4);
    assert(res.tree.full_slots == sa);
    assert(!res.remote_pin.empty());
    assert(res.remote_pin.full.end == 4 && res.remote_pin.swa.end == 4);
    assert(res.remote_pin.mamba_end == 4);
    assert(r.ep.faas == 5);

    pressure(t, 4);
    assert(hit(t, ha.data(), 4) == 4 && hit(t, hb.data(), 4) == 0);

    r.ep.reset_counters();
    assert(r.walker.dec_ref_components_remote(r.info, ha.data(), 4,
                                              std::move(res.remote_pin)));
    assert(r.ep.faas == 5);
    pressure(t, 8);
    assert(hit(t, ha.data(), 4) == 0);

    insert_fresh(t, ha.data(), 4);
    publish_at(t, ha.data(), 4, 4);
    r.ep.reset_counters();
    auto tie = r.walker.walk_components(r.info, ha.data(), 4, FS, true,
                                        /*incumbent=*/4, /*local_full_hit=*/4);
    assert(tie.tree.common_hit == 4 && tie.remote_pin.empty());
    assert(r.ep.faas == 2);

    r.ep.reset_counters();
    auto cut = r.walker.walk_components(r.info, ha.data(), 4, FS, true,
                                        /*incumbent=*/2, /*local_full_hit=*/2);
    assert(cut.tree.common_hit == 4 && !cut.remote_pin.empty());
    // The node straddling the cut c=2 serves [2,4): pinned whole.
    assert(cut.remote_pin.full.start == 0 && cut.remote_pin.full.end == 4 &&
           cut.remote_pin.tail.empty());
    assert(cut.remote_pin.swa.start == 0 && cut.remote_pin.swa.end == 4);
    assert(r.walker.dec_ref_components_remote(r.info, ha.data(), 4,
                                              std::move(cut.remote_pin)));

    // Partial terminal over the wire: pinned whole, tail keys release it.
    r.ep.reset_counters();
    auto part = r.walker.walk_components(r.info, ha.data(), 2,
                                         COMPONENT_MASK_FULL, /*lock=*/true);
    assert(part.tree.common_hit == 2 && part.remote_pin.full.end == 4 &&
           part.remote_pin.tail.size() == 2);
    assert(r.ep.faas == 3);
    r.ep.reset_counters();
    assert(r.walker.dec_ref_components_remote(r.info, ha.data(), 2,
                                              std::move(part.remote_pin)));
    assert(r.ep.faas == 3);
    pressure(t, 8);
    assert(hit(t, ha.data(), 4) == 0);

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_remote_throw_releases_reader_count() {
    TEST_BEGIN("remote: an op throwing mid-walk leaves no reader count");

    const char* name = "/test_cq_remote_throw";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 4, 8));
    auto h = seq(2600, 4);
    insert_fresh(t, h.data(), 4);
    publish_at(t, h.data(), 4, 4);
    mamba_ok(t, h.data(), 4);
    auto* hdr = reinterpret_cast<ShmHeader*>(t.shm_base());

    Rig r(t);
    for (int at = 2; at <= 6; at++) {
        r.ep.reset_counters();
        r.ep.fail_read_at = at;
        bool threw = false;
        try {
            r.walker.walk_components(r.info, h.data(), 4, FSM, /*lock=*/true);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        assert(threw);
        assert(hdr->rdma_reader_count.load() == 0);
    }
    r.ep.fail_read_at = 0;

    r.ep.reset_counters();
    auto res = r.walker.walk_components(r.info, h.data(), 4, FSM, /*lock=*/true);
    assert(res.tree.common_hit == 4 && !res.remote_pin.empty());
    r.ep.reset_counters();
    r.ep.fail_read_at = 2;
    bool threw = false;
    try {
        r.walker.dec_ref_components_remote(r.info, h.data(), 4,
                                           std::move(res.remote_pin));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
    assert(hdr->rdma_reader_count.load() == 0);

    t.destroy();
    TEST_PASS();
}

static void test_remote_hints_and_failstop() {
    TEST_BEGIN("remote: hint fast path parity; injected failures fail stop");

    const char* name = "/test_cq_remote_hint";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 1, 64));
    auto h = seq(6000, 12);
    auto d = seq(9600, 13);
    for (int k = 1; k <= 12; k++) {
        insert_fresh(t, h.data(), k, k - 1);
        if (k < 12) {
            std::vector<uint64_t> fork(h.begin(), h.begin() + k);
            fork.push_back(d[(size_t)k]);
            insert_fresh(t, fork.data(), k + 1, k);
        }
    }
    for (int e = 1; e <= 12; e++) publish_at(t, h.data(), e, 4);
    mamba_ok(t, h.data(), 12);

    Rig r(t);
    auto base = r.walker.walk_components(r.info, h.data(), 12, FSM, false);
    assert(base.tree.common_hit == 12);

    std::vector<RemoteHit> hints;
    for (const auto& b : t.get_node_boundaries(h.data(), 12))
        hints.push_back(RemoteHit{(int)b.offset, b.tree_node_id});
    assert(hints.size() == 12);
    auto fast = r.walker.walk_components_batched(r.info, h.data(), 12, FSM,
                                                 false, 0, 0, hints.data(),
                                                 (int)hints.size());
    assert_same(base.tree, fast.tree);
    assert(fast.rdma_read_count < base.rdma_read_count);

    auto stale = hints;
    stale.back().tree_node_id = 199;
    auto sres = r.walker.walk_components_batched(r.info, h.data(), 12, FSM,
                                                 false, 0, 0, stale.data(),
                                                 (int)stale.size());
    assert(sres.tree.status == RemoteStatus::OK && sres.tree.common_hit == 0);
    stale.back().tree_node_id = 0x0FFFFFFF;
    auto ires = r.walker.walk_components_batched(r.info, h.data(), 12, FSM,
                                                 false, 0, 0, stale.data(),
                                                 (int)stale.size());
    assert(ires.tree.status == RemoteStatus::OK && ires.tree.common_hit == 0);

    t.write_lock();
    auto to = r.walker.walk_components(r.info, h.data(), 12, FS, false);
    assert(to.tree.status == RemoteStatus::TIMEOUT);
    t.write_unlock();

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_remote_hint_chain_is_the_walk() {
    TEST_BEGIN("remote: the hinted chain bounds the walk; no descent past its end");

    const char* name = "/test_cq_remote_hint_chain";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 4, 64));   // cs=4: nodes [0,4) [4,8) [8,12)
    auto h = seq(7000, 16);
    insert_fresh(t, h.data(), 12);
    auto bounds = t.get_node_boundaries(h.data(), 12);
    assert(bounds.size() == 3);

    Rig r(t);
    const uint32_t F = COMPONENT_MASK_FULL;
    std::vector<RemoteHit> hints;
    for (const auto& b : bounds) hints.push_back(RemoteHit{(int)b.offset, b.tree_node_id});
    auto res = r.walker.walk_components_batched(r.info, h.data(), 16, F, false,
                                                /*incumbent=*/8, /*local_full_hit=*/8,
                                                hints.data(), 3);
    assert(res.tree.status == RemoteStatus::OK && res.tree.common_hit == 12);
    assert(res.tree.full_slots.size() == 12);
    const int full_reads = res.rdma_read_count;

    // Hints stop at [4,8): the walk stops there too, whatever the peer holds.
    res = r.walker.walk_components_batched(r.info, h.data(), 16, F, false, 4, 4,
                                           hints.data(), 2);
    assert(res.tree.common_hit == 8 && res.rdma_read_count <= full_reads);

    // The last hint ends at the incumbent: empty miss after the metadata read.
    res = r.walker.walk_components_batched(r.info, h.data(), 16, F, false, 8, 8,
                                           hints.data(), 2);
    assert(res.tree.status == RemoteStatus::OK && res.tree.common_hit == 0);
    assert(res.rdma_read_count == 3);   // lock flag + 2 hint metas

    hints[2].tree_node_id = 199;
    res = r.walker.walk_components_batched(r.info, h.data(), 16, F, false, 8, 8,
                                           hints.data(), 3);
    assert(res.tree.status == RemoteStatus::OK && res.tree.common_hit == 0);

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_remote_pin_swa_terminal_past_query() {
    TEST_BEGIN("remote: SWA pin on a terminal longer than the query releases");

    // Same shape as the local case over the RDMA walker: the peer holds
    // [0,4)[4,8)[8,12)[12,16), the caller queries ten blocks with SWA live
    // on [6,10). Pins: Full on three nodes, SWA on two -> 5 FAAs + 2 lock.
    const char* name = "/test_cq_remote_swa_tail";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 4, 64));
    auto h = seq(17000, 16);
    insert_fresh(t, h.data(), 16);
    publish_at(t, h.data(), 10, 4);
    Rig r(t);

    r.ep.reset_counters();
    auto res = r.walker.walk_components(r.info, h.data(), 10, FS, /*lock=*/true);
    assert(res.tree.status == RemoteStatus::OK && res.tree.common_hit == 10);
    assert(res.tree.swa_start == 6 && res.tree.swa_slots.size() == 4);
    assert(res.remote_pin.full.start == 0 && res.remote_pin.full.end == 12 &&
           res.remote_pin.tail.size() == 2);
    assert(res.remote_pin.swa.start == 4 && res.remote_pin.swa.end == 12);
    assert(r.ep.faas == 7);

    // Release with the caller's ten keys must run and balance every FAA.
    r.ep.reset_counters();
    assert(r.walker.dec_ref_components_remote(r.info, h.data(), 10,
                                              std::move(res.remote_pin)));
    assert(r.ep.faas == 7);
    pressure(t, 64);
    assert(hit(t, h.data(), 16) == 0);

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_remote_pin_swa_only_terminal_past_query() {
    TEST_BEGIN("remote: SWA-only pin on a terminal longer than the query releases");

    // SWA alone over the walker: 2 lock FAAs + swa_ref_count on N1 and N2.
    const char* name = "/test_cq_remote_swa_only_tail";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 4, 64));
    auto h = seq(17100, 16);
    insert_fresh(t, h.data(), 16);
    publish_at(t, h.data(), 10, 4);
    Rig r(t);

    r.ep.reset_counters();
    auto res = r.walker.walk_components(r.info, h.data(), 10,
                                        COMPONENT_MASK_SWA, /*lock=*/true);
    assert(res.tree.status == RemoteStatus::OK && res.tree.common_hit == 10);
    assert(res.tree.swa_start == 6 && res.tree.swa_slots.size() == 4);
    assert(res.tree.full_slots.empty() && res.remote_pin.full.empty());
    assert(res.remote_pin.swa.start == 4 && res.remote_pin.swa.end == 12);
    assert(r.ep.faas == 4);

    r.ep.reset_counters();
    assert(r.walker.dec_ref_components_remote(r.info, h.data(), 10,
                                              std::move(res.remote_pin)));
    assert(r.ep.faas == 4);
    pressure(t, 64);
    assert(hit(t, h.data(), 16) == 0);

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

static void test_remote_multinode_cut_and_release_guards() {
    TEST_BEGIN("remote: fold pins only past the cut; release refuses bad records");

    const char* name = "/test_cq_remote_cut2";
    shm_unlink(name);
    auto t = RadixShmemTree::create(name, cfg(4, 4, 64));
    auto h = seq(14000, 8);
    insert_fresh(t, h.data(), 4);
    std::vector<uint64_t> fork(h.begin(), h.begin() + 4);
    auto d = seq(14900, 4);
    fork.insert(fork.end(), d.begin(), d.end());
    insert_fresh(t, fork.data(), 8, /*start=*/4);   // A[0,4) keeps two children
    insert_fresh(t, h.data(), 8, /*start=*/4);      // B[4,8)
    assert(t.get_node_boundaries(h.data(), 8).size() == 2);
    Rig r(t);

    auto zero = r.walker.walk_components(r.info, nullptr, 0, FSM, false);
    assert(zero.tree.status == RemoteStatus::OK && zero.tree.common_hit == 0);

    // SWA live on A only: the joint hit stops at 4 while Full reaches 8, so
    // the Full pin over [c,L) = [2,4) stops before B.
    publish_at(t, h.data(), 4, 4);
    r.ep.reset_counters();
    auto brk = r.walker.walk_components(r.info, h.data(), 8, FS, true,
                                        /*incumbent=*/2, /*local_full_hit=*/2);
    assert(brk.tree.common_hit == 4 && brk.tree.topology_match_length == 8);
    assert(brk.remote_pin.full.start == 0 && brk.remote_pin.full.end == 4 &&
           brk.remote_pin.tail.empty());
    assert(brk.remote_pin.swa.start == 0 && brk.remote_pin.swa.end == 4);
    assert(r.ep.faas == 4);
    r.ep.reset_counters();
    assert(r.walker.dec_ref_components_remote(r.info, h.data(), 8,
                                              std::move(brk.remote_pin)));
    assert(r.ep.faas == 4);

    // Local incumbent covers A whole: the remote pin skips A on both lanes.
    publish_at(t, h.data(), 8, 4);
    r.ep.reset_counters();
    auto cut = r.walker.walk_components(r.info, h.data(), 8, FS, true,
                                        /*incumbent=*/4, /*local_full_hit=*/4);
    assert(cut.tree.common_hit == 8);
    assert(cut.remote_pin.full.start == 4 && cut.remote_pin.full.end == 8);
    assert(cut.remote_pin.swa.start == 4 && cut.remote_pin.swa.end == 8);
    assert(r.ep.faas == 4);
    pressure(t, 56);   // only the fork leaf can go
    assert(hit(t, h.data(), 8) == 8 && hit(t, fork.data(), 8) == 4);
    r.ep.reset_counters();
    assert(r.walker.dec_ref_components_remote(r.info, h.data(), 8,
                                              std::move(cut.remote_pin)));
    assert(r.ep.faas == 4);

    // A CST target outside the peer MR refuses the pin before any atomic.
    RemoteTreeInfo oob = r.info;
    oob.component_state_offset = oob.mr.length;
    r.ep.reset_counters();
    auto bad_mr = r.walker.walk_components(oob, h.data(), 8, FS, true);
    assert(bad_mr.tree.status == RemoteStatus::INTERNAL && r.ep.faas == 2);

    // Malformed release records are refused before the lock.
    r.ep.reset_counters();
    auto refused = [&](PinRecord p, const uint64_t* hh, int n) {
        return !r.walker.dec_ref_components_remote(r.info, hh, n, std::move(p));
    };
    PinRecord ok;
    ok.full = {0, 4};
    assert(refused(ok, h.data(), 0));
    assert(refused(ok, nullptr, 4));
    assert(refused(PinRecord{}, h.data(), 4));
    RemoteTreeInfo broken = r.info;
    broken.component_info_valid = false;
    assert(!r.walker.dec_ref_components_remote(broken, h.data(), 4,
                                               PinRecord{ok}));
    PinRecord inv;
    inv.full = {5, 2};
    assert(refused(inv, h.data(), 8));
    PinRecord sinv;
    sinv.swa = {5, 2};
    assert(refused(sinv, h.data(), 8));
    PinRecord long_tail;
    long_tail.full = {0, 2};
    long_tail.tail = {1, 2, 3};
    assert(refused(long_tail, h.data(), 8));
    PinRecord shrt;
    shrt.full = {0, 8};
    assert(refused(shrt, h.data(), 4));
    assert(r.ep.faas == 0);

    // An unknown prefix stops the descent: lock round-trip only, no decrement.
    auto hx = seq(15000, 4);
    PinRecord miss;
    miss.full = {0, 4};
    assert(r.walker.dec_ref_components_remote(r.info, hx.data(), 4,
                                              std::move(miss)));
    assert(r.ep.faas == 2);
    pressure(t, 64);
    assert(hit(t, h.data(), 8) == 0);

    expect_healthy(t);
    t.destroy();
    TEST_PASS();
}

#endif

int main() {
    printf("test_component_query\n");
    test_swa_any_boundary_acceptance();
    test_swa_gap_fallback_and_reaccept();
    test_mamba_node_end();
    test_joint_common_hit();
    test_joint_alternating_descent();
    test_mask_validation_and_node_fww();
    test_swa_publish_window_refresh();
    test_swa_positional_publish();
    test_update_meta_component_gating();
    test_update_meta_touches_full_extent();
    test_component_slot_allocation();
    test_mamba_merge_keeps_checkpoint();
    test_pin_release_lifecycle();
    test_pin_per_lane_ends_and_overpin();
    test_pin_partial_terminal_then_split();
    test_pin_swa_terminal_past_query();
    test_pin_swa_only_terminal_past_query();
    test_pinned_split_then_release();
    test_release_trust_model();
    test_insert_arg_validation();
    test_mamba_only_pin_release();
    test_facade_shape_and_finalize();
    test_facade_bad_mask_and_crop_to_zero();
    test_client_component_insert();
#ifdef RADIXSHMEM_NO_RDMA
    test_nordma_client_boundaries();
#endif
#ifndef RADIXSHMEM_NO_RDMA
    test_remote_parity_and_abi_gate();
    test_remote_pin_incumbent_release();
    test_remote_throw_releases_reader_count();
    test_remote_pin_swa_terminal_past_query();
    test_remote_pin_swa_only_terminal_past_query();
    test_remote_hints_and_failstop();
    test_remote_hint_chain_is_the_walk();
    test_remote_multinode_cut_and_release_guards();
#endif
    return tests_report();
}
