// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// test_data_registry — the data plane's slot geometry in the shared etcd
// registry: publish, look up, and the bootstrap gate that refuses a cluster
// whose nodes disagree on the slot stride.
//
// Links shmradix_registry alone, and includes no indexer header (hence the data
// plane's runner) — which is what proves src/registry is plane-neutral.
//
// Single process throughout. Several logical nodes are just several node names,
// which is enough for everything here — and the Go etcd wrapper deadlocks on
// its first cgo call in a bare-forked child, so a fork-per-node shape would not
// work anyway.
//
// SKIPs (exit 77) when no etcd binary is on the box.

#include "data_test_utils.h"
#include "etcd_test_harness.hpp"

#include "shmradix/config.h"

#ifndef SHMRADIX_HAS_ETCD
int main() { printf("built without etcd — SKIP\n"); return 77; }
#else

#include "shmradix/data_registry.h"
#include "etcd_client.hpp"

#include <string>

using namespace shmradix;

static constexpr uint64_t kSlotBytes = 65536;
static constexpr uint64_t kSlots     = 4096;

static DataPlaneEntry entry(uint64_t slot_bytes, uint64_t num_slots,
                            const std::string& seg) {
    DataPlaneEntry e{};
    assert(make_data_plane_entry(slot_bytes, num_slots, seg, e));
    return e;
}

static void test_entry_builder() {
    TEST_BEGIN("make_data_plane_entry: stamps, copies, bounds-checks");

    DataPlaneEntry e{};
    assert(make_data_plane_entry(kSlotBytes, kSlots, "10.0.0.1:12345", e));
    assert(e.magic == DATA_PLANE_ENTRY_MAGIC);
    assert(e.version == DATA_PLANE_ENTRY_VERSION);
    assert(e.pool_mask == COMPONENT_MASK_FULL);
    assert(e.pools[0].slot_bytes == kSlotBytes);
    assert(e.pools[0].num_slots == kSlots);
    assert(e.pools[0].data_offset == 0);
    assert(e.pools[1].num_slots == 0 && e.pools[2].slot_bytes == 0);
    assert(std::string(e.segment) == "10.0.0.1:12345");

    // The longest name that still leaves room for the terminator fits; one more
    // is refused rather than silently truncated.
    const std::string fits(DATA_PLANE_SEGMENT_LEN - 1, 'x');
    assert(make_data_plane_entry(kSlotBytes, kSlots, fits, e));
    assert(std::string(e.segment) == fits);
    assert(!make_data_plane_entry(kSlotBytes, kSlots, fits + "x", e));

    TEST_PASS();
}

static void test_publish_lookup(DataPlaneRegistry& reg) {
    TEST_BEGIN("publish/lookup: round-trips every field");

    reg.publish("nodeA", entry(kSlotBytes, kSlots, "10.0.0.1:12345"));

    DataPlaneEntry got{};
    assert(reg.lookup("nodeA", got, 5));
    assert(got.pools[0].slot_bytes == kSlotBytes);
    assert(got.pools[0].num_slots == kSlots);
    assert(std::string(got.segment) == "10.0.0.1:12345");

    // An unpublished node is absent, not a decode failure or a hang.
    DataPlaneEntry missing{};
    assert(!reg.lookup("nodeZ", missing, 0));

    TEST_PASS();
}

static void test_list(DataPlaneRegistry& reg) {
    TEST_BEGIN("list: every entry, sorted, node names recovered");

    reg.publish("nodeB", entry(kSlotBytes, kSlots, "10.0.0.2:12345"));
    reg.publish("nodeC", entry(kSlotBytes, kSlots, "10.0.0.3:12345"));

    int skipped = -1;
    auto all = reg.list(&skipped);
    assert(skipped == 0);
    assert(all.size() == 3);
    assert(all[0].first == "nodeA");
    assert(all[1].first == "nodeB");
    assert(all[2].first == "nodeC");
    assert(std::string(all[2].second.segment) == "10.0.0.3:12345");

    TEST_PASS();
}

// The gate. Everyone agrees on the stride, so bootstrap proceeds.
static void test_validate_agreeing(DataPlaneRegistry& reg) {
    TEST_BEGIN("validate_uniform: an agreeing cluster passes");

    std::string who, why;
    assert(reg.validate_uniform(entry(kSlotBytes, kSlots, "local"), who, why));
    assert(who.empty() && why.empty());

    TEST_PASS();
}

// A node with a different slot count is fine: ids still mean the same bytes,
// and attach_peer range-checks against the published length.
static void test_validate_count_differs(DataPlaneRegistry& reg) {
    TEST_BEGIN("validate_uniform: a differing slot count is not a failure");

    reg.publish("nodeD", entry(kSlotBytes, kSlots / 2, "10.0.0.4:12345"));

    std::string who, why;
    assert(reg.validate_uniform(entry(kSlotBytes, kSlots, "local"), who, why));

    reg.unpublish("nodeD");
    TEST_PASS();
}

// The failure this whole path exists for: a stride that divides evenly into
// ours, so every length-based check downstream would have waved it through.
static void test_validate_stride_differs(DataPlaneRegistry& reg) {
    TEST_BEGIN("validate_uniform: a divisible-but-wrong stride is caught");

    const uint64_t bad_stride = 4096;
    assert(kSlotBytes % bad_stride == 0);   // the premise
    reg.publish("nodeE", entry(bad_stride, kSlots, "10.0.0.5:12345"));

    std::string who, why;
    assert(!reg.validate_uniform(entry(kSlotBytes, kSlots, "local"), who, why));
    assert(who == "nodeE");
    assert(why.find("4096") != std::string::npos);
    assert(why.find("65536") != std::string::npos);

    // And the offender is named precisely enough to act on: looking it up gives
    // the stride an orchestrator would have fed to attach_peer.
    DataPlaneEntry e{};
    assert(reg.lookup("nodeE", e, 0));
    assert(e.pools[0].slot_bytes == bad_stride);

    reg.unpublish("nodeE");
    std::string who2, why2;
    assert(reg.validate_uniform(entry(kSlotBytes, kSlots, "local"), who2, why2));

    TEST_PASS();
}

static void test_unpublish(DataPlaneRegistry& reg) {
    TEST_BEGIN("unpublish: removes one node, leaves the rest");

    reg.unpublish("nodeC");
    DataPlaneEntry e{};
    assert(!reg.lookup("nodeC", e, 0));
    assert(reg.lookup("nodeB", e, 0));
    assert(reg.list().size() == 2);

    // Unpublishing what was never there is not an error.
    reg.unpublish("nodeC");
    reg.unpublish("nodeNeverExisted");

    TEST_PASS();
}

// A blob from an incompatible build must be refused everywhere, not decoded
// into plausible-looking geometry.
static void test_undecodable_blob(const std::string& registry,
                                  const std::string& cluster_id,
                                  DataPlaneRegistry& reg) {
    TEST_BEGIN("undecodable blob: refused by lookup, list and the gate");

    etcd::Client raw;
    raw.init(etcd::strip_scheme(registry));
    etcd::Keys keys;
    keys.cluster_id = cluster_id;

    // Right size, wrong magic — the shape a future/foreign encoding would have.
    DataPlaneEntry bogus{};
    bogus.magic      = 0xDEADBEEF;
    bogus.version    = DATA_PLANE_ENTRY_VERSION;
    bogus.pool_mask  = COMPONENT_MASK_FULL;
    bogus.pools[0]   = {kSlots, kSlotBytes, 0};
    raw.put(keys.data("nodeX"), etcd::b64_encode(&bogus, sizeof(bogus)), 0);

    DataPlaneEntry got{};
    assert(!reg.lookup("nodeX", got, 0));

    int skipped = 0;
    auto all = reg.list(&skipped);
    assert(skipped == 1);
    for (auto& [node, e] : all) { (void)e; assert(node != "nodeX"); }

    // The gate must fail rather than validate the entries it *could* read: an
    // unreadable peer is an unknown stride, not an absent one.
    std::string who, why;
    assert(!reg.validate_uniform(entry(kSlotBytes, kSlots, "local"), who, why));
    assert(!why.empty());

    raw.del(keys.data("nodeX"));
    TEST_PASS();
}

// magic and version say "readable", but the segment fills all 128 bytes with no
// terminator — reading it as a C string would run off the end of the struct.
static void test_unterminated_segment(const std::string& registry,
                                      const std::string& cluster_id,
                                      DataPlaneRegistry& reg) {
    TEST_BEGIN("unterminated segment: refused, not read past the end");

    etcd::Client raw;
    raw.init(etcd::strip_scheme(registry));
    etcd::Keys keys;
    keys.cluster_id = cluster_id;

    DataPlaneEntry bogus{};
    bogus.magic      = DATA_PLANE_ENTRY_MAGIC;
    bogus.version    = DATA_PLANE_ENTRY_VERSION;
    bogus.pool_mask  = COMPONENT_MASK_FULL;
    bogus.pools[0]   = {kSlots, kSlotBytes, 0};
    memset(bogus.segment, 'x', DATA_PLANE_SEGMENT_LEN);   // no '\0' anywhere
    raw.put(keys.data("nodeY"), etcd::b64_encode(&bogus, sizeof(bogus)), 0);

    DataPlaneEntry got{};
    assert(!reg.lookup("nodeY", got, 0));

    int skipped = 0;
    auto all = reg.list(&skipped);
    assert(skipped == 1);
    for (auto& [node, e] : all) { (void)e; assert(node != "nodeY"); }

    // An unreadable segment is an unusable peer, so the gate fails too — even
    // though this entry's stride agrees with ours.
    std::string who, why;
    assert(!reg.validate_uniform(entry(kSlotBytes, kSlots, "local"), who, why));

    raw.del(keys.data("nodeY"));
    TEST_PASS();
}

static void test_empty_registry(const std::string& registry) {
    TEST_BEGIN("validate_uniform: an empty namespace is agreement");

    // A fresh cluster_id nobody has published under.
    DataPlaneRegistry fresh(registry, "radix-test-empty-" +
                                      std::to_string((int)::getpid()));
    assert(fresh.list().empty());

    std::string who, why;
    assert(fresh.validate_uniform(entry(kSlotBytes, kSlots, "local"), who, why));

    TEST_PASS();
}


// Three-pool layouts: the whole table round-trips, and the gate compares per pool.
static StoreGeometry layout3(uint64_t swa_stride = 4096, uint64_t full_slots = kSlots) {
    StoreGeometry g;
    g.pool_mask = COMPONENT_MASK_ALL;
    g[ComponentType::FULL]  = {0, full_slots, kSlotBytes};
    g[ComponentType::SWA]   = {full_slots * kSlotBytes, 64, swa_stride};
    g[ComponentType::MAMBA] = {full_slots * kSlotBytes + 64 * swa_stride, 8, 8192};
    return g;
}

static void test_multi_pool_entry() {
    TEST_BEGIN("make_data_plane_entry(layout): three pools in, same three out");

    DataPlaneEntry e{};
    assert(make_data_plane_entry(layout3(), "10.0.0.9:1", e));
    assert(e.pool_mask == COMPONENT_MASK_ALL);
    assert(e.pools[1].num_slots == 64 && e.pools[1].slot_bytes == 4096);
    assert(e.pools[1].data_offset == kSlots * kSlotBytes);
    assert(e.pools[2].slot_bytes == 8192);

    StoreGeometry back = data_plane_layout(e);
    assert(back.pool_mask == COMPONENT_MASK_ALL);
    assert(back[ComponentType::MAMBA].base == e.pools[2].data_offset);
    assert(back.same_strides(layout3()));

    // Malformed layouts are refused.
    StoreGeometry bad = layout3();
    bad.pool_mask = COMPONENT_MASK_SWA;                 // no FULL
    assert(!make_data_plane_entry(bad, "x", e));
    bad = layout3();
    bad[ComponentType::SWA].num_slots = 0;              // present pool, zero count
    assert(!make_data_plane_entry(bad, "x", e));

    TEST_PASS();
}

static void test_validate_pools(const std::string& registry) {
    TEST_BEGIN("validate_uniform: pool mask and per-pool stride, counts free");

    // Own namespace: the shared one holds FULL-only nodes.
    DataPlaneRegistry reg(registry, "radix-test-pools-" + std::to_string((int)::getpid()));
    DataPlaneEntry local{};
    assert(make_data_plane_entry(layout3(), "local", local));

    DataPlaneEntry peer{};
    assert(make_data_plane_entry(layout3(4096, kSlots / 2), "10.0.0.6:1", peer));
    reg.publish("nodeF", peer);
    std::string who, why;
    assert(reg.validate_uniform(local, who, why));     // fewer FULL slots: fine

    assert(make_data_plane_entry(layout3(8192), "10.0.0.7:1", peer));
    reg.publish("nodeG", peer);                        // SWA stride differs
    assert(!reg.validate_uniform(local, who, why));
    assert(who == "nodeG" && why.find("SWA") != std::string::npos);
    reg.unpublish("nodeG");

    // A FULL-only peer in a three-pool cluster.
    reg.publish("nodeH", entry(kSlotBytes, kSlots, "10.0.0.8:1"));
    assert(!reg.validate_uniform(local, who, why));
    assert(who == "nodeH" && why.find("pool mask") != std::string::npos);
    reg.unpublish("nodeH");
    reg.unpublish("nodeF");

    TEST_PASS();
}

int main() {
    printf("=== data-plane geometry registry ===\n");

    test_entry_builder();
    test_multi_pool_entry();

    shmradix_test::EtcdHarness etcd_srv;
    if (!etcd_srv.usable()) {
        printf("  no etcd binary — SKIP\n");
        return shmradix_test::kSkipExitCode;
    }

    const std::string registry   = etcd_srv.registry();
    const std::string cluster_id = etcd_srv.cluster_id();

    DataPlaneRegistry reg(registry, cluster_id);

    test_publish_lookup(reg);
    test_list(reg);
    test_validate_agreeing(reg);
    test_validate_count_differs(reg);
    test_validate_stride_differs(reg);
    test_validate_pools(registry);
    test_unpublish(reg);
    test_undecodable_blob(registry, cluster_id, reg);
    test_unterminated_segment(registry, cluster_id, reg);
    test_empty_registry(registry);

    return tests_report();
}

#endif // SHMRADIX_HAS_ETCD
