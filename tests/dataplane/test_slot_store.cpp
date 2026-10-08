// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// test_slot_store — SlotStore contract: geometry, addressing, cross-process
// attach, lifecycle. Links radixshmem_data only (no indexer symbols available).
//
//   - geometry: slot_bytes rounded up to slot_align, data_bytes/num_slots exact
//   - addressing: slot i at data_base + i*stride, out-of-range rejected
//   - header excluded from [data_base, data_bytes) so a remote write can't reach it
//   - fork: child attaches by name, sees the parent's bytes, writes back
//   - attach waits/fails: absent name, bad magic, wrong version, corrupt geometry
//   - lifecycle: owner destructor unlinks; destroy() clears a leftover
//   - hugepage path falls back to shm_open when unavailable

#include "data_test_utils.h"
#include "slot_store.h"

#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

using namespace radixshmem;

// Plain shm keeps every test deterministic on hosts without reserved hugepages.
static SlotStoreConfig cfg(const std::string& name, uint64_t slots,
                           uint64_t slot_bytes, uint64_t align = 4096) {
    SlotStoreConfig c;
    c.name = name;
    c.full.num_slots = slots;
    c.full.slot_bytes = slot_bytes;
    c.slot_align = align;
    c.hugepage_path = "";
    c.prefault = false;
    return c;
}

// ==================================================================
static void test_geometry() {
    TEST_BEGIN("create: slot_bytes rounded up to slot_align, sizes exact");

    SlotStore::destroy("/test_ss_01", "");
    SlotStore s = SlotStore::create(cfg("/test_ss_01", 64, 100));

    // 100 B payload, 4 KiB alignment -> 4 KiB stride.
    assert(s.slot_bytes() == 4096);
    assert(s.num_slots() == 64);
    assert(s.data_bytes() == 64ull * 4096);
    assert(s.is_owner());
    assert(!s.is_hugepage());
    assert(s.valid());
    assert(s.name() == "/test_ss_01");

    // An already-aligned request is left alone.
    SlotStore::destroy("/test_ss_01b", "");
    SlotStore t = SlotStore::create(cfg("/test_ss_01b", 8, 8192));
    assert(t.slot_bytes() == 8192);
    assert(t.data_bytes() == 8ull * 8192);

    // slot_align=1 means "no padding": stride is the raw payload size.
    SlotStore::destroy("/test_ss_01c", "");
    SlotStore u = SlotStore::create(cfg("/test_ss_01c", 16, 100, 1));
    assert(u.slot_bytes() == 100);
    assert(u.data_bytes() == 1600);

    TEST_PASS();
}

// ==================================================================
static void test_addressing() {
    TEST_BEGIN("slot_addr/slot_offset: stride exact, out-of-range rejected");

    SlotStore::destroy("/test_ss_02", "");
    SlotStore s = SlotStore::create(cfg("/test_ss_02", 32, 4096));

    char* base = (char*)s.data_base();
    assert(base != nullptr);
    assert(s.slot_addr(0) == base);
    for (uint64_t i = 0; i < s.num_slots(); ++i) {
        assert((char*)s.slot_addr(i) == base + i * s.slot_bytes());
        assert(s.slot_offset(i) == i * s.slot_bytes());
    }

    assert(s.slot_addr(s.num_slots()) == nullptr);
    assert(s.slot_addr(~0ull) == nullptr);
    assert(s.slot_offset(s.num_slots()) == UINT64_MAX);

    // Every slot is inside the registerable range, and the range is exact.
    assert((char*)s.slot_addr(s.num_slots() - 1) + s.slot_bytes()
           == base + s.data_bytes());

    TEST_PASS();
}

// ==================================================================
static void test_header_outside_data() {
    TEST_BEGIN("header sits below data_base (never inside the registered range)");

    SlotStore::destroy("/test_ss_03", "");
    SlotStore s = SlotStore::create(cfg("/test_ss_03", 8, 4096));

    // Map the raw object independently and locate the header at offset 0. This is
    // a distinct mapping, so only file offsets — never addresses — are comparable
    // between it and `s`.
    int fd = ::shm_open("/test_ss_03", O_RDWR, 0666);
    assert(fd >= 0);
    const uint64_t map_bytes = 8ull * 4096 + 4096;
    void* raw = ::mmap(nullptr, map_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    assert(raw != MAP_FAILED);
    auto* hdr = (SlotStoreHeader*)raw;
    assert(hdr->magic == SLOT_STORE_MAGIC);
    assert(hdr->version == SLOT_STORE_VERSION);
    assert(hdr->pool_mask == COMPONENT_MASK_FULL);
    assert(hdr->pools[0].num_slots == 8);
    assert(hdr->owner_pid == (int32_t)::getpid());

    // Slot 0 starts strictly above the header, on a page boundary.
    assert(hdr->data_offset >= sizeof(SlotStoreHeader));
    assert(hdr->data_offset % 4096 == 0);
    assert(((uintptr_t)s.data_base() % 4096) == 0);
    assert(hdr->total_bytes == hdr->data_offset + s.data_bytes());

    // A write anywhere in the registered range lands at data_offset or beyond —
    // it can never reach the header.
    std::memset(s.slot_addr(0), 0x77, s.slot_bytes());
    assert(hdr->magic == SLOT_STORE_MAGIC);
    assert(((unsigned char*)raw)[hdr->data_offset] == 0x77);

    ::munmap(raw, map_bytes);
    ::close(fd);
    TEST_PASS();
}

// ==================================================================
static void test_read_write_roundtrip() {
    TEST_BEGIN("payload roundtrip through slot_addr, neighbours untouched");

    SlotStore::destroy("/test_ss_04", "");
    SlotStore s = SlotStore::create(cfg("/test_ss_04", 16, 4096));

    for (uint64_t i = 0; i < s.num_slots(); ++i)
        std::memset(s.slot_addr(i), (int)(i & 0xff), s.slot_bytes());

    for (uint64_t i = 0; i < s.num_slots(); ++i) {
        auto* p = (unsigned char*)s.slot_addr(i);
        for (uint64_t b = 0; b < s.slot_bytes(); ++b)
            assert(p[b] == (unsigned char)(i & 0xff));
    }

    TEST_PASS();
}

// ==================================================================
static void test_fork_attach() {
    TEST_BEGIN("fork: child attaches by name, shares geometry and bytes");

    SlotStore::destroy("/test_ss_05", "");
    SlotStore s = SlotStore::create(cfg("/test_ss_05", 32, 4096));
    std::memset(s.slot_addr(3), 0xAB, s.slot_bytes());

    run_child([] {
        SlotStore c = SlotStore::attach("/test_ss_05", "");
        assert(!c.is_owner());
        assert(c.num_slots() == 32);
        assert(c.slot_bytes() == 4096);
        assert(c.data_bytes() == 32ull * 4096);

        auto* p = (unsigned char*)c.slot_addr(3);
        for (uint64_t b = 0; b < c.slot_bytes(); ++b) assert(p[b] == 0xAB);

        std::memset(c.slot_addr(7), 0xCD, c.slot_bytes());
        // A non-owner must not unlink on destruction.
    });

    auto* p = (unsigned char*)s.slot_addr(7);
    for (uint64_t b = 0; b < s.slot_bytes(); ++b) assert(p[b] == 0xCD);

    // The child's destructor left the region alive for the owner.
    SlotStore again = SlotStore::attach("/test_ss_05", "");
    assert(again.num_slots() == 32);

    TEST_PASS();
}

// ==================================================================
static void test_attach_failures() {
    TEST_BEGIN("attach: absent name, bad magic, wrong version all throw");

    SlotStore::destroy("/test_ss_06_absent", "");
    assert(throws([] { SlotStore::attach("/test_ss_06_absent", ""); }));

    SlotStore::destroy("/test_ss_06", "");
    SlotStore s = SlotStore::create(cfg("/test_ss_06", 8, 4096));

    int fd = ::shm_open("/test_ss_06", O_RDWR, 0666);
    assert(fd >= 0);
    void* raw = ::mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    assert(raw != MAP_FAILED);
    auto* hdr = (SlotStoreHeader*)raw;

    const uint64_t good_magic = hdr->magic;
    hdr->magic = 0xdeadbeef;
    assert(throws([] { SlotStore::attach("/test_ss_06", "", 200); }));
    hdr->magic = good_magic;

    const uint32_t good_ver = hdr->version;
    hdr->version = good_ver + 1;
    assert(throws([] { SlotStore::attach("/test_ss_06", "", 200); }));
    hdr->version = good_ver;

    // Restored header attaches again.
    SlotStore ok = SlotStore::attach("/test_ss_06", "");
    assert(ok.num_slots() == 8);

    // Geometry the hot path would use must be bounded, not just total_bytes:
    // data_offset feeds data_base_, num_slots*slot_bytes feeds the registered MR.
    const uint64_t good_off = hdr->data_offset;
    hdr->data_offset = 0;                       // would put slot 0 over the header
    assert(throws([] { SlotStore::attach("/test_ss_06", "", 200); }));
    hdr->data_offset = good_off;

    SlotPoolDesc& full = hdr->pools[0];
    const uint64_t good_slots = full.num_slots;
    full.num_slots = good_slots * 1000;         // total_bytes still passes its own check
    assert(throws([] { SlotStore::attach("/test_ss_06", "", 200); }));
    full.num_slots = good_slots;

    const uint64_t good_stride = full.slot_bytes;
    full.slot_bytes = ~0ull;                    // num_slots * slot_bytes wraps
    assert(throws([] { SlotStore::attach("/test_ss_06", "", 200); }));
    full.slot_bytes = good_stride;

    // Pool table corruption: no FULL, unknown bit, absent pool not zeroed.
    const uint32_t good_mask = hdr->pool_mask;
    hdr->pool_mask = COMPONENT_MASK_SWA;
    assert(throws([] { SlotStore::attach("/test_ss_06", "", 200); }));
    hdr->pool_mask = good_mask | (1u << 5);
    assert(throws([] { SlotStore::attach("/test_ss_06", "", 200); }));
    hdr->pool_mask = good_mask;
    hdr->pools[1].num_slots = 4;
    assert(throws([] { SlotStore::attach("/test_ss_06", "", 200); }));
    hdr->pools[1].num_slots = 0;

    assert(SlotStore::attach("/test_ss_06", "").num_slots() == 8);

    // ready==0 (owner still initialising) makes attach time out rather than
    // hand out a half-built region.
    hdr->ready.store(0, std::memory_order_release);
    assert(throws([] { SlotStore::attach("/test_ss_06", "", 50); }));
    hdr->ready.store(1, std::memory_order_release);

    ::munmap(raw, 4096);
    ::close(fd);
    TEST_PASS();
}

// ==================================================================
static void test_bad_config() {
    TEST_BEGIN("create: empty name / zero size / non-power-of-2 align throw");

    assert(throws([] { SlotStore::create(cfg("", 8, 4096)); }));
    assert(throws([] { SlotStore::create(cfg("/test_ss_07", 0, 4096)); }));
    assert(throws([] { SlotStore::create(cfg("/test_ss_07", 8, 0)); }));
    assert(throws([] { SlotStore::create(cfg("/test_ss_07", 8, 4096, 3000)); }));
    assert(throws([] { SlotStore::create(cfg("/test_ss_07", 8, 4096, 0)); }));
    // num_slots * stride must not wrap.
    assert(throws([] { SlotStore::create(cfg("/test_ss_07", ~0ull, 4096)); }));
    // Nor may the slot_bytes -> stride round-up: a wrap gives stride 0, which
    // would pass every size check and alias all slots onto one address.
    assert(throws([] { SlotStore::create(cfg("/test_ss_07", 8, ~0ull)); }));

    TEST_PASS();
}

// ==================================================================
static void test_lifecycle() {
    TEST_BEGIN("lifecycle: owner unlinks on destruction, destroy() clears leftovers");

    SlotStore::destroy("/test_ss_08", "");
    {
        SlotStore s = SlotStore::create(cfg("/test_ss_08", 8, 4096));
        assert(s.valid());
    }
    // Owner gone -> region gone.
    assert(throws([] { SlotStore::attach("/test_ss_08", "", 50); }));

    // A moved-from owner must not double-unlink, and the moved-to one still owns.
    SlotStore a = SlotStore::create(cfg("/test_ss_08", 8, 4096));
    char* base = (char*)a.data_base();
    SlotStore b = std::move(a);
    assert(!a.valid());
    assert(b.valid() && b.is_owner() && (char*)b.data_base() == base);
    assert(b.num_slots() == 8);

    SlotStore::destroy("/test_ss_08", "");
    // destroy() on an absent name is a silent no-op.
    SlotStore::destroy("/test_ss_08", "");

    TEST_PASS();
}

// ==================================================================
static void test_hugepage_fallback() {
    TEST_BEGIN("hugepage path unavailable -> falls back to shm_open");

    SlotStore::destroy("/test_ss_09", "/nonexistent_hugepages");
    SlotStoreConfig c = cfg("/test_ss_09", 8, 4096);
    c.hugepage_path = "/nonexistent_hugepages";   // open() fails -> shm fallback
    SlotStore s = SlotStore::create(c);
    assert(!s.is_hugepage());
    assert(s.num_slots() == 8);
    std::memset(s.slot_addr(0), 0x5A, s.slot_bytes());

    // An attacher passing the same (dead) hugepage path still finds the shm.
    SlotStore t = SlotStore::attach("/test_ss_09", "/nonexistent_hugepages");
    assert(((unsigned char*)t.slot_addr(0))[0] == 0x5A);

    TEST_PASS();
}

// ==================================================================
static void test_prefault_large() {
    TEST_BEGIN("prefault=true over a multi-MB region");

    SlotStore::destroy("/test_ss_10", "");
    SlotStoreConfig c = cfg("/test_ss_10", 4096, 65536);   // 256 MiB
    c.prefault = true;
    SlotStore s = SlotStore::create(c);
    assert(s.data_bytes() == 4096ull * 65536);

    std::memset(s.slot_addr(4095), 0x11, s.slot_bytes());
    assert(((unsigned char*)s.slot_addr(4095))[65535] == 0x11);

    TEST_PASS();
}


// ==================================================================
static SlotStoreConfig cfg3(const std::string& name, uint64_t align = 4096) {
    SlotStoreConfig c = cfg(name, 16, 65536, align);
    c.swa   = {8, 4096};
    c.mamba = {3, 100000};
    return c;
}

static void test_multi_pool_layout() {
    TEST_BEGIN("three pools: ordered, aligned, disjoint, one contiguous range");

    SlotStore::destroy("/test_ss_11", "");
    SlotStore s = SlotStore::create(cfg3("/test_ss_11"));
    const ComponentType F = ComponentType::FULL, W = ComponentType::SWA, M = ComponentType::MAMBA;

    assert(s.pool_mask() == COMPONENT_MASK_ALL);
    assert(s.has_pool(F) && s.has_pool(W) && s.has_pool(M));
    assert(s.num_slots(F) == 16 && s.slot_bytes(F) == 65536);
    assert(s.num_slots(W) == 8  && s.slot_bytes(W) == 4096);
    assert(s.num_slots(M) == 3  && s.slot_bytes(M) == 102400);   // 100000 -> 4 KiB stride
    assert(s.num_slots() == 16 && s.slot_bytes() == 65536);      // FULL shorthands

    // FULL first, then SWA, then MAMBA; each start page-aligned.
    assert(s.pool_offset(F) == 0);
    assert(s.pool_offset(W) == 16 * 65536);
    assert(s.pool_offset(M) == 16 * 65536 + 8 * 4096);
    assert(s.pool_offset(M) % 4096 == 0);
    assert(s.pool_bytes(M) == 3 * 102400);
    assert(s.data_bytes() == s.pool_offset(M) + s.pool_bytes(M));
    assert((char*)s.pool_base(W) == (char*)s.data_base() + s.pool_offset(W));

    StoreGeometry g = s.layout();
    assert(g.pool_mask == COMPONENT_MASK_ALL);
    assert(g[W].base == s.pool_offset(W) && g[W].num_slots == 8 && g[W].slot_bytes == 4096);
    assert(g[M].base == s.pool_offset(M));

    // Pool alignment follows a larger slot_align.
    SlotStore::destroy("/test_ss_11b", "");
    SlotStore t = SlotStore::create(cfg3("/test_ss_11b", 1 << 16));
    assert(t.pool_offset(W) % (1 << 16) == 0);
    assert(t.slot_bytes(W) == 65536 && t.slot_bytes(M) == 131072);

    TEST_PASS();
}

static void test_multi_pool_addressing() {
    TEST_BEGIN("(component, slot) addressing: same id, different pool, different bytes");

    SlotStore::destroy("/test_ss_12", "");
    SlotStore s = SlotStore::create(cfg3("/test_ss_12"));
    const ComponentType F = ComponentType::FULL, W = ComponentType::SWA, M = ComponentType::MAMBA;

    for (uint64_t i = 0; i < 8; ++i) {
        assert(s.slot_addr(W, i) == (char*)s.data_base() + s.pool_offset(W) + i * 4096);
        assert(s.slot_offset(W, i) == s.pool_offset(W) + i * 4096);
    }
    assert(s.slot_addr(F, 2) != s.slot_addr(W, 2) && s.slot_addr(W, 2) != s.slot_addr(M, 2));
    assert(s.slot_offset(F, 0) == s.slot_offset(0));

    // Out of range per pool; unknown component.
    assert(s.slot_addr(W, 8) == nullptr && s.slot_offset(W, 8) == UINT64_MAX);
    assert(s.slot_addr(M, 3) == nullptr);
    assert(s.slot_addr(F, 16) == nullptr);
    assert(s.slot_addr((ComponentType)7, 0) == nullptr);
    assert(!s.has_pool((ComponentType)3));

    // Writes stay inside their pool.
    std::memset(s.slot_addr(F, 15), 0xF1, s.slot_bytes(F));
    std::memset(s.slot_addr(W, 0), 0xA2, s.slot_bytes(W));
    std::memset(s.slot_addr(W, 7), 0xA3, s.slot_bytes(W));
    std::memset(s.slot_addr(M, 0), 0xB4, s.slot_bytes(M));
    assert(((unsigned char*)s.slot_addr(F, 15))[65535] == 0xF1);
    assert(((unsigned char*)s.slot_addr(W, 0))[0] == 0xA2);
    assert(((unsigned char*)s.slot_addr(W, 7))[4095] == 0xA3);
    assert(((unsigned char*)s.slot_addr(M, 0))[0] == 0xB4);
    assert(((unsigned char*)s.slot_addr(M, 0))[102399] == 0xB4);
    // Last FULL byte and first SWA byte are neighbours, and distinct.
    assert((char*)s.slot_addr(F, 15) + 65536 == (char*)s.slot_addr(W, 0));

    TEST_PASS();
}

static void test_partial_pools() {
    TEST_BEGIN("FULL+MAMBA without SWA: absent pool takes no space");

    SlotStore::destroy("/test_ss_13", "");
    SlotStoreConfig c = cfg("/test_ss_13", 4, 4096);
    c.mamba = {2, 4096};
    SlotStore s = SlotStore::create(c);
    assert(s.pool_mask() == (COMPONENT_MASK_FULL | COMPONENT_MASK_MAMBA));
    assert(!s.has_pool(ComponentType::SWA));
    assert(s.num_slots(ComponentType::SWA) == 0 && s.slot_addr(ComponentType::SWA, 0) == nullptr);
    assert(s.pool_offset(ComponentType::MAMBA) == 4 * 4096);
    assert(s.data_bytes() == 6 * 4096);

    // Half-specified pool is refused.
    SlotStoreConfig bad = cfg("/test_ss_13b", 4, 4096);
    bad.swa = {8, 0};
    assert(throws([&] { SlotStore::create(bad); }));
    bad.swa = {0, 4096};
    assert(throws([&] { SlotStore::create(bad); }));

    TEST_PASS();
}

static void test_multi_pool_fork_attach() {
    TEST_BEGIN("fork: child reads the pool table and addresses every pool");

    SlotStore::destroy("/test_ss_14", "");
    SlotStore s = SlotStore::create(cfg3("/test_ss_14"));
    std::memset(s.slot_addr(ComponentType::SWA, 5), 0x5A, s.slot_bytes(ComponentType::SWA));
    std::memset(s.slot_addr(ComponentType::MAMBA, 2), 0x2B, s.slot_bytes(ComponentType::MAMBA));

    run_child([] {
        SlotStore t = SlotStore::attach("/test_ss_14", "");
        assert(t.pool_mask() == COMPONENT_MASK_ALL);
        assert(t.num_slots(ComponentType::SWA) == 8 && t.slot_bytes(ComponentType::MAMBA) == 102400);
        assert(((unsigned char*)t.slot_addr(ComponentType::SWA, 5))[100] == 0x5A);
        assert(((unsigned char*)t.slot_addr(ComponentType::MAMBA, 2))[102399] == 0x2B);
        std::memset(t.slot_addr(ComponentType::MAMBA, 0), 0x77, 16);
    });
    assert(((unsigned char*)s.slot_addr(ComponentType::MAMBA, 0))[15] == 0x77);

    // Overlapping pools in the header are refused on attach.
    int fd = ::shm_open("/test_ss_14", O_RDWR, 0666);
    assert(fd >= 0);
    void* raw = ::mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    assert(raw != MAP_FAILED);
    auto* hdr = (SlotStoreHeader*)raw;
    const uint64_t good = hdr->pools[1].data_offset;
    hdr->pools[1].data_offset = hdr->pools[0].data_offset;     // SWA over FULL
    assert(throws([] { SlotStore::attach("/test_ss_14", "", 200); }));
    hdr->pools[1].data_offset = good;
    hdr->pools[2].num_slots = 1000;                             // MAMBA past the range
    assert(throws([] { SlotStore::attach("/test_ss_14", "", 200); }));
    hdr->pools[2].num_slots = 3;
    assert(SlotStore::attach("/test_ss_14", "").num_slots(ComponentType::MAMBA) == 3);
    ::munmap(raw, 4096);
    ::close(fd);

    TEST_PASS();
}

// ==================================================================
int main() {
    printf("=== SlotStore tests ===\n");
    test_geometry();
    test_addressing();
    test_header_outside_data();
    test_read_write_roundtrip();
    test_fork_attach();
    test_attach_failures();
    test_bad_config();
    test_lifecycle();
    test_hugepage_fallback();
    test_prefault_large();
    test_multi_pool_layout();
    test_multi_pool_addressing();
    test_partial_pools();
    test_multi_pool_fork_attach();
    return tests_report();
}
