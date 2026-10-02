// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// test_slot_geometry — SlotTransfer's geometry contract, exercised against a
// stub backend rather than a real engine. No RDMA, no Mooncake, no fork: the
// thing under test is pure arithmetic and bookkeeping, and giving it a backend
// whose segment geometry we dictate is the only way to reach the failure this
// file exists for.
//
// The failure: a peer built with a different slot stride, whose published
// length still divides evenly by ours. Every length-based check passes, the
// slot ids get rescaled, and the transfer moves the wrong bytes with nothing
// reported anywhere. attach_peer must refuse it on the stride alone.
//
// Also covers the peer lifetime rule: detach_peer must refuse while a batch
// against that peer is still unreleased.

#include "data_test_utils.h"
#include "shmradix/slot_store.h"
#include "shmradix/slot_transfer.h"
#include "shmradix/transfer.h"

#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

using namespace shmradix;

static constexpr uint64_t kSlots     = 64;
static constexpr uint64_t kSlotBytes = 65536;

static SlotStoreConfig store_cfg(const std::string& name) {
    SlotStoreConfig c;
    c.name          = name;
    c.full.num_slots     = kSlots;
    c.full.slot_bytes    = kSlotBytes;
    c.slot_align    = 4096;
    c.hugepage_path = "";
    c.prefault      = false;
    return c;
}

// A backend that publishes whatever geometry the test asks for and records the
// descriptors it was handed. Transfers complete instantly and move no bytes —
// this file is about what reaches submit(), not about what the wire does.
namespace {

class StubBackend : public ITransferBackend {
public:
    // Register a peer name the test wants open_segment() to resolve.
    void offer(const std::string& name, uint64_t base, uint64_t length) {
        offered_[name] = SegmentInfo{INVALID_SEGMENT, base, length};
    }

    int register_memory(void* addr, uint64_t len, const std::string&) override {
        reg_addr_ = addr;
        reg_len_  = len;
        return XFER_OK;
    }
    int unregister_memory(void*) override { return XFER_OK; }

    int open_segment(const std::string& seg_name, SegmentInfo& out) override {
        auto it = offered_.find(seg_name);
        if (it == offered_.end()) return XFER_ENOENT;
        ++opens_;
        out    = it->second;
        out.id = next_id_++;
        return XFER_OK;
    }
    void close_segment(SegmentId) override { ++closes_; }

    const std::string& local_segment() const override { return local_; }

    BatchHandle submit(const TransferDesc* descs, int n) override {
        last_.assign(descs, descs + n);
        return next_batch_++;
    }
    XferState poll(BatchHandle, uint64_t* done) override {
        if (done) *done = 0;
        return XferState::COMPLETED;
    }
    XferState wait(BatchHandle, int) override { return XferState::COMPLETED; }
    int       release(BatchHandle) override { return release_rc_; }
    const char* backend_name() const override { return "stub"; }

    // Stand in for a batch mooncake refuses to free: still running.
    void refuse_release(bool on) { release_rc_ = on ? XFER_EBUSY : XFER_OK; }

    void*    reg_addr() const { return reg_addr_; }
    uint64_t reg_len()  const { return reg_len_; }
    int      opens()    const { return opens_; }
    int      closes()   const { return closes_; }
    const std::vector<TransferDesc>& last() const { return last_; }

private:
    std::unordered_map<std::string, SegmentInfo> offered_;
    std::vector<TransferDesc> last_;
    std::string local_ = "stub-local";
    void*       reg_addr_  = nullptr;
    uint64_t    reg_len_   = 0;
    SegmentId   next_id_   = 1;
    BatchHandle next_batch_= 100;
    int         opens_     = 0;
    int         closes_    = 0;
    int         release_rc_= XFER_OK;
};

} // namespace

// A peer whose stride matches ours attaches, and its slot count is the
// published length divided by that stride.
static void test_matching_stride() {
    TEST_BEGIN("attach_peer: matching stride resolves the slot count");

    const std::string shm = "/tsg_match";
    SlotStore::destroy(shm, "");
    SlotStore s = SlotStore::create(store_cfg(shm));

    StubBackend be;
    be.offer("peerA", 0x700000000000ULL, 128 * kSlotBytes);
    SlotTransfer x(s, be);
    assert(x.register_store() == XFER_OK);
    assert(be.reg_addr() == s.data_base());
    assert(be.reg_len() == s.data_bytes());

    SlotGeometry g;
    assert(x.attach_peer("peerA", kSlotBytes, &g) == XFER_OK);
    assert(g.slot_bytes == kSlotBytes);
    assert(g.num_slots == 128);
    assert(g.base == 0x700000000000ULL);

    TEST_PASS();
}

// The bug this file exists for: peer stride 4 KiB, ours 64 KiB, published
// length 4 MiB. 4 MiB is a whole number of 64 KiB slots, so a length-only
// check accepts it and silently rescales every slot id by 16x.
static void test_mismatched_stride_that_divides() {
    TEST_BEGIN("attach_peer: divisible-but-wrong stride is refused");

    const std::string shm = "/tsg_divis";
    SlotStore::destroy(shm, "");
    SlotStore s = SlotStore::create(store_cfg(shm));

    StubBackend be;
    const uint64_t peer_stride = 4096;
    const uint64_t peer_slots  = 1024;
    be.offer("peerB", 0x700000000000ULL, peer_stride * peer_slots);

    // The premise: this really would have passed the length check.
    assert((peer_stride * peer_slots) % kSlotBytes == 0);

    SlotTransfer x(s, be);
    assert(x.register_store() == XFER_OK);

    SlotGeometry g;
    assert(x.attach_peer("peerB", peer_stride, &g) == XFER_EGEOMETRY);

    // Refused means not attached, and refused before the segment was opened —
    // no handle to leak.
    assert(!x.peer_geometry("peerB", g));
    assert(be.opens() == 0);

    // And nothing can be transferred to it.
    const int32_t one = 0;
    assert(x.read("peerB", &one, &one, 1) == INVALID_BATCH);
    assert(x.write("peerB", &one, &one, 1) == INVALID_BATCH);

    TEST_PASS();
}

// A length that is not a whole number of slots is still rejected, and the
// segment opened to discover that is closed again.
static void test_ragged_length() {
    TEST_BEGIN("attach_peer: ragged published length is refused");

    const std::string shm = "/tsg_ragged";
    SlotStore::destroy(shm, "");
    SlotStore s = SlotStore::create(store_cfg(shm));

    StubBackend be;
    be.offer("peerC", 0x700000000000ULL, 8 * kSlotBytes + 4096);
    SlotTransfer x(s, be);
    assert(x.register_store() == XFER_OK);

    assert(x.attach_peer("peerC", kSlotBytes) == XFER_EGEOMETRY);
    assert(be.opens() == 1);
    assert(be.closes() == 1);

    // Shorter than a single slot is the same class of failure.
    be.offer("peerD", 0x700000000000ULL, kSlotBytes / 2);
    assert(x.attach_peer("peerD", kSlotBytes) == XFER_EGEOMETRY);

    TEST_PASS();
}

// Re-attaching is idempotent, but only while the caller keeps telling us the
// same stride. A registry that now reports a different one must not be
// answered out of the cache.
static void test_reattach_stride_change() {
    TEST_BEGIN("attach_peer: cached geometry never outvotes a new stride");

    const std::string shm = "/tsg_reattach";
    SlotStore::destroy(shm, "");
    SlotStore s = SlotStore::create(store_cfg(shm));

    StubBackend be;
    be.offer("peerE", 0x700000000000ULL, 32 * kSlotBytes);
    SlotTransfer x(s, be);
    assert(x.register_store() == XFER_OK);

    assert(x.attach_peer("peerE", kSlotBytes) == XFER_OK);
    assert(be.opens() == 1);

    // Same stride again: served from the cache, no second open.
    assert(x.attach_peer("peerE", kSlotBytes) == XFER_OK);
    assert(be.opens() == 1);

    // Different stride: refused, and the cached entry left alone.
    assert(x.attach_peer("peerE", 4096) == XFER_EGEOMETRY);
    SlotGeometry g;
    assert(x.peer_geometry("peerE", g));
    assert(g.slot_bytes == kSlotBytes);

    TEST_PASS();
}

// The explicit opt-out still works, for bring-up paths with no registry.
static void test_unchecked_opt_out() {
    TEST_BEGIN("attach_peer: UNCHECKED assumes the local stride");

    const std::string shm = "/tsg_unchecked";
    SlotStore::destroy(shm, "");
    SlotStore s = SlotStore::create(store_cfg(shm));

    StubBackend be;
    be.offer("peerF", 0x700000000000ULL, 16 * kSlotBytes);
    SlotTransfer x(s, be);
    assert(x.register_store() == XFER_OK);

    SlotGeometry g;
    assert(x.attach_peer("peerF", PEER_SLOT_BYTES_UNCHECKED, &g) == XFER_OK);
    assert(g.slot_bytes == kSlotBytes);
    assert(g.num_slots == 16);

    // Opting out does not poison the cache: a later caller that does know the
    // stride still gets a straight answer.
    assert(x.attach_peer("peerF", kSlotBytes) == XFER_OK);
    assert(x.attach_peer("peerF", 4096) == XFER_EGEOMETRY);

    // A zero stride is a caller bug, not an opt-out.
    assert(x.attach_peer("peerG", uint64_t{0}) == XFER_EINVAL);

    TEST_PASS();
}

// Once attached, the addresses handed to the backend must be built from that
// stride — the check above is only worth having if the arithmetic depends on it.
static void test_descriptor_addressing() {
    TEST_BEGIN("read/write: descriptors are built from the attached stride");

    const std::string shm = "/tsg_addr";
    SlotStore::destroy(shm, "");
    SlotStore s = SlotStore::create(store_cfg(shm));

    StubBackend be;
    const uint64_t base = 0x700000000000ULL;
    be.offer("peerH", base, 64 * kSlotBytes);
    SlotTransfer x(s, be);
    assert(x.register_store() == XFER_OK);
    assert(x.attach_peer("peerH", kSlotBytes) == XFER_OK);

    // A run contiguous on both sides fuses; the outlier does not.
    const int32_t local[4]  = {10, 11, 12, 40};
    const int32_t remote[4] = {20, 21, 22, 55};
    int ndesc = 0;
    BatchHandle h = x.read("peerH", remote, local, 4, &ndesc);
    assert(h != INVALID_BATCH);
    assert(ndesc == 2);

    const auto& d = be.last();
    assert(d.size() == 2);
    assert(d[0].op == XferOp::READ);
    assert(d[0].local_addr == s.slot_addr(10));
    assert(d[0].target_offset == base + 20 * kSlotBytes);
    assert(d[0].length == 3 * kSlotBytes);
    assert(d[1].local_addr == s.slot_addr(40));
    assert(d[1].target_offset == base + 55 * kSlotBytes);
    assert(d[1].length == kSlotBytes);

    x.release(h);

    TEST_PASS();
}

// Slot ids are range-checked against each side's own count, which differ here.
static void test_slot_range_checks() {
    TEST_BEGIN("read/write: slot ids checked against each side's own count");

    const std::string shm = "/tsg_range";
    SlotStore::destroy(shm, "");
    SlotStore s = SlotStore::create(store_cfg(shm));

    StubBackend be;
    be.offer("peerI", 0x700000000000ULL, 16 * kSlotBytes);   // peer is smaller
    SlotTransfer x(s, be);
    assert(x.register_store() == XFER_OK);

    SlotGeometry g;
    assert(x.attach_peer("peerI", kSlotBytes, &g) == XFER_OK);
    assert(g.num_slots == 16 && s.num_slots() == kSlots);

    const int32_t ok_local = 63, ok_remote = 15;
    assert(x.read("peerI", &ok_remote, &ok_local, 1) != INVALID_BATCH);

    // Valid locally, past the end of the peer.
    const int32_t bad_remote = 16;
    assert(x.read("peerI", &bad_remote, &ok_local, 1) == INVALID_BATCH);

    // Valid on the peer, past the end of us.
    const int32_t bad_local = (int32_t)kSlots;
    assert(x.read("peerI", &ok_remote, &bad_local, 1) == INVALID_BATCH);

    const int32_t neg = -1;
    assert(x.read("peerI", &ok_remote, &neg, 1) == INVALID_BATCH);
    assert(x.write("peerI", &ok_local, &neg, 1) == INVALID_BATCH);

    TEST_PASS();
}

static void test_detach() {
    TEST_BEGIN("detach_peer: closes the segment and forgets the geometry");

    const std::string shm = "/tsg_detach";
    SlotStore::destroy(shm, "");
    SlotStore s = SlotStore::create(store_cfg(shm));

    StubBackend be;
    be.offer("peerJ", 0x700000000000ULL, 8 * kSlotBytes);
    SlotTransfer x(s, be);
    assert(x.register_store() == XFER_OK);
    assert(x.attach_peer("peerJ", kSlotBytes) == XFER_OK);

    assert(x.detach_peer("peerJ") == XFER_OK);
    assert(be.closes() == 1);
    SlotGeometry g;
    assert(!x.peer_geometry("peerJ", g));

    // Detaching twice is not an error, and re-attaching opens afresh.
    assert(x.detach_peer("peerJ") == XFER_OK);
    assert(be.closes() == 1);
    assert(x.attach_peer("peerJ", kSlotBytes) == XFER_OK);
    assert(be.opens() == 2);

    TEST_PASS();
}

// ==================================================================
static void test_detach_while_inflight() {
    TEST_BEGIN("detach_peer: refused while a batch is unreleased");

    const std::string shm = "/tsg_detach_busy";
    SlotStore::destroy(shm, "");
    SlotStore s = SlotStore::create(store_cfg(shm));

    StubBackend be;
    be.offer("peerK", 0x700000000000ULL, 8 * kSlotBytes);
    SlotTransfer x(s, be);
    assert(x.register_store() == XFER_OK);
    assert(x.attach_peer("peerK", kSlotBytes) == XFER_OK);

    const int32_t rs[2] = {0, 1}, ls[2] = {0, 1};
    const BatchHandle h1 = x.read("peerK", rs, ls, 2);
    assert(h1 != INVALID_BATCH);

    // In flight: the segment stays open and the geometry stays visible.
    assert(x.detach_peer("peerK") == XFER_EBUSY);
    assert(be.closes() == 0);
    SlotGeometry g;
    assert(x.peer_geometry("peerK", g));

    // A release the backend refuses frees nothing and keeps the peer held.
    be.refuse_release(true);
    assert(x.release(h1) == XFER_EBUSY);
    assert(x.detach_peer("peerK") == XFER_EBUSY);
    assert(be.closes() == 0);
    be.refuse_release(false);

    // Two batches: releasing one is not enough.
    const BatchHandle h2 = x.write("peerK", ls, rs, 2);
    assert(h2 != INVALID_BATCH && h2 != h1);
    assert(x.release(h1) == XFER_OK);
    assert(x.detach_peer("peerK") == XFER_EBUSY);
    // Releasing the same handle twice must not double-decrement.
    x.release(h1);
    assert(x.detach_peer("peerK") == XFER_EBUSY);
    assert(be.closes() == 0);

    x.release(h2);
    assert(x.detach_peer("peerK") == XFER_OK);
    assert(be.closes() == 1);

    // A rejected batch reserves nothing.
    assert(x.attach_peer("peerK", kSlotBytes) == XFER_OK);
    const int32_t bad[2] = {0, 99};
    assert(x.read("peerK", bad, ls, 2) == INVALID_BATCH);
    assert(x.detach_peer("peerK") == XFER_OK);
    assert(be.closes() == 2);

    TEST_PASS();
}


// ==================================================================
// Multi-pool stores: the peer's layout comes from the registry, and every
// check and address is per pool.
static SlotStoreConfig store_cfg3(const std::string& name) {
    SlotStoreConfig c = store_cfg(name);
    c.swa   = {32, 4096};
    c.mamba = {4, 8192};
    return c;
}

static void test_multi_pool_attach() {
    TEST_BEGIN("attach_peer(layout): same pools+strides pass, counts may differ");

    const std::string shm = "/tsg_mp";
    SlotStore::destroy(shm, "");
    SlotStore s = SlotStore::create(store_cfg3(shm));
    const ComponentType F = ComponentType::FULL, W = ComponentType::SWA, M = ComponentType::MAMBA;

    // Peer: twice the FULL slots, half the SWA, same strides.
    StoreGeometry pub;
    pub.pool_mask = COMPONENT_MASK_ALL;
    pub[F] = {0, 128, kSlotBytes};
    pub[W] = {128 * kSlotBytes, 16, 4096};
    pub[M] = {128 * kSlotBytes + 16 * 4096, 4, 8192};
    const uint64_t len  = pub[M].base + 4 * 8192;
    const uint64_t base = 0x700000000000ULL;

    StubBackend be;
    be.offer("peerM", base, len);
    SlotTransfer x(s, be);
    assert(x.register_store() == XFER_OK);
    assert(be.reg_len() == s.data_bytes());

    StoreGeometry g;
    assert(x.attach_peer("peerM", &pub, &g) == XFER_OK);
    assert(g.pool_mask == COMPONENT_MASK_ALL);
    assert(g[F].base == base && g[F].num_slots == 128);
    assert(g[W].base == base + 128 * kSlotBytes && g[W].num_slots == 16);
    assert(g[M].base == base + pub[M].base && g[M].slot_bytes == 8192);

    // The stride form is for FULL-only stores.
    assert(x.attach_peer("peerM", kSlotBytes) == XFER_EINVAL);

    // Cached geometry vs a contradicting layout.
    StoreGeometry other = pub;
    other[W].slot_bytes = 8192;
    assert(x.attach_peer("peerM", &other) == XFER_EGEOMETRY);
    assert(x.attach_peer("peerM", &pub) == XFER_OK);
    assert(be.opens() == 1);

    TEST_PASS();
}

static void test_multi_pool_refusals() {
    TEST_BEGIN("attach_peer(layout): mask / per-pool stride / fit mismatches refused");

    const std::string shm = "/tsg_mp_bad";
    SlotStore::destroy(shm, "");
    SlotStore s = SlotStore::create(store_cfg3(shm));
    const ComponentType F = ComponentType::FULL, W = ComponentType::SWA, M = ComponentType::MAMBA;

    StubBackend be;
    const uint64_t base = 0x700000000000ULL;
    be.offer("peerN", base, s.data_bytes());
    SlotTransfer x(s, be);
    assert(x.register_store() == XFER_OK);

    const StoreGeometry mine = s.layout();

    StoreGeometry no_mamba = mine;                 // fewer pools
    no_mamba.pool_mask = COMPONENT_MASK_FULL | COMPONENT_MASK_SWA;
    assert(x.attach_peer("peerN", &no_mamba) == XFER_EGEOMETRY);

    StoreGeometry swa_stride = mine;               // FULL agrees, SWA does not
    swa_stride[W].slot_bytes = 8192;
    assert(x.attach_peer("peerN", &swa_stride) == XFER_EGEOMETRY);

    StoreGeometry too_big = mine;                  // MAMBA runs past the segment
    too_big[M].num_slots = 5;
    assert(x.attach_peer("peerN", &too_big) == XFER_EGEOMETRY);

    StoreGeometry overlap = mine;                  // SWA over FULL
    overlap[W].base = mine[F].base;
    assert(x.attach_peer("peerN", &overlap) == XFER_EGEOMETRY);

    StoreGeometry no_full = mine;
    no_full.pool_mask = COMPONENT_MASK_SWA | COMPONENT_MASK_MAMBA;
    assert(x.attach_peer("peerN", &no_full) == XFER_EINVAL);

    assert(be.opens() == 2 && be.closes() == 2);   // only the fit checks opened
    assert(x.attach_peer("peerN", &mine) == XFER_OK);

    // nullptr on a multi-pool store: our layout assumed, must still fit.
    be.offer("peerO", base, s.data_bytes() - 1);
    assert(x.attach_peer("peerO", nullptr) == XFER_EGEOMETRY);
    be.offer("peerP", base, s.data_bytes());
    StoreGeometry g;
    assert(x.attach_peer("peerP", nullptr, &g) == XFER_OK);
    assert(g[M].base == base + mine[M].base);

    TEST_PASS();
}

static void test_multi_pool_descriptors() {
    TEST_BEGIN("read/write(component): addresses from that pool on both sides");

    const std::string shm = "/tsg_mp_desc";
    SlotStore::destroy(shm, "");
    SlotStore s = SlotStore::create(store_cfg3(shm));
    const ComponentType F = ComponentType::FULL, W = ComponentType::SWA, M = ComponentType::MAMBA;

    StubBackend be;
    const uint64_t base = 0x700000000000ULL;
    be.offer("peerQ", base, s.data_bytes());
    SlotTransfer x(s, be);
    assert(x.register_store() == XFER_OK);
    const StoreGeometry mine = s.layout();
    assert(x.attach_peer("peerQ", &mine) == XFER_OK);

    // Same ids, SWA pool: local from our SWA pool, remote at the peer's SWA base.
    const int32_t local[3]  = {2, 3, 4};
    const int32_t remote[3] = {10, 11, 12};
    int ndesc = 0;
    BatchHandle h = x.read("peerQ", W, remote, local, 3, &ndesc);
    assert(h != INVALID_BATCH && ndesc == 1);
    const auto& d = be.last();
    assert(d[0].local_addr == s.slot_addr(W, 2));
    assert(d[0].local_addr != s.slot_addr(F, 2));
    assert(d[0].target_offset == base + mine[W].base + 10 * 4096);
    assert(d[0].length == 3 * 4096);
    x.release(h);

    // FULL overload == FULL component.
    h = x.read("peerQ", remote, local, 3, &ndesc);
    assert(h != INVALID_BATCH);
    assert(be.last()[0].local_addr == s.slot_addr(F, 2));
    assert(be.last()[0].target_offset == base + 10 * kSlotBytes);
    x.release(h);

    // MAMBA write, single slot.
    const int32_t one_l = 3, one_r = 0;
    h = x.write("peerQ", M, &one_l, &one_r, 1, &ndesc);
    assert(h != INVALID_BATCH && ndesc == 1);
    assert(be.last()[0].op == XferOp::WRITE);
    assert(be.last()[0].local_addr == s.slot_addr(M, 3));
    assert(be.last()[0].target_offset == base + mine[M].base);
    assert(be.last()[0].length == 8192);
    x.release(h);

    // Range checks are per pool; unknown component refused.
    const int32_t swa_edge = 32, mamba_edge = 4;
    assert(x.read("peerQ", W, &one_r, &swa_edge, 1) == INVALID_BATCH);
    assert(x.read("peerQ", M, &mamba_edge, &one_l, 1) == INVALID_BATCH);
    assert(x.read("peerQ", (ComponentType)3, &one_r, &one_l, 1) == INVALID_BATCH);

    // A FULL-only local store has no SWA pool.
    const std::string shm1 = "/tsg_mp_desc1";
    SlotStore::destroy(shm1, "");
    SlotStore s1 = SlotStore::create(store_cfg(shm1));
    StubBackend be1;
    be1.offer("peerR", base, 8 * kSlotBytes);
    SlotTransfer x1(s1, be1);
    assert(x1.register_store() == XFER_OK);
    assert(x1.attach_peer("peerR", kSlotBytes) == XFER_OK);
    assert(x1.read("peerR", W, &one_r, &one_l, 1) == INVALID_BATCH);

    TEST_PASS();
}

int main() {
    printf("=== SlotTransfer geometry contract ===\n");

    test_matching_stride();
    test_mismatched_stride_that_divides();
    test_ragged_length();
    test_reattach_stride_change();
    test_unchecked_opt_out();
    test_descriptor_addressing();
    test_slot_range_checks();
    test_multi_pool_attach();
    test_multi_pool_refusals();
    test_multi_pool_descriptors();
    test_detach();
    test_detach_while_inflight();

    for (const char* n : {"/tsg_match", "/tsg_divis", "/tsg_ragged",
                          "/tsg_reattach", "/tsg_unchecked", "/tsg_addr",
                          "/tsg_range", "/tsg_detach",
                          "/tsg_detach_busy"})
        SlotStore::destroy(n, "");

    return tests_report();
}
