// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// test_slot_transfer_e2e — SlotTransfer driving a real TransferEngine between
// two processes. This is where the slot arithmetic above the backend — run
// coalescing included — is proven, on the wire it will actually run on.
//
//   - the initiator attaches the publisher's segment by name and derives the
//     geometry from it (no out-of-band agreement on slot count)
//   - a contiguous run of N slots crosses as ONE request, and every byte of it
//     is verified against a per-slot pattern
//   - a scattered write lands in exactly the named remote slots, neighbours
//     untouched
//   - it runs over rdma and over tcp, since the layer above must not care
//
// Every engine lives inside a forked child (ibverbs contexts do not survive
// fork), and the whole file SKIPs when no engine can come up.

#include "data_test_utils.h"
#include "shmradix/slot_store.h"
#include "shmradix/slot_transfer.h"
#include "shmradix/transfer.h"

#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>

using namespace shmradix;

static constexpr uint64_t kSlots     = 64;
static constexpr uint64_t kSlotBytes = 65536;

static SlotStoreConfig store_cfg(const std::string& name) {
    SlotStoreConfig c;
    c.name = name;
    c.full.num_slots = kSlots;
    c.full.slot_bytes = kSlotBytes;
    c.slot_align = 4096;
    c.hugepage_path = "";
    c.prefault = true;      // an MR over unfaulted pages is a bad idea
    return c;
}

static TransferConfig mc_cfg(bool publish, const std::string& protocol) {
    TransferConfig c;
    c.backend = "mooncake";
    c.metadata = "P2PHANDSHAKE";
    c.ip_or_host = "127.0.0.1";
    c.rpc_port = 0;         // the engine binds what it can; we read the name back
    c.protocol = protocol;
    c.publish = publish;
    return c;
}

// Slot i is filled with a byte derived from i, so a mis-addressed slot shows up
// as the wrong pattern rather than as plausible-looking data.
static unsigned char pattern(uint64_t slot) { return (unsigned char)(slot * 37 + 11); }

static void fill(SlotStore& s, uint64_t slot, unsigned char v) {
    std::memset(s.slot_addr(slot), v, s.slot_bytes());
}
static bool holds(const SlotStore& s, uint64_t slot, unsigned char v) {
    auto* p = (const unsigned char*)s.slot_addr(slot);
    for (uint64_t i = 0; i < s.slot_bytes(); ++i)
        if (p[i] != v) return false;
    return true;
}

// Children report where they gave up through their exit code; print it, because
// "the child failed" on its own is not a debuggable message.
static bool reap_step(pid_t pid, const char* who) {
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return false;
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) return true;
    if (WIFEXITED(status))
        fprintf(stderr, "\n    %s child exited with step code %d\n", who, WEXITSTATUS(status));
    else
        fprintf(stderr, "\n    %s child died on signal %d\n", who, WTERMSIG(status));
    return false;
}

static void write_str(int fd, const std::string& s) {
    uint32_t n = (uint32_t)s.size();
    assert(::write(fd, &n, sizeof(n)) == (ssize_t)sizeof(n));
    assert(::write(fd, s.data(), n) == (ssize_t)n);
}
static bool read_str(int fd, std::string& out) {
    uint32_t n = 0;
    if (::read(fd, &n, sizeof(n)) != (ssize_t)sizeof(n)) return false;
    out.resize(n);
    return ::read(fd, &out[0], n) == (ssize_t)n;
}

// ==================================================================
static bool mooncake_available() {
    static int cached = -1;
    if (cached >= 0) return cached == 1;
    cached = child_succeeds([] {
        auto be = create_transfer_backend(mc_cfg(true, "rdma"));
        _exit(be ? 0 : 1);
    }) ? 1 : 0;
    return cached == 1;
}

// ==================================================================
// The whole exchange, at the SlotTransfer level: nobody here builds a
// TransferDesc or touches an address.
static bool run_e2e(const std::string& protocol, const std::string& tag) {
    int name_pipe[2], done_pipe[2];
    assert(::pipe(name_pipe) == 0 && ::pipe(done_pipe) == 0);

    const std::string shm_a = "/test_ste_" + tag + "_a";
    const std::string shm_b = "/test_ste_" + tag + "_b";
    SlotStore::destroy(shm_a, "");
    SlotStore::destroy(shm_b, "");

    // Slots the initiator will push back into the publisher, deliberately not a
    // single run so the descriptor count is interesting.
    // Kept clear of the range the initiator reads (8..23) so the two halves of
    // the exchange cannot mask each other.
    const int32_t push_local[]  = {40, 41, 42, 50};
    const int32_t push_remote[] = {60, 61, 62, 55};
    const int      push_n       = 4;

    // Publisher: owns the data, then audits what came back.
    pid_t pa = spawn_child([&] {
        ::close(name_pipe[0]);
        ::close(done_pipe[1]);

        SlotStore s = SlotStore::create(store_cfg(shm_a));
        auto be = create_transfer_backend(mc_cfg(true, protocol));
        if (!be) _exit(2);
        SlotTransfer xfer(s, *be);
        if (xfer.register_store("cpu:0") != XFER_OK) _exit(3);

        for (uint64_t i = 0; i < kSlots; ++i) fill(s, i, pattern(i));
        // Landing zone for the push: zeroed so a no-op would be caught.
        for (int i = 0; i < push_n; ++i) fill(s, (uint64_t)push_remote[i], 0x00);

        write_str(name_pipe[1], xfer.local_segment());

        char sig = 0;
        if (::read(done_pipe[0], &sig, 1) != 1 || sig != 'k') _exit(4);

        // Exactly the named slots changed, and to the right thing.
        for (int i = 0; i < push_n; ++i)
            if (!holds(s, (uint64_t)push_remote[i], pattern((uint64_t)push_local[i]))) _exit(5);
        for (uint64_t i = 0; i < kSlots; ++i) {
            bool pushed = false;
            for (int k = 0; k < push_n; ++k) pushed |= ((uint64_t)push_remote[k] == i);
            if (!pushed && !holds(s, i, pattern(i))) _exit(6);
        }
        _exit(0);
    });

    // Initiator: publishes nothing, pulls a run and pushes a scattered set.
    pid_t pb = spawn_child([&] {
        ::close(name_pipe[1]);
        ::close(done_pipe[0]);

        SlotStore s = SlotStore::create(store_cfg(shm_b));
        auto be = create_transfer_backend(mc_cfg(false, protocol));
        if (!be) _exit(2);
        SlotTransfer xfer(s, *be);
        if (xfer.register_store("cpu:0") != XFER_OK) _exit(3);

        std::string peer;
        if (!read_str(name_pipe[0], peer) || peer.empty()) _exit(4);

        // Slot count comes off the segment; the stride is the contract both
        // sides were configured with, standing in for the registry here.
        SlotGeometry geo;
        if (xfer.attach_peer(peer, kSlotBytes, &geo) != XFER_OK) _exit(5);
        if (geo.num_slots != kSlots || geo.slot_bytes != s.slot_bytes()) _exit(6);

        // READ a 16-slot run: contiguous on both sides, so one request of 1 MiB.
        const int kRun = 16;
        std::vector<int32_t> rl(kRun), rr(kRun);
        for (int i = 0; i < kRun; ++i) { rl[i] = i; rr[i] = 8 + i; }
        for (int i = 0; i < kRun; ++i) fill(s, (uint64_t)rl[i], 0x00);
        fill(s, (uint64_t)kRun, 0xEE);          // sentinel just past the run

        int ndesc = 0;
        BatchHandle h = xfer.read(peer, rr.data(), rl.data(), kRun, &ndesc);
        if (h == INVALID_BATCH) _exit(7);
        if (ndesc != 1) _exit(8);
        if (xfer.wait(h, 30000) != XferState::COMPLETED) _exit(9);
        uint64_t done = 0;
        xfer.poll(h, &done);
        xfer.release(h);
        if (done != (uint64_t)kRun * s.slot_bytes()) _exit(10);
        for (int i = 0; i < kRun; ++i)
            if (!holds(s, (uint64_t)rl[i], pattern((uint64_t)rr[i]))) _exit(11);
        // The slot just past the run was not dragged along.
        if (!holds(s, (uint64_t)kRun, 0xEE)) _exit(12);

        // WRITE a scattered set back: {40,41,42}->{60,61,62} fuses, {50}->{55}
        // does not, so two requests.
        for (int i = 0; i < push_n; ++i)
            fill(s, (uint64_t)push_local[i], pattern((uint64_t)push_local[i]));
        BatchHandle h2 = xfer.write(peer, push_local, push_remote, push_n, &ndesc);
        if (h2 == INVALID_BATCH) _exit(13);
        if (ndesc != 2) _exit(14);
        if (xfer.wait(h2, 30000) != XferState::COMPLETED) _exit(15);
        xfer.release(h2);

        // Read one of them back to confirm the peer really has it.
        const int32_t back_l = 30, back_r = 61;
        fill(s, (uint64_t)back_l, 0x00);
        BatchHandle h3 = xfer.read(peer, &back_r, &back_l, 1);
        if (h3 == INVALID_BATCH) _exit(16);
        if (xfer.wait(h3, 30000) != XferState::COMPLETED) _exit(17);
        xfer.release(h3);
        if (!holds(s, (uint64_t)back_l, pattern(41))) _exit(18);

        char sig = 'k';
        if (::write(done_pipe[1], &sig, 1) != 1) _exit(19);
        _exit(0);
    });

    ::close(name_pipe[0]); ::close(name_pipe[1]);
    ::close(done_pipe[0]); ::close(done_pipe[1]);

    const bool ok_b = reap_step(pb, "initiator");
    const bool ok_a = reap_step(pa, "publisher");
    SlotStore::destroy(shm_a, "");
    SlotStore::destroy(shm_b, "");
    return ok_a && ok_b;
}

static void test_e2e_rdma() {
    TEST_BEGIN("two processes over rdma: attach, coalesced read, scattered write");
    if (!mooncake_available()) { TEST_SKIP("no Mooncake engine"); return; }
    assert(run_e2e("rdma", "rdma"));
    TEST_PASS();
}

static void test_e2e_tcp() {
    TEST_BEGIN("two processes over tcp: same exchange, no RDMA involved");
    if (!mooncake_available()) { TEST_SKIP("no Mooncake engine"); return; }
    assert(run_e2e("tcp", "tcp"));
    TEST_PASS();
}

// ==================================================================
// A whole store pushed out and pulled back, byte for byte — the shape the bench
// times, run once here so a correctness break shows up in ctest and not only in
// a benchmark number.
static void test_roundtrip_all_slots() {
    TEST_BEGIN("round trip: every slot written to a peer and read back, verified");
    if (!mooncake_available()) { TEST_SKIP("no Mooncake engine"); return; }

    int name_pipe[2], done_pipe[2];
    assert(::pipe(name_pipe) == 0 && ::pipe(done_pipe) == 0);
    const std::string shm_a = "/test_ste_rt_a", shm_b = "/test_ste_rt_b";
    SlotStore::destroy(shm_a, "");
    SlotStore::destroy(shm_b, "");

    // Publisher is passive: it only holds memory and waits.
    pid_t pa = spawn_child([&] {
        ::close(name_pipe[0]);
        ::close(done_pipe[1]);
        SlotStore s = SlotStore::create(store_cfg(shm_a));
        auto be = create_transfer_backend(mc_cfg(true, "rdma"));
        if (!be) _exit(2);
        SlotTransfer xfer(s, *be);
        if (xfer.register_store("cpu:0") != XFER_OK) _exit(3);
        std::memset(s.data_base(), 0x00, s.data_bytes());
        write_str(name_pipe[1], xfer.local_segment());
        char sig = 0;
        if (::read(done_pipe[0], &sig, 1) != 1 || sig != 'k') _exit(4);
        _exit(0);
    });

    pid_t pb = spawn_child([&] {
        ::close(name_pipe[1]);
        ::close(done_pipe[0]);
        SlotStore s = SlotStore::create(store_cfg(shm_b));
        auto be = create_transfer_backend(mc_cfg(false, "rdma"));
        if (!be) _exit(2);
        SlotTransfer xfer(s, *be);
        if (xfer.register_store("cpu:0") != XFER_OK) _exit(3);

        std::string peer;
        if (!read_str(name_pipe[0], peer) || peer.empty()) _exit(4);
        if (xfer.attach_peer(peer, kSlotBytes) != XFER_OK) _exit(5);

        // Half the store out, then the same half back into the other half.
        const int n = (int)kSlots / 2;
        std::vector<int32_t> src(n), rem(n), dst(n);
        for (int i = 0; i < n; ++i) {
            src[i] = i;
            rem[i] = i;
            dst[i] = n + i;
            fill(s, (uint64_t)src[i], pattern((uint64_t)i));
            fill(s, (uint64_t)dst[i], 0x00);
        }

        int ndesc = 0;
        BatchHandle hw = xfer.write(peer, src.data(), rem.data(), n, &ndesc);
        if (hw == INVALID_BATCH || ndesc != 1) _exit(6);
        if (xfer.wait(hw, 30000) != XferState::COMPLETED) _exit(7);
        xfer.release(hw);

        BatchHandle hr = xfer.read(peer, rem.data(), dst.data(), n, &ndesc);
        if (hr == INVALID_BATCH || ndesc != 1) _exit(8);
        if (xfer.wait(hr, 30000) != XferState::COMPLETED) _exit(9);
        xfer.release(hr);

        for (int i = 0; i < n; ++i)
            if (std::memcmp(s.slot_addr((uint64_t)src[i]), s.slot_addr((uint64_t)dst[i]),
                            s.slot_bytes()) != 0) _exit(10);

        char sig = 'k';
        if (::write(done_pipe[1], &sig, 1) != 1) _exit(11);
        _exit(0);
    });

    ::close(name_pipe[0]); ::close(name_pipe[1]);
    ::close(done_pipe[0]); ::close(done_pipe[1]);
    const bool ok_b = reap_step(pb, "initiator");
    const bool ok_a = reap_step(pa, "publisher");
    SlotStore::destroy(shm_a, "");
    SlotStore::destroy(shm_b, "");
    assert(ok_a && ok_b);

    TEST_PASS();
}


// ==================================================================
// Three-pool stores on both sides: SWA and MAMBA slots move through their own
// pools, FULL bytes untouched.
static SlotStoreConfig store_cfg3(const std::string& name) {
    SlotStoreConfig c = store_cfg(name);
    c.swa   = {32, 4096};
    c.mamba = {4, 8192};
    return c;
}
static unsigned char pattern3(ComponentType c, uint64_t slot) {
    return (unsigned char)((uint64_t)c * 101 + slot * 37 + 11);
}
static void fill3(SlotStore& s, ComponentType c, uint64_t slot, unsigned char v) {
    std::memset(s.slot_addr(c, slot), v, s.slot_bytes(c));
}
static bool holds3(const SlotStore& s, ComponentType c, uint64_t slot, unsigned char v) {
    auto* p = (const unsigned char*)s.slot_addr(c, slot);
    for (uint64_t i = 0; i < s.slot_bytes(c); ++i)
        if (p[i] != v) return false;
    return true;
}

static void test_e2e_pools() {
    TEST_BEGIN("two processes, three pools: SWA run read, MAMBA slot written");
    if (!mooncake_available()) { TEST_SKIP("no Mooncake engine"); return; }

    int name_pipe[2], done_pipe[2];
    assert(::pipe(name_pipe) == 0 && ::pipe(done_pipe) == 0);
    const std::string shm_a = "/test_ste_p_a", shm_b = "/test_ste_p_b";
    SlotStore::destroy(shm_a, "");
    SlotStore::destroy(shm_b, "");
    const ComponentType F = ComponentType::FULL, W = ComponentType::SWA, M = ComponentType::MAMBA;

    pid_t pa = spawn_child([&] {
        ::close(name_pipe[0]);
        ::close(done_pipe[1]);
        SlotStore s = SlotStore::create(store_cfg3(shm_a));
        auto be = create_transfer_backend(mc_cfg(true, "rdma"));
        if (!be) _exit(2);
        SlotTransfer xfer(s, *be);
        if (xfer.register_store("cpu:0") != XFER_OK) _exit(3);
        for (uint64_t i = 0; i < s.num_slots(F); ++i) fill3(s, F, i, pattern3(F, i));
        for (uint64_t i = 0; i < s.num_slots(W); ++i) fill3(s, W, i, pattern3(W, i));
        for (uint64_t i = 0; i < s.num_slots(M); ++i) fill3(s, M, i, 0x00);
        write_str(name_pipe[1], xfer.local_segment());
        char sig = 0;
        if (::read(done_pipe[0], &sig, 1) != 1 || sig != 'k') _exit(4);
        // MAMBA slot 2 was written; everything else untouched.
        if (!holds3(s, M, 2, pattern3(M, 1))) _exit(5);
        for (uint64_t i = 0; i < s.num_slots(M); ++i)
            if (i != 2 && !holds3(s, M, i, 0x00)) _exit(6);
        for (uint64_t i = 0; i < s.num_slots(F); ++i)
            if (!holds3(s, F, i, pattern3(F, i))) _exit(7);
        for (uint64_t i = 0; i < s.num_slots(W); ++i)
            if (!holds3(s, W, i, pattern3(W, i))) _exit(8);
        _exit(0);
    });

    pid_t pb = spawn_child([&] {
        ::close(name_pipe[1]);
        ::close(done_pipe[0]);
        SlotStore s = SlotStore::create(store_cfg3(shm_b));
        auto be = create_transfer_backend(mc_cfg(false, "rdma"));
        if (!be) _exit(2);
        SlotTransfer xfer(s, *be);
        if (xfer.register_store("cpu:0") != XFER_OK) _exit(3);
        std::string peer;
        if (!read_str(name_pipe[0], peer) || peer.empty()) _exit(4);

        // Identical configs: our layout stands in for the registry entry.
        const StoreGeometry layout = s.layout();
        StoreGeometry geo;
        if (xfer.attach_peer(peer, &layout, &geo) != XFER_OK) _exit(5);
        if (geo[W].num_slots != 32 || geo[M].slot_bytes != 8192) _exit(6);

        // SWA: peer slots 8..15 -> our SWA slots 0..7, one request.
        const int n = 8;
        std::vector<int32_t> rl(n), rr(n);
        for (int i = 0; i < n; ++i) { rl[i] = i; rr[i] = 8 + i; fill3(s, W, (uint64_t)i, 0x00); }
        fill3(s, F, 0, 0xEE);                          // FULL slot 0 must stay
        int ndesc = 0;
        BatchHandle h = xfer.read(peer, W, rr.data(), rl.data(), n, &ndesc);
        if (h == INVALID_BATCH || ndesc != 1) _exit(7);
        if (xfer.wait(h, 30000) != XferState::COMPLETED) _exit(8);
        uint64_t done = 0;
        xfer.poll(h, &done);
        xfer.release(h);
        if (done != (uint64_t)n * 4096) _exit(9);
        for (int i = 0; i < n; ++i)
            if (!holds3(s, W, (uint64_t)rl[i], pattern3(W, (uint64_t)rr[i]))) _exit(10);
        if (!holds3(s, F, 0, 0xEE)) _exit(11);

        // MAMBA: our slot 1 -> peer slot 2.
        fill3(s, M, 1, pattern3(M, 1));
        const int32_t ml = 1, mr = 2;
        BatchHandle h2 = xfer.write(peer, M, &ml, &mr, 1, &ndesc);
        if (h2 == INVALID_BATCH || ndesc != 1) _exit(12);
        if (xfer.wait(h2, 30000) != XferState::COMPLETED) _exit(13);
        xfer.release(h2);

        char sig = 'k';
        if (::write(done_pipe[1], &sig, 1) != 1) _exit(14);
        _exit(0);
    });

    ::close(name_pipe[0]); ::close(name_pipe[1]);
    ::close(done_pipe[0]); ::close(done_pipe[1]);
    const bool ok_b = reap_step(pb, "initiator");
    const bool ok_a = reap_step(pa, "publisher");
    SlotStore::destroy(shm_a, "");
    SlotStore::destroy(shm_b, "");
    assert(ok_a && ok_b);

    TEST_PASS();
}

// ==================================================================
int main() {
    printf("=== SlotTransfer end-to-end (two processes, real engine) ===\n");
    if (!mooncake_available())
        printf("  (no usable Mooncake engine on this host — tests will SKIP)\n");
    test_e2e_rdma();
    test_e2e_tcp();
    test_roundtrip_all_slots();
    test_e2e_pools();
    return tests_report();
}
