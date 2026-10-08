// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// test_mooncake_loopback — MooncakeBackend against a real TransferEngine.
//
//   - self segment: same-process transfers short-circuit to memcpy rather than
//     hand-shaking with themselves (which Mooncake cannot do anyway)
//   - validation: a malformed descriptor rejects the whole batch, and nothing
//     in it is issued
//   - loopback: two processes, one publishing and one initiator-only, moving
//     bytes both directions over rdma and over tcp
//   - device filter: the same loopback with both endpoints pinned to one named
//     HCA, which is how the data plane is kept off the management NIC
//   - scatter addressing: a multi-descriptor batch lands byte for byte where a
//     plain memcpy of the same slot pairs would put it
//
// Every engine lives inside a forked child. An ibverbs context does not survive
// fork(), so a parent that had built an engine could not safely fork children
// that build their own -- the parent here never touches one.
//
// Without RDMA hardware (or without Mooncake compiled in) the engine cannot come
// up; those cases report SKIP instead of failing, so the suite still runs on a
// plain host.

#include "data_test_utils.h"
#include "slot_store.h"
#include "transfer.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include <dirent.h>
#include <unistd.h>

using namespace radixshmem;

// The HCAs this host exposes to ibverbs, in sysfs order. Reading sysfs rather
// than calling ibv_get_device_list keeps the test off libibverbs directly --
// and it is safe before a fork, unlike opening a device context.
static std::vector<std::string> rdma_devices() {
    std::vector<std::string> out;
    DIR* d = ::opendir("/sys/class/infiniband");
    if (!d) return out;
    while (struct dirent* e = ::readdir(d)) {
        const std::string n = e->d_name;
        if (n != "." && n != "..") out.push_back(n);
    }
    ::closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

static SlotStoreConfig store_cfg(const std::string& name, uint64_t slots,
                                 uint64_t slot_bytes) {
    SlotStoreConfig c;
    c.name = name;
    c.full.num_slots = slots;
    c.full.slot_bytes = slot_bytes;
    c.slot_align = 4096;
    c.hugepage_path = "";
    c.prefault = true;      // an MR over unfaulted pages is a bad idea
    return c;
}

static TransferConfig mc_cfg(bool publish, const std::string& protocol,
                             uint64_t port,
                             const std::vector<std::string>& devices = {}) {
    TransferConfig c;
    c.backend  = "mooncake";
    c.metadata = "P2PHANDSHAKE";     // no metadata service to deploy
    c.ip_or_host = "127.0.0.1";
    c.rpc_port = port;
    c.protocol = protocol;
    c.device_filter = devices;
    c.publish  = publish;
    return c;
}

// The engine binds whatever port it can get, so tests must not assume one.
// Children hand their real segment name to their peer over a pipe.
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
        auto be = create_transfer_backend(mc_cfg(true, "rdma", 0));
        _exit(be ? 0 : 1);
    }) ? 1 : 0;
    return cached == 1;
}

// ==================================================================
static void test_self_segment() {
    TEST_BEGIN("self segment: same-process transfer short-circuits to memcpy");
    if (!mooncake_available()) { TEST_SKIP("no Mooncake engine"); return; }

    run_child([] {
        SlotStore::destroy("/test_mc_self", "");
        SlotStore s = SlotStore::create(store_cfg("/test_mc_self", 16, 4096));
        auto be = create_transfer_backend(mc_cfg(true, "rdma", 0));
        assert(be);
        assert(std::string(be->backend_name()) == "mooncake");
        assert(!be->local_segment().empty());
        assert(be->register_memory(s.data_base(), s.data_bytes(), "cpu:0") == XFER_OK);

        SegmentInfo self;
        assert(be->open_segment(be->local_segment(), self) == XFER_OK);
        assert(self.base == (uint64_t)s.data_base());
        assert(self.length == s.data_bytes());

        const uint64_t sb = s.slot_bytes();
        std::memset(s.slot_addr(2), 0x5C, sb);
        std::memset(s.slot_addr(11), 0x00, sb);

        TransferDesc d;
        d.op = XferOp::WRITE;
        d.local_addr = s.slot_addr(2);
        d.target = self.id;
        d.target_offset = self.base + 11 * sb;
        d.length = sb;

        BatchHandle h = be->submit(&d, 1);
        assert(h != INVALID_BATCH);
        uint64_t done = 0;
        assert(be->wait(h, 5000) == XferState::COMPLETED);
        assert(be->poll(h, &done) == XferState::COMPLETED && done == sb);
        assert(be->release(h) == XFER_OK);
        assert(be->poll(h, nullptr) == XferState::INVALID);

        auto* p = (unsigned char*)s.slot_addr(11);
        for (uint64_t i = 0; i < sb; ++i) assert(p[i] == 0x5C);

        // READ direction, and a neighbour left alone.
        std::memset(s.slot_addr(4), 0x00, sb);
        d.op = XferOp::READ;
        d.local_addr = s.slot_addr(4);
        d.target_offset = self.base + 11 * sb;
        BatchHandle h2 = be->submit(&d, 1);
        assert(h2 != INVALID_BATCH);
        assert(be->wait(h2, 5000) == XferState::COMPLETED);
        be->release(h2);
        auto* q = (unsigned char*)s.slot_addr(4);
        for (uint64_t i = 0; i < sb; ++i) assert(q[i] == 0x5C);
        auto* nb = (unsigned char*)s.slot_addr(5);
        for (uint64_t i = 0; i < sb; ++i) assert(nb[i] == 0x00);
    });

    TEST_PASS();
}

// ==================================================================
static void test_validation_parity() {
    TEST_BEGIN("a malformed descriptor rejects the batch, issuing none of it");
    if (!mooncake_available()) { TEST_SKIP("no Mooncake engine"); return; }

    run_child([] {
        SlotStore::destroy("/test_mc_val", "");
        SlotStore s = SlotStore::create(store_cfg("/test_mc_val", 16, 4096));
        auto be = create_transfer_backend(mc_cfg(true, "rdma", 0));
        assert(be);
        assert(be->register_memory(nullptr, 4096, "") == XFER_EINVAL);
        assert(be->register_memory(s.data_base(), 0, "") == XFER_EINVAL);
        assert(be->register_memory(s.data_base(), s.data_bytes(), "") == XFER_OK);
        assert(be->register_memory(s.data_base(), s.data_bytes(), "") == XFER_EINVAL);

        SegmentInfo self;
        assert(be->open_segment("", self) == XFER_EINVAL);
        assert(be->open_segment("127.0.0.1:1", self) == XFER_ENOENT);   // nobody there
        assert(be->open_segment(be->local_segment(), self) == XFER_OK);

        const uint64_t sb = s.slot_bytes();
        TransferDesc ok;
        ok.op = XferOp::WRITE;
        ok.local_addr = s.slot_addr(0);
        ok.target = self.id;
        ok.target_offset = self.base;
        ok.length = sb;

        assert(be->submit(nullptr, 1) == INVALID_BATCH);
        assert(be->submit(&ok, 0) == INVALID_BATCH);
        assert(be->submit(&ok, -1) == INVALID_BATCH);

        auto reject = [&](TransferDesc d) { assert(be->submit(&d, 1) == INVALID_BATCH); };
        { TransferDesc d = ok; d.length = 0;                      reject(d); }
        { TransferDesc d = ok; d.local_addr = nullptr;            reject(d); }
        { TransferDesc d = ok; d.target = INVALID_SEGMENT;        reject(d); }
        { TransferDesc d = ok; d.target = self.id + 77;           reject(d); }
        { TransferDesc d = ok; char stack[64]; d.local_addr = stack; d.length = 64; reject(d); }
        { TransferDesc d = ok; d.local_addr = s.slot_addr(15); d.length = sb * 2; reject(d); }
        { TransferDesc d = ok; d.target_offset = self.base + 15 * sb; d.length = sb * 2; reject(d); }
        { TransferDesc d = ok; d.target_offset = self.base - 8;   reject(d); }
        { TransferDesc d = ok; d.target_offset = UINT64_MAX - 8;  reject(d); }

        // A bad second descriptor must not apply the first.
        std::memset(s.slot_addr(0), 0xEE, sb);
        std::memset((char*)s.data_base(), 0x00, sb);
        TransferDesc pair[2] = {ok, ok};
        pair[1].target_offset = self.base + 15 * sb;
        pair[1].length = sb * 2;
        assert(be->submit(pair, 2) == INVALID_BATCH);
        auto* untouched = (unsigned char*)s.data_base();
        for (uint64_t i = 0; i < sb; ++i) assert(untouched[i] == 0x00);

        // Handle hygiene.
        assert(be->poll(INVALID_BATCH, nullptr) == XferState::INVALID);
        assert(be->wait(999999, 10) == XferState::INVALID);
        assert(be->release(INVALID_BATCH) == XFER_ENOENT);
        assert(be->release(4242) == XFER_ENOENT);

        // close_segment invalidates, reopen works.
        be->close_segment(self.id);
        assert(be->submit(&ok, 1) == INVALID_BATCH);
        be->close_segment(self.id);   // idempotent
        SegmentInfo again;
        assert(be->open_segment(be->local_segment(), again) == XFER_OK);
        assert(again.base == self.base && again.length == self.length);
    });

    TEST_PASS();
}

// ==================================================================
// Two processes, one publishing and one initiator-only (plan D3), moving bytes
// both directions. This is the shape of one node serving KV blocks to another.
static bool run_loopback(const std::string& protocol, const std::string& tag,
                         const std::vector<std::string>& devices = {}) {
    int name_pipe[2], done_pipe[2];
    assert(::pipe(name_pipe) == 0 && ::pipe(done_pipe) == 0);

    const std::string store_a = "/test_mc_" + tag + "_a";
    const std::string store_b = "/test_mc_" + tag + "_b";
    SlotStore::destroy(store_a, "");
    SlotStore::destroy(store_b, "");

    // Publisher: owns the data, waits for the initiator, then checks what it got.
    pid_t pa = spawn_child([&] {
        ::close(name_pipe[0]);
        ::close(done_pipe[1]);
        SlotStore s = SlotStore::create(store_cfg(store_a, 32, 65536));
        auto be = create_transfer_backend(mc_cfg(true, protocol, 0, devices));
        if (!be) _exit(2);
        if (be->register_memory(s.data_base(), s.data_bytes(), "cpu:0") != XFER_OK) _exit(3);

        std::memset(s.slot_addr(3), 0xA5, s.slot_bytes());   // for the peer to READ
        std::memset(s.slot_addr(9), 0x00, s.slot_bytes());   // for the peer to WRITE
        std::memset(s.slot_addr(10), 0x00, s.slot_bytes());  // must stay untouched

        write_str(name_pipe[1], be->local_segment());

        char sig = 0;
        if (::read(done_pipe[0], &sig, 1) != 1 || sig != 'k') _exit(4);

        auto* w = (unsigned char*)s.slot_addr(9);
        for (uint64_t i = 0; i < s.slot_bytes(); ++i) if (w[i] != 0xCC) _exit(5);
        auto* nb = (unsigned char*)s.slot_addr(10);
        for (uint64_t i = 0; i < s.slot_bytes(); ++i) if (nb[i] != 0x00) _exit(6);
        _exit(0);
    });

    // Initiator: publishes nothing, reaches into the peer both ways.
    pid_t pb = spawn_child([&] {
        ::close(name_pipe[1]);
        ::close(done_pipe[0]);
        SlotStore s = SlotStore::create(store_cfg(store_b, 32, 65536));
        auto be = create_transfer_backend(mc_cfg(false, protocol, 0, devices));
        if (!be) _exit(2);
        if (be->register_memory(s.data_base(), s.data_bytes(), "cpu:0") != XFER_OK) _exit(3);

        std::string peer;
        if (!read_str(name_pipe[0], peer) || peer.empty()) _exit(4);

        SegmentInfo seg;
        if (be->open_segment(peer, seg) != XFER_OK) _exit(5);
        const uint64_t sb = s.slot_bytes();
        if (seg.length != 32 * sb) _exit(6);      // geometry must agree

        // READ the peer's slot 3 into our slot 0.
        std::memset(s.slot_addr(0), 0x00, sb);
        TransferDesc r;
        r.op = XferOp::READ;
        r.local_addr = s.slot_addr(0);
        r.target = seg.id;
        r.target_offset = seg.base + 3 * sb;
        r.length = sb;
        BatchHandle h = be->submit(&r, 1);
        if (h == INVALID_BATCH) _exit(7);
        if (be->wait(h, 20000) != XferState::COMPLETED) _exit(8);
        uint64_t done = 0;
        be->poll(h, &done);
        // A batch mooncake just reported COMPLETED must be freeable outright.
        if (be->release(h) != XFER_OK) _exit(14);
        if (done != sb) _exit(9);
        auto* got = (unsigned char*)s.slot_addr(0);
        for (uint64_t i = 0; i < sb; ++i) if (got[i] != 0xA5) _exit(10);

        // WRITE our slot 1 into the peer's slot 9.
        std::memset(s.slot_addr(1), 0xCC, sb);
        TransferDesc w;
        w.op = XferOp::WRITE;
        w.local_addr = s.slot_addr(1);
        w.target = seg.id;
        w.target_offset = seg.base + 9 * sb;
        w.length = sb;
        BatchHandle h2 = be->submit(&w, 1);
        if (h2 == INVALID_BATCH) _exit(11);
        if (be->wait(h2, 20000) != XferState::COMPLETED) _exit(12);
        be->release(h2);

        char sig = 'k';
        if (::write(done_pipe[1], &sig, 1) != 1) _exit(13);
        _exit(0);
    });

    ::close(name_pipe[0]); ::close(name_pipe[1]);
    ::close(done_pipe[0]); ::close(done_pipe[1]);

    const bool ok_b = reap_child(pb);
    const bool ok_a = reap_child(pa);
    SlotStore::destroy(store_a, "");
    SlotStore::destroy(store_b, "");
    return ok_a && ok_b;
}

static void test_loopback_rdma() {
    TEST_BEGIN("loopback over rdma: publisher + initiator, READ and WRITE verified");
    if (!mooncake_available()) { TEST_SKIP("no Mooncake engine"); return; }
    assert(run_loopback("rdma", "rdma"));
    TEST_PASS();
}

static void test_loopback_tcp() {
    TEST_BEGIN("loopback over tcp: same transfers, no RDMA involved");
    if (!mooncake_available()) { TEST_SKIP("no Mooncake engine"); return; }
    assert(run_loopback("tcp", "tcp"));
    TEST_PASS();
}

// ==================================================================
// device_filter is what keeps the data plane on the east-west fabric: without
// it the engine may pick the management NIC on one end and a fabric HCA on the
// other, and that pair never reaches RTR. Pinning both endpoints to one named
// HCA is the deterministic case -- they cannot land on different planes.
static void test_device_filter() {
    TEST_BEGIN("device_filter pins both endpoints to one named HCA");
    if (!mooncake_available()) { TEST_SKIP("no Mooncake engine"); return; }
    const std::vector<std::string> devs = rdma_devices();
    if (devs.empty()) { TEST_SKIP("no RDMA device in sysfs"); return; }

    assert(run_loopback("rdma", "dev1", {devs[0]}));

    // A list, and one entry that does not exist: the filter is an allowlist
    // intersected with what the host has, not a list of requirements, so the
    // real device survives and the transfers still land.
    assert(run_loopback("rdma", "dev2", {devs[0], "mlx5_no_such_device"}));

    TEST_PASS();
}

// ==================================================================
static void test_scatter_addressing() {
    TEST_BEGIN("a scattered multi-descriptor batch lands exactly where memcpy would");
    if (!mooncake_available()) { TEST_SKIP("no Mooncake engine"); return; }

    // The reference is a memcpy model of the same slot pairs, computed here.
    // A divergence means the backend mis-addresses a slot — the failure mode a
    // single-descriptor test cannot see, because one wrong base still round
    // trips against itself.
    const int32_t src_slots[] = {0, 5, 6, 7, 31};
    const int32_t dst_slots[] = {20, 21, 22, 8, 19};
    const int n = 5;

    assert(child_succeeds([&] {
        const char* shm = "/test_mc_scatter";
        SlotStore::destroy(shm, "");
        SlotStore s = SlotStore::create(store_cfg(shm, 32, 4096));
        auto be = create_transfer_backend(mc_cfg(true, "rdma", 0));
        assert(be);
        assert(be->register_memory(s.data_base(), s.data_bytes(), "cpu:0") == XFER_OK);
        SegmentInfo self;
        assert(be->open_segment(be->local_segment(), self) == XFER_OK);

        const uint64_t sb = s.slot_bytes();
        for (uint64_t i = 0; i < s.num_slots(); ++i)
            std::memset(s.slot_addr(i), (int)(i * 7 + 1) & 0xff, sb);

        // What the store must hold afterwards, by definition of the batch.
        std::vector<unsigned char> want(s.data_bytes());
        std::memcpy(want.data(), s.data_base(), want.size());
        for (int i = 0; i < n; ++i)
            std::memcpy(want.data() + (uint64_t)dst_slots[i] * sb,
                        want.data() + (uint64_t)src_slots[i] * sb, sb);

        std::vector<TransferDesc> descs;
        for (int i = 0; i < n; ++i) {
            TransferDesc d;
            d.op = XferOp::WRITE;
            d.local_addr = s.slot_addr(src_slots[i]);
            d.target = self.id;
            d.target_offset = self.base + dst_slots[i] * sb;
            d.length = sb;
            descs.push_back(d);
        }
        BatchHandle h = be->submit(descs.data(), n);
        assert(h != INVALID_BATCH);
        assert(be->wait(h, 20000) == XferState::COMPLETED);
        be->release(h);

        const bool same = std::memcmp(want.data(), s.data_base(), want.size()) == 0;
        SlotStore::destroy(shm, "");
        _exit(same ? 0 : 1);
    }));

    TEST_PASS();
}

// ==================================================================
int main() {
    printf("=== Mooncake transfer backend tests ===\n");
    if (!mooncake_available())
        printf("  (no usable Mooncake engine on this host — engine tests will SKIP)\n");
    test_self_segment();
    test_validation_parity();
    test_loopback_rdma();
    test_loopback_tcp();
    test_device_filter();
    test_scatter_addressing();
    return tests_report();
}
