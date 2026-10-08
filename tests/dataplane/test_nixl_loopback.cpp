// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// test_nixl_loopback — NixlBackend against a real nixlAgent: the self-segment
// memcpy path, two processes moving bytes both ways over UCX while the
// release() contract is checked (XFER_EBUSY, handle kept, while in flight),
// and the metadata exchange surviving a peer that is not listening yet and a
// close/reopen. Without a working UCX agent the cases report SKIP.

#include "data_test_utils.h"
#include "slot_store.h"
#include "transfer.h"

#include <chrono>
#include <cstring>
#include <string>
#include <thread>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

using namespace radixshmem;

static SlotStoreConfig store_cfg(const std::string& name, uint64_t slots,
                                 uint64_t slot_bytes) {
    SlotStoreConfig c;
    c.name = name;
    c.full.num_slots = slots;
    c.full.slot_bytes = slot_bytes;
    c.slot_align = 4096;
    c.hugepage_path = "";
    c.prefault = true;
    return c;
}

static TransferConfig nixl_cfg(uint64_t port) {
    TransferConfig c;
    c.backend = "nixl";
    c.ip_or_host = "127.0.0.1";
    c.rpc_port = port;
    return c;
}

static int reserve_port(uint64_t& port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    assert(fd >= 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    assert(::bind(fd, (sockaddr*)&a, sizeof(a)) == 0);
    socklen_t len = sizeof(a);
    assert(::getsockname(fd, (sockaddr*)&a, &len) == 0);
    port = ntohs(a.sin_port);
    return fd;
}

static uint64_t free_port() {
    uint64_t port = 0;
    ::close(reserve_port(port));
    return port;
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

static bool nixl_available() {
    static int cached = -1;
    if (cached >= 0) return cached == 1;
    cached = child_succeeds([] {
        auto be = create_transfer_backend(nixl_cfg(free_port()));
        _exit(be ? 0 : 1);
    }) ? 1 : 0;
    return cached == 1;
}

// ==================================================================
static void test_self_segment() {
    TEST_BEGIN("self segment: same-process transfer short-circuits to memcpy");
    if (!nixl_available()) { TEST_SKIP("no NIXL agent"); return; }

    run_child([] {
        SlotStore::destroy("/test_nixl_self", "");
        SlotStore s = SlotStore::create(store_cfg("/test_nixl_self", 16, 4096));
        const uint64_t port = free_port();
        auto be = create_transfer_backend(nixl_cfg(port));
        assert(be);
        assert(std::string(be->backend_name()) == "nixl");
        assert(be->local_segment() == "127.0.0.1:" + std::to_string(port));
        assert(be->register_memory(s.data_base(), s.data_bytes(), "") == XFER_OK);

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
        assert(be->poll(h, &done) == XferState::COMPLETED && done == sb);
        assert(be->release(h) == XFER_OK);
        assert(be->poll(h, nullptr) == XferState::INVALID);
        assert(be->release(h) == XFER_ENOENT);

        auto* p = (unsigned char*)s.slot_addr(11);
        for (uint64_t i = 0; i < sb; ++i) assert(p[i] == 0x5C);

        be->close_segment(self.id);
        assert(be->submit(&d, 1) == INVALID_BATCH);
        SegmentInfo again;
        assert(be->open_segment(be->local_segment(), again) == XFER_OK);
        assert(again.base == self.base && again.length == self.length);
    });

    TEST_PASS();
}

// ==================================================================
// Publisher A holds a pattern; initiator B READs 60 slots of it in one batch,
// calling release() from the moment of submit: every EBUSY must leave the
// handle alive, and the bytes must all be there once release() says OK
// (checked tail first, so a release that returned early is caught).
// B then WRITEs one slot back and A checks it, plus an untouched neighbour.
static void test_loopback_release_contract() {
    TEST_BEGIN("loopback: bytes both ways; release() is EBUSY while in flight");
    if (!nixl_available()) { TEST_SKIP("no NIXL agent"); return; }

    int name_pipe[2], done_pipe[2];
    assert(::pipe(name_pipe) == 0 && ::pipe(done_pipe) == 0);

    const std::string store_a = "/test_nixl_lb_a";
    const std::string store_b = "/test_nixl_lb_b";
    SlotStore::destroy(store_a, "");
    SlotStore::destroy(store_b, "");
    const uint64_t slots = 64, sb = 1u << 20;

    pid_t pa = spawn_child([&] {
        ::close(name_pipe[0]);
        ::close(done_pipe[1]);
        SlotStore s = SlotStore::create(store_cfg(store_a, slots, sb));
        auto be = create_transfer_backend(nixl_cfg(free_port()));
        if (!be) _exit(2);
        if (be->register_memory(s.data_base(), s.data_bytes(), "") != XFER_OK) _exit(3);
        for (uint64_t i = 0; i < slots; ++i) std::memset(s.slot_addr(i), (int)(0x10 + i), sb);

        write_str(name_pipe[1], be->local_segment());
        ::close(name_pipe[1]);

        std::string done;
        if (!read_str(done_pipe[0], done) || done != "done") _exit(4);
        auto* w = (unsigned char*)s.slot_addr(9);
        for (uint64_t i = 0; i < sb; ++i) if (w[i] != 0xC3) _exit(5);
        auto* nb = (unsigned char*)s.slot_addr(10);
        for (uint64_t i = 0; i < sb; ++i) if (nb[i] != 0x1A) _exit(6);
    });

    pid_t pb = spawn_child([&] {
        ::close(name_pipe[1]);
        ::close(done_pipe[0]);
        SlotStore s = SlotStore::create(store_cfg(store_b, slots, sb));
        auto be = create_transfer_backend(nixl_cfg(free_port()));
        if (!be) _exit(2);
        if (be->register_memory(s.data_base(), s.data_bytes(), "") != XFER_OK) _exit(3);
        std::memset(s.data_base(), 0x00, s.data_bytes());

        std::string peer;
        if (!read_str(name_pipe[0], peer)) _exit(4);
        SegmentInfo seg;
        if (be->open_segment(peer, seg) != XFER_OK) _exit(5);
        if (seg.length != slots * sb) _exit(6);

        TransferDesc d;
        d.op = XferOp::READ;
        d.local_addr = s.slot_addr(0);
        d.target = seg.id;
        d.target_offset = seg.base;
        d.length = 60 * sb;
        BatchHandle h = be->submit(&d, 1);
        if (h == INVALID_BATCH) _exit(7);

        bool saw_busy = false;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        for (;;) {
            const int rc = be->release(h);
            if (rc == XFER_EBUSY) {
                if (be->poll(h, nullptr) == XferState::INVALID) _exit(8);
                if (std::chrono::steady_clock::now() > deadline) _exit(9);
                saw_busy = true;
                continue;
            }
            if (rc != XFER_OK) _exit(10);
            if (be->poll(h, nullptr) != XferState::INVALID) _exit(11);
            break;
        }
        for (uint64_t i = 60; i-- > 0;) {
            auto* p = (unsigned char*)s.slot_addr(i);
            for (uint64_t j = 0; j < sb; ++j) if (p[j] != (unsigned char)(0x10 + i)) _exit(12);
        }
        auto* untouched = (unsigned char*)s.slot_addr(60);
        for (uint64_t j = 0; j < sb; ++j) if (untouched[j] != 0x00) _exit(13);

        std::memset(s.slot_addr(5), 0xC3, sb);
        d.op = XferOp::WRITE;
        d.local_addr = s.slot_addr(5);
        d.target_offset = seg.base + 9 * sb;
        d.length = sb;
        BatchHandle h2 = be->submit(&d, 1);
        if (h2 == INVALID_BATCH) _exit(14);
        if (be->wait(h2, 30000) != XferState::COMPLETED) _exit(15);
        if (be->release(h2) != XFER_OK) _exit(16);
        be->close_segment(seg.id);

        write_str(done_pipe[1], "done");
        printf(saw_busy ? "(EBUSY seen) " : "(no EBUSY: batch finished before first release) ");
        fflush(stdout);
    });

    ::close(name_pipe[0]);
    ::close(name_pipe[1]);
    ::close(done_pipe[0]);
    ::close(done_pipe[1]);
    const bool ok_b = reap_child(pb);
    const bool ok_a = reap_child(pa);
    assert(ok_b);
    assert(ok_a);
    SlotStore::destroy(store_a, "");
    SlotStore::destroy(store_b, "");

    TEST_PASS();
}

// ==================================================================
// A starts listening 1.5 s after B began open_segment(), so B's first
// metadata exchange hits a closed port. B then closes and reopens the
// segment. Both opens must end in a working READ.
static void test_exchange_retry_and_reopen() {
    TEST_BEGIN("open_segment retries the exchange; close/reopen refetches");
    if (!nixl_available()) { TEST_SKIP("no NIXL agent"); return; }

    int name_pipe[2], done_pipe[2];
    assert(::pipe(name_pipe) == 0 && ::pipe(done_pipe) == 0);

    const std::string store_a = "/test_nixl_rt_a";
    const std::string store_b = "/test_nixl_rt_b";
    SlotStore::destroy(store_a, "");
    SlotStore::destroy(store_b, "");
    const uint64_t slots = 16, sb = 65536;

    pid_t pa = spawn_child([&] {
        ::close(name_pipe[0]);
        ::close(done_pipe[1]);
        uint64_t port = 0;
        const int hold = reserve_port(port);
        write_str(name_pipe[1], "127.0.0.1:" + std::to_string(port));
        ::close(name_pipe[1]);
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));
        ::close(hold);

        SlotStore s = SlotStore::create(store_cfg(store_a, slots, sb));
        auto be = create_transfer_backend(nixl_cfg(port));
        if (!be) _exit(2);
        if (be->register_memory(s.data_base(), s.data_bytes(), "") != XFER_OK) _exit(3);
        for (uint64_t i = 0; i < slots; ++i) std::memset(s.slot_addr(i), (int)(0x40 + i), sb);

        std::string done;
        if (!read_str(done_pipe[0], done) || done != "done") _exit(4);
    });

    pid_t pb = spawn_child([&] {
        ::close(name_pipe[1]);
        ::close(done_pipe[0]);
        SlotStore s = SlotStore::create(store_cfg(store_b, slots, sb));
        auto be = create_transfer_backend(nixl_cfg(free_port()));
        if (!be) _exit(2);
        if (be->register_memory(s.data_base(), s.data_bytes(), "") != XFER_OK) _exit(3);
        std::string peer;
        if (!read_str(name_pipe[0], peer)) _exit(4);

        auto pull = [&](uint64_t slot, int code, bool release) {
            SegmentInfo seg;
            if (be->open_segment(peer, seg) != XFER_OK) _exit(code);
            std::memset(s.slot_addr(slot), 0x00, sb);
            TransferDesc d;
            d.op = XferOp::READ;
            d.local_addr = s.slot_addr(slot);
            d.target = seg.id;
            d.target_offset = seg.base + slot * sb;
            d.length = sb;
            BatchHandle h = be->submit(&d, 1);
            if (h == INVALID_BATCH) _exit(code + 1);
            if (be->wait(h, 30000) != XferState::COMPLETED) _exit(code + 2);
            if (release && be->release(h) != XFER_OK) _exit(code + 3);
            auto* p = (unsigned char*)s.slot_addr(slot);
            for (uint64_t j = 0; j < sb; ++j) if (p[j] != (unsigned char)(0x40 + slot)) _exit(code + 4);
            be->close_segment(seg.id);
        };
        pull(3, 10, true);
        pull(4, 20, false);

        write_str(done_pipe[1], "done");
    });

    ::close(name_pipe[0]);
    ::close(name_pipe[1]);
    ::close(done_pipe[0]);
    ::close(done_pipe[1]);
    const bool ok_b = reap_child(pb);
    const bool ok_a = reap_child(pa);
    assert(ok_b);
    assert(ok_a);
    SlotStore::destroy(store_a, "");
    SlotStore::destroy(store_b, "");

    TEST_PASS();
}

int main() {
    printf("test_nixl_loopback\n");
    test_self_segment();
    test_loopback_release_contract();
    test_exchange_retry_and_reopen();
    return tests_report();
}
