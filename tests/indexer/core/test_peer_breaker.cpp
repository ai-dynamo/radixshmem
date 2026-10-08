// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// test_peer_breaker — the per-peer circuit breaker of the one-sided index
// planes: a failure skips the peer for a backoff that doubles up to a cap,
// then lets one probe through; a success closes it; skips are counted in the
// shared counters. RDMA builds only (the breaker lives in the peer plane).

#include "test_utils.h"

#ifndef RADIXSHMEM_NO_RDMA
#include "peer_breaker.hpp"

#include <cassert>
#include <chrono>
#include <thread>

using namespace radixshmem;
using Gate = PeerBreaker::Gate;

static void sleep_ms(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

static void test_closed_by_default() {
    TEST_BEGIN("healthy peers pass; success on a healthy peer logs nothing");
    PeerFaultCounters c;
    c.clear();
    PeerBreaker b("[test]", &c, 50, 200);
    assert(b.gate(1) == Gate::PASS);
    b.succeeded(1);
    assert(!b.is_down(1) && c.down_events.load() == 0);
    TEST_PASS();
}

static void test_skip_probe_backoff() {
    TEST_BEGIN("failure skips until the backoff, one probe, doubling to the cap");
    PeerFaultCounters c;
    c.clear();
    PeerBreaker b("[test]", &c, 100, 250);
    b.name(1, "10.0.0.1");
    b.failed(1, "RHT flush", "WC error status=12", 0.5);
    assert(b.is_down(1) && c.down_events.load() == 1);
    assert(b.gate(1) == Gate::SKIP);
    assert(b.gate(2) == Gate::PASS);              // other peers unaffected
    sleep_ms(120);
    assert(b.gate(1) == Gate::PROBE);
    assert(b.gate(1) == Gate::SKIP);              // one probe at a time
    b.failed(1, "RHT read", "WC error status=12", 0.5);   // backoff 200 ms
    sleep_ms(80);
    assert(b.gate(1) == Gate::SKIP);
    sleep_ms(150);
    assert(b.gate(1) == Gate::PROBE);
    b.failed(1, "RHT read", "WC error status=12", 0.5);   // capped at 250 ms
    sleep_ms(270);
    assert(b.gate(1) == Gate::PROBE);
    assert(c.down_events.load() == 1);            // one outage, one event
    b.succeeded(1);
    assert(!b.is_down(1) && b.gate(1) == Gate::PASS);
    TEST_PASS();
}

static void test_skips_counted() {
    TEST_BEGIN("skipped ops go to the shared counters by kind");
    PeerFaultCounters c;
    c.clear();
    PeerBreaker b("[test]", &c, 50, 200);
    b.failed(3, "remote walk", "WC error status=12", 0.5);
    b.skipped(3, PeerBreaker::Op::RHT_WRITE, 4);
    b.skipped(3, PeerBreaker::Op::RHT_REMOVE);
    b.skipped(3, PeerBreaker::Op::RHT_LOOKUP, 7);
    b.skipped(3, PeerBreaker::Op::WALK, 2);
    assert(c.rht_writes.load() == 4 && c.rht_removes.load() == 1);
    assert(c.rht_lookups.load() == 7 && c.walks.load() == 2);
    PeerBreaker quiet("[test]", nullptr);         // no shared counters is fine
    quiet.skipped(3, PeerBreaker::Op::WALK);
    TEST_PASS();
}

static void test_new_outage_after_recovery() {
    TEST_BEGIN("a new failure after recovery starts over at the base backoff");
    PeerFaultCounters c;
    c.clear();
    PeerBreaker b("[test]", &c, 100, 1000);
    b.failed(1, "RHT flush", "x", 0.1);
    sleep_ms(120);
    assert(b.gate(1) == Gate::PROBE);
    b.failed(1, "RHT flush", "x", 0.1);           // backoff 200 ms
    sleep_ms(220);
    assert(b.gate(1) == Gate::PROBE);
    b.succeeded(1);
    b.failed(1, "RHT flush", "x", 0.1);           // back to 100 ms
    assert(c.down_events.load() == 2);
    sleep_ms(120);
    assert(b.gate(1) == Gate::PROBE);
    TEST_PASS();
}

int main() {
    printf("=== test_peer_breaker ===\n\n");
    test_closed_by_default();
    test_skip_probe_backoff();
    test_skips_counted();
    test_new_outage_after_recovery();
    return tests_report();
}

#else // RADIXSHMEM_NO_RDMA

int main() {
    printf("=== test_peer_breaker ===\n\n  (skipped: built without RDMA)\n");
    return 0;
}

#endif
