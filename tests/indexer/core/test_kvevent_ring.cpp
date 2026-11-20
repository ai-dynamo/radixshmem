#include "kvevent.h"
#include <atomic>
#include <cassert>
#include <cstdio>
#include <thread>
#include <unordered_set>
#include <vector>
using namespace radixshmem;

// Removed hashes of a drain, in order.
static std::vector<uint64_t> removed(const KvEventRingReader::DrainResult& d) {
    std::vector<uint64_t> out;
    for (const auto& r : d.records)
        if (r.kind == KVEVENT_REMOVED) out.insert(out.end(), r.hashes.begin(), r.hashes.end());
    return out;
}

static std::vector<char> make_region(uint32_t cap) {
    std::vector<char> buf(kvevent_ring_bytes(cap));
    KvEventRingWriter::init(buf.data(), cap);
    return buf;
}

int main() {
    // capacity normalization
    assert(kvevent_ring_normalize_capacity(0) == 0);
    assert(kvevent_ring_normalize_capacity(1) == 64);
    assert(kvevent_ring_normalize_capacity(64) == 64);
    assert(kvevent_ring_normalize_capacity(65) == 128);

    // 1) basic round-trip, multi-entry split
    {
        auto buf = make_region(64);
        KvEventRingWriter w(buf.data());
        KvEventRingReader r(buf.data());
        std::vector<uint64_t> in;
        for (uint64_t i = 0; i < 100; i++) in.push_back(i + 1);
        w.publish(KVEVENT_REMOVED, in.data(), (uint32_t)in.size());
        auto out = r.drain();
        assert(!out.overflow);
        assert(removed(out) == in);
        // drained: nothing left
        auto out2 = r.drain();
        assert(removed(out2).empty() && !out2.overflow);
        printf("ok: round-trip 100 hashes across %u-entry ring\n", 64u);
    }

    // 2) seek_to_head skips history
    {
        auto buf = make_region(64);
        KvEventRingWriter w(buf.data());
        std::vector<uint64_t> in{1,2,3};
        w.publish(KVEVENT_REMOVED, in.data(), 3);
        KvEventRingReader r(buf.data());   // ctor seeks to head
        assert(removed(r.drain()).empty());
        w.publish(KVEVENT_REMOVED, in.data(), 3);
        assert(removed(r.drain()).size() == 3);
        printf("ok: late attach skips history\n");
    }

    // 3) overflow: producer laps the consumer
    {
        auto buf = make_region(64);
        KvEventRingWriter w(buf.data());
        KvEventRingReader r(buf.data());
        std::vector<uint64_t> one{42};
        for (int i = 0; i < 200; i++) w.publish(KVEVENT_REMOVED, one.data(), 1);  // 200 entries > 64
        auto out = r.drain();
        assert(out.overflow);
        assert(removed(out).size() == 64);   // only the surviving lap
        printf("ok: lapped consumer reports overflow, keeps %zu\n", removed(out).size());
    }

    // 4) max_hashes bounds the batch, remainder comes next
    {
        auto buf = make_region(64);
        KvEventRingWriter w(buf.data());
        KvEventRingReader r(buf.data());
        std::vector<uint64_t> in;
        for (uint64_t i = 0; i < 140; i++) in.push_back(i);
        w.publish(KVEVENT_REMOVED, in.data(), 140);   // 10 entries
        auto a = r.drain(20);
        assert(!a.overflow && removed(a).size() >= 20 && removed(a).size() <= 28);
        auto b = r.drain();
        assert(!b.overflow);
        std::vector<uint64_t> all = removed(a);
        const auto removed_b = removed(b);
        all.insert(all.end(), removed_b.begin(), removed_b.end());
        assert(all == in);
        printf("ok: max_hashes split %zu + %zu == 140\n", removed(a).size(), removed(b).size());
    }

    // 5) disabled ring is a total no-op
    {
        assert(kvevent_ring_bytes(0) == 0);
        KvEventRingWriter w(nullptr);
        assert(!w.enabled());
        w.publish(KVEVENT_REMOVED, nullptr, 0);
        KvEventRingReader r(nullptr);
        assert(!r.enabled());
        assert(removed(r.drain()).empty());
        printf("ok: disabled ring is inert\n");
    }

    // 6) concurrent producers: no slot collisions, no lost/duplicated values
    {
        const uint32_t cap = 1 << 16;
        auto buf = make_region(cap);
        KvEventRingReader r(buf.data());
        const int kThreads = 8, kPerThread = 500;
        std::vector<std::thread> ts;
        for (int t = 0; t < kThreads; t++) {
            ts.emplace_back([&, t] {
                KvEventRingWriter w(buf.data());
                for (int i = 0; i < kPerThread; i++) {
                    uint64_t h[7];
                    for (int k = 0; k < 7; k++)
                        h[k] = (uint64_t)t * 1000000 + (uint64_t)i * 10 + k;
                    w.publish(KVEVENT_REMOVED, h, 7);
                }
            });
        }
        for (auto& th : ts) th.join();
        std::unordered_set<uint64_t> seen;
        size_t total = 0;
        for (;;) {
            auto out = r.drain();
            assert(!out.overflow);
            if (removed(out).empty()) break;
            for (uint64_t h : removed(out)) {
                assert(seen.insert(h).second && "duplicate hash -> slot collision");
                total++;
            }
        }
        assert(total == (size_t)kThreads * kPerThread * 7);
        printf("ok: %d concurrent producers, %zu distinct hashes, no collision\n",
               kThreads, total);
    }

    // 7) reader survives a producer restart (region recreated under it)
    {
        auto buf = make_region(64);
        KvEventRingWriter w(buf.data());
        std::vector<uint64_t> in{7,8,9};
        w.publish(KVEVENT_REMOVED, in.data(), 3);
        KvEventRingReader r(buf.data());
        w.publish(KVEVENT_REMOVED, in.data(), 3);
        assert(removed(r.drain()).size() == 3);
        KvEventRingWriter::init(buf.data(), 64);   // restart: head resets to 0
        auto out = r.drain();
        assert(out.overflow);
        printf("ok: producer restart flagged as overflow\n");
    }

    // 8) kinds survive the ring: adjacent same-kind entries merge into one
    //    record, a kind change or a cleared record starts a new one, and only
    //    removed hashes land in the eviction-only view
    {
        auto buf = make_region(64);
        KvEventRingWriter w(buf.data());
        KvEventRingReader r(buf.data());
        std::vector<uint64_t> s1(20), s2{100, 101}, rm{5, 6, 7};
        for (int i = 0; i < 20; i++) s1[i] = 1000 + i;   // 2 entries
        w.publish(KVEVENT_STORED, s1.data(), 20);
        w.publish(KVEVENT_STORED, s2.data(), 2);
        w.publish(KVEVENT_REMOVED, rm.data(), 3);
        w.publish(KVEVENT_CLEARED, nullptr, 0);
        w.publish(KVEVENT_CLEARED, rm.data(), 3);         // hashes ignored on cleared
        w.publish(KVEVENT_STORED, s2.data(), 2);
        auto out = r.drain();
        assert(!out.overflow);
        assert(out.records.size() == 5);
        assert(out.records[0].kind == KVEVENT_STORED && out.records[0].hashes.size() == 22);
        assert(out.records[0].hashes[20] == 100);
        assert(out.records[1].kind == KVEVENT_REMOVED && out.records[1].hashes == rm);
        assert(out.records[2].kind == KVEVENT_CLEARED && out.records[2].hashes.empty());
        assert(out.records[3].kind == KVEVENT_CLEARED && out.records[3].hashes.empty());
        assert(out.records[4].kind == KVEVENT_STORED && out.records[4].hashes == s2);
        assert(removed(out) == rm);
        printf("ok: stored/removed/cleared kinds round-trip\n");
    }

    printf("ALL RING TESTS PASSED\n");
    return 0;
}
