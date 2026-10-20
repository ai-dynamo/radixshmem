#include "kvevent.h"
#include <atomic>
#include <cassert>
#include <cstdio>
#include <thread>
#include <unordered_set>
#include <vector>
using namespace radixshmem;

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
        w.publish(in.data(), (uint32_t)in.size());
        auto out = r.drain();
        assert(!out.overflow);
        assert(out.hashes == in);
        // drained: nothing left
        auto out2 = r.drain();
        assert(out2.hashes.empty() && !out2.overflow);
        printf("ok: round-trip 100 hashes across %u-entry ring\n", 64u);
    }

    // 2) seek_to_head skips history
    {
        auto buf = make_region(64);
        KvEventRingWriter w(buf.data());
        std::vector<uint64_t> in{1,2,3};
        w.publish(in.data(), 3);
        KvEventRingReader r(buf.data());   // ctor seeks to head
        assert(r.drain().hashes.empty());
        w.publish(in.data(), 3);
        assert(r.drain().hashes.size() == 3);
        printf("ok: late attach skips history\n");
    }

    // 3) overflow: producer laps the consumer
    {
        auto buf = make_region(64);
        KvEventRingWriter w(buf.data());
        KvEventRingReader r(buf.data());
        std::vector<uint64_t> one{42};
        for (int i = 0; i < 200; i++) w.publish(one.data(), 1);  // 200 entries > 64
        auto out = r.drain();
        assert(out.overflow);
        assert(out.hashes.size() == 64);   // only the surviving lap
        printf("ok: lapped consumer reports overflow, keeps %zu\n", out.hashes.size());
    }

    // 4) max_hashes bounds the batch, remainder comes next
    {
        auto buf = make_region(64);
        KvEventRingWriter w(buf.data());
        KvEventRingReader r(buf.data());
        std::vector<uint64_t> in;
        for (uint64_t i = 0; i < 140; i++) in.push_back(i);
        w.publish(in.data(), 140);   // 10 entries
        auto a = r.drain(20);
        assert(!a.overflow && a.hashes.size() >= 20 && a.hashes.size() <= 28);
        auto b = r.drain();
        assert(!b.overflow);
        std::vector<uint64_t> all = a.hashes;
        all.insert(all.end(), b.hashes.begin(), b.hashes.end());
        assert(all == in);
        printf("ok: max_hashes split %zu + %zu == 140\n", a.hashes.size(), b.hashes.size());
    }

    // 5) disabled ring is a total no-op
    {
        assert(kvevent_ring_bytes(0) == 0);
        KvEventRingWriter w(nullptr);
        assert(!w.enabled());
        w.publish(nullptr, 0);
        KvEventRingReader r(nullptr);
        assert(!r.enabled());
        assert(r.drain().hashes.empty());
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
                    w.publish(h, 7);
                }
            });
        }
        for (auto& th : ts) th.join();
        std::unordered_set<uint64_t> seen;
        size_t total = 0;
        for (;;) {
            auto out = r.drain();
            assert(!out.overflow);
            if (out.hashes.empty()) break;
            for (uint64_t h : out.hashes) {
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
        w.publish(in.data(), 3);
        KvEventRingReader r(buf.data());
        w.publish(in.data(), 3);
        assert(r.drain().hashes.size() == 3);
        KvEventRingWriter::init(buf.data(), 64);   // restart: head resets to 0
        auto out = r.drain();
        assert(out.overflow);
        printf("ok: producer restart flagged as overflow\n");
    }

    printf("ALL RING TESTS PASSED\n");
    return 0;
}
