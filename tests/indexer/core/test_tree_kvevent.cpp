// End-to-end for the KV-event ring inside a real RadixShmemTree: inserts must
// put exactly the added blocks on the ring as stored, LRU eviction exactly the
// evicted node's blocks as removed, reset a cleared record -- in the order a
// consumer can act on. test_kvevent_ring.cpp covers the ring; this the wiring.

#include "kvevent.h"
#include "radix_tree.h"
#include "types.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <set>
#include <sys/mman.h>
#include <string>
#include <vector>

using namespace radixshmem;

// Removed hashes of a drain, in order.
static std::vector<uint64_t> removed(const KvEventRingReader::DrainResult& d) {
    std::vector<uint64_t> out;
    for (const auto& r : d.records)
        if (r.kind == KVEVENT_REMOVED) out.insert(out.end(), r.hashes.begin(), r.hashes.end());
    return out;
}

static std::vector<uint64_t> seq(uint64_t base, int n) {
    std::vector<uint64_t> h(n);
    for (int i = 0; i < n; i++) h[i] = base + (uint64_t)i;
    return h;
}

static ShmConfig cfg(uint32_t max_blocks, uint32_t ring_capacity) {
    ShmConfig c;
    c.max_nodes = 512;
    c.max_blocks = max_blocks;
    c.register_chunk_tokens = 16;   // block_size 1, so 16 tokens = 16 blocks
    c.hugepage_path = "";
    c.component_mask = COMPONENT_MASK_FULL;
    c.kvevent_ring_capacity = ring_capacity;
    return c;
}

static void insert_path(RadixShmemTree& t, const std::vector<uint64_t>& h) {
    std::vector<int32_t> slots;
    int got = t.allocate_slots((int)h.size(), slots);
    assert(got == (int)h.size());
    auto r = t.insert(h.data(), (int)h.size(), slots.data(), (int)slots.size());
    assert(r.error == InsertError::OK);
}

int main() {
    // ---- 1. A ring-less region publishes nothing, and the reader knows it.
    {
        const std::string name = "/kvevent_tree_noring";
        shm_unlink(name.c_str());
        auto t = RadixShmemTree::create(name, cfg(/*max_blocks=*/32, /*ring=*/0));
        assert(t.kvevent_ring_offset() == 0);
        auto h = seq(1000, 8);
        insert_path(t, h);
        t.destroy();
        printf("ok: kvevent_ring_capacity=0 leaves no ring\n");
    }

    // ---- 2. Eviction publishes the evicted node's hashes onto the ring.
    {
        const std::string name = "/kvevent_tree_evict";
        shm_unlink(name.c_str());
        // 16 blocks total: two 8-block paths fit, the third forces eviction.
        auto t = RadixShmemTree::create(name, cfg(/*max_blocks=*/16, /*ring=*/64));
        assert(t.kvevent_ring_offset() != 0);

        KvEventRingReader reader((char*)t.shm_base() + t.kvevent_ring_offset());
        reader.seek_to_head();

        auto a = seq(10000, 8);
        auto b = seq(20000, 8);
        insert_path(t, a);
        insert_path(t, b);
        // Nothing evicted yet: the pool held both paths exactly.
        auto quiet = reader.drain();
        assert(removed(quiet).empty() && !quiet.overflow);

        // Allocating past capacity evicts the LRU path (a, inserted first).
        std::vector<int32_t> slots;
        int got = t.allocate_slots(8, slots);
        assert(got == 8);

        auto res = reader.drain();
        assert(!res.overflow);
        assert(removed(res).size() == 8);
        const auto removed_res = removed(res);
        std::set<uint64_t> seen(removed_res.begin(), removed_res.end());
        std::set<uint64_t> want(a.begin(), a.end());
        assert(seen == want);
        // A drained cursor does not re-report.
        assert(removed(reader.drain()).empty());
        t.recycle_slots(slots.data(), (int)slots.size());
        t.destroy();
        printf("ok: LRU eviction publishes exactly the evicted path's 8 hashes\n");
    }

    // ---- 3. reset() drops the tree and says so: a consumer must forget the
    //         node, which only an explicit cleared record can tell it.
    {
        const std::string name = "/kvevent_tree_reset";
        shm_unlink(name.c_str());
        auto t = RadixShmemTree::create(name, cfg(/*max_blocks=*/16, /*ring=*/64));
        KvEventRingReader reader((char*)t.shm_base() + t.kvevent_ring_offset());
        reader.seek_to_head();
        insert_path(t, seq(30000, 8));
        t.reset();
        auto res = reader.drain();
        assert(!res.overflow);
        assert(res.records.size() == 2);
        assert(res.records[0].kind == KVEVENT_STORED && res.records[0].hashes.size() == 8);
        assert(res.records[1].kind == KVEVENT_CLEARED && res.records[1].hashes.empty());
        assert(removed(res).empty());   // a reset is not a per-block eviction
        t.destroy();
        printf("ok: reset() publishes one cleared record\n");
    }

    // ---- 4. A second reader attaching late must not replay history.
    {
        const std::string name = "/kvevent_tree_late";
        shm_unlink(name.c_str());
        auto t = RadixShmemTree::create(name, cfg(/*max_blocks=*/16, /*ring=*/64));
        KvEventRingReader early((char*)t.shm_base() + t.kvevent_ring_offset());
        early.seek_to_head();
        insert_path(t, seq(40000, 8));
        insert_path(t, seq(50000, 8));
        std::vector<int32_t> slots;
        assert(t.allocate_slots(8, slots) == 8);
        auto first = early.drain();
        assert(removed(first).size() == 8);

        KvEventRingReader late((char*)t.shm_base() + t.kvevent_ring_offset());
        late.seek_to_head();
        assert(removed(late.drain()).empty());
        t.recycle_slots(slots.data(), (int)slots.size());
        t.destroy();
        printf("ok: a late attacher sees no pre-attach evictions\n");
    }

    // ---- 5. Insert announces exactly the blocks it added: an extension of a
    //         stored path names only the new suffix, a duplicate names nothing,
    //         and the eviction it causes comes first.
    {
        const std::string name = "/kvevent_tree_store";
        shm_unlink(name.c_str());
        auto t = RadixShmemTree::create(name, cfg(/*max_blocks=*/16, /*ring=*/256));
        KvEventRingReader reader((char*)t.shm_base() + t.kvevent_ring_offset());
        reader.seek_to_head();

        auto a = seq(60000, 4);
        insert_path(t, a);
        auto ab = seq(60000, 8);              // a plus 4 new blocks
        {
            std::vector<int32_t> slots;
            assert(t.allocate_slots(4, slots) == 4);
            auto r = t.insert(ab.data(), 8, slots.data(), 4, /*start=*/4);
            assert(r.error == InsertError::OK);
        }
        {
            std::vector<int32_t> slots;     // same path again: nothing new
            assert(t.allocate_slots(4, slots) == 4);
            auto r = t.insert(a.data(), 4, slots.data(), 4);
            assert(r.error == InsertError::OK);
        }
        auto res = reader.drain();
        assert(!res.overflow);
        assert(res.records.size() == 1);    // adjacent stored entries merge
        assert(res.records[0].kind == KVEVENT_STORED);
        assert(res.records[0].hashes == ab);

        // A fresh 8-block path over a full pool evicts ab first.
        auto c = seq(70000, 8);
        insert_path(t, seq(80000, 8));      // pool now full (ab + this)
        reader.drain();
        insert_path(t, c);
        res = reader.drain();
        assert(res.records.size() == 2);
        assert(res.records[0].kind == KVEVENT_REMOVED);
        assert(res.records[1].kind == KVEVENT_STORED && res.records[1].hashes == c);
        t.destroy();
        printf("ok: insert publishes only the added suffix, after its evictions\n");
    }

    printf("ALL TREE KVEVENT TESTS PASSED\n");
    return 0;
}
