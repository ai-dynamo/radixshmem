// End-to-end for the eviction ring inside a real RadixShmemTree: an LRU eviction
// must put exactly the evicted node's block hashes on the ring, in the order a
// consumer can act on. ring_test.cpp covers the ring; this covers the wiring.

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
        assert(quiet.hashes.empty() && !quiet.overflow);

        // Allocating past capacity evicts the LRU path (a, inserted first).
        std::vector<int32_t> slots;
        int got = t.allocate_slots(8, slots);
        assert(got == 8);

        auto res = reader.drain();
        assert(!res.overflow);
        assert(res.hashes.size() == 8);
        std::set<uint64_t> seen(res.hashes.begin(), res.hashes.end());
        std::set<uint64_t> want(a.begin(), a.end());
        assert(seen == want);
        // A drained cursor does not re-report.
        assert(reader.drain().hashes.empty());
        t.recycle_slots(slots.data(), (int)slots.size());
        t.destroy();
        printf("ok: LRU eviction publishes exactly the evicted path's 8 hashes\n");
    }

    // ---- 3. reset() drops the tree; a consumer that only watched the ring
    //         would keep advertising everything, which is why the Python tier
    //         pairs reset() with an explicit clear rather than trusting this.
    {
        const std::string name = "/kvevent_tree_reset";
        shm_unlink(name.c_str());
        auto t = RadixShmemTree::create(name, cfg(/*max_blocks=*/16, /*ring=*/64));
        KvEventRingReader reader((char*)t.shm_base() + t.kvevent_ring_offset());
        reader.seek_to_head();
        insert_path(t, seq(30000, 8));
        t.reset();
        auto res = reader.drain();
        printf("ok: reset() published %zu hashes (a clear is the tier's job)\n",
               res.hashes.size());
        t.destroy();
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
        assert(first.hashes.size() == 8);

        KvEventRingReader late((char*)t.shm_base() + t.kvevent_ring_offset());
        late.seek_to_head();
        assert(late.drain().hashes.empty());
        t.recycle_slots(slots.data(), (int)slots.size());
        t.destroy();
        printf("ok: a late attacher sees no pre-attach evictions\n");
    }

    printf("ALL TREE KVEVENT TESTS PASSED\n");
    return 0;
}
