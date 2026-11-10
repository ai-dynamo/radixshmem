// Node-wide record of which blocks enter and leave a radixshmem tree.
//
// Every process attached to a region inserts into and evicts from the same tree,
// so no single engine knows what the node holds. Writers publish block hashes
// tagged stored / removed / cleared into this ring in the shm region; the
// radix-server drains it into the node's placement stream (radixshmem.kv_events).
// A reader that falls more than `capacity` entries behind reports overflow and
// must treat the node's state as unknown.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace radixshmem {

// Larger batches span consecutive entries; each entry stands alone.
static constexpr uint32_t KVEVENT_HASHES_PER_ENTRY = 14;

enum KvEventKind : uint32_t {
    KVEVENT_REMOVED = 0,
    KVEVENT_STORED = 1,
    KVEVENT_CLEARED = 2,   // tree reset; no hashes
};

struct KvEventEntry {
    // 1-based slot sequence, written last; a reader checks it before and after
    // copying so it never keeps a torn entry. 0 = never written.
    std::atomic<uint64_t> seq;
    uint32_t num_hashes;
    uint32_t kind;   // KvEventKind
    uint64_t hashes[KVEVENT_HASHES_PER_ENTRY];
};
static_assert(sizeof(KvEventEntry) == 128, "KvEventEntry must stay 128 bytes");

// Followed by `capacity` entries.
struct KvEventRingHeader {
    uint32_t capacity;       // power of two; 0 = no ring
    uint32_t _pad;
    std::atomic<uint64_t> head;   // entries ever reserved
};

inline uint64_t kvevent_ring_bytes(uint32_t capacity) {
    if (capacity == 0) return 0;
    return sizeof(KvEventRingHeader) + (uint64_t)capacity * sizeof(KvEventEntry);
}

// Power of two, at least 64.
uint32_t kvevent_ring_normalize_capacity(uint32_t requested);

// Lock-free, multi-producer: a publish reserves all its entries first. Never
// blocks on readers; a full ring overwrites the oldest entries.
class KvEventRingWriter {
public:
    KvEventRingWriter() = default;
    explicit KvEventRingWriter(void* base);   // ring header in the region; null = disabled

    bool enabled() const { return hdr_ != nullptr; }
    void publish(KvEventKind kind, const uint64_t* hashes, uint32_t count);
    static void init(void* base, uint32_t capacity);   // region owner, at creation

private:
    KvEventRingHeader* hdr_ = nullptr;
    KvEventEntry* entries_ = nullptr;
};

// Each reader owns its cursor.
class KvEventRingReader {
public:
    struct Record {
        KvEventKind kind;
        std::vector<uint64_t> hashes;
    };
    struct DrainResult {
        std::vector<Record> records;   // in order; adjacent entries of one kind merged
        bool overflow = false;         // entries were lost
    };

    KvEventRingReader() = default;
    explicit KvEventRingReader(void* base);

    bool enabled() const { return hdr_ != nullptr; }

    void seek_to_head();
    // Everything since the last drain, up to about `max_hashes`.
    DrainResult drain(size_t max_hashes = 65536);

private:
    KvEventRingHeader* hdr_ = nullptr;
    KvEventEntry* entries_ = nullptr;
    uint64_t cursor_ = 0;
};

}  // namespace radixshmem
