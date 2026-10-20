// KV-cache eviction event ring.
//
// The background evictor runs inside the radix_server process while the
// consumers (one per inference-engine rank attached to this shm) live in
// other processes, so eviction is invisible to them without an explicit
// channel. This is that channel: a multi-producer / multi-consumer ring in
// the shm region itself, carrying block hashes only. Multiple producers are
// possible because the same region is also evicted from by the remote-insert
// handler and by client processes, not just the evictor thread.
//
// Hashes are all a consumer needs. The router-side BlockRemoved wire format
// (llm-d and dynamo alike) is hash-only, and neither the radix tree nor the
// engine retains token ids past insertion.
//
// Consumers poll; they are never signalled. A consumer that falls more than
// `capacity` entries behind has lost events it can never recover, and must
// resynchronise by clearing everything it has published for this node --
// see KvEventRingReader::drain's `overflow` result.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace radixshmem {

// Hashes carried inline per entry. Evicted nodes holding more than this are
// split across consecutive entries; a consumer treats each entry
// independently, so no reassembly is required.
static constexpr uint32_t KVEVENT_HASHES_PER_ENTRY = 14;

// One eviction record. Fixed size so the ring is index-addressable and a
// reader can detect being overwritten mid-copy by re-reading `seq`.
struct KvEventEntry {
    // Publication sequence, 1-based. seq == 0 means never written. The
    // producer stamps this last (release) and the consumer reads it first
    // and last (acquire) to validate the payload it copied out.
    std::atomic<uint64_t> seq;
    uint32_t num_hashes;
    uint32_t _pad;
    uint64_t hashes[KVEVENT_HASHES_PER_ENTRY];
};
static_assert(sizeof(KvEventEntry) == 128, "KvEventEntry must stay 128 bytes");

// Ring header, followed immediately by `capacity` KvEventEntry slots.
struct KvEventRingHeader {
    uint32_t capacity;       // entry count; power of two, 0 = ring disabled
    uint32_t _pad;
    // Total entries ever published. Producer-only writes. A consumer's cursor
    // is a value of this counter; `head - cursor > capacity` means overflow.
    std::atomic<uint64_t> head;
};

// Bytes needed for a ring of `capacity` entries (0 for a disabled ring).
inline uint64_t kvevent_ring_bytes(uint32_t capacity) {
    if (capacity == 0) return 0;
    return sizeof(KvEventRingHeader) + (uint64_t)capacity * sizeof(KvEventEntry);
}

// Round up to a power of two, minimum 64. Capacity must be a power of two so
// the ring can mask instead of divide.
uint32_t kvevent_ring_normalize_capacity(uint32_t requested);

// Ring writer. Lock-free and safe for concurrent producers: each publish
// atomically reserves its slots before filling them. Publication happens
// after the tree's write lock is dropped -- the ring has no interaction with
// the tree lock.
class KvEventRingWriter {
public:
    KvEventRingWriter() = default;
    // `base` points at the ring header inside the shm region; null disables.
    explicit KvEventRingWriter(void* base);

    bool enabled() const { return hdr_ != nullptr; }

    // Publish `count` hashes, splitting across entries as needed. Overwrites
    // the oldest entries when full: eviction must never block on a consumer.
    void publish(const uint64_t* hashes, uint32_t count);

    // Initialise a freshly created region. Owner only, before any consumer
    // attaches.
    static void init(void* base, uint32_t capacity);

private:
    KvEventRingHeader* hdr_ = nullptr;
    KvEventEntry* entries_ = nullptr;
};

// Multi-consumer reader. Each consumer owns its cursor; consumers do not
// coordinate and cannot block the producer.
class KvEventRingReader {
public:
    struct DrainResult {
        // Hashes read, in publication order.
        std::vector<uint64_t> hashes;
        // True when the producer lapped this cursor: an unknown number of
        // events was missed. The caller must fall back to a full resync
        // (publish AllBlocksCleared) rather than trust the partial batch.
        bool overflow = false;
    };

    KvEventRingReader() = default;
    explicit KvEventRingReader(void* base);

    bool enabled() const { return hdr_ != nullptr; }

    // Start reading from the current head, skipping history. Called once on
    // attach so a late consumer does not replay evictions it never published
    // stores for.
    void seek_to_head();

    // Read everything published since the last drain. `max_hashes` bounds the
    // work done in one call; the remainder is returned by the next drain.
    DrainResult drain(size_t max_hashes = 65536);

private:
    KvEventRingHeader* hdr_ = nullptr;
    KvEventEntry* entries_ = nullptr;
    uint64_t cursor_ = 0;
};

}  // namespace radixshmem
