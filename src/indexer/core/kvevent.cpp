#include "kvevent.h"

#include <algorithm>
#include <cstring>

namespace radixshmem {

uint32_t kvevent_ring_normalize_capacity(uint32_t requested) {
    if (requested == 0) return 0;
    uint32_t cap = 64;
    while (cap < requested && cap < (1u << 22)) cap <<= 1;
    return cap;
}

// ==================== Writer ====================

void KvEventRingWriter::init(void* base, uint32_t capacity) {
    if (base == nullptr || capacity == 0) return;
    auto* hdr = static_cast<KvEventRingHeader*>(base);
    hdr->capacity = capacity;
    hdr->_pad = 0;
    hdr->head.store(0, std::memory_order_relaxed);
    auto* entries = reinterpret_cast<KvEventEntry*>(hdr + 1);
    std::memset(entries, 0, (size_t)capacity * sizeof(KvEventEntry));
}

KvEventRingWriter::KvEventRingWriter(void* base) {
    if (base == nullptr) return;
    auto* hdr = static_cast<KvEventRingHeader*>(base);
    if (hdr->capacity == 0) return;
    hdr_ = hdr;
    entries_ = reinterpret_cast<KvEventEntry*>(hdr + 1);
}

void KvEventRingWriter::publish(const uint64_t* hashes, uint32_t count) {
    if (hdr_ == nullptr || hashes == nullptr || count == 0) return;

    const uint32_t cap = hdr_->capacity;
    const uint32_t mask = cap - 1;

    // Reserve the whole batch up front so concurrent producers (the evictor
    // thread, the remote-insert handler, and any client process evicting on
    // the same region) never land on the same slot.
    const uint32_t entries_needed =
        (count + KVEVENT_HASHES_PER_ENTRY - 1) / KVEVENT_HASHES_PER_ENTRY;
    uint64_t slot = hdr_->head.fetch_add(entries_needed, std::memory_order_acq_rel);

    uint32_t written = 0;
    while (written < count) {
        const uint32_t chunk =
            std::min<uint32_t>(KVEVENT_HASHES_PER_ENTRY, count - written);
        KvEventEntry* e = &entries_[slot & mask];

        // Invalidate before mutating: a reader that observes seq == 0, or a
        // seq it did not expect, discards the slot instead of copying a torn
        // payload out of it.
        e->seq.store(0, std::memory_order_release);
        e->num_hashes = chunk;
        std::memcpy(e->hashes, hashes + written, (size_t)chunk * sizeof(uint64_t));
        // Publish. seq is 1-based so slot zero is distinguishable from unwritten.
        e->seq.store(slot + 1, std::memory_order_release);

        slot += 1;
        written += chunk;
    }
}

// ==================== Reader ====================

KvEventRingReader::KvEventRingReader(void* base) {
    if (base == nullptr) return;
    auto* hdr = static_cast<KvEventRingHeader*>(base);
    if (hdr->capacity == 0) return;
    hdr_ = hdr;
    entries_ = reinterpret_cast<KvEventEntry*>(hdr + 1);
    cursor_ = hdr->head.load(std::memory_order_acquire);
}

void KvEventRingReader::seek_to_head() {
    if (hdr_ == nullptr) return;
    cursor_ = hdr_->head.load(std::memory_order_acquire);
}

KvEventRingReader::DrainResult KvEventRingReader::drain(size_t max_hashes) {
    DrainResult out;
    if (hdr_ == nullptr) return out;

    const uint32_t cap = hdr_->capacity;
    const uint32_t mask = cap - 1;
    uint64_t head = hdr_->head.load(std::memory_order_acquire);

    if (head < cursor_) {
        // Producer restarted (shm recreated). Treat as a full resync.
        out.overflow = true;
        cursor_ = head;
        return out;
    }
    if (head - cursor_ > cap) {
        // Lapped: the oldest unread entries are already overwritten. Report
        // the gap and restart from whatever is still intact; the caller must
        // resync rather than trust a partial batch.
        out.overflow = true;
        cursor_ = head - cap;
    }

    while (cursor_ < head && out.hashes.size() < max_hashes) {
        KvEventEntry* e = &entries_[cursor_ & mask];
        const uint64_t want = cursor_ + 1;

        uint64_t seq = e->seq.load(std::memory_order_acquire);
        if (seq > want) {
            // Overwritten by a newer lap: the events that belonged here are
            // gone. Same class of loss as an overflow.
            out.overflow = true;
            cursor_ += 1;
            continue;
        }
        if (seq != want) {
            // Slot reserved by a producer that has not finished writing yet
            // (head runs ahead of published slots). Stop here and pick it up
            // on the next drain rather than skipping past it.
            break;
        }

        const uint32_t n = std::min(e->num_hashes, KVEVENT_HASHES_PER_ENTRY);
        const size_t before = out.hashes.size();
        out.hashes.insert(out.hashes.end(), e->hashes, e->hashes + n);

        // Re-validate: if the producer lapped this slot while we copied, the
        // bytes we took are garbage. Drop them and flag the gap.
        if (e->seq.load(std::memory_order_acquire) != want) {
            out.hashes.resize(before);
            out.overflow = true;
        }
        cursor_ += 1;
    }
    return out;
}

}  // namespace radixshmem
