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

void KvEventRingWriter::publish(KvEventKind kind, const uint64_t* hashes, uint32_t count) {
    if (hdr_ == nullptr) return;
    if (kind == KVEVENT_CLEARED) {
        count = 0;
    } else if (hashes == nullptr || count == 0) {
        return;
    }

    const uint32_t cap = hdr_->capacity;
    const uint32_t mask = cap - 1;

    const uint32_t entries_needed = count == 0
        ? 1 : (count + KVEVENT_HASHES_PER_ENTRY - 1) / KVEVENT_HASHES_PER_ENTRY;
    uint64_t slot = hdr_->head.fetch_add(entries_needed, std::memory_order_acq_rel);

    uint32_t written = 0;
    for (uint32_t i = 0; i < entries_needed; ++i) {
        const uint32_t chunk =
            std::min<uint32_t>(KVEVENT_HASHES_PER_ENTRY, count - written);
        KvEventEntry* e = &entries_[slot & mask];

        e->seq.store(0, std::memory_order_release);
        e->num_hashes = chunk;
        e->kind = kind;
        if (chunk > 0) std::memcpy(e->hashes, hashes + written, (size_t)chunk * sizeof(uint64_t));
        e->seq.store(slot + 1, std::memory_order_release);

        slot += 1;
        written += chunk;
    }
}

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
        // Region recreated.
        out.overflow = true;
        cursor_ = head;
        return out;
    }
    if (head - cursor_ > cap) {
        // Lapped: resume at the oldest entry still intact.
        out.overflow = true;
        cursor_ = head - cap;
    }

    size_t total = 0;
    while (cursor_ < head && total < max_hashes) {
        KvEventEntry* e = &entries_[cursor_ & mask];
        const uint64_t want = cursor_ + 1;

        uint64_t seq = e->seq.load(std::memory_order_acquire);
        if (seq > want) {
            out.overflow = true;
            cursor_ += 1;
            continue;
        }
        if (seq != want) {
            // Reserved but not written yet: pick it up next time.
            break;
        }

        const uint32_t n = std::min(e->num_hashes, KVEVENT_HASHES_PER_ENTRY);
        const auto kind = static_cast<KvEventKind>(e->kind);
        uint64_t copied[KVEVENT_HASHES_PER_ENTRY];
        std::memcpy(copied, e->hashes, (size_t)n * sizeof(uint64_t));

        // Overwritten while copying.
        if (e->seq.load(std::memory_order_acquire) != want) {
            out.overflow = true;
            cursor_ += 1;
            continue;
        }
        cursor_ += 1;
        if (kind == KVEVENT_CLEARED) {
            out.records.push_back({kind, {}});
            continue;
        }
        if (out.records.empty() || out.records.back().kind != kind ||
            out.records.back().hashes.empty())
            out.records.push_back({kind, {}});
        auto& rec = out.records.back().hashes;
        rec.insert(rec.end(), copied, copied + n);
        total += n;
    }
    return out;
}

}  // namespace radixshmem
