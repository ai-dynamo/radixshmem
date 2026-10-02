// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "rht_core.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <random>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace shmradix {

// align_up comes from types.h (via cluster_manifest.h in rht_types.h).

uint32_t rht_normalize_capacity(uint32_t requested) {
    uint32_t v = requested < RHT_MIN_CAPACITY ? RHT_MIN_CAPACITY : requested;
    if (v > (1u << 31)) return 1u << 31;   // next_pow2 would wrap to 0
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

// ==================== RhtShard — one shard's shm base ====================

RhtShard::~RhtShard() {
    if (base_ && base_ != MAP_FAILED) munmap(base_, size_);
    if (fd_ >= 0) close(fd_);
    if (is_owner_ && !name_.empty()) shm_unlink(name_.c_str());
}

RhtShard::RhtShard(RhtShard&& o) noexcept { *this = std::move(o); }

RhtShard& RhtShard::operator=(RhtShard&& o) noexcept {
    if (this != &o) {
        if (base_ && base_ != MAP_FAILED) munmap(base_, size_);
        if (fd_ >= 0) close(fd_);
        if (is_owner_ && !name_.empty()) shm_unlink(name_.c_str());
        fd_ = o.fd_; base_ = o.base_; size_ = o.size_;
        buckets_ = o.buckets_; capacity_ = o.capacity_;
        slots_per_bucket_ = o.slots_per_bucket_;
        shard_id_ = o.shard_id_; holder_rank_ = o.holder_rank_;
        is_owner_ = o.is_owner_; name_ = std::move(o.name_);
        o.fd_ = -1; o.base_ = nullptr; o.size_ = 0;
        o.buckets_ = nullptr; o.capacity_ = 0; o.slots_per_bucket_ = 1;
        o.is_owner_ = false;
        o.shard_id_ = -1; o.holder_rank_ = -1;
    }
    return *this;
}

uint64_t RhtShard::data_offset() const {
    return align_up(sizeof(RhtShardHeader), 64);
}

void RhtShard::reset() {
    if (!buckets_) return;
    memset(buckets_, 0,
           (size_t)capacity_ * slots_per_bucket_ * sizeof(RhtSlot));
}

RhtShard RhtShard::create(const std::string& name, uint32_t capacity,
                          uint32_t slots_per_bucket,
                          int shard_id, int holder_rank) {
    if (slots_per_bucket < 1) slots_per_bucket = 1;
    capacity = rht_normalize_capacity(capacity);
    RhtShard s;
    s.name_            = name;
    s.is_owner_        = true;
    s.capacity_        = capacity;
    s.slots_per_bucket_ = slots_per_bucket;
    s.shard_id_        = shard_id;
    s.holder_rank_     = holder_rank;

    uint64_t data_off = align_up(sizeof(RhtShardHeader), 64);
    uint64_t total_slots = (uint64_t)capacity * slots_per_bucket;
    s.size_ = align_up(data_off + total_slots * sizeof(RhtSlot), 4096);

    shm_unlink(name.c_str());
    s.fd_ = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
    if (s.fd_ < 0)
        throw std::runtime_error("RhtShard: shm_open create failed: " + name);
    if (ftruncate(s.fd_, s.size_) < 0) {
        close(s.fd_); shm_unlink(name.c_str());
        throw std::runtime_error("RhtShard: ftruncate failed: " + name);
    }
    s.base_ = mmap(nullptr, s.size_, PROT_READ | PROT_WRITE,
                   MAP_SHARED, s.fd_, 0);
    if (s.base_ == MAP_FAILED) {
        close(s.fd_); shm_unlink(name.c_str());
        throw std::runtime_error("RhtShard: mmap failed: " + name);
    }
    memset(s.base_, 0, s.size_);

    auto* hdr = static_cast<RhtShardHeader*>(s.base_);
    hdr->capacity         = capacity;
    hdr->slots_per_bucket = slots_per_bucket;
    hdr->data_offset      = data_off;
    hdr->shard_id         = shard_id;
    hdr->holder_rank      = holder_rank;
    hdr->slot_bytes       = RHT_SLOT_BYTES;
    hdr->magic            = RHT_SHARD_MAGIC;   // written last

    s.buckets_ = reinterpret_cast<RhtSlot*>(
        static_cast<char*>(s.base_) + data_off);
    return s;
}

RhtShard RhtShard::attach(const std::string& name, int timeout_sec) {
    RhtShard s;
    s.name_     = name;
    s.is_owner_ = false;

    auto deadline = std::chrono::steady_clock::now()
                  + std::chrono::seconds(timeout_sec);
    int fd = -1;
    while (true) {
        fd = shm_open(name.c_str(), O_RDWR, 0666);
        if (fd >= 0) break;
        if (std::chrono::steady_clock::now() > deadline)
            throw std::runtime_error("RhtShard::attach: shm_open timed out: " + name);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    s.fd_ = fd;

    // Map the header first to learn the full size, then remap.
    struct stat st{};
    if (fstat(fd, &st) < 0 || (size_t)st.st_size < sizeof(RhtShardHeader)) {
        close(fd); s.fd_ = -1;
        throw std::runtime_error("RhtShard::attach: fstat/size invalid: " + name);
    }
    s.size_ = (size_t)st.st_size;
    s.base_ = mmap(nullptr, s.size_, PROT_READ | PROT_WRITE,
                   MAP_SHARED, fd, 0);
    if (s.base_ == MAP_FAILED) {
        close(fd); s.fd_ = -1;
        throw std::runtime_error("RhtShard::attach: mmap failed: " + name);
    }

    auto* hdr = static_cast<RhtShardHeader*>(s.base_);
    while (reinterpret_cast<std::atomic<uint64_t>*>(&hdr->magic)
               ->load(std::memory_order_acquire) != RHT_SHARD_MAGIC) {
        // Region exists but the holder has not finished initializing it yet.
        if (std::chrono::steady_clock::now() > deadline)
            throw std::runtime_error("RhtShard::attach: magic timed out: " + name);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (hdr->slot_bytes != RHT_SLOT_BYTES) {
        munmap(s.base_, s.size_); s.base_ = nullptr;
        close(s.fd_); s.fd_ = -1;
        throw std::runtime_error("RhtShard::attach: slot layout mismatch: " + name);
    }
    s.capacity_         = hdr->capacity;
    s.slots_per_bucket_ = hdr->slots_per_bucket ? hdr->slots_per_bucket : 1;
    s.shard_id_         = hdr->shard_id;
    s.holder_rank_      = hdr->holder_rank;
    s.buckets_          = reinterpret_cast<RhtSlot*>(
        static_cast<char*>(s.base_) + hdr->data_offset);
    return s;
}

// ==================== RhtEngine — routing + read/write ====================

// Cheap thread-local PRNG for slot selection (no crypto strength needed).
static uint32_t fast_rand() {
    static thread_local uint32_t x = 0x9e3779b9u ^ (uint32_t)(uintptr_t)&x;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return x;
}

RhtEngine::RhtEngine(int num_shards, uint32_t shard_capacity,
                     uint32_t slots_per_bucket,
                     std::vector<void*> shard_ptrs)
    : num_shards_(num_shards),
      shard_capacity_(rht_normalize_capacity(shard_capacity)),
      shard_mask_(shard_capacity_ - 1),
      slots_per_bucket_(slots_per_bucket < 1 ? 1 : slots_per_bucket),
      shards_(std::move(shard_ptrs)) {
    assert((int)shards_.size() == num_shards);
}

void RhtEngine::enable_rdma(std::vector<IRdmaEndpoint*> endpoints,
                            std::vector<RdmaMrInfo> shard_mrs,
                            void* local_buf, uint32_t local_lkey,
                            size_t local_buf_bytes) {
    assert((int)endpoints.size() == num_shards_);
    assert((int)shard_mrs.size() == num_shards_);
    if (local_buf_bytes < (size_t)slots_per_bucket_ * RHT_SLOT_BYTES)
        throw std::invalid_argument(
            "RhtEngine::enable_rdma: scratch smaller than one bucket");
    endpoints_ = std::move(endpoints);
    shard_mrs_ = std::move(shard_mrs);
    rdma_buf_ = local_buf;
    rdma_lkey_ = local_lkey;
    rdma_buf_bytes_ = local_buf_bytes;
}

void RhtEngine::set_fault_handling(std::vector<int> holder_of_shard,
                                   PeerBreaker* breaker,
                                   std::function<IRdmaEndpoint*(int)> connect,
                                   std::function<void(int)> release) {
    assert((int)holder_of_shard.size() == num_shards_);
    holder_  = std::move(holder_of_shard);
    breaker_ = breaker;
    connect_ = std::move(connect);
    release_ = std::move(release);
}

// ==================== Remote faults ====================

static double secs_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

IRdmaEndpoint* RhtEngine::remote_ep(int shard) {
    if (!breaker_) return endpoints_[shard];
    const int h = holder_[shard];
    if (breaker_->gate(h) == PeerBreaker::Gate::SKIP) return nullptr;
    if (endpoints_[shard]) return endpoints_[shard];
    auto t0 = Clock::now();
    IRdmaEndpoint* ep = nullptr;
    std::string err = "no endpoint";
    try {
        if (connect_) ep = connect_(h);
    } catch (const std::exception& e) {
        err = e.what();
    }
    if (!ep) {
        breaker_->failed(h, "connect", err.c_str(), secs_since(t0));
        return nullptr;
    }
    for (int s = 0; s < num_shards_; s++)
        if (holder_[s] == h && shards_[s] == nullptr) endpoints_[s] = ep;
    return ep;
}

// Drops the holder's endpoint: a failed QP is replaced, never reused, and
// destroying it takes its unpolled completions off the shared CQ.
void RhtEngine::release_holder(int holder) {
    for (int s = 0; s < num_shards_; s++)
        if (holder_[s] == holder && shards_[s] == nullptr) endpoints_[s] = nullptr;
    if (release_) release_(holder);
}

void RhtEngine::holder_changed(int holder, const RdmaMrInfo* mr) {
    for (int s = 0; s < num_shards_; s++)
        if (mr && holder_[s] == holder && shards_[s] == nullptr) shard_mrs_[s] = *mr;
    release_holder(holder);
}

// On a shared DCI an error can surface on an op to another target: blame the
// endpoints that report broken(), else the op's own holder.
void RhtEngine::fault(int shard, const char* op, const std::exception& e,
                      Clock::time_point t0) {
    std::vector<int> blamed;
    for (int s = 0; s < num_shards_; s++)
        if (shards_[s] == nullptr && endpoints_[s] && endpoints_[s]->broken() &&
            std::find(blamed.begin(), blamed.end(), holder_[s]) == blamed.end())
            blamed.push_back(holder_[s]);
    if (blamed.empty()) blamed.push_back(holder_[shard]);
    const double secs = secs_since(t0);
    for (int h : blamed) {
        release_holder(h);
        breaker_->failed(h, op, e.what(), secs);
    }
}

void RhtEngine::dropped(int shard, PeerBreaker::Op op, uint64_t n) {
    if (breaker_ && n) breaker_->skipped(holder_[shard], op, n);
}

void RhtEngine::ok(int shard) {
    if (breaker_) breaker_->succeeded(holder_[shard]);
}

// ==================== Pack / Unpack ====================

uint64_t RhtEngine::pack(uint16_t tag, int32_t node_id,
                         uint32_t tree_node_id) {
    return ((uint64_t)tag << 48) |
           ((uint64_t)(uint16_t)(node_id + 1) << 32) |
           (uint64_t)tree_node_id;
}

bool RhtEngine::unpack(uint64_t head, uint16_t expected_tag,
                       int32_t& node_id, uint32_t& tree_node_id) {
    if (head == 0) return false;
    uint16_t tag = (uint16_t)(head >> 48);
    if (tag != expected_tag) return false;
    node_id = (int32_t)((head >> 32) & 0xFFFF) - 1;
    tree_node_id = (uint32_t)(head & 0xFFFFFFFF);
    return true;
}

static uint16_t ext_chk(uint64_t head, uint32_t num_blocks) {
    uint64_t x = head ^ ((uint64_t)num_blocks * 0x9E3779B97F4A7C15ULL);
    x ^= x >> 32; x ^= x >> 16;
    return (uint16_t)x;
}

uint64_t RhtEngine::pack_ext(uint64_t head, uint32_t num_blocks) {
    return ((uint64_t)ext_chk(head, num_blocks) << 48) | (uint64_t)num_blocks;
}

bool RhtEngine::unpack_ext(uint64_t head, uint64_t ext, uint32_t& num_blocks) {
    num_blocks = (uint32_t)(ext & 0xFFFFFFFF);
    return (uint16_t)(ext >> 48) == ext_chk(head, num_blocks);
}

// ==================== Routing ====================

int RhtEngine::shard_of(uint64_t key) const {
    return (int)(key % (uint64_t)num_shards_);
}

uint32_t RhtEngine::bucket_of(uint64_t key) const {
    return (uint32_t)(key / (uint64_t)num_shards_) & shard_mask_;
}

uint16_t RhtEngine::tag_of(uint64_t key) const {
    return (uint16_t)(key >> 48);
}

// ==================== Slot access (W slots per bucket) ====================
//
// No locks (RHT is soft state). Invariant: a write only ever touches ONE
// 16-byte cell (one 16B inline RDMA write / two 8B stores locally); a read
// pulls the whole bucket's W cells in one shot. A 16B cell is not atomic, so a
// reader racing a writer may see a torn cell; the tag in `head` (and a future
// hash check on `ext`) screens it out. Different peers writing different cells
// never collide; two writers racing the same cell lose an update, which is
// acceptable. Cell (b,j) lives at RhtSlot index b*W + j.

bool RhtEngine::read_bucket(int shard, uint32_t bucket, RhtSlot* out) {
    uint32_t W = slots_per_bucket_;
    uint64_t base = (uint64_t)bucket * W;
    if (shards_[shard] != nullptr || endpoints_.empty()) {
        auto* arr = (RhtSlot*)shards_[shard];
        for (uint32_t j = 0; j < W; j++) {
            out[j].head = reinterpret_cast<std::atomic<uint64_t>*>(&arr[base + j].head)
                              ->load(std::memory_order_relaxed);
            out[j].ext  = reinterpret_cast<std::atomic<uint64_t>*>(&arr[base + j].ext)
                              ->load(std::memory_order_relaxed);
        }
        return true;
    }
    IRdmaEndpoint* ep = remote_ep(shard);
    if (!ep) return false;
    // One RDMA read of the contiguous W*16 bytes.
    uint64_t remote_addr = shard_mrs_[shard].addr + base * RHT_SLOT_BYTES;
    auto t0 = Clock::now();
    try {
        ep->rdma_read(rdma_buf_, rdma_lkey_, remote_addr,
                      shard_mrs_[shard].rkey, W * RHT_SLOT_BYTES);
    } catch (const std::exception& e) {
        if (!breaker_) throw;
        fault(shard, "RHT read", e, t0);
        return false;
    }
    ok(shard);
    memcpy(out, rdma_buf_, (size_t)W * RHT_SLOT_BYTES);
    return true;
}

bool RhtEngine::write_slot(int shard, uint32_t bucket, uint32_t slot,
                           RhtSlot value) {
    uint64_t idx = (uint64_t)bucket * slots_per_bucket_ + slot;
    if (shards_[shard] != nullptr || endpoints_.empty()) {
        RhtSlot& cell = ((RhtSlot*)shards_[shard])[idx];
        // ext first, head last: a reader that sees the new head sees the new ext.
        reinterpret_cast<std::atomic<uint64_t>*>(&cell.ext)
            ->store(value.ext, std::memory_order_relaxed);
        reinterpret_cast<std::atomic<uint64_t>*>(&cell.head)
            ->store(value.head, std::memory_order_release);
        return true;
    }
    IRdmaEndpoint* ep = remote_ep(shard);
    if (!ep) return false;
    uint64_t remote_addr = shard_mrs_[shard].addr + idx * RHT_SLOT_BYTES;
    auto t0 = Clock::now();
    try {
        ep->rdma_write_async_inline(remote_addr, shard_mrs_[shard].rkey, &value,
                                    RHT_SLOT_BYTES);
    } catch (const std::exception& e) {
        if (!breaker_) throw;
        fault(shard, "RHT write", e, t0);
        return false;
    }
    return true;   // landed only once flush() returns
}

// Among candidate slots pick the one holding the shortest node (ties random):
// routing prefers the longest candidate, so a full bucket sheds its least
// useful entry instead of a random one.
static int shortest_of(const RhtSlot* slots, const uint32_t* idx, uint32_t n) {
    uint32_t best[64]; uint32_t nb = 0; uint32_t best_len = UINT32_MAX;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t len = (uint32_t)(slots[idx[i]].ext & 0xFFFFFFFF);
        if (len < best_len) { best_len = len; nb = 0; }
        if (len == best_len) best[nb++] = idx[i];
    }
    return (int)best[fast_rand() % nb];
}

// Choose which slot in `slots[0..W)` to write `packed` (= pack(tag,node,tnid)).
// Returns -1 if an IDENTICAL slot already exists (true no-op). If a slot is
// already MINE with the same tag but a DIFFERENT value (e.g. tree_node_id
// remapped by a split), it is updated IN PLACE — skipping that would leave the
// RHT pointing at a stale/split node and the query would miss.
static int choose_slot(const RhtSlot* slots, uint32_t W,
                       uint16_t tag, int32_t node_id, RhtSlot packed) {
    // Rule 0: own slot (mine + same tag). Reuse it in place. Identical → no-op.
    for (uint32_t j = 0; j < W; j++) {
        if (slots[j].head == 0) continue;
        uint16_t t  = (uint16_t)(slots[j].head >> 48);
        int32_t  nd = (int32_t)((slots[j].head >> 32) & 0xFFFF) - 1;
        if (t == tag && nd == node_id)
            return (slots[j].head == packed.head && slots[j].ext == packed.ext)
                       ? -1 : (int)j;
    }
    // Rule 1: empty slots → random empty.
    uint32_t empties[64]; uint32_t ne = 0;
    for (uint32_t j = 0; j < W && ne < 64; j++)
        if (slots[j].head == 0) empties[ne++] = j;
    if (ne > 0) return (int)empties[fast_rand() % ne];
    // Rule 2: full → the longest route for this prefix wins: replace the
    // shortest same-tag entry only if the incoming one is longer, else drop it.
    uint32_t same[64]; uint32_t ns = 0;
    for (uint32_t j = 0; j < W && ns < 64; j++)
        if ((uint16_t)(slots[j].head >> 48) == tag) same[ns++] = j;
    if (ns > 0) {
        int j = shortest_of(slots, same, ns);
        return (uint32_t)(packed.ext & 0xFFFFFFFF) > (uint32_t)(slots[j].ext & 0xFFFFFFFF)
                   ? j : -1;
    }
    // Rule 3: full, none same-tag → shed the shortest entry in the bucket.
    uint32_t all[64]; uint32_t na = 0;
    for (uint32_t j = 0; j < W && na < 64; j++) all[na++] = j;
    return shortest_of(slots, all, na);
}

void RhtEngine::insert_batch(const RhtInsertEntry* entries, int count) {
    if (count <= 0) return;
    // W>1 needs read-modify-write (read the bucket, pick a slot). Group reads
    // by shard, batch them, choose slots locally, then write only the chosen
    // single slot per entry. W==1 keeps the old blind-chain-write fast path.
    uint32_t W = slots_per_bucket_;

    std::vector<RhtSlot> bucketbuf(W);
    for (int i = 0; i < count; i++) {
        const RhtInsertEntry& e = entries[i];
        int shard = shard_of(e.key);
        uint32_t bucket = bucket_of(e.key);
        uint16_t tag = tag_of(e.key);
        RhtSlot packed = make_slot(pack(tag, e.node_id, e.tree_node_id),
                                   e.num_blocks);

        if (W == 1) {
            if (!write_slot(shard, bucket, 0, packed))   // blind overwrite (legacy)
                dropped(shard, PeerBreaker::Op::RHT_WRITE);
            continue;
        }
        if (!read_bucket(shard, bucket, bucketbuf.data())) {
            dropped(shard, PeerBreaker::Op::RHT_WRITE);
            continue;
        }
        int j = choose_slot(bucketbuf.data(), W, tag, e.node_id, packed);
        if (j < 0) continue;                         // identical entry → no-op
        if (!write_slot(shard, bucket, (uint32_t)j, packed))
            dropped(shard, PeerBreaker::Op::RHT_WRITE);
    }
}

void RhtEngine::flush() {
    for (int s = 0; s < (int)endpoints_.size(); s++) {
        if (!endpoints_[s]) continue;
        auto t0 = Clock::now();
        try {
            endpoints_[s]->flush();
        } catch (const std::exception& e) {
            if (!breaker_) throw;
            fault(s, "RHT flush", e, t0);
            continue;
        }
        ok(s);
    }
}

// ==================== Core operations ====================

void RhtEngine::lookup_batch(const uint64_t* keys, int count,
                             std::vector<RHTValue>* results) {
    for (int i = 0; i < count; i++)
        results[i].clear();
    if (count <= 0) return;

    uint32_t W = slots_per_bucket_;
    auto collect = [&](int i, const RhtSlot* slots) {
        uint16_t tag = tag_of(keys[i]);
        for (uint32_t j = 0; j < W; j++) {
            int32_t nid; uint32_t tnid, nb;
            if (unpack(slots[j].head, tag, nid, tnid) &&
                unpack_ext(slots[j].head, slots[j].ext, nb))   // torn cell → drop
                results[i].push_back({nid, tnid, nb});
        }
    };

    // Local shards: read directly. Remote: group by endpoint (shards on one
    // holder share a QP).
    struct Probe { int shard; uint32_t bucket; int result_idx; };
    std::vector<IRdmaEndpoint*> eps;
    std::vector<std::vector<Probe>> groups;
    std::vector<RhtSlot> localbuf(W);
    size_t remaining = 0;

    for (int i = 0; i < count; i++) {
        int shard = shard_of(keys[i]);
        uint32_t bucket = bucket_of(keys[i]);
        if (shards_[shard] != nullptr || endpoints_.empty()) {
            read_bucket(shard, bucket, localbuf.data());
            collect(i, localbuf.data());
            continue;
        }
        IRdmaEndpoint* ep = remote_ep(shard);
        if (!ep) {
            dropped(shard, PeerBreaker::Op::RHT_LOOKUP);
            continue;
        }
        size_t g = 0;
        while (g < eps.size() && eps[g] != ep) g++;
        if (g == eps.size()) { eps.push_back(ep); groups.emplace_back(); }
        groups[g].push_back({shard, bucket, i});
        remaining++;
    }

    // Rounds: one chain (<= RDMA_READ_CHAIN_MAX buckets) per endpoint, laid
    // back-to-back in rdma_buf_, then one wait. Scratch bounds a round.
    const size_t bucket_bytes = (size_t)W * RHT_SLOT_BYTES;
    const size_t scratch_buckets = rdma_buf_bytes_ / bucket_bytes;
    assert(scratch_buckets >= 1);   // guaranteed by enable_rdma
    std::vector<size_t> next(groups.size(), 0);
    struct Posted { size_t g, from, off; int n; };
    std::vector<Posted> posted;
    RdmaReadOp ops[RDMA_READ_CHAIN_MAX];

    // A group whose holder failed: its unread probes miss.
    auto drop_group = [&](size_t g) {
        const size_t left = groups[g].size() - next[g];
        dropped(groups[g][0].shard, PeerBreaker::Op::RHT_LOOKUP, left);
        remaining -= left;
        next[g] = groups[g].size();
    };

    while (remaining > 0) {
        posted.clear();
        size_t used = 0;
        int pending = 0;
        IRdmaEndpoint* waiter = nullptr;
        size_t cur = SIZE_MAX;   // group being posted
        auto t0 = Clock::now();
        try {
            for (size_t g = 0; g < groups.size(); g++) {
                const size_t left = groups[g].size() - next[g];
                if (left == 0) continue;
                if (used == scratch_buckets) break;
                const int n = (int)std::min({left, (size_t)RDMA_READ_CHAIN_MAX,
                                             scratch_buckets - used});
                for (int j = 0; j < n; j++) {
                    const Probe& p = groups[g][next[g] + j];
                    ops[j].local_buf   = (char*)rdma_buf_ +
                                         (used + j) * bucket_bytes;
                    ops[j].remote_addr = shard_mrs_[p.shard].addr +
                                         (uint64_t)p.bucket * bucket_bytes;
                    ops[j].remote_rkey = shard_mrs_[p.shard].rkey;
                    ops[j].length      = (uint32_t)bucket_bytes;
                }
                cur = g;
                const int owed =
                    eps[g]->rdma_read_multi_post(ops, n, rdma_lkey_);
                cur = SIZE_MAX;
                if (owed > 0 && waiter == nullptr) waiter = eps[g];
                pending += owed;
                posted.push_back({g, next[g], used, n});
                next[g] += n;
                used += n;
                remaining -= n;
            }
            if (pending > 0) waiter->rdma_read_multi_wait(pending);
        } catch (const std::exception& e) {
            if (!breaker_) {
                // Collect what was posted, or a later sync poll takes it.
                if (cur != SIZE_MAX && pending > 0) waiter->rdma_read_multi_wait(pending);
                throw;
            }
            bool collected = false;
            if (cur != SIZE_MAX) {   // a post threw; the chains before it are intact
                fault(groups[cur][0].shard, "RHT lookup", e, t0);
                drop_group(cur);
                try {
                    if (pending > 0) waiter->rdma_read_multi_wait(pending);
                    collected = true;
                } catch (const std::exception&) {
                }
            }
            if (!collected) {
                // The wait threw: fault the broken endpoints and drop every
                // posted chain, releasing the other endpoints as well so their
                // unpolled completions leave the CQ with them.
                std::vector<bool> broken(posted.size());
                bool any = false;
                for (size_t k = 0; k < posted.size(); k++)
                    any = (broken[k] = eps[posted[k].g]->broken()) || any;
                for (size_t k = 0; k < posted.size(); k++) {
                    const Posted& p = posted[k];
                    const int shard = groups[p.g][0].shard;
                    dropped(shard, PeerBreaker::Op::RHT_LOOKUP, (uint64_t)p.n);
                    if (!any || broken[k])
                        fault(shard, "RHT lookup", e, t0);
                    else
                        release_holder(holder_[shard]);
                    drop_group(p.g);
                }
                continue;
            }
        }
        for (const Posted& p : posted) {
            ok(groups[p.g][0].shard);
            for (int j = 0; j < p.n; j++)
                collect(groups[p.g][p.from + j].result_idx,
                        (const RhtSlot*)((char*)rdma_buf_ +
                                         (p.off + j) * bucket_bytes));
        }
    }
}

void RhtEngine::dump_bucket(uint64_t key, std::vector<RhtSlot>& out) {
    out.assign(slots_per_bucket_, RhtSlot{0, 0});
    read_bucket(shard_of(key), bucket_of(key), out.data());
}

bool RhtEngine::remove(uint64_t key, int32_t expected_node_id,
                       uint32_t expected_tree_node_id) {
    int shard = shard_of(key);
    uint32_t bucket = bucket_of(key);
    uint16_t tag = tag_of(key);
    std::vector<RhtSlot> slots(slots_per_bucket_);
    if (!read_bucket(shard, bucket, slots.data())) {
        dropped(shard, PeerBreaker::Op::RHT_REMOVE);
        return false;
    }
    // Find MY slot: triple-match (tag + node_id + tree_node_id). Clears only it,
    // leaving collision victims / re-registered nodes intact.
    for (uint32_t j = 0; j < slots_per_bucket_; j++) {
        int32_t nid; uint32_t tnid;
        if (!unpack(slots[j].head, tag, nid, tnid)) continue;
        if (nid == expected_node_id && tnid == expected_tree_node_id) {
            if (!write_slot(shard, bucket, j, RhtSlot{0, 0})) {
                dropped(shard, PeerBreaker::Op::RHT_REMOVE);
                return false;
            }
            return true;
        }
    }
    return false;
}

void RhtEngine::remove_batch(const RhtRemoveEntry* entries, int count) {
    // No grouping/chaining optimization: removes are read-modify-(maybe)write
    // per cell (must read to validate tag+node+tnid before zeroing), and the
    // evictor's batch is small relative to inserts. Keep it simple + correct.
    // Entries that fail validation (re-registered node / hash-collision victim)
    // are correctly left untouched.
    for (int i = 0; i < count; i++)
        remove(entries[i].key, entries[i].node_id, entries[i].tree_node_id);
}

} // namespace shmradix
