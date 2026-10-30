// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// check_health(): lock-free audit of one attached RadixShmemTree. Reads the
// pthread lock words, the rdma_rwlock, then walks root->leaves once, tallying
// slot references and refs; finally diffs each mempool's used-bitmap against
// the tree. Never takes a lock, never writes shm.

#include "health_check.h"
#include "radix_tree.h"

#include <pthread.h>
#include <cstdio>
#include <vector>

namespace radixshmem {

namespace {

// glibc lock-word layout (nptl/pthread_rwlock_common.c, struct_mutex.h).
// pthread_rwlock_t.__data.__readers: bit0 WRPHASE, bit1 WRLOCKED, bit2 RWAITING,
// bits 3.. = reader count. __writers = writers blocked on __writers_futex.
constexpr unsigned kRwWrLocked   = 2u;
constexpr unsigned kRwReaderShift = 3u;

void inspect_locks(const ShmHeader* hdr, HealthReport& r) {
#if defined(__GLIBC__)
    r.locks_inspected = true;
    const auto& rw = hdr->rwlock.__data;
    unsigned readers = __atomic_load_n(&rw.__readers, __ATOMIC_ACQUIRE);
    r.rwlock_readers         = readers >> kRwReaderShift;
    r.rwlock_writer_held     = (readers & kRwWrLocked) != 0;
    r.rwlock_writers_waiting = __atomic_load_n(&rw.__writers, __ATOMIC_ACQUIRE);
    r.lru_locked = __atomic_load_n(&hdr->lru_mutex.__data.__lock, __ATOMIC_ACQUIRE) != 0;
    if (r.rwlock_readers) r.flags |= HEALTH_RWLOCK_READ_HELD;
    if (r.rwlock_writer_held || r.rwlock_writers_waiting) r.flags |= HEALTH_RWLOCK_WRITE_HELD;
    if (r.lru_locked) r.flags |= HEALTH_LRU_LOCK_HELD;
#else
    (void)hdr;
    r.locks_inspected = false;
#endif
    r.rdma_rwlock = hdr->rdma_write_flag.load(std::memory_order_acquire) |
                    hdr->rdma_reader_count.load(std::memory_order_acquire);
    if (r.rdma_rwlock != 0) r.flags |= HEALTH_RDMA_LOCK_HELD;
}

// Per-pool tally: `seen` is a bitmap over slot ids referenced by the tree.
struct PoolAudit {
    const Mempool* pool = nullptr;
    HealthReport::Pool* out = nullptr;
    std::vector<uint64_t> seen;

    void init(const Mempool& p, HealthReport::Pool& o) {
        pool = &p; out = &o;
        out->enabled = p.enabled();
        out->total = p.total();
        if (out->enabled) seen.assign(Mempool::bitmap_words(out->total), 0);
    }
    void mark(int32_t s) {
        out->in_tree++;
        if (!out->enabled) return;
        if (s < 0 || (uint32_t)s >= out->total) { out->dangling++; return; }
        uint64_t& w = seen[(uint32_t)s >> 6];
        uint64_t bit = 1ull << (s & 63);
        if (w & bit) out->duplicate++;
        else w |= bit;
    }
    void finish(HealthReport& r, HealthFlag leak_flag) {
        if (!out->enabled) {
            if (out->in_tree) r.flags |= leak_flag;   // refs into a pool that doesn't exist
            return;
        }
        out->used = pool->num_used();
        for (uint32_t s = 0; s < out->total; s++) {
            bool used = pool->is_used(s);
            bool ref  = (seen[s >> 6] >> (s & 63)) & 1ull;
            if (used && !ref) out->leaked++;
            if (!used && ref) out->dangling++;
        }
        if (!out->ok()) r.flags |= leak_flag;
    }
};

} // namespace

// Friend of RadixShmemTree: the only place outside radix_tree.cpp that reads the
// pools directly. Declared in radix_tree.h.
class HealthAuditor {
public:
    static HealthReport run(const RadixShmemTree& t) {
        HealthReport r;
        if (!t.base_) { r.flags |= HEALTH_NOT_ATTACHED; return r; }
        const ShmHeader* hdr = t.header();
        inspect_locks(hdr, r);

        PoolAudit full, swa, mamba;
        full.init(t.mempool_, r.full);
        swa.init(t.swa_mempool_, r.swa);
        mamba.init(t.mamba_mempool_, r.mamba);

        const uint32_t capacity = hdr->node_pool_capacity;
        std::vector<uint32_t> stack{hdr->root_node_id};
        std::vector<uint32_t> children;
        while (!stack.empty()) {
            uint32_t id = stack.back();
            stack.pop_back();
            if (id >= capacity) continue;                 // torn read under a racing writer
            if (++r.nodes_reachable > capacity) break;    // cycle guard, same cause
            const TreeNode* node = t.nodes_.get(id);

            int64_t refs = node->ref_count.load(std::memory_order_relaxed);
            if (const NodeComponentState* cs = t.nodes_.component_state(id)) {
                refs += cs->swa_ref_count.load(std::memory_order_relaxed);
                refs += cs->mamba_ref_count.load(std::memory_order_relaxed);
            }
            if (refs != 0) { r.nodes_with_refs++; r.ref_total += refs; }

            const int nb = (int)node->num_blocks;
            if (nb > 0 && node->data_offset != INVALID_OFFSET) {
                const int32_t* fs = t.data_.get_slots(node->data_offset, nb);
                for (int i = 0; i < nb; i++) full.mark(fs[i]);
                if (const int32_t* ss = t.data_.get_swa_slots(node->data_offset, nb))
                    for (int i = 0; i < nb; i++)
                        if (ss[i] != INVALID_COMPONENT_SLOT) swa.mark(ss[i]);
            }
            if (node->mamba_checkpoint_slot != INVALID_COMPONENT_SLOT)
                mamba.mark(node->mamba_checkpoint_slot);

            children.clear();
            t.child_table_.collect_children(node, children);
            stack.insert(stack.end(), children.begin(), children.end());
        }

        r.full.held  = hdr->held_slots[(int)ComponentType::FULL].load(std::memory_order_relaxed);
        r.swa.held   = hdr->held_slots[(int)ComponentType::SWA].load(std::memory_order_relaxed);
        r.mamba.held = hdr->held_slots[(int)ComponentType::MAMBA].load(std::memory_order_relaxed);
        full.finish(r, HEALTH_FULL_SLOT_LEAK);
        swa.finish(r, HEALTH_SWA_SLOT_LEAK);
        mamba.finish(r, HEALTH_MAMBA_SLOT_LEAK);
        if (r.nodes_with_refs) r.flags |= HEALTH_REF_LEAK;
        r.nodes_in_pool = hdr->node_pool_size.load(std::memory_order_relaxed);
        if (r.nodes_in_pool != r.nodes_reachable) r.flags |= HEALTH_NODE_LEAK;
        return r;
    }
};

HealthReport check_health(const RadixShmemTree& tree) {
    return HealthAuditor::run(tree);
}

std::string HealthReport::describe() const {
    if (ok()) return "healthy\n";
    std::string s;
    char buf[256];
    auto line = [&](const char* fmt, auto... args) {
        snprintf(buf, sizeof(buf), fmt, args...);
        s += buf; s += '\n';
    };
    if (flags & HEALTH_NOT_ATTACHED) { line("%s", "no local tree attached"); return s; }
    if (flags & HEALTH_RWLOCK_READ_HELD)
        line("rwlock: %u reader(s) inside", rwlock_readers);
    if (flags & HEALTH_RWLOCK_WRITE_HELD)
        line("rwlock: writer %s, %u writer(s) waiting",
             rwlock_writer_held ? "held" : "not held", rwlock_writers_waiting);
    if (flags & HEALTH_LRU_LOCK_HELD)
        line("%s", "lru_mutex: locked");
    if (flags & HEALTH_RDMA_LOCK_HELD)
        line("rdma_rwlock: %lld (write_bit=%d readers=%lld)",
             (long long)rdma_rwlock, rdma_rwlock < 0 ? 1 : 0,
             (long long)(rdma_rwlock & ~(1LL << 63)));
    struct { const char* name; const Pool& p; HealthFlag f; } pools[] = {
        {"full",  full,  HEALTH_FULL_SLOT_LEAK},
        {"swa",   swa,   HEALTH_SWA_SLOT_LEAK},
        {"mamba", mamba, HEALTH_MAMBA_SLOT_LEAK},
    };
    for (const auto& e : pools)
        if (flags & e.f)
            line("%s slots: used=%u in_tree=%u leaked=%u dangling=%u duplicate=%u held=%u (total=%u%s)",
                 e.name, e.p.used, e.p.in_tree, e.p.leaked, e.p.dangling,
                 e.p.duplicate, e.p.held, e.p.total, e.p.enabled ? "" : ", pool absent");
    if (flags & HEALTH_REF_LEAK)
        line("refs: %u node(s) pinned, ref_total=%lld", nodes_with_refs,
             (long long)ref_total);
    if (flags & HEALTH_NODE_LEAK)
        line("nodes: reachable=%u pool_size=%u", nodes_reachable, nodes_in_pool);
    return s;
}

} // namespace radixshmem
