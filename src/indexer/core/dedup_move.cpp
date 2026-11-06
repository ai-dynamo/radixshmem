// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radix_tree.h"
#include "component_types.hpp"

#include <algorithm>
#include <cstddef>

namespace radixshmem {

namespace {

constexpr uint32_t AUX_LRU_STEPS = 4096;

uint64_t age_of(uint64_t ts, uint64_t now) { return now > ts ? now - ts : 0; }

}  // namespace

struct RadixShmemTree::MoveCost {
    int blocks = 0;
    uint32_t slots[NUM_COMPONENT_TYPES] = {};
    bool pinned = false;

    void add(const MoveCost& o) {
        blocks += o.blocks;
        for (int k = 0; k < NUM_COMPONENT_TYPES; k++) slots[k] += o.slots[k];
        pinned |= o.pinned;
    }
};

void RadixShmemTree::scan_moved(uint32_t top, int skip, int base, int cap, uint64_t now,
                                MoveCost& cost, std::vector<MovedPath>& out) {
    struct Frame {
        uint32_t id;
        int skip;
        size_t depth;
    };
    struct Block {
        uint64_t hash;
        int32_t full;
        uint64_t age;
        int32_t swa = INVALID_COMPONENT_SLOT;
        uint64_t swa_age = 0;
        int32_t mamba = INVALID_COMPONENT_SLOT;
        uint64_t mamba_age = 0;
    };
    std::vector<Frame> stack{{top, skip, 0}};
    std::vector<Block> path;
    std::vector<uint32_t> kids;
    size_t fresh = SIZE_MAX;   // first block of `path` no emitted path holds yet
    while (!stack.empty() && !cost.pinned && cost.blocks <= cap) {
        const Frame f = stack.back();
        stack.pop_back();
        const TreeNode* nd = nodes_.get(f.id);
        const int nb = (int)nd->num_blocks;
        cost.pinned |= !component_refs_zero(f.id);
        path.resize(f.depth);
        fresh = std::min(fresh, f.depth);
        const uint64_t* h = data_.get_hashes(nd->data_offset);
        const int32_t* fs = data_.get_slots(nd->data_offset, nb);
        const int32_t* sw = has_swa_ ? data_.get_swa_slots(nd->data_offset, nb) : nullptr;
        const uint64_t age = age_of(nd->last_access_time.load(std::memory_order_relaxed), now);
        for (int i = f.skip; i < nb; i++) {
            Block b{h[i], fs[i], age};
            if (sw && sw[i] != INVALID_COMPONENT_SLOT) {
                b.swa = sw[i];
                b.swa_age = age_of(swa_lru_.entry(sw[i]).ts, now);
                cost.slots[(int)ComponentType::SWA]++;
            }
            path.push_back(b);
        }
        if (has_mamba_ && nb > f.skip && nd->mamba_checkpoint_slot != INVALID_COMPONENT_SLOT) {
            path.back().mamba = nd->mamba_checkpoint_slot;
            path.back().mamba_age = age_of(mamba_lru_.entry(nd->mamba_checkpoint_slot).ts, now);
            cost.slots[(int)ComponentType::MAMBA]++;
        }
        cost.blocks += nb - f.skip;
        cost.slots[(int)ComponentType::FULL] += nb - f.skip;
        kids.clear();
        child_table_.collect_children(nd, kids);
        for (uint32_t c : kids) stack.push_back({c, 0, path.size()});
        if (!kids.empty()) continue;
        MovedPath p;
        p.base = base;
        p.full_from = base + (int)fresh;
        for (size_t i = 0; i < path.size(); i++) {
            const Block& b = path[i];
            p.hashes.push_back(b.hash);
            if (i < fresh) continue;
            const int32_t pos = base + (int32_t)i;
            p.full.push_back(b.full);
            p.full_age_ns.push_back(b.age);
            if (b.swa != INVALID_COMPONENT_SLOT)
                p.cells.push_back({ComponentType::SWA, pos, b.swa, b.swa_age});
            if (b.mamba != INVALID_COMPONENT_SLOT)
                p.cells.push_back({ComponentType::MAMBA, pos, b.mamba, b.mamba_age});
        }
        out.push_back(std::move(p));
        fresh = SIZE_MAX;
    }
}

bool RadixShmemTree::plan_dedup(const uint64_t* hashes, int n, int tail_cap, int branch_cap,
                                const uint32_t budget[NUM_COMPONENT_TYPES], DedupPlan& plan) {
    plan = DedupPlan{};
    if (n <= 0 || hashes == nullptr) return false;
    set_write_op("dedup");
    write_lock();
    const std::vector<MatchedNodeSpanView> spans =
        path_spans(query_internal(hashes, n, 0, false));
    if (spans.empty()) {
        write_unlock();
        return false;
    }
    const uint64_t now = now_ts();
    const MatchedNodeSpanView& last = spans.back();
    const int m = (int)(last.absolute_start + last.matched_blocks);
    uint32_t used[NUM_COMPONENT_TYPES] = {};
    auto fits = [&](const MoveCost& c) {
        for (int k = 0; k < NUM_COMPONENT_TYPES; k++)
            if (used[k] + c.slots[k] > budget[k]) return false;
        return true;
    };
    auto take = [&](const MoveCost& c) {
        for (int k = 0; k < NUM_COMPONENT_TYPES; k++) used[k] += c.slots[k];
    };

    std::vector<MovedPath> moved;
    std::vector<uint32_t> kids;
    MoveCost tail;
    if (last.matched_blocks < last.node_blocks) {
        scan_moved(last.node_id, (int)last.matched_blocks, m, tail_cap, now, tail, moved);
    } else {
        child_table_.collect_children(nodes_.get(last.node_id), kids);
        for (uint32_t c : kids) scan_moved(c, 0, m, tail_cap, now, tail, moved);
    }
    if (tail.pinned || tail.blocks > tail_cap || !fits(tail)) {
        write_unlock();
        return false;
    }
    take(tail);

    MovedPath cells;
    cells.base = m;
    cells.full_from = m;
    int top = -1;
    for (int i = (int)spans.size() - 1; i >= 0; i--) {
        const MatchedNodeSpanView& v = spans[i];
        if (!component_refs_zero(v.node_id)) break;
        MoveCost c;
        std::vector<MovedPath> sides;
        bool ok = true;
        if (i + 1 < (int)spans.size()) {
            kids.clear();
            child_table_.collect_children(nodes_.get(v.node_id), kids);
            for (uint32_t k : kids) {
                if (k == spans[i + 1].node_id) continue;
                MoveCost one;
                scan_moved(k, 0, (int)spans[i + 1].absolute_start, branch_cap, now, one, sides);
                if (one.pinned || one.blocks > branch_cap) {
                    ok = false;
                    break;
                }
                c.add(one);
            }
        }
        if (!ok) break;
        std::vector<MovedCell> mine;
        for (uint32_t b = 0; has_swa_ && b < v.matched_blocks; b++)
            if (v.swa_slots[b] != INVALID_COMPONENT_SLOT) {
                mine.push_back({ComponentType::SWA, (int32_t)(v.absolute_start + b), v.swa_slots[b],
                                age_of(swa_lru_.entry(v.swa_slots[b]).ts, now)});
                c.slots[(int)ComponentType::SWA]++;
            }
        if (has_mamba_ && v.matched_blocks == v.node_blocks &&
            v.mamba_checkpoint_slot != INVALID_COMPONENT_SLOT) {
            mine.push_back({ComponentType::MAMBA, (int32_t)(v.absolute_start + v.node_blocks - 1),
                            v.mamba_checkpoint_slot,
                            age_of(mamba_lru_.entry(v.mamba_checkpoint_slot).ts, now)});
            c.slots[(int)ComponentType::MAMBA]++;
        }
        if (!fits(c)) break;
        take(c);
        for (MovedPath& p : sides) moved.push_back(std::move(p));
        cells.cells.insert(cells.cells.end(), mine.begin(), mine.end());
        top = i;
    }
    if (top < 0) {
        write_unlock();
        return false;
    }
    if (!cells.cells.empty()) moved.push_back(std::move(cells));

    const uint32_t t = spans[top].node_id;
    plan.top_hash = data_.get_hashes(nodes_.get(t)->data_offset)[0];
    plan.generation = header()->reset_generation.load(std::memory_order_relaxed);
    std::vector<uint32_t> stack{t};
    while (!stack.empty()) {
        const uint32_t x = stack.back();
        stack.pop_back();
        TreeNode* nd = nodes_.get(x);
        const int nb = (int)nd->num_blocks;
        const uint64_t* h = data_.get_hashes(nd->data_offset);
        const int32_t* sw = has_swa_ ? data_.get_swa_slots(nd->data_offset, nb) : nullptr;
        for (int i = 0; i < nb; i++) {
            plan.blocks.insert(h[i]);
            if (sw && sw[i] != INVALID_COMPONENT_SLOT) plan.swa.insert(h[i]);
        }
        nd->local_ref_count.fetch_add(1, std::memory_order_relaxed);
        if (has_aux_)
            nodes_.component_state(x)->local_swa_ref_count.fetch_add(1, std::memory_order_relaxed);
        if (has_mamba_ && nd->mamba_checkpoint_slot != INVALID_COMPONENT_SLOT) {
            plan.mamba.insert(nd->mamba_checkpoint_slot);
            nodes_.component_state(x)->local_mamba_ref_count.fetch_add(1, std::memory_order_relaxed);
        }
        kids.clear();
        child_table_.collect_children(nd, kids);
        stack.insert(stack.end(), kids.begin(), kids.end());
    }
    plan.moved = std::move(moved);
    write_unlock();
    return true;
}

int RadixShmemTree::finish_dedup(DedupPlan& plan, bool commit) {
    set_write_op("dedup");
    write_lock();
    // Pinned nodes leave only through reset(), which may hand their slots out again.
    const uint32_t t = header()->reset_generation.load(std::memory_order_relaxed) == plan.generation
                           ? child_table_.lookup(INVALID_NODE_ID, plan.top_hash)
                           : INVALID_NODE_ID;
    if (t == INVALID_NODE_ID) {
        write_unlock();
        return -1;
    }
    // Nodes split since the plan carry its pins and its first hashes; nodes added
    // since do not, so they are neither unpinned nor dropped.
    bool same = true;
    std::vector<uint32_t> order, kids;
    std::vector<uint32_t> stack{t};
    while (!stack.empty()) {
        const uint32_t x = stack.back();
        stack.pop_back();
        TreeNode* nd = nodes_.get(x);
        const int nb = (int)nd->num_blocks;
        const uint64_t* h = data_.get_hashes(nd->data_offset);
        if (!plan.blocks.count(h[0])) {
            same = false;
            continue;
        }
        (void)dec_ref(nd->local_ref_count);
        if (has_aux_) (void)dec_ref(nodes_.component_state(x)->local_swa_ref_count);
        const int32_t ck = has_mamba_ ? nd->mamba_checkpoint_slot : INVALID_COMPONENT_SLOT;
        if (ck != INVALID_COMPONENT_SLOT) {
            if (plan.mamba.count(ck))
                (void)dec_ref(nodes_.component_state(x)->local_mamba_ref_count);
            else
                same = false;
        }
        const int32_t* sw = has_swa_ ? data_.get_swa_slots(nd->data_offset, nb) : nullptr;
        for (int i = 0; i < nb; i++)
            if (!plan.blocks.count(h[i]) ||
                (sw && sw[i] != INVALID_COMPONENT_SLOT && !plan.swa.count(h[i])))
                same = false;
        order.push_back(x);
        kids.clear();
        child_table_.collect_children(nd, kids);
        stack.insert(stack.end(), kids.begin(), kids.end());
    }
    int freed = 0;
    if (commit && same &&
        std::all_of(order.begin(), order.end(), [&](uint32_t x) { return component_refs_zero(x); })) {
        const uint32_t parent = nodes_.get(t)->parent_id;
        std::vector<int32_t> slots;
        for (auto it = order.rbegin(); it != order.rend(); ++it)
            freed += detach_node(*it, slots);
        post_evict_fixup(parent);
    }
    write_unlock();
    flush_evicted();
    return freed;
}

InsertResult RadixShmemTree::adopt(const uint64_t* hashes, int n, const MovedPath& p,
                                   std::vector<int32_t> unused[NUM_COMPONENT_TYPES]) {
    InsertResult r;
    const uint64_t now = now_ts();
    auto at = [now](uint64_t age) { return now > age ? now - age : 1; };
    if (!p.full.empty()) {
        std::vector<uint64_t> ts(p.full.size());
        for (size_t i = 0; i < ts.size(); i++) ts[i] = at(p.full_age_ns[i]);
        r = insert_impl(hashes, n, p.full.data(), (int)p.full.size(), p.full_from, false,
                        ts.data());
        std::vector<int32_t>& u = unused[(int)ComponentType::FULL];
        u.insert(u.end(), r.unused_slots.begin(), r.unused_slots.end());
    }
    if (p.cells.empty()) return r;
    set_write_op("adopt");
    write_lock();
    for (const MovedCell& c : p.cells) {
        const int k = (int)c.component;
        const bool swa = c.component == ComponentType::SWA;
        const int end = c.pos + 1;
        QueryInternalDetail m;
        if (c.pos >= 0 && end <= n && (swa ? has_swa_ : has_mamba_))
            m = query_internal(hashes, end, 0, false);
        if (m.matched_blocks < end) {
            unused[k].push_back(c.slot);
            continue;
        }
        uint32_t id = m.last_node_id;
        const int before = end - m.last_node_matched_blocks;
        TreeNode* nd = nodes_.get(id);
        if (swa) {
            int32_t& cell = data_.get_swa_slots(nd->data_offset, (int)nd->num_blocks)[c.pos - before];
            if (cell != INVALID_COMPONENT_SLOT) {
                unused[k].push_back(c.slot);
                continue;
            }
            cell = c.slot;
            nodes_.component_state(id)->swa_live_blocks++;
            swa_lru_.insert_by_ts(c.slot, id, (uint32_t)(c.pos - before), at(c.age_ns),
                                  AUX_LRU_STEPS);
            continue;
        }
        if (before + (int)nd->num_blocks > end) {
            id = split_node(id, end - before);
            if (id == INVALID_NODE_ID) {
                unused[k].push_back(c.slot);
                continue;
            }
            r.topology_changed = true;
            nd = nodes_.get(id);
        }
        if (nd->mamba_checkpoint_slot != INVALID_COMPONENT_SLOT) {
            unused[k].push_back(c.slot);
            continue;
        }
        nd->mamba_checkpoint_slot = c.slot;
        mamba_lru_.insert_by_ts(c.slot, id, 0, at(c.age_ns), AUX_LRU_STEPS);
    }
    write_unlock();
    return r;
}

}  // namespace radixshmem
