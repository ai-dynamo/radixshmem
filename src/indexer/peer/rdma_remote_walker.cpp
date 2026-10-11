// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include "rdma_remote_walker.h"

#include "component_set.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace radixshmem {

static constexpr int MAX_LOCK_RETRIES = 10000;

RdmaRemoteWalker::RdmaRemoteWalker(IRdmaEndpoint& ep, void* local_buf,
                                   uint32_t local_lkey, size_t buf_size)
    : ep_(ep),
      local_buf_(local_buf),
      local_lkey_(local_lkey),
      local_buf_size_(buf_size) {}

// ==================== Address helpers ====================

uint64_t RdmaRemoteWalker::node_addr(const RemoteTreeInfo& r,
                                     uint32_t node_id) const {
    return r.base_addr + r.node_pool_offset +
           (uint64_t)node_id * sizeof(TreeNode);
}

uint64_t RdmaRemoteWalker::ht_bucket_addr(const RemoteTreeInfo& r,
                                          uint32_t bucket) const {
    return r.base_addr + r.global_ht_offset +
           (uint64_t)bucket * sizeof(uint32_t);
}

uint64_t RdmaRemoteWalker::ht_entry_addr(const RemoteTreeInfo& r,
                                         uint32_t index) const {
    uint64_t entries_start = r.base_addr + r.global_ht_offset +
                             (uint64_t)r.global_ht_bucket_count * sizeof(uint32_t);
    return entries_start + (uint64_t)index * sizeof(ChainHTEntry);
}

uint64_t RdmaRemoteWalker::data_addr(const RemoteTreeInfo& r,
                                     uint64_t offset) const {
    return r.base_addr + offset;
}

uint64_t RdmaRemoteWalker::ref_count_addr(const RemoteTreeInfo& r,
                                           uint32_t node_id) const {
    return node_addr(r, node_id) + offsetof(TreeNode, ref_count);
}

// The peer's payload carries a SWA lane iff its config enables SWA.
static inline bool remote_swa_lane(const RemoteTreeInfo& r) {
    return component_has_swa(r.component_config.component_mask);
}

// Lane refcount words inside NodeComponentState: the FAA targets peers hit.
static constexpr uint64_t SWA_REF_OFF =
    offsetof(NodeComponentState, swa_ref_count);
static constexpr uint64_t MAMBA_REF_OFF =
    offsetof(NodeComponentState, mamba_ref_count);

uint64_t RdmaRemoteWalker::cst_addr(const RemoteTreeInfo& r,
                                    uint32_t node_id) const {
    return r.base_addr + r.component_state_offset +
           (uint64_t)node_id * r.component_state_entry_size;
}

void RdmaRemoteWalker::dec_ref_node(const RemoteTreeInfo& remote,
                                     uint32_t node_id) {
    ep_.rdma_fetch_add(ref_count_addr(remote, node_id), remote.rkey,
                       (uint64_t)-1LL, local_buf_, local_lkey_);
    atomic_count_++;
}

// ==================== RDMA rwlock ====================

// Read-before-FAA (fixes the remote-reader writer-livelock).
bool RdmaRemoteWalker::rdma_read_lock(const RemoteTreeInfo& remote) {
    auto flag_set = [&]() {
        ep_.rdma_read(local_buf_, local_lkey_,
                      remote.rdma_rwlock_addr, remote.rkey,
                      (uint32_t)sizeof(int64_t));
        read_count_++;
        return *reinterpret_cast<const int64_t*>(local_buf_) != 0;
    };
    for (int i = 0; i < MAX_LOCK_RETRIES; i++) {
        ep_.rdma_fetch_add(remote.rdma_reader_count_addr, remote.rkey, 1,
                           local_buf_, local_lkey_);
        atomic_count_++;
        if (!flag_set())
            return true;
        ep_.rdma_fetch_add(remote.rdma_reader_count_addr, remote.rkey,
                           (uint64_t)-1LL, local_buf_, local_lkey_);
        atomic_count_++;
        while (i < MAX_LOCK_RETRIES && flag_set()) i++;
    }
    return false;
}

void RdmaRemoteWalker::rdma_read_unlock(const RemoteTreeInfo& remote) {
    ep_.rdma_fetch_add(
        remote.rdma_reader_count_addr, remote.rkey, (uint64_t)-1LL,
        local_buf_, local_lkey_);
    atomic_count_++;
}

// ==================== Remote reads ====================

RemoteTreeInfo RdmaRemoteWalker::read_remote_info(const RdmaMrInfo& remote_mr) {
    RemoteTreeInfo info;
    info.mr = remote_mr;
    info.rkey = remote_mr.rkey;
    info.base_addr = remote_mr.addr;

    static_assert(sizeof(ShmHeader) <= 4096, "ShmHeader too large");
    ep_.rdma_read(local_buf_, local_lkey_,
                  remote_mr.addr, remote_mr.rkey,
                  (uint32_t)sizeof(ShmHeader));
    read_count_++;

    auto* hdr = (ShmHeader*)local_buf_;
    info.node_pool_offset = hdr->node_pool_offset;
    info.node_pool_capacity = hdr->node_pool_capacity;
    info.data_pool_offset = hdr->data_pool_offset;
    info.global_ht_offset = hdr->global_ht_offset;
    info.global_ht_bucket_count = hdr->global_ht_bucket_count;
    info.global_ht_entry_capacity = hdr->global_ht_entry_capacity;
    info.root_node_id = hdr->root_node_id;
    info.rdma_rwlock_addr = remote_mr.addr + offsetof(ShmHeader, rdma_write_flag);
    info.rdma_reader_count_addr = info.rdma_rwlock_addr + RDMA_READER_COUNT_DELTA;
    // Component fields come from the peer's manifest entry (PeerConnector);
    // component walks fail closed while component_info_valid is false.

    return info;
}

TreeNodeData RdmaRemoteWalker::read_node(const RemoteTreeInfo& remote,
                                         uint32_t node_id) {
    TreeNodeData node;
    memset(&node, 0, sizeof(node));
    // Range-check node_id against peer NodePool; out-of-range returns zeroed
    // node (num_blocks==0 stops the walk cleanly).
    if (remote.node_pool_capacity != 0 && node_id >= remote.node_pool_capacity) {
        return node;
    }
    ep_.rdma_read(local_buf_, local_lkey_,
                  node_addr(remote, node_id), remote.rkey,
                  sizeof(TreeNode));
    read_count_++;
    memcpy(&node, local_buf_, sizeof(TreeNodeData));
    return node;
}

// Validate against garbage from a stale slot: num_blocks fits local_buf_
// and data_offset+total lies inside the peer MR; else bail (return false).
bool RdmaRemoteWalker::payload_read_ok(const RemoteTreeInfo& remote,
                                       uint64_t data_offset, int num_blocks,
                                       uint64_t total) const {
    if (num_blocks <= 0) return false;
    if (total > local_buf_size_) return false;
    if (remote.mr.length != 0 &&
        (data_offset >= remote.mr.length ||
         total > remote.mr.length - data_offset)) {
        return false;   // garbage / out-of-bounds data_offset
    }
    return true;
}

bool RdmaRemoteWalker::read_node_lanes(const RemoteTreeInfo& remote,
                                       uint64_t data_offset, int num_blocks,
                                       uint64_t* out_hashes, int32_t* out_full,
                                       int32_t* out_swa) {
    const bool swa = remote_swa_lane(remote);
    uint64_t total = (uint64_t)data_block_bytes(num_blocks, swa);
    if (!payload_read_ok(remote, data_offset, num_blocks, total)) return false;
    uint32_t hash_bytes = (uint32_t)(num_blocks * sizeof(uint64_t));
    uint32_t lane_bytes = (uint32_t)(num_blocks * sizeof(int32_t));
    ep_.rdma_read(local_buf_, local_lkey_, data_addr(remote, data_offset),
                  remote.rkey, (uint32_t)total);
    read_count_++;
    memcpy(out_hashes, local_buf_, hash_bytes);
    memcpy(out_full, (char*)local_buf_ + hash_bytes, lane_bytes);
    if (swa)
        memcpy(out_swa, (char*)local_buf_ + hash_bytes + lane_bytes, lane_bytes);
    else
        std::fill_n(out_swa, num_blocks, INVALID_COMPONENT_SLOT);
    return true;
}

uint32_t RdmaRemoteWalker::ht_lookup(const RemoteTreeInfo& remote,
                                     uint32_t parent_id,
                                     uint64_t token_hash) {
    uint32_t mask = remote.global_ht_bucket_count - 1;
    uint32_t bucket = (uint32_t)(token_hash ^ (token_hash >> 32)) & mask;

    uint32_t entry_idx;
    ep_.rdma_read(local_buf_, local_lkey_,
                  ht_bucket_addr(remote, bucket), remote.rkey,
                  sizeof(uint32_t));
    read_count_++;
    memcpy(&entry_idx, local_buf_, sizeof(uint32_t));

    // A torn entry can point outside the pool or loop; fail closed before a READ leaves the MR.
    uint32_t hops = 0;
    while (entry_idx != HT_END) {
        if (entry_idx >= remote.global_ht_entry_capacity ||
            ++hops > remote.global_ht_entry_capacity)
            return INVALID_NODE_ID;
        ChainHTEntry entry;
        ep_.rdma_read(local_buf_, local_lkey_,
                      ht_entry_addr(remote, entry_idx), remote.rkey,
                      sizeof(ChainHTEntry));
        read_count_++;
        memcpy(&entry, local_buf_, sizeof(ChainHTEntry));

        if (entry.token_hash == token_hash)
            return entry.child_node_id;

        entry_idx = entry.hash_next;
    }

    return INVALID_NODE_ID;
}

// ==================== Walk with global rwlock ====================

// ==================== Bottom-up walk ====================

// ==================== Batched walk ====================

// ==================== Component walk / release ====================

// Pin and release must relocate the same hash path. `anchor` is metadata-only
// (root, or the hinted deepest node whose payload was already consumed).
int RdmaRemoteWalker::build_path_by_descent(
        const RemoteTreeInfo& remote, const uint64_t* hashes, int n,
        std::vector<ComponentPathNode>& out,
        uint32_t anchor, int anchor_matched) {
    uint32_t current_id = anchor;
    int matched = anchor_matched;
    // Not `current_id != anchor`: a corrupt remote HT entry pointing back at
    // the anchor must fail hash verification, not skip it (walk must halt).
    bool at_anchor = true;

    while (matched < n) {
        TreeNodeData node = read_node(remote, current_id);

        if (!at_anchor) {
            int nb = (int)node.num_blocks;
            if (nb <= 0) break;
            int cmp = std::min(nb, n - matched);
            ComponentPathNode pn;
            pn.node_id = current_id;
            pn.absolute_start = (uint32_t)matched;
            pn.meta = node;
            pn.hashes.resize(nb);
            pn.full.resize(nb);
            pn.swa.resize(nb);

            if (!read_node_lanes(remote, node.data_offset, nb,
                                 pn.hashes.data(), pn.full.data(),
                                 pn.swa.data()))
                break;

            // The remote walk always re-verifies hashes (no local
            // edge-key trust across the wire).
            int ml = 0;
            for (int i = 0; i < cmp; i++) {
                if (pn.hashes[i] == hashes[matched + i])
                    ml++;
                else
                    break;
            }

            if (ml > 0) {
                pn.match_len = ml;
                out.push_back(std::move(pn));
            }

            matched += ml;
            if (ml < nb) break;
        }
        at_anchor = false;

        if (matched < n && node.children_count > 0) {
            uint32_t child_id = ht_lookup(remote, current_id,
                                          hashes[matched]);
            if (child_id != INVALID_NODE_ID) {
                current_id = child_id;
                continue;
            }
        }
        break;
    }
    return matched;
}

// Hashes are verified over the whole stitched path (no cross-rank edge-key
// trust); any divergence returns -1 and the caller bails to a miss.
int RdmaRemoteWalker::build_path_from_hints(
        const RemoteTreeInfo& remote, const uint64_t* hashes, int n,
        int incumbent, const RemoteHit* hints, int num_hints,
        std::vector<ComponentPathNode>& out, int& matched) {
    matched = 0;

    // Phase 1: batch-read metadata for all hint nodes, chunked by buffer size.
    int metas_per_chunk = (int)(local_buf_size_ / sizeof(TreeNode));
    if (metas_per_chunk < 1) metas_per_chunk = 1;
    std::vector<TreeNodeData> hint_meta(num_hints);
    for (int base = 0; base < num_hints; base += metas_per_chunk) {
        int cnt = std::min(metas_per_chunk, num_hints - base);
        std::vector<RdmaReadOp> ops(cnt);
        for (int i = 0; i < cnt; i++) {
            const uint32_t nid = hints[base + i].tree_node_id;
            // An out-of-pool hint id would aim the READ past the node pool
            // (and, on the fake endpoint, out of process memory): stale.
            if (remote.node_pool_capacity != 0 &&
                nid >= remote.node_pool_capacity)
                return -1;
            ops[i].local_buf = (char*)local_buf_ + i * sizeof(TreeNodeData);
            ops[i].remote_addr = node_addr(remote, nid);
            ops[i].remote_rkey = remote.rkey;
            ops[i].length = sizeof(TreeNode);
        }
        ep_.rdma_read_multi(ops.data(), cnt, local_lkey_);
        read_count_ += cnt;

        for (int i = 0; i < cnt; i++)
            memcpy(&hint_meta[base + i],
                   (char*)local_buf_ + i * sizeof(TreeNodeData),
                   sizeof(TreeNodeData));
    }

    // Cache tree_node_id -> meta for O(1) Phase-4 gap-fill lookups.
    std::unordered_map<uint32_t, TreeNodeData> cache;
    cache.reserve(num_hints * 2);
    for (int i = 0; i < num_hints; i++)
        cache.emplace(hints[i].tree_node_id, hint_meta[i]);

    // Phase 2: the deepest hint is the chain end the router found; it must
    // still reach past the incumbent with its real length.
    int deepest_idx = 0;
    for (int i = 1; i < num_hints; i++)
        if (hints[i].position > hints[deepest_idx].position) deepest_idx = i;
    if (hints[deepest_idx].position + (int)hint_meta[deepest_idx].num_blocks <=
        incumbent)
        return 0;   // nothing extends past the incumbent: empty miss

    // Phase 3: Validate deepest node — read its data and check first hash
    int deepest_pos = hints[deepest_idx].position;
    TreeNodeData& deepest_meta = hint_meta[deepest_idx];
    int deepest_nb = (int)deepest_meta.num_blocks;

    if (deepest_nb <= 0 ||
        (uint64_t)data_block_bytes(deepest_nb, remote_swa_lane(remote)) >
            local_buf_size_)
        return -1;
    std::vector<uint64_t> deepest_hashes(deepest_nb);
    std::vector<int32_t> deepest_full(deepest_nb), deepest_swa(deepest_nb);
    if (!read_node_lanes(remote, deepest_meta.data_offset, deepest_nb,
                         deepest_hashes.data(), deepest_full.data(),
                         deepest_swa.data()))
        return -1;

    if (deepest_pos >= n || deepest_hashes[0] != hashes[deepest_pos])
        return -1;   // stale deepest hint

    // Phase 4: Build path from root to deepest, gap-filling between hints
    struct PathEntry {
        uint32_t node_id;
        TreeNodeData meta;
    };

    auto get_meta = [&](uint32_t nid) -> TreeNodeData {
        auto it = cache.find(nid);
        if (it != cache.end()) return it->second;
        return read_node(remote, nid);
    };

    // Walk from 'start' upward until parent == stop_parent.
    // Returns top-down segment. Empty = couldn't reach stop_parent (stale).
    auto fill_to = [&](uint32_t start, uint32_t stop_parent)
        -> std::vector<PathEntry> {
        std::vector<PathEntry> seg;
        uint32_t cur = start;
        for (int lim = 256; lim > 0 && cur != stop_parent
                            && cur != INVALID_NODE_ID; lim--) {
            TreeNodeData m = get_meta(cur);
            seg.push_back({cur, m});
            cur = m.parent_id;
        }
        if (cur != stop_parent) return {};
        std::reverse(seg.begin(), seg.end());
        return seg;
    };

    std::vector<PathEntry> path;

    // Segment 0: hints[0] → root
    {
        auto seg = fill_to(hints[0].tree_node_id, remote.root_node_id);
        if (seg.empty())
            return -1;
        path = std::move(seg);
    }

    // Segments 1..deepest_idx: check parent linkage, fill gaps if broken
    for (int h = 1; h <= deepest_idx; h++) {
        auto seg = fill_to(hints[h].tree_node_id,
                           hints[h - 1].tree_node_id);
        if (seg.empty())
            return -1;
        path.insert(path.end(), seg.begin(), seg.end());
    }

    // Phase 4.5: batch-read the three-lane payload of every path node in
    // one rdma_read_multi; overflow past the buffer reads per-node below.
    struct Slice { size_t idx; uint32_t nb; size_t off; uint32_t bytes; };
    std::vector<Slice> slices;
    size_t cursor = 0;
    for (size_t i = 0; i < path.size(); i++) {
        int nb = (int)path[i].meta.num_blocks;
        uint64_t bytes = (uint64_t)data_block_bytes(nb, remote_swa_lane(remote));
        if (nb <= 0 ||
            !payload_read_ok(remote, path[i].meta.data_offset, nb, bytes))
            return -1;   // stale/garbage node on the stitched path
        if (cursor + bytes > local_buf_size_) break;  // staging buffer full
        slices.push_back({i, (uint32_t)nb, cursor, (uint32_t)bytes});
        cursor += bytes;
    }
    // Issue the batched reads (rdma_read_multi self-chunks to SQ depth).
    if (!slices.empty()) {
        std::vector<RdmaReadOp> ops(slices.size());
        for (size_t k = 0; k < slices.size(); k++) {
            ops[k].local_buf   = (char*)local_buf_ + slices[k].off;
            ops[k].remote_addr = data_addr(remote, path[slices[k].idx].meta.data_offset);
            ops[k].remote_rkey = remote.rkey;
            ops[k].length      = slices[k].bytes;
        }
        ep_.rdma_read_multi(ops.data(), (int)ops.size(), local_lkey_);
        read_count_ += (int)ops.size();
    }

    // Phase 5: Match hashes top-down, emitting one ComponentPathNode per node
    out.reserve(path.size());
    for (size_t i = 0; i < path.size(); i++) {
        auto& entry = path[i];

        int nb = (int)entry.meta.num_blocks;
        ComponentPathNode pn;
        pn.node_id = entry.node_id;
        pn.absolute_start = (uint32_t)matched;
        pn.meta = entry.meta;
        pn.hashes.resize(nb);
        pn.full.resize(nb);
        pn.swa.resize(nb);

        // Need data — use cache or read
        if (i < slices.size()) {
            const char* base = (char*)local_buf_ + slices[i].off;
            const size_t hb = (size_t)slices[i].nb * sizeof(uint64_t);
            const size_t lb = (size_t)slices[i].nb * sizeof(int32_t);
            memcpy(pn.hashes.data(), base, hb);
            memcpy(pn.full.data(), base + hb, lb);
            if (remote_swa_lane(remote))
                memcpy(pn.swa.data(), base + hb + lb, lb);
            else
                std::fill(pn.swa.begin(), pn.swa.end(), INVALID_COMPONENT_SLOT);
        } else if (!read_node_lanes(remote, entry.meta.data_offset, nb,
                                    pn.hashes.data(), pn.full.data(),
                                    pn.swa.data())) {
            return -1;
        }

        int cmp = std::min(nb, n - matched);
        int ml = 0;
        for (int b = 0; b < cmp; b++) {
            if (pn.hashes[b] == hashes[matched + b])
                ml++;
            else
                break;
        }

        if (ml == 0 || (ml < nb && i + 1 < path.size())) {
            return -1;
        }
        pn.match_len = ml;
        out.push_back(std::move(pn));

        matched += ml;
        if (ml < nb)
            return 1;   // clean partial terminus at the deepest
    }
    return 1;   // the route chain ends here; no top-down descent past it
}

RemoteComponentWalkResult RdmaRemoteWalker::walk_components(
        const RemoteTreeInfo& remote, const uint64_t* hashes, int n,
        uint32_t mask, bool lock, int incumbent_common_hit,
        int local_full_hit) {
    return walk_components_impl(remote, hashes, n, mask, lock,
                                incumbent_common_hit, local_full_hit,
                                nullptr, 0);
}

RemoteComponentWalkResult RdmaRemoteWalker::walk_components_batched(
        const RemoteTreeInfo& remote, const uint64_t* hashes, int n,
        uint32_t mask, bool lock, int incumbent_common_hit,
        int local_full_hit, const RemoteHit* hints, int num_hints) {
    return walk_components_impl(remote, hashes, n, mask, lock,
                                incumbent_common_hit, local_full_hit,
                                hints, num_hints);
}

RemoteComponentWalkResult RdmaRemoteWalker::walk_components_impl(
        const RemoteTreeInfo& remote, const uint64_t* hashes, int n,
        uint32_t mask, bool lock, int incumbent_common_hit, int local_full_hit,
        const RemoteHit* hints, int num_hints) {
    read_count_ = 0;
    atomic_count_ = 0;
    auto error_result = [&](RemoteStatus st) {
        RemoteComponentWalkResult r;
        r.tree.status = st;
        r.rdma_read_count = read_count_;
        r.rdma_atomic_count = atomic_count_;
        return r;
    };

    // Same fail-closed gate as the local tree, evaluated against the REMOTE
    // config; component walks additionally require validated component info.
    if (!remote.component_info_valid || mask == 0 ||
        (mask & ~COMPONENT_MASK_ALL) != 0 ||
        (mask & remote.component_config.component_mask) != mask ||
        n < 0 || (n > 0 && hashes == nullptr) ||
        incumbent_common_hit < 0 || local_full_hit < 0)
        return error_result(RemoteStatus::BAD_REQUEST);
    if (n == 0)
        return error_result(RemoteStatus::OK);
    if (!rdma_read_lock(remote))
        return error_result(RemoteStatus::TIMEOUT);

    std::vector<ComponentPathNode> nodes;
    int matched = 0;
    if (hints && num_hints > 0) {
        if (build_path_from_hints(remote, hashes, n, incumbent_common_hit,
                                  hints, num_hints, nodes, matched) <= 0) {
            nodes.clear();
            matched = 0;
        }
    } else {
        matched = build_path_by_descent(remote, hashes, n, nodes,
                                        remote.root_node_id, 0);
    }
    std::vector<MatchedNodeSpanView> spans;
    spans.reserve(nodes.size());
    for (const ComponentPathNode& pn : nodes)
        spans.push_back(MatchedNodeSpanView{
            pn.node_id, pn.absolute_start, pn.meta.num_blocks,
            (uint32_t)pn.match_len, pn.hashes.data(), pn.full.data(),
            pn.swa.data(), pn.meta.mamba_checkpoint_slot});

    ComponentQueryContext ctx;
    ctx.config = remote.component_config;
    ctx.mask = mask;
    ctx.path = MatchedPathView{spans.data(), (uint32_t)spans.size(),
                               (uint32_t)matched};
    TreeComponentQueryResult tree = ComponentSet().query(ctx);
    PinRecord rp;

    // Pin only when the remote result actually extends the incumbent;
    // Full covers the continuation suffix [min(f,L), L) whole-node.
    if (lock && tree.status == RemoteStatus::OK &&
        (int)tree.common_hit > incumbent_common_hit) {
        const uint32_t L = tree.common_hit;
        const uint32_t c = std::min<uint32_t>((uint32_t)local_full_hit, L);
        // Shared coverage math (with a hit, tree.swa_start < L iff SWA
        // slots were accepted, so the local guard applies unchanged).
        PinCoverage cov = compute_pin_coverage(
            spans.data(), spans.size(), mask, L, tree.swa_start, c, L);

        // Refuse aux pins whose CST target falls outside the MR; build
        // the release record before the first atomic.
        if ((cov.swa_from < spans.size() || cov.pin_mamba) &&
            remote.mr.length != 0 &&
            cst_addr(remote, spans[cov.aux_k].node_id) +
                    remote.component_state_entry_size >
                remote.base_addr + remote.mr.length) {
            rdma_read_unlock(remote);
            return error_result(RemoteStatus::INTERNAL);
        }
        record_pin_tail(spans.data(), spans.size(), cov, L, L, rp);
        if (cov.swa_from < spans.size())
            rp.swa = {spans[cov.swa_from].absolute_start, cov.aux_cover_end};
        if (cov.pin_mamba)
            rp.mamba_end = (int32_t)L;

        auto pin1 = [&](uint64_t addr) {
            ep_.rdma_fetch_add(addr, remote.rkey, 1, local_buf_, local_lkey_);
            atomic_count_++;
        };
        if (cov.has_full)
            for (size_t i = cov.full_from; i <= cov.full_k; i++)
                pin1(ref_count_addr(remote, spans[i].node_id));
        for (size_t i = cov.swa_from; i <= cov.aux_k && i < spans.size(); i++)
            pin1(cst_addr(remote, spans[i].node_id) + SWA_REF_OFF);
        if (cov.pin_mamba)
            pin1(cst_addr(remote, spans[cov.aux_k].node_id) + MAMBA_REF_OFF);
    }
    rdma_read_unlock(remote);
    return RemoteComponentWalkResult{std::move(tree), std::move(rp),
                                     read_count_, atomic_count_};
}

bool RdmaRemoteWalker::dec_ref_components_remote(const RemoteTreeInfo& remote,
                                                 const uint64_t* hashes, int n,
                                                 PinRecord&& pin) {
    read_count_ = 0;
    atomic_count_ = 0;

    if (n <= 0 || hashes == nullptr) return false;
    if (pin.empty() || !remote.component_info_valid) return false;
    const bool full_present = !pin.full.empty();
    const bool swa_present = !pin.swa.empty();
    const bool mamba_present = pin.mamba_end > 0;
    if (full_present && pin.full.start > pin.full.end) return false;
    if (swa_present && pin.swa.start >= pin.swa.end) return false;
    int end = 0;
    if (full_present) end = (int)pin.full.end;
    if (swa_present && (int)pin.swa.end > end) end = (int)pin.swa.end;
    if (mamba_present && (int)pin.mamba_end > end) end = (int)pin.mamba_end;
    if (end <= 0) return false;
    // Caller keys plus the record's tail must reach the deepest lane END;
    // the tail holds the terminal's blocks past the caller's keys, whichever
    // lane pinned that node. The descent below indexes k[matched] for
    // matched < end, so `end` keys are exactly what it needs.
    const int tail = (int)pin.tail.size();
    if (tail > end) return false;
    const int need = end - tail;
    if (need > n) return false;
    std::vector<uint64_t> keys;
    const uint64_t* k = hashes;
    if (tail > 0) {
        keys.assign(hashes, hashes + need);
        keys.insert(keys.end(), pin.tail.begin(), pin.tail.end());
        k = keys.data();
    }

    if (!rdma_read_lock(remote)) return false;

    // Legacy trust model: descend by hash keys exactly like the local
    // RadixShmemTree::dec_ref and decrement unconditionally inside each lane's
    // recorded range.
    auto dec1 = [&](uint64_t addr) {
        ep_.rdma_fetch_add(addr, remote.rkey, (uint64_t)-1LL, local_buf_,
                           local_lkey_);
        atomic_count_++;
    };
    uint32_t parent_id = remote.root_node_id;
    int matched = 0;
    // An FAA outside the peer's MR kills the QP before the unlock and leaks its reader count.
    const uint64_t mr_end = remote.base_addr + remote.mr.length;
    auto cst_in_mr = [&](uint32_t node_id) {
        return remote.mr.length != 0 &&
               cst_addr(remote, node_id) + remote.component_state_entry_size <= mr_end;
    };

    while (matched < end) {
        // Descend to the child keyed by k[matched].
        uint32_t child_id = ht_lookup(remote, parent_id, k[matched]);
        if (child_id == INVALID_NODE_ID) break;
        if (child_id >= remote.node_pool_capacity) {
            fprintf(stderr, "[RdmaRemoteWalker] release: node id %u beyond the "
                    "peer's node pool (%u); stopping the descent\n",
                    child_id, remote.node_pool_capacity);
            break;
        }

        TreeNodeData child = read_node(remote, child_id);
        int nb = (int)child.num_blocks;
        int node_end = matched + nb;

        // Dec whole node if it starts in [start, end); tail partial node
        // too (symmetric with inc). Each lane applies its own window.
        if (full_present && matched >= (int)pin.full.start &&
            matched < (int)pin.full.end)
            dec_ref_node(remote, child_id);
        const bool want_swa = swa_present && matched >= (int)pin.swa.start &&
                              matched < (int)pin.swa.end;
        const bool want_mamba = mamba_present && node_end == (int)pin.mamba_end;
        if ((want_swa || want_mamba) && !cst_in_mr(child_id)) {
            fprintf(stderr, "[RdmaRemoteWalker] release: component state of node "
                    "%u lies outside the peer's MR; skipping its lane refcounts\n",
                    child_id);
        } else {
            if (want_swa)
                dec1(cst_addr(remote, child_id) + SWA_REF_OFF);
            if (want_mamba)
                dec1(cst_addr(remote, child_id) + MAMBA_REF_OFF);
        }

        matched = node_end;
        parent_id = child_id;

        // Once we're past the user's window there's nothing more to do.
        if (matched >= end) break;
    }

    rdma_read_unlock(remote);
    return true;
}

} // namespace radixshmem
