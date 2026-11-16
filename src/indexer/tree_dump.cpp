// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// dump_tree(): lock-free BFS copy of one attached RadixShmemTree for the
// monitor. Like check_health() it never takes a lock and never writes shm, so
// a killed monitor cannot leave the tree's rwlock held. Every id, offset and
// child entry is bounds-checked before use; what fails a check is counted in
// `torn` and skipped with its subtree.

#include "tree_dump.h"
#include "radix_tree.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace radixshmem {

// Friend of RadixShmemTree, declared in radix_tree.h.
class TreeDumper {
public:
    static TreeDump run(const RadixShmemTree& t, uint32_t max_nodes, uint64_t max_blocks) {
        if (!t.base_) throw std::runtime_error("dump_tree: no tree attached");
        TreeDump d;
        const ShmHeader* hdr = t.header();
        d.now = (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
        d.nodes_in_pool = hdr->node_pool_size.load(std::memory_order_relaxed);
        d.total_blocks = hdr->total_blocks.load(std::memory_order_relaxed);
        d.block_size = hdr->block_size;
        d.register_chunk_size = hdr->register_chunk_size;
        d.reset_generation = t.reset_generation();
        d.is_distributed = hdr->is_distributed != 0;
        d.rank = hdr->my_rank;
        d.node_name.assign(hdr->node_name, strnlen(hdr->node_name, sizeof(hdr->node_name)));
        d.cluster_id.assign(hdr->cluster_id, strnlen(hdr->cluster_id, sizeof(hdr->cluster_id)));

        const uint32_t capacity = hdr->node_pool_capacity;
        const uint32_t entry_cap = hdr->global_ht_entry_capacity;
        const auto* entries = reinterpret_cast<const ChainHTEntry*>(
            (const char*)t.base_ + hdr->global_ht_offset
            + (uint64_t)hdr->global_ht_bucket_count * sizeof(uint32_t));
        const uint64_t data_lo = hdr->data_pool_offset;
        const uint64_t data_hi = std::min<uint64_t>(data_lo + hdr->data_pool_total_size, t.size_);
        const bool swa_lane = t.data_.has_swa_lane();
        if (max_nodes == 0) max_nodes = capacity;

        std::vector<uint64_t> seen((capacity + 63) / 64, 0);
        std::vector<uint32_t> ids;   // node id per dump entry
        auto push = [&](uint32_t id, int32_t parent, uint32_t level, uint32_t depth) {
            seen[id >> 6] |= 1ull << (id & 63);
            ids.push_back(id);
            d.parent.push_back(parent);
            d.level.push_back(level);
            d.depth.push_back(depth);
        };
        if (hdr->root_node_id >= capacity) throw std::runtime_error("dump_tree: bad root id");
        push(hdr->root_node_id, -1, 0, 0);

        for (size_t i = 0; i < ids.size(); i++) {
            const TreeNode* node = t.nodes_.get(ids[i]);
            uint32_t nb = node->num_blocks;
            const uint64_t off = node->data_offset;
            bool torn = nb > (uint32_t)BuddyHeader::MAX_DATA_BLOCKS ||
                        (nb > 0 && (off == INVALID_OFFSET || off < data_lo ||
                                    off + data_block_bytes((int)nb, swa_lane) > data_hi));
            if (torn) {
                d.torn++;
                nb = 0;
            }
            if (max_blocks && d.hashes.size() + nb > max_blocks) {
                // Keep the nodes copied so far; the queued ones have no payload yet.
                d.truncated = true;
                ids.resize(i);
                d.parent.resize(i);
                d.level.resize(i);
                d.depth.resize(i);
                break;
            }
            const size_t h0 = d.hashes.size();
            d.hashes.resize(h0 + nb);
            uint32_t swa = 0;
            if (nb > 0) {
                std::memcpy(&d.hashes[h0], t.data_.get_hashes(off), nb * sizeof(uint64_t));
                if (const int32_t* ss = t.data_.get_swa_slots(off, (int)nb))
                    for (uint32_t b = 0; b < nb; b++)
                        swa += ss[b] != INVALID_COMPONENT_SLOT;
            }
            int64_t refs = node->local_ref_count.load(std::memory_order_relaxed) +
                           node->remote_ref_count.load(std::memory_order_relaxed);
            if (const NodeComponentState* cs = t.nodes_.component_state(ids[i])) {
                refs += cs->local_swa_ref_count.load(std::memory_order_relaxed) +
                        cs->remote_swa_ref_count.load(std::memory_order_relaxed);
                refs += cs->local_mamba_ref_count.load(std::memory_order_relaxed) +
                        cs->remote_mamba_ref_count.load(std::memory_order_relaxed);
            }
            d.num_blocks.push_back(nb);
            d.swa_blocks.push_back(swa);
            d.ref_count.push_back(refs);
            d.last_access.push_back(node->last_access_time.load(std::memory_order_relaxed));
            d.mamba.push_back(node->mamba_checkpoint_slot != INVALID_COMPONENT_SLOT);

            uint32_t e = torn ? HT_END : node->first_child_entry;
            uint32_t prev = HT_END;
            for (uint32_t steps = 0; e != HT_END; steps++) {
                if (e >= entry_cap || steps >= entry_cap) { d.torn++; break; }
                const ChainHTEntry ent = entries[e];
                const uint32_t back = prev;
                prev = e;
                e = ent.sibling_next;
                const uint32_t c = ent.child_node_id;
                if (ent.sibling_prev != back || c >= capacity || (seen[c >> 6] >> (c & 63)) & 1 ||
                    t.nodes_.get(c)->parent_id != ids[i]) {
                    d.torn++;
                    continue;
                }
                if (ids.size() >= max_nodes) { d.truncated = true; break; }
                push(c, (int32_t)i, d.level[i] + 1, d.depth[i] + nb);
            }
        }
        return d;
    }
};

TreeDump dump_tree(const RadixShmemTree& tree, uint32_t max_nodes, uint64_t max_blocks) {
    return TreeDumper::run(tree, max_nodes, max_blocks);
}

} // namespace radixshmem
