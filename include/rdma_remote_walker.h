// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "component_types.h"
#include "rdma_transport.hpp"
#include "types.h"
#include <cstdint>
#include <vector>

namespace radixshmem {

struct RemoteTreeInfo {
    RdmaMrInfo mr;             // remote memory region (shm base + size)
    uint32_t rkey;             // for convenience
    uint64_t base_addr;        // shm base address on remote

    // Pool offsets (read from ShmHeader once)
    uint64_t node_pool_offset;
    uint32_t node_pool_capacity;   // remote NodePool slot count; node_id must be < this
    uint64_t data_pool_offset;
    uint64_t global_ht_offset;
    uint32_t global_ht_bucket_count;
    uint32_t global_ht_entry_capacity;
    uint32_t root_node_id;

    uint64_t rdma_rwlock_addr;        // remote ShmHeader::rdma_write_flag (CPU-owned, RDMA-read only)
    uint64_t rdma_reader_count_addr;  // remote ShmHeader::rdma_reader_count (NIC FAA only)

    ComponentConfig component_config{};
    uint64_t component_state_offset = 0;
    uint32_t component_state_entry_size = 0;
    bool component_info_valid = false;
};

struct TreeNodeData {
    uint32_t parent_id;
    uint32_t num_blocks;
    uint32_t children_count;
    int32_t  mamba_checkpoint_slot;
    uint64_t data_offset;
    uint64_t last_access_time;
    int64_t ref_count;
    uint32_t next_free;
    uint32_t leaf_prev;
    uint32_t leaf_next;
    uint32_t first_child_entry;
    uint64_t _pad_version;
};
static_assert(sizeof(TreeNodeData) == sizeof(TreeNode),
              "TreeNodeData must match TreeNode layout");

struct RemoteHit {
    int position;
    uint32_t tree_node_id;
};

struct RemoteComponentWalkResult {
    TreeComponentQueryResult tree;
    PinRecord remote_pin;
    int rdma_read_count = 0;
    int rdma_atomic_count = 0;
};

class RdmaRemoteWalker {
public:
    RdmaRemoteWalker(IRdmaEndpoint& ep, void* local_buf,
                     uint32_t local_lkey, size_t buf_size);

    RemoteTreeInfo read_remote_info(const RdmaMrInfo& remote_mr);

    void dec_ref_node(const RemoteTreeInfo& remote, uint32_t node_id);

    RemoteComponentWalkResult walk_components(
        const RemoteTreeInfo& remote, const uint64_t* hashes, int n,
        uint32_t mask, bool lock, int incumbent_common_hit = 0,
        int local_full_hit = 0);
    RemoteComponentWalkResult walk_components_batched(
        const RemoteTreeInfo& remote, const uint64_t* hashes, int n,
        uint32_t mask, bool lock, int incumbent_common_hit,
        int local_full_hit, const RemoteHit* hints, int num_hints);

    bool dec_ref_components_remote(const RemoteTreeInfo& remote,
                                   const uint64_t* hashes, int n,
                                   PinRecord&& pin);

private:
    TreeNodeData read_node(const RemoteTreeInfo& remote, uint32_t node_id);

    bool payload_read_ok(const RemoteTreeInfo& remote, uint64_t data_offset,
                         int num_blocks, uint64_t total) const;

    uint32_t ht_lookup(const RemoteTreeInfo& remote,
                       uint32_t parent_id, uint64_t token_hash);

    bool rdma_read_lock(const RemoteTreeInfo& remote);
    void rdma_read_unlock(const RemoteTreeInfo& remote);

    uint64_t node_addr(const RemoteTreeInfo& r, uint32_t node_id) const;
    uint64_t ht_bucket_addr(const RemoteTreeInfo& r, uint32_t bucket) const;
    uint64_t ht_entry_addr(const RemoteTreeInfo& r, uint32_t index) const;
    uint64_t data_addr(const RemoteTreeInfo& r, uint64_t offset) const;
    uint64_t ref_count_addr(const RemoteTreeInfo& r, uint32_t node_id) const;
    uint64_t cst_addr(const RemoteTreeInfo& r, uint32_t node_id) const;

    bool read_node_lanes(const RemoteTreeInfo& remote,
                         uint64_t data_offset, int num_blocks,
                         uint64_t* out_hashes, int32_t* out_full,
                         int32_t* out_swa);

    struct ComponentPathNode {
        uint32_t node_id;
        uint32_t absolute_start;
        TreeNodeData meta;
        int match_len;
        std::vector<uint64_t> hashes;
        std::vector<int32_t> full, swa;
    };

    RemoteComponentWalkResult walk_components_impl(
        const RemoteTreeInfo& remote, const uint64_t* hashes, int n,
        uint32_t mask, bool lock, int incumbent_common_hit,
        int local_full_hit, const RemoteHit* hints, int num_hints);
    int build_path_by_descent(const RemoteTreeInfo& remote,
                              const uint64_t* hashes, int n,
                              std::vector<ComponentPathNode>& out,
                              uint32_t anchor, int anchor_matched);

    int build_path_from_hints(const RemoteTreeInfo& remote,
                                  const uint64_t* hashes, int n,
                                  int incumbent,
                                  const RemoteHit* hints, int num_hints,
                                  std::vector<ComponentPathNode>& out,
                                  int& matched);

    IRdmaEndpoint& ep_;
    void* local_buf_;
    uint32_t local_lkey_;
    size_t local_buf_size_;

    int read_count_ = 0;
    int atomic_count_ = 0;
};

} // namespace radixshmem
