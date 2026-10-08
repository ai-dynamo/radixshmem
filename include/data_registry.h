// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "radixshmem_config.h"

#ifdef RADIXSHMEM_HAS_ETCD

#include "component_kind.h"
#include "slot_geometry.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace radixshmem {

inline constexpr uint32_t DATA_PLANE_ENTRY_MAGIC   = 0x52584450;  // "RXDP"
inline constexpr uint32_t DATA_PLANE_ENTRY_VERSION = 2;           // 2: per-component pools
inline constexpr uint32_t DATA_PLANE_SEGMENT_LEN   = 128;

// One pool; data_offset relative to the published segment (SlotStore::data_base()).
struct DataPlanePool {
    uint64_t num_slots;    // 0 = absent
    uint64_t slot_bytes;   // stride; must be uniform cluster-wide per pool
    uint64_t data_offset;
};

// One node's data-plane geometry. POD, base64'd whole into etcd.
struct DataPlaneEntry {
    uint32_t magic;
    uint32_t version;
    uint32_t pool_mask;    // COMPONENT_MASK_* of the pools present
    uint32_t _pad;
    DataPlanePool pools[NUM_COMPONENT_TYPES];   // indexed by ComponentType
    // Transfer backend's local_segment(); empty if none; NUL-terminated.
    char     segment[DATA_PLANE_SEGMENT_LEN];
};
static_assert(sizeof(DataPlaneEntry) == 16 + 24 * NUM_COMPONENT_TYPES + DATA_PLANE_SEGMENT_LEN,
              "DataPlaneEntry layout changed — bump DATA_PLANE_ENTRY_VERSION");

// False (nothing written) on an over-long segment name or a malformed layout
// (no FULL pool, unknown bits, present pool with a zero field).
// `layout` is SlotStore::layout().
bool make_data_plane_entry(const StoreGeometry& layout, const std::string& segment,
                           DataPlaneEntry& out);

// FULL-only store of num_slots x slot_bytes at offset 0.
bool make_data_plane_entry(uint64_t slot_bytes, uint64_t num_slots,
                           const std::string& segment, DataPlaneEntry& out);

// Published layout, for SlotTransfer::attach_peer(peer, &layout).
StoreGeometry data_plane_layout(const DataPlaneEntry& e);

// Reads and writes radix/<cluster_id>/data/. Shares the process-global etcd
// connection, so one next to a RadixServer/RadixClient costs no second one.
class DataPlaneRegistry {
public:
    // `registry` is the indexer's bootstrap endpoint, scheme optional
    // ("etcd://host:2379" or "host:2379"). Throws if etcd is unreachable.
    DataPlaneRegistry(const std::string& registry, const std::string& cluster_id);
    ~DataPlaneRegistry();

    DataPlaneRegistry(const DataPlaneRegistry&) = delete;
    DataPlaneRegistry& operator=(const DataPlaneRegistry&) = delete;

    void publish(const std::string& node, const DataPlaneEntry& e,
                 int64_t lease = 0);

    void unpublish(const std::string& node);

    bool lookup(const std::string& node, DataPlaneEntry& out,
                int timeout_sec = 0);

    std::vector<std::pair<std::string, DataPlaneEntry>> list(int* skipped = nullptr);

    std::string node_of_rank(int rank);

    bool validate_uniform(const DataPlaneEntry& local, std::string& who,
                          std::string& why);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace radixshmem

#endif // RADIXSHMEM_HAS_ETCD
