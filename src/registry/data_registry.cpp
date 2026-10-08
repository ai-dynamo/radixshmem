// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "data_registry.h"

#ifdef RADIXSHMEM_HAS_ETCD

#include "etcd_client.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace radixshmem {

namespace {

std::string encode_entry(const DataPlaneEntry& e) {
    return etcd::b64_encode(&e, sizeof(e));
}

bool layout_valid(uint32_t mask, const DataPlanePool* pools) {
    if ((mask & COMPONENT_MASK_FULL) == 0 || (mask & ~COMPONENT_MASK_ALL) != 0)
        return false;
    for (int i = 0; i < NUM_COMPONENT_TYPES; ++i) {
        const DataPlanePool& p = pools[i];
        if (mask & (1u << i)) {
            if (p.num_slots == 0 || p.slot_bytes == 0) return false;
        } else if (p.num_slots != 0 || p.slot_bytes != 0 || p.data_offset != 0) {
            return false;
        }
    }
    return true;
}

bool decode_entry(const std::string& b64, DataPlaneEntry& out) {
    const int n = etcd::b64_decode(b64.data(), b64.size(), &out, sizeof(out));
    return n == (int)sizeof(out) &&
           out.magic == DATA_PLANE_ENTRY_MAGIC &&
           out.version == DATA_PLANE_ENTRY_VERSION &&
           layout_valid(out.pool_mask, out.pools) &&
           // Callers read segment as a C string; an unterminated one would run
           // off the end. make_data_plane_entry always terminates, so this only
           // ever rejects a foreign or corrupted blob.
           std::memchr(out.segment, '\0', DATA_PLANE_SEGMENT_LEN) != nullptr;
}

// "radix/<id>/data/nodeA" -> "nodeA"
std::string node_of(const std::string& key) {
    const auto pos = key.rfind('/');
    return pos == std::string::npos ? key : key.substr(pos + 1);
}

} // namespace

bool make_data_plane_entry(const StoreGeometry& layout, const std::string& segment,
                           DataPlaneEntry& out) {
    if (segment.size() >= DATA_PLANE_SEGMENT_LEN) return false;
    DataPlaneEntry e;
    std::memset(&e, 0, sizeof(e));
    e.magic     = DATA_PLANE_ENTRY_MAGIC;
    e.version   = DATA_PLANE_ENTRY_VERSION;
    e.pool_mask = layout.pool_mask;
    for (int i = 0; i < NUM_COMPONENT_TYPES; ++i) {
        if (!(layout.pool_mask & (1u << i))) continue;
        e.pools[i].num_slots   = layout.pools[i].num_slots;
        e.pools[i].slot_bytes  = layout.pools[i].slot_bytes;
        e.pools[i].data_offset = layout.pools[i].base;
    }
    if (!layout_valid(e.pool_mask, e.pools)) return false;
    std::memcpy(e.segment, segment.data(), segment.size());
    out = e;
    return true;
}

bool make_data_plane_entry(uint64_t slot_bytes, uint64_t num_slots,
                           const std::string& segment, DataPlaneEntry& out) {
    StoreGeometry g;
    g.pool_mask = COMPONENT_MASK_FULL;
    g[ComponentType::FULL] = PoolGeometry{0, num_slots, slot_bytes};
    return make_data_plane_entry(g, segment, out);
}

StoreGeometry data_plane_layout(const DataPlaneEntry& e) {
    StoreGeometry g;
    g.pool_mask = e.pool_mask;
    for (int i = 0; i < NUM_COMPONENT_TYPES; ++i) {
        if (!(e.pool_mask & (1u << i))) continue;
        g.pools[i].base       = e.pools[i].data_offset;
        g.pools[i].num_slots  = e.pools[i].num_slots;
        g.pools[i].slot_bytes = e.pools[i].slot_bytes;
    }
    return g;
}

struct DataPlaneRegistry::Impl {
    etcd::Client cli;
    etcd::Keys   keys;
};

DataPlaneRegistry::DataPlaneRegistry(const std::string& registry,
                                     const std::string& cluster_id)
    : impl_(new Impl) {
    impl_->keys.cluster_id = cluster_id;
    impl_->cli.init(etcd::strip_scheme(registry));
}

DataPlaneRegistry::~DataPlaneRegistry() = default;

void DataPlaneRegistry::publish(const std::string& node, const DataPlaneEntry& e,
                                int64_t lease) {
    impl_->cli.put(impl_->keys.data(node), encode_entry(e), lease);
}

void DataPlaneRegistry::unpublish(const std::string& node) {
    // Exact key: delete_prefix here would take nodeB down with nodeB2.
    impl_->cli.del(impl_->keys.data(node));
}

bool DataPlaneRegistry::lookup(const std::string& node, DataPlaneEntry& out,
                               int timeout_sec) {
    std::string val;
    auto read_once = [&] { return impl_->cli.get(impl_->keys.data(node), val); };
    if (timeout_sec > 0) {
        if (!etcd::poll_until(timeout_sec, read_once)) return false;
    } else if (!read_once()) {
        return false;
    }
    if (!decode_entry(val, out)) {
        fprintf(stderr, "[radixshmem] DataPlaneRegistry: node '%s' published a blob "
                        "this build cannot decode (wrong version, or malformed "
                        "— e.g. an unterminated segment name)\n", node.c_str());
        return false;
    }
    return true;
}

std::vector<std::pair<std::string, DataPlaneEntry>>
DataPlaneRegistry::list(int* skipped) {
    std::vector<std::pair<std::string, DataPlaneEntry>> out;
    int bad = 0;
    for (auto& kv : impl_->cli.get_prefix(impl_->keys.data_prefix())) {
        DataPlaneEntry e{};
        if (!decode_entry(kv.value, e)) { ++bad; continue; }
        out.emplace_back(node_of(kv.key), e);
    }
    std::sort(out.begin(), out.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    if (skipped) *skipped = bad;
    return out;
}

std::string DataPlaneRegistry::node_of_rank(int rank) {
    std::string node;
    if (rank < 0 || !impl_->cli.get(impl_->keys.rank_slot(rank), node)) return {};
    return node;
}

bool DataPlaneRegistry::validate_uniform(const DataPlaneEntry& local,
                                         std::string& who, std::string& why) {
    who.clear();
    why.clear();

    int bad = 0;
    auto all = list(&bad);
    if (bad > 0) {
        why = "a peer published " + std::to_string(bad) +
              " entry/entries this build cannot decode";
        return false;
    }

    // Invariants: pool set and per-pool stride. Counts/offsets may differ.
    for (auto& [node, e] : all) {
        if (e.pool_mask != local.pool_mask) {
            who = node;
            why = "pool mask " + std::to_string(e.pool_mask) +
                  " != ours " + std::to_string(local.pool_mask) +
                  " (different KV components stored)";
            return false;
        }
        for (int i = 0; i < NUM_COMPONENT_TYPES; ++i) {
            if (!(local.pool_mask & (1u << i))) continue;
            if (e.pools[i].slot_bytes != local.pools[i].slot_bytes) {
                who = node;
                why = std::string(component_type_name(static_cast<ComponentType>(i))) +
                      " slot stride " + std::to_string(e.pools[i].slot_bytes) +
                      " != ours " + std::to_string(local.pools[i].slot_bytes);
                return false;
            }
        }
    }
    return true;
}

} // namespace radixshmem

#endif // RADIXSHMEM_HAS_ETCD
