// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "hash.h"
#include "health_check.h"
#include "radix_client.h"
#include "radix_server.h"
#include "types.h"

#include "cluster_protocol.h"

#ifdef RADIXSHMEM_HAS_ETCD
#include "etcd_client.hpp"   // src/registry: the Go-wrapper etcd client, bound as _core.EtcdKv
#endif

#include <optional>

namespace py = pybind11;
using namespace radixshmem;

static py::array_t<int32_t> vec_to_npy_i32(const std::vector<int32_t>& v) {
    py::array_t<int32_t> arr(static_cast<py::ssize_t>(v.size()));
    std::memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(int32_t));
    return arr;
}


static py::dict peer_fault_dict(const PeerFaultCounters* c) {
    auto v = [&](const std::atomic<uint64_t> PeerFaultCounters::*f) {
        return c ? (c->*f).load(std::memory_order_relaxed) : 0;
    };
    py::dict d;
    d["peer_down_events"]    = v(&PeerFaultCounters::down_events);
    d["rht_writes_dropped"]  = v(&PeerFaultCounters::rht_writes);
    d["rht_removes_dropped"] = v(&PeerFaultCounters::rht_removes);
    d["rht_lookups_skipped"] = v(&PeerFaultCounters::rht_lookups);
    d["walks_skipped"]       = v(&PeerFaultCounters::walks);
    return d;
}

template <typename T>
static std::pair<const T*, int> npy_1d(const py::array_t<T>& a,
                                       const char* what) {
    auto buf = a.request();
    if (buf.ndim != 1)
        throw std::runtime_error(std::string(what) + " must be 1-D array");
    return { static_cast<const T*>(buf.ptr), static_cast<int>(buf.shape[0]) };
}

static py::array_t<uint64_t> py_compute_hashes(py::array_t<int64_t> token_ids,
                                                int tokens_per_block) {
    auto [ptr, len] = npy_1d(token_ids, "token_ids");
    std::vector<uint64_t> vec;
    {
        py::gil_scoped_release release;
        vec = RadixClient::compute_hashes(ptr, len, tokens_per_block);
    }
    py::array_t<uint64_t> arr(static_cast<py::ssize_t>(vec.size()));
    std::memcpy(arr.mutable_data(), vec.data(), vec.size() * sizeof(uint64_t));
    return arr;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "radixshmem: shared-memory radix tree for KV cache prefix matching";

    // ==================== Enums ====================

    py::enum_<InsertError>(m, "InsertError")
        .value("OK", InsertError::OK)
        .value("NODE_POOL_FULL", InsertError::NODE_POOL_FULL)
        .value("DATA_POOL_FULL", InsertError::DATA_POOL_FULL)
        .value("HT_POOL_FULL", InsertError::HT_POOL_FULL)
        .value("MEMPOOL_FULL", InsertError::MEMPOOL_FULL)
        .value("REMOTE_FAILED", InsertError::REMOTE_FAILED)
        .value("BAD_REQUEST", InsertError::BAD_REQUEST)
        .value("FULL_PATH_MISSING", InsertError::FULL_PATH_MISSING);

    py::enum_<EvictPolicy>(m, "EvictPolicy")
        .value("LRU", EvictPolicy::LRU);


    py::enum_<RemoteStatus>(m, "RemoteStatus")
        .value("OK", RemoteStatus::OK)
        .value("BAD_REQUEST", RemoteStatus::BAD_REQUEST)
        .value("UNKNOWN_OP", RemoteStatus::UNKNOWN_OP)
        .value("TIMEOUT", RemoteStatus::TIMEOUT)
        .value("BAD_REPLY", RemoteStatus::BAD_REPLY)
        .value("BAD_TARGET", RemoteStatus::BAD_TARGET)
        .value("INTERNAL", RemoteStatus::INTERNAL);


    m.attr("MAX_REMOTE_OP_BLOCKS") = MAX_REMOTE_OP_BLOCKS;
    // Largest register chunk in blocks (register_chunk_tokens / block_size) RadixShmemTree::create accepts.
    m.attr("MAX_REGISTER_CHUNK_SIZE") = BuddyHeader::MAX_DATA_BLOCKS;
    py::enum_<ComponentType>(m, "ComponentType")
        .value("FULL", ComponentType::FULL)
        .value("SWA", ComponentType::SWA)
        .value("MAMBA", ComponentType::MAMBA);

    // Create-time component ABI, read back from the shm header.
    py::class_<ComponentConfig>(m, "ComponentConfig")
        .def_readonly("abi_version", &ComponentConfig::abi_version)
        .def_readonly("component_mask", &ComponentConfig::component_mask)
        .def_readonly("logical_block_tokens", &ComponentConfig::logical_block_tokens)
        .def_readonly("swa_window_blocks", &ComponentConfig::swa_window_blocks)
        .def_readonly("mamba_state_bytes", &ComponentConfig::mamba_state_bytes)
        .def("__repr__", [](const ComponentConfig& c) {
            return "ComponentConfig(component_mask=" + std::to_string(c.component_mask) +
                   ", logical_block_tokens=" + std::to_string(c.logical_block_tokens) +
                   ", swa_window_blocks=" + std::to_string(c.swa_window_blocks) +
                   ", mamba_state_bytes=" + std::to_string(c.mamba_state_bytes) + ")";
        });

    m.attr("COMPONENT_MASK_FULL") = COMPONENT_MASK_FULL;
    m.attr("COMPONENT_MASK_SWA") = COMPONENT_MASK_SWA;
    m.attr("COMPONENT_MASK_MAMBA") = COMPONENT_MASK_MAMBA;
    m.attr("INVALID_COMPONENT_SLOT") = INVALID_COMPONENT_SLOT;

    // ==================== Config ====================

    py::class_<ShmConfig>(m, "ShmConfig")
        .def(py::init<>())
        .def(py::init([](uint32_t max_nodes, uint32_t max_blocks,
                         int block_size, int register_chunk_tokens,
                         float background_evict_ratio,
                         EvictPolicy evict_policy, float data_pool_ratio,
                         const std::string& hugepage_path,
                         uint32_t component_mask, uint32_t swa_window_blocks,
                         uint64_t mamba_state_bytes, uint32_t swa_max_blocks,
                         uint32_t mamba_max_slots) {
            ShmConfig c;
            c.max_nodes = max_nodes;
            c.max_blocks = max_blocks;
            c.block_size = block_size;
            c.register_chunk_tokens = register_chunk_tokens;
            c.background_evict_ratio = background_evict_ratio;
            c.evict_policy = evict_policy;
            c.data_pool_ratio = data_pool_ratio;
            c.hugepage_path = hugepage_path;
            c.component_mask = component_mask;
            c.swa_window_blocks = swa_window_blocks;
            c.mamba_state_bytes = mamba_state_bytes;
            c.swa_max_blocks = swa_max_blocks;
            c.mamba_max_slots = mamba_max_slots;
            return c;
        }),
        py::arg("max_nodes") = 0,
        py::arg("max_blocks") = 0,
        py::arg("block_size") = 1,
        py::arg("register_chunk_tokens") = DEFAULT_REGISTER_CHUNK_TOKENS,
        py::arg("background_evict_ratio") = 0.05f,
        py::arg("evict_policy") = EvictPolicy::LRU,
        py::arg("data_pool_ratio") = 4.0f,
        py::arg("hugepage_path") = "/dev/hugepages",
        py::arg("component_mask") = COMPONENT_MASK_FULL,
        py::arg("swa_window_blocks") = 0,
        py::arg("mamba_state_bytes") = 0,
        py::arg("swa_max_blocks") = 0,
        py::arg("mamba_max_slots") = 0)
        .def_readwrite("max_nodes", &ShmConfig::max_nodes)
        .def_readwrite("max_blocks", &ShmConfig::max_blocks)
        .def_readwrite("block_size", &ShmConfig::block_size)
        .def_readwrite("register_chunk_tokens", &ShmConfig::register_chunk_tokens)
        .def_readwrite("background_evict_ratio", &ShmConfig::background_evict_ratio)
        .def_readwrite("evict_policy", &ShmConfig::evict_policy)
        .def_readwrite("data_pool_ratio", &ShmConfig::data_pool_ratio)
        .def_readwrite("hugepage_path", &ShmConfig::hugepage_path)
        .def_readwrite("component_mask", &ShmConfig::component_mask)
        .def_readwrite("swa_window_blocks", &ShmConfig::swa_window_blocks)
        .def_readwrite("mamba_state_bytes", &ShmConfig::mamba_state_bytes)
        .def_readwrite("swa_max_blocks", &ShmConfig::swa_max_blocks)
        .def_readwrite("mamba_max_slots", &ShmConfig::mamba_max_slots)
        .def("effective_max_nodes", &ShmConfig::effective_max_nodes)
        .def("effective_register_chunk_size", &ShmConfig::effective_register_chunk_size,
             "register_chunk_tokens in blocks, what the tree is created with.")
        .def("effective_data_pool_bytes", &ShmConfig::effective_data_pool_bytes)
        .def("__repr__", [](const ShmConfig& c) {
            const char* policy = c.evict_policy == EvictPolicy::LRU ? "LRU" : "unknown";
            return "ShmConfig(max_nodes=" + std::to_string(c.max_nodes) +
                   ", max_blocks=" + std::to_string(c.max_blocks) +
                   ", block_size=" + std::to_string(c.block_size) +
                   ", register_chunk_tokens=" + std::to_string(c.register_chunk_tokens) +
                   ", background_evict_ratio=" + std::to_string(c.background_evict_ratio) +
                   ", evict_policy=" + std::string(policy) +
                   ", data_pool_ratio=" + std::to_string(c.data_pool_ratio) +
                   ", hugepage_path=" + c.hugepage_path +
                   ", component_mask=" + std::to_string(c.component_mask) +
                   ", swa_window_blocks=" + std::to_string(c.swa_window_blocks) +
                   ", mamba_state_bytes=" + std::to_string(c.mamba_state_bytes) +
                   ", swa_max_blocks=" + std::to_string(c.swa_max_blocks) +
                   ", mamba_max_slots=" + std::to_string(c.mamba_max_slots) + ")";
        });

    py::class_<RadixServerConfig>(m, "RadixServerConfig")
        .def(py::init<>())
        .def_readwrite("name", &RadixServerConfig::name)
        .def_readwrite("shm",  &RadixServerConfig::shm)
        .def_readwrite("rank", &RadixServerConfig::rank)
        .def_readwrite("world_size", &RadixServerConfig::world_size)
        .def_readwrite("registry", &RadixServerConfig::registry)
        .def_readwrite("cluster_id", &RadixServerConfig::cluster_id)
        .def_readwrite("node_name", &RadixServerConfig::node_name)
        // Distributed-mode bootstrap IP peers connect to (required when world_size>1;
        // "0.0.0.0" for a single-node test). rpc_interface resolves the IP from a NIC
        // and wins over rpc_address.
        .def_readwrite("rpc_address", &RadixServerConfig::rpc_address)
        .def_readwrite("rpc_interface", &RadixServerConfig::rpc_interface)
        // Bootstrap / XRC listen port; 0 = OS-assigned (the real port is published).
        .def_property("rpc_port",
                      [](const RadixServerConfig& c) { return (int)c.rpc_port; },
                      [](RadixServerConfig& c, int v) { c.rpc_port = (uint16_t)v; })
        .def_readwrite("index_dev", &RadixServerConfig::index_dev)
        .def_readwrite("gid_idx", &RadixServerConfig::gid_idx)
        .def_readwrite("bootstrap_timeout_sec",
                       &RadixServerConfig::bootstrap_timeout_sec)
        .def_readwrite("lease_ttl_sec", &RadixServerConfig::lease_ttl_sec)
        .def_readwrite("register_chunk_tokens",
                       &RadixServerConfig::register_chunk_tokens)
        // Membership gate: bootstrap waits for max(num_rht_shards, expected_min_nodes)
        // peers, stable for settle_ms. > 1 also enables distributed mode.
        .def_readwrite("expected_min_nodes",
                       &RadixServerConfig::expected_min_nodes)
        .def_readwrite("settle_ms", &RadixServerConfig::settle_ms)
        .def_readwrite("num_rht_shards", &RadixServerConfig::num_rht_shards)
        .def_readwrite("rht_shard_holders", &RadixServerConfig::rht_shard_holders)
        .def_readwrite("rht_transport", &RadixServerConfig::rht_transport)
        .def_readwrite("peer_index_transport",
                       &RadixServerConfig::peer_index_transport)
        .def_readwrite("rht_slots_per_bucket",
                       &RadixServerConfig::rht_slots_per_bucket)
        .def_readwrite("enable_remote_insert",
                       &RadixServerConfig::enable_remote_insert)
        .def_readwrite("enable_remote_query",
                       &RadixServerConfig::enable_remote_query)
        // Control-plane transport backend: "zmq" or "dc"; rank 0 authoritative.
        .def_readwrite("remote_op_transport",
                       &RadixServerConfig::remote_op_transport)
        .def_property("zmq_listen_port",
                      [](const RadixServerConfig& c) { return (int)c.zmq_listen_port; },
                      [](RadixServerConfig& c, int v) { c.zmq_listen_port = (uint16_t)v; });

    // ==================== Result types ====================

    // One-shot cleanup callable: r.finalize() runs once, later calls are no-ops.
    py::class_<FinalizeFn>(m, "FinalizeFn")
        .def("__call__", [](FinalizeFn& f) {
            py::gil_scoped_release rel;
            f();
        })
        .def("__bool__", [](const FinalizeFn& f) { return (bool)f; })
        .def("empty", &FinalizeFn::empty);

    py::class_<ComponentQueryResult>(m, "ComponentQueryResult")
        .def_readonly("status",       &ComponentQueryResult::status)
        .def_readonly("common_hit",   &ComponentQueryResult::common_hit)
        // Source-tagged Full runs sorted by offset, tiling
        // [0, common_hit): list of (source_rank, offset, slot_ids).
        .def_property_readonly("full_fragments", [](const ComponentQueryResult& r) {
            py::list runs;
            for (const auto& run : r.full_fragments)
                runs.append(py::make_tuple(run.source_rank,
                                           run.offset,
                                           vec_to_npy_i32(run.slot_ids)));
            return runs;
        })
        .def_property_readonly("swa_source_rank",
                               [](const ComponentQueryResult& r) {
            return r.swa.source_rank;
        })
        .def_property_readonly("swa_start", [](const ComponentQueryResult& r) {
            return r.swa.offset;
        })
        .def_property_readonly("swa_slots", [](const ComponentQueryResult& r) {
            return vec_to_npy_i32(r.swa.slot_ids);
        })
        // None when no checkpoint; else (source_rank, slot_id) at common_hit.
        .def_property_readonly("mamba", [](const ComponentQueryResult& r)
                                            -> py::object {
            if (!r.mamba) return py::none();
            return py::make_tuple(r.mamba->source_rank, r.mamba->slot_id);
        })
        .def_readonly("rdma_read_count",   &ComponentQueryResult::rdma_read_count)
        .def_readonly("rdma_atomic_count", &ComponentQueryResult::rdma_atomic_count)
        .def_property_readonly("finalize",
            [](ComponentQueryResult& r) -> FinalizeFn& { return r.finalize; },
            py::return_value_policy::reference_internal);

    py::class_<QueryResult>(m, "QueryResult")
        // Request-level transport status (remote_query fills the real one).
        .def_readonly("status",            &QueryResult::status)
        // Totals (local + remote when distributed)
        .def_readonly("total_hit_length",  &QueryResult::total_hit_length)
        // Local-tree details
        .def_readonly("local_hit_length",  &QueryResult::local_hit_length)
        // Local matched-prefix slots
        .def_property_readonly("prefix_slots", [](const QueryResult& r) {
            return vec_to_npy_i32(r.prefix_slots);
        })
        // Remote-tree details (single-node: remote_node_id == -1)
        .def_readonly("remote_node_id",       &QueryResult::remote_node_id)
        .def_readonly("remote_hit_length",    &QueryResult::remote_hit_length)
        .def_property_readonly("remote_slots", [](const QueryResult& r) {
            return vec_to_npy_i32(r.remote_slots);
        })
        // RDMA stats
        .def_readonly("rdma_read_count",      &QueryResult::rdma_read_count)
        .def_readonly("rdma_atomic_count",    &QueryResult::rdma_atomic_count)
        // One-shot cleanup handle; exposed by reference so __bool__ flips after call.
        .def_property_readonly("finalize",
            [](QueryResult& r) -> FinalizeFn& { return r.finalize; },
            py::return_value_policy::reference_internal);

    py::class_<InsertResult>(m, "InsertResult")
        .def_readonly("error", &InsertResult::error)
        .def_property_readonly("unused_slots", [](const InsertResult& r) {
            return vec_to_npy_i32(r.unused_slots);
        });

    // ==================== Health audit ====================

    py::enum_<HealthFlag>(m, "HealthFlag", py::arithmetic(),
                          "check_health() failure bits; 0 == healthy.")
        .value("OK",                HEALTH_OK)
        .value("RWLOCK_READ_HELD",  HEALTH_RWLOCK_READ_HELD)
        .value("RWLOCK_WRITE_HELD", HEALTH_RWLOCK_WRITE_HELD)
        .value("LRU_LOCK_HELD",     HEALTH_LRU_LOCK_HELD)
        .value("RDMA_LOCK_HELD",    HEALTH_RDMA_LOCK_HELD)
        .value("FULL_SLOT_LEAK",    HEALTH_FULL_SLOT_LEAK)
        .value("SWA_SLOT_LEAK",     HEALTH_SWA_SLOT_LEAK)
        .value("MAMBA_SLOT_LEAK",   HEALTH_MAMBA_SLOT_LEAK)
        .value("REF_LEAK",          HEALTH_REF_LEAK)
        .value("NODE_LEAK",         HEALTH_NODE_LEAK)
        .value("NOT_ATTACHED",      HEALTH_NOT_ATTACHED);

    py::class_<HealthReport::Pool>(m, "HealthPoolReport")
        .def_readonly("enabled",   &HealthReport::Pool::enabled)
        .def_readonly("total",     &HealthReport::Pool::total)
        .def_readonly("used",      &HealthReport::Pool::used)
        .def_readonly("in_tree",   &HealthReport::Pool::in_tree)
        .def_readonly("leaked",    &HealthReport::Pool::leaked)
        .def_readonly("dangling",  &HealthReport::Pool::dangling)
        .def_readonly("duplicate", &HealthReport::Pool::duplicate)
        .def("ok", &HealthReport::Pool::ok)
        .def("__repr__", [](const HealthReport::Pool& p) {
            return "HealthPoolReport(enabled=" + std::to_string(p.enabled) +
                   ", total=" + std::to_string(p.total) +
                   ", used=" + std::to_string(p.used) +
                   ", in_tree=" + std::to_string(p.in_tree) +
                   ", leaked=" + std::to_string(p.leaked) +
                   ", dangling=" + std::to_string(p.dangling) +
                   ", duplicate=" + std::to_string(p.duplicate) + ")";
        });

    py::class_<HealthReport>(m, "HealthReport")
        .def_readonly("flags",                  &HealthReport::flags)
        .def_readonly("locks_inspected",        &HealthReport::locks_inspected)
        .def_readonly("rwlock_readers",         &HealthReport::rwlock_readers)
        .def_readonly("rwlock_writers_waiting", &HealthReport::rwlock_writers_waiting)
        .def_readonly("rwlock_writer_held",     &HealthReport::rwlock_writer_held)
        .def_readonly("lru_locked",             &HealthReport::lru_locked)
        .def_readonly("rdma_rwlock",            &HealthReport::rdma_rwlock)
        .def_readonly("full",                   &HealthReport::full)
        .def_readonly("swa",                    &HealthReport::swa)
        .def_readonly("mamba",                  &HealthReport::mamba)
        .def_readonly("nodes_with_refs",        &HealthReport::nodes_with_refs)
        .def_readonly("ref_total",              &HealthReport::ref_total)
        .def_readonly("nodes_reachable",        &HealthReport::nodes_reachable)
        .def_readonly("nodes_in_pool",          &HealthReport::nodes_in_pool)
        .def("ok", &HealthReport::ok)
        .def("describe", &HealthReport::describe)
        .def("__bool__", &HealthReport::ok)
        .def("__repr__", [](const HealthReport& r) {
            return "HealthReport(flags=" + std::to_string(r.flags) + ")\n" + r.describe();
        });

    // ==================== RadixServer ====================

    py::class_<RadixServer>(m, "RadixServer")
        .def(py::init<const RadixServerConfig&>(), py::arg("config"))
        .def(py::init([](const std::string& name, const ShmConfig& shm) {
            RadixServerConfig cfg;
            cfg.name = name;
            cfg.shm  = shm;
            auto srv = std::make_unique<RadixServer>(cfg);
            srv->bootstrap();   // single-node form auto-boots
            return srv;
        }), py::arg("name"), py::arg("config") = ShmConfig{},
            "Convenience overload — single-node deployment. Auto-bootstraps.")
        .def("bootstrap", &RadixServer::bootstrap,
             py::call_guard<py::gil_scoped_release>())
        .def("is_distributed", &RadixServer::is_distributed)
        .def("rank", &RadixServer::rank)
        .def("world_size", &RadixServer::world_size)
        .def("shm_name", &RadixServer::shm_name)
        .def("node_name", &RadixServer::node_name)
        .def("lease_id", &RadixServer::lease_id)
        .def("total_radix_nodes", &RadixServer::total_radix_nodes)
        .def("total_blocks", &RadixServer::total_blocks)
        .def("data_pool_utilization", &RadixServer::data_pool_utilization)
        .def("mempool_total", &RadixServer::mempool_total)
        .def("mempool_free", &RadixServer::mempool_free)
        .def("mempool_used", &RadixServer::mempool_used)
        .def("swa_mempool_total", &RadixServer::swa_mempool_total)
        .def("mamba_mempool_total", &RadixServer::mamba_mempool_total)
        .def_property_readonly("component_config", &RadixServer::component_config)
        .def("insert_success_count", &RadixServer::insert_success_count)
        .def("peer_fault_counts",
             [](const RadixServer& s) { return peer_fault_dict(s.peer_faults()); },
             "Node-wide index-plane ops lost to unreachable peers.")
        .def("insert_error_node_pool_count", &RadixServer::insert_error_node_pool_count)
        .def("insert_error_data_pool_count", &RadixServer::insert_error_data_pool_count)
        .def("insert_error_ht_pool_count", &RadixServer::insert_error_ht_pool_count)
        .def("check_health", [](const RadixServer& s) { return s.check_health(); },
             "Lock-free leak audit; returns HealthFlag bits (0 == healthy).")
        .def("health_report", [](const RadixServer& s) {
            HealthReport r; s.check_health(&r); return r;
        }, "Lock-free leak audit; returns the full HealthReport.");

#ifdef RADIXSHMEM_HAS_ETCD
    // ==================== EtcdKv ====================
    // Plain key/value access to the cluster's etcd, sharing the process-global
    // connection with RadixServer / DataPlaneRegistry. Values are text (JSON,
    // base64). Used by radixshmem.server for radix/<cluster_id>/geometry/<node>.

    py::class_<etcd::Client>(m, "EtcdKv")
        .def(py::init([](const std::string& registry) {
            auto c = std::make_unique<etcd::Client>();
            py::gil_scoped_release nogil;
            c->init(etcd::strip_scheme(registry));
            return c;
        }), py::arg("registry"), "registry: 'etcd://host:port' or 'host:port'; throws when unreachable.")
        .def("put", [](etcd::Client& c, const std::string& k, const std::string& v, int64_t lease) {
            c.put(k, v, lease);
        }, py::arg("key"), py::arg("value"), py::arg("lease") = 0,
           py::call_guard<py::gil_scoped_release>())
        .def("create", [](etcd::Client& c, const std::string& k, const std::string& v, int64_t lease) {
            return c.create(k, v, lease);
        }, py::arg("key"), py::arg("value"), py::arg("lease") = 0,
           py::call_guard<py::gil_scoped_release>(), "Put-if-absent; True when this call created the key.")
        .def("get", [](etcd::Client& c, const std::string& k) -> py::object {
            std::string out;
            bool found;
            {
                py::gil_scoped_release nogil;
                found = c.get(k, out);
            }
            if (!found) return py::none();
            return py::str(out);
        }, py::arg("key"), "Value or None.")
        .def("get_prefix", [](etcd::Client& c, const std::string& prefix) {
            std::vector<std::pair<std::string, std::string>> out;
            {
                py::gil_scoped_release nogil;
                for (auto& kv : c.get_prefix(prefix)) out.emplace_back(kv.key, kv.value);
            }
            return out;
        }, py::arg("prefix"), "[(key, value), ...] sorted by key.")
        .def("count", [](etcd::Client& c, const std::string& prefix) { return c.count(prefix); },
             py::arg("prefix"), py::call_guard<py::gil_scoped_release>())
        .def("delete", [](etcd::Client& c, const std::string& k) { c.del(k); },
             py::arg("key"), py::call_guard<py::gil_scoped_release>(), "Exactly this key.")
        .def("grant_lease", [](etcd::Client& c, int64_t ttl_s) { return c.grant_lease(ttl_s); },
             py::arg("ttl_s"), py::call_guard<py::gil_scoped_release>())
        .def("keep_alive", [](etcd::Client& c, int64_t lease) { c.keep_alive(lease); },
             py::arg("lease"), py::call_guard<py::gil_scoped_release>())
        .def("revoke", [](etcd::Client& c, int64_t lease) { c.revoke(lease); },
             py::arg("lease"), py::call_guard<py::gil_scoped_release>());
#endif

    // ==================== RadixClient ====================

    py::class_<RadixClient>(m, "RadixClient")
        .def(py::init<const std::string&, bool>(),
             py::arg("name"), py::arg("local_read_only") = false,
             "Attach to an existing shm region by name. local_read_only=True attaches "
             "the tree only (no etcd / RDMA / RHT plane even on a cluster region): "
             "query() is always local, is_distributed() is False, and insert / "
             "allocate_slots / recycle_slots / reset raise RuntimeError.")
        .def(py::init<RadixServer&>(),
             py::arg("server"),
             "Embed inside a same-process RadixServer (enables RDMA routing).")

        .def("insert", [](RadixClient& c,
                          py::array_t<uint64_t> hashes,
                          py::array_t<int32_t> slots,
                          int start, bool auto_recycle,
                          ComponentType component) {
            auto [hptr, hlen] = npy_1d(hashes, "hashes");
            auto [sptr, slen] = npy_1d(slots, "slots");
            InsertResult result;
            {
                py::gil_scoped_release rel;
                result = c.insert(hptr, hlen, sptr, slen, start, auto_recycle,
                                  component);
            }
            return result;
        }, py::arg("hashes"), py::arg("slots"), py::arg("start") = 0,
           py::arg("auto_recycle") = true, py::kw_only(),
           py::arg("component") = ComponentType::FULL,
           "Publish slots for the prefix `hashes` (root-anchored, len n).\n"
           "FULL: slots[i] <-> hashes[start+i], len(slots) == n - start.\n"
           "SWA: the slots cover the trailing positions [n - len(slots), n), "
           "1 <= len(slots) <= n, start must be 0; the FULL path [0, n) must "
           "already exist (FULL_PATH_MISSING otherwise). The window W only "
           "gates queries, so a publisher may cover a single position or a "
           "whole node.\n"
           "MAMBA: one slot at END n, start must be 0.\n"
           "Slots the tree did not take (duplicates, refusals) come back in "
           "unused_slots and are recycled when auto_recycle is set.")

        // keep_alive: the result's finalize releases through this client.
        .def("query", [](RadixClient& c, py::array_t<uint64_t> hashes,
                         py::object mask, bool local_only,
                         bool lock, bool update_meta) {
            if (py::isinstance<ComponentType>(mask))
                throw py::type_error(
                    "mask must be a bitwise-OR of COMPONENT_MASK_* constants, "
                    "not a ComponentType");
            uint32_t m32 = mask.cast<uint32_t>();
            auto [ptr, len] = npy_1d(hashes, "hashes");
            ComponentQueryResult result;
            {
                py::gil_scoped_release rel;
                result = c.query(ptr, len, m32, local_only, lock,
                                 update_meta);
            }
            return result;
        }, py::arg("hashes"), py::arg("mask") = COMPONENT_MASK_FULL,
           py::arg("local_only") = false,
           py::arg("lock") = false, py::arg("update_meta") = true,
           py::keep_alive<0, 1>())

        // Empty array on a component-pool shortfall (all-or-none); bad args raise.
        .def("allocate_slots", [](RadixClient& c, int n,
                                  ComponentType component) {
            std::vector<int32_t> out;
            int got;
            {
                py::gil_scoped_release rel;
                got = c.allocate_slots(n, out, component);
            }
            if (got < 0)
                throw py::value_error("allocate_slots: bad num_slots or component");
            return vec_to_npy_i32(out);
        }, py::arg("num_slots"), py::arg("component") = ComponentType::FULL)

        .def("recycle_slots", [](RadixClient& c, py::array_t<int32_t> slots,
                                 ComponentType component) {
            auto [ptr, len] = npy_1d(slots, "slots");
            py::gil_scoped_release rel;
            c.recycle_slots(ptr, len, component);
        }, py::arg("slots"), py::arg("component") = ComponentType::FULL)

        .def("reset", &RadixClient::reset,
             py::call_guard<py::gil_scoped_release>(),
             "Clear all cached data in place; invalidates prior slot IDs.")

        .def("flush", &RadixClient::flush,
             py::call_guard<py::gil_scoped_release>(),
             "Drain pending async RHT writes; no-op on single-node.")
        .def("peer_fault_counts",
             [](const RadixClient& c) { return peer_fault_dict(c.peer_faults()); },
             "Node-wide index-plane ops lost to unreachable peers.")
        .def("is_distributed", &RadixClient::is_distributed)
        .def("rank", &RadixClient::rank)
        .def("world_size", &RadixClient::world_size)
        .def("total_radix_nodes", &RadixClient::total_radix_nodes)
        .def("total_blocks", &RadixClient::total_blocks)
        .def("block_size", &RadixClient::block_size)
        .def("data_pool_utilization", &RadixClient::data_pool_utilization)
        .def("mempool_total", &RadixClient::mempool_total)
        .def("mempool_free", &RadixClient::mempool_free)
        .def("mempool_used", &RadixClient::mempool_used)
        .def("swa_mempool_total", &RadixClient::swa_mempool_total)
        .def("mamba_mempool_total", &RadixClient::mamba_mempool_total)
        .def_property_readonly("component_config", &RadixClient::component_config)
        .def("print_tree", &RadixClient::print_tree)
        .def("name", &RadixClient::name)
        .def("check_health", [](const RadixClient& c) { return c.check_health(); },
             "Lock-free leak audit of the local tree; returns HealthFlag bits (0 == healthy).")
        .def("health_report", [](const RadixClient& c) {
            HealthReport r; c.check_health(&r); return r;
        }, "Lock-free leak audit; returns the full HealthReport.")
#ifndef RADIXSHMEM_NO_RDMA
        // Remote ops are rank-first; the client caches one session per rank.
        .def("remote_allocate", [](RadixClient& c, int rank, int n) {
            AllocateReply rep;
            { py::gil_scoped_release rel; rep = c.remote_allocate(rank, n); }
            return rep;
        }, py::arg("rank"), py::arg("n"),
           "Phase 1: allocate `n` slots on `rank` (no hashes, no dedup).")

        .def("remote_insert", [](RadixClient& c, int rank,
                                   py::array_t<uint64_t> hashes,
                                   int start, int length,
                                   py::array_t<int32_t> slots,
                                   bool auto_recycle) {
            auto [ptr, len] = npy_1d(hashes, "hashes");
            auto [sptr, slen] = npy_1d(slots, "slots");
            if (slen < length)
                throw std::runtime_error("slots must carry `length` entries");
            InsertResult res;
            { py::gil_scoped_release rel;
              res = c.remote_insert(rank, ptr, len, start, length, sptr,
                                    auto_recycle); }
            return res;
        }, py::arg("rank"), py::arg("hashes"), py::arg("start"),
           py::arg("length"), py::arg("slots"),
           py::arg("auto_recycle") = true,
           "Phase 2: publish `slots` on `rank`; auto_recycle frees the unmatched surplus.")

        .def("remote_recycle", [](RadixClient& c, int rank,
                                  py::array_t<int32_t> slots) {
            auto [sptr, slen] = npy_1d(slots, "slots");
            RemoteStatus st;
            { py::gil_scoped_release rel; st = c.remote_recycle(rank, sptr, slen); }
            return st;
        }, py::arg("rank"), py::arg("slots"),
           "Return unpublished slots to `rank`'s pool (abort / surplus); returns RemoteStatus.")

        .def("remote_query", [](RadixClient& c, int rank,
                                py::array_t<uint64_t> hashes, bool lock,
                                bool update_meta) {
            auto [ptr, len] = npy_1d(hashes, "hashes");
            QueryResult result;
            {
                py::gil_scoped_release rel;
                result = c.remote_query(rank, ptr, len, lock, update_meta);
            }
            return result;
        }, py::arg("rank"), py::arg("hashes"), py::arg("lock") = false,
           py::arg("update_meta") = true,
           // keep_alive<0,1>: lock=True result's finalize calls back into client.
           py::keep_alive<0, 1>(),
           "Query `rank`'s own tree (1 RTT); lock=True releases via finalize().")

        .def("remote_unlock", [](RadixClient& c, int rank,
                                 py::array_t<uint64_t> hashes, int hit_length) {
            auto [ptr, len] = npy_1d(hashes, "hashes");
            RemoteStatus st;
            { py::gil_scoped_release rel;
              st = c.remote_unlock(rank, ptr, len, hit_length); }
            return st;
        }, py::arg("rank"), py::arg("hashes"), py::arg("hit_length"),
           "Manually drop a remote_query(lock=True) pin: dec_ref over [0, hit_length); returns RemoteStatus.")
#endif

        .def_static("compute_hashes", &py_compute_hashes,
                     py::arg("token_ids"), py::arg("tokens_per_block"));

#ifndef RADIXSHMEM_NO_RDMA
    // ==================== Remote-op control plane ====================
    // (RemoteStatus is registered unconditionally in the Enums section above.)

    py::class_<AllocateReply>(m, "AllocateReply")
        .def_readonly("status", &AllocateReply::status)
        .def_property_readonly("slots", [](const AllocateReply& r) {
            return vec_to_npy_i32(r.slots);
        });
#endif  // !RADIXSHMEM_NO_RDMA

    // ==================== Free functions ====================

    m.def("compute_hashes", &py_compute_hashes,
          py::arg("token_ids"), py::arg("tokens_per_block"),
          "Compute cumulative prefix hashes from token IDs.");
}
