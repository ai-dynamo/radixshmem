// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// radixshmem._data — Python bindings for the data plane (SlotStore + transfer
// backend + SlotTransfer). A separate extension from radixshmem._core: the data
// plane does not link the indexer.
//
// Lifetime: SlotTransfer borrows its store and backend; keep_alive pins both.
// Slot payloads are memoryviews over the shared mapping (windows, not copies).

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "component_kind.h"
#include "data_registry.h"
#include "slot_geometry.h"
#include "slot_store.h"
#include "slot_transfer.h"
#include "transfer.h"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace radixshmem;

namespace {

void check_pair(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    if (a.size() != b.size())
        throw std::invalid_argument("slot lists must have the same length (got " +
                                    std::to_string(a.size()) + " and " +
                                    std::to_string(b.size()) + ")");
    if (a.empty())
        throw std::invalid_argument("slot lists must not be empty");
}

void check_kind(ComponentType c) {
    if (!component_type_valid(c))
        throw std::invalid_argument("invalid ComponentType " +
                                    std::to_string((int)c));
}

// `kind` arguments take a ComponentType or a plain int (0/1/2).
ComponentType kind_of(const py::handle& h) {
    ComponentType c;
    if (py::isinstance<ComponentType>(h)) c = h.cast<ComponentType>();
    else if (py::isinstance<py::int_>(h)) c = static_cast<ComponentType>(h.cast<int>());
    else throw py::type_error("kind must be a ComponentType or an int");
    check_kind(c);
    return c;
}

#define KIND_ARG(name) py::arg_v(name, ComponentType::FULL, "ComponentType.FULL")

py::memoryview view_of(void* p, uint64_t n, const char* what) {
    if (p == nullptr) throw std::out_of_range(what);
    return py::memoryview::from_memory(p, (py::ssize_t)n, /*readonly=*/false);
}

void* slot_or_throw(SlotStore& s, ComponentType c, uint64_t slot_id) {
    check_kind(c);
    if (!s.has_pool(c))
        throw std::out_of_range(std::string("store has no ") + component_type_name(c) + " pool");
    void* p = s.slot_addr(c, slot_id);
    if (!p) throw std::out_of_range("slot id out of range");
    return p;
}

std::string pool_repr(const PoolGeometry& g) {
    return "slots=" + std::to_string(g.num_slots) + " slot_bytes=" +
           std::to_string(g.slot_bytes) + " base=" + std::to_string(g.base);
}

} // namespace

PYBIND11_MODULE(_data, m) {
    m.doc() = "radixshmem data plane: shared-memory slot pools and slot-granular transfers";

    // ==================== return codes / enums ====================
    m.attr("XFER_OK")        = XFER_OK;
    m.attr("XFER_EINVAL")    = XFER_EINVAL;
    m.attr("XFER_ENOENT")    = XFER_ENOENT;
    m.attr("XFER_EGEOMETRY") = XFER_EGEOMETRY;
    m.attr("XFER_EBACKEND")  = XFER_EBACKEND;
    m.attr("XFER_EBUSY")     = XFER_EBUSY;
    m.attr("INVALID_BATCH")  = INVALID_BATCH;
    m.attr("PEER_SLOT_BYTES_UNCHECKED") = PEER_SLOT_BYTES_UNCHECKED;
    // Transfer backends compiled in (tests SKIP on a missing one).
    py::list backends;
#ifdef RADIXSHMEM_HAS_MOONCAKE
    backends.append("mooncake");
#endif
#ifdef RADIXSHMEM_HAS_NIXL
    backends.append("nixl");
#endif
    m.attr("TRANSFER_BACKENDS") = backends;
    m.attr("COMPONENT_MASK_FULL")  = COMPONENT_MASK_FULL;
    m.attr("COMPONENT_MASK_SWA")   = COMPONENT_MASK_SWA;
    m.attr("COMPONENT_MASK_MAMBA") = COMPONENT_MASK_MAMBA;

    // ComponentType is shared with _core: reuse its registration when _core is
    // already loaded (pybind11 refuses to bind one C++ type twice).
    if (py::detail::get_type_info(typeid(ComponentType)) == nullptr) {
        py::enum_<ComponentType>(m, "ComponentType")
            .value("FULL", ComponentType::FULL)
            .value("SWA", ComponentType::SWA)
            .value("MAMBA", ComponentType::MAMBA);
    } else {
        m.attr("ComponentType") = py::detail::get_type_handle(typeid(ComponentType), true);
    }

    py::enum_<XferState>(m, "XferState")
        .value("PENDING", XferState::PENDING)
        .value("COMPLETED", XferState::COMPLETED)
        .value("FAILED", XferState::FAILED)
        .value("TIMEOUT", XferState::TIMEOUT)
        .value("INVALID", XferState::INVALID);

    // ==================== geometry ====================
    py::class_<PoolGeometry>(m, "PoolGeometry")
        .def(py::init<>())
        .def(py::init([](uint64_t base, uint64_t num_slots, uint64_t slot_bytes) {
                 return PoolGeometry{base, num_slots, slot_bytes};
             }),
             py::arg("base"), py::arg("num_slots"), py::arg("slot_bytes"))
        .def_readwrite("base", &PoolGeometry::base)
        .def_readwrite("num_slots", &PoolGeometry::num_slots)
        .def_readwrite("slot_bytes", &PoolGeometry::slot_bytes)
        .def("__repr__", [](const PoolGeometry& g) { return "<PoolGeometry " + pool_repr(g) + ">"; });
    m.attr("SlotGeometry") = m.attr("PoolGeometry");

    py::class_<StoreGeometry>(m, "StoreGeometry")
        .def(py::init<>())
        .def_readwrite("pool_mask", &StoreGeometry::pool_mask)
        .def("has", [](const StoreGeometry& g, py::object k) { return g.has(kind_of(k)); }, py::arg("kind"))
        .def("__getitem__",
             [](const StoreGeometry& g, py::object k) { return g[kind_of(k)]; },
             py::arg("kind"))
        .def("__setitem__",
             [](StoreGeometry& g, py::object k, const PoolGeometry& p) {
                 const ComponentType c = kind_of(k);
                 g[c] = p;
                 g.pool_mask |= component_mask_of(c);
             },
             py::arg("kind"), py::arg("pool"))
        .def("rebased", &StoreGeometry::rebased, py::arg("segment_base"))
        .def("same_strides", &StoreGeometry::same_strides, py::arg("other"))
        .def("__repr__", [](const StoreGeometry& g) {
            std::string r = "<StoreGeometry mask=" + std::to_string(g.pool_mask);
            for (int i = 0; i < NUM_COMPONENT_TYPES; ++i) {
                const ComponentType c = (ComponentType)i;
                if (g.has(c)) r += std::string(" ") + component_type_name(c) + "{" + pool_repr(g[c]) + "}";
            }
            return r + ">";
        });

    // ==================== storage ====================
    py::class_<SlotPoolConfig>(m, "SlotPoolConfig")
        .def(py::init<>())
        .def(py::init([](uint64_t num_slots, uint64_t slot_bytes) {
                 SlotPoolConfig p;
                 p.num_slots = num_slots;
                 p.slot_bytes = slot_bytes;
                 return p;
             }),
             py::arg("num_slots"), py::arg("slot_bytes"))
        .def_readwrite("num_slots", &SlotPoolConfig::num_slots)
        .def_readwrite("slot_bytes", &SlotPoolConfig::slot_bytes)
        .def_property_readonly("enabled", &SlotPoolConfig::enabled)
        .def("__repr__", [](const SlotPoolConfig& p) {
            return "SlotPoolConfig(num_slots=" + std::to_string(p.num_slots) +
                   ", slot_bytes=" + std::to_string(p.slot_bytes) + ")";
        });

    py::class_<SlotStoreConfig>(m, "SlotStoreConfig")
        .def(py::init<>())
        .def(py::init([](const std::string& name, const SlotPoolConfig& full,
                         const SlotPoolConfig& swa, const SlotPoolConfig& mamba,
                         uint64_t slot_align, const std::string& hugepage_path,
                         bool prefault) {
                 SlotStoreConfig c;
                 c.name = name;
                 c.full = full;
                 c.swa = swa;
                 c.mamba = mamba;
                 c.slot_align = slot_align;
                 c.hugepage_path = hugepage_path;
                 c.prefault = prefault;
                 return c;
             }),
             py::arg("name"), py::arg("full"),
             py::arg("swa") = SlotPoolConfig{},
             py::arg("mamba") = SlotPoolConfig{},
             py::arg("slot_align") = 4096,
             py::arg("hugepage_path") = "/dev/hugepages",
             py::arg("prefault") = true)
        .def_readwrite("name", &SlotStoreConfig::name)
        // reference_internal: `cfg.full.num_slots = n` edits in place.
        .def_readwrite("full", &SlotStoreConfig::full, py::return_value_policy::reference_internal)
        .def_readwrite("swa", &SlotStoreConfig::swa, py::return_value_policy::reference_internal)
        .def_readwrite("mamba", &SlotStoreConfig::mamba, py::return_value_policy::reference_internal)
        .def_readwrite("slot_align", &SlotStoreConfig::slot_align)
        .def_readwrite("hugepage_path", &SlotStoreConfig::hugepage_path)
        .def_readwrite("prefault", &SlotStoreConfig::prefault)
        .def("pool", [](const SlotStoreConfig& c, py::object k) { return c.pool(kind_of(k)); },
             py::arg("kind"))
        .def_property_readonly("pool_mask", &SlotStoreConfig::pool_mask)
        .def("__repr__", [](const SlotStoreConfig& c) {
            return "SlotStoreConfig(name='" + c.name + "', full=(" +
                   std::to_string(c.full.num_slots) + "x" + std::to_string(c.full.slot_bytes) +
                   "), swa=(" + std::to_string(c.swa.num_slots) + "x" + std::to_string(c.swa.slot_bytes) +
                   "), mamba=(" + std::to_string(c.mamba.num_slots) + "x" + std::to_string(c.mamba.slot_bytes) +
                   "), slot_align=" + std::to_string(c.slot_align) + ")";
        });

    py::class_<SlotStore>(m, "SlotStore")
        .def_static("create", &SlotStore::create, py::arg("config"),
                    "Owner: create the region and publish its geometry.")
        .def_static("attach", &SlotStore::attach, py::arg("name"),
                    py::arg("hugepage_path") = "/dev/hugepages",
                    py::arg("timeout_ms") = 5000,
                    "Non-owner: map an existing region and read its geometry.")
        .def_static("destroy", &SlotStore::destroy, py::arg("name"),
                    py::arg("hugepage_path") = "/dev/hugepages",
                    "Remove a leftover region by name. Never throws.")
        .def_property_readonly("name", &SlotStore::name)
        // FULL-pool shorthands.
        .def_property_readonly("num_slots", [](const SlotStore& s) { return s.num_slots(); })
        .def_property_readonly("slot_bytes", [](const SlotStore& s) { return s.slot_bytes(); })
        .def_property_readonly("data_bytes", &SlotStore::data_bytes)
        .def_property_readonly("valid", &SlotStore::valid)
        .def_property_readonly("is_owner", &SlotStore::is_owner)
        .def_property_readonly("is_hugepage", &SlotStore::is_hugepage)
        .def_property_readonly("pool_mask", &SlotStore::pool_mask)
        .def("has_pool", [](const SlotStore& s, py::object k) { return s.has_pool(kind_of(k)); },
             py::arg("kind"))
        .def("pool",
             [](const SlotStore& s, py::object k) {
                 const ComponentType c = kind_of(k);
                 return PoolGeometry{s.pool_offset(c), s.num_slots(c), s.slot_bytes(c)};
             },
             py::arg("kind"),
             "Geometry of one pool (base = offset from data_view); all zero if absent.")
        .def("layout", &SlotStore::layout,
             "StoreGeometry with pool offsets relative to data_view(): what a node publishes.")
        .def("slot_offset",
             [](const SlotStore& s, uint64_t slot_id, py::object k) {
                 return s.slot_offset(kind_of(k), slot_id);
             },
             py::arg("slot_id"), KIND_ARG("kind"),
             "Offset from data_view(); UINT64_MAX if out of range / no pool.")
        .def("slot_view",
             [](SlotStore& s, uint64_t slot_id, py::object k) {
                 const ComponentType c = kind_of(k);
                 return view_of(slot_or_throw(s, c, slot_id), s.slot_bytes(c), "slot");
             },
             py::arg("slot_id"), KIND_ARG("kind"), py::keep_alive<0, 1>(),
             "Writable memoryview over one slot.")
        .def("pool_view",
             [](SlotStore& s, py::object k) {
                 const ComponentType c = kind_of(k);
                 if (!s.has_pool(c))
                     throw std::out_of_range(std::string("store has no ") + component_type_name(c) + " pool");
                 return view_of(s.pool_base(c), s.pool_bytes(c), "pool");
             },
             py::arg("kind"), py::keep_alive<0, 1>(),
             "Writable memoryview over one whole pool (num_slots x slot_bytes).")
        .def("data_view",
             [](SlotStore& s) { return view_of(s.data_base(), s.data_bytes(), "empty store"); },
             py::keep_alive<0, 1>(),
             "Writable memoryview over the whole registerable range.")
        .def("read_slot",
             [](SlotStore& s, uint64_t slot_id, py::object k) {
                 const ComponentType c = kind_of(k);
                 void* p = slot_or_throw(s, c, slot_id);
                 return py::bytes((const char*)p, (size_t)s.slot_bytes(c));
             },
             py::arg("slot_id"), KIND_ARG("kind"),
             "Copy one slot out as bytes.")
        .def("write_slot",
             [](SlotStore& s, uint64_t slot_id, py::buffer data, py::object k) {
                 const ComponentType c = kind_of(k);
                 void* p = slot_or_throw(s, c, slot_id);
                 py::buffer_info info = data.request();
                 if ((uint64_t)info.size * (uint64_t)info.itemsize > s.slot_bytes(c))
                     throw std::invalid_argument("payload larger than one slot");
                 std::memcpy(p, info.ptr, (size_t)info.size * (size_t)info.itemsize);
             },
             py::arg("slot_id"), py::arg("data"), KIND_ARG("kind"),
             "Copy bytes into the front of one slot.")
        .def("__repr__", [](const SlotStore& s) {
            std::string r = "<SlotStore name='" + s.name() + "'";
            for (int i = 0; i < NUM_COMPONENT_TYPES; ++i) {
                const ComponentType c = (ComponentType)i;
                if (s.has_pool(c))
                    r += std::string(" ") + component_type_name(c) + "=" +
                         std::to_string(s.num_slots(c)) + "x" + std::to_string(s.slot_bytes(c));
            }
            return r + ">";
        });

    // ==================== transfer backend ====================
    py::class_<TransferConfig>(m, "TransferConfig")
        .def(py::init<>())
        .def(py::init([](const std::string& backend, const std::string& metadata,
                         const std::string& local_segment, const std::string& ip_or_host,
                         uint64_t rpc_port, const std::string& protocol,
                         const std::vector<std::string>& device_filter, bool publish) {
                 TransferConfig c;
                 c.backend = backend;
                 c.metadata = metadata;
                 c.local_segment = local_segment;
                 c.ip_or_host = ip_or_host;
                 c.rpc_port = rpc_port;
                 c.protocol = protocol;
                 c.device_filter = device_filter;
                 c.publish = publish;
                 return c;
             }),
             py::arg("backend") = "mooncake",
             py::arg("metadata") = "P2PHANDSHAKE",
             py::arg("local_segment") = "",
             py::arg("ip_or_host") = "",
             py::arg("rpc_port") = 12345,
             py::arg("protocol") = "rdma",
             py::arg("device_filter") = std::vector<std::string>{},
             py::arg("publish") = true)
        .def_readwrite("backend", &TransferConfig::backend)
        .def_readwrite("metadata", &TransferConfig::metadata)
        .def_readwrite("local_segment", &TransferConfig::local_segment)
        .def_readwrite("ip_or_host", &TransferConfig::ip_or_host)
        .def_readwrite("rpc_port", &TransferConfig::rpc_port)
        .def_readwrite("protocol", &TransferConfig::protocol)
        .def_readwrite("device_filter", &TransferConfig::device_filter)
        .def_readwrite("publish", &TransferConfig::publish);

    py::class_<ITransferBackend>(m, "TransferBackend")
        .def_property_readonly("backend_name", &ITransferBackend::backend_name)
        .def_property_readonly("local_segment", &ITransferBackend::local_segment)
        .def("__repr__", [](const ITransferBackend& b) {
            return std::string("<TransferBackend '") + b.backend_name() + "' segment='" +
                   b.local_segment() + "'>";
        });

    // None when the backend name is unknown or was not compiled in.
    m.def("create_transfer_backend",
          [](const TransferConfig& cfg) { return create_transfer_backend(cfg); },
          py::arg("config"),
          py::call_guard<py::gil_scoped_release>(),
          "Build a transfer backend, or None if it is unavailable.");

    // ==================== slot-granular layer ====================
    py::class_<SlotTransfer>(m, "SlotTransfer")
        .def(py::init<SlotStore&, ITransferBackend&>(),
             py::arg("store"), py::arg("backend"),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
        .def("register_store", &SlotTransfer::register_store,
             py::arg("location") = "",
             py::call_guard<py::gil_scoped_release>(),
             "Hand the store's registerable range to the backend.")
        .def("attach_peer",
             [](SlotTransfer& t, const std::string& peer, py::object published) -> py::tuple {
                 // int: FULL-only stride form -> (rc, PoolGeometry).
                 if (py::isinstance<py::int_>(published)) {
                     const uint64_t stride = published.cast<uint64_t>();
                     SlotGeometry g;
                     int rc;
                     {
                         py::gil_scoped_release nogil;
                         rc = t.attach_peer(peer, stride, &g);
                     }
                     return py::make_tuple(rc, rc == XFER_OK ? py::cast(g) : py::none());
                 }
                 // StoreGeometry or None -> (rc, StoreGeometry).
                 const StoreGeometry* layout = nullptr;
                 StoreGeometry holder;
                 if (!published.is_none()) {
                     holder = published.cast<StoreGeometry>();
                     layout = &holder;
                 }
                 StoreGeometry g;
                 int rc;
                 {
                     py::gil_scoped_release nogil;
                     rc = t.attach_peer(peer, layout, &g);
                 }
                 return py::make_tuple(rc, rc == XFER_OK ? py::cast(g) : py::none());
             },
             py::arg("peer"), py::arg("published"),
             "Resolve a peer segment and check its geometry. `published` is the "
             "peer's StoreGeometry from the registry, None to assume our layout, "
             "or (FULL-only stores) the published stride / PEER_SLOT_BYTES_UNCHECKED. "
             "Returns (rc, geometry or None).")
        .def("detach_peer", &SlotTransfer::detach_peer, py::arg("peer"),
             "Close the peer segment. XFER_EBUSY while any batch is unreleased.")
        .def("peer_geometry",
             [](const SlotTransfer& t, const std::string& peer) -> py::object {
                 StoreGeometry g;
                 if (!t.peer_geometry(peer, g)) return py::none();
                 return py::cast(g);
             },
             py::arg("peer"), "Attached peer's StoreGeometry (absolute bases), or None.")
        .def_property_readonly("local_segment", &SlotTransfer::local_segment)
        .def("read",
             [](SlotTransfer& t, const std::string& peer,
                const std::vector<int32_t>& remote_slots,
                const std::vector<int32_t>& local_slots, py::object k) {
                 check_pair(remote_slots, local_slots);
                 const ComponentType c = kind_of(k);
                 BatchHandle h;
                 int ndesc = 0;
                 {
                     py::gil_scoped_release nogil;
                     h = t.read(peer, c, remote_slots.data(), local_slots.data(),
                                (int)local_slots.size(), &ndesc);
                 }
                 return py::make_tuple(h, ndesc);
             },
             py::arg("peer"), py::arg("remote_slots"), py::arg("local_slots"),
             KIND_ARG("kind"),
             "local_slots[i] <- peer's remote_slots[i] in pool `kind`. Returns "
             "(batch_handle, descriptors); handle is INVALID_BATCH on rejection.")
        .def("write",
             [](SlotTransfer& t, const std::string& peer,
                const std::vector<int32_t>& local_slots,
                const std::vector<int32_t>& remote_slots, py::object k) {
                 check_pair(local_slots, remote_slots);
                 const ComponentType c = kind_of(k);
                 BatchHandle h;
                 int ndesc = 0;
                 {
                     py::gil_scoped_release nogil;
                     h = t.write(peer, c, local_slots.data(), remote_slots.data(),
                                 (int)local_slots.size(), &ndesc);
                 }
                 return py::make_tuple(h, ndesc);
             },
             py::arg("peer"), py::arg("local_slots"), py::arg("remote_slots"),
             KIND_ARG("kind"),
             "peer's remote_slots[i] <- local_slots[i] in pool `kind`. Returns "
             "(batch_handle, descriptors).")
        .def("poll",
             [](SlotTransfer& t, BatchHandle h) {
                 uint64_t done = 0;
                 XferState s;
                 {
                     py::gil_scoped_release nogil;
                     s = t.poll(h, &done);
                 }
                 return py::make_tuple(s, done);
             },
             py::arg("handle"), "Returns (XferState, bytes_done).")
        .def("wait", &SlotTransfer::wait, py::arg("handle"), py::arg("timeout_ms"),
             py::call_guard<py::gil_scoped_release>())
        .def("release", &SlotTransfer::release, py::arg("handle"),
             py::call_guard<py::gil_scoped_release>(),
             "XFER_OK, or XFER_EBUSY while the batch is still running.");

#ifdef RADIXSHMEM_HAS_ETCD
    // ==================== data-plane registry (etcd) ====================
    py::class_<DataPlaneEntry>(m, "DataPlaneEntry")
        .def_readonly("pool_mask", &DataPlaneEntry::pool_mask)
        .def_property_readonly("segment",
                               [](const DataPlaneEntry& e) { return std::string(e.segment); })
        .def("layout", &data_plane_layout, "StoreGeometry with offsets relative to the segment.")
        .def("__repr__", [](const DataPlaneEntry& e) {
            return "<DataPlaneEntry segment='" + std::string(e.segment) +
                   "' mask=" + std::to_string(e.pool_mask) + ">";
        });

    m.def("make_data_plane_entry",
          [](const StoreGeometry& layout, const std::string& segment) {
              DataPlaneEntry e{};
              if (!make_data_plane_entry(layout, segment, e))
                  throw std::invalid_argument("malformed layout or over-long segment name");
              return e;
          },
          py::arg("layout"), py::arg("segment"));

    py::class_<DataPlaneRegistry>(m, "DataPlaneRegistry")
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("registry"), py::arg("cluster_id"),
             py::call_guard<py::gil_scoped_release>())
        .def("publish", &DataPlaneRegistry::publish,
             py::arg("node"), py::arg("entry"), py::arg("lease") = 0,
             py::call_guard<py::gil_scoped_release>())
        .def("unpublish", &DataPlaneRegistry::unpublish, py::arg("node"),
             py::call_guard<py::gil_scoped_release>())
        .def("lookup",
             [](DataPlaneRegistry& r, const std::string& node, int timeout_sec) -> py::object {
                 DataPlaneEntry e{};
                 bool ok;
                 {
                     py::gil_scoped_release nogil;
                     ok = r.lookup(node, e, timeout_sec);
                 }
                 if (!ok) return py::none();
                 return py::cast(e);
             },
             py::arg("node"), py::arg("timeout_sec") = 0)
        .def("list", [](DataPlaneRegistry& r) { return r.list(); },
             py::call_guard<py::gil_scoped_release>())
        .def("node_of_rank", &DataPlaneRegistry::node_of_rank, py::arg("rank"),
             py::call_guard<py::gil_scoped_release>())
        .def("validate_uniform",
             [](DataPlaneRegistry& r, const DataPlaneEntry& local) {
                 std::string who, why;
                 bool ok;
                 {
                     py::gil_scoped_release nogil;
                     ok = r.validate_uniform(local, who, why);
                 }
                 return py::make_tuple(ok, who, why);
             },
             py::arg("local"), "Returns (ok, offending node, reason).");
#endif
}
