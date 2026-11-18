// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Data-plane bindings (SlotStore, C++ transfer backends, slot planning), part of radixshmem._core.
//
// Slot payloads are memoryviews over the shared mapping (windows, not copies).

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "component_kind.h"
#include "data_registry.h"
#include "slot_geometry.h"
#include "slot_store.h"
#include "slot_plan.h"
#include "transfer.h"

#include <climits>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace radixshmem;

namespace {

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

// No forcecast: numpy's safe casting only, so a value that does not fit is a TypeError, not a wrapped address.
using U64Array = py::array_t<uint64_t, py::array::c_style>;

// Slot ids from a sequence of ints or a 1-d integer array, read exactly: anything else is a TypeError (numpy's
// casts would truncate 1.7 to 1). Ids past int64 become -1, so plan_slots refuses them as out of range.
std::vector<int64_t> slot_ids(py::handle h, const char* what) {
    std::vector<int64_t> out;
    if (py::isinstance<py::array>(h)) {
        auto a = py::reinterpret_borrow<py::array>(h);
        const char k = a.dtype().kind();
        if ((k != 'i' && k != 'u') || a.ndim() != 1)
            throw py::type_error(std::string(what) + " must be a 1-d integer array");
        // uint64 ids past int64 wrap negative: out of range like any other.
        auto v = py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(a);
        if (!v) throw std::runtime_error(std::string("cannot read ") + what);
        out.assign(v.data(), v.data() + v.size());
        return out;
    }
    if (!PySequence_Check(h.ptr()) || PyUnicode_Check(h.ptr()) || PyBytes_Check(h.ptr()))
        throw py::type_error(std::string(what) + " must be a sequence of integers or an integer array");
    const Py_ssize_t n = PyObject_Size(h.ptr());
    if (n < 0) throw py::error_already_set();
    out.reserve((size_t)n);
    for (py::handle item : h) {
        PyObject* o = item.ptr();
        int overflow = 0;
        long long v;
        if (PyLong_CheckExact(o)) {
            v = PyLong_AsLongLongAndOverflow(o, &overflow);
        } else {
            if (PyBool_Check(o) || PyFloat_Check(o) || !PyIndex_Check(o))
                throw py::type_error(std::string(what) + " must be integers");
            py::object i = py::reinterpret_steal<py::object>(PyNumber_Index(o));
            if (!i) throw py::error_already_set();
            v = PyLong_AsLongLongAndOverflow(i.ptr(), &overflow);
        }
        if (v == -1 && PyErr_Occurred()) throw py::error_already_set();
        out.push_back(overflow ? -1 : (int64_t)v);
    }
    return out;
}

py::ssize_t check_ranges(const U64Array& local, const U64Array& remote, const U64Array& lengths) {
    if (local.ndim() != 1 || remote.ndim() != 1 || lengths.ndim() != 1 || local.shape(0) != remote.shape(0) ||
        local.shape(0) != lengths.shape(0))
        throw std::invalid_argument("local, remote and lengths must be 1-d and the same length");
    return local.shape(0);
}

py::array_t<uint64_t> to_array(const std::vector<uint64_t>& v) {
    py::array_t<uint64_t> a(static_cast<py::ssize_t>(v.size()));
    std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(uint64_t));
    return a;
}

// Calls take a reference before they release the GIL; close() drops ours, and the engine goes with the last one.
struct NativeBackend {
    std::mutex mu;
    std::shared_ptr<ITransferBackend> be;
    std::shared_ptr<ITransferBackend> get() {
        std::lock_guard<std::mutex> lk(mu);
        if (!be) throw std::runtime_error("transfer backend is closed");
        return be;
    }
    std::shared_ptr<ITransferBackend> take() {
        std::lock_guard<std::mutex> lk(mu);
        return std::move(be);
    }
};

std::string pool_repr(const PoolGeometry& g) {
    return "slots=" + std::to_string(g.num_slots) + " slot_bytes=" +
           std::to_string(g.slot_bytes) + " base=" + std::to_string(g.base);
}

} // namespace

void bind_data(py::module_& m) {
    // ==================== return codes / enums ====================
    m.attr("XFER_OK")        = XFER_OK;
    m.attr("XFER_EINVAL")    = XFER_EINVAL;
    m.attr("XFER_ENOENT")    = XFER_ENOENT;
    m.attr("XFER_EGEOMETRY") = XFER_EGEOMETRY;
    m.attr("XFER_EBACKEND")  = XFER_EBACKEND;
    m.attr("XFER_EBUSY")     = XFER_EBUSY;
    m.attr("INVALID_BATCH")  = INVALID_BATCH;
    m.attr("PEER_SLOT_BYTES_UNCHECKED") = PEER_SLOT_BYTES_UNCHECKED;
    // Transfer backends whose plugin is installed (tests SKIP on a missing one).
    m.attr("TRANSFER_BACKENDS") = available_transfer_backends();

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
        .def_property_readonly("data_addr", [](const SlotStore& s) { return reinterpret_cast<uint64_t>(s.data_base()); })
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

    // ==================== transfer backends implemented in C++ ====================
    py::enum_<XferOp>(m, "XferOp")
        .value("READ", XferOp::READ)
        .value("WRITE", XferOp::WRITE);

    py::class_<SegmentInfo>(m, "SegmentInfo")
        .def(py::init([](int32_t id, uint64_t base, uint64_t length) { return SegmentInfo{id, base, length}; }),
             py::arg("id"), py::arg("base"), py::arg("length"))
        .def_readwrite("id", &SegmentInfo::id)
        .def_readwrite("base", &SegmentInfo::base)
        .def_readwrite("length", &SegmentInfo::length)
        .def("__repr__", [](const SegmentInfo& i) {
            return "<SegmentInfo id=" + std::to_string(i.id) + " base=" + std::to_string(i.base) +
                   " length=" + std::to_string(i.length) + ">";
        });

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

    // A plugin's ITransferBackend behind radixshmem.transfer.TransferBackend.
    py::class_<NativeBackend>(m, "NativeTransferBackend")
        .def_property_readonly("name", [](NativeBackend& b) { return std::string(b.get()->backend_name()); })
        .def_property_readonly("local_segment", [](NativeBackend& b) { return b.get()->local_segment(); })
        .def("register_memory",
             [](NativeBackend& b, uint64_t addr, uint64_t length, const std::string& location) {
                 auto be = b.get();
                 py::gil_scoped_release nogil;
                 return be->register_memory(reinterpret_cast<void*>(addr), length, location);
             },
             py::arg("addr"), py::arg("length"), py::arg("location") = "")
        .def("unregister_memory",
             [](NativeBackend& b, uint64_t addr) {
                 auto be = b.get();
                 py::gil_scoped_release nogil;
                 return be->unregister_memory(reinterpret_cast<void*>(addr));
             },
             py::arg("addr"))
        .def("open_segment",
             [](NativeBackend& b, const std::string& name) -> py::tuple {
                 auto be = b.get();
                 SegmentInfo info;
                 int rc;
                 {
                     py::gil_scoped_release nogil;
                     rc = be->open_segment(name, info);
                 }
                 return py::make_tuple(rc, rc == XFER_OK ? py::cast(info) : py::none());
             },
             py::arg("name"), "Returns (rc, SegmentInfo or None).")
        .def("close_segment",
             [](NativeBackend& b, int32_t id) {
                 auto be = b.get();
                 py::gil_scoped_release nogil;
                 be->close_segment(id);
             },
             py::arg("seg_id"))
        .def("submit",
             [](NativeBackend& b, XferOp op, int32_t seg, const U64Array& local, const U64Array& remote,
                const U64Array& lengths) {
                 const py::ssize_t n = check_ranges(local, remote, lengths);
                 const uint64_t* l = local.data();
                 const uint64_t* r = remote.data();
                 const uint64_t* len = lengths.data();
                 auto be = b.get();
                 py::gil_scoped_release nogil;
                 std::vector<TransferDesc> descs((size_t)n);
                 for (py::ssize_t i = 0; i < n; ++i)
                     descs[i] = TransferDesc{op, reinterpret_cast<void*>(l[i]), seg, r[i], len[i]};
                 return be->submit(descs.data(), (int)n);
             },
             py::arg("op"), py::arg("seg_id"), py::arg("local"), py::arg("remote"), py::arg("lengths"),
             "Batch handle, INVALID_BATCH on rejection.")
        .def("poll",
             [](NativeBackend& b, BatchHandle h) {
                 auto be = b.get();
                 uint64_t done = 0;
                 XferState s;
                 {
                     py::gil_scoped_release nogil;
                     s = be->poll(h, &done);
                 }
                 return py::make_tuple(s, done);
             },
             py::arg("handle"), "Returns (XferState, bytes_done).")
        .def("wait",
             [](NativeBackend& b, BatchHandle h, int timeout_ms) {
                 auto be = b.get();
                 py::gil_scoped_release nogil;
                 return be->wait(h, timeout_ms);
             },
             py::arg("handle"), py::arg("timeout_ms"))
        .def("release",
             [](NativeBackend& b, BatchHandle h) {
                 auto be = b.get();
                 py::gil_scoped_release nogil;
                 return be->release(h);
             },
             py::arg("handle"), "XFER_OK, or XFER_EBUSY while the batch is still running.")
        .def("close",
             [](NativeBackend& b) {
                 auto be = b.take();
                 py::gil_scoped_release nogil;
                 be.reset();
             })
        .def("__repr__", [](NativeBackend& b) {
            std::shared_ptr<ITransferBackend> be;
            {
                std::lock_guard<std::mutex> lk(b.mu);
                be = b.be;
            }
            if (!be) return std::string("<NativeTransferBackend closed>");
            return std::string("<NativeTransferBackend '") + be->backend_name() + "' segment='" +
                   be->local_segment() + "'>";
        });

    m.def("create_native_backend",
          [](const TransferConfig& cfg) -> std::unique_ptr<NativeBackend> {
              std::unique_ptr<ITransferBackend> be;
              {
                  py::gil_scoped_release nogil;
                  be = create_transfer_backend(cfg);
              }
              if (!be) return nullptr;
              auto nb = std::unique_ptr<NativeBackend>(new NativeBackend());
              nb->be = std::move(be);
              return nb;
          },
          py::arg("config"), "The plugin backend named by config.backend, or None if it is unavailable.");

    // ==================== slot planning ====================
    m.def("resolve_peer_geometry",
          [](const StoreGeometry& mine, py::object published, uint64_t seg_base, uint64_t seg_len) {
              StoreGeometry holder, out;
              const StoreGeometry* layout = nullptr;
              if (!published.is_none()) {
                  holder = published.cast<StoreGeometry>();
                  layout = &holder;
              }
              std::string why;
              const bool ok = resolve_peer_geometry(mine, layout, seg_base, seg_len, out, why);
              return py::make_tuple(ok, ok ? py::cast(out) : py::none(), why);
          },
          py::arg("mine"), py::arg("published"), py::arg("seg_base"), py::arg("seg_len"),
          "Returns (ok, the peer's StoreGeometry with absolute bases or None, why).");

    m.def("plan_slots",
          [](const SlotStore& store, const StoreGeometry& peer, py::object k, py::handle local_slots,
             py::handle remote_slots) -> py::tuple {
              const ComponentType c = kind_of(k);
              const std::vector<int64_t> local = slot_ids(local_slots, "local slots");
              const std::vector<int64_t> remote = slot_ids(remote_slots, "remote slots");
              if (local.size() != remote.size())
                  throw std::invalid_argument("slot lists must be the same length");
              if (local.empty() || local.size() > (size_t)INT_MAX)
                  throw std::invalid_argument("slot lists must not be empty or longer than INT_MAX");
              SlotPlan plan;
              std::string why;
              bool ok;
              {
                  py::gil_scoped_release nogil;
                  ok = plan_slots(store, peer, c, local.data(), remote.data(), (int)local.size(), plan, why);
              }
              if (!ok) return py::make_tuple(py::none(), py::none(), py::none(), why);
              return py::make_tuple(to_array(plan.local), to_array(plan.remote), to_array(plan.length), "");
          },
          py::arg("store"), py::arg("peer"), py::arg("kind"), py::arg("local_slots"), py::arg("remote_slots"),
          "Slot ids: sequences of ints or integer arrays. Returns (local, remote, lengths, \"\") as uint64 arrays, runs "
          "fused, or (None, None, None, why).");

    m.def("check_batch",
          [](const U64Array& local, const U64Array& remote, const U64Array& lengths,
             const std::vector<std::pair<uint64_t, uint64_t>>& regions, uint64_t seg_base,
             uint64_t seg_len) -> py::tuple {
              const py::ssize_t n = check_ranges(local, remote, lengths);
              const uint64_t* l = local.data();
              const uint64_t* r = remote.data();
              const uint64_t* len = lengths.data();
              uint64_t seg_end = 0, total = 0;
              if (n == 0) return py::make_tuple(0, "empty batch");
              if (__builtin_add_overflow(seg_base, seg_len, &seg_end)) return py::make_tuple(0, "segment wraps");
              for (py::ssize_t i = 0; i < n; ++i) {
                  uint64_t lend = 0, rend = 0;
                  if (l[i] == 0 || len[i] == 0)
                      return py::make_tuple(0, "a range has a null address or zero length");
                  if (__builtin_add_overflow(l[i], len[i], &lend) || __builtin_add_overflow(r[i], len[i], &rend))
                      return py::make_tuple(0, "a range wraps around the address space");
                  bool inside = false;
                  for (const auto& [addr, size] : regions)
                      if (l[i] >= addr && lend - addr <= size) { inside = true; break; }
                  if (!inside) return py::make_tuple(0, "a local range is not inside any registered region");
                  if (r[i] < seg_base || rend > seg_end)
                      return py::make_tuple(0, "a remote range escapes the segment");
                  total += len[i];
              }
              return py::make_tuple(total, "");
          },
          py::arg("local"), py::arg("remote"), py::arg("lengths"), py::arg("regions"), py::arg("seg_base"),
          py::arg("seg_len"),
          "(total bytes, \"\") if every range is non-empty, inside one of the local regions ((addr, length) pairs) "
          "and inside [seg_base, +seg_len); else (0, why).");

    m.def("copy_ranges",
          [](XferOp op, const U64Array& local, const U64Array& remote, const U64Array& lengths) {
              const py::ssize_t n = check_ranges(local, remote, lengths);
              const uint64_t* l = local.data();
              const uint64_t* r = remote.data();
              const uint64_t* len = lengths.data();
              py::gil_scoped_release nogil;
              for (py::ssize_t i = 0; i < n; ++i) {
                  void* lp = reinterpret_cast<void*>(l[i]);
                  void* rp = reinterpret_cast<void*>(r[i]);
                  if (op == XferOp::READ) std::memcpy(lp, rp, len[i]);
                  else                    std::memcpy(rp, lp, len[i]);
              }
          },
          py::arg("op"), py::arg("local"), py::arg("remote"), py::arg("lengths"),
          "memcpy each range: READ copies remote to local, WRITE local to remote.");

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
}
