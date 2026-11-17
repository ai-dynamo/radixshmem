// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Defines the rdma-core functions the RDMA planes call; each forwards to the library loaded by dlopen.
// The verbs.h calls that go through the context ops (post_send, poll_cq, XRC, ...) need no symbol.

#include "verbs_loader.hpp"

#include <cerrno>
#include <dlfcn.h>
#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>
#include <mutex>
#include <stdexcept>

// X(library, return type, name, parameters, arguments, value on failure)
#define RADIXSHMEM_VERBS_FUNCS(X)                                                                    \
    X(ibverbs, ibv_device**, ibv_get_device_list, (int* num_devices), (num_devices), nullptr)        \
    X(ibverbs, const char*, ibv_get_device_name, (ibv_device* device), (device), nullptr)            \
    X(ibverbs, ibv_context*, ibv_open_device, (ibv_device* device), (device), nullptr)               \
    X(ibverbs, int, ibv_close_device, (ibv_context* context), (context), ENOSYS)                     \
    X(ibverbs, int, ibv_query_gid, (ibv_context* context, uint8_t port_num, int index, ibv_gid* gid), \
      (context, port_num, index, gid), ENOSYS)                                                       \
    X(ibverbs, int, ibv_query_port,                                                                  \
      (ibv_context* context, uint8_t port_num, _compat_ibv_port_attr* port_attr),                    \
      (context, port_num, port_attr), ENOSYS)                                                        \
    X(ibverbs, ibv_pd*, ibv_alloc_pd, (ibv_context* context), (context), nullptr)                    \
    X(ibverbs, int, ibv_dealloc_pd, (ibv_pd* pd), (pd), ENOSYS)                                      \
    X(ibverbs, ibv_mr*, ibv_reg_mr_iova2,                                                            \
      (ibv_pd* pd, void* addr, size_t length, uint64_t iova, unsigned int access),                   \
      (pd, addr, length, iova, access), nullptr)                                                     \
    X(ibverbs, int, ibv_dereg_mr, (ibv_mr* mr), (mr), ENOSYS)                                        \
    X(ibverbs, ibv_cq*, ibv_create_cq,                                                               \
      (ibv_context* context, int cqe, void* cq_context, ibv_comp_channel* channel, int comp_vector), \
      (context, cqe, cq_context, channel, comp_vector), nullptr)                                     \
    X(ibverbs, int, ibv_destroy_cq, (ibv_cq* cq), (cq), ENOSYS)                                      \
    X(ibverbs, ibv_qp*, ibv_create_qp, (ibv_pd* pd, ibv_qp_init_attr* qp_init_attr),                 \
      (pd, qp_init_attr), nullptr)                                                                   \
    X(ibverbs, int, ibv_destroy_qp, (ibv_qp* qp), (qp), ENOSYS)                                      \
    X(ibverbs, int, ibv_modify_qp, (ibv_qp* qp, ibv_qp_attr* attr, int attr_mask),                   \
      (qp, attr, attr_mask), ENOSYS)                                                                 \
    X(ibverbs, int, ibv_query_qp,                                                                    \
      (ibv_qp* qp, ibv_qp_attr* attr, int attr_mask, ibv_qp_init_attr* init_attr),                   \
      (qp, attr, attr_mask, init_attr), ENOSYS)                                                      \
    X(ibverbs, ibv_qp_ex*, ibv_qp_to_qp_ex, (ibv_qp* qp), (qp), nullptr)                             \
    X(ibverbs, ibv_srq*, ibv_create_srq, (ibv_pd* pd, ibv_srq_init_attr* srq_init_attr),             \
      (pd, srq_init_attr), nullptr)                                                                  \
    X(ibverbs, int, ibv_destroy_srq, (ibv_srq* srq), (srq), ENOSYS)                                  \
    X(ibverbs, ibv_ah*, ibv_create_ah, (ibv_pd* pd, ibv_ah_attr* attr), (pd, attr), nullptr)         \
    X(ibverbs, int, ibv_destroy_ah, (ibv_ah* ah), (ah), ENOSYS)                                      \
    X(mlx5, ibv_qp*, mlx5dv_create_qp,                                                               \
      (ibv_context* context, ibv_qp_init_attr_ex* qp_attr, mlx5dv_qp_init_attr* mlx5_qp_attr),       \
      (context, qp_attr, mlx5_qp_attr), nullptr)                                                     \
    X(mlx5, mlx5dv_qp_ex*, mlx5dv_qp_ex_from_ibv_qp_ex, (ibv_qp_ex* qp), (qp), nullptr)

namespace radixshmem::verbs {
namespace {

struct Library {
    const char* soname;
    const char* package;
    std::once_flag once;
    void* handle = nullptr;
    std::string error;
};

Library ibverbs{"libibverbs.so.1", "rdma-core (Debian/Ubuntu: libibverbs1 ibverbs-providers; RHEL: libibverbs)"};
Library mlx5{"libmlx5.so.1", "the mlx5 provider of rdma-core (Debian/Ubuntu: ibverbs-providers; RHEL: libibverbs)"};

#define RADIXSHMEM_VERBS_NAME(lib, ret, name, params, args, fail) {&lib, #name},
struct Function {
    Library* lib;
    const char* name;
};
const Function kFunctions[] = {
    RADIXSHMEM_VERBS_FUNCS(RADIXSHMEM_VERBS_NAME){&ibverbs, "ibv_free_device_list"},
};

void* handle(Library& lib) {
    std::call_once(lib.once, [&lib] {
        lib.handle = ::dlopen(lib.soname, RTLD_NOW | RTLD_LOCAL);
        if (!lib.handle) {
            const char* e = ::dlerror();
            lib.error = std::string(lib.soname) + ": " + (e ? e : "cannot be loaded");
            return;
        }
        std::string missing;
        for (const Function& f : kFunctions)
            if (f.lib == &lib && !::dlsym(lib.handle, f.name)) missing += std::string(" ") + f.name;
        if (!missing.empty()) {
            lib.error = std::string(lib.soname) + " lacks" + missing + " (rdma-core too old)";
            lib.handle = nullptr;
        }
    });
    return lib.handle;
}

void require(Library& lib, const char* what) {
    if (!handle(lib)) throw std::runtime_error(std::string(what) + " needs " + lib.package + ": " + lib.error);
}

}  // namespace

void* symbol(Library& lib, const char* name) {
    void* h = handle(lib);
    return h ? ::dlsym(h, name) : nullptr;
}

void require_ibverbs() { require(ibverbs, "RDMA"); }

void require_mlx5() { require(mlx5, "The DC transport"); }

std::string ibverbs_error() { return handle(ibverbs) ? std::string() : ibverbs.error; }

}  // namespace radixshmem::verbs

#define RADIXSHMEM_VERBS_FORWARD(lib, ret, name, params, args, fail)       \
    extern "C" ret(name) params {                                          \
        static const auto fn = reinterpret_cast<decltype(&(name))>(        \
            radixshmem::verbs::symbol(radixshmem::verbs::lib, #name));     \
        if (!fn) {                                                         \
            errno = ENOSYS;                                                \
            return fail;                                                   \
        }                                                                  \
        return fn args;                                                    \
    }
RADIXSHMEM_VERBS_FUNCS(RADIXSHMEM_VERBS_FORWARD)

extern "C" void(ibv_free_device_list)(ibv_device** list) {
    static const auto fn = reinterpret_cast<decltype(&(ibv_free_device_list))>(
        radixshmem::verbs::symbol(radixshmem::verbs::ibverbs, "ibv_free_device_list"));
    if (fn) fn(list);
}
