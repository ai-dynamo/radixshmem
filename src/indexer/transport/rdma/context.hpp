// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RDMA device context + memory region — the verbs foundation every plane builds on.

#pragma once

#include "endpoint.hpp"

#include <cstddef>
#include <cstdint>
#include <infiniband/verbs.h>

namespace shmradix {

class RdmaContext {
public:
    explicit RdmaContext(const char* dev_name = nullptr,
                         int port = 1, int gid_idx = 3);
    ~RdmaContext();

    RdmaContext(const RdmaContext&) = delete;
    RdmaContext& operator=(const RdmaContext&) = delete;

    ibv_context* ctx() const { return ctx_; }
    ibv_pd* pd() const { return pd_; }
    ibv_cq* cq() const { return cq_; }
    int port() const { return port_; }
    int gid_index() const { return gid_idx_; }
    const union ibv_gid& gid() const { return gid_; }

    // QoS knobs at QP RTR (sl / traffic_class); env SHMRADIX_RDMA_SL/_TC, override via set_qos.
    int  sl() const { return sl_; }
    int  traffic_class() const { return tc_; }
    void set_qos(int sl, int tc) { sl_ = sl; tc_ = tc; }

private:
    ibv_context* ctx_ = nullptr;
    ibv_pd* pd_ = nullptr;
    ibv_cq* cq_ = nullptr;
    int port_ = 1;
    int gid_idx_ = 3;
    int sl_ = 5;
    int tc_ = 160;
    union ibv_gid gid_ = {};
};

class RdmaMemoryRegion {
public:
    RdmaMemoryRegion() = default;
    RdmaMemoryRegion(ibv_pd* pd, void* addr, size_t len, int access);
    ~RdmaMemoryRegion();

    RdmaMemoryRegion(RdmaMemoryRegion&& o) noexcept;
    RdmaMemoryRegion& operator=(RdmaMemoryRegion&& o) noexcept;
    RdmaMemoryRegion(const RdmaMemoryRegion&) = delete;
    RdmaMemoryRegion& operator=(const RdmaMemoryRegion&) = delete;

    uint32_t rkey() const;
    uint32_t lkey() const;
    void* addr() const;
    size_t length() const;
    RdmaMrInfo info() const;
    bool valid() const { return mr_ != nullptr; }

private:
    ibv_mr* mr_ = nullptr;
};

} // namespace shmradix
