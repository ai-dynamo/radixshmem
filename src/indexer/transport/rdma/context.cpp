// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "context.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <infiniband/verbs.h>
#include <stdexcept>

namespace radixshmem {

RdmaContext::RdmaContext(const char* dev_name, int port, int gid_idx)
    : port_(port), gid_idx_(gid_idx) {
    // QoS defaults from env; set_qos() may override before QPs connect.
    if (const char* s = ::getenv("RADIXSHMEM_RDMA_SL")) sl_ = atoi(s);
    if (const char* t = ::getenv("RADIXSHMEM_RDMA_TC")) tc_ = atoi(t);

    int num_devices = 0;
    ibv_device** dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list || num_devices == 0)
        throw std::runtime_error("No RDMA devices found");

    ibv_device* dev = nullptr;
    if (dev_name) {
        for (int i = 0; i < num_devices; i++) {
            if (strcmp(ibv_get_device_name(dev_list[i]), dev_name) == 0) {
                dev = dev_list[i];
                break;
            }
        }
        if (!dev) {
            ibv_free_device_list(dev_list);
            throw std::runtime_error(std::string("RDMA device not found: ") + dev_name);
        }
    } else {
        dev = dev_list[0];
    }

    ctx_ = ibv_open_device(dev);
    ibv_free_device_list(dev_list);
    if (!ctx_)
        throw std::runtime_error("ibv_open_device failed");

    pd_ = ibv_alloc_pd(ctx_);
    if (!pd_) {
        ibv_close_device(ctx_);
        throw std::runtime_error("ibv_alloc_pd failed");
    }

    cq_ = ibv_create_cq(ctx_, 128, nullptr, nullptr, 0);
    if (!cq_) {
        ibv_dealloc_pd(pd_);
        ibv_close_device(ctx_);
        throw std::runtime_error("ibv_create_cq failed");
    }

    if (ibv_query_gid(ctx_, port_, gid_idx_, &gid_) != 0) {
        ibv_destroy_cq(cq_);
        ibv_dealloc_pd(pd_);
        ibv_close_device(ctx_);
        throw std::runtime_error("ibv_query_gid failed");
    }
}

RdmaContext::~RdmaContext() {
    if (cq_) ibv_destroy_cq(cq_);
    if (pd_) ibv_dealloc_pd(pd_);
    if (ctx_) ibv_close_device(ctx_);
}

// ==================== RdmaMemoryRegion ====================

RdmaMemoryRegion::RdmaMemoryRegion(ibv_pd* pd, void* addr, size_t len,
                                   int access) {
    mr_ = ibv_reg_mr(pd, addr, len, access);
    if (!mr_)
        throw std::runtime_error("ibv_reg_mr failed: len=" +
                                 std::to_string(len / (1024 * 1024)) + "MB");
    fprintf(stderr, "[radixshmem] MR registered: %p len=%.1f MB rkey=%u\n",
            addr, len / (1024.0 * 1024.0), mr_->rkey);
}

RdmaMemoryRegion::~RdmaMemoryRegion() {
    if (mr_) ibv_dereg_mr(mr_);
}

RdmaMemoryRegion::RdmaMemoryRegion(RdmaMemoryRegion&& o) noexcept
    : mr_(o.mr_) {
    o.mr_ = nullptr;
}

RdmaMemoryRegion& RdmaMemoryRegion::operator=(RdmaMemoryRegion&& o) noexcept {
    if (this != &o) {
        if (mr_) ibv_dereg_mr(mr_);
        mr_ = o.mr_;
        o.mr_ = nullptr;
    }
    return *this;
}

uint32_t RdmaMemoryRegion::rkey() const { return mr_->rkey; }
uint32_t RdmaMemoryRegion::lkey() const { return mr_->lkey; }
void* RdmaMemoryRegion::addr() const { return mr_->addr; }
size_t RdmaMemoryRegion::length() const { return mr_->length; }

RdmaMrInfo RdmaMemoryRegion::info() const {
    return {mr_->rkey, (uint64_t)mr_->addr, mr_->length};
}

} // namespace radixshmem
