// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The RDMA side of a radix-server: the device, the XRC accept loop and DC target
// that peers' one-sided QPs connect to, and the tree MR their walks read. A server
// builds one only when its RHT or its peer index runs on RDMA.

#pragma once

#include "cluster_manifest.h"   // PeerEntry
#include "rdma_transport.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace radixshmem {

// The device and GID index come from RADIXSHMEM_INDEX_DEV / _GID_IDX (default:
// the first device, GID index 3).
struct RdmaHostConfig {
    int         sl = 5;         // the cluster's QoS
    int         tc = 160;
    uint16_t    port = 0;       // XRC accept loop, 0 = OS-assigned
    bool        dc = false;     // also a DC target
    uint64_t    dc_access_key = 0;
    int         rank = 0;       // log prefix
};

class RdmaHost {
public:
    // Opens the device and starts the XRC accept loop; throws on failure.
    explicit RdmaHost(const RdmaHostConfig& cfg);
    ~RdmaHost();

    RdmaHost(const RdmaHost&) = delete;
    RdmaHost& operator=(const RdmaHost&) = delete;

    ibv_pd* pd() const { return ctx_->pd(); }
    // Lets peers READ the tree shm and FAA its counters (remote walks and pins).
    void expose_tree(void* base, size_t size);
    // The RDMA fields of this node's /peers entry.
    void publish_to(PeerEntry& e) const;

private:
    void accept_loop();
    void handle_client(int conn_fd);

    // Declared in bring-up order: members go in reverse, each before what it uses.
    RdmaHostConfig                    cfg_;
    std::unique_ptr<RdmaContext>      ctx_;
    std::unique_ptr<RdmaXrcd>         xrcd_;
    std::unique_ptr<ibv_cq, int (*)(ibv_cq*)> cq_{nullptr, ibv_destroy_cq};   // dormant; XRC needs one
    std::unique_ptr<XrcSrq>           srq_;
    std::unique_ptr<DcTarget>         dct_;
    uint32_t                          dct_num_ = 0;
    std::unique_ptr<RdmaMemoryRegion> tree_mr_;

    int               listen_fd_ = -1;
    uint16_t          port_ = 0;
    std::thread       listener_;
    std::atomic<bool> stopping_{false};
    std::mutex        clients_mu_;
    std::vector<std::unique_ptr<XrcRecvQp>> clients_;   // one per connected SEND QP
};

} // namespace radixshmem
