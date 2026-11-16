// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Resolve RDMA device (env override) and SKIP (exit 77) if absent. CMake maps 77 → ctest "Skipped".
#pragma once

#include <infiniband/verbs.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <strings.h>

// A Redis RHT and the grpc peer index (both defaults): no plane opens an RDMA device.
inline bool cluster_without_rdma() {
    const char* rht = ::getenv("RADIXSHMEM_RHT_BACKEND");
    const char* peer = ::getenv("RADIXSHMEM_PEER_INDEX_BACKEND");
    return (!rht || strcasecmp(rht, "redis") == 0) && (!peer || strcasecmp(peer, "grpc") == 0);
}

// Sets RADIXSHMEM_INDEX_DEV (default `def`) for the planes on RDMA and returns it;
// exits 77 if a plane needs it and it is absent. Call at top of main(), before forking.
inline std::string resolve_index_dev_or_skip(const char* def) {
    const char* env = ::getenv("RADIXSHMEM_INDEX_DEV");
    std::string want = env ? env : def;
    setenv("RADIXSHMEM_INDEX_DEV", want.c_str(), 1);
    if (cluster_without_rdma()) return want;
    int n = 0;
    ibv_device** list = ibv_get_device_list(&n);
    bool found = false;
    for (int i = 0; list && i < n; i++) {
        if (want == ibv_get_device_name(list[i])) { found = true; break; }
    }
    if (list) ibv_free_device_list(list);
    if (!found) {
        printf("SKIP: RDMA device '%s' not found on this host (%d device(s) "
               "present; set RADIXSHMEM_INDEX_DEV to override)\n",
               want.c_str(), n);
        exit(77);
    }
    return want;
}
