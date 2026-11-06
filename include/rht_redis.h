// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rht_backend.h"

#include <memory>
#include <string>

namespace radixshmem {

struct PeerFaultCounters;

struct RedisRhtConfig {
    std::string seeds;        // "host:port,..." of any nodes of the cluster
    std::string key_prefix;   // a route lives in HASH <key_prefix><first_hash in hex>
    int         timeout_ms = 50;
};

// One HASH per key, field = rank. AUTH uses RADIXSHMEM_RHT_REDIS_PASSWORD when
// set. Throws on a bad config or a build without hiredis; an unreachable
// cluster only drops ops until it answers.
std::unique_ptr<RhtBackend> make_redis_rht(const RedisRhtConfig& cfg, int my_rank,
                                           PeerFaultCounters* faults);

// Empty if a seed answers with cluster_state:ok, else why not.
std::string redis_rht_check(const RedisRhtConfig& cfg);

} // namespace radixshmem
