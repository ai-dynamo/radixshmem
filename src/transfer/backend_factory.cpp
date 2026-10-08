// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "transfer.h"

#include "radixshmem_config.h"

#ifdef RADIXSHMEM_HAS_MOONCAKE
#include "mooncake/mooncake_backend.hpp"
#endif

#ifdef RADIXSHMEM_HAS_NIXL
#include "nixl/nixl_backend.hpp"
#endif

#include <cstdio>

namespace radixshmem {

std::unique_ptr<ITransferBackend> create_transfer_backend(const TransferConfig& cfg) {
    if (cfg.backend == "mooncake") {
#ifdef RADIXSHMEM_HAS_MOONCAKE
        return create_mooncake_backend(cfg);
#else
        fprintf(stderr, "[radixshmem] transfer: built without Mooncake "
                        "(ENABLE_MOONCAKE=OFF or headers/lib not found) — "
                        "backend 'mooncake' unavailable\n");
        return nullptr;
#endif
    }

    if (cfg.backend == "nixl") {
#ifdef RADIXSHMEM_HAS_NIXL
        return create_nixl_backend(cfg);
#else
        fprintf(stderr, "[radixshmem] transfer: built without NIXL "
                        "(ENABLE_NIXL=OFF or headers/lib not found) — "
                        "backend 'nixl' unavailable\n");
        return nullptr;
#endif
    }

    fprintf(stderr, "[radixshmem] transfer: unknown backend '%s' (want 'mooncake' or 'nixl')\n",
            cfg.backend.c_str());
    return nullptr;
}

} // namespace radixshmem
