// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "shmradix/transfer.h"

#include "shmradix/config.h"

#ifdef SHMRADIX_HAS_MOONCAKE
#include "mooncake/mooncake_backend.hpp"
#endif

#ifdef SHMRADIX_HAS_NIXL
#include "nixl/nixl_backend.hpp"
#endif

#include <cstdio>

namespace shmradix {

std::unique_ptr<ITransferBackend> create_transfer_backend(const TransferConfig& cfg) {
    if (cfg.backend == "mooncake") {
#ifdef SHMRADIX_HAS_MOONCAKE
        return create_mooncake_backend(cfg);
#else
        fprintf(stderr, "[shmradix] transfer: built without Mooncake "
                        "(ENABLE_MOONCAKE=OFF or headers/lib not found) — "
                        "backend 'mooncake' unavailable\n");
        return nullptr;
#endif
    }

    if (cfg.backend == "nixl") {
#ifdef SHMRADIX_HAS_NIXL
        return create_nixl_backend(cfg);
#else
        fprintf(stderr, "[shmradix] transfer: built without NIXL "
                        "(ENABLE_NIXL=OFF or headers/lib not found) — "
                        "backend 'nixl' unavailable\n");
        return nullptr;
#endif
    }

    fprintf(stderr, "[shmradix] transfer: unknown backend '%s' (want 'mooncake' or 'nixl')\n",
            cfg.backend.c_str());
    return nullptr;
}

} // namespace shmradix
