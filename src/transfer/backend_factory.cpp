// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "transfer.h"

#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <exception>
#include <unistd.h>

namespace radixshmem {

namespace {

const char* const kBackends[] = {"mooncake"};

std::string plugin_dir() {
    if (const char* dir = ::getenv("RADIXSHMEM_TRANSFER_PLUGIN_DIR")) return dir;
    Dl_info info{};
    if (::dladdr(reinterpret_cast<void*>(&create_transfer_backend), &info) && info.dli_fname) {
        const std::string self = info.dli_fname;
        const auto slash = self.rfind('/');
        if (slash != std::string::npos) return self.substr(0, slash);
    }
    return ".";
}

std::string plugin_path(const std::string& backend) {
    return plugin_dir() + "/libradixshmem_" + backend + ".so";
}

}  // namespace

std::vector<std::string> available_transfer_backends() {
    std::vector<std::string> found;
    for (const char* b : kBackends)
        if (::access(plugin_path(b).c_str(), F_OK) == 0) found.emplace_back(b);
    return found;
}

std::unique_ptr<ITransferBackend> create_transfer_backend(const TransferConfig& cfg) {
    bool known = false;
    for (const char* b : kBackends) known = known || cfg.backend == b;
    if (!known) {
        fprintf(stderr, "[radixshmem] transfer: unknown backend '%s' (the C++ plugins: 'mooncake')\n",
                cfg.backend.c_str());
        return nullptr;
    }
    const std::string path = plugin_path(cfg.backend);
    void* plugin = ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!plugin) {
        fprintf(stderr, "[radixshmem] transfer: backend '%s' unavailable: %s\n", cfg.backend.c_str(),
                ::dlerror());
        return nullptr;
    }
    using Create = ITransferBackend* (*)(const TransferConfig*);
    auto create = reinterpret_cast<Create>(::dlsym(plugin, "radixshmem_create_transfer_backend"));
    if (!create) {
        fprintf(stderr, "[radixshmem] transfer: %s is not a transfer plugin\n", path.c_str());
        return nullptr;
    }
    try {
        return std::unique_ptr<ITransferBackend>(create(&cfg));
    } catch (const std::exception& e) {
        fprintf(stderr, "[radixshmem] transfer: backend '%s' failed: %s\n", cfg.backend.c_str(), e.what());
        return nullptr;
    }
}

} // namespace radixshmem
