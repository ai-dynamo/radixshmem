// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// test_client_borrowed_tree — a client that borrows a tree, as radix-server's
// own RHT client does, and fails to bootstrap leaves the tree to its owner.

#include "local.h"
#include "radix_client.h"
#include "radix_tree.h"
#include "test_utils.h"

#include <sys/mman.h>

#include <cassert>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace radixshmem;

int main() {
    printf("=== test_client_borrowed_tree ===\n\n");
#ifdef RADIXSHMEM_NO_RDMA
    printf("  (skipped: built without RDMA)\n");
    return 77;
#else
    TEST_BEGIN("a borrowing client that fails to bootstrap leaves the tree to its owner");
    const char* name = "/test_client_borrowed_tree";
    shm_unlink(name);
    ShmConfig c;
    c.max_nodes = 256;
    c.max_blocks = 256;
    c.hugepage_path = "";
    auto t = RadixShmemTree::create(name, c);
    static_cast<ShmHeader*>(t.shm_base())->is_distributed = 1;
    auto tree = std::make_unique<LocalTreeConnector>(name);
    RadixClientConfig cc;
    cc.registry = "etcd://127.0.0.1:1";
    cc.bootstrap_timeout_sec = 1;
    bool threw = false;
    try {
        RadixClient client(cc, tree.get());
    } catch (const std::exception&) {
        threw = true;
    }
    assert(threw);
    std::vector<int32_t> slots;
    assert(tree->allocate_slots(4, slots) == 4);
    const uint64_t h[4] = {1, 2, 3, 4};
    assert(tree->insert(h, 4, slots.data(), 4).error == InsertError::OK);
    tree.reset();
    t.destroy();
    TEST_PASS();
    return tests_report();
#endif
}
