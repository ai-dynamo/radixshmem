// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Exercise the real backend's submit/poll/release through its public factory,
// with only TransferEngine replaced by a deterministic fault injector.
#include "data_test_utils.h"
#include "mooncake_backend.hpp"
#include "transfer_engine.h"

#include <array>

using namespace radixshmem;

static void test_release_retry(bool completed) {
    TEST_BEGIN(completed ? "completed batch: failed release keeps the handle"
                         : "pending batch: busy/error releases keep the handle");
    mooncake::test::release_calls.clear();
    mooncake::test::retired_batches = 0;
    mooncake::test::transfer_state = completed ? mooncake::TransferStatusEnum::COMPLETED
                                              : mooncake::TransferStatusEnum::PENDING;

    // The engine ID deliberately differs from the backend's opaque handle.
    std::array<char, 64> local{};
    TransferConfig cfg;
    cfg.local_segment = "local:1234";
    cfg.protocol = "tcp";
    auto backend = create_mooncake_backend(cfg);
    assert(backend);
    assert(backend->register_memory(local.data(), local.size()) == XFER_OK);
    SegmentInfo peer;
    assert(backend->open_segment("remote:5678", peer) == XFER_OK);
    TransferDesc desc;
    desc.local_addr = local.data();
    desc.target = peer.id;
    desc.target_offset = peer.base;
    desc.length = local.size();
    const BatchHandle h = backend->submit(&desc, 1);
    assert(h != INVALID_BATCH);

    for (auto code : {mooncake::Status::BUSY, mooncake::Status::ERROR,
                      mooncake::Status::ERROR}) {
        mooncake::test::release_code = code;
        assert(backend->release(h) == (code == mooncake::Status::BUSY
                                      ? XFER_EBUSY : XFER_EBACKEND));
        uint64_t bytes = UINT64_MAX;
        assert(backend->poll(h, &bytes) == (completed ? XferState::COMPLETED
                                                    : XferState::PENDING));
        assert(bytes == (completed ? local.size() : 0));
        assert(mooncake::test::retired_batches == 0);
    }

    mooncake::test::transfer_state = mooncake::TransferStatusEnum::COMPLETED;
    mooncake::test::release_code = mooncake::Status::OK;
    assert(backend->release(h) == XFER_OK);
    assert(mooncake::test::retired_batches == 1);
    assert(mooncake::test::release_calls.size() == 4);
    for (auto id : mooncake::test::release_calls) {
        assert(id == mooncake::test::release_calls.front());
        assert(id != h);
    }
    // Once retired, repeated release must never free the engine ID again.
    assert(backend->poll(h) == XferState::INVALID);
    assert(backend->release(h) == XFER_ENOENT);
    backend.reset();
    assert(mooncake::test::release_calls.size() == 4);
    assert(mooncake::test::retired_batches == 1);
    TEST_PASS();
}

int main() {
    test_release_retry(false);
    test_release_retry(true);
    return tests_report();
}
