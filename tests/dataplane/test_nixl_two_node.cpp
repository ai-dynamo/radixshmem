// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// test_nixl_two_node — manual, real cross-machine check for the NIXL backend.
//
// Not in ctest (needs two hosts on the same fabric with the peer already
// running). Exercises ITransferBackend exactly as SlotTransfer would: a
// heap buffer stands in for a SlotStore region.
//
// Usage, run concurrently on two machines on the same RDMA fabric:
//   host A (10.0.0.1): test_nixl_two_node --me 10.0.0.1:14300 \
//       --peer 10.0.0.2:14300 --role server
//   host B (10.0.0.2): test_nixl_two_node --me 10.0.0.2:14300 \
//       --peer 10.0.0.1:14300 --role client --devices mlx5_bond_0
//
// The server just registers a pattern-filled buffer and blocks. The client
// registers its own (zeroed) buffer, opens the server's segment, issues a
// READ pulling the pattern across, and verifies every byte.

#include "shmradix/transfer.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>

using namespace shmradix;

namespace {

unsigned char pattern(uint64_t i) { return (unsigned char)((i * 2654435761u) & 0xff); }

struct Args {
    std::string me, peer, role, devices;
    uint64_t bytes = 1u << 20;  // 1 MiB
};

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (k == "--me") a.me = next();
        else if (k == "--peer") a.peer = next();
        else if (k == "--role") a.role = next();
        else if (k == "--devices") a.devices = next();
        else if (k == "--bytes") a.bytes = strtoull(next().c_str(), nullptr, 0);
    }
    return a;
}

const char* xfer_state_name(XferState s) {
    switch (s) {
        case XferState::PENDING:   return "PENDING";
        case XferState::COMPLETED: return "COMPLETED";
        case XferState::FAILED:    return "FAILED";
        case XferState::TIMEOUT:   return "TIMEOUT";
        case XferState::INVALID:   return "INVALID";
    }
    return "?";
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size()) {
        size_t comma = s.find(',', start);
        if (comma == std::string::npos) comma = s.size();
        if (comma > start) out.push_back(s.substr(start, comma - start));
        start = comma + 1;
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const Args a = parse(argc, argv);
    if (a.me.empty() || a.peer.empty() || (a.role != "server" && a.role != "client")) {
        fprintf(stderr, "usage: %s --me host:port --peer host:port --role server|client "
                        "[--devices mlx5_0,mlx5_1] [--bytes N]\n", argv[0]);
        return 2;
    }

    std::vector<unsigned char> buf(a.bytes);
    if (a.role == "server") {
        for (uint64_t i = 0; i < a.bytes; ++i) buf[i] = pattern(i);
    } else {
        std::memset(buf.data(), 0, buf.size());
    }

    TransferConfig cfg;
    cfg.backend      = "nixl";
    cfg.local_segment = a.me;
    const size_t colon = a.me.rfind(':');
    cfg.ip_or_host   = a.me.substr(0, colon);
    cfg.rpc_port     = strtoull(a.me.substr(colon + 1).c_str(), nullptr, 10);
    cfg.device_filter = split_csv(a.devices);
    cfg.publish      = true;

    auto backend = create_transfer_backend(cfg);
    if (!backend) {
        fprintf(stderr, "create_transfer_backend(nixl) failed — see stderr above\n");
        return 1;
    }
    printf("[%s] agent up as %s (backend=%s)\n", a.role.c_str(),
           backend->local_segment().c_str(), backend->backend_name());

    if (backend->register_memory(buf.data(), buf.size(), "") != XFER_OK) {
        fprintf(stderr, "register_memory failed\n");
        return 1;
    }
    printf("[%s] registered %lu bytes at %p\n", a.role.c_str(),
           (unsigned long)buf.size(), (void*)buf.data());

    if (a.role == "server") {
        printf("[server] ready; waiting for the client to pull (Ctrl+C to stop)\n");
        for (;;) sleep(3600);
    }

    // client
    SegmentInfo seg{};
    const int rc = backend->open_segment(a.peer, seg);
    if (rc != XFER_OK) {
        fprintf(stderr, "open_segment('%s') failed rc=%d\n", a.peer.c_str(), rc);
        return 1;
    }
    printf("[client] opened '%s': base=0x%lx length=%lu\n", a.peer.c_str(),
           (unsigned long)seg.base, (unsigned long)seg.length);
    if (seg.length < a.bytes) {
        fprintf(stderr, "peer segment (%lu bytes) is smaller than requested (%lu)\n",
                (unsigned long)seg.length, (unsigned long)a.bytes);
        return 1;
    }

    TransferDesc d{};
    d.op            = XferOp::READ;
    d.local_addr    = buf.data();
    d.target        = seg.id;
    d.target_offset = seg.base;
    d.length        = a.bytes;

    const auto t0 = std::chrono::steady_clock::now();
    const BatchHandle h = backend->submit(&d, 1);
    if (h == INVALID_BATCH) {
        fprintf(stderr, "submit() failed\n");
        return 1;
    }
    const XferState st = backend->wait(h, 30000);
    const auto t1 = std::chrono::steady_clock::now();
    if (st != XferState::COMPLETED) {
        fprintf(stderr, "transfer did not complete: state=%s\n", xfer_state_name(st));
        return 1;
    }
    backend->release(h);

    uint64_t bad = 0;
    for (uint64_t i = 0; i < a.bytes; ++i)
        if (buf[i] != pattern(i)) ++bad;

    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    const double gib_s = (a.bytes / (double)(1u << 30)) / (us / 1e6);
    printf("[client] transfer done in %.1f us (%.3f GiB/s), mismatched bytes: %lu/%lu\n",
           us, gib_s, (unsigned long)bad, (unsigned long)a.bytes);

    return bad == 0 ? 0 : 1;
}
