// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// bench_slot_transfer — SlotTransfer throughput and the gain from run coalescing.
//
// Two processes: one publishes its SlotStore, the other initiates and times
// transfers. The working set is verified byte for byte before timing.
//
// Matrix, per slot size: batch 1 / 8 / 64 contiguous (one request per batch)
// and batch 64 strided (64 requests per batch, the no-coalescing control).
//
// Not in ctest; correctness of the same path is covered by test_slot_transfer_e2e.
//
// Usage:
//   bench_slot_transfer [--protocol rdma|tcp] [--iters N] [--slots N]
//                       [--out PATH] [--devices mlx5_0,mlx5_1,...]
//
// --devices restricts the NICs used; without it, hosts with a RoCE management
// NIC can see ~1 s retry spikes in the `max` column.

#include "slot_store.h"
#include "slot_transfer.h"
#include "transfer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>
#include <sys/wait.h>

using namespace radixshmem;
using Clock = std::chrono::steady_clock;

namespace {

struct Options {
    std::string protocol = "rdma";
    std::string backend  = "mooncake";     // the only transfer backend; printed
                                           // into the report footer
    int         iters    = 200;
    uint64_t    slots    = 256;
    std::string out;                       // empty = stdout only
    std::vector<std::string> devices;      // empty = whatever the engine finds
};

// Trivially copyable on purpose: the measuring child ships these straight down
// a pipe to the parent that prints them.
struct Row {
    bool        is_read;
    uint64_t    slot_bytes;
    int         batch;
    bool        contiguous;
    int         descs;                     // what the coalescer actually produced
    double      p50_us, p99_us, max_us;
    double      gib_s;
};

const uint64_t kSlotSizes[] = {4096, 65536, 1048576};

double percentile(std::vector<double>& v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t i = (size_t)std::llround(q * (double)(v.size() - 1));
    return v[i];
}

unsigned char pattern(uint64_t slot, uint64_t i) {
    return (unsigned char)((slot * 37 + i * 13 + 7) & 0xff);
}

void paint(SlotStore& s, uint64_t slot) {
    auto* p = (unsigned char*)s.slot_addr(slot);
    for (uint64_t i = 0; i < s.slot_bytes(); ++i) p[i] = pattern(slot, i);
}

// ---------------------------------------------------------------- correctness
// Push the lower half of the store into the peer's upper half, scrub locally,
// pull it back, compare. Runs before the timing so a broken path never gets
// reported as a fast one.
//
// Lower half out, upper half in — never slot i to slot i, so a transfer that
// silently addressed the local store cannot pass as a round trip.
bool verify_roundtrip(SlotStore& s, SlotTransfer& xfer, const std::string& peer,
                      uint64_t n) {
    const uint64_t h = n / 2;
    std::vector<int32_t> local(h), remote(h);
    for (uint64_t i = 0; i < h; ++i) {
        local[i]  = (int32_t)i;
        remote[i] = (int32_t)(h + i);
        paint(s, i);
    }

    std::vector<unsigned char> golden(h * s.slot_bytes());
    std::memcpy(golden.data(), s.slot_addr(0), golden.size());

    BatchHandle hw = xfer.write(peer, local.data(), remote.data(), (int)h);
    if (hw == INVALID_BATCH || xfer.wait(hw, 60000) != XferState::COMPLETED) return false;
    xfer.release(hw);

    std::memset(s.slot_addr(0), 0x00, golden.size());

    BatchHandle hr = xfer.read(peer, remote.data(), local.data(), (int)h);
    if (hr == INVALID_BATCH || xfer.wait(hr, 60000) != XferState::COMPLETED) return false;
    xfer.release(hr);

    return std::memcmp(golden.data(), s.slot_addr(0), golden.size()) == 0;
}

// -------------------------------------------------------------------- timing
Row measure(SlotStore& s, SlotTransfer& xfer, const std::string& peer, XferOp op,
            int batch, bool contiguous, int iters) {
    const uint64_t sb   = s.slot_bytes();
    const uint64_t half = s.num_slots() / 2;
    const int      step = contiguous ? 1 : 2;

    std::vector<int32_t> local(batch), remote(batch);
    for (int i = 0; i < batch; ++i) {
        local[i]  = (int32_t)(i * step);
        remote[i] = (int32_t)(half + (uint64_t)(i * step));
    }

    int descs = 0;
    auto once = [&]() -> BatchHandle {
        return op == XferOp::READ
                   ? xfer.read(peer, remote.data(), local.data(), batch, &descs)
                   : xfer.write(peer, local.data(), remote.data(), batch, &descs);
    };

    // Warm the path: first-touch page faults and RDMA endpoint setup are not
    // what this is measuring. Mooncake may need several attempts before it
    // settles on a device pair that hand-shakes, so this is generous.
    for (int i = 0; i < 32; ++i) {
        BatchHandle h = once();
        if (h == INVALID_BATCH) { fprintf(stderr, "bench: submit failed\n"); exit(1); }
        xfer.wait(h, 60000);
        xfer.release(h);
    }

    std::vector<double> us;
    us.reserve((size_t)iters);
    for (int i = 0; i < iters; ++i) {
        const auto a = Clock::now();
        BatchHandle h = once();
        if (h == INVALID_BATCH) { fprintf(stderr, "bench: submit failed\n"); exit(1); }
        if (xfer.wait(h, 60000) != XferState::COMPLETED) {
            fprintf(stderr, "bench: transfer did not complete\n"); exit(1);
        }
        xfer.release(h);
        us.push_back(std::chrono::duration<double, std::micro>(Clock::now() - a).count());
    }

    Row r;
    r.is_read    = (op == XferOp::READ);
    r.slot_bytes = sb;
    r.batch      = batch;
    r.contiguous = contiguous;
    r.descs      = descs;
    r.p50_us     = percentile(us, 0.50);
    r.p99_us     = percentile(us, 0.99);
    r.max_us     = percentile(us, 1.00);
    // Bandwidth off the median, not off the total: one endpoint re-handshake in
    // the middle of a run is a 1-second sample, and averaging that in reports a
    // link that does not exist. The tail is what `max` is for.
    r.gib_s      = ((double)batch * sb) / (r.p50_us * 1e-6) / (1024.0 * 1024.0 * 1024.0);
    return r;
}

// Everything measured for one slot size, on a store that is already attached.
void run_matrix(SlotStore& s, SlotTransfer& xfer, const std::string& peer,
                int iters, std::vector<Row>& out) {
    const int max_batch = (int)std::min<uint64_t>(64, s.num_slots() / 4);
    for (int batch : {1, 8, 64}) {
        if (batch > max_batch) continue;
        for (XferOp op : {XferOp::READ, XferOp::WRITE})
            out.push_back(measure(s, xfer, peer, op, batch, true, iters));
    }
    // The control: same byte count, coalescing defeated by the stride.
    if (max_batch >= 64)
        out.push_back(measure(s, xfer, peer, XferOp::READ, 64, false, iters));
}

// -------------------------------------------------------------------- report
std::string human(uint64_t bytes) {
    char b[32];
    if (bytes >= (1u << 20)) snprintf(b, sizeof b, "%lu MiB", (unsigned long)(bytes >> 20));
    else                     snprintf(b, sizeof b, "%lu KiB", (unsigned long)(bytes >> 10));
    return b;
}

// How the run was allowed to pick HCAs, for the banner and the report footer.
std::string device_list(const Options& o) {
    std::string devs;
    for (const auto& d : o.devices) devs += (devs.empty() ? "" : ",") + d;
    return devs.empty() ? "(all)" : devs;
}

void report(const std::vector<Row>& rows, const Options& o, FILE* f) {
    fprintf(f, "| slot | batch | layout | descs | op | p50 (us) | p99 (us) | max (us) | GiB/s @p50 |\n");
    fprintf(f, "|---|---|---|---|---|---|---|---|---|\n");
    for (const Row& r : rows) {
        fprintf(f, "| %s | %d | %s | %d | %s | %.1f | %.1f | %.1f | %.2f |\n",
                human(r.slot_bytes).c_str(), r.batch,
                r.contiguous ? "contiguous" : "strided", r.descs,
                r.is_read ? "read" : "write",
                r.p50_us, r.p99_us, r.max_us, r.gib_s);
    }
    // The device list belongs in the recorded numbers: which HCAs the run was
    // allowed to use is the difference between a clean tail and second-long
    // stalls, so a table without it cannot be compared against another.
    fprintf(f, "\nbackend=%s protocol=%s iters=%d slots=%lu devices=%s\n",
            o.backend.c_str(), o.protocol.c_str(), o.iters, (unsigned long)o.slots,
            device_list(o).c_str());
}

void write_report(const std::vector<Row>& rows, const Options& o) {
    report(rows, o, stdout);
    if (o.out.empty()) return;
    FILE* f = fopen(o.out.c_str(), "w");
    if (!f) { fprintf(stderr, "bench: cannot write %s\n", o.out.c_str()); return; }
    fprintf(f, "# bench_slot_transfer\n\n");
    fprintf(f, "One request per batch when the slots are contiguous on both sides; "
               "the strided row is the same bytes with coalescing defeated.\n\n");
    report(rows, o, f);
    fclose(f);
    fprintf(stderr, "wrote %s\n", o.out.c_str());
}

// -------------------------------------------------------------------- config
std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size()) {
        const size_t comma = s.find(',', start);
        const size_t end = (comma == std::string::npos) ? s.size() : comma;
        std::string tok = s.substr(start, end - start);
        if (!tok.empty()) out.push_back(tok);
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return out;
}

SlotStoreConfig store_cfg(const std::string& name, uint64_t slots, uint64_t sb) {
    SlotStoreConfig c;
    c.name = name;
    c.full.num_slots = slots;
    c.full.slot_bytes = sb;
    c.slot_align = 4096;
    c.hugepage_path = "";
    c.prefault = true;
    return c;
}

TransferConfig xfer_cfg(const Options& o, bool publish) {
    TransferConfig c;
    c.backend = o.backend;
    c.metadata = "P2PHANDSHAKE";
    c.ip_or_host = "127.0.0.1";
    c.rpc_port = 0;
    c.protocol = o.protocol;
    c.device_filter = o.devices;
    c.publish = publish;
    // local_segment is left empty on purpose: the Mooncake engine derives its
    // own name from ip_or_host and the port it manages to bind, and naming it
    // here would give it a host it cannot resolve.
    return c;
}

void write_str(int fd, const std::string& s) {
    uint32_t n = (uint32_t)s.size();
    if (::write(fd, &n, sizeof n) != (ssize_t)sizeof n) _exit(90);
    if (::write(fd, s.data(), n) != (ssize_t)n) _exit(91);
}
bool read_str(int fd, std::string& out) {
    uint32_t n = 0;
    if (::read(fd, &n, sizeof n) != (ssize_t)sizeof n) return false;
    out.resize(n);
    return ::read(fd, &out[0], n) == (ssize_t)n;
}

// ------------------------------------------------------------- engine (2 proc)
// One slot size per fork pair: an engine per process, and a fresh one per size
// keeps registrations from piling up.
bool run_size(const Options& o, uint64_t sb, std::vector<Row>& rows) {
    int name_pipe[2], done_pipe[2], res_pipe[2];
    if (::pipe(name_pipe) || ::pipe(done_pipe) || ::pipe(res_pipe)) return false;

    const std::string shm_a = "/bench_st_a", shm_b = "/bench_st_b";
    SlotStore::destroy(shm_a, "");
    SlotStore::destroy(shm_b, "");

    pid_t pa = fork();
    if (pa == 0) {
        ::close(name_pipe[0]); ::close(done_pipe[1]);
        ::close(res_pipe[0]);  ::close(res_pipe[1]);
        SlotStore s = SlotStore::create(store_cfg(shm_a, o.slots, sb));
        auto be = create_transfer_backend(xfer_cfg(o, true));
        if (!be) _exit(2);
        SlotTransfer xfer(s, *be);
        if (xfer.register_store("cpu:0") != XFER_OK) _exit(3);
        write_str(name_pipe[1], xfer.local_segment());
        char sig = 0;
        _exit(::read(done_pipe[0], &sig, 1) == 1 && sig == 'k' ? 0 : 4);
    }

    pid_t pb = fork();
    if (pb == 0) {
        ::close(name_pipe[1]); ::close(done_pipe[0]); ::close(res_pipe[0]);
        SlotStore s = SlotStore::create(store_cfg(shm_b, o.slots, sb));
        auto be = create_transfer_backend(xfer_cfg(o, false));
        if (!be) _exit(2);
        SlotTransfer xfer(s, *be);
        if (xfer.register_store("cpu:0") != XFER_OK) _exit(3);

        std::string peer;
        if (!read_str(name_pipe[0], peer) || peer.empty()) _exit(4);
        if (xfer.attach_peer(peer, sb) != XFER_OK) _exit(5);
        if (!verify_roundtrip(s, xfer, peer, o.slots)) {
            fprintf(stderr, "bench: round trip verification FAILED at %s slots\n",
                    human(sb).c_str());
            _exit(6);
        }

        std::vector<Row> mine;
        run_matrix(s, xfer, peer, o.iters, mine);

        // Rows are POD, so the parent can take them straight off the pipe.
        uint32_t n = (uint32_t)mine.size();
        if (::write(res_pipe[1], &n, sizeof n) != (ssize_t)sizeof n) _exit(7);
        for (const Row& r : mine)
            if (::write(res_pipe[1], &r, sizeof r) != (ssize_t)sizeof r) _exit(8);
        char sig = 'k';
        if (::write(done_pipe[1], &sig, 1) != 1) _exit(9);
        _exit(0);
    }

    ::close(name_pipe[0]); ::close(name_pipe[1]);
    ::close(done_pipe[0]); ::close(done_pipe[1]);
    ::close(res_pipe[1]);

    uint32_t n = 0;
    bool got = ::read(res_pipe[0], &n, sizeof n) == (ssize_t)sizeof n;
    for (uint32_t i = 0; got && i < n; ++i) {
        Row r{};
        if (::read(res_pipe[0], &r, sizeof r) != (ssize_t)sizeof r) { got = false; break; }
        rows.push_back(r);
    }
    ::close(res_pipe[0]);

    int st = 0;
    waitpid(pb, &st, 0);
    const bool ok_b = WIFEXITED(st) && WEXITSTATUS(st) == 0;
    waitpid(pa, &st, 0);
    const bool ok_a = WIFEXITED(st) && WEXITSTATUS(st) == 0;

    SlotStore::destroy(shm_a, "");
    SlotStore::destroy(shm_b, "");
    return got && ok_a && ok_b;
}

} // namespace

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (a == "--protocol") o.protocol = next();
        else if (a == "--iters")    o.iters    = atoi(next().c_str());
        else if (a == "--slots")    o.slots    = strtoull(next().c_str(), nullptr, 10);
        else if (a == "--out")      o.out      = next();
        else if (a == "--devices")  o.devices  = split_csv(next());
        else {
            fprintf(stderr, "usage: %s [--protocol rdma|tcp]"
                            " [--iters N] [--slots N] [--out PATH]"
                            " [--devices mlx5_0,mlx5_1,...]\n", argv[0]);
            return 2;
        }
    }
    if (o.iters <= 0 || o.slots < 8) {
        fprintf(stderr, "bench: need --iters > 0 and --slots >= 8\n");
        return 2;
    }

    printf("bench_slot_transfer: backend=%s protocol=%s slots=%lu iters=%d devices=%s\n",
           o.backend.c_str(), o.protocol.c_str(), (unsigned long)o.slots, o.iters,
           device_list(o).c_str());

    std::vector<Row> rows;
    for (uint64_t sb : kSlotSizes) {
        if (!run_size(o, sb, rows)) {
            fprintf(stderr, "bench: run failed at %s slots\n", human(sb).c_str());
            return 1;
        }
    }
    write_report(rows, o);
    return 0;
}
