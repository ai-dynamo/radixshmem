# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SlotTransfer latency and bandwidth, and the gain from run coalescing, between two local processes.

One process publishes its SlotStore, the other initiates and times transfers after a byte-for-byte round
trip check. Per slot size: batches of 1 / 8 / 64 contiguous slots (one range each) and 64 strided slots (64
ranges, the no-coalescing control). Not in ctest.

    python tests/bench_slot_transfer.py [--backend mooncake|nixl] [--protocol rdma|tcp] [--iters N]
                                        [--slots N] [--devices mlx5_0,mlx5_1] [--out report.md]

--devices restricts the NICs; without it, hosts with a RoCE management NIC can see ~1 s retry spikes in max.
"""

import argparse
import multiprocessing as mp
import sys
import time
import uuid

import numpy as np

from radixshmem._data import (INVALID_BATCH, XFER_OK, SlotPoolConfig, SlotStore, SlotStoreConfig, SlotTransfer,
                              TransferConfig, XferState, create_transfer_backend)

SLOT_SIZES = (4096, 65536, 1048576)


def human(n: int) -> str:
    return f"{n >> 20} MiB" if n >= 1 << 20 else f"{n >> 10} KiB"


def make(args, name: str, slot_bytes: int, publish: bool):
    store = SlotStore.create(SlotStoreConfig(name, SlotPoolConfig(args.slots, slot_bytes), slot_align=4096,
                                             hugepage_path="", prefault=True))
    backend = create_transfer_backend(TransferConfig(
        backend=args.backend, metadata="P2PHANDSHAKE", ip_or_host="127.0.0.1", rpc_port=0,
        protocol=args.protocol, device_filter=args.devices, publish=publish))
    if backend is None:
        sys.exit(f"bench: backend {args.backend} unavailable")
    xfer = SlotTransfer(store, backend)
    if xfer.register_store("cpu:0") != XFER_OK:
        sys.exit("bench: register_store failed")
    return store, backend, xfer


def publisher(args, name, slot_bytes, conn):
    store, backend, xfer = make(args, name, slot_bytes, True)
    conn.send(xfer.local_segment)
    conn.recv()
    backend.close()


def wait(xfer, h):
    if h == INVALID_BATCH:
        sys.exit("bench: submit refused")
    st = xfer.wait(h, 60000)
    xfer.release(h)
    if st != XferState.COMPLETED:
        sys.exit(f"bench: transfer ended {st}")


def verify_roundtrip(store, xfer, peer):
    # Lower half out to the peer's upper half, scrub, pull back: never slot i to slot i, so a transfer that
    # addressed the local store cannot pass.
    half = store.num_slots // 2
    local, remote = list(range(half)), list(range(half, 2 * half))
    view = np.frombuffer(store.pool_view(0), dtype=np.uint8)
    n = half * store.slot_bytes
    golden = np.random.default_rng(1).integers(0, 256, n, dtype=np.uint8)
    view[:n] = golden
    wait(xfer, xfer.write(peer, local, remote)[0])
    view[:n] = 0
    wait(xfer, xfer.read(peer, remote, local)[0])
    return bool(np.array_equal(view[:n], golden))


def measure(store, xfer, peer, read, batch, contiguous, iters):
    half, step = store.num_slots // 2, 1 if contiguous else 2
    local = np.arange(batch, dtype=np.int32) * step
    remote = local + half

    def once():
        return xfer.read(peer, remote, local) if read else xfer.write(peer, local, remote)

    for _ in range(32):   # first-touch faults and endpoint setup are not what this measures
        wait(xfer, once()[0])
    us = []
    for _ in range(iters):
        t = time.perf_counter()
        h, descs = once()
        wait(xfer, h)
        us.append((time.perf_counter() - t) * 1e6)
    us.sort()
    p50 = us[round(0.5 * (len(us) - 1))]
    # Bandwidth off the median: one re-handshake in the run is a one-second sample; max shows the tail.
    return dict(read=read, slot_bytes=store.slot_bytes, batch=batch, contiguous=contiguous, descs=descs,
                p50=p50, p99=us[round(0.99 * (len(us) - 1))], max=us[-1],
                gib_s=batch * store.slot_bytes / (p50 * 1e-6) / 2**30)


def initiator(args, name, slot_bytes, peer_conn, out_conn):
    store, backend, xfer = make(args, name, slot_bytes, False)
    peer = peer_conn.recv()
    if xfer.attach_peer(peer, slot_bytes)[0] != XFER_OK:
        sys.exit(f"bench: attach_peer({peer}) failed")
    if not verify_roundtrip(store, xfer, peer):
        sys.exit(f"bench: round trip verification FAILED at {human(slot_bytes)} slots")
    rows = []
    max_batch = min(64, args.slots // 4)
    for batch in (1, 8, 64):
        if batch <= max_batch:
            for read in (True, False):
                rows.append(measure(store, xfer, peer, read, batch, True, args.iters))
    if max_batch >= 64:
        rows.append(measure(store, xfer, peer, True, 64, False, args.iters))
    out_conn.send(rows)
    peer_conn.send("done")
    backend.close()


def run_size(args, slot_bytes):
    ctx = mp.get_context("fork")   # each child starts its own engine after the fork
    tag = uuid.uuid4().hex[:8]
    a, b = f"/bench_st_a_{tag}", f"/bench_st_b_{tag}"
    pa_conn, pb_conn = ctx.Pipe()
    res_r, res_w = ctx.Pipe(duplex=False)
    pa = ctx.Process(target=publisher, args=(args, a, slot_bytes, pa_conn))
    pb = ctx.Process(target=initiator, args=(args, b, slot_bytes, pb_conn, res_w))
    pa.start()
    pb.start()
    try:
        rows = res_r.recv() if res_r.poll(600) else None
    except EOFError:
        rows = None
    pb.join(60)
    pa.join(60)
    for name in (a, b):
        SlotStore.destroy(name, "")
    if rows is None or pa.exitcode or pb.exitcode:
        sys.exit(f"bench: run failed at {human(slot_bytes)} slots")
    return rows


def report(rows, args, f):
    print("| slot | batch | layout | descs | op | p50 (us) | p99 (us) | max (us) | GiB/s @p50 |", file=f)
    print("|---|---|---|---|---|---|---|---|---|", file=f)
    for r in rows:
        print(f"| {human(r['slot_bytes'])} | {r['batch']} | {'contiguous' if r['contiguous'] else 'strided'} "
              f"| {r['descs']} | {'read' if r['read'] else 'write'} | {r['p50']:.1f} | {r['p99']:.1f} "
              f"| {r['max']:.1f} | {r['gib_s']:.2f} |", file=f)
    print(f"\nbackend={args.backend} protocol={args.protocol} iters={args.iters} slots={args.slots} "
          f"devices={','.join(args.devices) or '(all)'}", file=f)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", default="mooncake", choices=["mooncake", "nixl"])
    ap.add_argument("--protocol", default="rdma", choices=["rdma", "tcp"])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--slots", type=int, default=256)
    ap.add_argument("--devices", type=lambda s: [d for d in s.split(",") if d], default=[])
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.iters <= 0 or args.slots < 8:
        ap.error("need --iters > 0 and --slots >= 8")
    print(f"bench_slot_transfer: backend={args.backend} protocol={args.protocol} slots={args.slots} "
          f"iters={args.iters} devices={','.join(args.devices) or '(all)'}")
    rows = [r for sb in SLOT_SIZES for r in run_size(args, sb)]
    report(rows, args, sys.stdout)
    if args.out:
        with open(args.out, "w") as f:
            print("# bench_slot_transfer\n\nOne range per batch when the slots are contiguous on both sides; "
                  "the strided row is the same bytes with coalescing defeated.\n", file=f)
            report(rows, args, f)


if __name__ == "__main__":
    main()
