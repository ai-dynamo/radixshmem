#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A stand-in for an inference engine on one node of a data-plane cluster,
driving its radix-server through the public client API only.

Per request: pull_async() the prompt's prefix (local hit + RDMA pull of what
a peer holds), "prefill" the rest (allocate slots, write the KV bytes,
insert), hold the pin while it "decodes", then release it. With SWA / Mamba
pools in the geometry it also publishes, like a hybrid model, the SWA slots
of the trailing window and a Mamba checkpoint at the end of each prompt. Prompts are shared
system prompts plus multi-turn sessions; a session belongs to one node and
sometimes runs on another (the cross-node reuse the pulls serve).

    python radix-monitor/loadgen.py --name /kv --node 0 --nodes 2 --configure   # brings the geometry
    python radix-monitor/loadgen.py --name /kv --node 1 --nodes 2               # takes it from the cluster
"""

import argparse
import random
import signal
import sys
import time

import numpy as np

SESSIONS = 300          # session ids per node


def prompts(block: int):
    return [[random.Random(1000 + p).randrange(1, 1 << 30)
             for _ in range(random.Random(p).randint(8, 48) * block)] for p in range(8)]


def session_tokens(sid: int, turns: int, block: int, system) -> list:
    """Session ``sid``'s first ``turns`` turns; every node derives the same tokens."""
    r = random.Random(sid)
    toks = list(system[min(int(r.paretovariate(1.2)) - 1, len(system) - 1)])
    for _ in range(turns):
        toks += [r.randrange(1, 1 << 30) for _ in range(r.randint(2, 20) * block)]
    return toks


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="the radix-server --name")
    p.add_argument("--endpoint", default="", help="(default: the server's default socket)")
    p.add_argument("--node", type=int, required=True, help="this node's index, 0-based")
    p.add_argument("--nodes", type=int, required=True)
    p.add_argument("--configure", action="store_true", help="bring the geometry (one node of the cluster)")
    p.add_argument("--block", type=int, default=64, help="tokens per block")
    p.add_argument("--slot-bytes", type=int, default=128 << 10, help="KV bytes per block")
    p.add_argument("--swa-slot-bytes", type=int, default=0, help="SWA pool slot bytes, 0 = no SWA pool")
    p.add_argument("--swa-window", type=int, default=4, help="SWA window, blocks")
    p.add_argument("--mamba-slot-bytes", type=int, default=0, help="Mamba checkpoint bytes, 0 = no Mamba pool")
    p.add_argument("--rate", type=float, default=4.0, help="requests per second")
    p.add_argument("--report", type=float, default=30.0, help="seconds between progress lines")
    a = p.parse_args()

    import radixshmem

    geo = radixshmem.Geometry(block_size=a.block, full_slot_bytes=a.slot_bytes,
                              swa_slot_bytes=a.swa_slot_bytes,
                              swa_window_blocks=a.swa_window if a.swa_slot_bytes else 0,
                              mamba_slot_bytes=a.mamba_slot_bytes) if a.configure else None
    c = radixshmem.RadixClient(a.name, geo, **({"endpoint": a.endpoint} if a.endpoint else {}))
    info = c.wait_ready(900)
    print(f"ready: rank {info.rank}/{info.world_size}, block {info.geometry['block_size']} tokens, "
          f"{info.geometry['pools']['full']['num_slots']} slots x {info.geometry['pools']['full']['slot_bytes']} B",
          flush=True)
    block = info.geometry["block_size"]
    pools = info.geometry["pools"]
    fill = {k: bytes(range(256)) * (p["slot_bytes"] // 256) for k, p in pools.items() if p.get("slot_bytes")}
    swa_window = pools.get("swa", {}).get("window_blocks", 0) if "swa" in fill else 0
    SWA, MAMBA = radixshmem.ComponentType.SWA, radixshmem.ComponentType.MAMBA
    system = prompts(block)
    rng = random.Random(a.node)
    mine, pins, stop = {}, [], []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    n = hit_local = hit_remote = written = 0
    last = time.monotonic()
    while not stop:
        u = rng.random()
        if mine and u < 0.6:                       # the next turn of one of ours
            sid = rng.choice(list(mine))
            mine[sid] = turns = min(mine[sid] + 1, 30)
        elif u < 0.8:                              # a session of any node runs here
            sid = a.nodes * rng.randrange(SESSIONS) + rng.randrange(a.nodes)
            turns = rng.randint(1, 6)
        else:                                      # a new session of ours
            sid = a.nodes * rng.randrange(SESSIONS) + a.node
            mine[sid] = turns = 1
        h = radixshmem.compute_hashes(np.asarray(session_tokens(sid, turns, block, system), np.int64), block)
        r = c.pull_async(h).wait(60)
        hit = int(r.common_hit)
        hit_remote += int(r.remote_blocks)
        hit_local += hit - int(r.remote_blocks)
        need = len(h) - hit
        if need > 0:
            try:
                slots = c.allocate_slots(need)
                for s in slots:
                    c.slot_view(int(s))[:] = fill["full"]
                c.insert(h, slots, start=hit)
                written += len(slots)
                if swa_window:                     # the window's SWA state for the trailing blocks
                    swa = c.allocate_slots(min(swa_window, len(h)), SWA)
                    for s in swa:
                        c.slot_view(int(s), SWA)[:] = fill["swa"]
                    c.insert(h, swa, component=SWA)
                if "mamba" in fill:                # one recurrent-state checkpoint at the end
                    m = c.allocate_slots(1, MAMBA)
                    c.slot_view(int(m[0]), MAMBA)[:] = fill["mamba"]
                    c.insert(h, m, component=MAMBA)
            except Exception as e:  # noqa: BLE001 - pool exhausted for a moment
                print(f"insert skipped: {type(e).__name__}: {e}", file=sys.stderr)
        pins.append((time.monotonic() + rng.uniform(0.5, 4.0), r))
        now = time.monotonic()
        for k in range(len(pins) - 1, -1, -1):
            if pins[k][0] <= now:
                pins.pop(k)[1].finalize()
        n += 1
        if now - last >= a.report:
            last = now
            print(f"{time.strftime('%H:%M:%S')} requests={n} blocks: local_hit={hit_local} "
                  f"remote_pulled={hit_remote} written={written}", flush=True)
        time.sleep(rng.expovariate(a.rate))
    for _, r in pins:
        r.finalize()
    c.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
