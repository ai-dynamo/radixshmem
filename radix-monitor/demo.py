#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A live radixshmem cluster on one host for the Grafana dashboard.

Starts a private etcd, --nodes index-only radix-servers with --monitor-port
(one cluster, ranks from etcd), configures each, and runs one writer process
per node: shared system prompts, multi-turn sessions that mostly stay on their
home node and sometimes run on another, short-lived pins. Point Prometheus at
the printed /sd URL. Ctrl-C stops everything.

    python radix-monitor/demo.py --nodes 4 --monitor-port 9810

Needs etcd on PATH and an RDMA device for the index plane (--index-dev).
Writers are re-exec'd, never fork()'d (the etcd Go runtime does not survive a fork).
"""

import argparse
import os
import random
import signal
import socket
import subprocess
import sys
import tempfile
import time

NAME = "/rmdemo"
BLOCK = 16            # tokens per block
SESSIONS = 200        # session ids per node


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(path: str, needle: str, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path) and needle in open(path, errors="replace").read():
            return
        time.sleep(0.2)
    raise RuntimeError(f"timeout waiting for {needle!r} in {path}")


# ---------------------------------------------------------------- writer

def session_tokens(sid: int, turns: int, prompts) -> list:
    """Session ``sid``'s first ``turns`` turns; any process derives the same tokens."""
    r = random.Random(sid)
    toks = list(prompts[min(int(r.paretovariate(1.2)) - 1, len(prompts) - 1)])
    for _ in range(turns):
        toks += [r.randrange(1, 1 << 30) for _ in range(r.randint(2, 20) * BLOCK)]
    return toks


def writer(i: int, nodes: int, shm: str, rate: float) -> None:
    import numpy as np
    import radixshmem
    from radixshmem import _core

    c = _core.RadixClient(shm)
    prompts = [[random.Random(1000 + p).randrange(1, 1 << 30) for _ in range(random.Random(p).randint(8, 48) * BLOCK)]
               for p in range(8)]
    rng = random.Random(i)
    mine = {}                      # sid -> turns, sessions homed here
    pins = []
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    while not stop:
        # session ids: node k owns k, k + nodes, k + 2 * nodes, ... (SESSIONS per node)
        u = rng.random()
        if mine and u < 0.6:                       # the next turn of one of ours
            sid = rng.choice(list(mine))
            mine[sid] = turns = min(mine[sid] + 1, 30)
        elif u < 0.8:                              # a session of another node lands here
            sid = nodes * rng.randrange(SESSIONS) + rng.randrange(nodes)
            turns = rng.randint(1, 6)
        else:                                      # a new session of ours
            sid = nodes * rng.randrange(SESSIONS) + i
            mine[sid] = turns = 1
        h = radixshmem.compute_hashes(np.asarray(session_tokens(sid, turns, prompts), np.int64), BLOCK)
        q = c.query(h, mask=radixshmem.COMPONENT_MASK_FULL, local_only=True, lock=rng.random() < 0.2)
        hit = int(q.common_hit)
        if q.finalize:
            pins.append((time.monotonic() + rng.uniform(1, 6), q))
        need = len(h) - hit
        if need > 0:
            try:
                slots = c.allocate_slots(need)
                if len(slots) == need:
                    c.insert(h, slots, start=hit)
                elif len(slots):
                    c.recycle_slots(slots)
            except Exception:  # noqa: BLE001 - pool exhausted for a moment
                pass
        now = time.monotonic()
        for k in range(len(pins) - 1, -1, -1):
            if pins[k][0] <= now:
                pins.pop(k)[1].finalize()
        time.sleep(rng.expovariate(rate))
    for _, q in pins:
        q.finalize()


# ---------------------------------------------------------------- orchestrator

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nodes", type=int, default=4)
    p.add_argument("--monitor-port", type=int, default=9810)
    p.add_argument("--monitor-interval", type=float, default=5.0)
    p.add_argument("--slots", type=int, default=6000, help="FULL slots per node")
    p.add_argument("--rate", type=float, default=10.0, help="requests per second per node")
    p.add_argument("--index-dev", default="mlx5_bond_0")
    p.add_argument("--gid-idx", type=int, default=3)
    p.add_argument("--writer", type=int, default=-1, help=argparse.SUPPRESS)
    p.add_argument("--shm", default="", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.writer >= 0:
        writer(a.writer, a.nodes, a.shm, a.rate)
        return 0

    import radixshmem

    work = tempfile.mkdtemp(prefix="radix-monitor-demo-")
    procs = []
    env = dict(os.environ, PYTHONUNBUFFERED="1")

    def launch(cmd, log):
        procs.append(subprocess.Popen(cmd, stdout=open(os.path.join(work, log), "ab"),
                                      stderr=subprocess.STDOUT, env=env))
        return procs[-1]

    try:
        cport, pport = free_port(), free_port()
        launch(["etcd", "--name", "demo", "--data-dir", os.path.join(work, "etcd"),
                "--listen-client-urls", f"http://127.0.0.1:{cport}",
                "--advertise-client-urls", f"http://127.0.0.1:{cport}",
                "--listen-peer-urls", f"http://127.0.0.1:{pport}",
                "--initial-advertise-peer-urls", f"http://127.0.0.1:{pport}",
                "--initial-cluster", f"demo=http://127.0.0.1:{pport}"], "etcd.log")
        time.sleep(2)
        ep = lambda i: f"unix://{work}/node{i}.sock"  # noqa: E731
        for i in range(a.nodes):
            launch([sys.executable, "-m", "radixshmem.cli", "--name", NAME, "--node-name", f"node{i}",
                    "--registry", f"etcd://127.0.0.1:{cport}", "--cluster-id", "rmdemo",
                    "--expected-min-nodes", str(a.nodes), "--rpc-address", "0.0.0.0",
                    "--endpoint", ep(i), "--index-dev", a.index_dev, "--gid-idx", str(a.gid_idx),
                    "--interval", "0", "--transfer-log", "", "--monitor-port", str(a.monitor_port),
                    "--monitor-interval", str(a.monitor_interval)]
                   + (["--decode"] if i == a.nodes - 1 else []), f"server{i}.log")
        for i in range(a.nodes):
            wait_for(os.path.join(work, f"server{i}.log"), "Waiting for a client's geometry", 60)
            cl = radixshmem.RadixClient(NAME, endpoint=ep(i))
            cl.configure(radixshmem.Geometry(block_size=BLOCK, full_slots=a.slots))
            cl.close()
        shms = []
        for i in range(a.nodes):
            wait_for(os.path.join(work, f"server{i}.log"), "Cluster ready", 120)
            cl = radixshmem.RadixClient(NAME, endpoint=ep(i))
            shms.append(cl.wait_ready(30).index_name)
            cl.close()
        for i in range(a.nodes):
            launch([sys.executable, os.path.abspath(__file__), "--writer", str(i), "--nodes", str(a.nodes),
                    "--shm", shms[i], "--rate", str(a.rate)], f"writer{i}.log")
        mons = [l.split()[1] for i in range(a.nodes)
                for l in open(os.path.join(work, f"server{i}.log")) if l.strip().startswith("Monitor:")]
        print(f"cluster rmdemo: {a.nodes} nodes, logs in {work}")
        print("monitors: " + " ".join(mons))
        print(f"Prometheus http_sd_configs url: {mons[0].rsplit('/', 1)[0]}/sd", flush=True)
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        while all(pr.poll() is None for pr in procs):
            time.sleep(1)
        print("a process exited; stopping", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        # writers, then servers (they unpublish from etcd on the way out), then etcd
        n = a.nodes
        for group in (procs[1 + n:], procs[1:1 + n], procs[:1]):
            for pr in group:
                if pr.poll() is None:
                    pr.send_signal(signal.SIGTERM)
            for pr in group:
                try:
                    pr.wait(30)
                except subprocess.TimeoutExpired:
                    pr.kill()


if __name__ == "__main__":
    sys.exit(main())
