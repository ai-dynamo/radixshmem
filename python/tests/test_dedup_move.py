#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dedup that moves data, over a real data plane. node0 holds a path with SWA
windows and Mamba checkpoints at 8, 16 and 24, a short branch at 8 and a long
one at 4; node1 pull_async()s [0, 16). node0 then writes the tail [16, 24),
the short branch and the cells at 8 into node1 and drops its copy up to the
long branch; node1 serves all of it with the writer's bytes.

    python3 test_dedup_move.py [--index-dev mlx5_bond_0] [--transfer-dev mlx5_bond_0] [--gid-idx 3]

Needs etcd, an RDMA device and the Mooncake backend; exits 77 when missing.
"""

import argparse
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import test_p2p_pull as p2p

NAME = "/py_dmove"
WINDOW = 2


def paths():
    import numpy as np
    import radixshmem

    main = list(range(1000, 1024))
    h = lambda tokens: radixshmem.compute_hashes(np.asarray(tokens, dtype=np.int64), 1)  # noqa: E731
    return {"main": h(main), "side": h(main[:8] + list(range(2000, 2004))),
            "long": h(main[:4] + list(range(3000, 3020)))}


def word(kind: int, block_hash) -> int:
    return (int(block_hash) ^ (0x5157 * (kind + 1))) & 0xFFFFFFFFFFFFFFFF


def client_main(role: str, endpoint: str) -> int:
    import numpy as np
    import radixshmem

    full, swa, mamba = radixshmem.ComponentType.FULL, radixshmem.ComponentType.SWA, radixshmem.ComponentType.MAMBA
    ALL = radixshmem.COMPONENT_MASK_FULL | radixshmem.COMPONENT_MASK_SWA | radixshmem.COMPONENT_MASK_MAMBA
    c = radixshmem.RadixClient(NAME, endpoint=endpoint)
    st = c.store
    pools = {k: np.frombuffer(st.pool_view(k), dtype=np.uint64).reshape(int(st.pool(k).num_slots),
                                                                         int(st.pool(k).slot_bytes) // 8)
             for k in (full, swa, mamba)}
    p = paths()
    out = {"role": role}

    def fill(kind, slots, hs):
        for s, x in zip(slots, hs):
            pools[kind][int(s)] = word(int(kind), x)

    def same(kind, slots, hs):
        return len(slots) == len(hs) and all(bool((pools[kind][int(s)] == word(int(kind), x)).all())
                                             for s, x in zip(slots, hs))

    def local(hs):
        q = c.query(hs, mask=ALL, local_only=True, lock=False, update_meta=False)
        n = int(q.common_hit)
        slots = np.concatenate([np.asarray(f[2]) for f in q.full_fragments])[:n] if q.full_fragments else []
        cells = list(q.swa_slots)
        ck = [int(q.mamba[1])] if q.mamba is not None else []
        return [n, same(full, slots, hs[:n]), same(swa, cells, hs[n - len(cells):n]) and len(cells) == WINDOW,
                same(mamba, ck, hs[n - 1:n])]

    if role == "writer":
        errors = set()
        for name, ends in (("main", (8, 16, 24)), ("side", (12,)), ("long", (24,))):
            h = p[name]
            have = int(c.query(h, local_only=True, update_meta=False).common_hit)
            slots = c.allocate_slots(len(h) - have)
            fill(full, slots, h[have:])
            errors.add(str(c.insert(h, slots, start=have).error))
            for e in ends:
                s = c.allocate_slots(WINDOW, swa)
                fill(swa, s, h[e - WINDOW:e])
                errors.add(str(c.insert(h[:e], s, start=0, component=swa).error))
                s = c.allocate_slots(1, mamba)
                fill(mamba, s, h[e - 1:e])
                errors.add(str(c.insert(h[:e], s, component=mamba).error))
        c.flush()
        out["insert_errors"] = sorted(errors)
    elif role == "reader":
        h = p["main"]
        r = c.pull_async(h[:16], mask=ALL).wait(120)
        out.update(hit=int(r.common_hit), remote=int(r.remote_blocks), source=int(r.source_rank))
        r.finalize()
        src, deadline = out["source"], time.monotonic() + 20
        while time.monotonic() < deadline and (
                int(c.remote_query(src, h[:16], False, False).total_hit_length) != 4 or local(h)[0] != 24):
            time.sleep(0.05)
        for name, hs in (("source_main", h), ("source_side", p["side"]), ("source_long", p["long"])):
            out[name] = int(c.remote_query(src, hs, False, False).total_hit_length)
        out.update(main=local(h), side=local(p["side"]), main8=local(h[:8]))
    elif role == "health":
        out["health"] = int(c.check_health())
        out["report"] = c.health_report().describe().strip()
    print("RESULT " + json.dumps(out), flush=True)
    c.close()
    return 0


def run_clients(roles_endpoints, timeout=240):
    procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "--client", role, ep],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for role, ep in roles_endpoints]
    results = []
    for proc, (role, _) in zip(procs, roles_endpoints):
        out, err = proc.communicate(timeout=timeout)
        lines = [ln for ln in out.splitlines() if ln.startswith("RESULT ")]
        if not lines:
            raise RuntimeError(f"client {role} rc={proc.returncode}\nstdout:\n{out}\nstderr:\n{err[-3000:]}")
        results.append(json.loads(lines[0][len("RESULT "):]))
    return results


def orchestrate(a) -> int:
    import radixshmem

    if (not shutil.which("etcd") or not os.path.isdir(f"/sys/class/infiniband/{a.index_dev}")
            or not hasattr(radixshmem, "_data") or "mooncake" not in radixshmem._data.TRANSFER_BACKENDS
            or not hasattr(radixshmem._core, "EtcdKv")):
        print(f"SKIP: needs etcd, RDMA device {a.index_dev!r} and radixshmem built with etcd + Mooncake")
        return 77
    workdir = tempfile.mkdtemp(prefix="radixshmem-dmove-")
    cluster_id = f"dmove{os.getpid()}"
    etcd_proc, servers, logs = None, [], []
    try:
        etcd_proc, cport = p2p.start_etcd(workdir)
        registry = f"etcd://127.0.0.1:{cport}"
        endpoints = [f"unix://{workdir}/node{i}.sock" for i in range(2)]
        print(f"etcd on {registry}   logs in {workdir}")

        print("\n[1] two radix-server processes with FULL, SWA and Mamba pools")
        for i in range(2):
            logs.append(os.path.join(workdir, f"server{i}.log"))
            cmd = [sys.executable, "-m", "radixshmem.cli", "--name", NAME, "--node-name", f"node{i}",
                   "--data-name", f"{NAME}_data{i}", "--data-bytes", "64M", "--no-prefault",
                   "--endpoint", endpoints[i], "--registry", registry, "--cluster-id", cluster_id,
                   "--expected-min-nodes", "2", "--rpc-address", "0.0.0.0", "--transfer-ip", "127.0.0.1",
                   "--bootstrap-timeout", str(a.bootstrap_timeout),
                   "--background-evict-ratio", "0", "--swa-ratio", "0.1",
                   "--mamba-ratio", "0.1"]
            cmd += [f"--transfer-dev={d}" for d in a.transfer_dev]
            servers.append(subprocess.Popen(cmd, stdout=open(logs[i], "wb"), stderr=subprocess.STDOUT,
                                            env=dict(os.environ, PYTHONUNBUFFERED="1")))
        try:
            p2p.wait_log(logs, servers, "Waiting for a client's geometry", 60)
        except RuntimeError as e:
            if "transfer backend" in str(e) and "unavailable" in str(e):
                print("SKIP: Mooncake transfer backend unavailable")
                return 77
            raise
        geometry = radixshmem.Geometry(block_size=1, full_slot_bytes=4096, swa_slot_bytes=512,
                                       swa_window_blocks=WINDOW, mamba_slot_bytes=1024)
        cl0 = radixshmem.RadixClient(NAME, geometry, endpoint=endpoints[0])
        p2p.wait_log(logs, servers, "Cluster ready", a.bootstrap_timeout + 30)
        cl0.close()
        ranks = [int(p2p.read(log).split("rank=")[1].split("/")[0]) for log in logs]

        print("\n[2] writer on node0: path, windows and checkpoints at 8/16/24, branches at 8 and 4")
        w, = run_clients([("writer", endpoints[0])])
        print(f"  {w}")
        p2p.check(w["insert_errors"] == ["InsertError.OK"], "writer inserted")

        print("\n[3] reader on node1 pulls [0, 16); node0 moves what keeps its copy and drops it")
        r, = run_clients([("reader", endpoints[1])])
        print(f"  {r}")
        p2p.check(r["hit"] == 16 and r["remote"] == 16 + WINDOW + 1 and r["source"] == ranks[0],
                  f"the pull hit 16 blocks, their last window and checkpoint from rank {ranks[0]}")
        p2p.check(r["source_main"] == 4 and r["source_side"] == 4 and r["source_long"] == 24,
                  "node0 keeps [0, 4) and the long branch only")
        p2p.check(r["main"] == [24, True, True, True], "node1 serves the moved tail with its window and checkpoint")
        p2p.check(r["side"] == [12, True, True, True], "node1 serves the moved short branch")
        p2p.check(r["main8"] == [8, True, True, True], "node1 serves the moved cells of the dropped path")

        print("\n[4] both trees are leak-free")
        for i, h in enumerate(run_clients([("health", endpoints[0]), ("health", endpoints[1])])):
            p2p.check(h["health"] == 0, f"node{i} check_health == 0 ({h['report']})")
        print("\nPASS")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAIL: {exc}", file=sys.stderr)
        return 1
    finally:
        for proc in servers:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        for proc in servers:
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        if etcd_proc is not None:
            etcd_proc.terminate()
            etcd_proc.wait(timeout=10)
        for stale in glob.glob(f"/dev/shm{NAME}*"):
            try:
                os.unlink(stale)
            except OSError:
                pass
        if os.environ.get("RADIXSHMEM_KEEP_LOGS"):
            print(f"logs kept in {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def main() -> int:
    if len(sys.argv) > 3 and sys.argv[1] == "--client":
        return client_main(sys.argv[2], sys.argv[3])
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index-dev", default="mlx5_bond_0")
    p.add_argument("--transfer-dev", action="append", default=None)
    p.add_argument("--gid-idx", type=int, default=3)
    p.add_argument("--bootstrap-timeout", type=int, default=120)
    a = p.parse_args()
    # The device of the planes that run on RDMA; the servers inherit it.
    os.environ["RADIXSHMEM_INDEX_DEV"] = a.index_dev
    os.environ["RADIXSHMEM_INDEX_GID_IDX"] = str(a.gid_idx)
    if a.transfer_dev is None:
        a.transfer_dev = [a.index_dev] if a.index_dev else []
    return orchestrate(a)


if __name__ == "__main__":
    sys.exit(main())
