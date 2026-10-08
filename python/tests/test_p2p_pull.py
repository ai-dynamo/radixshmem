#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two radix-server processes on one host, each owning index + SlotStore +
transfer engine; a client configures node0, node1 takes the geometry from
etcd, then a client on node1 pulls node0's blocks with concurrent
pull_async() jobs.

    python3 test_p2p_pull.py [--index-dev mlx5_bond_0] [--transfer-dev mlx5_bond_0] [--gid-idx 3]

Needs etcd, an RDMA device and the Mooncake backend; exits 77 when missing.
RADIXSHMEM_TEST_LEASE_S=45 sleeps that long before the reader attaches, past the
etcd lease TTL, to prove the lease keepalive. Clients are re-exec'd, never
forked (the etcd Go runtime and ibverbs do not survive fork()).
"""

import argparse
import glob
import json
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time

NAME = "/py_p2p"
PREFIXES = 8
PREFIX_LEN = 32
SLOT_BYTES = 65536


def _pattern(k: int, i: int) -> bytes:
    return (struct.pack("<i", k * 1000 + i * 7 + 3) * (SLOT_BYTES // 4))[:SLOT_BYTES]


def client_main(role: str, endpoint: str) -> int:
    import numpy as np
    import radixshmem

    c = radixshmem.RadixClient(NAME, endpoint=endpoint)
    assert c.geometry["pools"]["full"]["slot_bytes"] == SLOT_BYTES
    out = {"role": role, "rank": c.rank(), "world_size": c.world_size(), "mode": c.info.mode}
    hs = [radixshmem.compute_hashes(np.arange(1000 * (k + 1), 1000 * (k + 1) + PREFIX_LEN, dtype=np.int64), 1)
          for k in range(PREFIXES)]
    if role == "writer":
        errors = set()
        for k, h in enumerate(hs):
            slots = c.allocate_slots(PREFIX_LEN)
            for i, s in enumerate(slots):
                c.slot_view(int(s))[:] = _pattern(k, i)
            errors.add(str(c.insert(h, slots).error))
        c.flush()
        out["insert_errors"] = sorted(errors)
    elif role == "reader":
        jobs = [c.pull_async(h) for h in hs]
        results = [j.wait(120) for j in jobs]
        out.update(hits=sum(r.common_hit for r in results),
                   remote_blocks=sum(r.remote_blocks for r in results),
                   source_ranks=sorted({r.source_rank for r in results}),
                   bytes_ok=sum(bytes(c.slot_view(int(s))) == _pattern(k, i)
                                for k, r in enumerate(results) for i, s in enumerate(r.full_slots)))
        for r in results:
            r.finalize()
        out["local_hit_after"] = sum(int(c.query(h, local_only=True, update_meta=False).common_hit)
                                     for h in hs)
    elif role == "health":
        out["health"] = int(c.check_health())
        out["report"] = c.health_report().describe().strip()
    print("RESULT " + json.dumps(out), flush=True)
    c.close()
    return 0


# ------------------------------------------------------------ orchestrator

def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_etcd(workdir):
    cport, pport = free_port(), free_port()
    proc = subprocess.Popen(
        ["etcd", "--name", "p2p", "--data-dir", os.path.join(workdir, "etcd-data"),
         "--listen-client-urls", f"http://127.0.0.1:{cport}",
         "--advertise-client-urls", f"http://127.0.0.1:{cport}",
         "--listen-peer-urls", f"http://127.0.0.1:{pport}",
         "--initial-advertise-peer-urls", f"http://127.0.0.1:{pport}",
         "--initial-cluster", f"p2p=http://127.0.0.1:{pport}"],
        stdout=open(os.path.join(workdir, "etcd.log"), "wb"), stderr=subprocess.STDOUT)
    deadline = time.time() + 30
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("etcd died during startup")
        try:
            with socket.create_connection(("127.0.0.1", cport), timeout=0.5):
                return proc, cport
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("etcd did not come up")


def read(path):
    with open(path, "r", errors="replace") as f:
        return f.read()


def wait_log(logs, procs, needle, timeout):
    deadline = time.time() + timeout
    pending = set(range(len(logs)))
    while pending and time.time() < deadline:
        for i in list(pending):
            if procs[i].poll() is not None:
                raise RuntimeError(f"server {i} exited (rc={procs[i].returncode}):\n{read(logs[i])[-3000:]}")
            if needle in read(logs[i]):
                pending.discard(i)
        time.sleep(0.2)
    if pending:
        raise RuntimeError(f"timeout waiting for {needle!r} on {sorted(pending)}:\n"
                           + read(logs[sorted(pending)[0]])[-3000:])


def run_clients(roles_endpoints, timeout=240):
    procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "--client", role, ep],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for role, ep in roles_endpoints]
    results = []
    for p, (role, _) in zip(procs, roles_endpoints):
        out, err = p.communicate(timeout=timeout)
        lines = [l for l in out.splitlines() if l.startswith("RESULT ")]
        if not lines:
            raise RuntimeError(f"client {role} rc={p.returncode}\nstdout:\n{out}\nstderr:\n{err[-3000:]}")
        results.append(json.loads(lines[0][len("RESULT "):]))
    return results


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


def orchestrate(a) -> int:
    import radixshmem

    if (not shutil.which("etcd") or not os.path.isdir(f"/sys/class/infiniband/{a.index_dev}")
            or not hasattr(radixshmem, "_data") or "mooncake" not in radixshmem._data.TRANSFER_BACKENDS
            or not hasattr(radixshmem._core, "EtcdKv")):
        print(f"SKIP: needs etcd, RDMA device {a.index_dev!r} and radixshmem built with etcd + Mooncake")
        return 77
    workdir = tempfile.mkdtemp(prefix="radixshmem-p2p-")
    cluster_id = f"p2p{os.getpid()}"
    etcd_proc, servers, logs = None, [], []
    try:
        etcd_proc, cport = start_etcd(workdir)
        registry = f"etcd://127.0.0.1:{cport}"
        endpoints = [f"unix://{workdir}/node{i}.sock" for i in range(2)]
        print(f"etcd on {registry}   logs in {workdir}")

        print("\n[1] two radix-server processes, index + data plane, no geometry on the command line")
        for i in range(2):
            logs.append(os.path.join(workdir, f"server{i}.log"))
            cmd = [sys.executable, "-m", "radixshmem.cli", "--name", NAME, "--node-name", f"node{i}",
                   "--data-name", f"{NAME}_data{i}", "--data-bytes", "64M", "--no-prefault",
                   "--endpoint", endpoints[i], "--registry", registry, "--cluster-id", cluster_id,
                   "--expected-min-nodes", "2", "--rpc-address", "0.0.0.0", "--transfer-ip", "127.0.0.1",
                   "--gid-idx", str(a.gid_idx), "--bootstrap-timeout", str(a.bootstrap_timeout),
                   "--background-evict-ratio", "0", "--interval", "0"]
            cmd += ["--index-dev", a.index_dev] if a.index_dev else []
            cmd += [f"--transfer-dev={d}" for d in a.transfer_dev]
            servers.append(subprocess.Popen(cmd, stdout=open(logs[i], "wb"), stderr=subprocess.STDOUT,
                                            env=dict(os.environ, PYTHONUNBUFFERED="1")))
        try:
            wait_log(logs, servers, "Waiting for a client's geometry", 60)
        except RuntimeError as e:
            if "transfer backend" in str(e) and "unavailable" in str(e):
                print("SKIP: Mooncake transfer backend unavailable")
                return 77
            raise
        # this process only talks gRPC: a client that never calls the index never attaches shm
        probes = [radixshmem.RadixClient(NAME, endpoint=ep) for ep in endpoints]
        for i, cl in enumerate(probes):
            check(cl.info.mode == "waiting" and cl.info.geometry is None, f"node{i} is waiting for a geometry")

        print("\n[2] a client brings node0 the geometry; node1 takes it from etcd; the cluster forms")
        cl0 = radixshmem.RadixClient(NAME, radixshmem.Geometry(block_size=1, full_slot_bytes=SLOT_BYTES),
                                   endpoint=endpoints[0])
        info = cl0.info
        check(info.mode == "configuring" and info.geometry["pools"]["full"]["num_slots"] == (64 << 20) // SLOT_BYTES,
              f"the constructor returned at once with {info.geometry['pools']['full']['num_slots']} planned FULL slots")
        wait_log(logs, servers, "Cluster ready", a.bootstrap_timeout + 30)
        for i, cl in enumerate(probes):
            info = cl.wait_ready(10)
            check(info.mode == "ready" and info.geometry["pools"]["full"]["slot_bytes"] == SLOT_BYTES,
                  f"node{i} is ready with the client's slot shape")
            cl.close()
        cl0.close()
        ranks = [int(read(l).split("rank=")[1].split("/")[0]) for l in logs]
        check(sorted(ranks) == [0, 1], f"etcd handed out ranks {ranks}")
        try:
            radixshmem.RadixClient(NAME, radixshmem.Geometry(block_size=2, full_slot_bytes=SLOT_BYTES),
                                 endpoint=endpoints[1])
        except radixshmem.GeometryMismatch as e:
            check("block_size" in str(e), f"node1 refuses another geometry ({e})")
        else:
            raise AssertionError("node1 accepted a conflicting geometry")

        print(f"\n[3] writer on node0 fills slots and publishes {PREFIXES} prefixes")
        w, = run_clients([("writer", endpoints[0])])
        print(f"  {w}")
        check(w["insert_errors"] == ["InsertError.OK"] and w["mode"] == "ready", "writer inserted")

        lease_s = float(os.environ.get("RADIXSHMEM_TEST_LEASE_S", "0"))
        if lease_s > 0:
            print(f"\n[3b] sleeping {lease_s:.0f}s past the etcd lease TTL")
            time.sleep(lease_s)

        total = PREFIXES * PREFIX_LEN
        print(f"\n[4] reader on node1 pulls them with {PREFIXES} concurrent pull_async jobs")
        r, = run_clients([("reader", endpoints[1])])
        print(f"  {r}")
        check(r["hits"] == total, f"jobs hit all {total} blocks")
        check(r["remote_blocks"] == total and r["source_ranks"] == [ranks[0]],
              f"all blocks came from rank {ranks[0]}")
        check(r["bytes_ok"] == total, "every slot holds the writer's bytes")
        check(r["local_hit_after"] == total, "node1 now serves the prefixes locally")

        print("\n[5] both trees are leak-free")
        for i, h in enumerate(run_clients([("health", endpoints[0]), ("health", endpoints[1])])):
            check(h["health"] == 0, f"node{i} check_health == 0 ({h['report']})")
        print("\nPASS")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAIL: {exc}", file=sys.stderr)
        return 1
    finally:
        for p in servers:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        for p in servers:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
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
    if len(sys.argv) > 1 and sys.argv[1] == "--client":
        return client_main(sys.argv[2], sys.argv[3])
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index-dev", default="mlx5_bond_0")
    p.add_argument("--transfer-dev", action="append", default=None)
    p.add_argument("--gid-idx", type=int, default=3)
    p.add_argument("--bootstrap-timeout", type=int, default=120)
    a = p.parse_args()
    if a.transfer_dev is None:
        a.transfer_dev = [a.index_dev] if a.index_dev else []
    return orchestrate(a)


if __name__ == "__main__":
    sys.exit(main())
