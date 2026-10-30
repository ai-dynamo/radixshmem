#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A write-back handler on node1 pull_async()s node0's prefixes with its
in-process RadixClient, inside node1's server process.

    python3 test_writeback_pull.py [--index-dev mlx5_bond_0] [--transfer-dev mlx5_bond_0] [--gid-idx 3]

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
import threading
import time

import test_p2p_pull as p2p
from radixshmem.writeback import WritebackHandler


class PullHandler(WritebackHandler):
    def __init__(self, config):
        self.cfg = config
        self._stop = threading.Event()
        self._thread = None

    def start(self, client):
        self._thread = threading.Thread(target=self._run, args=(client,), daemon=True)
        self._thread.start()

    def _run(self, client):
        import numpy as np
        import radixshmem

        while not os.path.exists(self.cfg["go"]):
            if self._stop.wait(0.1):
                return
        hs = [radixshmem.compute_hashes(np.arange(1000 * (k + 1), 1000 * (k + 1) + p2p.PREFIX_LEN,
                                                  dtype=np.int64), 1) for k in range(p2p.PREFIXES)]
        try:
            results = [j.wait(120) for j in [client.pull_async(h) for h in hs]]
            out = {"hits": sum(r.common_hit for r in results),
                   "remote_blocks": sum(r.remote_blocks for r in results),
                   "bytes_ok": sum(bytes(client.slot_view(int(s))) == p2p._pattern(k, i)
                                   for k, r in enumerate(results) for i, s in enumerate(r.full_slots))}
            for r in results:
                r.finalize()
        except Exception as e:  # noqa: BLE001 - reported to the orchestrator
            out = {"error": f"{type(e).__name__}: {e}"}
        with open(self.cfg["out"] + ".tmp", "w") as f:
            json.dump(out, f)
        os.rename(self.cfg["out"] + ".tmp", self.cfg["out"])

    def write(self, tasks, done):
        done([t.task_id for t in tasks])

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(130)


def make(config):
    return PullHandler(config)


def orchestrate(a) -> int:
    import radixshmem

    if (not shutil.which("etcd") or not os.path.isdir(f"/sys/class/infiniband/{a.index_dev}")
            or not hasattr(radixshmem, "_data") or "mooncake" not in radixshmem._data.TRANSFER_BACKENDS
            or not hasattr(radixshmem._core, "EtcdKv")):
        print(f"SKIP: needs etcd, RDMA device {a.index_dev!r} and radixshmem built with etcd + Mooncake")
        return 77
    workdir = tempfile.mkdtemp(prefix="radixshmem-wbpull-")
    go, result = os.path.join(workdir, "go"), os.path.join(workdir, "pulled.json")
    cluster_id = f"wbpull{os.getpid()}"
    etcd_proc, servers, logs = None, [], []
    try:
        etcd_proc, cport = p2p.start_etcd(workdir)
        registry = f"etcd://127.0.0.1:{cport}"
        endpoints = [f"unix://{workdir}/node{i}.sock" for i in range(2)]
        env = dict(os.environ, PYTHONUNBUFFERED="1",
                   PYTHONPATH=os.pathsep.join([os.path.dirname(os.path.abspath(__file__)),
                                               os.environ.get("PYTHONPATH", "")]))
        print(f"etcd on {registry}   logs in {workdir}")

        print("\n[1] two radix-server processes; node1 loads the write-back handler")
        for i in range(2):
            logs.append(os.path.join(workdir, f"server{i}.log"))
            cmd = [sys.executable, "-m", "radixshmem.cli", "--name", p2p.NAME, "--node-name", f"node{i}",
                   "--data-name", f"{p2p.NAME}_data{i}", "--data-bytes", "64M", "--no-prefault",
                   "--endpoint", endpoints[i], "--registry", registry, "--cluster-id", cluster_id,
                   "--expected-min-nodes", "2", "--rpc-address", "0.0.0.0", "--transfer-ip", "127.0.0.1",
                   "--gid-idx", str(a.gid_idx), "--bootstrap-timeout", str(a.bootstrap_timeout),
                   "--interval", "0"]
            cmd += ["--index-dev", a.index_dev] if a.index_dev else []
            cmd += [f"--transfer-dev={d}" for d in a.transfer_dev]
            if i == 1:
                cmd += ["--background-evict-ratio", "0.01",
                        "--writeback-handler", "test_writeback_pull:make",
                        "--writeback-config", json.dumps({"go": go, "out": result})]
            else:
                cmd += ["--background-evict-ratio", "0"]
            servers.append(subprocess.Popen(cmd, stdout=open(logs[i], "wb"), stderr=subprocess.STDOUT, env=env))
        try:
            p2p.wait_log(logs, servers, "Waiting for a client's geometry", 60)
        except RuntimeError as e:
            if "transfer backend" in str(e) and "unavailable" in str(e):
                print("SKIP: Mooncake transfer backend unavailable")
                return 77
            raise
        p2p.check("Write-back:     test_writeback_pull:make" in p2p.read(logs[1]), "node1 loaded the handler")

        print("\n[2] a client brings node0 the geometry; the cluster forms")
        cl0 = radixshmem.RadixClient(p2p.NAME, radixshmem.Geometry(block_size=1, full_slot_bytes=p2p.SLOT_BYTES),
                                     endpoint=endpoints[0])
        p2p.wait_log(logs, servers, "Cluster ready", a.bootstrap_timeout + 30)
        cl0.close()
        ranks = [int(p2p.read(l).split("rank=")[1].split("/")[0]) for l in logs]

        print(f"\n[3] writer on node0 publishes {p2p.PREFIXES} prefixes")
        w, = p2p.run_clients([("writer", endpoints[0])])
        p2p.check(w["insert_errors"] == ["InsertError.OK"], "writer inserted")

        total = p2p.PREFIXES * p2p.PREFIX_LEN
        print("\n[4] node1's handler pulls them inside its server process")
        open(go, "w").close()
        deadline = time.time() + 150
        while not os.path.exists(result):
            if time.time() > deadline or servers[1].poll() is not None:
                raise RuntimeError("no result from the handler:\n" + p2p.read(logs[1])[-3000:])
            time.sleep(0.2)
        r = json.load(open(result))
        print(f"  {r}")
        p2p.check("error" not in r and r["hits"] == total, f"the in-process jobs hit all {total} blocks")
        p2p.check(r["remote_blocks"] == total, f"all blocks came from rank {ranks[0]} over the transfer engine")
        p2p.check(r["bytes_ok"] == total, "every pulled slot holds the writer's bytes")

        print("\n[5] both trees are leak-free")
        for i, h in enumerate(p2p.run_clients([("health", endpoints[0]), ("health", endpoints[1])])):
            p2p.check(h["health"] == 0, f"node{i} check_health == 0 ({h['report']})")
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
        for stale in glob.glob(f"/dev/shm{p2p.NAME}*"):
            try:
                os.unlink(stale)
            except OSError:
                pass
        if os.environ.get("RADIXSHMEM_KEEP_LOGS"):
            print(f"logs kept in {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def main() -> int:
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
