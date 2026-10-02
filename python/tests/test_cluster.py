#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-node smoke test driven entirely from Python.

Simulates a cluster on one host: a private etcd, N index-only `radix-server`
processes (`python -m shmradix.cli`, so the CLI itself is under test) that a
client configures one by one (index-only servers take their slot counts from
the client), and re-exec'd client processes that attach by shm name and
cross-check what they see.

    python3 test_cluster.py [--nodes 3] [--index-dev mlx5_bond_0] [--gid-idx 3]

Clients are re-exec'd, never fork()'d: the etcd Go wrapper starts its runtime
at library load and a bare-forked child deadlocks on the first cgo call.

Exit code 0 = pass.
"""

import argparse
import glob
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

SHM_PREFIX = "/pytest_cluster"
PREFIX_LEN = 64          # hashes the writer inserts
BLOCK_SIZE = 1
MAX_BLOCKS = 8192
TURN_LEN = 20            # blocks per turn of the alternating session
TURNS = 6
FILL_PREFIXES = 160      # x PREFIX_LEN > MAX_BLOCKS: forces eviction
LEASE_TTL = 10           # etcd lease of every server; [8] waits it out


# ---------------------------------------------------------------- client side

def _hashes(count: int, base: int = 1000):
    import numpy as np
    import shmradix
    tokens = np.arange(base, base + count, dtype=np.int64)
    return shmradix.compute_hashes(tokens, tokens_per_block=BLOCK_SIZE)


def _publish(cli, base):
    h = _hashes(PREFIX_LEN, base=base)
    res = cli.insert(h, cli.allocate_slots(len(h)))
    cli.flush()
    return {"insert_error": str(res.error)}


def _find(cli, base):
    import shmradix
    t0 = time.time()
    qr = cli.query(_hashes(PREFIX_LEN, base=base), mask=shmradix.COMPONENT_MASK_FULL)
    frags = qr.full_fragments
    return {"total_hit_length": int(qr.common_hit),
            "remote_node_id": int(frags[-1][0]) if frags else -1,
            "secs": time.time() - t0}


def client_main(role: str, shm_name: str) -> int:
    """Runs in a re-exec'd process; prints one JSON line on stdout."""
    import shmradix

    if role == "local_read_only":
        # Read-only attach without the cluster plane (no etcd, no RDMA): the
        # node's own tree and rank are visible, every mutation is refused.
        import numpy as np
        ro = shmradix._core.RadixClient(shm_name, local_read_only=True)
        h = _hashes(PREFIX_LEN)
        q = ro.query(h, mask=shmradix.COMPONENT_MASK_FULL, update_meta=False)
        refused = 0
        for call in (lambda: ro.allocate_slots(1),
                     lambda: ro.insert(h[:1], np.zeros(1, dtype=np.int32)), ro.reset):
            try:
                call()
            except RuntimeError:
                refused += 1
        print("RESULT " + json.dumps({
            "role": role, "shm": shm_name, "rank": ro.rank(), "world_size": ro.world_size(),
            "distributed": ro.is_distributed(), "hit": int(q.common_hit),
            "rdma_read_count": int(q.rdma_read_count), "refused": refused}), flush=True)
        return 0

    cli = shmradix._core.RadixClient(shm_name)
    out = {
        "role": role,
        "shm": shm_name,
        "rank": cli.rank(),
        "world_size": cli.world_size(),
        "distributed": cli.is_distributed(),
    }

    if role == "writer":
        h = _hashes(PREFIX_LEN)
        matched = int(cli.query(h, mask=shmradix.COMPONENT_MASK_FULL,
                                update_meta=False).common_hit)
        slots = cli.allocate_slots(len(h) - matched)
        res = cli.insert(h, slots, start=matched)
        cli.flush()
        out["insert_error"] = str(res.error)
        out["inserted"] = int(len(h) - matched)

    elif role == "reader":
        h = _hashes(PREFIX_LEN)
        qr = cli.query(h, mask=shmradix.COMPONENT_MASK_FULL)
        own = int(cli.rank())
        frags = qr.full_fragments
        local_hit = sum(len(f[2]) for f in frags if f[0] == own)
        remote_src = frags[-1][0] if frags else -1
        out.update(
            total_hit_length=int(qr.common_hit),
            local_hit_length=int(local_hit),
            remote_node_id=int(remote_src),
            remote_hit_length=int(qr.common_hit) - int(local_hit),
            rdma_read_count=int(qr.rdma_read_count),
        )

    elif role.startswith("publish"):     # publish[<base>]
        out.update(_publish(cli, int(role[7:] or 9000)))

    elif role.startswith("find"):        # find[<base>]
        out.update(_find(cli, int(role[4:] or 9000)))

    elif role == "session":
        # Long-lived client: one "publish|find <base>" per stdin line.
        print("RESULT " + json.dumps(out), flush=True)
        for line in sys.stdin:
            op, base = line.split()
            r = _publish(cli, int(base)) if op == "publish" else _find(cli, int(base))
            print("RESULT " + json.dumps(r), flush=True)
        return 0

    elif role == "fill":
        # Eviction then retracts RHT entries, some held by a dead node.
        worst = 0.0
        for k in range(FILL_PREFIXES):
            h = _hashes(PREFIX_LEN, base=20000 + k * 1000)
            t0 = time.time()
            cli.insert(h, cli.allocate_slots(len(h)))
            cli.flush()
            worst = max(worst, time.time() - t0)
        out["worst_secs"] = worst

    elif role == "faults":
        out.update(cli.peer_fault_counts())

    elif role == "health":
        # End-of-run leak audit of this node's tree (no locks taken).
        out["health"] = int(cli.check_health())
        out["health_report"] = cli.health_report().describe().strip()

    elif role == "rank_twice":
        # A second client on the same node must resolve the same etcd rank.
        cli2 = shmradix._core.RadixClient(shm_name)
        out["rank2"] = cli2.rank()

    elif role.startswith("turn"):
        # One turn of a session that alternates nodes; a Mamba checkpoint at
        # every turn end keeps the nodes unmerged, so the peers' node
        # boundaries never coincide.
        k = int(role[4:])
        n = (k + 1) * TURN_LEN
        h = _hashes(n, base=5000)
        fm = shmradix.COMPONENT_MASK_FULL | shmradix.COMPONENT_MASK_MAMBA
        q = cli.query(h[:k * TURN_LEN], mask=fm, update_meta=False) if k else None
        have = int(cli.query(h, mask=shmradix.COMPONENT_MASK_FULL, local_only=True,
                             update_meta=False).common_hit)
        res = cli.insert(h, cli.allocate_slots(n - have), start=have)
        mres = cli.insert(h, cli.allocate_slots(1, shmradix.ComponentType.MAMBA),
                          component=shmradix.ComponentType.MAMBA)
        cli.flush()
        out.update(hit=int(q.common_hit) if q else 0, expected=k * TURN_LEN,
                   rdma_read_count=int(q.rdma_read_count) if q else 0,
                   insert_error=str(res.error), mamba_error=str(mres.error))

    else:
        print(f"unknown role {role}", file=sys.stderr)
        return 2

    print("RESULT " + json.dumps(out), flush=True)
    return 0


# ------------------------------------------------------------ orchestrator

def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_etcd(workdir: str):
    """Private single-node etcd on free ports (never reuse a shared one:
    stale keys from an earlier run silently poison membership)."""
    cport, pport = free_port(), free_port()
    data_dir = os.path.join(workdir, "etcd-data")
    log = open(os.path.join(workdir, "etcd.log"), "wb")
    proc = subprocess.Popen(
        ["etcd",
         "--name", "pytest",
         "--data-dir", data_dir,
         "--listen-client-urls", f"http://127.0.0.1:{cport}",
         "--advertise-client-urls", f"http://127.0.0.1:{cport}",
         "--listen-peer-urls", f"http://127.0.0.1:{pport}",
         "--initial-advertise-peer-urls", f"http://127.0.0.1:{pport}",
         "--initial-cluster", f"pytest=http://127.0.0.1:{pport}"],
        stdout=log, stderr=subprocess.STDOUT)

    deadline = time.time() + 30
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("etcd died during startup; see etcd.log")
        try:
            with socket.create_connection(("127.0.0.1", cport), timeout=0.5):
                return proc, cport
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("etcd did not come up")


def server_cmd(i, n, registry, cluster_id, workdir, index_dev, gid_idx, transports):
    cmd = [
        sys.executable, "-m", "shmradix.cli",
        "--name", SHM_PREFIX,
        "--node-name", f"node{i}",
        "--registry", registry,
        "--cluster-id", cluster_id,
        "--expected-min-nodes", str(n),
        "--rpc-address", "0.0.0.0",
        "--rht-transport", transports[0],
        "--peer-index-transport", transports[1],
        # several servers on one host: the default socket is per name, not per node
        "--endpoint", endpoint(workdir, i),
        "--gid-idx", str(gid_idx),
        "--lease-ttl", str(LEASE_TTL),
        "--interval", "0",
    ]
    if index_dev:
        cmd += ["--index-dev", index_dev]
    return cmd


def launch(cmd, path):
    # stdout is a file here, so python block-buffers its prints and the
    # readiness banner would only land at exit.
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    return subprocess.Popen(cmd, stdout=open(path, "ab"), stderr=subprocess.STDOUT, env=env)


def start_servers(n, registry, cluster_id, workdir, index_dev, gid_idx, transports=("xrc", "xrc")):
    procs, logs = [], []
    for i in range(n):
        path = os.path.join(workdir, f"server{i}.log")
        procs.append(launch(server_cmd(i, n, registry, cluster_id, workdir, index_dev,
                                       gid_idx, transports), path))
        logs.append(path)
    return procs, logs


def endpoint(workdir, i):
    return f"unix://{workdir}/node{i}.sock"


def wait_banner(procs, logs, needle, timeout, marks=None):
    """Block until every server printed `needle` (past marks[i]); raise otherwise."""
    deadline = time.time() + timeout
    pending = set(range(len(procs)))
    while pending and time.time() < deadline:
        for i in list(pending):
            if procs[i].poll() is not None:
                raise RuntimeError(f"server {i} exited early "
                                   f"(rc={procs[i].returncode}); log:\n"
                                   + tail(logs[i]))
            if needle in read(logs[i])[marks[i] if marks else 0:]:
                pending.discard(i)
        if pending:
            time.sleep(0.2)
    if pending:
        raise RuntimeError(f"timeout waiting for {needle!r} on servers " + str(sorted(pending))
                           + "; log:\n" + tail(logs[sorted(pending)[0]]))


def configure_all(n, workdir):
    """Index-only servers take no geometry from etcd: every node's client hands
    over the slot counts. The clients here only talk gRPC (no index call, so
    no shm attach in this process); readiness is the 'Cluster ready' banner."""
    for i in range(n):
        configure_one(workdir, i)


def configure_one(workdir, i):
    import shmradix

    geo = shmradix.Geometry(block_size=BLOCK_SIZE, full_slots=MAX_BLOCKS, mamba_slots=64)
    cl = shmradix.RadixClient(SHM_PREFIX, endpoint=endpoint(workdir, i))
    info = cl.info
    check(info.mode == "waiting" and not info.data_plane and info.geometry is None,
          f"node{i} starts waiting for a geometry, index only")
    info = cl.configure(geo)
    check(info.mode == "configuring", f"node{i} accepted the geometry and returned at once")
    cl.close()


def read(path):
    with open(path, "r", errors="replace") as f:
        return f.read()


def tail(path, n=25):
    return "\n".join(read(path).splitlines()[-n:])


class Session:
    """A client process that lives across server restarts."""

    def __init__(self, shm_name):
        self.p = subprocess.Popen([sys.executable, os.path.abspath(__file__),
                                   "--client", "session", shm_name],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.attach = self._result()

    def _result(self):
        for line in self.p.stdout:
            if line.startswith("RESULT "):
                return json.loads(line[len("RESULT "):])
        raise RuntimeError(f"session client died (rc={self.p.poll()})")

    def do(self, op, base):
        self.p.stdin.write(f"{op} {base}\n")
        self.p.stdin.flush()
        return self._result()

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=30)


def run_client(role, shm_name, timeout=90):
    p = subprocess.run([sys.executable, os.path.abspath(__file__),
                        "--client", role, shm_name],
                       capture_output=True, text=True, timeout=timeout)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    raise RuntimeError(f"client {role} produced no result (rc={p.returncode})\n"
                       f"stdout:\n{p.stdout}\nstderr:\n{p.stderr[-2000:]}")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


def orchestrate(args) -> int:
    workdir = tempfile.mkdtemp(prefix="shmradix-cluster-")
    cluster_id = f"pytest{os.getpid()}"
    etcd_proc = servers = None
    logs = []
    try:
        etcd_proc, cport = start_etcd(workdir)
        registry = f"etcd://127.0.0.1:{cport}"
        print(f"etcd on {registry}   logs in {workdir}")

        print(f"\n[1] starting {args.nodes} radix-server processes, then a client configures each")
        servers, logs = start_servers(args.nodes, registry, cluster_id,
                                      workdir, args.index_dev, args.gid_idx,
                                      (args.rht_transport, args.peer_index_transport))
        wait_banner(servers, logs, "Waiting for a client's geometry", args.timeout)
        configure_all(args.nodes, workdir)
        wait_banner(servers, logs, "Cluster ready", args.timeout)

        import shmradix
        ranks = []
        for i, path in enumerate(logs):
            line = [l for l in read(path).splitlines() if "Cluster ready" in l][0]
            ranks.append(int(line.split("rank=")[1].split("/")[0]))
            print(f"  node{i}: {line.strip()}")
            cl = shmradix.RadixClient(SHM_PREFIX, endpoint=endpoint(workdir, i))
            info = cl.wait_ready(10)
            cl.close()
            check(info.mode == "ready" and info.rank == ranks[i] and info.world_size == args.nodes
                  and info.geometry["pools"] == {"full": {"slot_bytes": 0, "num_slots": MAX_BLOCKS},
                                                 "mamba": {"slot_bytes": 0, "num_slots": 64}},
                  f"node{i} publishes the client's counts and its rank")
        check(sorted(ranks) == list(range(args.nodes)),
              f"etcd handed out dense ranks {sorted(ranks)}")
        try:
            shmradix.RadixClient(SHM_PREFIX, shmradix.Geometry(block_size=BLOCK_SIZE, full_slots=MAX_BLOCKS),
                                 endpoint=endpoint(workdir, 1))
        except shmradix.GeometryMismatch as e:
            check("mamba" in str(e), f"node1 refuses a geometry without the Mamba pool ({e})")
        else:
            raise AssertionError("node1 accepted a conflicting geometry")

        print("\n[2] writer client on node0 inserts a prefix")
        w = run_client("writer", f"{SHM_PREFIX}_node0")
        print(f"  {w}")
        check(w["insert_error"] == "InsertError.OK", "insert returned OK")
        check(w["distributed"] and w["world_size"] == args.nodes,
              f"writer sees a {w['world_size']}-node cluster")
        check(w["rank"] == ranks[0], f"writer rank {w['rank']} == server rank")

        print("\n[2b] local_read_only clients see only their own node's tree and cannot write")
        for i in range(args.nodes):
            r = run_client("local_read_only", f"{SHM_PREFIX}_node{i}")
            print(f"  {r}")
            check(not r["distributed"] and r["world_size"] == 1 and r["rdma_read_count"] == 0,
                  f"node{i} local_read_only client never joined the cluster")
            check(r["rank"] == ranks[i], f"node{i} local_read_only client reports rank {ranks[i]}")
            check(r["hit"] == (PREFIX_LEN if i == 0 else 0),
                  f"node{i} local_read_only client hit {r['hit']} blocks locally")
            check(r["refused"] == 3, f"node{i} local_read_only client refused every mutation")

        print("\n[3] reader clients on the other nodes see it over RDMA")
        for i in range(1, args.nodes):
            r = run_client("reader", f"{SHM_PREFIX}_node{i}")
            print(f"  {r}")
            check(r["total_hit_length"] == PREFIX_LEN,
                  f"node{i} matched all {PREFIX_LEN} blocks")
            check(r["local_hit_length"] == 0, f"node{i} had nothing locally")
            check(r["remote_node_id"] == ranks[0],
                  f"node{i} resolved the holder to rank {ranks[0]}")
            check(r["rdma_read_count"] > 0, f"node{i} walked the peer over RDMA")

        print(f"\n[3b] a session alternating node0/node1 for {TURNS} turns keeps its whole history")
        for k in range(TURNS):
            r = run_client(f"turn{k}", f"{SHM_PREFIX}_node{k % 2}")
            check(r["insert_error"] == "InsertError.OK" and r["mamba_error"] == "InsertError.OK",
                  f"turn {k} on node{k % 2} inserted")
            check(r["hit"] == r["expected"],
                  f"turn {k} on node{k % 2} hit {r['hit']} of {r['expected']} "
                  f"({r['rdma_read_count']} RDMA reads)")

        print("\n[4] two clients on one node resolve the same rank")
        t = run_client("rank_twice", f"{SHM_PREFIX}_node1")
        check(t["rank"] == t["rank2"] == ranks[1],
              f"both clients on node1 report rank {t['rank']}")

        print("\n[5] every node's tree is leak-free (locks, slots, pins)")
        for i in range(args.nodes):
            h = run_client("health", f"{SHM_PREFIX}_node{i}")
            check(h["health"] == 0,
                  f"node{i} check_health == 0 ({h['health_report']})")

        v = args.nodes - 1
        session = Session(f"{SHM_PREFIX}_node0")
        print(f"\n[6] node{v} is SIGKILLed (its etcd lease stays) and restarts at once")
        servers[v].kill()
        servers[v].wait(timeout=15)
        for stale in glob.glob(f"/dev/shm{SHM_PREFIX}_node{v}*"):
            os.unlink(stale)
        mark = len(read(logs[v]))
        servers[v] = launch(server_cmd(v, args.nodes, registry, cluster_id, workdir, args.index_dev,
                                       args.gid_idx, (args.rht_transport, args.peer_index_transport)),
                            logs[v])
        wait_banner([servers[v]], [logs[v]], "Waiting for a client's geometry", args.timeout, [mark])
        configure_one(workdir, v)
        wait_banner([servers[v]], [logs[v]], "Cluster ready", args.timeout, [mark])
        text = read(logs[v])[mark:]
        check("removed the keys of a previous run" in text and "role=rejoin" in text,
              f"node{v} replaced its previous run's etcd keys and rejoined")
        check(f"rank={ranks[v]}/{args.nodes}" in text, f"node{v} got its rank {ranks[v]} back")
        r = run_client("publish", f"{SHM_PREFIX}_node{v}")
        check(r["insert_error"] == "InsertError.OK" and r["rank"] == ranks[v],
              f"a client on the restarted node{v} publishes")
        r = run_client("find", f"{SHM_PREFIX}_node0")
        check(r["total_hit_length"] == PREFIX_LEN and r["remote_node_id"] == ranks[v],
              f"a new node0 client finds it on node{v} over RDMA")
        time.sleep(1.5)   # the session's watcher takes node{v}'s new entry
        r = session.do("find", 9000)
        check(r["total_hit_length"] == PREFIX_LEN and r["remote_node_id"] == ranks[v],
              "a node0 client from before the restart finds it too")
        r = session.do("publish", 9500)
        check(r["insert_error"] == "InsertError.OK", "that client publishes")
        r = run_client("find9500", f"{SHM_PREFIX}_node{v}")
        check(r["total_hit_length"] == PREFIX_LEN and r["remote_node_id"] == ranks[0],
              f"node{v} finds what it published through its RHT")

        # Its etcd entry outlives it by the lease TTL: clients still route to it.
        print(f"\n[7] node{v} is SIGKILLed and stays down; node0 carries on")
        servers[v].kill()
        servers[v].wait(timeout=15)
        r = run_client("find", f"{SHM_PREFIX}_node0")
        check(r["total_hit_length"] == 0 and r["secs"] < 3,
              f"a query routed to node{v} misses in {r['secs']:.2f} s instead of failing")
        r = run_client("fill", f"{SHM_PREFIX}_node0")
        check(r["worst_secs"] < 3,
              f"{FILL_PREFIXES} inserts with eviction go on (slowest {r['worst_secs']:.2f} s)")
        check(servers[0].poll() is None, "node0's server survives evicting entries held by node"
              f"{v}")
        r = run_client("faults", f"{SHM_PREFIX}_node0")
        check(r["peer_down_events"] >= 1 and r["rht_removes_dropped"] + r["rht_writes_dropped"] > 0,
              f"node0 counts the lost ops ({r})")

        print(f"\n[8] node{v}'s lease expires; nothing waits for it")
        time.sleep(LEASE_TTL + 3)
        t0 = time.time()
        r = run_client("publish9700", f"{SHM_PREFIX}_node0")
        check(r["insert_error"] == "InsertError.OK" and time.time() - t0 < 10,
              f"a new node0 client attaches and publishes in {time.time() - t0:.1f} s")
        r = session.do("find", 9700)
        check(r["total_hit_length"] == PREFIX_LEN and r["secs"] < 1,
              "the long-lived node0 client still works")
        session.close()

        print("\nPASS")
        return 0
    except Exception as exc:                       # noqa: BLE001 — test driver
        print(f"\nFAIL: {exc}", file=sys.stderr)
        return 1
    finally:
        for p in servers or []:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        for p in servers or []:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
        if etcd_proc and etcd_proc.poll() is None:
            etcd_proc.terminate()
            try:
                etcd_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                etcd_proc.kill()
        # A killed server leaves its shm (tree + RHT shard) behind and the next
        # run would attach to the stale region instead of a fresh one.
        for stale in glob.glob(f"/dev/shm{SHM_PREFIX}_node*"):
            try:
                os.unlink(stale)
            except OSError:
                pass
        if os.environ.get("SHMRADIX_KEEP_LOGS"):
            print(f"logs kept in {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--client":
        return client_main(sys.argv[2], sys.argv[3])

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nodes", type=int, default=3)
    p.add_argument("--index-dev", default="mlx5_bond_0")
    p.add_argument("--rht-transport", default="xrc", choices=["xrc", "dc"])
    p.add_argument("--peer-index-transport", default="xrc", choices=["xrc", "dc"])
    p.add_argument("--gid-idx", type=int, default=3)
    p.add_argument("--timeout", type=int, default=120,
                   help="seconds to wait for the cluster to form")
    return orchestrate(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
