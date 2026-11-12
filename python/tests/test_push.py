#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""push_async and the PeerWriteback write-back handler across four
radix-server processes on one host.

node0 is a prefill node with background eviction and PeerWriteback; node1 and
node2 are empty decode nodes, node3 an empty prefill node. Prefixes come in pairs sharing a 3-block stem,
each with a FULL path and an SWA window and a Mamba checkpoint at block MID and at its END.

  push       a client on node0 push_async()s prefixes to node1: a new path, its
             sibling (the stem is deduplicated), the first one again, one whose
             valid() says no and one to its own rank. node1 then holds exactly
             the two that landed, bytes intact, and leaks nothing. A push to
             node2 with its Mamba pool taken writes nothing and leaves nothing.
  writeback  a writer puts twice node0's FULL pool through it; every prefix
             background eviction takes from node0 is on node1 or node2 with
             FULL + SWA + Mamba at both ends and intact bytes, on the same node as its
             sibling, both nodes got some and node3 none, node0's cluster
             query finds each one whole, and node0 pulls some back.

    python3 test_push.py [--index-dev mlx5_bond_0] [--transfer-dev mlx5_bond_0] [--gid-idx 3]

Needs etcd, an RDMA device and the Mooncake backend; exits 77 when missing.
Clients are re-exec'd, never forked (etcd's Go runtime and ibverbs do not
survive fork()).
"""

import argparse
import glob
import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time

import test_p2p_pull as p2p

NAME = "/py_push"
STEM, TAIL = 3, 5
N = STEM + TAIL
WINDOW = 4
FULL_BYTES, SWA_BYTES, MAMBA_BYTES = 65536, 16384, 32768
SALT = {"full": 0x46554C4C, "swa": 0x535741, "mamba": 0x4D414D42}
EVICT_RATIO = 0.2
PUSHED = 4          # prefixes [0, PUSHED) are for push_async; write-back gets the ones after
LONG = 600
MID = STEM + 3


def prefix_tokens(k: int):
    import numpy as np
    stem = np.arange(10_000 * (k // 2 + 1), 10_000 * (k // 2 + 1) + STEM, dtype=np.int64)
    tail = np.arange(1_000_000 + 100 * k, 1_000_000 + 100 * k + TAIL, dtype=np.int64)
    return np.concatenate([stem, tail])


def pattern(h: int, salt: int, size: int) -> bytes:
    return (struct.pack("<QQ", int(h), salt) * (size // 16))[:size]


def client_main(role: str, endpoint: str) -> int:
    import numpy as np
    import radixshmem
    from radixshmem import rpc
    from radixshmem._core import COMPONENT_MASK_FULL, COMPONENT_MASK_MAMBA, COMPONENT_MASK_SWA, ComponentType

    FULL, SWA, MAMBA = ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA
    ALL = COMPONENT_MASK_FULL | COMPONENT_MASK_SWA | COMPONENT_MASK_MAMBA
    c = radixshmem.RadixClient(NAME, endpoint=endpoint)
    out = {"role": role, "rank": c.rank()}

    def hs(k):
        return radixshmem.compute_hashes(prefix_tokens(k), 1)

    def span(arg):
        first, last = (int(x) for x in arg.split(":"))
        return range(first, last)

    def view(slot, kind):
        return c.slot_view(int(slot), kind)

    def joint(h):
        """Lock-held FULL | SWA | MAMBA hit of h on this node."""
        return c.query(h, mask=ALL, local_only=True, lock=True, update_meta=False)

    def full_of(q):
        return np.concatenate([np.asarray(s) for _, _, s in q.full_fragments]) if q.full_fragments else []

    def verify(h, q):
        """Bad slot count of a FULL | SWA | MAMBA hit on this node."""
        bad = sum(bytes(view(s, FULL)) != pattern(h[i], SALT["full"], FULL_BYTES) for i, s in enumerate(full_of(q)))
        start = int(q.swa_start)
        bad += sum(bytes(view(s, SWA)) != pattern(h[start + j], SALT["swa"], SWA_BYTES)
                   for j, s in enumerate(q.swa_slots))
        if q.mamba is not None:
            bad += bytes(view(q.mamba[1], MAMBA)) != pattern(h[-1], SALT["mamba"], MAMBA_BYTES)
        return bad

    def put(h, ends=None):
        errors = set()
        for e in ends or (len(h),):
            m = int(c.query(h[:e], local_only=True, update_meta=False).common_hit)
            slots = c.allocate_slots(e - m)
            for i, s in enumerate(slots):
                view(s, FULL)[:] = pattern(h[m + i], SALT["full"], FULL_BYTES)
            errors.add(str(c.insert(h[:e], slots, start=m).error))
            sw = c.allocate_slots(WINDOW, SWA)
            for i, s in enumerate(sw):
                view(s, SWA)[:] = pattern(h[e - WINDOW + i], SALT["swa"], SWA_BYTES)
            errors.add(str(c.insert(h[:e], sw, component=SWA).error))
            mb = c.allocate_slots(1, MAMBA)
            view(mb[0], MAMBA)[:] = pattern(h[e - 1], SALT["mamba"], MAMBA_BYTES)
            errors.add(str(c.insert(h[:e], mb, component=MAMBA).error))
        return errors

    def push(h, rank, valid=None):
        q = joint(h)
        swa = [(len(h), q.swa_slots)] if len(q.swa_slots) else []
        mamba = [(len(h), int(q.mamba[1]))] if q.mamba else []
        r = c.push_async(rank, h, full_of(q), swa, mamba, valid=valid).wait(60)
        q.finalize()
        return [r.common_hit, r.remote_hit, r.remote_blocks, r.remote_bytes, r.complete]

    long_path = radixshmem.compute_hashes(np.arange(5_000_000, 5_000_000 + LONG, dtype=np.int64), 1)

    if role.startswith("writer:"):
        errors = set()
        for k in span(role.split(":", 1)[1]):
            deadline = time.time() + 60
            while c.mempool_free() < N and time.time() < deadline:
                time.sleep(0.005)     # background eviction makes room, so allocate_slots never evicts
            errors |= put(hs(k), (MID, N))
            time.sleep(0.005)
        c.flush()
        out["insert_errors"] = sorted(errors)
    elif role.startswith("push:"):
        target = int(role.split(":", 1)[1])
        out["new"] = push(hs(0), target)
        out["sibling"] = push(hs(1), target)
        out["again"] = push(hs(0), target)
        out["refused"] = push(hs(2), target, valid=lambda: False)
        out["self"] = push(hs(3), c.rank())
    elif role.startswith("nospace:"):
        target, h = int(role.split(":", 1)[1]), hs(3)
        out["insert_errors"] = sorted(put(h))
        held, n = [], 64
        while n:
            r = c.remote_allocate_components(target, [(MAMBA, n)])
            if r.error == radixshmem.InsertError.OK and r.status == radixshmem.RemoteStatus.OK:
                held.append(r.slots[0])
            else:
                n //= 2
        q = joint(h)
        job = c.push_async(target, h, full_of(q), [(N, q.swa_slots)], [(N, int(q.mamba[1]))])
        r = job.wait(60)
        q.finalize()
        out["nospace"] = [r.common_hit, r.remote_hit, r.remote_blocks, r.remote_bytes, r.complete, job.no_space]
        out["recycled"] = sorted({str(c.remote_recycle(target, s, MAMBA)) for s in held})
    elif role.startswith("long:"):
        out["insert_errors"] = sorted(put(long_path))
        out["long"] = push(long_path, int(role.split(":", 1)[1]))
    elif role == "longcensus":
        q = joint(long_path)
        out["row"] = [int(c.query(long_path, local_only=True, update_meta=False).common_hit), int(q.common_hit),
                      verify(long_path, q) if int(q.common_hit) == LONG else 0]
        q.finalize()
    elif role.startswith("midcensus:"):
        rows = []
        for k in span(role.split(":", 1)[1]):
            h = hs(k)[:MID]
            q = joint(h)
            rows.append([int(q.common_hit), verify(h, q) if int(q.common_hit) == MID else 0])
            q.finalize()
        out["rows"] = rows
    elif role.startswith("census:"):
        rows = []   # per prefix: FULL hit, joint FULL | SWA | MAMBA hit, bad slots
        for k in span(role.split(":", 1)[1]):
            h = hs(k)
            q = joint(h)
            rows.append([int(c.query(h, local_only=True, update_meta=False).common_hit), int(q.common_hit),
                         verify(h, q) if int(q.common_hit) == N else 0])
            if q.finalize is not None:
                q.finalize()
        out["rows"] = rows
    elif role.startswith("puller:"):
        ks = [int(x) for x in role.split(":", 1)[1].split(",")]
        res = [j.wait(120) for j in [c.pull_async(hs(k), ALL) for k in ks]]
        out["hits"] = [int(r.common_hit) for r in res]
        for r in res:
            r.finalize()
        bad = 0
        for k in ks:
            q = joint(hs(k))
            bad += verify(hs(k), q) if int(q.common_hit) == N else 1000
            if q.finalize is not None:
                q.finalize()
        out["bad"] = bad
    elif role.startswith("route:"):
        out["hits"] = [int(c.query(hs(k), local_only=False, update_meta=False).common_hit)
                       for k in span(role.split(":", 1)[1])]
    elif role == "stats":
        stub = rpc.Stub(endpoint)
        out["stats"] = stub.call("Stats", timeout=30)
        stub.close()
    elif role == "health":
        out["health"] = int(c.check_health())
        out["report"] = c.health_report().describe().strip()
    print("RESULT " + json.dumps(out), flush=True)
    c.close()
    return 0


# ------------------------------------------------------------ orchestrator

def run_client(role, endpoint, timeout=240):
    p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--client", role, endpoint],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate(timeout=timeout)
    lines = [l for l in out.splitlines() if l.startswith("RESULT ")]
    if not lines:
        raise RuntimeError(f"client {role} rc={p.returncode}\nstdout:\n{out}\nstderr:\n{err[-3000:]}")
    return json.loads(lines[0][len("RESULT "):])


def wait_writeback(ep, threshold, timeout=90):
    """node0's stats once background eviction is done: free at the threshold, nothing queued or open."""
    deadline = time.time() + timeout
    while True:
        st = run_client("stats", ep)["stats"]
        wb = st["writeback"]
        free = st["mempool_total"] - st["mempool_used"]
        if (free >= threshold and wb["queued"] == 0 and wb["popped"] == 0) or time.time() > deadline:
            return st
        time.sleep(0.5)


def test_push(eps, ranks):
    print("\n[push] a client on node0 push_async()s prefixes to node1")
    w = run_client(f"writer:0:{PUSHED}", eps[0])
    p2p.check(w["insert_errors"] == ["InsertError.OK"], f"node0 holds prefixes 0..{PUSHED - 1}")
    r = run_client(f"push:{ranks[1]}", eps[0])
    print(f"  {r}")
    rest = WINDOW * SWA_BYTES + MAMBA_BYTES
    p2p.check(r["new"] == [N, 0, N + WINDOW + 1, N * FULL_BYTES + rest, True],
          "a new path lands whole: FULL, SWA window and Mamba checkpoint")
    p2p.check(r["sibling"] == [N, STEM, TAIL + WINDOW + 1, TAIL * FULL_BYTES + rest, True],
          "its sibling writes only the tail: the stem is deduplicated")
    p2p.check(r["again"][:2] == [N, N] and r["again"][4], "the same path again writes no FULL block")
    p2p.check(r["refused"] == [0, 0, 0, 0, False], "valid() == False publishes nothing")
    p2p.check(r["self"][0] == 0 and not r["self"][4], "a push to its own rank is refused")
    rows = run_client(f"census:0:{PUSHED}", eps[1])["rows"]
    p2p.check(rows[0][1:] == [N, 0] and rows[1][1:] == [N, 0], "node1 holds prefixes 0 and 1, bytes intact")
    p2p.check(rows[2][0] == 0 and rows[3][0] == 0, "and nothing of prefixes 2 and 3")
    h = run_client("health", eps[1])
    p2p.check(h["health"] == 0, f"node1 leaks no slot and no pin ({h['report']})")
    r = run_client(f"nospace:{ranks[2]}", eps[3])
    p2p.check(r["insert_errors"] == ["InsertError.OK"] and r["nospace"] == [0, 0, 0, 0, False, True] and
              r["recycled"] == ["RemoteStatus.OK"], "a target without a free Mamba slot gets nothing written")
    p2p.check(run_client("census:3:4", eps[2])["rows"][0][0] == 0, "node2 holds no block of it")
    h = run_client("health", eps[2])
    p2p.check(h["health"] == 0, f"node2 leaks no slot and no pin ({h['report']})")
    r = run_client(f"long:{ranks[2]}", eps[3])
    p2p.check(r["insert_errors"] == ["InsertError.OK"] and
              r["long"] == [LONG, 0, LONG + WINDOW + 1, LONG * FULL_BYTES + rest, True],
              f"node3 pushes a {LONG}-block path to node2 whole")
    p2p.check(run_client("longcensus", eps[2])["row"] == [LONG, LONG, 0], "node2 holds it, bytes intact")


def test_writeback(eps, full_slots):
    pairs = 2 * full_slots // (STEM + 2 * TAIL)
    first, last = PUSHED, PUSHED + 2 * pairs
    print(f"\n[writeback] a writer puts {last - first} prefixes ({pairs * (STEM + 2 * TAIL)} blocks) "
          f"through node0's {full_slots} FULL slots")
    w = run_client(f"writer:{first}:{last}", eps[0])
    p2p.check(w["insert_errors"] == ["InsertError.OK"], "node0's writer inserted every prefix")
    st = wait_writeback(eps[0], int(EVICT_RATIO * full_slots))
    wb = st["writeback"]
    print(f"  node0 writeback {wb}  transfers {st['transfers']} ({st['transfer_failures']} failed)")
    p2p.check(wb["active"], "background eviction holds for write-back while decode nodes are ready")
    p2p.check(wb["produced"] > 0 and wb["completed"] == wb["produced"] and wb["failed"] == 0 and wb["dropped"] == 0,
          "background eviction produced tasks and PeerWriteback completed all of them")
    p2p.check(wb["pushed"] == wb["produced"] and wb["incomplete"] == wb["no_target"] == wb["too_long"] == 0,
          "PeerWriteback pushed every task whole")
    p2p.check(st["transfers"] > 0 and st["transfer_failures"] == 0, "every push's RDMA write completed")
    span = f"{first}:{last}"
    src = run_client(f"census:{span}", eps[0])["rows"]
    dst = {i: run_client(f"census:{span}", eps[i])["rows"] for i in (1, 2, 3)}
    moved = [k for k, r in enumerate(src) if r[0] < N]
    print(f"  {len(moved)} of {len(src)} prefixes left node0")
    p2p.check(len(moved) >= pairs, "background eviction took at least half of them")
    p2p.check(all(r[1] == N and r[2] == 0 for r in src if r[0] == N), "the prefixes node0 kept are intact")
    holders = {k: [i for i in (1, 2) if dst[i][k][1] == N and dst[i][k][2] == 0] for k in moved}
    lost = [first + k for k, hs in holders.items() if not hs]
    p2p.check(not lost, f"every prefix that left node0 is on node1 or node2 with FULL + SWA + Mamba, bytes intact "
                    f"(lost: {lost})")
    mid = {i: run_client(f"midcensus:{span}", eps[i])["rows"] for i in (1, 2)}
    short = [first + k for k, who in holders.items() if mid[who[0]][k] != [MID, 0]]
    p2p.check(not short, f"and with the SWA window and Mamba checkpoint of its turn at block {MID} (short: {short})")
    pair = {}
    for k, hs in holders.items():
        pair.setdefault((first + k) // 2, set()).add(tuple(hs))
    p2p.check(all(len(hs) == 1 and len(next(iter(hs))) == 1 for hs in pair.values()),
          "each prefix went to one node, the one its sibling went to")
    p2p.check(all(any(i in hs for hs in holders.values()) for i in (1, 2)), "the first blocks spread them over both: "
          f"node1 got {sum(1 in hs for hs in holders.values())}, node2 {sum(2 in hs for hs in holders.values())}")
    p2p.check(all(r[0] == 0 for r in dst[3]), "the prefill node3 got none of them")
    route = run_client(f"route:{span}", eps[0])["hits"]
    miss = [first + k for k in moved if route[k] != N]
    p2p.check(not miss, f"node0's cluster query finds every one of them whole (short: {miss})")
    back = [first + k for k in moved[:6]]
    r = run_client("puller:" + ",".join(map(str, back)), eps[0])
    p2p.check(r["hits"] == [N] * len(back) and r["bad"] == 0, f"node0 pulls {back} back intact ({r})")
    wait_writeback(eps[0], int(EVICT_RATIO * full_slots))   # held ancestors keep refs until done


def orchestrate(a) -> int:
    import radixshmem

    if (not shutil.which("etcd") or not os.path.isdir(f"/sys/class/infiniband/{a.index_dev}")
            or not hasattr(radixshmem, "_data") or "mooncake" not in radixshmem._data.TRANSFER_BACKENDS
            or not hasattr(radixshmem._core, "EtcdKv")):
        print(f"SKIP: needs etcd, RDMA device {a.index_dev!r} and radixshmem built with etcd + Mooncake")
        return 77
    workdir = tempfile.mkdtemp(prefix="radixshmem-push-")
    etcd_proc, servers, logs = None, [], []
    try:
        etcd_proc, cport = p2p.start_etcd(workdir)
        registry = f"etcd://127.0.0.1:{cport}"
        print(f"etcd on {registry}   logs in {workdir}")
        eps = [f"unix://{workdir}/node{i}.sock" for i in range(4)]
        for i, size in enumerate(("16M", "64M", "64M", "64M")):
            logs.append(os.path.join(workdir, f"node{i}.log"))
            cmd = [sys.executable, "-m", "radixshmem.cli", "--name", NAME, "--node-name", f"node{i}",
                   "--data-name", f"{NAME}_data{i}", "--data-bytes", size, "--no-prefault",
                   "--swa-ratio", "0.2", "--mamba-ratio", "0.1", "--endpoint", eps[i],
                   "--registry", registry, "--cluster-id", f"push{os.getpid()}",
                   "--expected-min-nodes", "4", "--rpc-address", "0.0.0.0", "--transfer-ip", "127.0.0.1",
                   "--bootstrap-timeout", str(a.bootstrap_timeout), "--interval", "1"]
            if i == 0:
                cmd += ["--background-evict-ratio", str(EVICT_RATIO), "--writeback-max-held", "0.25"]
            else:
                cmd += ["--background-evict-ratio", "0"] + (["--decode"] if i < 3 else [])
            cmd += [f"--transfer-dev={d}" for d in a.transfer_dev]
            servers.append(subprocess.Popen(cmd, stdout=open(logs[i], "wb"), stderr=subprocess.STDOUT,
                                            env=dict(os.environ, PYTHONUNBUFFERED="1")))
        p2p.wait_log(logs, servers, "Waiting for a client's geometry", 60)
        geo = radixshmem.Geometry(block_size=1, full_slot_bytes=FULL_BYTES, swa_slot_bytes=SWA_BYTES,
                                  swa_window_blocks=WINDOW, mamba_slot_bytes=MAMBA_BYTES)
        cl0 = radixshmem.RadixClient(NAME, geo, endpoint=eps[0])
        p2p.wait_log(logs, servers, "Cluster ready", a.bootstrap_timeout + 30)
        for ep in eps:
            radixshmem.RadixClient(NAME, endpoint=ep).wait_ready(30)
        full_slots = cl0.wait_ready(10).geometry["pools"]["full"]["num_slots"]
        cl0.close()
        ranks = [run_client("stats", ep)["stats"]["rank"] for ep in eps]
        print(f"  ranks {ranks}; node0 has {full_slots} FULL slots")

        test_push(eps, ranks)
        test_writeback(eps, full_slots)
        print("\n[health]")
        for i, ep in enumerate(eps):
            h = run_client("health", ep)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dev", default="mlx5_bond_0")
    ap.add_argument("--transfer-dev", action="append", default=None)
    ap.add_argument("--gid-idx", type=int, default=3)
    ap.add_argument("--bootstrap-timeout", type=int, default=60)
    a = ap.parse_args()
    # The device of the planes that run on RDMA; the servers inherit it.
    os.environ["RADIXSHMEM_INDEX_DEV"] = a.index_dev
    os.environ["RADIXSHMEM_INDEX_GID_IDX"] = str(a.gid_idx)
    a.transfer_dev = a.transfer_dev or [a.index_dev]
    return orchestrate(a)


if __name__ == "__main__":
    sys.exit(main())
