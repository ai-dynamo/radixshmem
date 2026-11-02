# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The monitoring endpoint's tree algorithms against brute force on synthetic
radix trees (numpy only, no shm): anchors, the merged tree's distinct blocks
and duplication, view() bookkeeping, and the depth-first rows the Grafana
flame graph reads."""

import os
import random

import numpy as np

from radixshmem import treeview
from radixshmem.monitor import view_rows


def prefix_hashes(blocks):
    """Chained hashes like gen_hashes: hash i covers blocks [0, i]."""
    out, h = [], 1469598103934665603
    for b in blocks:
        h = (h * 1099511628211 ^ (b + 1)) & 0xFFFFFFFFFFFFFFFF
        out.append(h or 1)
    return out


def radix_dump(seqs):
    """A dump_tree()-shaped dict of the radix tree holding every sequence."""
    trie = {}
    for s in seqs:
        node = trie
        for h in prefix_hashes(s):
            node = node.setdefault(h, {})

    # compress unary chains into radix nodes, BFS order
    parent, level, depth, nb, hashes = [-1], [0], [0], [0], []
    queue = [(trie, 0, 0, 0)]
    q = 0
    while q < len(queue):
        node, idx, lvl, dep = queue[q]
        q += 1
        for h, child in node.items():
            run = [h]
            while len(child) == 1:
                (h2, child), = child.items()
                run.append(h2)
            parent.append(idx)
            level.append(lvl + 1)
            depth.append(dep)
            nb.append(len(run))
            hashes.extend(run)
            queue.append((child, len(parent) - 1, lvl + 1, dep + len(run)))
    # BFS by construction only within a level if the queue is level-ordered: it is.
    n = len(parent)
    return {
        "parent": np.array(parent, np.int32), "level": np.array(level, np.uint32),
        "depth": np.array(depth, np.uint32), "num_blocks": np.array(nb, np.uint32),
        "swa_blocks": np.zeros(n, np.uint32), "ref_count": np.zeros(n, np.int64),
        "last_access": np.full(n, 10**9, np.uint64), "mamba": np.zeros(n, np.uint8),
        "hashes": np.array(hashes, np.uint64), "now": 2 * 10**9,
    }


def workload(rng, ntrees):
    prompts = [[rng.randrange(1000) for _ in range(rng.randint(3, 12))] for _ in range(4)]
    trees = []
    for _ in range(ntrees):
        seqs = []
        for _ in range(rng.randint(5, 25)):
            p = list(rng.choice(prompts)) if rng.random() < 0.8 else []
            seqs.append(p + [rng.randrange(1000) for _ in range(rng.randint(1, 20))])
        trees.append(seqs)
    return trees


def test_node_tree_totals():
    rng = random.Random(1)
    for seqs in workload(rng, 5):
        d = radix_dump(seqs)
        idx = treeview.node_tree(d)
        assert idx.total == len(d["hashes"])
        distinct = {h for s in seqs for h in prefix_hashes(s)}
        assert idx.total == len(distinct)


def test_merge_stride1_matches_brute_force():
    rng = random.Random(2)
    trees = workload(rng, 6)
    dumps = [radix_dump(s) for s in trees]
    parts = [treeview.anchors(d, 1) for d in dumps]
    for d, p in zip(dumps, parts):
        assert int(p["blocks"].sum()) == len(d["hashes"])
    mt = treeview.MergedTree(parts)
    sets = [{h for s in seqs for h in prefix_hashes(s)} for seqs in trees]
    union = set().union(*sets)
    assert mt.unique_blocks == len(union)
    assert mt.total_blocks == sum(len(s) for s in sets)
    # every merged group: dup equals the number of trees holding its head prefix
    idx = mt.index
    for g in range(1, idx.n):
        k = int(idx.key[g])
        assert idx.attrs["dup"][g] == sum(k in s for s in sets)
    assert idx.total == len(union)


def test_merge_coarse_stride_keeps_structure():
    rng = random.Random(3)
    trees = workload(rng, 4)
    dumps = [radix_dump(s) for s in trees]
    for stride in (2, 4, 8):
        parts = [treeview.anchors(d, stride) for d in dumps]
        for d, p in zip(dumps, parts):
            # anchors cover every path, so a block shared by two paths may count twice
            assert int(p["blocks"].sum()) >= len(d["hashes"])
            # parents are always anchors of the same tree (or the root)
            known = set(p["hash"].tolist()) | {0}
            assert set(p["parent"].tolist()) <= known
            assert np.all((p["depth"] % stride == 0) | (p["blocks"] < stride))
        mt = treeview.MergedTree(parts)
        assert mt.index.total == mt.unique_blocks


def test_view_bookkeeping():
    rng = random.Random(4)
    dumps = [radix_dump(s) for s in workload(rng, 8)]
    mt = treeview.MergedTree([treeview.anchors(d, 1) for d in dumps])
    for budget in (5, 30, 10000):
        v = mt.index.view(0, budget, extra=mt.owners)
        n = len(v["id"])
        assert n <= budget
        shown = np.zeros(n)
        for i in range(1, n):
            shown[v["parent"][i]] += v["w"][i]
        for i in range(n):
            assert abs(v["w"][i] - (v["self"][i] + shown[i] + v["rest_w"][i])) < 1e-6
            assert len(v["owners"][i]) == (v["dup"][i] if i else 0)
    # zoom: the root's subtree only, with breadcrumbs back to the top
    g = int(np.argmax(np.where(mt.index.level == 2, mt.index.sub, -1)))
    v = mt.index.view(g, 100)
    assert v["total"] == mt.index.sub[g]
    assert v["crumbs"][0]["key"] == "0" * 16 and len(v["crumbs"]) == 2


def test_pack_roundtrip_and_stride():
    d = radix_dump([[1, 2, 3, 4, 5], [1, 2, 9]])
    a = treeview.anchors(d, 2)
    back, meta = treeview.unpack(treeview.pack(a, stride=2, blocks=6))
    assert meta == {"stride": 2, "blocks": 6}
    for f in treeview.ANCHOR_FIELDS:
        np.testing.assert_array_equal(back[f], a[f])
    assert treeview.pick_stride(1000, 50_000) == 1
    assert treeview.pick_stride(1_000_000, 50_000) == 32
    m = treeview.MergedTree([])
    assert m.index.n == 1 and m.unique_blocks == 0


def test_view_rows_flamegraph_order():
    """Rows are depth-first: a parent comes first, its subtree is contiguous,
    levels step by one, and value = self + the children's values."""
    rng = random.Random(5)
    dumps = [radix_dump(s) for s in workload(rng, 5)]
    mt = treeview.MergedTree([treeview.anchors(d, 1) for d in dumps])
    for idx, merged in ((mt.index, True), (treeview.node_tree(dumps[0]), False)):
        for budget in (4, 25, 5000):
            rows = view_rows(idx.view(0, budget), merged)
            assert [r["order"] for r in rows] == list(range(len(rows)))
            assert rows[0]["parent"] == -1 and rows[0]["label"] == ("cluster" if merged else "root")
            kids = {}
            for r in rows[1:]:
                assert r["parent"] < r["order"]
                assert r["level"] == rows[r["parent"]]["level"] + 1
                kids.setdefault(r["parent"], []).append(r)
            for r in rows:
                assert abs(r["value"] - r["self"] - sum(c["value"] for c in kids.get(r["order"], []))) < 1e-6
            # depth-first: every subtree is one contiguous run of orders
            end = {}
            for r in reversed(rows):
                end[r["order"]] = max([r["order"]] + [end[c["order"]] for c in kids.get(r["order"], [])])
            for r in rows[1:]:
                assert r["order"] <= end[r["parent"]]
            assert abs(rows[0]["value"] - idx.total) < 1e-6


def test_server_monitor_endpoint(tmp_path):
    """A standalone server with monitor_port: the monitor runs as a child
    process; /metrics carries the counters, the tree size and its rows; /sd
    lists itself; close() ends the process and frees the port."""
    import json
    import socket
    import time
    import urllib.request

    import radixshmem
    from radixshmem.server import RadixServer, ServerConfig

    with socket.socket() as s:
        s.bind(("0.0.0.0", 0))
        port = s.getsockname()[1]
    get = lambda u: urllib.request.build_opener(urllib.request.ProxyHandler({})).open(u, timeout=5).read()  # noqa: E731
    ep = f"unix://{tmp_path}/mon.sock"
    srv = RadixServer(ServerConfig(name="/py_monitor", endpoint=ep, monitor_port=port,
                                   monitor_interval_s=1, monitor_tree_nodes=50)).start()
    try:
        assert srv.monitor.port == port and srv.nodes()[0]["monitor"] == srv.monitor.url
        child = srv.monitor.proc
        assert child.poll() is None and child.pid != os.getpid()
        cl = radixshmem.RadixClient("/py_monitor", radixshmem.Geometry(block_size=4, full_slots=1024), endpoint=ep)
        core = radixshmem._core.RadixClient(cl.wait_ready(30).index_name)
        for k in range(4):
            h = radixshmem.compute_hashes(np.array([7] * 8 + [k * 100 + i for i in range(8)], np.int64), 4)
            m = int(core.query(h, update_meta=False).common_hit)
            core.insert(h, core.allocate_slots(len(h) - m), start=m)
        deadline = time.time() + 30
        while True:                                        # the process' next round sees the inserts
            text = get(f"http://127.0.0.1:{port}/metrics").decode()
            if 'radixshmem_tree_blocks{' in text and text.split("radixshmem_tree_blocks{")[1].split("\n")[0] \
                    .endswith(" 10.0") or time.time() > deadline:
                break
            time.sleep(0.5)
        assert 'radixshmem_tree_blocks{' in text and text.count("radixshmem_tree_blocks{") == 1
        blocks = [l for l in text.splitlines() if l.startswith("radixshmem_tree_blocks{")][0]
        assert blocks.endswith(" 10.0")                    # 2 shared + 4 x 2
        rows = [l for l in text.splitlines() if l.startswith("radixshmem_treeview_blocks{")]
        node_rows = [l for l in rows if 'view="node"' in l]
        assert len(node_rows) == 6 and 'label="root"' in node_rows[0] and node_rows[0].endswith(" 10.0")
        assert 'level="0",self="0"' in node_rows[0] and "radixshmem_treeview_edge" not in text
        pools = [l for l in text.splitlines() if l.startswith("radixshmem_pool_slots_")]
        assert len(pools) == 2 and all('pool="full"' in l for l in pools)      # no SWA / Mamba pool here
        assert any(l.startswith("radixshmem_pool_slots_used{") and l.endswith(" 10.0") for l in pools)
        assert any(l.startswith("radixshmem_pool_slots_total{") and l.endswith(" 1024.0") for l in pools)
        assert 'text="10 / 1024"} 1.0' in text and "radixshmem_data_bytes{" in text
        assert text.count("# TYPE radixshmem_treeview_blocks ") == 1       # node + cluster rows, one family
        assert any('view="cluster"' in l for l in rows)    # standalone: its own leader
        assert 'radixshmem_cluster_dup_factor{cluster="default"} 1.0' in text
        sd = json.loads(get(f"http://127.0.0.1:{port}/sd"))
        assert sd[0]["targets"] == [srv.monitor.url.split("//")[1]]
        cl.close()
    finally:
        srv.close()
    assert child.poll() is not None                        # ended with the server
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", port))                          # released
        s.listen()


def test_top_shared_counts_servers_per_node_head():
    """A radix node's first block hash counted once per server; only counts
    above one, most servers first, then the larger node."""
    common = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 9, 9]]           # node heads: [1..3], [4..6], [9, 9]
    trees = [common + [[7, 7]], common + [[8]], [[1, 2, 3, 4, 5, 6], [30, 31]]]
    parts = []
    for seqs in trees:
        d = radix_dump(seqs)
        parts.append(treeview.node_heads(d, treeview.node_tree(d), 0))
    top = treeview.top_shared(parts, 20)
    h = prefix_hashes
    # the third tree holds [1..6] as one node (nothing branches at block 3 there), so the node
    # starting at block 4 counts on two servers only: heads match where node boundaries match
    assert [(x[0], x[1]) for x in top] == [(h([1])[0], 3), (h([1, 2, 3, 4])[-1], 2), (h([1, 2, 3, 9])[-1], 2)]
    assert top[0][2:] == (3, 0) and top[1][2:] == (3, 3) and top[2][2:] == (2, 3)
    assert treeview.top_shared(parts[:1], 20) == []


def test_monitor_process_exits_with_a_killed_server(tmp_path):
    """SIGKILL the server: the monitor process notices (PR_SET_PDEATHSIG, or
    its parent-pid check) and exits."""
    import signal
    import subprocess
    import sys
    import time

    script = f"""
import time
from radixshmem.server import RadixServer, ServerConfig
srv = RadixServer(ServerConfig(name="/py_monitor_kill", endpoint="unix://{tmp_path}/kill.sock",
                               monitor_port=19899, monitor_interval_s=1)).start()
print(srv.monitor.proc.pid, flush=True)
time.sleep(600)
"""
    server = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    child = int(server.stdout.readline())
    os.kill(child, 0)                                      # running
    server.send_signal(signal.SIGKILL)
    server.wait()
    def gone():   # exited: no such process, or a zombie its new parent (pid 1) has not reaped
        try:
            with open(f"/proc/{child}/status") as f:
                return any(l.startswith("State:") and "Z" in l.split()[1] for l in f)
        except FileNotFoundError:
            return True

    deadline = time.time() + 15
    while time.time() < deadline:
        if gone():
            break
        time.sleep(0.2)
    else:
        os.kill(child, signal.SIGKILL)
        raise AssertionError("monitor process outlived its server")
    for f in os.listdir("/dev/shm"):                       # what the killed server left
        if f.startswith("py_monitor_kill"):
            os.unlink(os.path.join("/dev/shm", f))


if __name__ == "__main__":
    import tempfile
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(tempfile.mkdtemp()) if name == "test_server_monitor_endpoint" else fn()
            print("PASS", name)
