# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Monitoring child process of a radix-server (``ServerConfig.monitor_port``): copies the tree every interval and serves
/metrics (Prometheus; the tree as ``radixshmem_treeview_blocks`` rows, the leader adds the merged cluster view), /sd, /summary, /anchors, /nodes."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import errno
import json
import os
import select
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlsplit

import numpy as np

from radixshmem import treeview

# Never through http_proxy: peers are cluster-internal.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_PORT_TRIES = 16           # several servers on one host take the next free port
_MAX_NODES = 4_000_000     # snapshot caps; ``truncated`` reports a hit
_MAX_BLOCKS = 64_000_000
_ANCHORS_PER_TREE = 50_000
_ANCHOR_BUDGET = 4_000_000
_HEADS_PER_TREE = 20_000   # heaviest radix nodes each server offers for the shared-node count
_TOP_SHARED = 20


def _fetch(url: str, timeout: float) -> bytes:
    with _OPENER.open(url, timeout=timeout) as r:
        return r.read()


def default_ip() -> str:
    """Source address of the default route (no packet is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def _esc(v: Any) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class _Text:
    """Samples grouped by family; ``render`` writes each family once, with one
    HELP/TYPE, even when its samples come from several ``_Text``."""

    def __init__(self, base: Dict[str, Any]):
        self.base = base
        self.families: Dict[str, list] = {}     # name -> [help, kind, lines]

    def add(self, name: str, value, help_: str = "", kind: str = "gauge",
            labels: Optional[Dict[str, Any]] = None) -> None:
        lab = ",".join(f'{k}="{_esc(v)}"' for k, v in {**self.base, **(labels or {})}.items())
        self.families.setdefault(name, [help_, kind, []])[2].append(f"{name}{{{lab}}} {float(value)}")

    @staticmethod
    def render(*texts: Optional["_Text"]) -> str:
        merged: Dict[str, list] = {}
        for t in texts:
            for name, (help_, kind, lines) in (t.families.items() if t else ()):
                merged.setdefault(name, [help_, kind, []])[2].extend(lines)
        out = []
        for name, (help_, kind, lines) in merged.items():
            out += [f"# HELP {name} {help_}", f"# TYPE {name} {kind}", *lines]
        return "\n".join(out) + "\n" if out else ""


def view_rows(v: Dict, merged: bool) -> List[Dict]:
    """A ``TreeIndex.view`` in depth-first order, heaviest child first, with
    each node's summed rest as its last child."""
    n = len(v["id"])
    kids: List[List[int]] = [[] for _ in range(n)]
    for i in range(1, n):
        kids[v["parent"][i]].append(i)
    for k in kids:
        k.sort(key=lambda c: -v["w"][c])
    rows: List[Dict] = []
    stack: List[tuple] = [("node", 0, -1)]
    while stack:
        kind, i, parent = stack.pop()
        order = len(rows)
        if kind == "rest":
            rows.append({"order": order, "parent": parent, "level": v["level"][i] + 1,
                         "self": v["rest_w"][i], "value": v["rest_w"][i], "dup": 0,
                         "key": v["key"][i] + "~rest", "label": f"+{v['rest_n'][i]} smaller"})
            continue
        dup = int(v["dup"][i]) if merged else 1
        if i == 0:
            label = "cluster" if merged else "root"
        else:
            label = v["key"][i][:8] + (f" x{dup}" if merged else "")
        rows.append({"order": order, "parent": parent, "level": v["level"][i], "self": v["self"][i],
                     "value": v["w"][i], "dup": dup, "key": v["key"][i], "label": label})
        if v["rest_w"][i] > 0:
            stack.append(("rest", i, order))
        for c in reversed(kids[i]):
            stack.append(("node", c, order))
    return rows


def _inode(name: str, hugepage_path: str = "") -> Optional[tuple]:
    base = name.lstrip("/")
    for d in (hugepage_path, "/dev/hugepages", "/dev/shm"):
        if d:
            try:
                st = os.stat(os.path.join(d, base))
                return st.st_dev, st.st_ino
            except OSError:
                continue
    return None


class _Server:
    """The monitor's view of its radix-server: identity, counters and the
    cluster's node records over the server's RPCs, the tree through a
    read-only attach of the index shm (reattached when the region changes)."""

    def __init__(self, endpoint: str):
        from radixshmem import rpc
        self._stub = rpc.Stub(endpoint)
        self._index = None
        self._attached: Optional[tuple] = None

    def describe(self) -> Dict[str, Any]:
        return self._stub.call("Describe", timeout=5)

    def stats(self) -> Dict[str, Any]:
        return self._stub.call("Stats", timeout=5)

    def nodes(self) -> List[Dict[str, Any]]:
        return self._stub.call("Nodes", timeout=5)["nodes"]

    def index(self, info: Dict[str, Any]):
        name = info["index_name"]
        key = (name, _inode(name, info.get("hugepage_path", "")))
        if self._index is None or self._attached != key:
            from radixshmem import _core
            self._index = None
            self._index = _core.RadixClient(name, True)
            self._attached = key
        return self._index

    def drop(self) -> None:
        self._index, self._attached = None, None


class Monitor:
    def __init__(self, server: _Server, port: int, interval_s: float = 10.0, tree_nodes: int = 150,
                 cluster_nodes: int = 300, host: str = "", name: str = "", cluster_id: str = "default",
                 distributed: bool = False):
        self.srv = server
        self.name, self.cluster_id, self.distributed = name, cluster_id, distributed
        self.interval = max(float(interval_s), 1.0)
        self.tree_nodes = tree_nodes
        self.cluster_nodes = cluster_nodes
        self._stop = threading.Event()
        self._mu = threading.Lock()
        self._dump: Optional[Dict] = None
        self._summary: Dict[str, Any] = {}
        self._tree_text: Optional[_Text] = None
        self._cluster_text: Optional[_Text] = None
        self._anchor_cache: Dict[tuple, bytes] = {}
        self._heads: Optional[bytes] = None
        self._pool = cf.ThreadPoolExecutor(max_workers=16, thread_name_prefix="radixshmem-monitor-peer")
        self._httpd = self._bind(port)
        self.port = self._httpd.server_address[1]
        self.url = f"http://{host or default_ip()}:{self.port}"
        self._threads: List[threading.Thread] = []

    # ------------------------------------------------------------ lifecycle

    def _bind(self, port: int) -> ThreadingHTTPServer:
        handler = self._handler()
        last: Optional[OSError] = None
        for p in range(port, port + _PORT_TRIES):
            try:
                srv = ThreadingHTTPServer(("0.0.0.0", p), handler)
                srv.daemon_threads = True
                return srv
            except OSError as e:
                if e.errno != errno.EADDRINUSE:
                    raise
                last = e
        raise OSError(f"monitor: ports {port}..{port + _PORT_TRIES - 1} all in use") from last

    def start(self) -> "Monitor":
        for target, name in ((self._httpd.serve_forever, "radixshmem-monitor-http"),
                             (self._loop, "radixshmem-monitor")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        return self

    def close(self) -> None:
        self._stop.set()
        self._httpd.shutdown()
        self._httpd.server_close()
        for t in self._threads:
            t.join(5)
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _loop(self) -> None:
        while True:
            t0 = time.monotonic()
            try:
                self.refresh()
            except Exception as e:  # noqa: BLE001 - monitoring must never take the server down
                print(f"radixshmem monitor: {type(e).__name__}: {e}", file=sys.stderr)
            if self._stop.wait(max(0.1, self.interval - (time.monotonic() - t0))):
                return

    # ------------------------------------------------------------ identity

    def _labels(self, info: Dict[str, Any]) -> Dict[str, Any]:
        node = info.get("node_name") if self.distributed and info.get("node_name") else socket.gethostname()
        return {"cluster": self.cluster_id, "tree": f"{node}{self.name}", "node": node,
                "rank": info.get("rank", -1), "role": info.get("role", "")}

    def _records(self) -> List[Dict[str, Any]]:
        try:
            return self.srv.nodes()
        except Exception:  # noqa: BLE001 - server busy or etcd hiccup: no peers this round
            return []

    # ------------------------------------------------------------ snapshot

    def refresh(self) -> None:
        info = self.srv.describe()
        labels = self._labels(info)
        t0 = time.monotonic()
        d = None
        if info.get("mode") == "ready":
            # This process maps the region itself: a server tearing it down leaves the
            # mapping valid, and the next round attaches the new one.
            d = self.srv.index(info).dump_tree(_MAX_NODES, _MAX_BLOCKS)
            pools = self.srv.stats().get("pools", {})
        else:
            self.srv.drop()
        if d is None:
            with self._mu:
                self._dump, self._summary, self._tree_text, self._cluster_text = None, {}, None, None
                self._anchor_cache, self._heads = {}, None
            return
        ti = treeview.node_tree(d)
        nb = d["num_blocks"]
        nkids = np.bincount(d["parent"][1:], minlength=len(nb)) if len(nb) > 1 else np.zeros(len(nb))
        summary = {
            "url": self.url, "mode": "ready", **labels,
            "nodes": int(len(nb)), "blocks": int(nb.sum()),
            "leaves": int(np.sum(nkids[1:] == 0)) if len(nb) > 1 else 0,
            "max_depth": int((d["depth"].astype(np.int64) + nb).max()) if len(nb) else 0,
            "pinned_nodes": int(np.sum(d["ref_count"] > 0)),
            "swa_blocks": int(d["swa_blocks"].sum()), "mamba_ckpts": int(d["mamba"].sum()),
            "pools": {k: {"used": u, "total": n} for k, (u, n) in pools.items() if n > 0},  # from the server
            "block_size": int(d["block_size"]),
            "torn": int(d["torn"]), "truncated": bool(d["truncated"]),
            "snapshot_s": time.monotonic() - t0,
        }
        t = _Text(labels)
        for key, name, h in (
                ("blocks", "radixshmem_tree_blocks", "blocks in this server's radix tree"),
                ("nodes", "radixshmem_tree_nodes", "radix nodes"),
                ("leaves", "radixshmem_tree_leaves", "leaf nodes"),
                ("max_depth", "radixshmem_tree_max_depth_blocks", "deepest prefix, blocks"),
                ("pinned_nodes", "radixshmem_tree_pinned_nodes", "nodes with a nonzero ref count"),
                ("swa_blocks", "radixshmem_tree_swa_blocks", "blocks holding an SWA slot"),
                ("mamba_ckpts", "radixshmem_tree_mamba_checkpoints", "Mamba checkpoints in the tree"),
                ("torn", "radixshmem_tree_torn_entries", "entries skipped as inconsistent in the last copy"),
                ("truncated", "radixshmem_tree_truncated", "1 when the last copy hit its size cap"),
                ("snapshot_s", "radixshmem_tree_snapshot_seconds", "time to copy and index the tree")):
            t.add(name, summary[key], h)
        if "full" in summary["pools"]:   # a string for the dashboard's table; Grafana cannot join two numbers
            t.add("radixshmem_tree_blocks_of_slots", 1, "1; text = blocks in the tree / FULL slots",
                  labels={"text": f"{summary['blocks']} / {summary['pools']['full']['total']}"})
        for pool, c in summary["pools"].items():     # pools the geometry has: full, swa, mamba
            t.add("radixshmem_pool_slots_used", c["used"], "slots in use, per pool", labels={"pool": pool})
            t.add("radixshmem_pool_slots_total", c["total"], "slots, per pool", labels={"pool": pool})
        if self.tree_nodes > 0:
            self._add_rows(t, view_rows(ti.view(0, self.tree_nodes), merged=False), "node")
        heads = treeview.pack(treeview.node_heads(d, ti, _HEADS_PER_TREE))
        with self._mu:
            self._dump, self._summary = d, summary
            self._anchor_cache, self._heads = {}, heads
        cluster = None
        if self._is_leader():
            try:
                cluster = self._merge(labels)
            except Exception as e:  # noqa: BLE001 - a failed merge must not hold back this node's rows
                print(f"radixshmem monitor: merge: {type(e).__name__}: {e}", file=sys.stderr)
        with self._mu:   # one step: a reader never sees the new tree rows without the cluster rows
            self._tree_text, self._cluster_text = t, cluster

    @staticmethod
    def _add_rows(t: _Text, rows: List[Dict], view: str) -> None:
        """Grafana's flame graph looks fields up by name (its rename transformation
        only sets display names), so the labels carry ``label``, ``level`` and
        ``self`` under exactly those names."""
        width = len(str(len(rows)))
        order = lambda i: str(i).zfill(width)  # noqa: E731
        for r in rows:
            t.add("radixshmem_treeview_blocks", r["value"],
                  "one node of a tree view per series, depth-first by order; value = blocks under it",
                  labels={"view": view, "order": order(r["order"]),
                          "parent": "" if r["parent"] < 0 else order(r["parent"]),
                          "level": r["level"], "self": int(r["self"]), "dup": r["dup"], "key": r["key"],
                          "label": r["label"]})

    # ------------------------------------------------------------ cluster tree

    def _peers(self) -> List[Dict[str, Any]]:
        return [r for r in self._records() if r.get("monitor") and r.get("mode") == "ready"]

    def _is_leader(self) -> bool:
        peers = self._peers()
        if not peers:
            return True
        return min(peers, key=lambda r: (r.get("rank", -1), r["monitor"]))["monitor"] == self.url

    def _remote(self, url: str, path: str, timeout: float):
        try:
            return _fetch(url + path, timeout)
        except Exception:  # noqa: BLE001 - a peer down or restarting
            return None

    def _merge(self, labels: Dict[str, Any]) -> _Text:
        t0 = time.monotonic()
        others = [r["monitor"] for r in self._peers() if r["monitor"] != self.url]
        sums = [self._summary] + [json.loads(b) for b in self._pool.map(
            lambda u: self._remote(u, "/summary", 3.0), others) if b]
        sums = [s for s in sums if s.get("blocks") is not None]
        per_tree = max(1000, min(_ANCHORS_PER_TREE, _ANCHOR_BUDGET // max(len(sums), 1)))
        room = max(per_tree - max(s.get("leaves", 0) for s in sums), per_tree // 4)
        stride = treeview.pick_stride(max(s["blocks"] for s in sums), room)
        q = f"/anchors?stride={stride}&max={2 * per_tree}"
        parts = [treeview.anchors(self._dump, stride, 2 * per_tree)]
        failed = 0
        for b in self._pool.map(lambda u: self._remote(u, q, 10.0), others):
            if b is None:
                failed += 1
            else:
                parts.append(treeview.unpack(b)[0])
        mt = treeview.MergedTree(parts)
        heads = [treeview.unpack(b)[0] for b in [self._heads] + list(self._pool.map(
            lambda u: self._remote(u, "/nodes", 10.0), others)) if b]
        base = {"cluster": labels["cluster"]}
        t = _Text(base)
        t.add("radixshmem_cluster_leader", 1, "1 on the server that merges the cluster tree",
              labels={"tree": labels["tree"]})
        t.add("radixshmem_cluster_trees", len(parts), "trees in the last merge")
        t.add("radixshmem_cluster_merge_failures", failed, "peers whose anchors could not be fetched")
        t.add("radixshmem_cluster_blocks", mt.total_blocks, "blocks over all trees, as merged")
        t.add("radixshmem_cluster_unique_blocks", mt.unique_blocks, "distinct prefix blocks over all trees")
        t.add("radixshmem_cluster_dup_factor", mt.total_blocks / mt.unique_blocks if mt.unique_blocks else 0,
              "blocks over all trees / distinct prefix blocks")
        t.add("radixshmem_cluster_stride_blocks", stride, "anchor stride of the merge, blocks")
        t.add("radixshmem_cluster_merge_seconds", time.monotonic() - t0, "time to fetch and merge")
        for n, (h, servers, blocks, depth) in enumerate(treeview.top_shared(heads, _TOP_SHARED)):
            t.add("radixshmem_cluster_top_shared_nodes", servers,
                  "radix nodes starting at the same block hash on the most servers (> 1); value = servers",
                  labels={"order": str(n + 1).zfill(2), "hash": treeview.hex_key(h), "blocks": blocks,
                          "depth": depth})
        if self.cluster_nodes > 0:
            self._add_rows(t, view_rows(mt.index.view(0, self.cluster_nodes), merged=True), "cluster")
        return t

    # ------------------------------------------------------------ HTTP

    def metrics(self) -> str:
        info, s = self.srv.describe(), self.srv.stats()
        t = _Text(self._labels(info))
        t.add("radixshmem_ready", s.get("mode") == "ready", "1 when the server is ready")
        t.add("radixshmem_data_bytes", info.get("data_bytes", 0),
              "memory budget of the slot store (--data-bytes), bytes; 0 = index only")
        t.add("radixshmem_info", 1, "server identity",
              labels={"mode": s.get("mode", ""), "name": self.name,
                      "index_name": info.get("index_name", ""), "monitor": self.url})
        for key, name, kind, h in (
                ("insert_success", "radixshmem_inserts_total", "counter", "successful inserts"),
                ("insert_errors", "radixshmem_insert_errors_total", "counter", "inserts refused for a full pool"),
                ("transfers", "radixshmem_transfers_total", "counter", "transfers that moved data"),
                ("transfer_failures", "radixshmem_transfer_failures_total", "counter", "failed transfers"),
                ("transfer_bytes", "radixshmem_transfer_bytes_total", "counter", "bytes moved by transfers"),
                ("pending_jobs", "radixshmem_pending_jobs", "gauge", "queued + running + uncollected jobs"),
                ("queued_jobs", "radixshmem_queued_jobs", "gauge", "jobs waiting for the transfer thread"),
                ("peer_down_events", "radixshmem_peer_down_events_total", "counter", "peers marked unreachable"),
                ("rht_writes_dropped", "radixshmem_rht_writes_dropped_total", "counter", "RHT publishes dropped"),
                ("rht_removes_dropped", "radixshmem_rht_removes_dropped_total", "counter", "RHT removals dropped"),
                ("rht_lookups_skipped", "radixshmem_rht_lookups_skipped_total", "counter",
                 "RHT probes answered as misses"),
                ("walks_skipped", "radixshmem_walks_skipped_total", "counter", "remote walks skipped")):
            if key in s:
                t.add(name, s[key], h, kind)
        for key, v in (s.get("writeback") or {}).items():
            if isinstance(v, (int, float)):
                kind = "gauge" if key in ("queued", "popped") else "counter"
                t.add(f"radixshmem_writeback_{key}" + ("_total" if kind == "counter" else ""), v,
                      f"write-back {key}", kind)
        with self._mu:
            return _Text.render(t, self._tree_text, self._cluster_text)

    def sd(self) -> List[Dict[str, Any]]:
        targets = sorted({urlsplit(r["monitor"]).netloc for r in self._records() if r.get("monitor")})
        return [{"targets": targets, "labels": {}}]

    def anchors(self, stride: int, cap: int) -> Optional[bytes]:
        with self._mu:
            d, hit = self._dump, self._anchor_cache.get((stride, cap))
        if d is None or hit is not None:
            return hit
        data = treeview.pack(treeview.anchors(d, stride, cap))
        with self._mu:
            if self._dump is d:
                self._anchor_cache = {(stride, cap): data}
        return data

    def _handler(self):
        mon = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):  # noqa: N802 - http.server API
                u = urlsplit(self.path)
                q = parse_qs(u.query)
                try:
                    if u.path == "/metrics":
                        body, ctype = mon.metrics().encode(), "text/plain; version=0.0.4"
                    elif u.path == "/sd":
                        body, ctype = json.dumps(mon.sd()).encode(), "application/json"
                    elif u.path == "/summary":
                        body, ctype = json.dumps(mon._summary).encode(), "application/json"
                    elif u.path == "/nodes":
                        if mon._heads is None:
                            return self._send(503, b"tree not ready", "text/plain")
                        body, ctype = mon._heads, "application/octet-stream"
                    elif u.path == "/anchors":
                        data = mon.anchors(int(q.get("stride", ["1"])[0]), int(q.get("max", ["0"])[0]))
                        if data is None:
                            return self._send(503, b"tree not ready", "text/plain")
                        body, ctype = data, "application/octet-stream"
                    else:
                        return self._send(404, b"/metrics /sd /summary /anchors /nodes", "text/plain")
                except Exception as e:  # noqa: BLE001
                    return self._send(500, f"{type(e).__name__}: {e}".encode(), "text/plain")
                self._send(200, body, ctype)

            def _send(self, status: int, body: bytes, ctype: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):
                pass

        return Handler


# ------------------------------------------------------------ the monitor process

_READY = "RADIXSHMEM-MONITOR"


def _die_with_parent(parent_pid: int) -> None:
    """SIGTERM from the kernel when the server exits, even on SIGKILL (Linux)."""
    try:
        import ctypes
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM)   # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001 - not Linux: the main loop's ppid check still ends it
        pass
    if os.getppid() != parent_pid:
        sys.exit(0)


def main(argv=None) -> None:
    """Run by the server as ``python -m radixshmem.monitor``: everything the
    monitor does happens here, so it shares no GIL with the server."""
    p = argparse.ArgumentParser(prog="radixshmem.monitor")
    p.add_argument("--endpoint", required=True, help="the server's RPC endpoint")
    p.add_argument("--name", required=True, help="the server's --name")
    p.add_argument("--cluster-id", default="default")
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--host", default="", help="address peers and Prometheus reach (default: the default route's)")
    p.add_argument("--interval", type=float, default=10.0)
    p.add_argument("--tree-nodes", type=int, default=150)
    p.add_argument("--cluster-nodes", type=int, default=300)
    p.add_argument("--parent-pid", type=int, required=True)
    a = p.parse_args(argv)
    _die_with_parent(a.parent_pid)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    mon = Monitor(_Server(a.endpoint), a.port, a.interval, a.tree_nodes, a.cluster_nodes, a.host,
                  a.name, a.cluster_id, a.distributed)
    print(f"{_READY} {mon.port} {mon.url}", flush=True)
    os.dup2(2, 1)                      # nothing more on the pipe the server read the port from
    mon.start()
    while not stop.wait(1.0) and os.getppid() == a.parent_pid:
        pass
    mon.close()


class MonitorProcess:
    """The server's handle on its monitor process: starts it, learns the port
    it bound, and ends it on close()."""

    def __init__(self, cfg, host: str = "", timeout_s: float = 60.0):
        cmd = [sys.executable, "-m", "radixshmem.monitor", "--endpoint", cfg.resolved_endpoint,
               "--name", cfg.name, "--cluster-id", cfg.cluster.cluster_id, "--port", str(cfg.monitor_port),
               "--interval", str(cfg.monitor_interval_s), "--tree-nodes", str(cfg.monitor_tree_nodes),
               "--cluster-nodes", str(cfg.monitor_cluster_nodes), "--parent-pid", str(os.getpid())]
        cmd += ["--host", host] if host else []
        cmd += ["--distributed"] if cfg.distributed else []
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, close_fds=True)
        line = b""
        if select.select([self.proc.stdout], [], [], timeout_s)[0]:
            line = self.proc.stdout.readline()
        self.proc.stdout.close()
        parts = line.decode(errors="replace").split()
        if len(parts) != 3 or parts[0] != _READY:
            self.close()
            raise RuntimeError(f"monitor process did not start (exit {self.proc.returncode}); see its stderr")
        self.port, self.url = int(parts[1]), parts[2]

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


if __name__ == "__main__":
    main()
