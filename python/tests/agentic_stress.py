#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agentic multi-turn stress test for a RadixShmem cluster on one host.

launcher (this process)
  |- etcd                private single-node on a free port (nodes > 1)
  |- worker[0..N)        re-exec'd: RadixServer(rank) + RadixClient + TCP server
  `- generator           asyncio, --concurrency session coroutines

Each session is a multi-turn conversation whose hashes are drawn once; every
turn appends isl_step input and osl output blocks and is sent to a random
worker, so most turns cross nodes. Theoretical hit = history / input.

Data mode (default): every node owns a small SlotStore (128 B FULL, 64 B SWA,
256 B Mamba slots) and a request runs pull_async -> wait -> verify the bytes ->
generate the new blocks' bytes -> think -> allocate, write, insert. Block
bytes are a function of the block hash h: FULL [h, h+1, ...], SWA [~h, ~h+1,
...], Mamba [h, h-1, ...] as uint64 words, so a reader can check what it got.
--index-only keeps the old index-only flow: query -> think -> insert.

    python tests/agentic_stress.py                      # defaults: 8 nodes, 128 sessions, 60 s
    python tests/agentic_stress.py --nodes 2 --concurrency 16 --duration 20 --index-only
"""

import argparse
import asyncio
import glob
import json
import os
import queue
import random
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

import numpy as np

try:
    import radixshmem
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "python"))
    import radixshmem

MSG_REQ, MSG_DONE, MSG_STOP, MSG_STATS = 1, 2, 3, 4


# ======================================================================
# Config
# ======================================================================

@dataclass
class Config:
    nodes: int = 8
    concurrency: int = 128
    duration: float = 60.0
    drain_timeout: float = 0.0
    warmup_sec: float = 10.0
    turns: List[int] = field(default_factory=lambda: [50, 100])
    isl_step: List[int] = field(default_factory=lambda: [1000, 2000])
    osl: List[int] = field(default_factory=lambda: [100, 500])
    block_size: int = 16
    think_ms: float = 5.0
    aux: str = "swa"                  # swa | mamba | none
    swa_window: int = 1
    max_blocks: int = 10_000_000
    max_nodes: int = 1_000_000
    swa_max_blocks: int = 1_000_000
    mamba_max_slots: int = 1_000_000
    register_chunk_size: int = 1024
    data_pool_ratio: float = 4.0
    index_only: bool = False          # False: the server owns a SlotStore, requests move bytes
    full_slot_bytes: int = 128
    swa_slot_bytes: int = 64
    mamba_slot_bytes: int = 256
    executors_per_worker: int = 1
    worker_buffer: int = 4096
    rdma_devs: str = ""               # comma list; empty = auto from /sys
    gid_idx: int = 3
    base_port: int = 19000
    sticky: float = 0.0
    seed: int = 0
    json_out: str = ""
    workdir: str = ""
    report_sec: float = 10.0
    bootstrap_timeout: int = 120
    diag_misses: bool = False         # log every actual < history request
    slow_us: float = 0.0              # log every request slower than this (query or insert)
    rht_slots: int = 4                # RHT slots per bucket (W)
    flush_after_insert: bool = False  # drain async RHT writes before DONE
    writeback_delay_ms: float = -1.0  # >= 0: background eviction writes back, done after this delay
    # internal (filled by the launcher)
    run_id: str = ""
    etcd_port: int = 0


def parse_range(s: str) -> List[int]:
    a, b = (int(x) for x in s.split(","))
    return [a, b]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    p.add_argument("--nodes", type=int, default=d.nodes)
    p.add_argument("--concurrency", type=int, default=d.concurrency)
    p.add_argument("--duration", type=float, default=d.duration,
                   help="seconds after warmup; then no new sessions start")
    p.add_argument("--drain-timeout", type=float, default=d.drain_timeout,
                   help="0 = wait for every in-flight session to finish")
    p.add_argument("--warmup-sec", type=float, default=d.warmup_sec)
    p.add_argument("--turns", type=parse_range, default=d.turns, metavar="LO,HI")
    p.add_argument("--isl-step", type=parse_range, default=d.isl_step, metavar="LO,HI")
    p.add_argument("--osl", type=parse_range, default=d.osl, metavar="LO,HI")
    p.add_argument("--block-size", type=int, default=d.block_size)
    p.add_argument("--think-ms", type=float, default=d.think_ms)
    p.add_argument("--aux", choices=["swa", "mamba", "none"], default=d.aux)
    p.add_argument("--swa-window", type=int, default=d.swa_window)
    p.add_argument("--max-blocks", type=int, default=d.max_blocks)
    p.add_argument("--max-nodes", type=int, default=d.max_nodes)
    p.add_argument("--swa-max-blocks", type=int, default=d.swa_max_blocks)
    p.add_argument("--mamba-max-slots", type=int, default=d.mamba_max_slots)
    p.add_argument("--register-chunk-size", type=int, default=d.register_chunk_size)
    p.add_argument("--data-pool-ratio", type=float, default=d.data_pool_ratio)
    p.add_argument("--index-only", action="store_true", default=d.index_only,
                   help="no SlotStore: query -> think -> insert, nothing moves bytes")
    p.add_argument("--full-slot-bytes", type=int, default=d.full_slot_bytes)
    p.add_argument("--swa-slot-bytes", type=int, default=d.swa_slot_bytes)
    p.add_argument("--mamba-slot-bytes", type=int, default=d.mamba_slot_bytes)
    p.add_argument("--executors-per-worker", type=int, default=d.executors_per_worker)
    p.add_argument("--worker-buffer", type=int, default=d.worker_buffer)
    p.add_argument("--rdma-devs", default=d.rdma_devs)
    p.add_argument("--gid-idx", type=int, default=d.gid_idx)
    p.add_argument("--base-port", type=int, default=d.base_port)
    p.add_argument("--sticky", type=float, default=d.sticky)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--json", dest="json_out", default=d.json_out)
    p.add_argument("--workdir", default=d.workdir)
    p.add_argument("--report-sec", type=float, default=d.report_sec)
    p.add_argument("--bootstrap-timeout", type=int, default=d.bootstrap_timeout)
    p.add_argument("--slow-us", type=float, default=d.slow_us,
                   help="log requests whose query or insert exceeds this many us")
    p.add_argument("--diag-misses", action="store_true", default=d.diag_misses,
                   help="re-probe and log requests whose hit trails history")
    p.add_argument("--rht-slots", type=int, default=d.rht_slots,
                   choices=[1, 2, 4, 8], help="RHT slots per bucket")
    p.add_argument("--flush-after-insert", action="store_true",
                   default=d.flush_after_insert,
                   help="client.flush() after insert so RHT routes land before DONE")
    # internal
    p.add_argument("--writeback-delay-ms", type=float, default=d.writeback_delay_ms,
                   help=">= 0 enables write-back with a handler that stores nothing and calls "
                        "done after this delay (default: off)")
    p.add_argument("--role", choices=["launcher", "worker"], default="launcher")
    p.add_argument("--rank", type=int, default=-1)
    p.add_argument("--config-json", default="")
    a = p.parse_args()
    if a.config_json:
        cfg = Config(**json.loads(a.config_json))
    else:
        cfg = Config(**{k: getattr(a, k) for k in Config.__dataclass_fields__
                        if hasattr(a, k)})
    return cfg, a.role, a.rank


def query_mask(cfg: Config) -> int:
    m = radixshmem.COMPONENT_MASK_FULL
    if cfg.aux == "swa":
        m |= radixshmem.COMPONENT_MASK_SWA
    elif cfg.aux == "mamba":
        m |= radixshmem.COMPONENT_MASK_MAMBA
    return m


def server_config(cfg: Config, rank: int) -> radixshmem.ServerConfig:
    """One node; a cluster when nodes > 1. Per-rank names keep the shm regions
    and the gRPC sockets of one host apart. Data mode: the budget is exactly
    the index pools' bytes and the ratios hand the aux pools their share, so
    the server plans the same counts the index-only run configures directly;
    slot_align 64 keeps the small strides small."""
    swa, mamba = cfg.aux == "swa", cfg.aux == "mamba"
    swa_slots, mamba_slots = cfg.swa_max_blocks if swa else 0, cfg.mamba_max_slots if mamba else 0
    swa_bytes, mamba_bytes = swa_slots * cfg.swa_slot_bytes, mamba_slots * cfg.mamba_slot_bytes
    data_bytes = 0 if cfg.index_only else cfg.max_blocks * cfg.full_slot_bytes + swa_bytes + mamba_bytes
    devs = rdma_devices(cfg)
    dev = [devs[rank % len(devs)]] if devs else []
    cluster = radixshmem.ClusterConfig()
    if cfg.nodes > 1:
        cluster = radixshmem.ClusterConfig(
            expected_min_nodes=cfg.nodes, registry=f"etcd://127.0.0.1:{cfg.etcd_port}",
            cluster_id=f"agentic_{cfg.run_id}", node_name=f"node{rank}", rpc_address="0.0.0.0",
            num_rht_shards=cfg.nodes, rht_shard_holders=list(range(cfg.nodes)),
            index_dev=dev[0] if dev else "", gid_idx=cfg.gid_idx,
            bootstrap_timeout_sec=cfg.bootstrap_timeout, rht_slots_per_bucket=cfg.rht_slots,
            enable_remote_query=cfg.diag_misses)
    return radixshmem.ServerConfig(
        name=f"/agentic_{cfg.run_id}_r{rank}", data_bytes=data_bytes,
        swa_ratio=swa_bytes / data_bytes if data_bytes else 0.0,
        mamba_ratio=mamba_bytes / data_bytes if data_bytes else 0.0,
        slot_align=64, transfer_devices=dev, max_nodes=cfg.max_nodes,
        data_pool_ratio=cfg.data_pool_ratio, writeback=cfg.writeback_delay_ms >= 0, cluster=cluster)


class DelayedWriteback(radixshmem.WritebackHandler):
    """Writes nothing: hands the tasks back after a fixed delay."""

    def __init__(self, delay_ms: float):
        self.delay = delay_ms / 1000

    def write(self, tasks, done):
        ids = [t.task_id for t in tasks]
        if self.delay <= 0:
            done(ids)
            return
        t = threading.Timer(self.delay, done, args=(ids,))
        t.daemon = True
        t.start()


def geometry(cfg: Config) -> radixshmem.Geometry:
    """What the engine would send: bytes per slot in data mode, slot counts
    in index-only mode (there the slots are the caller's own storage)."""
    swa, mamba = cfg.aux == "swa", cfg.aux == "mamba"
    if cfg.index_only:
        return radixshmem.Geometry(
            block_size=cfg.block_size, full_slots=cfg.max_blocks,
            swa_slots=cfg.swa_max_blocks if swa else 0, swa_window_blocks=cfg.swa_window if swa else 0,
            mamba_slots=cfg.mamba_max_slots if mamba else 0,
            register_chunk_tokens=cfg.register_chunk_size * cfg.block_size)
    return radixshmem.Geometry(
        block_size=cfg.block_size, full_slot_bytes=cfg.full_slot_bytes,
        swa_slot_bytes=cfg.swa_slot_bytes if swa else 0, swa_window_blocks=cfg.swa_window if swa else 0,
        mamba_slot_bytes=cfg.mamba_slot_bytes if mamba else 0, slot_align=64,
        register_chunk_tokens=cfg.register_chunk_size * cfg.block_size)


FULL, SWA, MAMBA = radixshmem.ComponentType.FULL, radixshmem.ComponentType.SWA, radixshmem.ComponentType.MAMBA


def kv_pattern(kind, hashes: np.ndarray, words: int) -> np.ndarray:
    """(k, words) uint64 block bytes: FULL counts up from the hash, SWA from
    its complement, Mamba counts down."""
    h = np.asarray(hashes, dtype=np.uint64)
    if kind == SWA:
        h = ~h
    step = np.arange(words, dtype=np.uint64)
    return h[:, None] - step if kind == MAMBA else h[:, None] + step


def rdma_devices(cfg: Config) -> List[str]:
    if cfg.rdma_devs:
        return [d for d in cfg.rdma_devs.split(",") if d]
    devs = sorted(os.path.basename(p) for p in glob.glob("/sys/class/infiniband/mlx5_*")
                  if "bond" not in p)
    return devs


# ======================================================================
# Framing: !III (type, header_len, payload_len) | json header | payload
# ======================================================================

def send_msg(sock: socket.socket, msg_type: int, header: dict,
             payload: bytes = b""):
    hdr = json.dumps(header, separators=(",", ":")).encode()
    sock.sendall(struct.pack("!III", msg_type, len(hdr), len(payload)) + hdr + payload)


def recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def recv_msg(sock: socket.socket):
    fh = recv_exact(sock, 12)
    if fh is None:
        return None, None, b""
    t, hl, pl = struct.unpack("!III", fh)
    hdr = recv_exact(sock, hl)
    if hdr is None:
        return None, None, b""
    payload = recv_exact(sock, pl) if pl else b""
    if payload is None:
        return None, None, b""
    return t, json.loads(hdr), payload


# ======================================================================
# Worker
# ======================================================================

def create_node(cfg: Config, rank: int):
    """The engine's bootstrap: start the node's server, build the client with
    the geometry (handed to the server at once), wait for the cluster."""
    sc = server_config(cfg, rank)
    handler = DelayedWriteback(cfg.writeback_delay_ms) if cfg.writeback_delay_ms >= 0 else None
    server = radixshmem.RadixServer(sc, writeback=handler).start()
    client = radixshmem.RadixClient(sc.name, geometry(cfg))
    client.wait_ready(cfg.bootstrap_timeout + 120)
    return server, client


class Worker:
    def __init__(self, cfg: Config, rank: int, client, server):
        self.cfg = cfg
        self.rank = rank
        self.client = client
        self.server = server
        self.own = int(client.rank()) if client.is_distributed() else 0
        self.mask = query_mask(cfg)
        self.think = cfg.think_ms / 1000.0
        self.q: "queue.Queue" = queue.Queue(maxsize=cfg.worker_buffer)
        self.out: "queue.Queue" = queue.Queue()
        self.peak_depth = 0
        self.errors: Dict[str, int] = {}
        self.bad_blocks = 0
        self.lock = threading.Lock()
        # data mode: one uint64 view per pool, (num_slots, stride / 8)
        self.pools: Dict[object, np.ndarray] = {}
        self.words = {FULL: cfg.full_slot_bytes // 8, SWA: cfg.swa_slot_bytes // 8,
                      MAMBA: cfg.mamba_slot_bytes // 8}
        if not cfg.index_only:
            st = client.store
            for kind in (FULL, SWA, MAMBA):
                if st.has_pool(kind):
                    p = st.pool(kind)
                    self.pools[kind] = np.frombuffer(st.pool_view(kind), dtype=np.uint64).reshape(
                        int(p.num_slots), int(p.slot_bytes) // 8)

    def local_full_hit(self, hashes: np.ndarray) -> int:
        return int(self.client.query(hashes, mask=radixshmem.COMPONENT_MASK_FULL, local_only=True,
                                     lock=False, update_meta=False).common_hit)

    def verify(self, r, inp: np.ndarray) -> int:
        """Blocks of a PullResult whose bytes do not follow kv_pattern."""
        n, bad = int(r.common_hit), 0
        if n and len(r.full_slots):
            w = self.words[FULL]
            bad += int(np.count_nonzero(
                (self.pools[FULL][r.full_slots, :w] != kv_pattern(FULL, inp[:n], w)).any(axis=1)))
        if len(r.swa_slots):
            w = self.words[SWA]
            bad += int(np.count_nonzero(
                (self.pools[SWA][r.swa_slots, :w] != kv_pattern(SWA, inp[r.swa_start:n], w)).any(axis=1)))
        if r.mamba_slot >= 0:
            w = self.words[MAMBA]
            bad += int((self.pools[MAMBA][r.mamba_slot, :w] != kv_pattern(MAMBA, inp[n - 1:n], w)[0]).any())
        return bad

    def execute(self, hdr: dict, payload: bytes) -> dict:
        c = self.client
        hashes = np.frombuffer(payload, dtype=np.uint64)
        inp = hashes[:hdr["input_blocks"]]
        n = n0 = len(hashes)
        verify_us, xfer_us, remote_bytes, err, gen, have0 = 0.0, 0.0, 0, "", {}, 0

        t0 = time.perf_counter()
        if self.cfg.index_only:
            qr = c.query(inp, mask=self.mask, lock=True, update_meta=True)
            query_us = (time.perf_counter() - t0) * 1e6
            time.sleep(0.001)
            t_f = time.perf_counter()
            qr.finalize()
            finalize_us = (time.perf_counter() - t_f) * 1e6
            actual = int(qr.common_hit)
            local = sum(len(f[2]) for f in qr.full_fragments if f[0] == self.own)
            rdma_reads, rdma_atomics = int(qr.rdma_read_count), int(qr.rdma_atomic_count)
        else:
            job = c.pull_async(inp, mask=self.mask, lock=True)
            t_x = time.perf_counter()
            r = job.wait()
            xfer_us = (time.perf_counter() - t_x) * 1e6
            query_us = (time.perf_counter() - t0) * 1e6
            remote_bytes = int(r.remote_bytes)
            t_v = time.perf_counter()
            bad = self.verify(r, inp)
            verify_us = (time.perf_counter() - t_v) * 1e6
            t_f = time.perf_counter()
            r.finalize()
            finalize_us = (time.perf_counter() - t_f) * 1e6
            actual, local = int(r.common_hit), int(r.common_hit) - int(r.remote_blocks)
            rdma_reads, rdma_atomics = int(job._q.rdma_read_count), int(job._q.rdma_atomic_count)
            if bad:
                err = "VERIFY_MISMATCH"
                with self.lock:
                    self.bad_blocks += bad
            have0 = self.local_full_hit(hashes)
            gen[FULL] = kv_pattern(FULL, hashes[have0:], self.words[FULL])
            if self.cfg.aux == "swa":
                gen[SWA] = kv_pattern(SWA, hashes[n - min(self.cfg.swa_window, n):], self.words[SWA])
            elif self.cfg.aux == "mamba":
                gen[MAMBA] = kv_pattern(MAMBA, hashes[n - 1:], self.words[MAMBA])

        diag = None
        swa_src = int(qr.swa_source_rank) if self.cfg.index_only else int(r.source_rank)
        if self.cfg.diag_misses and actual < hdr["history_blocks"]:
            lo = c.query(inp, mask=self.mask, local_only=True, lock=False,
                         update_meta=False)
            time.sleep(0.002)
            rq = c.query(inp, mask=self.mask, lock=False, update_meta=False)
            fo = c.query(inp, mask=radixshmem.COMPONENT_MASK_FULL, lock=False,
                         update_meta=False)
            peer_hits = {}
            if c.is_distributed():
                for r in range(c.world_size()):
                    if r == self.own:
                        continue
                    try:   # remote_query caps requests at 512 blocks
                        peer_hits[r] = int(c.remote_query(r, inp[:512], lock=False,
                                                          update_meta=False).total_hit_length)
                    except Exception as e:
                        peer_hits[r] = f"err:{type(e).__name__}"
            cs = self.cfg.register_chunk_size
            diag = {"h0": "%08x" % (int(hashes[0]) & 0xffffffff),
                    "hk": {p: "%08x" % (int(hashes[p]) & 0xffffffff)
                           for p in range(0, len(inp), cs)},
                    "peer_hits": peer_hits,
                    "local_only_hit": int(lo.common_hit),
                    "requery_hit": int(rq.common_hit),
                    "requery_rdma_reads": int(rq.rdma_read_count),
                    "full_only_hit": int(fo.common_hit),
                    "full_only_sources": sorted({int(f[0]) for f in fo.full_fragments}),
                    "swa_source_rank": swa_src}

        if self.think > 0:
            time.sleep(self.think)

        t1 = time.perf_counter()
        have = self.local_full_hit(hashes)
        need = n - have
        if need > 0:
            slots = c.allocate_slots(need)
            n = have + len(slots)
            if len(slots) < need:
                err = err or "FULL_POOL_SHORT"
            if len(slots) > 0:
                if gen:
                    rows = (gen[FULL][have - have0:have - have0 + len(slots)] if have >= have0
                            else kv_pattern(FULL, hashes[have:n], self.words[FULL]))
                    self.pools[FULL][slots, :self.words[FULL]] = rows
                r = c.insert(hashes[:n], slots, start=have)
                if r.error != radixshmem.InsertError.OK:
                    err = err or r.error.name
        insert_full_us = (time.perf_counter() - t1) * 1e6

        if n > 0 and self.cfg.aux == "swa":
            k = min(self.cfg.swa_window, n)
            s = c.allocate_slots(k, SWA)
            if len(s) == k:
                if gen:
                    self.pools[SWA][s, :self.words[SWA]] = (
                        gen[SWA] if n == n0 else kv_pattern(SWA, hashes[n - k:n], self.words[SWA]))
                r = c.insert(hashes[:n], s, start=0, component=SWA)
                if r.error != radixshmem.InsertError.OK:
                    err = err or "SWA_" + r.error.name
            else:
                err = err or "SWA_POOL"
        elif n > 0 and self.cfg.aux == "mamba":
            s = c.allocate_slots(1, MAMBA)
            if len(s) == 1:
                if gen:
                    self.pools[MAMBA][s, :self.words[MAMBA]] = (
                        gen[MAMBA] if n == n0 else kv_pattern(MAMBA, hashes[n - 1:n], self.words[MAMBA]))
                r = c.insert(hashes[:n], s, component=MAMBA)
                if r.error != radixshmem.InsertError.OK:
                    err = err or "MAMBA_" + r.error.name
            else:
                err = err or "MAMBA_POOL"
        if self.cfg.flush_after_insert:
            c.flush()
        insert_us = (time.perf_counter() - t1) * 1e6

        if err:
            with self.lock:
                self.errors[err] = self.errors.get(err, 0) + 1
        return {
            "req_id": hdr["req_id"], "worker": self.rank, "own_rank": self.own,
            "query_us": query_us, "insert_us": insert_us,
            "insert_full_us": insert_full_us, "verify_us": verify_us,
            "xfer_us": xfer_us, "remote_bytes": remote_bytes,
            "actual_hit": actual, "local_hit": int(local),
            "remote_hit": actual - int(local),
            "rdma_reads": rdma_reads, "rdma_atomics": rdma_atomics,
            "inserted_blocks": max(0, n - have), "finalize_us": finalize_us, "error": err,
            "diag": diag,
        }

    def recv_loop(self, sock: socket.socket):
        while True:
            t, hdr, payload = recv_msg(sock)
            if t is None or t == MSG_STOP:
                break
            self.q.put((hdr, payload))
        for _ in range(self.cfg.executors_per_worker):
            self.q.put(None)

    def exec_loop(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            self.peak_depth = max(self.peak_depth, self.q.qsize() + 1)
            hdr, payload = item
            try:
                done = self.execute(hdr, payload)
            except Exception as e:  # keep the pipeline alive; report the fault
                done = {"req_id": hdr["req_id"], "worker": self.rank,
                        "query_us": 0.0, "insert_us": 0.0, "insert_full_us": 0.0, "verify_us": 0.0,
                        "xfer_us": 0.0, "remote_bytes": 0,
                        "actual_hit": 0, "local_hit": 0, "remote_hit": 0,
                        "rdma_reads": 0, "rdma_atomics": 0, "inserted_blocks": 0, "finalize_us": 0.0,
                        "error": f"EXC:{type(e).__name__}:{e}"[:120]}
            self.out.put(done)

    def send_loop(self, sock: socket.socket):
        while True:
            item = self.out.get()
            if item is None:
                break
            send_msg(sock, MSG_DONE, item)

    def drain_writeback(self, timeout: float = 30.0) -> Optional[dict]:
        if self.cfg.writeback_delay_ms < 0:
            return None
        deadline, quiet = time.monotonic() + timeout, 0
        while time.monotonic() < deadline and quiet < 3:
            wb = self.server.stats()["writeback"]
            quiet = quiet + 1 if wb["queued"] == 0 and wb["popped"] == 0 else 0
            time.sleep(0.1)
        return self.server.stats()["writeback"]

    def stats(self) -> dict:
        c = self.client
        wb = self.drain_writeback()   # held slots would show as leaks in the audit
        return {
            "worker": self.rank, "own_rank": self.own,
            "total_blocks": int(c.total_blocks()),
            "total_radix_nodes": int(c.total_radix_nodes()),
            "mempool_total": int(c.mempool_total()),
            "mempool_used": int(c.mempool_used()),
            "data_pool_utilization": float(c.data_pool_utilization()),
            "peak_queue_depth": self.peak_depth,
            "errors": self.errors,
            "bad_blocks": self.bad_blocks,
            "transfer_failures": c.transfer_failures,
            "staging_shortfalls": c.staging_shortfalls,
            "promotion_shortfalls": c.promotion_shortfalls,
            # end-of-run leak audit (locks / slots / pins); 0 == clean
            "health": int(c.check_health()),
            "health_report": c.health_report().describe().strip(),
            "writeback": wb,
        }


def run_worker(cfg: Config, rank: int) -> int:
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print(f"[W{rank}] creating RadixServer (pid={os.getpid()})")
    server, client = create_node(cfg, rank)
    print(f"[W{rank}] ready: shm={client.info.index_name} rank={client.rank()} "
          f"world={client.world_size()}")

    ls = socket.socket()
    ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind(("127.0.0.1", cfg.base_port + rank))
    ls.listen(1)
    print(f"READY {cfg.base_port + rank}")
    conn, _ = ls.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    w = Worker(cfg, rank, client, server)
    threads = [threading.Thread(target=w.recv_loop, args=(conn,), daemon=True),
               threading.Thread(target=w.send_loop, args=(conn,), daemon=True)]
    execs = [threading.Thread(target=w.exec_loop, daemon=True)
             for _ in range(cfg.executors_per_worker)]
    for t in threads + execs:
        t.start()
    threads[0].join()
    for t in execs:
        t.join()
    w.out.put(None)
    threads[1].join()
    send_msg(conn, MSG_STATS, w.stats())
    conn.close()
    print(f"[W{rank}] done, errors={w.errors}")
    client.close()
    server.close()
    return 0


# ======================================================================
# Generator
# ======================================================================

@dataclass
class Session:
    sid: int
    turns: int
    isl_blocks: List[int]
    osl_blocks: List[int]
    hashes: np.ndarray


def make_session(cfg: Config, sid: int, rng: random.Random) -> Session:
    turns = rng.randint(cfg.turns[0], cfg.turns[1])
    isl = rng.randint(cfg.isl_step[0], cfg.isl_step[1]) // cfg.block_size
    osl = rng.randint(cfg.osl[0], cfg.osl[1]) // cfg.block_size
    total = turns * (isl + osl)
    nrng = np.random.default_rng((cfg.seed << 32) ^ sid ^ 0x9E3779B97F4A7C15)
    hashes = nrng.integers(1, 2**63 - 1, size=total, dtype=np.uint64)
    return Session(sid, turns, [isl] * turns, [osl] * turns, hashes)


class WorkerConn:
    def __init__(self, idx: int, reader, writer):
        self.idx = idx
        self.reader = reader
        self.writer = writer
        self.lock = asyncio.Lock()
        self.futures: Dict[int, asyncio.Future] = {}
        self.stats: Optional[dict] = None
        self.closed = asyncio.Event()

    async def send(self, msg_type: int, hdr: dict, payload: bytes = b""):
        h = json.dumps(hdr, separators=(",", ":")).encode()
        async with self.lock:
            self.writer.write(struct.pack("!III", msg_type, len(h), len(payload)))
            self.writer.write(h)
            if payload:
                self.writer.write(payload)
            await self.writer.drain()

    async def read_loop(self):
        try:
            while True:
                fh = await self.reader.readexactly(12)
                t, hl, pl = struct.unpack("!III", fh)
                hdr = json.loads(await self.reader.readexactly(hl))
                if pl:
                    await self.reader.readexactly(pl)
                if t == MSG_DONE:
                    fut = self.futures.pop(hdr["req_id"], None)
                    if fut is not None and not fut.done():
                        fut.set_result(hdr)
                elif t == MSG_STATS:
                    self.stats = hdr
                    break
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            for fut in self.futures.values():
                if not fut.done():
                    fut.set_exception(ConnectionError(f"worker {self.idx} gone"))
            self.futures.clear()
            self.closed.set()


class Recorder:
    FIELDS = ["t", "worker", "turn", "input_blocks", "history_blocks",
              "actual_hit", "local_hit", "remote_hit", "query_us", "insert_us",
              "insert_full_us", "rdma_reads", "rdma_atomics", "inserted_blocks", "verify_us",
              "xfer_us", "remote_bytes"]

    def __init__(self):
        self.rows: List[tuple] = []
        self.errors: Dict[str, int] = {}
        self.sessions_done = 0
        self.sessions_truncated = 0
        self.t_start = time.time()
        self.t_steady = 0.0
        self.t_drain = 0.0
        self.t_end = 0.0
        self.win = []   # (t, actual, hist, query_us, insert_us) for progress
        self.miss_log = None
        self.slow_log = None

    def record(self, req: dict, done: dict):
        now = time.time()
        self.rows.append((now, done["worker"], req["turn"], req["input_blocks"],
                          req["history_blocks"], done["actual_hit"],
                          done["local_hit"], done["remote_hit"], done["query_us"],
                          done["insert_us"], done["insert_full_us"],
                          done["rdma_reads"], done["rdma_atomics"],
                          done["inserted_blocks"], done.get("verify_us", 0.0),
                          done.get("xfer_us", 0.0), done.get("remote_bytes", 0)))
        if done["error"]:
            self.errors[done["error"]] = self.errors.get(done["error"], 0) + 1
        if self.miss_log is not None and done["actual_hit"] < req["history_blocks"]:
            self.miss_log.write(json.dumps({
                "t": round(now - self.t_start, 3), "phase": self.phase(now),
                **{k: req[k] for k in ("session_id", "turn", "history_blocks",
                                       "input_blocks")},
                "visits": req["visits"], "worker": done["worker"],
                "own_rank": done.get("own_rank"),
                **{k: done[k] for k in ("actual_hit", "local_hit", "remote_hit",
                                        "rdma_reads", "error")},
                "diag": done.get("diag")}) + "\n")
            self.miss_log.flush()
        if self.slow_log is not None and max(done["query_us"], done["insert_us"]) > self.slow_us:
            self.slow_log.write(json.dumps({
                "t": round(now - self.t_start, 3), "ts": round(now, 3), "phase": self.phase(now),
                **{k: req[k] for k in ("session_id", "turn", "history_blocks",
                                       "input_blocks")},
                "worker": done["worker"],
                **{k: (round(done[k]) if k.endswith("_us") else done[k])
                   for k in ("query_us", "insert_us", "insert_full_us", "finalize_us", "actual_hit",
                             "local_hit", "remote_hit", "rdma_reads", "rdma_atomics",
                             "inserted_blocks")}}) + "\n")
            self.slow_log.flush()
        self.win.append((now, done["actual_hit"], req["history_blocks"],
                         done["query_us"], done["insert_us"], done.get("remote_bytes", 0)))

    def phase(self, now: float) -> str:
        if self.t_drain and now >= self.t_drain:
            return "drain"
        return "steady" if self.t_steady and now >= self.t_steady else "warmup"

    def progress_line(self, now: float, in_flight: int) -> str:
        w = [x for x in self.win if x[0] >= now - 10]
        self.win = w
        if not w:
            return f"[{now - self.t_start:6.1f}s] idle, in-flight={in_flight}"
        act = sum(x[1] for x in w)
        th = sum(x[2] for x in w)
        q99 = np.percentile([x[3] for x in w], 99)
        i99 = np.percentile([x[4] for x in w], 99)
        mbps = sum(x[5] for x in w) / 10 / 1e6
        return (f"[{now - self.t_start:6.1f}s] {len(w) / 10:7.1f} req/s  "
                f"hit/theory={act / max(th, 1):.3f}  query p99={q99:8.0f}us  "
                f"insert p99={i99:8.0f}us  " + (f"xfer={mbps:7.2f}MB/s  " if mbps else "")
                + f"sessions={self.sessions_done}  in-flight={in_flight}")


async def session_loop(cfg: Config, conns: List[WorkerConn], rec: Recorder,
                       rng: random.Random, stop_new: asyncio.Event,
                       drain_deadline: List[float], next_sid: List[int],
                       in_flight: List[int]):
    while not stop_new.is_set():
        sid = next_sid[0]
        next_sid[0] += 1
        sess = make_session(cfg, sid, rng)
        hist = 0
        prev_w = -1
        visits: List[int] = []
        truncated = False
        for turn in range(sess.turns):
            if drain_deadline[0] and time.time() > drain_deadline[0]:
                truncated = True
                break
            inp = hist + sess.isl_blocks[turn]
            out = sess.osl_blocks[turn]
            if prev_w >= 0 and rng.random() < cfg.sticky:
                w = prev_w
            else:
                w = rng.randrange(len(conns))
            conn = conns[w]
            req = {"req_id": (sid << 20) | turn, "session_id": sid, "turn": turn,
                   "history_blocks": hist, "input_blocks": inp,
                   "output_blocks": out, "visits": list(visits)}
            visits.append(w)
            fut = asyncio.get_running_loop().create_future()
            conn.futures[req["req_id"]] = fut
            in_flight[0] += 1
            try:
                await conn.send(MSG_REQ, req, sess.hashes[:inp + out].tobytes())
                done = await fut
            finally:
                in_flight[0] -= 1
            rec.record(req, done)
            hist = inp + out
            prev_w = w
        if truncated:
            rec.sessions_truncated += 1
        else:
            rec.sessions_done += 1


async def run_generator(cfg: Config) -> dict:
    conns: List[WorkerConn] = []
    for i in range(cfg.nodes):
        r, w = await asyncio.open_connection("127.0.0.1", cfg.base_port + i)
        w.get_extra_info("socket").setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conns.append(WorkerConn(i, r, w))
    readers = [asyncio.create_task(c.read_loop()) for c in conns]

    rec = Recorder()
    if cfg.diag_misses:
        rec.miss_log = open(os.path.join(cfg.workdir, "misses.jsonl"), "w")
    if cfg.slow_us > 0:
        rec.slow_us = cfg.slow_us
        rec.slow_log = open(os.path.join(cfg.workdir, "slow.jsonl"), "w")
    rng = random.Random(cfg.seed)
    stop_new = asyncio.Event()
    drain_deadline = [0.0]
    next_sid = [0]
    in_flight = [0]
    tasks = [asyncio.create_task(session_loop(cfg, conns, rec, rng, stop_new,
                                              drain_deadline, next_sid, in_flight))
             for _ in range(cfg.concurrency)]

    async def clock():
        await asyncio.sleep(cfg.warmup_sec)
        rec.t_steady = time.time()
        print(f"--- warmup done, measuring {cfg.duration:.0f}s ---", flush=True)
        await asyncio.sleep(cfg.duration)
        rec.t_drain = time.time()
        stop_new.set()
        if cfg.drain_timeout > 0:
            drain_deadline[0] = rec.t_drain + cfg.drain_timeout
        print(f"--- duration reached, draining {in_flight[0]} in-flight "
              f"sessions ---", flush=True)

    async def progress():
        while True:
            await asyncio.sleep(cfg.report_sec)
            print(rec.progress_line(time.time(), in_flight[0]), flush=True)

    clock_task = asyncio.create_task(clock())
    prog_task = asyncio.create_task(progress())
    try:
        await asyncio.gather(*tasks)
    finally:
        prog_task.cancel()
        clock_task.cancel()
        rec.t_end = time.time()
        for c in conns:
            try:
                await c.send(MSG_STOP, {})
            except Exception:
                pass
        await asyncio.wait([asyncio.ensure_future(c.closed.wait()) for c in conns],
                           timeout=60)
        for c in conns:
            c.writer.close()
        for t in readers:
            t.cancel()
    return build_report(cfg, rec, [c.stats for c in conns])


# ======================================================================
# Report
# ======================================================================

def pct(a: np.ndarray, ps=(50, 90, 99, 99.9)) -> dict:
    if a.size == 0:
        return {}
    d = {f"p{p:g}": float(np.percentile(a, p)) for p in ps}
    d["max"] = float(a.max())
    d["mean"] = float(a.mean())
    return d


def hit_block(rows: np.ndarray) -> dict:
    inp = rows[:, 3].sum()
    hist = rows[:, 4].sum()
    act = rows[:, 5].sum()
    loc = rows[:, 6].sum()
    rem = rows[:, 7].sum()
    ratio = rows[:, 5] / np.maximum(rows[:, 4], 1)
    ratio = ratio[rows[:, 4] > 0]
    return {
        "requests": int(rows.shape[0]),
        "actual_hit_rate": float(act / max(inp, 1)),
        "theoretical_hit_rate": float(hist / max(inp, 1)),
        "actual_over_theoretical": float(act / max(hist, 1)),
        "local_share": float(loc / max(act, 1)),
        "remote_share": float(rem / max(act, 1)),
        "ratio_p50": float(np.percentile(ratio, 50)) if ratio.size else 0.0,
        "ratio_p10": float(np.percentile(ratio, 10)) if ratio.size else 0.0,
        "ratio_min": float(ratio.min()) if ratio.size else 0.0,
        "rdma_reads_mean": float(rows[:, 11].mean()),
        "rdma_atomics_mean": float(rows[:, 12].mean()),
    }


def build_report(cfg: Config, rec: Recorder, stats: List[Optional[dict]]) -> dict:
    all_rows = np.array(rec.rows, dtype=np.float64) if rec.rows else np.zeros((0, 17))
    steady = all_rows[all_rows[:, 0] >= rec.t_steady] if rec.t_steady else all_rows
    drain = steady[steady[:, 0] >= rec.t_drain] if rec.t_drain else steady[:0]
    span = max(rec.t_end - (rec.t_steady or rec.t_start), 1e-9)
    rep = {
        "config": asdict(cfg),
        "timing": {
            "warmup_sec": (rec.t_steady - rec.t_start) if rec.t_steady else 0.0,
            "measured_sec": span,
            "drain_sec": (rec.t_end - rec.t_drain) if rec.t_drain else 0.0,
            "drain_requests": int(drain.shape[0]),
        },
        "throughput": {
            "req_per_sec": steady.shape[0] / span,
            "blocks_per_sec": float(steady[:, 3].sum() / span) if steady.size else 0.0,
            "sessions_done": rec.sessions_done,
            "sessions_truncated": rec.sessions_truncated,
        },
        "hit": hit_block(steady) if steady.size else {},
        "hit_by_worker": {},
        "latency_us": {},
        "latency_by_worker": {},
        "latency_by_input_blocks": {},
        "errors": rec.errors,
        "workers": [s for s in stats if s],
    }
    if steady.size:
        rep["latency_us"] = {"query": pct(steady[:, 8]), "insert": pct(steady[:, 9]),
                             "insert_full": pct(steady[:, 10])}
        if not cfg.index_only:
            rep["latency_us"]["verify"] = pct(steady[:, 14])
            rb, xs = steady[:, 16], steady[:, 15] / 1e6
            remote = rb > 0
            rep["transfer"] = {
                "remote_bytes": float(rb.sum()), "remote_requests": int(remote.sum()),
                "aggregate_mbps": float(rb.sum() / span / 1e6),
                "bandwidth_mbps": pct(rb[remote] / np.maximum(xs[remote], 1e-9) / 1e6) if remote.any() else {},
                "xfer_us": pct(steady[remote, 15]) if remote.any() else {},
            }
        for w in range(cfg.nodes):
            r = steady[steady[:, 1] == w]
            if r.size:
                rep["hit_by_worker"][w] = hit_block(r)
                rep["latency_by_worker"][w] = {"query": pct(r[:, 8]),
                                               "insert": pct(r[:, 9])}
        for name, lo, hi in (("<2k", 0, 2000), ("2k-5k", 2000, 5000),
                             (">=5k", 5000, 1 << 62)):
            r = steady[(steady[:, 3] >= lo) & (steady[:, 3] < hi)]
            if r.size:
                rep["latency_by_input_blocks"][name] = {
                    "requests": int(r.shape[0]), "query": pct(r[:, 8]),
                    "insert": pct(r[:, 9]),
                    "actual_over_theoretical": float(r[:, 5].sum() / max(r[:, 4].sum(), 1))}
    return rep


def fmt_lat(d: dict) -> str:
    if not d:
        return "n/a"
    return (f"p50 {d['p50']:9.0f}  p90 {d['p90']:9.0f}  p99 {d['p99']:9.0f}  "
            f"p99.9 {d['p99.9']:9.0f}  max {d['max']:9.0f}  mean {d['mean']:9.0f}")


def print_report(rep: dict):
    t, th, h = rep["timing"], rep["throughput"], rep["hit"]
    print("\n" + "=" * 96)
    print(f"agentic_stress  mode={'index' if rep['config']['index_only'] else 'data'}  "
          f"nodes={rep['config']['nodes']}  "
          f"concurrency={rep['config']['concurrency']}  aux={rep['config']['aux']}  "
          f"W={rep['config']['swa_window']}  cs={rep['config']['register_chunk_size']}  "
          f"rht_slots={rep['config']['rht_slots']}  flush={rep['config']['flush_after_insert']}  "
          f"max_blocks={rep['config']['max_blocks']}")
    print(f"warmup {t['warmup_sec']:.1f}s  measured {t['measured_sec']:.1f}s "
          f"(incl. drain {t['drain_sec']:.1f}s / {t['drain_requests']} req)")
    print(f"throughput: {th['req_per_sec']:.1f} req/s  {th['blocks_per_sec']:.0f} blocks/s  "
          f"sessions done={th['sessions_done']} truncated={th['sessions_truncated']}")
    if h:
        print(f"hit: actual {h['actual_hit_rate']:.4f}  theoretical {h['theoretical_hit_rate']:.4f}  "
              f"actual/theoretical {h['actual_over_theoretical']:.4f}  "
              f"(p50 {h['ratio_p50']:.3f} p10 {h['ratio_p10']:.3f} min {h['ratio_min']:.3f})")
        print(f"     local {h['local_share']:.3f}  remote {h['remote_share']:.3f}  "
              f"rdma reads/req {h['rdma_reads_mean']:.2f}  atomics/req {h['rdma_atomics_mean']:.2f}  "
              f"requests {h['requests']}")
    x = rep.get("transfer")
    if x:
        print(f"transfer: remote {x['remote_bytes'] / 1e6:.2f} MB in {x['remote_requests']} requests  "
              f"aggregate {x['aggregate_mbps']:.2f} MB/s")
        if x["bandwidth_mbps"]:
            print(f"     per-request MB/s {fmt_lat(x['bandwidth_mbps'])}")
            print(f"     wait us         {fmt_lat(x['xfer_us'])}")
    print("latency (us):" + ("" if rep["config"]["index_only"] else "   query = pull_async + wait"))
    for k in ("query", "verify", "insert", "insert_full"):
        if k in rep["latency_us"] or k != "verify":
            print(f"  {k:12s} {fmt_lat(rep['latency_us'].get(k, {}))}")
    if rep["latency_by_input_blocks"]:
        print("by input blocks:")
        for k, v in rep["latency_by_input_blocks"].items():
            print(f"  {k:6s} n={v['requests']:6d}  act/theory {v['actual_over_theoretical']:.3f}  "
                  f"query p50 {v['query']['p50']:8.0f} p99 {v['query']['p99']:8.0f}  "
                  f"insert p50 {v['insert']['p50']:8.0f} p99 {v['insert']['p99']:8.0f}")
    if rep["hit_by_worker"]:
        print("by worker:")
        for w, v in rep["hit_by_worker"].items():
            lq = rep["latency_by_worker"][w]
            print(f"  W{w}: n={v['requests']:6d}  act/theory {v['actual_over_theoretical']:.3f}  "
                  f"local {v['local_share']:.2f}  query p50/p99 {lq['query']['p50']:7.0f}/"
                  f"{lq['query']['p99']:7.0f}  insert p50/p99 {lq['insert']['p50']:7.0f}/"
                  f"{lq['insert']['p99']:7.0f}")
    if rep["workers"]:
        print("tree state:")
        for s in rep["workers"]:
            print(f"  W{s['worker']}: blocks={s['total_blocks']}  nodes={s['total_radix_nodes']}  "
                  f"mempool {s['mempool_used']}/{s['mempool_total']} "
                  f"({100.0 * s['mempool_used'] / max(s['mempool_total'], 1):.1f}%)  "
                  f"peak queue={s['peak_queue_depth']}  errors={s['errors']}  "
                  f"bad_blocks={s.get('bad_blocks', 0)}  xfer_fail={s.get('transfer_failures', 0)}  "
                  f"staging_short={s.get('staging_shortfalls', 0)}  "
                  f"promo_short={s.get('promotion_shortfalls', 0)}  "
                  f"health={'ok' if s.get('health', 0) == 0 else f'0x{s["health"]:x}'}")
            if s.get("writeback"):
                wb = s["writeback"]
                print(f"      write-back: tasks {wb['produced']} produced, {wb['completed']} completed, "
                      f"{wb['dropped']} dropped, {wb['failed']} failed; held {wb['held_slots']}")
            if s.get("health", 0):
                for line in s["health_report"].splitlines():
                    print(f"      {line}")
    if rep["errors"]:
        print(f"request errors: {rep['errors']}")
    print("=" * 96)


# ======================================================================
# Launcher
# ======================================================================

def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_etcd(workdir: str):
    cport, pport = free_port(), free_port()
    log = open(os.path.join(workdir, "etcd.log"), "wb")
    proc = subprocess.Popen(
        ["etcd", "--name", "agentic",
         "--data-dir", os.path.join(workdir, "etcd-data"),
         "--listen-client-urls", f"http://127.0.0.1:{cport}",
         "--advertise-client-urls", f"http://127.0.0.1:{cport}",
         "--listen-peer-urls", f"http://127.0.0.1:{pport}",
         "--initial-advertise-peer-urls", f"http://127.0.0.1:{pport}",
         "--initial-cluster", f"agentic=http://127.0.0.1:{pport}"],
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


def spawn_workers(cfg: Config, workdir: str):
    procs, ready = [], [threading.Event() for _ in range(cfg.nodes)]
    cfg_json = json.dumps(asdict(cfg))
    for i in range(cfg.nodes):
        log = open(os.path.join(workdir, f"worker{i}.log"), "wb")
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__),
                              "--role", "worker", "--rank", str(i),
                              "--config-json", cfg_json],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             env=dict(os.environ, PYTHONUNBUFFERED="1"))
        procs.append(p)

        def tee(proc=p, f=log, ev=ready[i], idx=i):
            for line in proc.stdout:
                f.write(line)
                f.flush()
                if line.startswith(b"READY "):
                    ev.set()
                elif line.startswith(b"[W") and b"done" in line:
                    sys.stdout.write(line.decode(errors="replace"))
            f.close()
        threading.Thread(target=tee, daemon=True).start()
    return procs, ready


def tail(path: str, n: int = 20) -> str:
    try:
        with open(path, "rb") as f:
            return b"".join(f.readlines()[-n:]).decode(errors="replace")
    except OSError:
        return ""


def cleanup(procs, etcd, cfg: Config):
    for p in procs:
        if p.poll() is None:
            p.terminate()
    for p in procs:
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()
    if etcd is not None and etcd.poll() is None:
        etcd.terminate()
        try:
            etcd.wait(timeout=10)
        except subprocess.TimeoutExpired:
            etcd.kill()
    for path in glob.glob(f"/dev/shm/agentic_{cfg.run_id}*"):
        try:
            os.unlink(path)
        except OSError:
            pass


def run_launcher(cfg: Config) -> int:
    if cfg.nodes > 1:
        backends = getattr(getattr(radixshmem, "_data", None), "TRANSFER_BACKENDS", [])
        missing = [what for what, ok in (
            ("etcd", shutil.which("etcd")),
            ("an RDMA device", rdma_devices(cfg)),
            ("radixshmem built with etcd", hasattr(radixshmem._core, "EtcdKv")),
            ("radixshmem built with Mooncake", cfg.index_only or "mooncake" in backends)) if not ok]
        if missing:
            print(f"SKIP: --nodes {cfg.nodes} needs {', '.join(missing)}")
            return 77
    cfg.run_id = f"{os.getpid():x}{int(time.time()) & 0xffff:x}"
    workdir = cfg.workdir or f"/tmp/agentic_stress_{os.getpid()}"
    os.makedirs(workdir, exist_ok=True)
    cfg.workdir = workdir
    print(f"run_id={cfg.run_id} workdir={workdir}")
    print(f"per-node index: full_slots={cfg.max_blocks} max_nodes={cfg.max_nodes} aux={cfg.aux} "
          f"data_pool_ratio={cfg.data_pool_ratio}  rdma_devs={rdma_devices(cfg) or ['<auto>']}")
    if not cfg.index_only:
        sc = server_config(cfg, 0)
        print(f"per-node store: {sc.data_bytes / 2**30:.2f} GiB  ratios swa={sc.swa_ratio:.4f} "
              f"mamba={sc.mamba_ratio:.4f}  slots FULL {cfg.full_slot_bytes} B"
              f"  SWA {cfg.swa_slot_bytes} B  MAMBA {cfg.mamba_slot_bytes} B")

    etcd = None
    procs = []
    try:
        if cfg.nodes > 1:
            etcd, cfg.etcd_port = start_etcd(workdir)
            print(f"etcd on 127.0.0.1:{cfg.etcd_port}")
        procs, ready = spawn_workers(cfg, workdir)
        deadline = time.time() + cfg.bootstrap_timeout + 60
        for i, ev in enumerate(ready):
            while not ev.wait(timeout=1.0):
                if procs[i].poll() is not None:
                    print(f"worker {i} exited early (rc={procs[i].returncode}):\n"
                          f"{tail(os.path.join(workdir, f'worker{i}.log'))}")
                    return 1
                if time.time() > deadline:
                    print(f"worker {i} did not become ready; log tail:\n"
                          f"{tail(os.path.join(workdir, f'worker{i}.log'))}")
                    return 1
        print(f"all {cfg.nodes} workers ready; starting {cfg.concurrency} sessions")
        rep = asyncio.run(run_generator(cfg))
        print_report(rep)
        if cfg.json_out:
            with open(cfg.json_out, "w") as f:
                json.dump(rep, f, indent=1, default=str)
            print(f"report written to {cfg.json_out}")
        unhealthy = [s["worker"] for s in rep["workers"] if s.get("health", 0)]
        if unhealthy:
            print(f"FAIL: check_health reported leaks on worker(s) {unhealthy}")
            return 1
        bad = sum(s.get("bad_blocks", 0) for s in rep["workers"])
        if bad:
            print(f"FAIL: {bad} block(s) came back with bytes that do not match their hash")
            return 1
        return 0
    except KeyboardInterrupt:
        print("interrupted")
        return 130
    finally:
        cleanup(procs, etcd, cfg)


def main() -> int:
    cfg, role, rank = parse_args()
    if role == "worker":
        return run_worker(cfg, rank)
    return run_launcher(cfg)


if __name__ == "__main__":
    sys.exit(main())
