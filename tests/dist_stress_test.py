#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Distributed stress test for RadixShmem KV cache cluster.

Each rank runs as a separate OS process with its own RadixServer (single-node or
for world_size=1).  Rank 0 acts as the central scheduler: it generates
conversations, dispatches requests to ranks via TCP, and aggregates metrics.

Launch modes
------------
Local simulation (spawns all ranks via multiprocessing):
    python tests/dist_stress_test.py --world-size 4 --max-blocks 500000 --duration 60

Single-rank (for multi-machine deployment):
    # Machine A (rank 0 — full config):
    python tests/dist_stress_test.py --rank 0 --world-size 2 \
        --master-addr 10.0.0.1 --max-blocks 500000
    # Machine B (worker — only connection params, config synced from rank 0):
    python tests/dist_stress_test.py --rank 1 --world-size 2 \
        --master-addr 10.0.0.1
"""

import argparse
import json
import multiprocessing
import os
import queue
import random
import signal
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import shmradix


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class StressConfig:
    rank: int = -1
    world_size: int = 1
    master_addr: str = "127.0.0.1"
    master_port: int = 18500
    dispatch_port: int = 18700
    index_dev: str = ""
    gid_idx: int = 3
    bootstrap_timeout: int = 60

    max_nodes: int = 0
    max_blocks: int = 500000
    block_size: int = 16
    register_chunk_size: int = 128
    background_evict_ratio: float = 0.05

    target_qps: float = 100.0
    isl_min: int = 32
    isl_max: int = 512
    osl_min: int = 16
    osl_max: int = 256
    system_prompt_length: int = 256
    max_turns: int = 10
    num_templates: int = 5
    conv_pool_size: int = 200
    workers_per_rank: int = 1

    transfer_ms_per_block: float = 0.01
    prefill_ms_per_block: float = 0.05
    decode_ms_per_token: float = 0.02

    duration_seconds: int = 0
    seed: int = 42
    report_interval: int = 10


# ======================================================================
# Message protocol (length-prefixed JSON header + binary payload)
# ======================================================================

MSG_REGISTER = 0
MSG_REQUEST = 1
MSG_COMPLETION = 2
MSG_SHUTDOWN = 3
MSG_CONFIG = 4

# Fields NOT synced from rank 0 (local to each machine)
_LOCAL_FIELDS = frozenset({
    "rank", "master_addr", "master_port", "dispatch_port",
    "index_dev", "gid_idx", "bootstrap_timeout",
})


def _config_to_dict(cfg: "StressConfig") -> dict:
    """Serialize all sync-able StressConfig fields to a dict."""
    return {k: getattr(cfg, k)
            for k in cfg.__dataclass_fields__
            if k not in _LOCAL_FIELDS}


def _config_apply(cfg: "StressConfig", d: dict):
    """Apply received config dict onto cfg, skipping local fields."""
    for k, v in d.items():
        if k not in _LOCAL_FIELDS and hasattr(cfg, k):
            setattr(cfg, k, v)


def _recv_exact(sock: socket.socket, n: int, buf: bytearray) -> bool:
    """Read exactly n bytes, appending to *buf*.  Returns True on success.
    On timeout the partial data stays in *buf* so the caller can retry.
    Returns False (peer closed / OS error) only on fatal conditions."""
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except socket.timeout:
            raise
        except (OSError, ConnectionError):
            return False
        if not chunk:
            return False
        buf += chunk
    return True


def send_msg(sock: socket.socket, msg_type: int,
             header: dict, payload: bytes = b""):
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode()
    frame = struct.pack("!III", msg_type, len(hdr_bytes), len(payload))
    sock.sendall(frame + hdr_bytes + payload)


class MsgReader:
    """Stateful message reader that survives socket timeouts without
    losing partially-received data."""

    def __init__(self):
        self._buf = bytearray()
        self._state = 0        # 0=frame header, 1=json header, 2=payload
        self._msg_type = 0
        self._hdr_len = 0
        self._pay_len = 0
        self._hdr_bytes = None

    def recv(self, sock: socket.socket) -> Tuple[Optional[int], Optional[dict], bytes]:
        # Phase 0: read 12-byte frame header
        if self._state == 0:
            if not _recv_exact(sock, 12, self._buf):
                return None, None, b""
            self._msg_type, self._hdr_len, self._pay_len = struct.unpack(
                "!III", bytes(self._buf[:12]))
            self._buf.clear()
            self._state = 1

        # Phase 1: read JSON header
        if self._state == 1:
            if not _recv_exact(sock, self._hdr_len, self._buf):
                return None, None, b""
            self._hdr_bytes = bytes(self._buf[:self._hdr_len])
            self._buf.clear()
            self._state = 2

        # Phase 2: read binary payload
        if self._state == 2:
            if self._pay_len > 0:
                if not _recv_exact(sock, self._pay_len, self._buf):
                    return None, None, b""
                payload = bytes(self._buf[:self._pay_len])
            else:
                payload = b""
            header = json.loads(self._hdr_bytes.decode())
            msg_type = self._msg_type

            # Reset for next message
            self._buf.clear()
            self._state = 0
            self._hdr_bytes = None
            return msg_type, header, payload

        return None, None, b""


def recv_msg(sock: socket.socket, reader: "MsgReader" = None
             ) -> Tuple[Optional[int], Optional[dict], bytes]:
    if reader is not None:
        return reader.recv(sock)
    # Legacy path (no reader) — only safe when timeouts are disabled
    buf = bytearray()
    if not _recv_exact(sock, 12, buf):
        return None, None, b""
    msg_type, hdr_len, pay_len = struct.unpack("!III", bytes(buf[:12]))
    buf.clear()
    if not _recv_exact(sock, hdr_len, buf):
        return None, None, b""
    header = json.loads(bytes(buf[:hdr_len]).decode())
    buf.clear()
    payload = b""
    if pay_len > 0:
        if not _recv_exact(sock, pay_len, buf):
            return None, None, b""
        payload = bytes(buf[:pay_len])
    return msg_type, header, payload


# ======================================================================
# Data structures
# ======================================================================

@dataclass
class TurnData:
    user_input: np.ndarray
    model_output: np.ndarray


@dataclass
class ConvState:
    id: int = 0
    template_id: int = 0
    num_turns: int = 0
    current_turn: int = 0
    in_flight: bool = False
    complete: bool = False
    turns: List[TurnData] = field(default_factory=list)
    system_prompt: Optional[np.ndarray] = None
    hashes_inserted: int = 0


@dataclass
class Request:
    conv_id: int
    conv_pool_idx: int
    turn: int
    template_id: int
    input_tokens: np.ndarray
    new_token_count: int
    osl: int
    expected_hit_hashes: int
    output_tokens: np.ndarray


@dataclass
class Completion:
    pool_idx: int
    full_num_hashes: int
    template_id: int
    executing_rank: int
    actual_hit: int
    local_hit: int
    remote_hit: int
    expected_hit: int
    num_hashes: int
    query_us: float
    insert_us: float
    rdma_reads: int
    rdma_atomics: int
    error: str  # "" for OK, otherwise InsertError name


# ======================================================================
# Aggregated metrics (thread-safe, on rank 0)
# ======================================================================

class AggMetrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.queries = 0
        self.actual_hit = 0
        self.local_hit = 0
        self.remote_hit = 0
        self.expected_hit = 0
        self.total_blocks = 0
        self.query_us = 0.0
        self.max_query_us = 0.0
        self.insert_us = 0.0
        self.max_insert_us = 0.0
        self.done = 0
        self.errors = 0
        self.error_types: Dict[str, int] = {}
        self.rdma_reads = 0
        self.rdma_atomics = 0
        self.query_lat_buf: List[float] = []
        self.insert_lat_buf: List[float] = []
        self.per_rank_done: Dict[int, int] = {}

    def record(self, comp: Completion):
        with self.lock:
            self.queries += 1
            self.actual_hit += comp.actual_hit
            self.local_hit += comp.local_hit
            self.remote_hit += comp.remote_hit
            self.expected_hit += comp.expected_hit
            self.total_blocks += comp.num_hashes
            self.query_us += comp.query_us
            if comp.query_us > self.max_query_us:
                self.max_query_us = comp.query_us
            self.insert_us += comp.insert_us
            if comp.insert_us > self.max_insert_us:
                self.max_insert_us = comp.insert_us
            self.done += 1
            if comp.error:
                self.errors += 1
                self.error_types[comp.error] = (
                    self.error_types.get(comp.error, 0) + 1
                )
            self.rdma_reads += comp.rdma_reads
            self.rdma_atomics += comp.rdma_atomics
            self.query_lat_buf.append(comp.query_us)
            self.insert_lat_buf.append(comp.insert_us)
            self.per_rank_done[comp.executing_rank] = (
                self.per_rank_done.get(comp.executing_rank, 0) + 1
            )

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "queries": self.queries, "actual_hit": self.actual_hit,
                "local_hit": self.local_hit, "remote_hit": self.remote_hit,
                "expected_hit": self.expected_hit, "total_blocks": self.total_blocks,
                "query_us": self.query_us, "max_query_us": self.max_query_us,
                "insert_us": self.insert_us, "max_insert_us": self.max_insert_us,
                "done": self.done, "errors": self.errors,
                "rdma_reads": self.rdma_reads, "rdma_atomics": self.rdma_atomics,
                "per_rank_done": dict(self.per_rank_done),
                "error_types": dict(self.error_types),
            }

    def drain_latencies(self) -> Tuple[List[float], List[float]]:
        with self.lock:
            qlats = self.query_lat_buf[:]
            self.query_lat_buf.clear()
            ilats = self.insert_lat_buf[:]
            self.insert_lat_buf.clear()
            return qlats, ilats


# ======================================================================
# Workload generation
# ======================================================================

def generate_templates(cfg: StressConfig, rng: random.Random) -> List[np.ndarray]:
    templates = []
    for _ in range(cfg.num_templates):
        tokens = np.array([rng.randint(0, 0x7FFFFFFF)
                           for _ in range(cfg.system_prompt_length)], dtype=np.int64)
        templates.append(tokens)
    return templates


def generate_conversation(conv: ConvState, conv_id: int,
                          templates: List[np.ndarray],
                          cfg: StressConfig, rng: random.Random):
    conv.id = conv_id
    conv.current_turn = 0
    conv.in_flight = False
    conv.complete = False
    conv.hashes_inserted = 0

    conv.template_id = rng.randint(0, len(templates) - 1)
    conv.system_prompt = templates[conv.template_id]
    conv.num_turns = rng.randint(1, cfg.max_turns)

    isl_blk_min = max(1, cfg.isl_min // cfg.block_size)
    isl_blk_max = max(isl_blk_min, cfg.isl_max // cfg.block_size)
    osl_blk_min = max(1, cfg.osl_min // cfg.block_size)
    osl_blk_max = max(osl_blk_min, cfg.osl_max // cfg.block_size)

    conv.turns = []
    for _ in range(conv.num_turns):
        isl_tokens = rng.randint(isl_blk_min, isl_blk_max) * cfg.block_size
        osl_tokens = rng.randint(osl_blk_min, osl_blk_max) * cfg.block_size
        user_input = np.array([rng.randint(0, 0x7FFFFFFF)
                               for _ in range(isl_tokens)], dtype=np.int64)
        model_output = np.array([rng.randint(0, 0x7FFFFFFF)
                                 for _ in range(osl_tokens)], dtype=np.int64)
        conv.turns.append(TurnData(user_input=user_input, model_output=model_output))


def build_input_tokens(conv: ConvState, turn: int) -> np.ndarray:
    parts = [conv.system_prompt]
    for t in range(turn):
        parts.append(conv.turns[t].user_input)
        parts.append(conv.turns[t].model_output)
    parts.append(conv.turns[turn].user_input)
    return np.concatenate(parts)


# ======================================================================
# Core: execute a request on a RadixServer (standalone or distributed)
# ======================================================================

def _insert_full(node, full_hashes, is_standalone):
    """allocate_slots (LRU-evicting) + insert — slots are now mandatory and
    caller-supplied. Peek the LOCAL match only (a remote hit must not shrink
    the local allocation) without perturbing LRU, allocate just the shortfall,
    insert, and recycle any unused tail. Call under the writer lock, if any."""
    matched = int(
        node.query(full_hashes, mask=shmradix.COMPONENT_MASK_FULL,
                   local_only=not is_standalone,
                   lock=False, update_meta=False).common_hit
    )
    need = len(full_hashes) - matched
    slots = (node.allocate_slots(need) if need > 0
             else np.empty(0, dtype=np.int32))
    # Slots bind hashes[matched:] via start=matched; insert auto-recycles unused.
    r = node.insert(full_hashes, slots, start=matched, auto_recycle=True)
    return r


def execute_request(node, req: Request,
                    cfg: StressConfig, executing_rank: int,
                    insert_lock: Optional[threading.Lock] = None) -> Completion:
    is_standalone = not node.is_distributed()

    num_tokens = len(req.input_tokens)
    num_hashes = num_tokens // cfg.block_size

    if num_hashes <= 0:
        return Completion(
            pool_idx=req.conv_pool_idx, full_num_hashes=0,
            template_id=req.template_id, executing_rank=executing_rank,
            actual_hit=0, local_hit=0, remote_hit=0,
            expected_hit=req.expected_hit_hashes, num_hashes=0,
            query_us=0.0, insert_us=0.0,
            rdma_reads=0, rdma_atomics=0, error="",
        )

    input_hashes = shmradix.compute_hashes(req.input_tokens, cfg.block_size)

    t0 = time.perf_counter()
    qr = node.query(input_hashes, mask=shmradix.COMPONENT_MASK_FULL)
    t1 = time.perf_counter()
    query_us = (t1 - t0) * 1e6
    if query_us > 500000:
        print(f"[R{executing_rank}] SLOW query: {query_us/1e6:.2f}s "
              f"h={num_hashes}", flush=True)

    actual_hit = qr.common_hit
    own = int(node.rank()) if not is_standalone else 0
    local_hit = sum(len(f[2]) for f in qr.full_fragments if f[0] == own) \
        if not is_standalone else actual_hit
    remote_hit = actual_hit - local_hit if not is_standalone else 0
    rdma_reads = qr.rdma_read_count if not is_standalone else 0
    rdma_atomics = qr.rdma_atomic_count if not is_standalone else 0

    if actual_hit > 0:
        us = int(actual_hit * cfg.transfer_ms_per_block * 1000)
        if us > 0:
            time.sleep(us * 1e-6)

    unmatched = num_hashes - actual_hit
    compute_us = (int(unmatched * cfg.prefill_ms_per_block * 1000) +
                  int(req.osl * cfg.decode_ms_per_token * 1000))
    if compute_us > 0:
        time.sleep(compute_us * 1e-6)

    full_tokens = np.concatenate([req.input_tokens, req.output_tokens])
    full_num_hashes = len(full_tokens) // cfg.block_size
    ins_error = ""
    insert_us = 0.0

    if full_num_hashes > 0:
        full_hashes = shmradix.compute_hashes(full_tokens, cfg.block_size)
        t_ins = time.perf_counter()
        if insert_lock:
            with insert_lock:
                ins_result = _insert_full(node, full_hashes, is_standalone)
        else:
            ins_result = _insert_full(node, full_hashes, is_standalone)
        insert_us = (time.perf_counter() - t_ins) * 1e6
        if insert_us > 500000:
            print(f"[R{executing_rank}] SLOW insert: {insert_us/1e6:.2f}s "
                  f"h={full_num_hashes}", flush=True)

        new_blocks = full_num_hashes - actual_hit
        if new_blocks > 0:
            us = int(new_blocks * cfg.transfer_ms_per_block * 1000)
            if us > 0:
                time.sleep(us * 1e-6)

        if ins_result.error != shmradix.InsertError.OK:
            ins_error = ins_result.error.name

    return Completion(
        pool_idx=req.conv_pool_idx, full_num_hashes=full_num_hashes,
        template_id=req.template_id, executing_rank=executing_rank,
        actual_hit=actual_hit, local_hit=local_hit, remote_hit=remote_hit,
        expected_hit=req.expected_hit_hashes, num_hashes=num_hashes,
        query_us=query_us, insert_us=insert_us,
        rdma_reads=rdma_reads, rdma_atomics=rdma_atomics, error=ins_error,
    )


# ======================================================================
# Worker thread helpers (multi-worker-per-rank)
# ======================================================================

def _worker_thread_fn(node, cfg: StressConfig, rank: int,
                      req_queue: queue.Queue, comp_queue: queue.Queue,
                      insert_lock: Optional[threading.Lock],
                      running_flag: list):
    """Worker thread: pulls requests, executes query+insert, pushes completions."""
    while running_flag[0]:
        try:
            req = req_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        comp = execute_request(node, req, cfg, rank, insert_lock)
        comp_queue.put(comp)


def _completion_sender_fn(sock: socket.socket,
                          comp_queue: queue.Queue,
                          running_flag: list):
    """Sender thread: reads completions from queue and sends via TCP."""
    while running_flag[0]:
        try:
            comp = comp_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            send_completion(sock, comp)
        except (OSError, ConnectionError):
            break


# ======================================================================
# Serialize / deserialize requests over TCP
# ======================================================================

def send_request(sock: socket.socket, req: Request):
    header = {
        "conv_id": int(req.conv_id), "pool_idx": int(req.conv_pool_idx),
        "turn": int(req.turn), "template_id": int(req.template_id),
        "new_token_count": int(req.new_token_count), "osl": int(req.osl),
        "expected_hit": int(req.expected_hit_hashes),
        "input_len": len(req.input_tokens), "output_len": len(req.output_tokens),
    }
    payload = req.input_tokens.tobytes() + req.output_tokens.tobytes()
    send_msg(sock, MSG_REQUEST, header, payload)


def decode_request(header: dict, payload: bytes) -> Request:
    input_len = header["input_len"]
    input_tokens = np.frombuffer(payload[:input_len * 8], dtype=np.int64).copy()
    output_tokens = np.frombuffer(payload[input_len * 8:], dtype=np.int64).copy()
    return Request(
        conv_id=header["conv_id"], conv_pool_idx=header["pool_idx"],
        turn=header["turn"], template_id=header["template_id"],
        input_tokens=input_tokens, new_token_count=header["new_token_count"],
        osl=header["osl"], expected_hit_hashes=header["expected_hit"],
        output_tokens=output_tokens,
    )


def send_completion(sock: socket.socket, comp: Completion):
    send_msg(sock, MSG_COMPLETION, {
        "pool_idx": comp.pool_idx, "full_num_hashes": comp.full_num_hashes,
        "template_id": comp.template_id, "executing_rank": comp.executing_rank,
        "actual_hit": comp.actual_hit, "local_hit": comp.local_hit,
        "remote_hit": comp.remote_hit, "expected_hit": comp.expected_hit,
        "num_hashes": comp.num_hashes, "query_us": comp.query_us,
        "insert_us": comp.insert_us,
        "rdma_reads": comp.rdma_reads, "rdma_atomics": comp.rdma_atomics,
        "error": comp.error,
    })


def decode_completion(header: dict) -> Completion:
    return Completion(**header)


# ======================================================================
# Percentile helper
# ======================================================================

def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    data.sort()
    idx = int(p / 100.0 * (len(data) - 1))
    return data[min(idx, len(data) - 1)]


# ======================================================================
# Rank 0: Scheduler
# ======================================================================

def _scheduler_accept_workers(cfg: StressConfig,
                              send_config: bool = False,
                              ) -> Dict[int, socket.socket]:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", cfg.dispatch_port))
    srv.listen(cfg.world_size)
    srv.settimeout(cfg.bootstrap_timeout)

    config_dict = _config_to_dict(cfg) if send_config else None

    peers: Dict[int, socket.socket] = {}
    print(f"[R0] Waiting for {cfg.world_size - 1} worker(s) on port {cfg.dispatch_port}...")

    while len(peers) < cfg.world_size - 1:
        try:
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            msg_type, header, _ = recv_msg(conn)
            if msg_type == MSG_REGISTER:
                rank = header["rank"]
                if send_config:
                    send_msg(conn, MSG_CONFIG, config_dict)
                    print(f"[R0]   Worker rank {rank} connected from "
                          f"{addr[0]}:{addr[1]} (config synced)")
                else:
                    print(f"[R0]   Worker rank {rank} connected from "
                          f"{addr[0]}:{addr[1]}")
                peers[rank] = conn
        except socket.timeout:
            print("ERROR: Timed out waiting for workers", file=sys.stderr)
            sys.exit(1)

    srv.close()
    return peers


def _worker_connect_and_recv_config(
        cfg: StressConfig) -> Tuple[StressConfig, socket.socket]:
    """Connect to rank 0, send REGISTER, receive CONFIG, return (updated cfg, sock)."""
    print(f"[R{cfg.rank}] Connecting to scheduler at "
          f"{cfg.master_addr}:{cfg.dispatch_port} for config sync...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    deadline = time.monotonic() + cfg.bootstrap_timeout
    connected = False
    while time.monotonic() < deadline and not connected:
        try:
            sock.connect((cfg.master_addr, cfg.dispatch_port))
            connected = True
        except ConnectionRefusedError:
            time.sleep(0.5)

    if not connected:
        print(f"[R{cfg.rank}] ERROR: Could not connect to scheduler",
              file=sys.stderr)
        sys.exit(1)

    send_msg(sock, MSG_REGISTER, {"rank": cfg.rank})

    msg_type, header, _ = recv_msg(sock)
    if msg_type != MSG_CONFIG or header is None:
        print(f"[R{cfg.rank}] ERROR: Expected MSG_CONFIG, got {msg_type}",
              file=sys.stderr)
        sys.exit(1)

    _config_apply(cfg, header)
    print(f"[R{cfg.rank}] Config received from rank 0: "
          f"max_blocks={cfg.max_blocks} block_size={cfg.block_size} "
          f"qps={cfg.target_qps} workers_per_rank={cfg.workers_per_rank}")
    return cfg, sock


def _remote_recv_fn(rank: int, sock: socket.socket,
                    comp_queue: queue.Queue, running_flag):
    """Receive completions from a remote worker rank (lightweight thread)."""
    sock.settimeout(0.5)
    reader = MsgReader()
    while running_flag[0]:
        try:
            msg_type, header, _ = recv_msg(sock, reader)
        except socket.timeout:
            continue
        except (OSError, ConnectionError):
            break
        if msg_type is None:
            break
        if msg_type == MSG_COMPLETION:
            comp_queue.put(decode_completion(header))


def run_scheduler(cfg: StressConfig, node,
                  peer_socks: Optional[Dict[int, socket.socket]] = None):
    running = [True]

    def signal_handler(signum, frame):
        running[0] = False
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if peer_socks is None:
        peer_socks = {}
        if cfg.world_size > 1:
            peer_socks = _scheduler_accept_workers(cfg)
    print(f"[R0] All {cfg.world_size} rank(s) ready.\n")

    rng = random.Random(cfg.seed)
    templates = generate_templates(cfg, rng)

    convs: List[ConvState] = [ConvState() for _ in range(cfg.conv_pool_size)]
    next_conv_id = 0
    for i in range(cfg.conv_pool_size):
        generate_conversation(convs[i], next_conv_id, templates, cfg, rng)
        next_conv_id += 1

    template_inserted = [False] * cfg.num_templates
    dispatch_rng = random.Random(cfg.seed + 7)

    comp_queue: queue.Queue = queue.Queue()
    agg = AggMetrics()

    # Local worker thread pool for rank 0
    local_req_queue: queue.Queue = queue.Queue()
    insert_lock = threading.Lock() if cfg.workers_per_rank > 1 else None
    local_workers = []
    for w in range(cfg.workers_per_rank):
        t = threading.Thread(
            target=_worker_thread_fn,
            args=(node, cfg, 0, local_req_queue, comp_queue, insert_lock, running),
            daemon=True,
        )
        t.start()
        local_workers.append(t)

    for rank, sock in peer_socks.items():
        t = threading.Thread(
            target=_remote_recv_fn,
            args=(rank, sock, comp_queue, running),
            daemon=True,
        )
        t.start()

    # ---- Main loop: generate, dispatch, execute local, collect ----
    total_workers = cfg.world_size * cfg.workers_per_rank
    print(f"[R0] Generating requests at {cfg.target_qps:.1f} QPS "
          f"across {cfg.world_size} rank(s) x {cfg.workers_per_rank} worker(s) "
          f"= {total_workers} worker(s)...\n")

    test_start = time.monotonic()
    request_interval = 1.0 / cfg.target_qps if cfg.target_qps > 0 else 0
    next_dispatch = time.monotonic()
    total_dispatched = 0
    convs_completed = 0

    prev_report = time.monotonic()
    prev_queries = 0
    prev_query_us = 0.0
    prev_insert_us = 0.0

    while running[0]:
        if cfg.duration_seconds > 0:
            if time.monotonic() - test_start >= cfg.duration_seconds:
                break

        # --- Drain completion queue (non-blocking) ---
        while True:
            try:
                comp = comp_queue.get_nowait()
            except queue.Empty:
                break
            agg.record(comp)
            conv = convs[comp.pool_idx]
            conv.hashes_inserted = comp.full_num_hashes
            conv.current_turn += 1
            conv.in_flight = False
            if conv.current_turn >= conv.num_turns:
                conv.complete = True
            template_inserted[comp.template_id] = True

        # --- Periodic report ---
        now = time.monotonic()
        if now - prev_report >= cfg.report_interval:
            _print_report(agg, node, cfg, test_start, prev_queries,
                         prev_query_us, prev_insert_us)
            snap = agg.snapshot()
            prev_queries = snap["queries"]
            prev_query_us = snap["query_us"]
            prev_insert_us = snap["insert_us"]
            prev_report = now

        # --- Rate limit ---
        now = time.monotonic()
        if now < next_dispatch:
            time.sleep(min(next_dispatch - now, 0.05))
            if not running[0]:
                break
            now = time.monotonic()
            if now < next_dispatch:
                continue
        next_dispatch += request_interval
        if next_dispatch < now:
            next_dispatch = now

        # --- Pick target rank: round-robin by (conv_id + turn) so adjacent
        #     turns of the same conversation land on different ranks,
        #     maximising remote cache hits. ---

        # --- Find a ready conversation ---
        pool_idx = -1
        for i in range(cfg.conv_pool_size):
            if not convs[i].in_flight and not convs[i].complete:
                pool_idx = i
                break

        if pool_idx < 0:
            for i in range(cfg.conv_pool_size):
                if convs[i].complete:
                    convs_completed += 1
                    generate_conversation(convs[i], next_conv_id, templates,
                                          cfg, dispatch_rng)
                    next_conv_id += 1
                    pool_idx = i
                    break

        if pool_idx < 0:
            continue

        convs[pool_idx].in_flight = True

        conv = convs[pool_idx]
        turn = conv.current_turn

        input_tokens = build_input_tokens(conv, turn)
        total_input_hashes = len(input_tokens) // cfg.block_size
        sys_hashes = cfg.system_prompt_length // cfg.block_size

        if conv.hashes_inserted > 0:
            expected_hit = min(conv.hashes_inserted, total_input_hashes)
        elif template_inserted[conv.template_id]:
            expected_hit = min(sys_hashes, total_input_hashes)
        else:
            expected_hit = 0

        req = Request(
            conv_id=conv.id, conv_pool_idx=pool_idx, turn=turn,
            template_id=conv.template_id, input_tokens=input_tokens,
            new_token_count=len(conv.turns[turn].user_input),
            osl=len(conv.turns[turn].model_output),
            expected_hit_hashes=expected_hit,
            output_tokens=conv.turns[turn].model_output,
        )

        # --- Dispatch ---
        target_rank = (conv.id + turn // 4) % cfg.world_size
        if target_rank == 0:
            local_req_queue.put(req)
        else:
            try:
                send_request(peer_socks[target_rank], req)
            except (OSError, ConnectionError):
                print(f"WARNING: Lost connection to rank {target_rank}",
                      file=sys.stderr)
                convs[pool_idx].in_flight = False
                continue

        total_dispatched += 1

    # ---- Shutdown ----
    print("\n[R0] Shutting down scheduler...")

    # Wait for local worker threads to drain
    drain_deadline = time.monotonic() + 5.0
    while not local_req_queue.empty() and time.monotonic() < drain_deadline:
        while True:
            try:
                comp = comp_queue.get_nowait()
            except queue.Empty:
                break
            agg.record(comp)
            conv = convs[comp.pool_idx]
            conv.hashes_inserted = comp.full_num_hashes
            conv.current_turn += 1
            conv.in_flight = False
            if conv.current_turn >= conv.num_turns:
                conv.complete = True
            template_inserted[comp.template_id] = True
        time.sleep(0.05)

    running[0] = False

    for rank, sock in peer_socks.items():
        try:
            send_msg(sock, MSG_SHUTDOWN, {})
        except OSError:
            pass

    time.sleep(1.0)

    # Drain any remaining completions
    while True:
        try:
            comp = comp_queue.get_nowait()
        except queue.Empty:
            break
        agg.record(comp)

    for sock in peer_socks.values():
        try:
            sock.close()
        except OSError:
            pass

    # ---- Final summary ----
    total_time = time.monotonic() - test_start
    _print_final_summary(agg, node, cfg, total_time, total_dispatched, convs_completed)

    snap = agg.snapshot()
    return snap["errors"]


def _print_report(agg: AggMetrics, node, cfg: StressConfig,
                  start_time: float, prev_queries: int,
                  prev_query_us: float, prev_insert_us: float):
    snap = agg.snapshot()
    qlats, ilats = agg.drain_latencies()

    elapsed = time.monotonic() - start_time
    elapsed_m = int(elapsed) // 60
    elapsed_s = int(elapsed) % 60

    dq = snap["queries"] - prev_queries
    interval_qps = dq / cfg.report_interval if cfg.report_interval > 0 else 0

    tb = snap["total_blocks"]
    hit_pct = (100.0 * snap["actual_hit"] / tb) if tb > 0 else 0
    local_pct = (100.0 * snap["local_hit"] / tb) if tb > 0 else 0
    remote_pct = (100.0 * snap["remote_hit"] / tb) if tb > 0 else 0
    theo_pct = (100.0 * snap["expected_hit"] / tb) if tb > 0 else 0

    dq_us = snap["query_us"] - prev_query_us
    avg_q_us = dq_us / dq if dq > 0 else 0
    di_us = snap["insert_us"] - prev_insert_us
    avg_i_us = di_us / dq if dq > 0 else 0

    qp50 = percentile(qlats, 50)
    qp90 = percentile(qlats, 90)
    qp99 = percentile(qlats, 99)
    qmx = max(qlats) if qlats else 0

    ip50 = percentile(ilats, 50)
    ip90 = percentile(ilats, 90)
    ip99 = percentile(ilats, 99)
    imx = max(ilats) if ilats else 0

    per_rank = snap["per_rank_done"]
    rank_str = " ".join(f"R{r}={c}" for r, c in sorted(per_rank.items()))

    print(
        f"[{elapsed_m:02d}:{elapsed_s:02d}] "
        f"QPS={interval_qps:.1f}  Total={snap['done']}  "
        f"Hit={hit_pct:.1f}%(L={local_pct:.1f}% R={remote_pct:.1f}%)  "
        f"Theo={theo_pct:.1f}%  "
        f"RDMA(r={snap['rdma_reads']},a={snap['rdma_atomics']})  "
        f"Err={snap['errors']}"
    )
    print(
        f"  [query  Avg={avg_q_us:.1f}us  P50={qp50:.1f}  P90={qp90:.1f}  "
        f"P99={qp99:.1f}  Max={qmx:.0f}us]"
    )
    print(
        f"  [insert Avg={avg_i_us:.1f}us  P50={ip50:.1f}  P90={ip90:.1f}  "
        f"P99={ip99:.1f}  Max={imx:.0f}us]"
    )
    et = snap.get("error_types", {})
    if et:
        err_detail = " ".join(f"{k}={v}" for k, v in sorted(et.items()))
        print(f"  [errors: {err_detail}]")
    print(f"  [{rank_str}]")
    tree_line = (
        f"  [local: nodes={node.total_radix_nodes()}  blocks={node.total_blocks()}  "
        f"dp={node.data_pool_utilization() * 100:.1f}%"
    )
    if cfg.max_blocks > 0:
        tree_line += f"  mp={node.mempool_used()}/{node.mempool_total()}"
    tree_line += "]"
    print(tree_line)
    sys.stdout.flush()


def _print_final_summary(agg: AggMetrics, node, cfg: StressConfig,
                         total_time: float, total_dispatched: int,
                         convs_completed: int):
    snap = agg.snapshot()

    print(f"\n=== Final Summary (scheduler) ===\n")
    print(f"  Duration:           {total_time:.1f} s")
    print(f"  Total dispatched:   {total_dispatched}")
    print(f"  Total completed:    {snap['done']}")
    print(f"  Conversations done: {convs_completed}")
    eff_qps = snap["done"] / total_time if total_time > 0 else 0
    print(f"  Effective QPS:      {eff_qps:.1f}")
    print()

    tb = snap["total_blocks"]
    hit = (100.0 * snap["actual_hit"] / tb) if tb > 0 else 0
    lhit = (100.0 * snap["local_hit"] / tb) if tb > 0 else 0
    rhit = (100.0 * snap["remote_hit"] / tb) if tb > 0 else 0
    theo = (100.0 * snap["expected_hit"] / tb) if tb > 0 else 0
    ratio = snap["actual_hit"] / snap["expected_hit"] if snap["expected_hit"] > 0 else 0

    print(f"  Actual hit rate:    {hit:.2f}%")
    print(f"    Local hit:        {lhit:.2f}%")
    print(f"    Remote hit:       {rhit:.2f}%")
    print(f"  Theoretical hit:    {theo:.2f}%")
    print(f"  Hit ratio:          {ratio:.3f}")
    print()

    avg_q_us = snap["query_us"] / snap["queries"] if snap["queries"] > 0 else 0
    avg_i_us = snap["insert_us"] / snap["queries"] if snap["queries"] > 0 else 0
    print(f"  Query latency (cumulative):")
    print(f"    avg:    {avg_q_us:10.1f} us")
    print(f"    max:    {snap['max_query_us']:10.1f} us")
    print(f"  Insert latency (cumulative):")
    print(f"    avg:    {avg_i_us:10.1f} us")
    print(f"    max:    {snap['max_insert_us']:10.1f} us")
    print()

    print(f"  RDMA reads:         {snap['rdma_reads']}")
    print(f"  RDMA atomics:       {snap['rdma_atomics']}")
    print(f"  Insert errors:      {snap['errors']}")
    et = snap.get("error_types", {})
    if et:
        for etype, count in sorted(et.items()):
            print(f"    {etype:20s} {count}")
    print()

    per_rank = snap["per_rank_done"]
    print(f"  Per-rank completions:")
    for r in sorted(per_rank.keys()):
        print(f"    Rank {r}: {per_rank[r]}")
    print()

    print(f"  Local tree stats:")
    print(f"    Nodes:            {node.total_radix_nodes()}")
    print(f"    Blocks:           {node.total_blocks()}")
    print(f"    DataPool:         {node.data_pool_utilization() * 100:.1f}%")
    if cfg.max_blocks > 0:
        print(f"    Mempool:          {node.mempool_used()}/{node.mempool_total()}")
    print()

    status = "PASSED (no errors)" if snap["errors"] == 0 else "FAILED (errors detected)"
    print(status)


# ======================================================================
# Rank N: Worker
# ======================================================================

def run_worker(cfg: StressConfig, node,
               sock: Optional[socket.socket] = None):
    running = [True]

    def signal_handler(signum, frame):
        running[0] = False
    signal.signal(signal.SIGTERM, signal_handler)

    if sock is None:
        # Connect to rank 0 (legacy path without config sync)
        print(f"[R{cfg.rank}] Connecting to scheduler at "
              f"{cfg.master_addr}:{cfg.dispatch_port}...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        deadline = time.monotonic() + cfg.bootstrap_timeout
        connected = False
        while time.monotonic() < deadline and not connected:
            try:
                sock.connect((cfg.master_addr, cfg.dispatch_port))
                connected = True
            except ConnectionRefusedError:
                time.sleep(0.5)

        if not connected:
            print(f"[R{cfg.rank}] ERROR: Could not connect to scheduler",
                  file=sys.stderr)
            sys.exit(1)

        send_msg(sock, MSG_REGISTER, {"rank": cfg.rank})

    print(f"[R{cfg.rank}] Connected to scheduler. "
          f"{cfg.workers_per_rank} worker thread(s). Processing requests...\n")

    # Worker thread pool
    req_queue: queue.Queue = queue.Queue()
    comp_out_queue: queue.Queue = queue.Queue()
    insert_lock = threading.Lock() if cfg.workers_per_rank > 1 else None

    workers = []
    for w in range(cfg.workers_per_rank):
        t = threading.Thread(
            target=_worker_thread_fn,
            args=(node, cfg, cfg.rank, req_queue, comp_out_queue,
                  insert_lock, running),
            daemon=True,
        )
        t.start()
        workers.append(t)

    # Sender thread: sends completions back to scheduler
    sender = threading.Thread(
        target=_completion_sender_fn,
        args=(sock, comp_out_queue, running),
        daemon=True,
    )
    sender.start()

    done_count = 0
    report_start = time.monotonic()
    prev_done = 0
    prev_report = time.monotonic()

    sock.settimeout(0.5)
    reader = MsgReader()

    while running[0]:
        try:
            msg_type, header, payload = recv_msg(sock, reader)
        except socket.timeout:
            # Periodic local report (on timeout, cheap check)
            now = time.monotonic()
            if now - prev_report >= cfg.report_interval:
                cur_done = done_count + comp_out_queue.qsize()
                elapsed = now - report_start
                elapsed_m = int(elapsed) // 60
                elapsed_s = int(elapsed) % 60
                dq = cur_done - prev_done
                interval_qps = dq / cfg.report_interval
                print(
                    f"[R{cfg.rank} {elapsed_m:02d}:{elapsed_s:02d}] "
                    f"QPS={interval_qps:.1f}  Done~={cur_done}  "
                    f"InQ={req_queue.qsize()}  "
                    f"nodes={node.total_radix_nodes()}  blocks={node.total_blocks()}  "
                    f"dp={node.data_pool_utilization() * 100:.1f}%"
                    + (f"  mp={node.mempool_used()}/{node.mempool_total()}"
                       if cfg.max_blocks > 0 else "")
                )
                sys.stdout.flush()
                prev_done = cur_done
                prev_report = now
            continue
        except (OSError, ConnectionError):
            break

        if msg_type is None or msg_type == MSG_SHUTDOWN:
            break

        if msg_type == MSG_REQUEST:
            req = decode_request(header, payload)
            req_queue.put(req)
            done_count += 1

    # Drain remaining requests
    running[0] = False

    try:
        sock.close()
    except OSError:
        pass

    print(f"\n[R{cfg.rank}] Done. Dispatched {done_count} requests.")
    print(f"  Tree: nodes={node.total_radix_nodes()}  blocks={node.total_blocks()}  "
          f"dp={node.data_pool_utilization() * 100:.1f}%")
    return 0


# ======================================================================
# Per-rank entry point
# ======================================================================

def _create_cluster_node(rank: int, cfg: StressConfig):
    """Create RadixServerConfig + RadixServer and bootstrap (distributed)."""
    shm_cfg = shmradix._core.ShmConfig(
        max_nodes=cfg.max_nodes, max_blocks=cfg.max_blocks,
        block_size=cfg.block_size, register_chunk_tokens=cfg.register_chunk_size * cfg.block_size,
        background_evict_ratio=cfg.background_evict_ratio,
    )
    rscfg = shmradix._core.RadixServerConfig()
    rscfg.shm = shm_cfg
    rscfg.rank = rank
    rscfg.world_size = cfg.world_size
    rscfg.registry = f"etcd://{cfg.master_addr}:2379"
    rscfg.cluster_id = "dist_stress"
    rscfg.node_name = f"node{rank}"
    rscfg.index_dev = cfg.index_dev
    rscfg.gid_idx = cfg.gid_idx
    rscfg.bootstrap_timeout_sec = cfg.bootstrap_timeout
    rscfg.register_chunk_tokens = cfg.register_chunk_size * cfg.block_size

    print(f"[R{rank}] Creating RadixServer (distributed)...")
    server = shmradix._core.RadixServer(rscfg)
    print(f"[R{rank}] Bootstrapping cluster...")
    if not server.bootstrap():
        print(f"[R{rank}] ERROR: Bootstrap failed!", file=sys.stderr)
        sys.exit(1)
    client = shmradix._core.RadixClient(server)
    print(f"[R{rank}] Cluster ready. shm={server.shm_name()}\n")
    return server, client


def run_rank(rank: int, cfg: StressConfig):
    """Entry point for each rank process."""
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    cfg = StressConfig(**{k: getattr(cfg, k) for k in cfg.__dataclass_fields__})
    cfg.rank = rank

    if rank != 0:
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    print(f"[R{rank}] Starting (pid={os.getpid()})...")

    if cfg.world_size == 1:
        # Standalone mode
        shm_cfg = shmradix._core.ShmConfig(
            max_nodes=cfg.max_nodes, max_blocks=cfg.max_blocks,
            block_size=cfg.block_size, register_chunk_tokens=cfg.register_chunk_size * cfg.block_size,
            background_evict_ratio=cfg.background_evict_ratio,
        )
        shm_name = f"/shmradix_stress_0"
        if cfg.max_nodes == 0:
            shm_cfg.max_nodes = cfg.max_blocks
        print(f"[R0] Creating RadixServer (standalone mode)...")
        server = shmradix._core.RadixServer(shm_name, shm_cfg)
        client = shmradix._core.RadixClient(server)
        print(f"[R0] Ready. shm={shm_name}\n")
        errors = run_scheduler(cfg, client)
    elif rank == 0:
        # Rank 0: accept workers and send config BEFORE creating RadixServer
        peer_socks = _scheduler_accept_workers(cfg, send_config=True)
        server, client = _create_cluster_node(rank, cfg)
        errors = run_scheduler(cfg, client, peer_socks=peer_socks)
    else:
        # Worker: connect, receive config, THEN create RadixServer
        cfg, dispatch_sock = _worker_connect_and_recv_config(cfg)
        server, client = _create_cluster_node(rank, cfg)
        errors = run_worker(cfg, client, sock=dispatch_sock)

    # End-of-run leak audit of this rank's tree (no locks taken).
    health = client.check_health()
    if health != 0:
        print(f"[R{rank}] check_health FAILED (flags=0x{health:x}):\n"
              f"{client.health_report().describe()}")
    else:
        print(f"[R{rank}] check_health OK")
    sys.exit(0 if errors == 0 and health == 0 else 1)


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> Tuple[StressConfig, bool]:
    """Returns (config, is_single_rank_mode)."""
    p = argparse.ArgumentParser(
        description="Distributed stress test for RadixShmem KV cache cluster.\n\n"
                    "By default, spawns all ranks locally via multiprocessing.\n"
                    "Use --rank to run a single rank (for multi-machine deployment).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = p.add_argument_group("cluster")
    g.add_argument("--rank", type=int, default=-1,
                   help="Run a single rank (for distributed deploy). "
                        "Omit to launch all ranks locally.")
    g.add_argument("--world-size", type=int, default=1)
    g.add_argument("--master-addr", default="127.0.0.1")
    g.add_argument("--master-port", type=int, default=18500)
    g.add_argument("--dispatch-port", type=int, default=18700)
    g.add_argument("--index-dev", default="")
    g.add_argument("--gid-idx", type=int, default=3)
    g.add_argument("--bootstrap-timeout", type=int, default=60)

    g = p.add_argument_group("tree")
    g.add_argument("--max-nodes", type=int, default=0)
    g.add_argument("--max-blocks", type=int, default=500000)
    g.add_argument("--block-size", type=int, default=16)
    g.add_argument("--register-chunk-size", type=int, default=128)
    g.add_argument("--background-evict-ratio", type=float, default=0.05)

    g = p.add_argument_group("workload")
    g.add_argument("--qps", type=float, default=100.0)
    g.add_argument("--isl-min", type=int, default=32)
    g.add_argument("--isl-max", type=int, default=512)
    g.add_argument("--osl-min", type=int, default=16)
    g.add_argument("--osl-max", type=int, default=256)
    g.add_argument("--sys-prompt", type=int, default=256)
    g.add_argument("--max-turns", type=int, default=10)
    g.add_argument("--num-templates", type=int, default=5)
    g.add_argument("--conv-pool", type=int, default=200)
    g.add_argument("--workers-per-rank", type=int, default=1,
                   help="Number of worker threads per rank (default: 1)")

    g = p.add_argument_group("timing simulation")
    g.add_argument("--transfer-ms", type=float, default=0.01)
    g.add_argument("--prefill-ms", type=float, default=0.05)
    g.add_argument("--decode-ms", type=float, default=0.02)

    g = p.add_argument_group("test control")
    g.add_argument("--duration", type=int, default=0,
                   help="Test duration in seconds, 0=infinite (default: 0)")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--report-interval", type=int, default=10)

    args = p.parse_args()

    is_single_rank = (args.rank >= 0)

    cfg = StressConfig(
        rank=args.rank, world_size=args.world_size,
        master_addr=args.master_addr, master_port=args.master_port,
        dispatch_port=args.dispatch_port,
        index_dev=args.index_dev, gid_idx=args.gid_idx,
        bootstrap_timeout=args.bootstrap_timeout,
        max_nodes=args.max_nodes, max_blocks=args.max_blocks,
        block_size=args.block_size, register_chunk_size=args.register_chunk_size,
        background_evict_ratio=args.background_evict_ratio,
        target_qps=args.qps,
        isl_min=args.isl_min, isl_max=args.isl_max,
        osl_min=args.osl_min, osl_max=args.osl_max,
        system_prompt_length=args.sys_prompt,
        max_turns=args.max_turns, num_templates=args.num_templates,
        conv_pool_size=args.conv_pool, workers_per_rank=args.workers_per_rank,
        transfer_ms_per_block=args.transfer_ms,
        prefill_ms_per_block=args.prefill_ms,
        decode_ms_per_token=args.decode_ms,
        duration_seconds=args.duration, seed=args.seed,
        report_interval=args.report_interval,
    )

    cfg.system_prompt_length = max(cfg.block_size,
                                   (cfg.system_prompt_length // cfg.block_size) * cfg.block_size)

    return cfg, is_single_rank


# ======================================================================
# Main
# ======================================================================

def main():
    cfg, is_single_rank = parse_args()

    print(f"=== Distributed Stress Test ===\n")
    print(f"  World size:       {cfg.world_size}")
    print(f"  Workers/rank:     {cfg.workers_per_rank}")
    print(f"  Master:           {cfg.master_addr}:{cfg.master_port}")
    print(f"  Dispatch port:    {cfg.dispatch_port}")
    print(f"  Max blocks:       {cfg.max_blocks}")
    print(f"  Block size:       {cfg.block_size} tokens")
    print(f"  Target QPS:       {cfg.target_qps:.1f}")
    print(f"  ISL range:        {cfg.isl_min}-{cfg.isl_max} tokens")
    print(f"  OSL range:        {cfg.osl_min}-{cfg.osl_max} tokens")
    print(f"  System prompt:    {cfg.system_prompt_length} tokens")
    print(f"  Max turns:        {cfg.max_turns}")
    print(f"  Templates:        {cfg.num_templates}")
    print(f"  Conv pool:        {cfg.conv_pool_size}")
    dur = f"{cfg.duration_seconds}s" if cfg.duration_seconds > 0 else "infinite (Ctrl+C)"
    print(f"  Duration:         {dur}")
    print(f"  Seed:             {cfg.seed}")
    print()

    if is_single_rank:
        # Distributed deploy: run a single rank
        if cfg.rank == 0:
            print(f"  Mode:             single-rank (rank 0, scheduler)")
        else:
            print(f"  Mode:             single-rank (rank {cfg.rank}, "
                  f"config will be synced from rank 0)")
        print()
        run_rank(cfg.rank, cfg)
    elif cfg.world_size == 1:
        # Standalone: single process
        print(f"  Mode:             standalone (1 process)")
        print()
        run_rank(0, cfg)
    else:
        # Local simulation: spawn all ranks via multiprocessing
        print(f"  Mode:             local simulation ({cfg.world_size} processes)")
        print()

        processes = []
        for r in range(cfg.world_size):
            p = multiprocessing.Process(target=run_rank, args=(r, cfg),
                                        name=f"rank-{r}")
            p.start()
            processes.append(p)

        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            print("\n[launcher] Ctrl+C received, waiting for processes...")
            for p in processes:
                p.join(timeout=10)
            for p in processes:
                if p.is_alive():
                    p.terminate()

        exit_codes = [p.exitcode or 0 for p in processes]
        if any(c != 0 for c in exit_codes):
            print(f"[launcher] Some ranks exited with errors: {exit_codes}")
            sys.exit(1)
        else:
            print("[launcher] All ranks completed successfully.")
            sys.exit(0)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
