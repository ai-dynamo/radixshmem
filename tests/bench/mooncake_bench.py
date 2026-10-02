#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""mooncake_bench.py — application-level mooncake-trace stress test for RadixShmem.

Counterpart to dynamo's `mooncake_bench.rs`, but exercising the *Python
integration layer* (shmradix._core.RadixServer / RadixClient) — the same surface the
vLLM OffloadingConnector integration will ultimately call.

MVP scope: single process holding both server (writer) and client (reader/writer).
Measures throughput, latency (query / insert split), and the prefix-hit three-way
split (full / partial / miss) that is RadixShmem's distinguishing strength.

Modes:
  single     one config, one run, print a full report line
  compare    several knob combos (lock / update_meta / block_size) side by side
  sweep      scan offered load (trace timeline compression) with early-stop
  repeated   same config N trials, print inter-trial percentiles

The C++ benches (bench_query / bench_insert / stress_test_llm) measure native
limits; this measures the realistic Python-integration-layer behaviour. They are
complementary, not a replacement.

API notes (verified against python/bindings.cpp):
  * Exports are RadixServer / RadixClient (NOT TreeServer / TreeClient).
  * client.insert(hashes, slots, start, auto_recycle) takes externally-
    allocated slots (mandatory), positionally bound to hashes[start:].
    Store = allocate_slots(shortfall) [LRU-evicts when the mempool is full] +
    insert(hashes, slots, start=matched). The shortfall is len(hashes) - matched,
    which the preceding query already reports as total_hit_length.
  * Pool metrics are already exposed: server.mempool_total/used/free(),
    server.total_radix_nodes(), server.total_blocks() — no pool_stats() patch
    needed. The engine exposes no eviction counter, so the CSV reports
    cumulative-inserted-minus-resident as an eviction proxy.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from time import perf_counter, perf_counter_ns

import numpy as np

try:
    import shmradix
except ImportError as e:  # pragma: no cover
    sys.stderr.write(
        "ERROR: cannot import shmradix. Run inside the radixshmem_vllm container "
        "or with python/ on PYTHONPATH.\n"
    )
    raise

MASK64 = 0xFFFFFFFFFFFFFFFF
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3


# ---------------------------------------------------------------------------
# Trace parsing  (counterpart to dynamo process_mooncake_trace)
# ---------------------------------------------------------------------------

@dataclass
class Request:
    worker_id: int
    hashes: np.ndarray      # uint64 block-hash sequence
    arrival_ms: float       # absolute trace timestamp (ms), used by trace-driven


def _fold_group(ids: list[int]) -> int:
    """Deterministic FNV-1a fold of a group of block ids into one uint64.

    Position-independent over the group contents only, so two requests sharing a
    prefix (identical hash_ids in aligned positions) fold to identical grouped
    block hashes — preserving prefix-sharing under block_size>1. Stable across
    runs (unlike Python's randomized hash()).
    """
    h = FNV_OFFSET
    for v in ids:
        x = v & MASK64
        for _ in range(8):  # mix 8 little-endian bytes
            h = ((h ^ (x & 0xFF)) * FNV_PRIME) & MASK64
            x >>= 8
    return h


def load_trace(
    path: str,
    block_size: int,
    length_factor: float,
    duplication_factor: int,
    num_workers: int,
    seed: int,
) -> list[Request]:
    """Parse a mooncake JSONL trace into scaled Request objects.

    hash_ids are already block-granular (one id ~= 512 tokens), so there is no
    token->block splitting. With block_size>1 we fold every `block_size`
    consecutive ids into one combined block hash (mirrors offloaded_block_size =
    gpu_block_size x factor); the tail that does not fill a block is dropped,
    matching vLLM.
    """
    rng = np.random.default_rng(seed)
    raw: list[tuple[int, list[int]]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ids = d.get("hash_ids")
            if not ids:
                continue
            ts = float(d.get("timestamp", 0))
            raw.append((ts, ids))

    requests: list[Request] = []
    for dup in range(max(1, duplication_factor)):
        # Offset duplicated copies into a disjoint id space so duplication injects
        # *conflict* on the timeline but each copy is independently addressable
        # only when dup==0; matching dynamo, dup copies reuse the SAME ids to
        # create cross-copy prefix collisions, so we keep ids identical.
        for ts, ids in raw:
            # length scaling: truncate or replicate the block list
            if length_factor != 1.0:
                target = max(1, int(round(len(ids) * length_factor)))
                if target <= len(ids):
                    ids2 = ids[:target]
                else:
                    reps = (target + len(ids) - 1) // len(ids)
                    ids2 = (ids * reps)[:target]
            else:
                ids2 = ids

            # group into blocks of `block_size`
            if block_size <= 1:
                blocks = [v & MASK64 for v in ids2]
            else:
                n_full = len(ids2) // block_size
                blocks = [
                    _fold_group(ids2[i * block_size:(i + 1) * block_size])
                    for i in range(n_full)
                ]
            if not blocks:
                continue

            # worker assignment: hash first block into one of N virtual workers
            wid = int(blocks[0] % num_workers) if num_workers > 1 else 0
            requests.append(
                Request(
                    worker_id=wid,
                    hashes=np.asarray(blocks, dtype=np.uint64),
                    arrival_ms=ts,
                )
            )

    # sort by arrival so trace-driven replay is monotonic
    requests.sort(key=lambda r: r.arrival_ms)
    return requests


# ---------------------------------------------------------------------------
# Latency collection
# ---------------------------------------------------------------------------

class Latency:
    __slots__ = ("samples",)

    def __init__(self) -> None:
        self.samples: list[int] = []   # nanoseconds

    def add(self, ns: int) -> None:
        self.samples.append(ns)

    def pct_us(self, p: float) -> float:
        if not self.samples:
            return 0.0
        s = sorted(self.samples)
        k = min(len(s) - 1, int(p / 100.0 * len(s)))
        return s[k] / 1000.0


# ---------------------------------------------------------------------------
# Background shm-state sampler
# ---------------------------------------------------------------------------

class ShmSampler(threading.Thread):
    """Sample pool state every interval_ms; write a time-series CSV."""

    def __init__(self, server, interval_ms: int, cum_blocks: list[int]):
        super().__init__(daemon=True)
        self.server = server
        self.interval = interval_ms / 1000.0
        self.cum_blocks = cum_blocks          # shared 1-elem list: blocks ever inserted
        self.rows: list[tuple] = []
        self._stop_evt = threading.Event()
        self._t0 = perf_counter()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            elapsed_ms = int((perf_counter() - self._t0) * 1000)
            used = self.server.mempool_used()
            free = self.server.mempool_free()
            total = self.server.mempool_total()
            nodes = self.server.total_radix_nodes()
            used_pct = (100.0 * used / total) if total else 0.0
            # eviction proxy: blocks ever inserted but no longer resident
            evicted = max(0, self.cum_blocks[0] - used)
            self.rows.append((elapsed_ms, used, free, round(used_pct, 2), nodes, evicted))
            self._stop_evt.wait(self.interval)

    def stop(self) -> None:
        self._stop_evt.set()

    def write_csv(self, path: str) -> None:
        with open(path, "w") as f:
            f.write("elapsed_ms,used_blocks,free_blocks,used_pct,node_count,evict_proxy\n")
            for r in self.rows:
                f.write(",".join(str(x) for x in r) + "\n")
        print(f"  shm metrics CSV -> {path}")


# ---------------------------------------------------------------------------
# Benchmark config + result
# ---------------------------------------------------------------------------

_EMPTY_I32 = np.empty(0, dtype=np.int32)


def store(client, h, matched, lock):
    """Store a request's blocks: allocate the shortfall past the already-known
    matched prefix (LRU-evicts when full), insert (slots bind hashes[matched:]
    via start=matched, insert auto-recycles anything unused), and optionally pin
    the whole prefix against eviction: query(lock=True) whose finalize handle
    is dropped, so the pin stays for the run (old lock=True behavior)."""
    need = len(h) - matched
    slots = client.allocate_slots(need) if need > 0 else _EMPTY_I32
    ins = client.insert(h, slots, start=matched, auto_recycle=True)
    if lock:
        client.query(h, local_only=True, lock=True, update_meta=False)
    return ins


@dataclass
class BenchConfig:
    label: str
    block_size: int = 1
    lock: bool = False
    update_meta: bool = True
    local_only: bool = True


@dataclass
class BenchResult:
    label: str
    duration_s: float
    ops: int
    ops_per_s: float
    offered_ops_per_s: float
    block_throughput: float
    offered_block_throughput: float
    full_pct: float
    partial_pct: float
    miss_pct: float
    q_p50: float
    q_p90: float
    q_p99: float
    i_p50: float
    i_p90: float
    i_p99: float
    kept_up: bool
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core workload
# ---------------------------------------------------------------------------

def run_workload(
    requests: list[Request],
    cfg: BenchConfig,
    mode: str,
    duration_ms: int,
    shm_name: str,
    max_blocks: int,
    data_pool_ratio: float,
    sample_interval_ms: int,
    csv_path: str | None,
) -> BenchResult:
    """Build a fresh shm region, replay the trace, return metrics.

    Each request: query (lookup) then insert (store) — mirroring the KV-cache
    pattern (look up cached prefix, then store the request's blocks for reuse).
    """
    # IMPORTANT: max_nodes must be sized generously. The default (max_nodes=0)
    # derives node capacity from max_blocks, which badly under-provisions for
    # mooncake traces (they create ~1 radix node per ~10 blocks, far denser than
    # the default heuristic assumes). Overflowing the node pool currently
    # *segfaults* the engine instead of returning NODE_POOL_FULL, so we give it
    # 4x max_blocks worth of nodes — comfortably above observed node counts.
    shm_cfg = shmradix._core.ShmConfig(
        max_nodes=max(1 << 20, max_blocks * 4),
        max_blocks=max_blocks,
        block_size=1,               # we feed precomputed block hashes (1 hash = 1 block)
        data_pool_ratio=data_pool_ratio,
    )
    server = shmradix._core.RadixServer(shm_name, shm_cfg)
    client = shmradix._core.RadixClient(shm_name)

    cum_blocks = [0]
    sampler = None
    if csv_path:
        sampler = ShmSampler(server, sample_interval_ms, cum_blocks)
        sampler.start()

    q_lat, i_lat = Latency(), Latency()
    full = partial = miss = 0
    total_blocks = 0
    ops = 0
    kept_up = True

    n_req = len(requests)
    if n_req == 0:
        raise RuntimeError("trace produced 0 requests")

    # offered-rate model
    span_ms = requests[-1].arrival_ms - requests[0].arrival_ms
    base_ms = requests[0].arrival_ms

    if mode == "trace-driven":
        # compress/stretch the trace timeline to fit duration_ms
        time_scale = (duration_ms / span_ms) if span_ms > 0 else 0.0
        offered_ops = n_req / (duration_ms / 1000.0)
        t0 = perf_counter()
        for req in requests:
            target = t0 + (req.arrival_ms - base_ms) * time_scale / 1000.0
            now = perf_counter()
            if now < target:
                time.sleep(target - now)
            elif now - target > 0.05:   # >50ms behind => falling behind
                kept_up = False

            h = req.hashes
            t = perf_counter_ns()
            qr = client.query(h, mask=shmradix.COMPONENT_MASK_FULL,
                              local_only=cfg.local_only, lock=cfg.lock,
                              update_meta=cfg.update_meta)
            q_lat.add(perf_counter_ns() - t)
            n = qr.common_hit
            if n == len(h):
                full += 1
            elif n > 0:
                partial += 1
            else:
                miss += 1

            t = perf_counter_ns()
            store(client, h, n, cfg.lock)
            i_lat.add(perf_counter_ns() - t)

            total_blocks += len(h)
            cum_blocks[0] += len(h)
            ops += 1
        dur = perf_counter() - t0

    elif mode == "closed-loop":
        # saturation: loop over the trace as fast as possible for duration_ms
        deadline = perf_counter() + duration_ms / 1000.0
        t0 = perf_counter()
        i = 0
        while perf_counter() < deadline:
            req = requests[i % n_req]
            i += 1
            h = req.hashes
            t = perf_counter_ns()
            qr = client.query(h, mask=shmradix.COMPONENT_MASK_FULL,
                              local_only=cfg.local_only, lock=cfg.lock,
                              update_meta=cfg.update_meta)
            q_lat.add(perf_counter_ns() - t)
            n = qr.common_hit
            if n == len(h):
                full += 1
            elif n > 0:
                partial += 1
            else:
                miss += 1

            t = perf_counter_ns()
            store(client, h, n, cfg.lock)
            i_lat.add(perf_counter_ns() - t)

            total_blocks += len(h)
            cum_blocks[0] += len(h)
            ops += 1
        dur = perf_counter() - t0
        offered_ops = ops / dur     # offered == achieved (no pacing)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    if sampler:
        sampler.stop()
        sampler.join(timeout=2.0)
        if csv_path:
            sampler.write_csv(csv_path)

    return BenchResult(
        label=cfg.label,
        duration_s=dur,
        ops=ops,
        ops_per_s=ops / dur if dur else 0.0,
        offered_ops_per_s=offered_ops,
        block_throughput=total_blocks / dur if dur else 0.0,
        offered_block_throughput=offered_ops * (total_blocks / ops if ops else 0.0),
        full_pct=100.0 * full / ops if ops else 0.0,
        partial_pct=100.0 * partial / ops if ops else 0.0,
        miss_pct=100.0 * miss / ops if ops else 0.0,
        q_p50=q_lat.pct_us(50), q_p90=q_lat.pct_us(90), q_p99=q_lat.pct_us(99),
        i_p50=i_lat.pct_us(50), i_p90=i_lat.pct_us(90), i_p99=i_lat.pct_us(99),
        kept_up=kept_up,
        extra={
            "resident_blocks": server.mempool_used(),
            "node_count": server.total_radix_nodes(),
        },
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

HEADER = (
    f"{'config':<22}{'dur(s)':>8}{'ops':>10}{'ops/s':>11}"
    f"{'full%':>8}{'part%':>8}{'miss%':>8}"
    f"{'q_p50':>8}{'q_p90':>8}{'q_p99':>9}{'i_p50':>8}{'i_p99':>9}"
)
RULE = "-" * len(HEADER)


def print_row(r: BenchResult) -> None:
    print(
        f"{r.label:<22}{r.duration_s:>8.2f}{r.ops:>10}{r.ops_per_s:>11.0f}"
        f"{r.full_pct:>8.1f}{r.partial_pct:>8.1f}{r.miss_pct:>8.1f}"
        f"{r.q_p50:>8.2f}{r.q_p90:>8.2f}{r.q_p99:>9.2f}{r.i_p50:>8.2f}{r.i_p99:>9.2f}"
    )
    if not r.kept_up:
        print(
            "  WARNING: benchmarker fell >5% behind the issue rate; "
            "rerun with a larger --benchmark-duration-ms."
        )


def print_trace_stats(requests: list[Request]) -> None:
    lens = sorted(len(r.hashes) for r in requests)
    n = len(lens)
    def pct(p):
        return lens[min(n - 1, int(p / 100 * n))]
    span = (requests[-1].arrival_ms - requests[0].arrival_ms) / 1000.0
    print(f"  trace: {n} requests, timeline span {span:.1f}s")
    print(
        f"  block-length: p50={pct(50)} p90={pct(90)} p99={pct(99)} "
        f"max={lens[-1]} mean={sum(lens)/n:.1f}"
    )
    n1 = sum(1 for x in lens if x <= 1)
    print(f"  single-block (<=1) requests: {n1} ({100.0*n1/n:.1f}%)")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

COMPARE_PRESETS = {
    "baseline":        BenchConfig("baseline", block_size=1, lock=False, update_meta=True),
    "lock-on":         BenchConfig("lock-on", block_size=1, lock=True, update_meta=True),
    "update-meta-off": BenchConfig("update-meta-off", block_size=1, lock=False, update_meta=False),
    "block-4":         BenchConfig("block-4", block_size=4, lock=False, update_meta=True),
    "block-8":         BenchConfig("block-8", block_size=8, lock=False, update_meta=True),
}


def result_to_json(r: BenchResult) -> dict:
    return {
        "duration_ms": int(r.duration_s * 1000),
        "offered_ops_throughput": r.offered_ops_per_s,
        "ops_throughput": r.ops_per_s,
        "offered_block_throughput": r.offered_block_throughput,
        "block_throughput": r.block_throughput,
        "latency_p99_us": r.q_p99,
        "full_pct": r.full_pct,
        "partial_pct": r.partial_pct,
        "miss_pct": r.miss_pct,
    }


def shm_name_for(base: str, suffix: str) -> str:
    return f"{base}_{suffix}"


def main() -> None:
    ap = argparse.ArgumentParser(description="mooncake trace bench for RadixShmem")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--block-size", type=int, default=1,
                    help="application block grouping of hash_ids (default 1 = 1:1)")
    ap.add_argument("--max-blocks", type=int, default=200000)
    ap.add_argument("--data-pool-ratio", type=float, default=4.0)
    ap.add_argument("--mode", choices=["trace-driven", "closed-loop"], default="closed-loop")
    ap.add_argument("--trace-length-factor", type=float, default=1.0)
    ap.add_argument("--trace-duplication-factor", type=int, default=1)
    ap.add_argument("--num-unique-inference-workers", type=int, default=1)
    ap.add_argument("--benchmark-duration-ms", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shm-name", default="/mooncake_bench")
    ap.add_argument("--shm-metrics-csv", default="")
    ap.add_argument("--shm-metrics-interval-ms", type=int, default=200)
    ap.add_argument("--sweep-json-output", default="sweep_results.json")
    ap.add_argument("--benchmark-runs", type=int, default=1)
    ap.add_argument("--compare", default="",
                    help="comma list of presets: " + ",".join(COMPARE_PRESETS))

    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("single")
    sub.add_parser("compare")
    sub.add_parser("sweep")
    sub.add_parser("repeated")
    args = ap.parse_args()

    def build_requests(block_size: int) -> list[Request]:
        return load_trace(
            args.trace, block_size, args.trace_length_factor,
            args.trace_duplication_factor, args.num_unique_inference_workers,
            args.seed,
        )

    print(f"=== mooncake_bench (mode={args.mode}, trace={args.trace}) ===\n")

    # ---------------- single ----------------
    if args.cmd == "single":
        reqs = build_requests(args.block_size)
        print_trace_stats(reqs)
        print()
        cfg = BenchConfig("single", block_size=args.block_size)
        print(HEADER); print(RULE)
        csv = args.shm_metrics_csv or None
        r = run_workload(
            reqs, cfg, args.mode, args.benchmark_duration_ms,
            shm_name_for(args.shm_name, "single"), args.max_blocks,
            args.data_pool_ratio, args.shm_metrics_interval_ms, csv,
        )
        print_row(r)
        print(f"\n  resident_blocks={r.extra['resident_blocks']} "
              f"node_count={r.extra['node_count']} "
              f"block_throughput={r.block_throughput:.0f} blk/s")

    # ---------------- compare ----------------
    elif args.cmd == "compare":
        names = [s.strip() for s in (args.compare or
                 "baseline,lock-on,update-meta-off,block-4,block-8").split(",") if s.strip()]
        # build per-block-size request sets once
        cache: dict[int, list[Request]] = {}
        print(HEADER); print(RULE)
        for name in names:
            cfg = COMPARE_PRESETS.get(name)
            if cfg is None:
                print(f"  (skip unknown preset {name})"); continue
            if cfg.block_size not in cache:
                cache[cfg.block_size] = build_requests(cfg.block_size)
            reqs = cache[cfg.block_size]
            csv = (args.shm_metrics_csv.replace(".csv", f"_{name}.csv")
                   if args.shm_metrics_csv else None)
            r = run_workload(
                reqs, cfg, args.mode, args.benchmark_duration_ms,
                shm_name_for(args.shm_name, name), args.max_blocks,
                args.data_pool_ratio, args.shm_metrics_interval_ms, csv,
            )
            print_row(r)

    # ---------------- sweep ----------------
    elif args.cmd == "sweep":
        reqs = build_requests(args.block_size)
        print_trace_stats(reqs); print()
        cfg = BenchConfig("sweep", block_size=args.block_size)
        # sweep offered load by compressing the timeline: shorter duration = higher rate
        durations = [8000, 4000, 2000, 1000, 500, 250]
        print(HEADER); print(RULE)
        steps = []
        for dur in durations:
            r = run_workload(
                reqs, cfg, "trace-driven", dur,
                shm_name_for(args.shm_name, f"sw{dur}"), args.max_blocks,
                args.data_pool_ratio, args.shm_metrics_interval_ms, None,
            )
            r.label = f"sweep-{dur}ms"
            print_row(r)
            steps.append(result_to_json(r))
            # early stop: offered throughput >5x achieved (saturated)
            if r.offered_block_throughput > r.block_throughput * 5.0:
                print("  Early stop: offered throughput >5x achieved")
                break
        out = {"sweep": steps}
        with open(args.sweep_json_output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSweep results saved to {args.sweep_json_output}")

    # ---------------- repeated ----------------
    elif args.cmd == "repeated":
        reqs = build_requests(args.block_size)
        cfg = BenchConfig("repeated", block_size=args.block_size)
        print(HEADER); print(RULE)
        trials: list[BenchResult] = []
        for k in range(args.benchmark_runs):
            csv = (args.shm_metrics_csv.replace(".csv", f"_run{k+1}.csv")
                   if args.shm_metrics_csv else None)
            r = run_workload(
                reqs, cfg, args.mode, args.benchmark_duration_ms,
                shm_name_for(args.shm_name, f"r{k}"), args.max_blocks,
                args.data_pool_ratio, args.shm_metrics_interval_ms, csv,
            )
            r.label = f"trial-{k+1}"
            print_row(r)
            trials.append(r)
        # inter-trial percentiles
        opss = [t.ops_per_s for t in trials]
        p99s = [t.q_p99 for t in trials]
        print(RULE)
        print(f"  ops/s     min={min(opss):.0f} median={statistics.median(opss):.0f} "
              f"max={max(opss):.0f}")
        print(f"  q_p99_us  min={min(p99s):.2f} median={statistics.median(p99s):.2f} "
              f"max={max(p99s):.2f}")


if __name__ == "__main__":
    main()
