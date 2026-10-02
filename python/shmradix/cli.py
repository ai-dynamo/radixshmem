# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""radix-server: run a RadixServer until SIGINT / SIGTERM.

The server starts with what the operator knows (identity, cluster, NICs, the
SlotStore budget and how it is split between the pools) and waits for a client
to bring the geometry, tokens per block and bytes per slot:
``shmradix.RadixClient(name, shmradix.Geometry(...))``. In a cluster the first
node configured spreads its geometry to the others through etcd.

Index only (no data plane; slot ids belong to the client's own storage, whose
geometry brings the slot counts):
    radix-server --name /kv

Index + data plane (--data-bytes selects it; the server owns a SlotStore and
sizes the pools from the budget: SWA and Mamba take their ratio, FULL the rest):
    radix-server --name /kv --data-bytes 64G
    radix-server --name /kv --data-bytes 64G --swa-ratio 0.15 --mamba-ratio 0.05

Distributed (one per node, discovery via etcd, rank assigned by etcd):
    radix-server --name /kv --data-bytes 64G --expected-min-nodes 3 \\
                 --registry etcd://10.0.0.1:2379 --rpc-address 10.0.0.2 \\
                 --transfer-dev mlx5_0,mlx5_1

Byte sizes are the client's business; budgets and ratios may differ per node,
the slot shape may not (the servers cross-check it through etcd).
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time

from shmradix import _geometry
from shmradix.server import ClusterConfig, RadixServer, ServerConfig

_SIZE = {"": 1, "B": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}

# argparse dest -> (ServerConfig field, flag). All default to None so that a
# flag given without --data-bytes can be reported instead of ignored.
_DATA_FLAGS = {
    "swa_ratio": ("swa_ratio", "--swa-ratio"),
    "mamba_ratio": ("mamba_ratio", "--mamba-ratio"),
    "slot_align": ("slot_align", "--slot-align"),
    "data_name": ("data_name", "--data-name"),
    "no_prefault": ("prefault", "--no-prefault"),
    "transfer_backend": ("transfer_backend", "--transfer-backend"),
    "transfer_protocol": ("transfer_protocol", "--transfer-protocol"),
    "transfer_dev": ("transfer_devices", "--transfer-dev"),
    "transfer_ip": ("transfer_ip", "--transfer-ip"),
    "transfer_port": ("transfer_port", "--transfer-port"),
    "transfer_metadata": ("transfer_metadata", "--transfer-metadata"),
    "max_inflight": ("max_inflight", "--max-inflight"),
    "max_pending_jobs": ("max_pending_jobs", "--max-pending-jobs"),
    "job_ttl": ("job_ttl_s", "--job-ttl"),
}


def parse_size(text: str) -> int:
    """'64G', '512M', '1073741824' -> bytes (binary units)."""
    t = str(text).strip().upper()
    for suffix in ("IB", "B", "I"):
        if t.endswith(suffix) and len(t) > len(suffix):
            t = t[:-len(suffix)]
    unit = t[-1] if t and t[-1] in "KMGT" else ""
    try:
        return int(float(t[:-1] if unit else t) * _SIZE[unit])
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"bad size {text!r}") from e


def fmt_size(nbytes: int) -> str:
    for shift, unit in ((40, "TB"), (30, "GB")):
        if nbytes >= 1 << shift:
            return f"{nbytes / (1 << shift):.2f} {unit}"
    return f"{nbytes / (1 << 20):.2f} MB"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="radix-server", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("index (both modes)")
    g.add_argument("--name", default="/shmradix", help="shm name / prefix (default: /shmradix)")
    g.add_argument("--max-nodes", type=int, default=0, help="radix node pool capacity, 0 = auto")
    g.add_argument("--register-chunk-tokens", type=int, default=4096,
                   help="RHT registration granularity in tokens unless the client's geometry "
                        "brings its own (default: 4096)")
    g.add_argument("--background-evict-ratio", type=float, default=0.05)
    g.add_argument("--data-pool-ratio", type=float, default=4.0,
                   help="index DataPool = full slots x ratio x bytes per block (default: 4.0)")
    g.add_argument("--evict-policy", default="lru", choices=["lru"])
    g.add_argument("--hugepage-path", default="",
                   help="hugetlbfs mount for the index and the slot store (default: plain /dev/shm)")

    g = p.add_argument_group("data plane (--data-bytes selects the mode; the others need it)")
    g.add_argument("--data-bytes", type=parse_size, default=None,
                   help="slot store budget, e.g. 64G; absent = index only")
    g.add_argument("--swa-ratio", type=float, default=None,
                   help="share of the budget for SWA slots when the model has SWA layers (default: 0)")
    g.add_argument("--mamba-ratio", type=float, default=None,
                   help="share of the budget for Mamba checkpoint slots (default: 0); "
                        "FULL slots fill what the aux pools leave")
    g.add_argument("--slot-align", type=int, default=None,
                   help="slot stride alignment unless the geometry brings its own (default: 4096)")
    g.add_argument("--data-name", default=None, help="SlotStore shm name (default: <name>_data)")
    g.add_argument("--no-prefault", action="store_const", const=True, default=None,
                   help="do not touch every slot page at creation")
    g.add_argument("--transfer-backend", default=None, choices=["mooncake", "nixl"],
                   help="(default: mooncake)")
    g.add_argument("--transfer-protocol", default=None, choices=["rdma", "tcp"], help="(default: rdma)")
    g.add_argument("--transfer-dev", action="append", default=None,
                   help="HCA allowlist of the transfer engine, repeatable or comma-separated (default: all)")
    g.add_argument("--transfer-ip", default=None, help="data-plane IP (default: the rpc address)")
    g.add_argument("--transfer-port", type=int, default=None,
                   help="(default: 0 = engine picks; nixl backend requires a nonzero port)")
    g.add_argument("--transfer-metadata", default=None, help="(default: P2PHANDSHAKE)")
    g.add_argument("--max-inflight", type=int, default=None,
                   help="batches in flight at the engine (default: 256)")
    g.add_argument("--max-pending-jobs", type=int, default=None,
                   help="queued + running + uncollected jobs before Submit is refused (default: 4096)")
    g.add_argument("--job-ttl", type=float, default=None,
                   help="seconds an uncollected job is kept (default: 60)")

    g = p.add_argument_group("server")
    g.add_argument("--endpoint", default="",
                   help="gRPC endpoint clients connect to (default: unix:///dev/shm/<name>.sock)")
    g.add_argument("--rpc-workers", type=int, default=32,
                   help="gRPC threads; one blocks per client with jobs in flight (default: 32)")
    g.add_argument("--interval", type=int, default=5, help="stats interval in seconds, 0 = silent")

    g = p.add_argument_group("distributed (--expected-min-nodes or --num-rht-shards > 1)")
    g.add_argument("--expected-min-nodes", type=int, default=0)
    g.add_argument("--settle-ms", type=int, default=500)
    g.add_argument("--registry", default="etcd://127.0.0.1:2379")
    g.add_argument("--cluster-id", default="default")
    g.add_argument("--node-name", default="")
    g.add_argument("--rpc-address", default="", help="bootstrap IP peers dial; 0.0.0.0 on one host")
    g.add_argument("--rpc-interface", default="", help="NIC to resolve the bootstrap IP from")
    g.add_argument("--rpc-port", type=int, default=0)
    g.add_argument("--index-dev", default="",
                   help="HCA of the index control plane (default: first available); "
                        "the data plane's is --transfer-dev")
    g.add_argument("--gid-idx", type=int, default=3)
    g.add_argument("--bootstrap-timeout", type=int, default=60)
    g.add_argument("--lease-ttl", type=int, default=30,
                   help="etcd lease TTL: how long a killed node's keys outlive it")
    g.add_argument("--rht-transport", default="xrc", choices=["xrc", "dc"],
                   help="client -> RHT shard holder QPs")
    g.add_argument("--peer-index-transport", default="xrc", choices=["xrc", "dc"],
                   help="remote walks into peer trees")
    g.add_argument("--num-rht-shards", type=int, default=0)
    g.add_argument("--rht-shard-holders", default="", help="comma-separated ranks")
    g.add_argument("--rht-slots", type=int, default=1, choices=[1, 2, 4, 8])
    g.add_argument("--enable-remote-insert", action="store_true")
    g.add_argument("--enable-remote-query", action="store_true")
    g.add_argument("--zmq-listen-port", type=int, default=0)
    g.add_argument("--remote-op-transport", default="zmq", choices=["zmq", "dc"])
    return p.parse_args(argv)


def build_config(a: argparse.Namespace) -> ServerConfig:
    """ValueError on a flag combination that does not describe one server."""
    given = {dest: getattr(a, dest) for dest in _DATA_FLAGS if getattr(a, dest) is not None}
    if a.data_bytes is None and given:
        flags = ", ".join(_DATA_FLAGS[d][1] for d in given)
        raise ValueError(f"{flags} need --data-bytes; without it the server is index only")
    kw = {_DATA_FLAGS[d][0]: v for d, v in given.items()}
    if "prefault" in kw:
        kw["prefault"] = not kw["prefault"]
    if "transfer_devices" in kw:
        kw["transfer_devices"] = [d.strip() for v in kw["transfer_devices"] for d in v.split(",") if d.strip()]
    cluster = ClusterConfig(
        expected_min_nodes=a.expected_min_nodes, settle_ms=a.settle_ms, registry=a.registry,
        cluster_id=a.cluster_id, node_name=a.node_name, rpc_address=a.rpc_address,
        rpc_interface=a.rpc_interface, rpc_port=a.rpc_port, index_dev=a.index_dev, gid_idx=a.gid_idx,
        bootstrap_timeout_sec=a.bootstrap_timeout, lease_ttl_sec=a.lease_ttl,
        rht_transport=a.rht_transport,
        peer_index_transport=a.peer_index_transport, num_rht_shards=a.num_rht_shards,
        rht_shard_holders=[int(x) for x in a.rht_shard_holders.split(",") if x.strip()],
        rht_slots_per_bucket=a.rht_slots, enable_remote_insert=a.enable_remote_insert,
        enable_remote_query=a.enable_remote_query, zmq_listen_port=a.zmq_listen_port,
        remote_op_transport=a.remote_op_transport)
    return ServerConfig(
        name=a.name, data_bytes=a.data_bytes or 0, hugepage_path=a.hugepage_path,
        max_nodes=a.max_nodes, register_chunk_tokens=a.register_chunk_tokens,
        background_evict_ratio=a.background_evict_ratio, data_pool_ratio=a.data_pool_ratio,
        evict_policy=a.evict_policy, endpoint=a.endpoint, rpc_workers=a.rpc_workers,
        cluster=cluster, **kw)


def _stats_line(srv: RadixServer) -> str:
    s = srv.stats()
    if "total_blocks" not in s:
        return f"[{s['mode']}]"
    tag = f"[rank {s['rank']}]" if s["world_size"] > 1 else "[stats]"
    line = (f"{tag} nodes={s['total_radix_nodes']}  blocks={s['total_blocks']}  "
            f"data_pool={s['data_pool_utilization'] * 100:.1f}%  ins_ok={s['insert_success']}")
    if s["mempool_total"]:
        line += f"  mempool={s['mempool_used']}/{s['mempool_total']}"
    if s["insert_errors"]:
        line += f"  ins_err={s['insert_errors']}"
    if s["peer_down_events"]:
        line += (f"  peer_down={s['peer_down_events']}  rht_drop={s['rht_writes_dropped']}"
                 f"/{s['rht_removes_dropped']}  lookup_skip={s['rht_lookups_skipped']}"
                 f"  walk_skip={s['walks_skipped']}")
    if s["transfers"] or s["pending_jobs"]:
        line += (f"  xfers={s['transfers']}({s['transfer_failures']} failed, "
                 f"{fmt_size(s['transfer_bytes'])})  jobs={s['pending_jobs']}")
    return line


def _ready_banner(srv: RadixServer) -> str:
    idx = srv.index
    if idx.is_distributed():
        return f"Cluster ready.  rank={idx.rank()}/{idx.world_size()}  shm={idx.shm_name()}"
    return f"Server ready.  shm={idx.shm_name()}"


def main(argv=None) -> None:
    a = parse_args(argv)
    try:
        cfg = build_config(a)
    except (ValueError, TypeError) as e:
        sys.exit(f"ERROR: {e}")

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    srv = RadixServer(cfg)
    print(f"Starting radix-server ({'index only' if cfg.index_only else 'index + data plane'})")
    print(f"  Name:           {cfg.name}")
    print(f"  Endpoint:       {cfg.resolved_endpoint}")
    if not cfg.index_only:
        print(f"  Data name:      {cfg.resolved_data_name}")
        print(f"  Data budget:    {fmt_size(cfg.data_bytes)}  (swa {cfg.swa_ratio:g}, "
              f"mamba {cfg.mamba_ratio:g}, FULL the rest)")
    if cfg.distributed:
        print(f"  Registry:       {cfg.cluster.registry}  cluster {cfg.cluster.cluster_id}")
    sys.stdout.flush()
    try:
        srv.start()
    except Exception as e:  # noqa: BLE001
        srv.close()
        sys.exit(f"Start FAILED: {e}")
    print("Waiting for a client's geometry (RadixClient(name, Geometry(...))) ...", flush=True)
    last, seen = 0.0, srv.mode
    try:
        while not stop.wait(0.2):
            mode = srv.mode
            if mode != seen:
                seen = mode
                if mode == "configuring":
                    print(f"Configuring: block {srv.geometry['block_size']} tokens, "
                          f"{_geometry.describe(srv.geometry)}", flush=True)
                elif mode == "ready":
                    print(_ready_banner(srv), flush=True)
                elif mode == "waiting":
                    print(f"Configuration FAILED: {srv.last_error}\nWaiting for a client's geometry ...",
                          flush=True)
            if a.interval > 0 and mode == "ready" and time.monotonic() - last >= a.interval:
                last = time.monotonic()
                sys.stdout.write(f"\r{_stats_line(srv)}        ")
                sys.stdout.flush()
    finally:
        print("\nShutting down ...", flush=True)
        srv.close()


if __name__ == "__main__":
    main()
