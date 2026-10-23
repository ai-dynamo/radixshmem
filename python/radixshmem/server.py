# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RadixServer: the one process per node that owns the index, the SlotStore,
the transfer engine and the etcd entries, and serves RadixClients over gRPC
(see ``radixshmem.rpc``).

``ServerConfig`` is what the operator knows when the node is deployed:
identity and cluster, the SlotStore budget (``data_bytes``, 0 = index-only)
and its split between the pools (``swa_ratio`` / ``mamba_ratio``; FULL takes
the rest), memory, transfer-engine and index tuning. What a slot looks like
comes from a client: ``configure(Geometry)`` (the Configure RPC) or, in a
cluster, the geometry another node's server already published in etcd.
Until then the server is ``waiting``. ``configure`` validates and plans
synchronously and returns; a thread then creates the regions and joins the
cluster (``configuring`` -> ``ready``, or back to ``waiting`` with
``last_error``). Clients never block unless they call ``wait_ready``.

Transfers are jobs: ``submit`` queues one, a single transfer thread feeds the
engine and polls completions, ``wait_any`` sleeps on an Event until one of the
caller's jobs is done. Nothing else polls. With ``transfer_log`` set, every
transfer that moved data is logged as one line (time, size, bandwidth); the
transfer thread only stamps times and queues a tuple, a log thread formats and
writes them.
"""

from __future__ import annotations

import fcntl
import json
import socket
import struct
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from radixshmem import _core, _geometry, rpc
from radixshmem._core import ComponentType, EvictPolicy
from radixshmem.client import (MODE_CLOSED, MODE_CONFIGURING, MODE_READY, MODE_STARTING, MODE_WAITING,
                             NodeInfo, TransferItem, TransferReply)
from radixshmem.rpc import DataPlaneUnavailable, GeometryMismatch, ServerBusy

# ClusterConfig fields copied one to one onto _core.RadixServerConfig.
_CLUSTER_FIELDS = (
    "expected_min_nodes", "settle_ms", "registry", "cluster_id", "node_name", "rpc_address",
    "rpc_interface", "rpc_port", "index_dev", "gid_idx", "bootstrap_timeout_sec", "lease_ttl_sec", "rht_transport",
    "peer_index_transport", "num_rht_shards", "rht_shard_holders", "rht_slots_per_bucket",
    "enable_remote_insert",
    "enable_remote_query", "zmq_listen_port", "remote_op_transport")

_POLL_S = 0.5        # how often a waiting cluster node looks for a published geometry
_LEASE_TTL_S = 10    # etcd lease of the geometry key; kept alive while the server runs


@dataclass
class ClusterConfig:
    """etcd rendezvous and the control-plane RDMA of the index (see
    _core.RadixServerConfig). Distributed when expected_min_nodes or
    num_rht_shards is > 1."""

    expected_min_nodes: int = 0
    settle_ms: int = 500
    registry: str = "etcd://127.0.0.1:2379"
    cluster_id: str = "default"
    node_name: str = ""
    rpc_address: str = ""
    rpc_interface: str = ""
    rpc_port: int = 0
    index_dev: str = ""              # HCA of the index control plane, "" = first available;
                                     # the data plane's HCAs are ServerConfig.transfer_devices
    gid_idx: int = 3
    bootstrap_timeout_sec: int = 60
    lease_ttl_sec: int = 30           # etcd lease TTL of the node and of the cluster keys
    rht_transport: str = "xrc"        # client -> RHT shard holder QPs: "xrc" | "dc"
    peer_index_transport: str = "xrc" # remote walk into peer trees: "xrc" | "dc"
    num_rht_shards: int = 0
    rht_shard_holders: List[int] = field(default_factory=list)
    rht_slots_per_bucket: int = 4
    enable_remote_insert: bool = False
    enable_remote_query: bool = False
    zmq_listen_port: int = 0
    remote_op_transport: str = "zmq"

    @property
    def distributed(self) -> bool:
        return self.expected_min_nodes > 1 or self.num_rht_shards > 1


@dataclass
class ServerConfig:
    """Everything the operator sets when the server starts; nothing here
    depends on the model. The slot shape arrives later from a client."""

    name: str = "/radixshmem"          # shm name; a cluster node gets "_<node_name>" appended
    # SlotStore budget and its split. 0 = index-only: no SlotStore, slot ids
    # belong to the client's own storage and the client's geometry brings the counts.
    data_bytes: int = 0
    swa_ratio: float = 0.0           # share of data_bytes for SWA slots; FULL takes what the aux pools leave
    mamba_ratio: float = 0.0
    data_name: str = ""              # SlotStore shm name, "" = <name>_data
    hugepage_path: str = ""          # hugetlbfs mount for both regions, "" = /dev/shm
    prefault: bool = True
    slot_align: int = 4096           # slot stride alignment unless the geometry brings its own
    # transfer engine (distributed data mode)
    transfer_backend: str = "mooncake"  # "mooncake" | "nixl"
    transfer_protocol: str = "rdma"
    transfer_devices: List[str] = field(default_factory=list)
    transfer_ip: str = ""            # "" = the cluster rpc address
    transfer_port: int = 0
    transfer_metadata: str = "P2PHANDSHAKE"
    # jobs
    max_inflight: int = 256          # batches in flight at the engine
    max_pending_jobs: int = 4096     # queued + running + uncollected jobs before Submit is refused
    job_ttl_s: float = 60.0          # seconds an uncollected job is kept
    transfer_log: str = ""           # one line per transfer with bytes > 0: "-" = stdout, a path appends, "" = off
    # index tuning
    max_nodes: int = 0               # radix node pool, 0 = auto
    register_chunk_tokens: int = 4096  # RHT registration granularity unless the geometry brings its own
    background_evict_ratio: float = 0.05
    data_pool_ratio: float = 4.0     # DataPool bytes = full_slots x ratio x (12 | 16 with SWA)
    evict_policy: str = "lru"
    # service
    endpoint: str = ""               # gRPC; "" = unix:///dev/shm/<name>.sock
    rpc_workers: int = 32            # one blocks per client with jobs in flight
    cluster: ClusterConfig = field(default_factory=ClusterConfig)

    def __post_init__(self) -> None:
        if self.data_bytes < 0:
            raise ValueError("data_bytes must be >= 0")
        for k in ("swa_ratio", "mamba_ratio"):
            if not 0.0 <= float(getattr(self, k)) < 1.0:
                raise ValueError(f"{k} must be in [0, 1)")
        if self.swa_ratio + self.mamba_ratio >= 1.0:
            raise ValueError("swa_ratio + mamba_ratio must stay below 1 to leave FULL slots")
        if self.index_only and (self.swa_ratio or self.mamba_ratio):
            raise ValueError("pool ratios need data_bytes > 0; without it the server is index only")
        if self.slot_align <= 0 or self.slot_align & (self.slot_align - 1):
            raise ValueError("slot_align must be a power of two")
        if self.register_chunk_tokens <= 0:
            raise ValueError("register_chunk_tokens must be > 0")
        if self.evict_policy not in ("lru",):
            raise ValueError(f"unknown evict_policy {self.evict_policy!r}")
        if self.transfer_backend not in ("mooncake", "nixl"):
            raise ValueError(f"unknown transfer_backend {self.transfer_backend!r}")

    @property
    def index_only(self) -> bool:
        return self.data_bytes <= 0

    @property
    def distributed(self) -> bool:
        return self.cluster.distributed

    @property
    def resolved_data_name(self) -> str:
        if self.index_only:
            return ""
        return self.data_name or f"{self.name}_data"

    @property
    def resolved_endpoint(self) -> str:
        return self.endpoint or rpc.endpoint_for(self.name)

    def resolved_transfer_ip(self) -> str:
        if self.transfer_ip:
            return self.transfer_ip
        c = self.cluster
        if c.rpc_interface:
            return _nic_ip(c.rpc_interface)
        return c.rpc_address if c.rpc_address not in ("", "0.0.0.0") else "127.0.0.1"


def _nic_ip(ifname: str) -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        packed = fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", ifname[:15].encode()))
    finally:
        s.close()
    return socket.inet_ntoa(packed[20:24])


class _GeometryRegistry:
    """``radix/<cluster_id>/geometry/<node>``: the uniform part of a node's
    geometry as JSON. Published when the node accepts a geometry, before its
    rendezvous (the peers it waits for at the membership gate may be waiting
    for exactly this key), on a lease this server keeps alive for its
    lifetime, so the key goes away with the process."""

    def __init__(self, registry: str, cluster_id: str):
        self._kv = _core.EtcdKv(registry)
        self._prefix = f"radix/{cluster_id}/geometry/"
        self.lease = self._kv.grant_lease(_LEASE_TTL_S)
        self._kv.keep_alive(self.lease)

    def list(self) -> List[Tuple[str, Dict[str, Any]]]:
        out = []
        for key, value in self._kv.get_prefix(self._prefix):
            try:
                out.append((key[len(self._prefix):], json.loads(value)))
            except ValueError:
                continue
        return out

    def publish(self, node: str, uniform: Dict[str, Any]) -> None:
        self._kv.put(self._prefix + node, json.dumps(uniform, sort_keys=True), self.lease)

    def unpublish(self, node: str) -> None:
        self._kv.delete(self._prefix + node)

    def close(self) -> None:
        try:
            self._kv.revoke(self.lease)
        except Exception:  # noqa: BLE001 - the lease expires on its own
            pass


class _Job:
    __slots__ = ("ticket", "items", "segs", "replies", "left", "deadline", "done", "done_at", "waiters",
                 "submitted_ns")

    def __init__(self, items: List[TransferItem], deadline: float):
        self.submitted_ns = time.perf_counter_ns()
        self.ticket = 0
        self.items = items
        self.segs: List[Optional[str]] = []
        self.replies: List[Optional[TransferReply]] = [None] * len(items)
        self.left = len(items)
        self.deadline = deadline
        self.done = False
        self.done_at = 0.0
        self.waiters: List[threading.Event] = []


class _TransferLog:
    """Per-transfer log lines. ``add`` (transfer thread) only appends a tuple
    of perf_counter_ns stamps and counts; a daemon thread formats and writes
    the queued records once per ``period_s``, and ``close`` writes the rest.

    Line: ``xfer <UTC end time> peer=<rank> kind=<pool> slots=<n> desc=<n>
    bytes=<n> us=<issue to completion> MBps=<bytes/us> wait_us=<submit to
    issue> inflight=<batches already in flight at issue>`` plus ``state=<..>``
    when the transfer did not complete."""

    def __init__(self, target: str, period_s: float = 1.0):
        self._q: Deque[tuple] = deque()
        self._own = target != "-"
        self._out = open(target, "a", buffering=1) if self._own else sys.stdout
        self._base_ns = time.time_ns() - time.perf_counter_ns()
        self._period = period_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="radixshmem-xferlog", daemon=True)
        self._thread.start()

    def add(self, rec: tuple) -> None:
        self._q.append(rec)

    def _run(self) -> None:
        while not self._stop.wait(self._period):
            self.flush()

    def flush(self) -> None:
        lines = []
        while True:
            try:
                submitted, issued, done, rank, kind, slots, ndesc, nbytes, state, inflight = self._q.popleft()
            except IndexError:
                break
            us = max(done - issued, 1) / 1000
            end = datetime.fromtimestamp((self._base_ns + done) / 1e9, timezone.utc)
            lines.append(f"xfer {end.strftime('%Y-%m-%dT%H:%M:%S.%f')}Z peer={rank} kind={_kind_name(kind)} "
                         f"slots={slots} desc={ndesc} bytes={nbytes} us={us:.1f} MBps={nbytes / us:.1f} "
                         f"wait_us={(issued - submitted) / 1000:.1f} inflight={inflight}"
                         f"{'' if not state else ' state=' + state}\n")
        if lines:
            self._out.write("".join(lines))
            self._out.flush()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(2 * self._period + 1)
        self.flush()
        if self._own:
            self._out.close()


def _kind_name(kind: int) -> str:
    try:
        return ComponentType(kind).name
    except Exception:  # noqa: BLE001
        return str(kind)


class RadixServer:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self._geometry: Optional[Dict[str, Any]] = None
        self.last_error = ""
        self._cv = threading.Condition(threading.RLock())
        self._mode = MODE_STARTING
        self._index = self._iclient = self._store = self._backend = self._xfer = self._dreg = None
        self._greg: Optional[_GeometryRegistry] = None
        self._rejected: Optional[Dict[str, Any]] = None   # cluster geometry this node could not take
        self._peers: Dict[int, str] = {}
        self._zombies: List[int] = []
        self._rpc = None
        self._stop = threading.Event()
        self._poller: Optional[threading.Thread] = None
        self._worker: Optional[threading.Thread] = None
        self._jobs_mu = threading.Lock()
        self._jobs: Dict[int, _Job] = {}
        self._queue: Deque[_Job] = deque()
        self._next_ticket = 1
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.transfers = self.transfer_failures = self.transfer_bytes = 0
        self._xlog: Optional[_TransferLog] = None

    # ------------------------------------------------------------ lifecycle

    def start(self) -> "RadixServer":
        """Serve the endpoint and bring up what needs no geometry (etcd, the
        transfer engine); ``waiting`` on return. A cluster node with a data
        plane then follows the geometry its peers publish; every server takes
        a client's ``configure``."""
        self._rpc = rpc.serve({
            "Describe": lambda _: vars(self.describe()),
            "WaitReady": lambda r: vars(self.wait_ready(r["timeout_ms"] / 1000)),
            "Configure": lambda r: vars(self.configure(r["geometry"])),
            "Submit": lambda r: {"ticket": self.submit(r["items"], r["timeout_ms"])},
            "WaitAny": lambda r: self.wait_any(r["tickets"], r["timeout_ms"]),
            "Stats": lambda _: self.stats(),
        }, self.cfg.resolved_endpoint, self.cfg.rpc_workers)
        if self.cfg.transfer_log and self._xlog is None:
            self._xlog = _TransferLog(self.cfg.transfer_log)
        try:
            self._preflight()
        except BaseException:
            self._rpc.stop(grace=0).wait()
            self._rpc = None
            with self._cv:
                self._mode = MODE_CLOSED
                self._cv.notify_all()
            raise
        with self._cv:
            self._mode = MODE_WAITING
            self._cv.notify_all()
        if self._greg is not None and not self.cfg.index_only:
            self._poller = threading.Thread(target=self._follow_cluster, name="radixshmem-geometry",
                                            daemon=True)
            self._poller.start()
        return self

    def _preflight(self) -> None:
        c = self.cfg
        if not c.index_only:
            from radixshmem import _data  # noqa: F401 - data mode needs the SlotStore extension
        if not c.distributed:
            return
        if not (c.cluster.rpc_address or c.cluster.rpc_interface):
            raise ValueError("distributed mode requires cluster.rpc_address or cluster.rpc_interface")
        if not hasattr(_core, "EtcdKv"):
            raise RuntimeError("radixshmem was built without etcd; a cluster needs it")
        self._greg = _GeometryRegistry(c.cluster.registry, c.cluster.cluster_id)
        if not c.index_only:
            self._backend = self._make_backend()

    def _make_backend(self):
        from radixshmem import _data

        c = self.cfg
        if c.transfer_backend == "nixl" and c.transfer_port <= 0:
            raise ValueError("transfer_port must be set for the nixl transfer backend; "
                             "it cannot pick a port")
        backend = _data.create_transfer_backend(_data.TransferConfig(
            backend=c.transfer_backend, metadata=c.transfer_metadata, ip_or_host=c.resolved_transfer_ip(),
            rpc_port=c.transfer_port, protocol=c.transfer_protocol,
            device_filter=list(c.transfer_devices), publish=True))
        if backend is None:
            raise RuntimeError(
                f"transfer backend {c.transfer_backend!r} unavailable (no RDMA "
                f"device, or that backend not built — see ENABLE_MOONCAKE/ENABLE_NIXL)")
        return backend

    def close(self) -> None:
        self._stop.set()
        if self._poller is not None:
            self._poller.join(2)
            self._poller = None
        if self._rpc is not None:
            self._rpc.stop(grace=1).wait()
            self._rpc = None
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(self.cfg.cluster.bootstrap_timeout_sec + 30)
        self._running = False
        self._wake.set()
        if self._thread is not None:
            self._thread.join(2)
            self._thread = None
        if self._xlog is not None:
            self._xlog.close()
            self._xlog = None
        if self._greg is not None:
            try:
                self._greg.unpublish(self._node_key())
            except Exception:  # noqa: BLE001
                pass
            self._greg.close()
        self._teardown_regions()
        self._backend = self._greg = None
        with self._cv:
            self._mode = MODE_CLOSED
            self._cv.notify_all()

    def _teardown_regions(self) -> None:
        """Drop the index, the SlotStore and the data-plane entry; the transfer
        engine and the etcd connection stay (they are the server's, not the
        geometry's)."""
        with self._cv:
            index, store, dreg = self._index, self._store, self._dreg
            self._index = self._iclient = self._store = self._xfer = self._dreg = None
            self._peers = {}
        if dreg is not None and index is not None:
            try:
                dreg.unpublish(index.node_name())
            except Exception:  # noqa: BLE001
                pass
        self._reap()
        if store is not None:
            from radixshmem import _data

            name, hp = store.name, self.cfg.hugepage_path
            del store
            _data.SlotStore.destroy(name, hp)
        del index

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def index(self):
        return self._index

    @property
    def geometry(self) -> Optional[Dict[str, Any]]:
        """Planned geometry from ``configuring`` on; None while waiting."""
        return self._geometry

    @property
    def endpoint(self) -> str:
        return self.cfg.resolved_endpoint

    # ------------------------------------------------------------- control

    def describe(self) -> NodeInfo:
        with self._cv:
            idx, g, c = self._index, self._geometry, self.cfg
            return NodeInfo(
                mode=self._mode,
                data_plane=not c.index_only,
                index_name=idx.shm_name() if idx else (g["index_name"] if g else c.name),
                data_name=c.resolved_data_name,
                hugepage_path=c.hugepage_path,
                geometry=g,
                rank=idx.rank() if idx else -1,
                world_size=idx.world_size() if idx else 0,
                node_name=idx.node_name() if idx else c.cluster.node_name,
                segment=self._xfer.local_segment if self._xfer else "",
                data_bytes=c.data_bytes,
                last_error=self.last_error,
                swa_ratio=float(c.swa_ratio),
                mamba_ratio=float(c.mamba_ratio))

    def configure(self, geometry) -> NodeInfo:
        """Take a client's geometry: validate it, plan the pools, check it
        against the cluster's, and start the bring-up thread. Returns at once
        (``configuring``). A server that already has a geometry only compares:
        the same one is accepted again, a different one raises
        ``GeometryMismatch``. ValueError on a geometry that does not fit this
        server (wrong mode, unplannable budget)."""
        spec = _geometry.as_spec(geometry)
        with self._cv:
            self._cv.wait_for(lambda: self._mode != MODE_STARTING)
            if self._mode == MODE_CLOSED:
                raise RuntimeError("server is closed")
            planned = _geometry.plan(spec, self.cfg)
            if self._mode in (MODE_CONFIGURING, MODE_READY):
                diffs = _geometry.diff_geometry(planned, self._geometry, counts=True)
                if diffs:
                    raise GeometryMismatch("server serves another geometry: " + "; ".join(diffs))
                return self.describe()
            if self._greg is not None:
                for node, published in self._greg.list():
                    diffs = _geometry.diff_geometry(planned, published, counts=False)
                    if diffs:
                        raise GeometryMismatch(f"geometry differs from node {node}'s: " + "; ".join(diffs))
                self._greg.publish(self._node_key(), _geometry.uniform_fields(planned))
            self._geometry, self.last_error = planned, ""
            self._mode = MODE_CONFIGURING
            self._worker = threading.Thread(target=self._configure_thread, name="radixshmem-configure",
                                            daemon=True)
            self._worker.start()
            self._cv.notify_all()
            return self.describe()

    def _node_key(self) -> str:
        """This node's name under geometry/: the configured node_name, else the
        one the index derives from its bind IP (node<ip>)."""
        c = self.cfg.cluster
        if c.node_name:
            return c.node_name
        return "node" + (_nic_ip(c.rpc_interface) if c.rpc_interface else c.rpc_address)

    def _configure_thread(self) -> None:
        try:
            self._bring_up()
            with self._cv:
                self._mode = MODE_READY
                self._cv.notify_all()
        except Exception as e:  # noqa: BLE001 - reported through last_error, the server stays up
            self._teardown_regions()
            if self._greg is not None:
                try:
                    self._greg.unpublish(self._node_key())
                except Exception:  # noqa: BLE001
                    pass
            with self._cv:
                self.last_error = f"{type(e).__name__}: {e}"
                self._rejected = _geometry.uniform_fields(self._geometry)
                self._geometry = None
                self._mode = MODE_WAITING
                self._cv.notify_all()

    def _follow_cluster(self) -> None:
        """Data-mode cluster node: adopt the geometry a peer published. A
        geometry this node rejected or failed on is not retried; a client's
        configure still is taken."""
        while not self._stop.wait(_POLL_S):
            with self._cv:
                if self._mode != MODE_WAITING:
                    continue
            try:
                entries = self._greg.list()
            except Exception:  # noqa: BLE001 - etcd hiccup, try again next round
                continue
            if not entries:
                continue
            node, published = entries[0]
            if self._rejected is not None and not _geometry.diff_geometry(published, self._rejected,
                                                                          counts=False):
                continue
            try:
                self.configure(published)
            except Exception as e:  # noqa: BLE001 - a client's configure may still succeed
                with self._cv:
                    self.last_error = f"cluster geometry from {node} rejected: {e}"
                    self._rejected = published

    def wait_ready(self, timeout_s: float) -> NodeInfo:
        """Block until ``ready``. TimeoutError when the server is still
        waiting for a geometry or configuring after ``timeout_s``;
        RuntimeError when it is waiting after a failed configuration (the
        error is ``last_error``; a new configure clears it) or closed."""
        with self._cv:
            settled = lambda: (self._mode in (MODE_READY, MODE_CLOSED)  # noqa: E731
                               or (self._mode == MODE_WAITING and bool(self.last_error)))
            if not self._cv.wait_for(settled, timeout_s):
                raise TimeoutError(f"server still {self._mode} after {timeout_s:.0f}s")
            if self._mode == MODE_READY:
                return self.describe()
            if self._mode == MODE_WAITING:
                raise RuntimeError(f"configuration failed: {self.last_error}")
            raise RuntimeError("server is closed")

    def stats(self) -> Dict[str, Any]:
        idx = self._index
        with self._jobs_mu:
            out = {"mode": self._mode, "transfers": self.transfers,
                   "transfer_failures": self.transfer_failures, "transfer_bytes": self.transfer_bytes,
                   "pending_jobs": len(self._jobs), "queued_jobs": len(self._queue)}
        if idx is not None:
            out.update(rank=idx.rank(), world_size=idx.world_size(),
                       total_radix_nodes=idx.total_radix_nodes(), total_blocks=idx.total_blocks(),
                       data_pool_utilization=idx.data_pool_utilization(),
                       mempool_total=idx.mempool_total(), mempool_used=idx.mempool_used(),
                       insert_success=idx.insert_success_count(),
                       insert_errors=idx.insert_error_node_pool_count()
                       + idx.insert_error_data_pool_count() + idx.insert_error_ht_pool_count(),
                       **idx.peer_fault_counts())
        return out

    # ------------------------------------------------------------ bring-up

    def _core_config(self, shm, g) -> "_core.RadixServerConfig":
        cc = _core.RadixServerConfig()
        cc.name, cc.shm = self.cfg.name, shm
        cc.register_chunk_tokens = int(g["register_chunk_tokens"])
        for k in _CLUSTER_FIELDS:
            setattr(cc, k, getattr(self.cfg.cluster, k))
        return cc

    def _bring_up(self) -> None:
        c, g = self.cfg, self._geometry
        shm = _geometry.shm_config(g, background_evict_ratio=c.background_evict_ratio,
                                   data_pool_ratio=c.data_pool_ratio)
        shm.evict_policy = {"lru": EvictPolicy.LRU}[c.evict_policy]
        shm.max_nodes = shm.effective_max_nodes()
        index = _core.RadixServer(self._core_config(shm, g))
        if not index.bootstrap():
            raise RuntimeError("index bootstrap failed")
        iclient = _core.RadixClient(index)
        g["index_name"] = index.shm_name()      # a cluster node's name carries "_<node_name>"
        store = None
        if not c.index_only:
            from radixshmem import _data

            _data.SlotStore.destroy(g["data_name"], g["hugepage_path"])
            store = _data.SlotStore.create(_geometry.store_config(g, prefault=c.prefault))
            _geometry.check_attached(g, iclient, store)
        else:
            _geometry.check_attached(g, iclient, None)
        with self._cv:
            self._index, self._iclient, self._store = index, iclient, store
        if store is not None and index.is_distributed():
            self._data_plane_up(index, store)

    def _data_plane_up(self, index, store) -> None:
        from radixshmem import _data

        c = self.cfg
        xfer = _data.SlotTransfer(store, self._backend)
        rc = xfer.register_store("cpu:0")
        if rc != _data.XFER_OK:
            raise RuntimeError(f"register_store failed rc={rc}")
        dreg = _data.DataPlaneRegistry(c.cluster.registry, c.cluster.cluster_id)
        entry = _data.make_data_plane_entry(store.layout(), xfer.local_segment)
        ok, who, why = dreg.validate_uniform(entry)
        if not ok:
            raise RuntimeError(f"data plane of {who} is incompatible: {why}")
        dreg.publish(index.node_name(), entry, index.lease_id())
        with self._cv:
            self._xfer, self._dreg = xfer, dreg

    # ---------------------------------------------------------------- jobs

    def submit(self, items: Sequence[TransferItem], timeout_ms: int) -> int:
        if self._mode != MODE_READY:
            raise DataPlaneUnavailable(f"server is {self._mode}")
        if self._store is None:
            raise DataPlaneUnavailable("index-only server has no data plane")
        if self._xfer is None:
            raise DataPlaneUnavailable("standalone server has no peers")
        job = _Job([(int(r), int(k), [int(x) for x in rs], [int(x) for x in ls])
                    for r, k, rs, ls in items], time.monotonic() + timeout_ms / 1000)
        for i, (rank, _, _, _) in enumerate(job.items):
            try:
                job.segs.append(self._peer(rank))
            except Exception as e:  # noqa: BLE001
                job.segs.append(None)
                job.replies[i] = (False, 0, 0, str(e))
                job.left -= 1
        with self._jobs_mu:
            if len(self._jobs) >= self.cfg.max_pending_jobs:
                raise ServerBusy(f"{len(self._jobs)} jobs pending")
            job.ticket = self._next_ticket
            self._next_ticket += 1
            self._jobs[job.ticket] = job
            if job.left == 0:
                self._mark_done(job)
            else:
                self._queue.append(job)
            if self._thread is None:
                self._running = True
                self._thread = threading.Thread(target=self._transfer_loop, name="radixshmem-transfer",
                                                daemon=True)
                self._thread.start()
        self._wake.set()
        return job.ticket

    def wait_any(self, tickets: Sequence[int], timeout_ms: int) -> Dict[str, Any]:
        ev = threading.Event()
        with self._jobs_mu:
            jobs = {int(t): self._jobs.get(int(t)) for t in tickets}
            unknown = [t for t, j in jobs.items() if j is None]
            wait = not unknown and not any(j.done for j in jobs.values())
            if wait:
                for j in jobs.values():
                    j.waiters.append(ev)
        if wait:
            ev.wait(min(int(timeout_ms), 5000) / 1000)
        done = {}
        with self._jobs_mu:
            for t, j in jobs.items():
                if j is None:
                    continue
                if j.done:
                    done[t] = j.replies
                    del self._jobs[t]
                elif ev in j.waiters:
                    j.waiters.remove(ev)
        return {"done": done, "unknown": unknown}

    def _mark_done(self, job: _Job) -> None:
        job.done, job.done_at = True, time.monotonic()
        for ev in job.waiters:
            ev.set()
        job.waiters.clear()

    def _resolve(self, job: _Job, i: int, reply: TransferReply) -> None:
        with self._jobs_mu:
            job.replies[i] = reply
            job.left -= 1
            self.transfers += 1
            self.transfer_failures += not reply[0]
            self.transfer_bytes += reply[1]
            if job.left == 0:
                self._mark_done(job)

    def _transfer_loop(self) -> None:
        from radixshmem import _data

        c = self.cfg
        xlog = self._xlog
        clock = time.perf_counter_ns
        inflight: List[tuple] = []
        last_gc = time.monotonic()
        while self._running:
            while len(inflight) < c.max_inflight:
                with self._jobs_mu:
                    job = self._queue.popleft() if self._queue else None
                if job is None:
                    break
                for i, (rank, kind, remote, local) in enumerate(job.items):
                    if job.replies[i] is not None:
                        continue
                    issued = clock() if xlog is not None else 0
                    try:
                        h, nd = self._xfer.read(job.segs[i], remote, local, kind=ComponentType(kind))
                    except Exception as e:  # noqa: BLE001
                        h, nd = _data.INVALID_BATCH, str(e)
                    if h == _data.INVALID_BATCH:
                        self._peers.pop(rank, None)
                        self._resolve(job, i, (False, 0, 0, nd if isinstance(nd, str) else "read rejected"))
                    else:
                        inflight.append((job, i, h, nd, issued, len(inflight)))
            progressed, now, still = False, time.monotonic(), []
            for job, i, h, nd, issued, ahead in inflight:
                st, done = self._xfer.poll(h)
                if st == _data.XferState.PENDING and now < job.deadline:
                    still.append((job, i, h, nd, issued, ahead))
                    continue
                if xlog is not None and done:
                    rank, kind, _, local = job.items[i]
                    xlog.add((job.submitted_ns, issued, clock(), rank, kind, len(local), int(nd), int(done),
                              "" if st == _data.XferState.COMPLETED else
                              ("TIMEOUT" if st == _data.XferState.PENDING else st.name), ahead))
                if st == _data.XferState.PENDING:
                    st = _data.XferState.TIMEOUT
                if self._xfer.release(h) == _data.XFER_EBUSY:
                    self._zombies.append(h)
                ok = st == _data.XferState.COMPLETED
                if not ok:
                    self._peers.pop(job.items[i][0], None)
                self._resolve(job, i, (ok, int(done), int(nd), "" if ok else str(st)))
                progressed = True
            inflight = still
            if now - last_gc > 1.0:
                last_gc = now
                self._reap()
                with self._jobs_mu:
                    for t in [t for t, j in self._jobs.items()
                              if j.done and now - j.done_at > c.job_ttl_s]:
                        del self._jobs[t]
            if inflight:
                time.sleep(0 if progressed else 0.00005)
            else:
                self._wake.wait(0.1)
                self._wake.clear()

    def _peer(self, rank: int) -> str:
        from radixshmem import _data

        with self._cv:
            seg = self._peers.get(rank)
            if seg:
                return seg
            if rank == self._index.rank():
                raise ValueError("peer rank is this node")
            node = self._dreg.node_of_rank(rank)
            if not node:
                raise LookupError(f"rank {rank} is not in the cluster")
            e = self._dreg.lookup(node, 0)
            if e is None:
                raise LookupError(f"{node} publishes no data plane")
            rc, _ = self._xfer.attach_peer(e.segment, e.layout())
            if rc != _data.XFER_OK:
                raise RuntimeError(f"attach_peer({e.segment}) rc={rc}")
            self._peers[rank] = e.segment
            return e.segment

    def _reap(self) -> None:
        from radixshmem import _data

        if self._xfer is not None:
            self._zombies = [h for h in self._zombies if self._xfer.release(h) == _data.XFER_EBUSY]
