# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RadixClient: index maintenance plus ``pull_async``, the one call that also
moves bytes. It attaches the index (and, in data mode, the SlotStore) its
RadixServer published; the server runs the transfers, this side stages local
slots, submits, and publishes the result once the server reports completion.

A client can be built before its server is ready. ``RadixClient(name,
geometry)`` hands the server the geometry (the Configure RPC) and returns at
once; ``status()`` asks the server where it stands without waiting;
``wait_ready()`` is the one call that blocks. The index and the SlotStore are
attached on the first index or data call, which raises ``ServerNotReady``
while the server is still waiting or configuring."""

from __future__ import annotations

import concurrent.futures
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from radixshmem import _core, _geometry, rpc
from radixshmem._core import COMPONENT_MASK_FULL, ComponentType
from radixshmem.rpc import DataPlaneUnavailable, ServerNotReady

FULL, SWA, MAMBA = ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA

# Server lifecycle. Whether there is a data plane is NodeInfo.data_plane.
MODE_STARTING = "starting"          # start() has not finished its preflight
MODE_WAITING = "waiting"            # up, no geometry yet: a client's configure or a peer's publish
MODE_CONFIGURING = "configuring"    # geometry accepted, regions being created / cluster joining
MODE_READY = "ready"
MODE_CLOSED = "closed"

_ATTACH_TIMEOUT_MS = 30000          # a ready server's SlotStore is complete; attach never spins long

# (peer_rank, kind, remote_slots, local_slots) -> (ok, bytes, descriptors, error)
TransferItem = Tuple[int, int, List[int], List[int]]
TransferReply = Tuple[bool, int, int, str]


@dataclass
class NodeInfo:
    mode: str
    data_plane: bool = False
    index_name: str = ""
    data_name: str = ""
    hugepage_path: str = ""
    geometry: Optional[Dict[str, Any]] = None
    rank: int = -1
    world_size: int = 0
    node_name: str = ""
    segment: str = ""
    data_bytes: int = 0
    last_error: str = ""            # why the last configuration failed; "" when none did
    swa_ratio: float = 0.0
    mamba_ratio: float = 0.0


@dataclass
class PullResult:
    """Everything in [0, common_hit) is in the local SlotStore when this exists.
    ``finalize()`` drops the local pin taken with ``lock=True`` (no-op otherwise)."""

    common_hit: int
    full_slots: np.ndarray
    swa_start: int
    swa_slots: np.ndarray
    mamba_slot: int
    remote_blocks: int
    remote_bytes: int
    source_rank: int
    finalize: Callable[[], None]


class PullJob:
    """Handle of one pull_async(). ``wait()`` blocks until the job completed and
    returns its PullResult; a failed or refused transfer completes the job with
    the local hit, it never raises."""

    def __init__(self, job_id: int, client: "RadixClient"):
        self.job_id = job_id
        self._client = client
        self._f: concurrent.futures.Future = concurrent.futures.Future()
        self._q = None
        self._hashes: np.ndarray = np.empty(0, dtype=np.uint64)
        self._mask = 0
        self._lock = True
        self._hit = 0
        self._items: List[TransferItem] = []
        self._staged: Dict[ComponentType, Tuple[int, np.ndarray]] = {}
        self._src = -1
        self._ticket = -1
        self._cancelled = False
        # Known when pull_async() returns: blocks the local tree already held, and
        # that plus the remote run being pulled (equal when nothing is pulled).
        self.local_hit = 0
        self.planned_hit = 0

    def done(self) -> bool:
        return self._f.done()

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def wait(self, timeout: Optional[float] = None) -> PullResult:
        if self._cancelled:
            raise RuntimeError(f"job {self.job_id} was cancelled")
        try:
            r = self._f.result(timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"job {self.job_id} still running") from None
        self._client._jobs.pop(self.job_id, None)
        return r

    def cancel(self) -> None:
        """Give up on the result: the job leaves the client's table now. A
        transfer still in flight finishes in the background (the RDMA READ into
        the staged slots cannot be aborted), its blocks are published as usual,
        and no pin is left behind. ``wait()`` afterwards raises."""
        self._client.cancel(self)


def _empty() -> np.ndarray:
    return np.empty(0, dtype=np.int32)


class _Rpc:
    def __init__(self, endpoint: str):
        self._stub = rpc.Stub(endpoint)

    def describe(self) -> NodeInfo:
        return NodeInfo(**self._stub.call("Describe", timeout=30))

    def wait_ready(self, timeout_s) -> NodeInfo:
        return NodeInfo(**self._stub.call("WaitReady", {"timeout_ms": int(timeout_s * 1000)},
                                          timeout=timeout_s + 5))

    def configure(self, spec: Dict[str, Any]) -> NodeInfo:
        return NodeInfo(**self._stub.call("Configure", {"geometry": spec}, timeout=30))

    def submit(self, items, timeout_ms) -> int:
        req = {"items": [[int(r), int(k), [int(x) for x in rs], [int(x) for x in ls]]
                         for r, k, rs, ls in items], "timeout_ms": int(timeout_ms)}
        return int(self._stub.call("Submit", req, timeout=30)["ticket"])

    def wait_any(self, tickets, timeout_ms) -> Dict[str, Any]:
        r = self._stub.call("WaitAny", {"tickets": list(tickets), "timeout_ms": int(timeout_ms)},
                            timeout=timeout_ms / 1000 + 5)
        return {"done": {int(t): [tuple(x) for x in v] for t, v in r["done"].items()},
                "unknown": [int(t) for t in r["unknown"]]}

    def close(self) -> None:
        self._stub.close()


class RadixClient:
    """``RadixClient(name)`` connects to the server started as ``--name <name>``
    on this node (socket ``unix:///dev/shm/<name>.sock``; ``endpoint`` overrides
    it for TCP or another path). The server need not be ready: the constructor
    only talks gRPC and returns at once. With ``geometry`` (a ``Geometry`` or
    its dict) it hands the server the geometry first (idempotent; a different
    one raises ``GeometryMismatch``, one this server cannot take
    ``ValueError``). ``status()`` refreshes ``info`` without waiting,
    ``wait_ready(timeout_s)`` blocks until the server is ready, ``configure()``
    sends a geometry later. The index and, when the server has a data plane,
    the SlotStore are attached on the first index or data call; before the
    server is ready that call raises ``ServerNotReady``.

    ``geometry`` is what the server published; callers compare their own page
    size and bytes against it. Index methods (query, insert, allocate_slots,
    recycle_slots, flush, reset, check_health, stats) are those of the
    underlying index client. ``max_outstanding`` bounds the jobs in flight;
    ``pull_async`` blocks while it is reached.

    ``local_read_only=True`` is the light form for processes that only read: it
    attaches the index and the SlotStore, never joins the cluster (no etcd,
    RDMA or RHT plane) and does not start a completer thread. ``query()`` is
    always local, ``is_distributed()`` is False, and ``pull_async`` / ``insert``
    / ``allocate_slots`` / ``recycle_slots`` / ``reset`` raise ``RuntimeError``;
    ``info`` still carries the server's rank and world size."""

    def __init__(self, name: str, geometry=None, *, endpoint: Optional[str] = None,
                 max_outstanding: int = 256, local_read_only: bool = False):
        self.name = name
        self.local_read_only = local_read_only
        self._index = None
        self._store = None
        self._attach_mu = threading.Lock()
        self._t = _Rpc(endpoint or rpc.endpoint_for(name))
        try:
            self.info = (self._t.configure(_geometry.as_spec(geometry)) if geometry is not None
                         else self._t.describe())
        except Exception:
            self._t.close()
            raise
        self.transfer_failures = 0
        self.staging_shortfalls = 0
        self.promotion_shortfalls = 0
        self.saturated = 0
        self._mu = threading.Lock()
        self._jobs: Dict[int, PullJob] = {}
        self._pending: Dict[int, PullJob] = {}
        self._next_id = 1
        self._slots = threading.Semaphore(max_outstanding)
        self._work = threading.Event()
        self._completer: Optional[threading.Thread] = None
        self._closed = False

    # ------------------------------------------------------------- control

    def status(self) -> NodeInfo:
        """The server's state now: one RPC, no waiting. Refreshes ``info``."""
        self.info = self._t.describe()
        return self.info

    def configure(self, geometry) -> NodeInfo:
        """Hand the server a geometry; returns as soon as it is validated and
        planned (``mode`` configuring, or ready when it already served it)."""
        self.info = self._t.configure(_geometry.as_spec(geometry))
        return self.info

    def wait_ready(self, timeout_s: float = 300.0) -> NodeInfo:
        """Block until the server is ready. ``TimeoutError`` while it is still
        waiting for a geometry or configuring, ``RuntimeError`` (with the
        server's ``last_error``) when its configuration failed."""
        self.info = self._t.wait_ready(timeout_s)
        return self.info

    def _attach(self):
        """The index client, attached on first use once the server is ready."""
        index = self._index
        if index is not None:
            return index
        with self._attach_mu:
            if self._index is not None:
                return self._index
            if self._closed:
                raise RuntimeError("client is closed")
            info = self.info if self.info.mode == MODE_READY else self.status()
            if info.mode != MODE_READY:
                raise ServerNotReady(info.mode, info.last_error)
            index = _core.RadixClient(info.index_name, local_read_only=self.local_read_only)
            store = None
            if info.data_plane:
                from radixshmem import _data

                store = _data.SlotStore.attach(info.data_name, info.hugepage_path, _ATTACH_TIMEOUT_MS)
            _geometry.check_attached(info.geometry, index, store)
            self._store, self._index = store, index
            return index

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._attach(), name)

    @property
    def index(self):
        return self._attach()

    @property
    def geometry(self) -> Optional[Dict[str, Any]]:
        return self.info.geometry

    @property
    def store(self):
        self._attach()
        if self._store is None:
            raise DataPlaneUnavailable("index-only server: no SlotStore to read")
        return self._store

    def slot_view(self, slot: int, kind: ComponentType = FULL):
        return self.store.slot_view(int(slot), kind)

    def close(self) -> None:
        self._closed = True
        self._work.set()
        if self._completer is not None:
            self._completer.join(2)
        self._store = None
        self._index = None
        self._t.close()

    # ------------------------------------------------------------ pull_async

    def pull_async(self, hashes, mask: int = COMPONENT_MASK_FULL, *, lock: bool = True,
                   timeout_ms: int = 30000, block: bool = True) -> PullJob:
        """query, stage local slots for every remote piece, hand the transfer to
        the server and return; the completer publishes the blocks when the
        server reports them and completes the job with local slots for
        [0, common_hit). With ``block=False`` a client at ``max_outstanding``
        does not wait: the job completes at once with the local hit."""
        if self.local_read_only:
            raise RuntimeError("local_read_only client: pull_async needs the cluster plane; "
                               "use query() for the local hit")
        self.store
        with self._mu:
            job = PullJob(self._next_id, self)
            self._next_id += 1
            self._jobs[job.job_id] = job
        job._hashes = np.ascontiguousarray(hashes, dtype=np.uint64)
        job._mask, job._lock = mask, lock
        job._q = q = self._index.query(job._hashes, mask=mask, local_only=False, lock=True,
                                       update_meta=True)
        job._hit = self._stage(job, int(q.common_hit) if int(q.status) == 0 else 0)
        job.planned_hit = job._hit
        if not job._items:
            self._complete(job, [])
            return job
        if block:
            self._slots.acquire()
        elif not self._slots.acquire(blocking=False):
            self.saturated += 1
            self._complete(job, [(False, 0, 0, "client saturated")] * len(job._items))
            return job
        try:
            job._ticket = self._t.submit(job._items, timeout_ms)
        except Exception as e:  # noqa: BLE001 - busy or unreachable server: local hit only
            self._slots.release()
            self._complete(job, [(False, 0, 0, str(e))] * len(job._items))
            return job
        with self._mu:
            self._pending[job._ticket] = job
            if self._completer is None:
                self._completer = threading.Thread(target=self._run, name="radixshmem-completer",
                                                   daemon=True)
                self._completer.start()
        self._work.set()
        return job

    def wait(self, job: Union[PullJob, int], timeout: Optional[float] = None) -> PullResult:
        return (job if isinstance(job, PullJob) else self._jobs[job]).wait(timeout)

    def cancel(self, job: Union[PullJob, int]) -> None:
        """See ``PullJob.cancel``. Under ``_mu`` so that a result set by the
        completer at the same moment still gets its pin released."""
        job = job if isinstance(job, PullJob) else self._jobs[job]
        with self._mu:
            job._cancelled = True
            self._jobs.pop(job.job_id, None)
            r = (job._f.result() if job._f.done() and job._f.exception() is None
                 else None)
        if r is not None:
            r.finalize()

    def wait_any(self, jobs: Sequence[PullJob], timeout: Optional[float] = None) -> List[PullJob]:
        concurrent.futures.wait([j._f for j in jobs], timeout,
                                return_when=concurrent.futures.FIRST_COMPLETED)
        return [j for j in jobs if j.done()]

    def _run(self) -> None:
        while not self._closed:
            with self._mu:
                tickets = list(self._pending)
            if not tickets:
                self._work.wait()
                self._work.clear()
                continue
            try:
                rep = self._t.wait_any(tickets, 1000)
                lost = "lost by server"
            except Exception as e:  # noqa: BLE001 - server gone: pending jobs degrade
                rep, lost = {"done": {}, "unknown": tickets}, str(e)
            for ticket in list(rep["done"]) + list(rep["unknown"]):
                with self._mu:
                    job = self._pending.pop(ticket, None)
                if job is None:
                    continue
                replies = rep["done"].get(ticket) or [(False, 0, 0, lost)] * len(job._items)
                try:
                    self._complete(job, replies)
                except Exception as e:  # noqa: BLE001
                    job._q.finalize()
                    job._f.set_exception(e)
                self._slots.release()

    def _stage(self, job: PullJob, hit: int) -> int:
        q, me = job._q, int(self._index.rank())
        remote = [(int(r), int(o), np.asarray(s, dtype=np.int32))
                  for r, o, s in q.full_fragments if r >= 0 and r != me]
        job.local_hit = int(remote[0][1]) if remote else hit
        if remote:
            job._src, off, rs = remote[0]
            st = self._index.allocate_slots(len(rs), FULL)
            if len(st) < len(rs):
                self._recycle(st, FULL)
                self.staging_shortfalls += 1
                return off
            job._items.append((job._src, int(FULL), rs.tolist(), st.tolist()))
            job._staged[FULL] = (off, st)
        swa_rank, swa = int(q.swa_source_rank), np.asarray(q.swa_slots, dtype=np.int32)
        if len(swa) and swa_rank >= 0 and swa_rank != me:
            st = self._index.allocate_slots(len(swa), SWA)
            if len(st) == len(swa):
                job._items.append((swa_rank, int(SWA), swa.tolist(), st.tolist()))
                job._staged[SWA] = (0, st)
            else:
                self._recycle(st, SWA)
                self.staging_shortfalls += 1
            job._src = swa_rank if job._src < 0 else job._src
        mamba = q.mamba
        if mamba is not None and int(mamba[0]) >= 0 and int(mamba[0]) != me:
            st = self._index.allocate_slots(1, MAMBA)
            if len(st) == 1:
                job._items.append((int(mamba[0]), int(MAMBA), [int(mamba[1])], st.tolist()))
                job._staged[MAMBA] = (0, st)
            else:
                self._recycle(st, MAMBA)
                self.staging_shortfalls += 1
            job._src = int(mamba[0]) if job._src < 0 else job._src
        return hit

    def _complete(self, job: PullJob, replies: Sequence[TransferReply]) -> None:
        ok = {ComponentType(k): rep[0] for (_, k, _, _), rep in zip(job._items, replies)}
        self.transfer_failures += sum(not v for v in ok.values())
        job._q.finalize()   # unpinned parent absorbs the promoted leaf; an evicted prefix fails the insert (slots auto-recycled)
        hit, pulled, nbytes = job._hit, 0, 0
        if FULL in job._staged:
            off, st = job._staged[FULL]
            if ok[FULL]:
                self._index.insert(job._hashes[:off + len(st)], st, start=off)
                pulled += len(st)
                nbytes += len(st) * int(self._store.pool(FULL).slot_bytes)
            else:
                self._recycle(st, FULL)
                hit = off
        for kind in (SWA, MAMBA):
            if kind not in job._staged:
                continue
            st = job._staged[kind][1]
            if ok.get(FULL, True) and ok[kind]:
                self._index.insert(job._hashes[:hit], st, component=kind)
                pulled += len(st)
                nbytes += len(st) * int(self._store.pool(kind).slot_bytes)
            else:
                self._recycle(st, kind)
        self._finish(job, hit, pulled, nbytes)

    def _finish(self, job: PullJob, hit: int, pulled: int = 0, nbytes: int = 0) -> None:
        lock = job._lock and not job._cancelled
        r = (self._index.query(job._hashes[:hit], mask=job._mask, local_only=True,
                               lock=lock, update_meta=False) if hit > 0 else None)
        common = int(r.common_hit) if r is not None and int(r.status) == 0 else 0
        if common < hit:
            self.promotion_shortfalls += 1
        if common == 0:
            if r is not None:
                r.finalize()
            res = PullResult(0, _empty(), 0, _empty(), -1, pulled, nbytes, job._src, lambda: None)
        else:
            runs = [np.asarray(s, dtype=np.int32) for _, _, s in r.full_fragments]
            full = np.concatenate(runs)[:common] if runs else _empty()
            res = PullResult(common, full, int(r.swa_start),
                            np.asarray(r.swa_slots, dtype=np.int32),
                            int(r.mamba[1]) if r.mamba else -1, pulled, nbytes, job._src,
                            r.finalize)
        with self._mu:
            job._f.set_result(res)
            if job._cancelled:      # cancelled between the pin above and here
                res.finalize()

    def _recycle(self, slots, kind: ComponentType) -> None:
        if len(slots):
            self._index.recycle_slots(np.asarray(slots, dtype=np.int32), kind)
