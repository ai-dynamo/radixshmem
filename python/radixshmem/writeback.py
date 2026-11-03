# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write-back of background-evicted KV: a thread of the server process hands
``WritebackTask`` batches to a ``WritebackHandler``; ``PeerWriteback`` pushes
them to the decode nodes."""

from __future__ import annotations

import importlib
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import numpy as np

from radixshmem._core import INVALID_COMPONENT_SLOT, MAX_REMOTE_OP_BLOCKS
from radixshmem.client import MODE_READY, PushJob

_STOP_WAIT_S = 5.0   # how long close() waits for a running write() before handler.stop()


@dataclass
class WritebackTask:
    """An evicted chain (a leaf and the ancestors it left childless) with its
    root path [0, end): [evicted_from, end) left the tree, [0, evicted_from) is
    still in it, and no slot is recycled until done."""

    task_id: int
    hashes: np.ndarray        # uint64[end]
    full_slots: np.ndarray    # int32[end]
    swa_slots: np.ndarray     # int32[end], -1 where the tree has none; empty without an SWA pool
    mamba_ends: np.ndarray    # uint32[k]: END positions of the path's Mamba checkpoints
    mamba_slots: np.ndarray   # int32[k]
    evicted_from: int
    generation: int           # the index's reset() count at eviction
    _check: Optional[Callable[[WritebackTask], bool]] = field(default=None, repr=False, compare=False)

    @property
    def end(self) -> int:
        return len(self.hashes)

    def valid(self) -> bool:
        return self._check is not None and self._check(self)


class WritebackHandler:
    """``start(client)`` runs once the server is ready. ``write`` calls
    ``done(task_ids)`` from any thread, possibly later; data read from the slots
    may be stored only if ``task.valid()`` is still True after the read (a
    ``reset()`` or ``done`` ends it). ``close()`` calls ``stop`` first, while the
    client can still pull, once ``write`` returned or after 5 s; tasks not done
    by then are dropped."""

    def start(self, client) -> None:
        pass

    def write(self, tasks: Sequence[WritebackTask], done: Callable[[Sequence[int]], None]) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        pass

    def stats(self) -> Dict[str, int]:
        return {}


class PeerWriteback(WritebackHandler):
    """Pushes each evicted path to a ready decode node with ``push_async``,
    picked by the hash of its first block so that a prefix tree stays on one
    node. On prefill nodes it extends them with the decode nodes' memory.
    ``config``: ``timeout_ms`` (default 30000)."""

    def __init__(self, config: Dict[str, Any]):
        self.timeout_ms = int(config.get("timeout_ms", 30000))
        self.client = None
        self._window = 0
        self._decode = False
        self._stats = dict.fromkeys(("pushed", "incomplete", "no_target", "too_long"), 0)

    def start(self, client) -> None:
        self.client = client
        self._window = int(client.geometry["pools"].get("swa", {}).get("window_blocks", 0))
        self._decode = client.info.role == "decode"
        if self._decode:
            print("radixshmem: PeerWriteback on a decode node pushes nothing", file=sys.stderr)

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def write(self, tasks: Sequence[WritebackTask], done: Callable[[Sequence[int]], None]) -> None:
        nodes = [] if self._decode else self.client.nodes()
        jobs = []
        for t in tasks:
            rank = self._choose_target(nodes, t)
            if rank < 0:
                self._stats["no_target"] += 1
            elif t.end > MAX_REMOTE_OP_BLOCKS:
                self._stats["too_long"] += 1
            else:
                jobs.append(self._push(t, rank))
        for job in jobs:
            self._stats["pushed" if job.wait().complete else "incomplete"] += 1
        done([t.task_id for t in tasks])

    def _choose_target(self, nodes: List[Dict[str, Any]], t: WritebackTask) -> int:
        """A ready decode node other than this one, by ``t``'s first block hash; -1 if none."""
        me = self.client.rank()
        ranks = sorted(int(r["rank"]) for r in nodes if r.get("mode") == MODE_READY and r.get("data_plane")
                       and r.get("role") == "decode" and int(r.get("rank", -1)) not in (-1, me))
        return ranks[int(t.hashes[0]) % len(ranks)] if ranks else -1

    def _push(self, t: WritebackTask, rank: int) -> PushJob:
        swa = t.swa_slots[max(0, t.end - self._window):]
        live = len(swa) > 0 and bool((swa != INVALID_COMPONENT_SLOT).all())
        mamba = int(t.mamba_slots[-1]) if len(t.mamba_ends) and int(t.mamba_ends[-1]) == t.end else -1
        return self.client.push_async(rank, t.hashes, t.full_slots, swa if live else None, mamba,
                                      valid=t.valid, timeout_ms=self.timeout_ms)


def load_handler(spec: str, config: Dict[str, Any]) -> WritebackHandler:
    module, _, attr = spec.partition(":")
    if not module or not attr:
        raise ValueError(f"write-back handler {spec!r} is not 'module:factory'")
    handler = getattr(importlib.import_module(module), attr)(config)
    if not isinstance(handler, WritebackHandler):
        raise TypeError(f"{spec} returned {type(handler).__name__}, not a WritebackHandler")
    return handler


class _Worker:
    def __init__(self, index, handler: WritebackHandler, client, batch: int = 64, poll_ms: int = 200):
        self._index, self._handler, self._client = index, handler, client
        self._batch, self._poll_ms = batch, poll_ms
        self._stop = threading.Event()
        self._life = threading.Lock()   # start() and stop() do not overlap
        self._started = False
        self._mu = threading.Lock()     # held across index calls, so stop() waits for them
        self._open: Set[int] = set()    # handed to the handler, not done yet
        self.failed = 0
        self._thread = threading.Thread(target=self._run, name="radixshmem-writeback", daemon=True)

    def start(self) -> None:
        with self._life:
            if self._stop.is_set():
                return
            self._handler.start(self._client)
            self._started = True
            self._thread.start()

    def done(self, task_ids: Sequence[int]) -> None:
        with self._mu:
            ids = [int(i) for i in task_ids if int(i) in self._open]
            self._open.difference_update(ids)
            if ids and self._index is not None:
                self._index.complete_writeback(ids)

    def _valid(self, task: WritebackTask) -> bool:
        with self._mu:
            return (task.task_id in self._open and self._index is not None
                    and self._index.reset_generation() == task.generation)

    def _run(self) -> None:
        while not self._stop.is_set():
            ids: List[int] = []
            try:
                raw = self._index.pop_writeback(self._batch, self._poll_ms)
                ids = [int(r[0]) for r in raw]
                with self._mu:
                    self._open.update(ids)
                if raw:
                    self._handler.write([WritebackTask(*r, _check=self._valid) for r in raw], self.done)
            except BaseException as e:  # noqa: BLE001 - the batch is given up, the thread goes on
                self._give_up(ids, e)

    def _give_up(self, ids: List[int], e: BaseException) -> None:
        with self._mu:
            left = [i for i in ids if i in self._open]
        self.failed += len(left)
        try:
            print(f"radixshmem: write-back failed on {len(left)} task(s): {type(e).__name__}: {e}",
                  file=sys.stderr)
        except Exception:  # noqa: BLE001 - no stderr: the tasks are given up all the same
            pass
        self.done(left)
        if not ids:
            self._stop.wait(self._poll_ms / 1000)

    def stop(self) -> None:
        self._stop.set()
        with self._life:
            if self._started:
                if self._thread.ident is not None:
                    self._thread.join(_STOP_WAIT_S)
                if self._thread.is_alive():
                    print(f"radixshmem: write-back handler still in write() after {_STOP_WAIT_S:g}s; "
                          "stopping it", file=sys.stderr)
                try:
                    self._handler.stop()
                except Exception as e:  # noqa: BLE001
                    print(f"radixshmem: write-back handler stop failed: {type(e).__name__}: {e}",
                          file=sys.stderr)
        # The handler may keep `done`, which must not keep the index alive.
        with self._mu:
            self._index = None
            self._open.clear()
        client, self._client = self._client, None
        if client is not None:
            client.close()
