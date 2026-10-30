# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write-back of background-evicted KV: a thread of the server process hands
``WritebackTask`` batches to a ``WritebackHandler``."""

from __future__ import annotations

import importlib
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import numpy as np

_STOP_WAIT_S = 5.0   # how long close() waits for a running write() before handler.stop()


@dataclass
class WritebackTask:
    """An evicted leaf with its root path [0, end): [evicted_from, end) left the
    tree, [0, evicted_from) is still in it, and no slot is recycled until done."""

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
