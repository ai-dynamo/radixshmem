# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write-back of background-evicted KV: a thread of the server process hands
``WritebackTask`` batches to a ``WritebackHandler``."""

from __future__ import annotations

import importlib
import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence, Set

import numpy as np


@dataclass
class WritebackTask:
    """An evicted leaf with its root path [0, end): [evicted_from, end) left the
    tree, [0, evicted_from) stays pinned; every slot stays put until done."""

    task_id: int
    hashes: np.ndarray        # uint64[end]
    full_slots: np.ndarray    # int32[end]
    swa_slots: np.ndarray     # int32[end], -1 where the tree has none; empty without an SWA pool
    mamba_ends: np.ndarray    # uint32[k]: END positions of the path's Mamba checkpoints
    mamba_slots: np.ndarray   # int32[k]
    evicted_from: int

    @property
    def end(self) -> int:
        return len(self.hashes)


class WritebackHandler:
    """``write`` calls ``done(task_ids)`` from any thread once the slots may be
    reused; tasks not done when ``stop`` returns are released anyway."""

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
    return getattr(importlib.import_module(module), attr)(config)


class _Worker:
    def __init__(self, index, handler: WritebackHandler, client, batch: int = 64, poll_ms: int = 200):
        self._index, self._handler, self._client = index, handler, client
        self._batch, self._poll_ms = batch, poll_ms
        self._stop = threading.Event()
        self._mu = threading.Lock()
        self._open: Set[int] = set()   # handed to the handler, not done yet
        self.failed = 0
        self._thread = threading.Thread(target=self._run, name="radixshmem-writeback", daemon=True)

    def start(self) -> None:
        self._handler.start(self._client)
        self._thread.start()

    def done(self, task_ids: Sequence[int]) -> None:
        with self._mu:
            ids = [int(i) for i in task_ids if int(i) in self._open]
            self._open.difference_update(ids)
            index = self._index
        if ids and index is not None:
            index.complete_writeback(ids)

    def _run(self) -> None:
        while not self._stop.is_set():
            raw = self._index.pop_writeback(self._batch, self._poll_ms)
            if not raw:
                continue
            tasks = [WritebackTask(*r) for r in raw]
            ids = [t.task_id for t in tasks]
            with self._mu:
                self._open.update(ids)
            try:
                self._handler.write(tasks, self.done)
            except Exception as e:  # noqa: BLE001 - the handler's tasks are given up, the thread goes on
                with self._mu:
                    left = [i for i in ids if i in self._open]
                self.failed += len(left)
                print(f"radixshmem: write-back handler failed on {len(left)} task(s): "
                      f"{type(e).__name__}: {e}", file=sys.stderr)
                self.done(left)

    def stop(self) -> None:
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join()
        try:
            self._handler.stop()
        except Exception as e:  # noqa: BLE001
            print(f"radixshmem: write-back handler stop failed: {type(e).__name__}: {e}", file=sys.stderr)
        # The handler may keep `done`, which must not keep the index alive.
        with self._mu:
            self._index = None
            self._open.clear()
        client, self._client = self._client, None
        if client is not None:
            client.close()
