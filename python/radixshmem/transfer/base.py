# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The transfer backend interface. A backend moves byte ranges between this process's registered memory and
a peer's published segment; SlotTransfer turns slot ids into those ranges.

Return codes and states are ``radixshmem._core``'s: ``XFER_*``, ``INVALID_BATCH``, ``XferState``. A batch is
validated whole before anything moves, targets one segment and carries one operation. Addresses on the peer
side are absolute, inside ``SegmentInfo.base`` .. ``base + length``.
"""

from __future__ import annotations

import abc
import os
import time
from typing import Optional, Tuple

import numpy as np

from radixshmem._core import SegmentInfo, XferOp, XferState


def integer_array(values, what: str) -> np.ndarray:
    """``values`` as an integer array, without numpy's silent truncation of floats; TypeError otherwise."""
    a = np.asarray(values)
    if a.size == 0:
        return a.astype(np.int64)
    if a.dtype.kind not in "iu":
        raise TypeError(f"{what} must be integers, not {a.dtype}")
    return a


def as_u64(values, what: str) -> Optional[np.ndarray]:
    """Non-negative integers as a uint64 array, or None if any is negative."""
    a = integer_array(values, what)
    if a.size and a.dtype.kind == "i" and a.min() < 0:
        return None
    return np.ascontiguousarray(a, dtype=np.uint64)


class TransferBackend(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    @abc.abstractmethod
    def local_segment(self) -> str:
        """The name peers open this process's memory by; published in the data-plane registry."""

    @abc.abstractmethod
    def register_memory(self, addr: int, length: int, location: str = "") -> int: ...

    @abc.abstractmethod
    def unregister_memory(self, addr: int) -> int: ...

    @abc.abstractmethod
    def open_segment(self, name: str) -> Tuple[int, Optional[SegmentInfo]]:
        """(XFER_OK, the peer's segment) or (error, None). Opening our own segment is allowed."""

    @abc.abstractmethod
    def close_segment(self, seg_id: int) -> None: ...

    @abc.abstractmethod
    def submit(self, op: XferOp, seg_id: int, local: np.ndarray, remote: np.ndarray, lengths: np.ndarray) -> int:
        """One batch of ranges (uint64 arrays). A batch handle, or INVALID_BATCH if any range is refused."""

    @abc.abstractmethod
    def poll(self, handle: int) -> Tuple[XferState, int]:
        """(state, bytes done)."""

    @abc.abstractmethod
    def release(self, handle: int) -> int:
        """XFER_OK, or XFER_EBUSY while the batch still runs (the handle stays valid)."""

    def wait(self, handle: int, timeout_ms: int) -> XferState:
        # Spin for the first millisecond: any sleep, even sleep(0), costs the 50 us timer slack.
        start = time.monotonic()
        deadline = None if timeout_ms < 0 else start + timeout_ms / 1000
        while True:
            state, _ = self.poll(handle)
            if state != XferState.PENDING:
                return state
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                return XferState.TIMEOUT
            if now - start < 0.001:
                os.sched_yield()
            else:
                time.sleep(0.00005)

    @abc.abstractmethod
    def close(self) -> None: ...
