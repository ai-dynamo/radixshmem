# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Slot-granular transfers between this node's SlotStore and peers' stores, over any TransferBackend.

A peer is attached by its segment name and its published StoreGeometry; slot ids then mean the same bytes on
both sides or the attach is refused. read/write turn slot pairs into address ranges (``_core.plan_slots``,
runs contiguous on both sides fused) and submit them as one batch. A peer cannot be detached while one of its
batches is unreleased.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

from radixshmem import _core
from radixshmem._core import (COMPONENT_MASK_FULL, COMPONENT_MASK_MAMBA, COMPONENT_MASK_SWA, INVALID_BATCH,
                              PEER_SLOT_BYTES_UNCHECKED, XFER_EBUSY, XFER_EGEOMETRY, XFER_EINVAL, XFER_OK,
                              ComponentType, StoreGeometry, XferOp, XferState)
from radixshmem.transfer.base import TransferBackend

COMPONENT_MASK_ALL = COMPONENT_MASK_FULL | COMPONENT_MASK_SWA | COMPONENT_MASK_MAMBA


def _warn(msg: str) -> None:
    print(f"radixshmem: SlotTransfer: {msg}", file=sys.stderr)


@dataclass
class _Peer:
    seg_id: int
    geo: StoreGeometry   # absolute bases
    inflight: int = 0    # submitted, not yet released


class SlotTransfer:
    def __init__(self, store, backend: TransferBackend):
        self._store = store
        self._backend = backend
        self._mu = threading.Lock()
        self._peers: Dict[str, _Peer] = {}
        self._batch_peer: Dict[int, str] = {}

    @property
    def store(self):
        return self._store

    @property
    def backend(self) -> TransferBackend:
        return self._backend

    @property
    def local_segment(self) -> str:
        return self._backend.local_segment

    def register_store(self, location: str = "") -> int:
        """Hand the store's registerable range to the backend."""
        if not self._store.valid:
            return XFER_EINVAL
        return self._backend.register_memory(self._store.data_addr, self._store.data_bytes, location)

    def attach_peer(self, peer: str, published: Union[StoreGeometry, int, None]) -> Tuple[int, object]:
        """Open the peer's segment and check its geometry. ``published`` is the peer's StoreGeometry from the
        registry, None to assume our layout, or (FULL-only stores) the peer's slot stride /
        PEER_SLOT_BYTES_UNCHECKED. Returns (rc, the peer's geometry with absolute bases, or None): a
        StoreGeometry, or for a stride the FULL PoolGeometry."""
        if isinstance(published, int):
            return self._attach_stride(peer, published)
        rc, geo = self._attach(peer, published)
        return rc, geo if rc == XFER_OK else None

    def _attach_stride(self, peer: str, stride: int) -> Tuple[int, object]:
        if not self._store.valid or stride == 0:
            return XFER_EINVAL, None
        if self._store.pool_mask != COMPONENT_MASK_FULL:
            _warn("attach_peer(stride) is for FULL-only stores; this one has more pools — pass the peer's "
                  "published StoreGeometry")
            return XFER_EINVAL, None
        sb = self._store.slot_bytes
        if stride != PEER_SLOT_BYTES_UNCHECKED and stride != sb:
            _warn(f"peer '{peer}' has a {stride}-byte slot stride, we have {sb} — slot ids would not mean the "
                  "same bytes on both sides")
            return XFER_EGEOMETRY, None
        rc, geo = self._attach(peer, None)
        return rc, geo[ComponentType.FULL] if rc == XFER_OK else None

    def _attach(self, peer: str, published: Optional[StoreGeometry]) -> Tuple[int, Optional[StoreGeometry]]:
        if not peer or not self._store.valid:
            return XFER_EINVAL, None
        if published is not None and (not published.pool_mask & COMPONENT_MASK_FULL or
                                      published.pool_mask & ~COMPONENT_MASK_ALL):
            return XFER_EINVAL, None
        mine = self._store.layout()
        # Held across open_segment: it keeps lookup-then-insert atomic.
        with self._mu:
            p = self._peers.get(peer)
            if p is not None:
                # The cache never outvotes the registry.
                if published is not None and not published.same_strides(p.geo):
                    _warn(f"peer '{peer}' was attached with a different pool layout than the one now published")
                    return XFER_EGEOMETRY, None
                return XFER_OK, p.geo
            # Strides are checked before opening; the segment fit only after.
            if published is not None and not published.same_strides(mine):
                _, _, why = _core.resolve_peer_geometry(mine, published, 0, 0)
                _warn(f"peer '{peer}' {why}")
                return XFER_EGEOMETRY, None
            rc, info = self._backend.open_segment(peer)
            if rc != XFER_OK:
                return rc, None
            ok, geo, why = _core.resolve_peer_geometry(mine, published, info.base, info.length)
            if not ok:
                _warn(f"peer '{peer}' {why}")
                self._backend.close_segment(info.id)
                return XFER_EGEOMETRY, None
            self._peers[peer] = _Peer(info.id, geo)
            return XFER_OK, geo

    def detach_peer(self, peer: str) -> int:
        """Close the peer's segment. XFER_EBUSY while any of its batches is unreleased."""
        with self._mu:
            p = self._peers.get(peer)
            if p is None:
                return XFER_OK
            if p.inflight > 0:
                _warn(f"peer '{peer}' has {p.inflight} batch(es) in flight; release them before detaching")
                return XFER_EBUSY
            del self._peers[peer]
        self._backend.close_segment(p.seg_id)
        return XFER_OK

    def peer_geometry(self, peer: str) -> Optional[StoreGeometry]:
        with self._mu:
            p = self._peers.get(peer)
            return p.geo if p is not None else None

    def read(self, peer: str, remote_slots, local_slots, kind=ComponentType.FULL) -> Tuple[int, int]:
        """local_slots[i] <- the peer's remote_slots[i] in pool ``kind``. Returns (batch handle, ranges);
        the handle is INVALID_BATCH on rejection."""
        return self._submit(XferOp.READ, peer, kind, local_slots, remote_slots)

    def write(self, peer: str, local_slots, remote_slots, kind=ComponentType.FULL) -> Tuple[int, int]:
        """The peer's remote_slots[i] <- local_slots[i] in pool ``kind``. Returns (batch handle, ranges)."""
        return self._submit(XferOp.WRITE, peer, kind, local_slots, remote_slots)

    def _submit(self, op: XferOp, peer: str, kind, local_slots, remote_slots) -> Tuple[int, int]:
        with self._mu:
            p = self._peers.get(peer)
        if p is None:
            _warn(f"peer '{peer}' is not attached")
            return INVALID_BATCH, 0
        local, remote, lengths, why = _core.plan_slots(self._store, p.geo, kind, local_slots, remote_slots)
        if local is None:
            _warn(f"{why} (peer '{peer}')")
            return INVALID_BATCH, 0
        # Hold the peer until release(), if it is still the segment the ranges were planned against.
        with self._mu:
            if self._peers.get(peer) is not p:
                _warn(f"peer '{peer}' was detached while the batch was being built")
                return INVALID_BATCH, 0
            p.inflight += 1
        h = self._backend.submit(op, p.seg_id, local, remote, lengths)
        with self._mu:
            if h == INVALID_BATCH:
                p.inflight -= 1
                return INVALID_BATCH, 0
            self._batch_peer[h] = peer
        return h, len(local)

    def poll(self, handle: int) -> Tuple[XferState, int]:
        """(XferState, bytes done)."""
        return self._backend.poll(handle)

    def wait(self, handle: int, timeout_ms: int) -> XferState:
        return self._backend.wait(handle, timeout_ms)

    def release(self, handle: int) -> int:
        """XFER_OK, or XFER_EBUSY while the batch still runs; a refused release keeps the peer held."""
        rc = self._backend.release(handle)
        if rc == XFER_EBUSY:
            return rc
        with self._mu:
            peer = self._batch_peer.pop(handle, None)
            p = self._peers.get(peer) if peer is not None else None
            if p is not None and p.inflight > 0:
                p.inflight -= 1
        return rc
