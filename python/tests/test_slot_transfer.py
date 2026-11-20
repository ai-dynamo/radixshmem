# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SlotTransfer's geometry contract and peer bookkeeping, against a stub backend: no engine, no RDMA.

The failure it guards: a peer built with a different slot stride whose published length still divides evenly
by ours. Every length check passes, slot ids get rescaled, and the transfer moves the wrong bytes. attach_peer
must refuse it on the stride alone. detach_peer must refuse while a batch against the peer is unreleased.
"""

import os
import uuid

import numpy as np
import pytest

from radixshmem._data import (COMPONENT_MASK_FULL, COMPONENT_MASK_MAMBA, COMPONENT_MASK_SWA, INVALID_BATCH,
                              PEER_SLOT_BYTES_UNCHECKED, XFER_EBUSY, XFER_EGEOMETRY, XFER_EINVAL, XFER_ENOENT,
                              XFER_OK, ComponentType, PoolGeometry, SegmentInfo, SlotPoolConfig, SlotStore,
                              SlotStoreConfig, SlotTransfer, StoreGeometry, TransferBackend, XferOp, XferState)

SLOTS, SLOT_BYTES = 64, 65536
BASE = 0x700000000000
F, W, M = ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA
MASK_ALL = COMPONENT_MASK_FULL | COMPONENT_MASK_SWA | COMPONENT_MASK_MAMBA


class StubBackend(TransferBackend):
    """Publishes whatever segment the test offers and records what submit() was handed. Batches complete at
    once and move nothing."""

    def __init__(self):
        self.offered, self.last = {}, None
        self.reg = None
        self.opens = self.closes = 0
        self.release_rc = XFER_OK
        self._next_id, self._next_batch = 1, 100

    def offer(self, name, base, length):
        self.offered[name] = (base, length)

    name = "stub"
    local_segment = "stub-local"

    def register_memory(self, addr, length, location=""):
        self.reg = (addr, length)
        return XFER_OK

    def unregister_memory(self, addr):
        return XFER_OK

    def open_segment(self, name):
        if name not in self.offered:
            return XFER_ENOENT, None
        self.opens += 1
        base, length = self.offered[name]
        self._next_id += 1
        return XFER_OK, SegmentInfo(self._next_id - 1, base, length)

    def close_segment(self, seg_id):
        self.closes += 1

    def submit(self, op, seg_id, local, remote, lengths):
        self.last = (op, list(local), list(remote), list(lengths))
        self._next_batch += 1
        return self._next_batch - 1

    def poll(self, handle):
        return XferState.COMPLETED, 0

    def release(self, handle):
        return self.release_rc

    def close(self):
        pass


@pytest.fixture
def make_store():
    names = []

    def make(pools=False):
        name = f"/tst_{uuid.uuid4().hex[:12]}"
        names.append(name)
        cfg = SlotStoreConfig(name, SlotPoolConfig(SLOTS, SLOT_BYTES),
                              swa=SlotPoolConfig(32, 4096) if pools else SlotPoolConfig(),
                              mamba=SlotPoolConfig(4, 8192) if pools else SlotPoolConfig(),
                              slot_align=4096, hugepage_path="", prefault=False)
        return SlotStore.create(cfg)

    yield make
    for n in names:
        SlotStore.destroy(n, "")


def attached(store, name="peer", length=None):
    be = StubBackend()
    be.offer(name, BASE, length if length is not None else 64 * SLOT_BYTES)
    x = SlotTransfer(store, be)
    assert x.register_store() == XFER_OK
    return be, x


def test_matching_stride(make_store):
    s = make_store()
    be, x = attached(s, "peerA", 128 * SLOT_BYTES)
    assert be.reg == (s.data_addr, s.data_bytes)
    rc, g = x.attach_peer("peerA", SLOT_BYTES)
    assert rc == XFER_OK
    assert (g.slot_bytes, g.num_slots, g.base) == (SLOT_BYTES, 128, BASE)


def test_mismatched_stride_that_divides(make_store):
    s = make_store()
    peer_stride, peer_slots = 4096, 1024
    assert (peer_stride * peer_slots) % SLOT_BYTES == 0   # a length-only check would pass
    be, x = attached(s, "peerB", peer_stride * peer_slots)
    assert x.attach_peer("peerB", peer_stride) == (XFER_EGEOMETRY, None)
    # Refused before the segment was opened, and nothing can be transferred to it.
    assert x.peer_geometry("peerB") is None and be.opens == 0
    assert x.read("peerB", [0], [0])[0] == INVALID_BATCH
    assert x.write("peerB", [0], [0])[0] == INVALID_BATCH


def test_ragged_length(make_store):
    s = make_store()
    be, x = attached(s, "peerC", 8 * SLOT_BYTES + 4096)
    assert x.attach_peer("peerC", SLOT_BYTES)[0] == XFER_EGEOMETRY
    assert be.opens == 1 and be.closes == 1   # the segment opened to find out is closed again
    be.offer("peerD", BASE, SLOT_BYTES // 2)
    assert x.attach_peer("peerD", SLOT_BYTES)[0] == XFER_EGEOMETRY


def test_reattach_stride_change(make_store):
    s = make_store()
    be, x = attached(s, "peerE", 32 * SLOT_BYTES)
    assert x.attach_peer("peerE", SLOT_BYTES)[0] == XFER_OK
    assert x.attach_peer("peerE", SLOT_BYTES)[0] == XFER_OK and be.opens == 1   # served from the cache
    assert x.attach_peer("peerE", 4096)[0] == XFER_EGEOMETRY                  # never outvotes a new stride
    assert x.peer_geometry("peerE")[F].slot_bytes == SLOT_BYTES


def test_unchecked_opt_out(make_store):
    s = make_store()
    be, x = attached(s, "peerF", 16 * SLOT_BYTES)
    rc, g = x.attach_peer("peerF", PEER_SLOT_BYTES_UNCHECKED)
    assert rc == XFER_OK and (g.slot_bytes, g.num_slots) == (SLOT_BYTES, 16)
    assert x.attach_peer("peerF", SLOT_BYTES)[0] == XFER_OK
    assert x.attach_peer("peerF", 4096)[0] == XFER_EGEOMETRY
    assert x.attach_peer("peerG", 0)[0] == XFER_EINVAL


def test_ranges_from_the_attached_stride(make_store):
    s = make_store()
    be, x = attached(s, "peerH")
    assert x.attach_peer("peerH", SLOT_BYTES)[0] == XFER_OK
    # A run contiguous on both sides fuses; the outlier does not.
    h, n = x.read("peerH", [20, 21, 22, 55], [10, 11, 12, 40])
    assert h != INVALID_BATCH and n == 2
    op, local, remote, lengths = be.last
    assert op == XferOp.READ
    assert local == [s.data_addr + s.slot_offset(10), s.data_addr + s.slot_offset(40)]
    assert remote == [BASE + 20 * SLOT_BYTES, BASE + 55 * SLOT_BYTES]
    assert lengths == [3 * SLOT_BYTES, SLOT_BYTES]
    assert x.release(h) == XFER_OK


def test_slot_range_checks(make_store):
    s = make_store()
    be, x = attached(s, "peerI", 16 * SLOT_BYTES)   # the peer is smaller
    assert x.attach_peer("peerI", SLOT_BYTES)[1].num_slots == 16
    assert x.read("peerI", [15], [63])[0] != INVALID_BATCH
    assert x.read("peerI", [16], [63])[0] == INVALID_BATCH       # past the end of the peer
    assert x.read("peerI", [15], [SLOTS])[0] == INVALID_BATCH    # past the end of us
    assert x.read("peerI", [15], [-1])[0] == INVALID_BATCH
    assert x.write("peerI", [63], [-1])[0] == INVALID_BATCH
    with pytest.raises(ValueError):
        x.read("peerI", [1, 2], [1])
    with pytest.raises(ValueError):
        x.read("peerI", [], [])


def test_slot_ids_are_not_cast_unsafely(make_store):
    s = make_store()
    be, x = attached(s, "peerU")
    assert x.attach_peer("peerU", SLOT_BYTES)[0] == XFER_OK
    # Values that an unsafe cast would wrap into valid slots (2**32 + 3 -> 3, 2**63 -> 0) are out of range.
    assert x.read("peerU", np.array([2**32 + 3], dtype=np.int64), [0])[0] == INVALID_BATCH
    assert x.read("peerU", [0], np.array([2**63], dtype=np.uint64))[0] == INVALID_BATCH
    assert x.read("peerU", [2**40], [0])[0] == INVALID_BATCH
    with pytest.raises(TypeError):
        x.read("peerU", [1.7], [0])
    with pytest.raises(TypeError):
        x.write("peerU", np.array([1.0]), [0])
    assert be.last is None


def test_native_arrays_are_not_cast_unsafely(make_store):
    from radixshmem import _core
    s = make_store()
    peer = StoreGeometry()
    peer[F] = PoolGeometry(BASE, 8, SLOT_BYTES)
    for big in (np.array([2**32 + 3], dtype=np.int64), np.array([2**63], dtype=np.uint64), [2**64 + 3]):
        assert _core.plan_slots(s, peer, F, big, np.array([0], dtype=np.int32))[0] is None
    for bad in (np.array([1.7]), [1.7], [True], "0"):
        with pytest.raises(TypeError):
            _core.plan_slots(s, peer, F, bad, [0])
    one = np.array([1], dtype=np.uint64)
    with pytest.raises(TypeError):
        _core.copy_ranges(XferOp.READ, np.array([-1], dtype=np.int64), one, one)


def test_numpy_slots(make_store):
    s = make_store()
    be, x = attached(s, "peerN")
    assert x.attach_peer("peerN", SLOT_BYTES)[0] == XFER_OK
    h, n = x.write("peerN", np.arange(4, 8, dtype=np.int32), np.arange(0, 4, dtype=np.int64))
    assert h != INVALID_BATCH and n == 1 and be.last[0] == XferOp.WRITE and be.last[3] == [4 * SLOT_BYTES]


def test_detach(make_store):
    s = make_store()
    be, x = attached(s, "peerJ", 8 * SLOT_BYTES)
    assert x.attach_peer("peerJ", SLOT_BYTES)[0] == XFER_OK
    assert x.detach_peer("peerJ") == XFER_OK and be.closes == 1
    assert x.peer_geometry("peerJ") is None
    assert x.detach_peer("peerJ") == XFER_OK and be.closes == 1
    assert x.attach_peer("peerJ", SLOT_BYTES)[0] == XFER_OK and be.opens == 2


def test_detach_while_inflight(make_store):
    s = make_store()
    be, x = attached(s, "peerK", 8 * SLOT_BYTES)
    assert x.attach_peer("peerK", SLOT_BYTES)[0] == XFER_OK
    h1, _ = x.read("peerK", [0, 1], [0, 1])
    assert h1 != INVALID_BATCH
    assert x.detach_peer("peerK") == XFER_EBUSY and be.closes == 0
    assert x.peer_geometry("peerK") is not None

    # A release the backend refuses frees nothing and keeps the peer held.
    be.release_rc = XFER_EBUSY
    assert x.release(h1) == XFER_EBUSY
    assert x.detach_peer("peerK") == XFER_EBUSY
    be.release_rc = XFER_OK

    h2, _ = x.write("peerK", [0, 1], [0, 1])
    assert h2 not in (INVALID_BATCH, h1)
    assert x.release(h1) == XFER_OK
    assert x.detach_peer("peerK") == XFER_EBUSY
    x.release(h1)   # a second release of the same handle must not count twice
    assert x.detach_peer("peerK") == XFER_EBUSY and be.closes == 0
    x.release(h2)
    assert x.detach_peer("peerK") == XFER_OK and be.closes == 1

    # A rejected batch reserves nothing.
    assert x.attach_peer("peerK", SLOT_BYTES)[0] == XFER_OK
    assert x.read("peerK", [0, 99], [0, 1])[0] == INVALID_BATCH
    assert x.detach_peer("peerK") == XFER_OK and be.closes == 2


def test_rejected_submit_releases_the_hold(make_store):
    s = make_store()
    be, x = attached(s, "peerL")
    assert x.attach_peer("peerL", SLOT_BYTES)[0] == XFER_OK
    be.submit = lambda *a: INVALID_BATCH
    assert x.read("peerL", [0], [0]) == (INVALID_BATCH, 0)
    assert x.detach_peer("peerL") == XFER_OK


def published_layout():
    pub = StoreGeometry()
    pub[F] = PoolGeometry(0, 128, SLOT_BYTES)
    pub[W] = PoolGeometry(128 * SLOT_BYTES, 16, 4096)
    pub[M] = PoolGeometry(128 * SLOT_BYTES + 16 * 4096, 4, 8192)
    return pub


def test_multi_pool_attach(make_store):
    s = make_store(pools=True)
    pub = published_layout()   # twice the FULL slots, half the SWA, same strides
    be, x = attached(s, "peerM", pub[M].base + 4 * 8192)
    rc, g = x.attach_peer("peerM", pub)
    assert rc == XFER_OK and g.pool_mask == MASK_ALL
    assert (g[F].base, g[F].num_slots) == (BASE, 128)
    assert (g[W].base, g[W].num_slots) == (BASE + 128 * SLOT_BYTES, 16)
    assert (g[M].base, g[M].slot_bytes) == (BASE + pub[M].base, 8192)
    assert x.attach_peer("peerM", SLOT_BYTES)[0] == XFER_EINVAL   # the stride form is FULL-only

    other = published_layout()
    other[W] = PoolGeometry(other[W].base, 16, 8192)
    assert x.attach_peer("peerM", other)[0] == XFER_EGEOMETRY
    assert x.attach_peer("peerM", pub)[0] == XFER_OK and be.opens == 1


def test_multi_pool_refusals(make_store):
    s = make_store(pools=True)
    be, x = attached(s, "peerN", s.data_bytes)
    mine = s.layout()

    def variant(**pools):
        g = s.layout()
        for kind, pool in pools.items():
            g[{"F": F, "W": W, "M": M}[kind]] = pool
        return g

    no_mamba = s.layout()
    no_mamba.pool_mask = COMPONENT_MASK_FULL | COMPONENT_MASK_SWA
    assert x.attach_peer("peerN", no_mamba)[0] == XFER_EGEOMETRY
    assert x.attach_peer("peerN", variant(W=PoolGeometry(mine[W].base, mine[W].num_slots, 8192)))[0] == XFER_EGEOMETRY
    assert x.attach_peer("peerN", variant(M=PoolGeometry(mine[M].base, 5, 8192)))[0] == XFER_EGEOMETRY
    assert x.attach_peer("peerN", variant(W=PoolGeometry(mine[F].base, 32, 4096)))[0] == XFER_EGEOMETRY
    no_full = s.layout()
    no_full.pool_mask = COMPONENT_MASK_SWA | COMPONENT_MASK_MAMBA
    assert x.attach_peer("peerN", no_full)[0] == XFER_EINVAL
    assert be.opens == 2 and be.closes == 2   # only the fit checks opened
    assert x.attach_peer("peerN", mine)[0] == XFER_OK

    # None on a multi-pool store: our layout assumed, still checked to fit.
    be.offer("peerO", BASE, s.data_bytes - 1)
    assert x.attach_peer("peerO", None)[0] == XFER_EGEOMETRY
    be.offer("peerP", BASE, s.data_bytes)
    rc, g = x.attach_peer("peerP", None)
    assert rc == XFER_OK and g[M].base == BASE + mine[M].base


def test_multi_pool_ranges(make_store):
    s = make_store(pools=True)
    be, x = attached(s, "peerQ", s.data_bytes)
    mine = s.layout()
    assert x.attach_peer("peerQ", mine)[0] == XFER_OK

    h, n = x.read("peerQ", [10, 11, 12], [2, 3, 4], kind=W)
    assert h != INVALID_BATCH and n == 1
    _, local, remote, lengths = be.last
    assert local == [s.data_addr + s.slot_offset(2, W)] and local != [s.data_addr + s.slot_offset(2, F)]
    assert remote == [BASE + mine[W].base + 10 * 4096] and lengths == [3 * 4096]

    x.read("peerQ", [10, 11, 12], [2, 3, 4])
    assert be.last[1] == [s.data_addr + s.slot_offset(2, F)] and be.last[2] == [BASE + 10 * SLOT_BYTES]

    h, n = x.write("peerQ", [3], [0], kind=M)
    assert be.last == (XferOp.WRITE, [s.data_addr + s.slot_offset(3, M)], [BASE + mine[M].base], [8192])

    assert x.read("peerQ", [0], [32], kind=W)[0] == INVALID_BATCH
    assert x.read("peerQ", [4], [3], kind=M)[0] == INVALID_BATCH
    with pytest.raises(ValueError):
        x.read("peerQ", [0], [3], kind=3)

    s1 = make_store()
    be1, x1 = attached(s1, "peerR", 8 * SLOT_BYTES)
    assert x1.attach_peer("peerR", SLOT_BYTES)[0] == XFER_OK
    assert x1.read("peerR", [0], [3], kind=W)[0] == INVALID_BATCH   # no SWA pool here


def test_plan_slots_direct(make_store):
    from radixshmem import _core
    s = make_store()
    peer = StoreGeometry()
    peer[F] = PoolGeometry(BASE, 8, SLOT_BYTES)
    local, remote, lengths, why = _core.plan_slots(s, peer, F, [0, 1, 2, 5], [3, 4, 5, 7])
    assert why == "" and local.dtype == np.uint64
    assert list(lengths) == [3 * SLOT_BYTES, SLOT_BYTES]
    assert _core.plan_slots(s, peer, F, [0], [8])[0] is None
    assert list(_core.plan_slots(s, peer, F, range(2), (3, 4))[2]) == [2 * SLOT_BYTES]


def test_copy_ranges(make_store):
    from radixshmem import _core
    s = make_store()
    s.write_slot(0, os.urandom(SLOT_BYTES))
    a = np.array([s.data_addr + s.slot_offset(1)], dtype=np.uint64)
    b = np.array([s.data_addr + s.slot_offset(0)], dtype=np.uint64)
    n = np.array([SLOT_BYTES], dtype=np.uint64)
    _core.copy_ranges(XferOp.READ, a, b, n)
    assert s.read_slot(1) == s.read_slot(0)
    s.write_slot(2, b"x" * SLOT_BYTES)
    _core.copy_ranges(XferOp.WRITE, np.array([s.data_addr + s.slot_offset(2)], dtype=np.uint64), b, n)
    assert s.read_slot(0) == b"x" * SLOT_BYTES


def test_check_batch():
    from radixshmem import _core

    def u64(*v):
        return np.array(v, dtype=np.uint64)

    regions = [(0x1000, 0x1000), (0x10000, 0x100)]
    assert _core.check_batch(u64(0x1000, 0x10080), u64(0x5000, 0x5800), u64(0x800, 0x80), regions,
                             0x5000, 0x1000) == (0x880, "")
    for local, remote, length, why in [
        (0x1000, 0x5000, 0, "zero length"),
        (0, 0x5000, 8, "null address"),
        (0x1c00, 0x5000, 0x800, "registered region"),     # runs past the first region's end
        (0x1000, 0x5c00, 0x800, "escapes the segment"),
        (0x1000, 0x4ff0, 0x10, "escapes the segment"),
        (0x1000, 2**64 - 8, 0x10, "wraps"),
    ]:
        total, msg = _core.check_batch(u64(local), u64(remote), u64(length), regions, 0x5000, 0x1000)
        assert total == 0 and why in msg, (local, remote, length, msg)
