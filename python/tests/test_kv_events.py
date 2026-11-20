# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Node placement stream: a real standalone server publishes what its tree
holds; a ZMQ subscriber sees inserts, evictions and a reset, in order."""

import struct
import uuid

import numpy as np
import pytest

import radixshmem
from radixshmem import Geometry, RadixClient, RadixServer, ServerConfig
from radixshmem.kv_events import records_to_events

pytestmark = pytest.mark.skipif(not hasattr(radixshmem, "_data"), reason="radixshmem._data not built")
zmq = pytest.importorskip("zmq")
msgpack = pytest.importorskip("msgpack")

OK = radixshmem.InsertError.OK
BLOCK = 4


def hashes(n, base=0):
    return radixshmem.compute_hashes(np.arange(base, base + BLOCK * n, dtype=np.int64), BLOCK)


def signed(h):
    return np.asarray(h, dtype=np.uint64).view(np.int64).tolist()


@pytest.fixture
def node(tmp_path):
    """A ready standalone server publishing on a free tcp port, a client on it,
    and a subscriber connected before anything is inserted."""
    tag = uuid.uuid4().hex[:8]
    cfg = ServerConfig(name=f"/py_kvev_{tag}", endpoint=f"unix://{tmp_path}/{tag}.sock",
                       data_bytes=16 * 8192, slot_align=64, prefault=False,
                       background_evict_ratio=0.0, kv_events_endpoint="tcp://127.0.0.1:*")
    srv = RadixServer(cfg).start()
    cl = RadixClient(cfg.name, Geometry(block_size=BLOCK, full_slot_bytes=8192,
                                        hash_scheme="test-scheme-v1"), endpoint=cfg.endpoint)
    cl.wait_ready(30)
    sub = zmq.Context.instance().socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.RCVTIMEO, 3000)
    sub.connect(srv._kv_events.endpoint)
    # PUB drops until the subscription lands; the heartbeat proves it has.
    _next(sub)
    try:
        yield srv, cl, sub
    finally:
        sub.close(0)
        cl.close()
        srv.close()


def _next(sub):
    topic, seq, payload = sub.recv_multipart()
    return struct.unpack(">Q", seq)[0], msgpack.unpackb(payload, raw=False)


def _events(sub, want):
    """Collect events until `want` of them arrived (heartbeats skipped)."""
    out, seqs = [], []
    while len(out) < want:
        seq, msg = _next(sub)
        seqs.append(seq)
        out += msg["events"]
    assert seqs == list(range(seqs[0], seqs[0] + len(seqs)))
    return out, msg


def test_insert_evict_reset_reach_the_subscriber_in_order(node):
    srv, cl, sub = node
    a, b = hashes(8, 0), hashes(8, 10_000)
    assert cl.insert(a, cl.allocate_slots(8)).error == OK
    events, msg = _events(sub, 1)
    assert events == [["stored", signed(a)]]
    assert msg["v"] == 1 and msg["hash_scheme"] == "test-scheme-v1" and msg["block_size"] == BLOCK
    assert msg["node"] == 0 and msg["incarnation"] == srv._kv_events.incarnation

    # The pool holds 16 slots: b fits, then 8 more slots evict a (the LRU path).
    assert cl.insert(b, cl.allocate_slots(8)).error == OK
    assert _events(sub, 1)[0] == [["stored", signed(b)]]
    extra = cl.allocate_slots(8)
    events, _ = _events(sub, 1)
    assert events[0][0] == "removed" and sorted(events[0][1]) == sorted(signed(a))
    cl.recycle_slots(extra)

    cl.reset()
    assert _events(sub, 1)[0] == [["cleared", []]]


def test_heartbeat_carries_the_header_and_advances_seq(node):
    _, _, sub = node
    s1, m1 = _next(sub)
    s2, m2 = _next(sub)
    assert m1["events"] == [] and m2["events"] == [] and s2 == s1 + 1
    assert m1["incarnation"] == m2["incarnation"]


def test_overflow_leads_with_cleared():
    recs = [(radixshmem.kv_events.KVEVENT_STORED, np.array([1, 2], dtype=np.uint64))]
    assert records_to_events(recs, True) == [["cleared", []], ["stored", [1, 2]]]
    big = np.array([2**64 - 1], dtype=np.uint64)
    assert records_to_events([(0, big)], False) == [["removed", [-1]]]


def test_hash_scheme_is_part_of_the_geometry(tmp_path):
    tag = uuid.uuid4().hex[:8]
    cfg = ServerConfig(name=f"/py_kvev_hs_{tag}", endpoint=f"unix://{tmp_path}/{tag}.sock",
                       data_bytes=16 * 8192, slot_align=64, prefault=False,
                       background_evict_ratio=0.0)
    srv = RadixServer(cfg).start()
    try:
        a = RadixClient(cfg.name, Geometry(block_size=BLOCK, full_slot_bytes=8192),
                        endpoint=cfg.endpoint)
        a.wait_ready(30)
        assert a.geometry["hash_scheme"] == "xxh64-v1"
        # Naming the default explicitly is the same geometry ...
        RadixClient(cfg.name, Geometry(block_size=BLOCK, full_slot_bytes=8192,
                                       hash_scheme="xxh64-v1"), endpoint=cfg.endpoint).close()
        # ... another scheme is not.
        with pytest.raises(radixshmem.GeometryMismatch, match="hash_scheme"):
            RadixClient(cfg.name, Geometry(block_size=BLOCK, full_slot_bytes=8192,
                                           hash_scheme="sglang-sha256-i64-v1"),
                        endpoint=cfg.endpoint)
        a.close()
    finally:
        srv.close()
