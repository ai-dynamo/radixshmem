# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""radix/<cluster_id>/geometry against a private etcd: created once, joined by
equal geometries, refused to different ones, and kept alive by its holders."""

import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import uuid

import pytest

import radixshmem
from radixshmem import _geometry, server

pytestmark = pytest.mark.skipif(not shutil.which("etcd") or not hasattr(radixshmem._core, "EtcdKv"),
                                reason="needs etcd and radixshmem built with etcd")

TTL_S = 3


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def registry():
    # One etcd for the module: the Go client is process-global and keeps the
    # first endpoint it was given.
    workdir = tempfile.mkdtemp(prefix="radixshmem-georeg-")
    cport, pport = _free_port(), _free_port()
    proc = subprocess.Popen(
        ["etcd", "--name", "georeg", "--data-dir", os.path.join(workdir, "data"),
         "--listen-client-urls", f"http://127.0.0.1:{cport}",
         "--advertise-client-urls", f"http://127.0.0.1:{cport}",
         "--listen-peer-urls", f"http://127.0.0.1:{pport}",
         "--initial-advertise-peer-urls", f"http://127.0.0.1:{pport}",
         "--initial-cluster", f"georeg=http://127.0.0.1:{pport}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 30
    while True:
        try:
            socket.create_connection(("127.0.0.1", cport), timeout=0.5).close()
            break
        except OSError:
            if proc.poll() is not None or time.time() > deadline:
                raise RuntimeError("etcd did not come up")
            time.sleep(0.1)
    try:
        yield f"etcd://127.0.0.1:{cport}"
    finally:
        proc.terminate()
        proc.wait(10)
        shutil.rmtree(workdir, ignore_errors=True)


@pytest.fixture
def regs(registry, monkeypatch):
    """regs(n) -> n _ServerRegistry on a fresh cluster id, short lease TTL."""
    monkeypatch.setattr(server, "_LEASE_TTL_S", TTL_S)
    cid = uuid.uuid4().hex[:8]
    made = []

    def make(n):
        out = [server._ServerRegistry(registry, cid) for _ in range(n)]
        made.extend(out)
        return out

    yield make
    for r in made:
        r.release_geometry()
        r.close()


def uniform(slot_bytes):
    return _geometry.uniform_fields({"block_size": 16, "slot_align": 64, "register_chunk_tokens": 4096,
                                     "pools": {"full": {"slot_bytes": slot_bytes, "num_slots": 8}}})


def hold(reg, mine, node):
    reg.hold_geometry(mine, node, lambda published: _geometry.diff_geometry(mine, published, counts=False))


def test_concurrent_different_geometries_one_wins(regs):
    rs = regs(8)
    go = threading.Barrier(len(rs))
    errors = [None] * len(rs)

    def run(i):
        go.wait()
        try:
            hold(rs[i], uniform(1024 if i % 2 else 2048), f"n{i}")
        except radixshmem.GeometryMismatch as e:
            errors[i] = e

    ts = [threading.Thread(target=run, args=(i,)) for i in range(len(rs))]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
    rec = rs[0].geometry()
    won = rec["geometry"]["pools"]["full"]["slot_bytes"]
    for i, r in enumerate(rs):
        if (1024 if i % 2 else 2048) == won:
            assert errors[i] is None and r.geometry_lease == rec["lease"]
        else:
            assert isinstance(errors[i], radixshmem.GeometryMismatch) and r.geometry_lease == 0
            assert "created by n" in str(errors[i]) and "cluster_id" in str(errors[i])


def test_equal_geometry_joins_the_creators_lease(regs):
    a, b = regs(2)
    hold(a, uniform(1024), "a")
    hold(b, uniform(1024), "b")
    rec = a.geometry()
    assert rec["node"] == "a" and a.geometry_lease == b.geometry_lease == rec["lease"]
    hold(b, uniform(1024), "b")                     # again: still one renewal
    assert server._ServerRegistry._renewals[rec["lease"]] == 2


def test_key_lives_while_any_holder_renews(regs):
    a, b = regs(2)
    hold(a, uniform(1024), "a")
    hold(b, uniform(1024), "b")
    a.release_geometry()                            # the creator leaves first
    time.sleep(TTL_S + 2)
    assert b.geometry() is not None
    b.release_geometry()
    deadline = time.time() + TTL_S + 5
    while b.geometry() is not None and time.time() < deadline:
        time.sleep(0.2)
    assert b.geometry() is None
    hold(a, uniform(2048), "a")                     # a new geometry once nobody holds the old one
    assert a.geometry()["geometry"]["pools"]["full"]["slot_bytes"] == 2048
