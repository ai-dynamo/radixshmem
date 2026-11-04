# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write-back of background-evicted KV on a standalone data-mode server."""

import asyncio
import math
import os
import struct
import threading
import time
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

import radixshmem
from radixshmem import (ClusterConfig, Geometry, PushJob, PushResult, RadixClient, RadixServer, ServerConfig,
                        WritebackHandler, writeback)

pytestmark = pytest.mark.skipif(not hasattr(radixshmem, "_data"), reason="radixshmem._data not built")

SLOTS = 512          # 4 MiB / 8 KiB
SEQ_BLOCKS = 40      # ten 4-block nodes per sequence
LIMIT = int(SLOTS * 0.05)
THRESHOLD = int(SLOTS * 0.25)


@pytest.fixture
def cfg(tmp_path):
    tag = uuid.uuid4().hex[:8]

    def make(**kw):
        kw = {"data_bytes": 4 << 20, "slot_align": 64, "prefault": False, "background_evict_ratio": 0.25,
              "writeback_max_held_ratio": 0.05, **kw}
        return ServerConfig(name=make.name, endpoint=make.endpoint, **kw)

    make.name, make.endpoint = f"/py_wb_{tag}", f"unix://{tmp_path}/{tag}.sock"
    return make


class Recorder(WritebackHandler):
    """Reads every task's slots; done at once unless ``hold`` (then ``release``).
    The first ``fail`` writes raise ``error``; with ``block`` write() waits for stop()."""

    def __init__(self, hold=False, fail=0, error=None, block=False):
        self.hold, self.fail, self.error, self.block = hold, fail, error, block
        self.client = None
        self.tasks, self.data, self.hits, self.valid = [], [], [], []
        self.pending, self.done = [], None
        self.stopped = False
        self.entered, self.unblock = threading.Event(), threading.Event()
        self.mu = threading.Lock()

    def start(self, client):
        self.client = client

    def write(self, tasks, done):
        if self.block:
            self.entered.set()
            self.unblock.wait()
            return
        if self.fail:
            self.fail -= 1
            raise self.error or RuntimeError("storage down")
        for t in tasks:
            self.data.append([bytes(self.client.slot_view(int(s))[:8]) for s in t.full_slots])
            q = self.client.query(t.hashes, mask=radixshmem.COMPONENT_MASK_FULL, local_only=True,
                                  lock=False, update_meta=False)
            self.hits.append(int(q.common_hit))
            self.valid.append(t.valid())
        with self.mu:
            self.tasks.extend(tasks)
            if self.hold:
                self.pending.extend(t.task_id for t in tasks)
                self.done = done
                return
        done([t.task_id for t in tasks])

    def release(self):
        with self.mu:
            ids, self.pending = self.pending, []
        self.done(ids)

    def stop(self):
        self.stopped = True
        self.unblock.set()

    def stats(self):
        with self.mu:
            return {"written": len(self.tasks)}


class ReadyAtStart(Recorder):
    def start(self, client):
        self.ready = client.wait_ready(5)
        super().start(client)


def with_handler(c, handler):
    class Server(RadixServer):
        def _make_writeback_handler(self):
            return handler

    return Server(c)


def up(c, handler):
    srv = with_handler(c, handler).start()
    cl = RadixClient(c.name, Geometry(block_size=4, full_slot_bytes=8192, register_chunk_tokens=16),
                     endpoint=c.endpoint)
    cl.wait_ready(30)
    wait_for(lambda: handler.client is not None)            # start() runs once the server is ready
    return srv, cl


def fill(cl, n_seqs, first=0):
    seqs = []
    for i in range(first, first + n_seqs):
        h = radixshmem.compute_hashes(np.arange(i * 100000, i * 100000 + 4 * SEQ_BLOCKS, dtype=np.int64), 4)
        slots = cl.allocate_slots(SEQ_BLOCKS)
        assert len(slots) == SEQ_BLOCKS
        for s, x in zip(slots, h):
            cl.slot_view(int(s))[:8] = struct.pack("<Q", int(x))   # each slot starts with its hash
        assert cl.insert(h, slots).error == radixshmem.InsertError.OK
        seqs.append(h)
    return seqs


def wait_for(cond, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def test_handler_gets_whole_prefixes_with_intact_data(cfg):
    rec = Recorder()
    srv, cl = up(cfg(), rec)
    try:
        assert rec.client is not None and rec.client.store is not None
        seqs = fill(cl, 11)                                  # 440 of 512: two sequences to evict
        wait_for(lambda: cl.mempool_free() >= THRESHOLD and srv.stats()["writeback"]["popped"] == 0)
        st = srv.stats()["writeback"]
        assert st["produced"] == st["completed"] == len(rec.tasks) == st["written"] > 1
        assert st["held_slots"] == (0, 0, 0) and st["inflight_slots"] == (0, 0, 0)
        for t, data, hit in zip(rec.tasks, rec.data, rec.hits):
            assert t.end == SEQ_BLOCKS and t.evicted_from == 0   # unshared: one chain, root to leaf
            owner = [h for h in seqs if np.array_equal(h[:t.end], t.hashes)]
            assert len(owner) == 1                           # a whole prefix of one sequence
            assert data == [struct.pack("<Q", int(x)) for x in t.hashes]
            assert hit == t.evicted_from                     # [0, evicted_from) is still in the tree
            assert len(t.swa_slots) == 0 and len(t.mamba_ends) == 0
        assert all(rec.valid) and not rec.tasks[0].valid()   # valid until done
        rec.client.geometry["pools"] = None
        assert srv.geometry["pools"] is not None            # the client's NodeInfo is a copy
        assert cl.check_health() == 0
    finally:
        cl.close()
        srv.close()
    assert rec.stopped


def test_unfinished_writebacks_pause_background_eviction(cfg):
    rec = Recorder(hold=True)
    srv, cl = up(cfg(), rec)
    try:
        fill(cl, 10)
        wait_for(lambda: len(rec.tasks) >= 1)
        time.sleep(0.3)
        st = srv.stats()["writeback"]
        # one task already covers SEQ_BLOCKS > LIMIT slots: nothing else is evicted
        assert st["produced"] == 1 and st["inflight_slots"][0] == SEQ_BLOCKS > LIMIT
        assert st["held_slots"][0] == SEQ_BLOCKS
        free = cl.mempool_free()
        assert free == SLOTS - 10 * SEQ_BLOCKS               # held slots are not free

        # foreground eviction still serves allocations and writes nothing back
        slots = cl.allocate_slots(free + 20)
        assert len(slots) == free + 20
        cl.recycle_slots(slots)
        assert srv.stats()["writeback"]["produced"] == 1

        rec.release()

        # pressure again: every round stays within one task past the watermark
        fill(cl, 3, first=10)
        deadline = time.monotonic() + 20
        while cl.mempool_free() < THRESHOLD:
            assert time.monotonic() < deadline
            wait_for(lambda: rec.pending)
            time.sleep(0.05)
            assert srv.stats()["writeback"]["inflight_slots"][0] < LIMIT + SEQ_BLOCKS
            rec.release()
        assert srv.stats()["writeback"]["produced"] > 1
        wait_for(lambda: srv.stats()["writeback"]["inflight_slots"][0] == 0 or rec.pending)
        if rec.pending:
            rec.release()
        assert cl.check_health() == 0
    finally:
        cl.close()
        srv.close()


def test_handler_failure_gives_the_tasks_up(cfg):
    rec = Recorder(fail=math.inf)
    srv, cl = up(cfg(), rec)
    try:
        fill(cl, 10)
        wait_for(lambda: cl.mempool_free() >= THRESHOLD and srv.stats()["writeback"]["popped"] == 0)
        st = srv.stats()["writeback"]
        assert st["failed"] == st["completed"] == st["produced"] > 0
        assert st["held_slots"] == (0, 0, 0)
        assert cl.check_health() == 0
    finally:
        cl.close()
        srv.close()


def test_base_exception_does_not_end_the_thread(cfg):
    rec = Recorder(fail=1, error=asyncio.CancelledError())
    srv, cl = up(cfg(), rec)
    try:
        fill(cl, 11)
        wait_for(lambda: cl.mempool_free() >= THRESHOLD and srv.stats()["writeback"]["popped"] == 0)
        st = srv.stats()["writeback"]
        assert st["failed"] == 1 and len(rec.tasks) > 0      # later batches were still written
        assert st["completed"] == st["produced"] == st["failed"] + len(rec.tasks)
        assert cl.check_health() == 0
    finally:
        cl.close()
        srv.close()


def test_reset_invalidates_unfinished_tasks(cfg):
    rec = Recorder(hold=True)
    srv, cl = up(cfg(), rec)
    try:
        fill(cl, 10)
        wait_for(lambda: len(rec.tasks) >= 1)
        t = rec.tasks[0]
        assert t.valid() and t.generation == srv.index.reset_generation()
        cl.reset()                                           # frees every slot, held ones included
        assert not t.valid()
        rec.release()
        st = srv.stats()["writeback"]
        assert st["inflight_slots"] == (0, 0, 0) and st["held_slots"] == (0, 0, 0)
        assert cl.check_health() == 0
    finally:
        cl.close()
        srv.close()


def test_start_runs_once_ready(cfg):
    rec = ReadyAtStart()
    srv, cl = up(cfg(), rec)
    try:
        assert rec.ready.mode == "ready" and rec.client.info.mode == "ready"
    finally:
        cl.close()
        srv.close()


def test_close_with_unfinished_writebacks(cfg):
    c = cfg()
    rec = Recorder(hold=True)
    srv, cl = up(c, rec)
    fill(cl, 10)
    wait_for(lambda: len(rec.tasks) >= 1)
    cl.close()
    srv.close()
    assert rec.stopped and srv.mode == "closed"
    assert not os.path.exists(f"/dev/shm{c.name}")
    assert not rec.tasks[0].valid()
    rec.release()                                            # late done: ignored


def test_close_stops_a_stuck_write(cfg, monkeypatch):
    monkeypatch.setattr(writeback, "_STOP_WAIT_S", 0.3)
    rec = Recorder(block=True)
    srv, cl = up(cfg(), rec)
    fill(cl, 10)
    assert rec.entered.wait(10)                              # write() waits for stop()
    cl.close()
    t0 = time.monotonic()
    srv.close()
    assert time.monotonic() - t0 < 5 and rec.stopped and srv.mode == "closed"


def test_config_and_cli(cfg):
    with pytest.raises(ValueError, match="writeback_max_held_ratio"):
        cfg(writeback_max_held_ratio=0.0)

    from radixshmem import cli

    c = cli.build_config(cli.parse_args(["--data-bytes", "64M", "--writeback-max-held", "0.1"]))
    assert c.writeback_max_held_ratio == 0.1


def test_role_picks_the_handler(cfg):
    cluster = ClusterConfig(expected_min_nodes=2, registry="etcd://127.0.0.1:2379", rpc_address="127.0.0.1")
    assert isinstance(RadixServer(cfg(cluster=cluster)).writeback_handler, writeback.PeerWriteback)
    assert RadixServer(cfg(cluster=cluster, role="decode")).writeback_handler is None
    assert RadixServer(cfg(cluster=cluster, background_evict_ratio=0.0)).writeback_handler is None
    assert RadixServer(cfg(cluster=cluster, data_bytes=0)).writeback_handler is None
    assert RadixServer(cfg()).writeback_handler is None


class StubJob:
    def __init__(self, complete):
        self.complete = complete

    def wait(self, timeout=None):
        return self


class StubClient:
    def __init__(self, role, complete=True):
        self.info = SimpleNamespace(role=role)
        self.geometry = {"pools": {"full": {"slot_bytes": 8192, "num_slots": 64}}}
        self.complete = complete
        self.pushes = []

    def rank(self):
        return 0

    def nodes(self):
        return [{"rank": r, "mode": "ready", "data_plane": True, "role": role}
                for r, role in ((0, self.info.role), (1, "decode"), (2, "decode"), (3, "prefill"))]

    def push_async(self, rank, hashes, full_slots, swa_slots=None, mamba_slot=-1, **kw):
        self.pushes.append((rank, len(hashes)))
        return StubJob(self.complete)


def task(task_id, n, first=1):
    return writeback.WritebackTask(task_id, np.arange(first, first + n, dtype=np.uint64),
                                   np.arange(n, dtype=np.int32), np.empty(0, np.int32),
                                   np.empty(0, np.uint32), np.empty(0, np.int32), 0, 0)


def peer_writeback(role, complete=True):
    h = writeback.PeerWriteback({})
    h.start(StubClient(role, complete))
    return h


def test_peer_writeback_counts_tasks_and_skips_long_paths():
    h = peer_writeback("prefill")
    done = []
    h.write([task(1, 8), task(2, 8, first=2), task(3, radixshmem.MAX_REMOTE_OP_BLOCKS + 1)], done.extend)
    assert done == [1, 2, 3]
    assert h.client.pushes == [(2, 8), (1, 8)]
    assert h.stats() == {"pushed": 2, "incomplete": 0, "no_target": 0, "too_long": 1}

    h = peer_writeback("prefill", complete=False)
    h.write([task(1, radixshmem.MAX_REMOTE_OP_BLOCKS)], done.extend)
    assert h.client.pushes == [(2, radixshmem.MAX_REMOTE_OP_BLOCKS)]
    assert h.stats() == {"pushed": 0, "incomplete": 1, "no_target": 0, "too_long": 0}


def test_peer_writeback_pushes_nothing_from_a_decode_node(capfd):
    h = peer_writeback("decode")
    assert "decode node" in capfd.readouterr().err
    done = []
    h.write([task(1, 8), task(2, 8, first=2)], done.extend)
    assert done == [1, 2] and h.client.pushes == []
    assert h.stats() == {"pushed": 0, "incomplete": 0, "no_target": 2, "too_long": 0}


def test_close_releases_the_pins_of_unfinished_pushes():
    released = []
    c = RadixClient.__new__(RadixClient)
    c._init("/py_wb_stub", SimpleNamespace(close=lambda: None), 4, False)
    job = PushJob(1, c, 2, np.arange(8, dtype=np.uint64))
    job._q, job._hit = SimpleNamespace(finalize=lambda: released.append(job.job_id)), 3
    c._pending[7] = job
    c.close()
    assert released == [1] and not c._pending
    assert job.wait(0) == PushResult(3, 3, 0, 0, False)
