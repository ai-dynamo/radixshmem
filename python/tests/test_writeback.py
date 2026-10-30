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

import numpy as np
import pytest

import radixshmem
from radixshmem import Geometry, RadixClient, RadixServer, ServerConfig, WritebackHandler, writeback
from radixshmem.writeback import load_handler

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
              "writeback": True, "writeback_max_held_ratio": 0.05, **kw}
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


class ReadyAtStart(Recorder):
    def start(self, client):
        self.ready = client.wait_ready(5)
        super().start(client)


def make_handler(config):
    return Recorder(**config)


def returns_none(config):
    return None


def up(c, handler):
    srv = RadixServer(c, writeback=handler).start()
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
        seqs = fill(cl, 10)                                  # 400 of 512: below the 25% free mark
        wait_for(lambda: cl.mempool_free() >= THRESHOLD and srv.stats()["writeback"]["popped"] == 0)
        st = srv.stats()["writeback"]
        assert st["produced"] == st["completed"] == len(rec.tasks) > 1
        assert st["held_slots"] == (0, 0, 0) and st["inflight_slots"] == (0, 0, 0)
        for t, data, hit in zip(rec.tasks, rec.data, rec.hits):
            assert t.end - t.evicted_from == 4 and t.end % 4 == 0
            owner = [h for h in seqs if np.array_equal(h[:t.end], t.hashes)]
            assert len(owner) == 1                           # a whole prefix of one sequence
            assert data == [struct.pack("<Q", int(x)) for x in t.hashes]
            assert hit == t.evicted_from                     # the prefix was still in the tree
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
        assert st["held_slots"][0] == 4
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
        fill(cl, 10)
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
    with pytest.raises(ValueError, match="go together"):
        RadixServer(cfg())
    with pytest.raises(ValueError, match="go together"):
        RadixServer(cfg(writeback=False), writeback=Recorder())
    with pytest.raises(ValueError, match="background eviction"):
        cfg(background_evict_ratio=0.0)
    with pytest.raises(ValueError, match="writeback_max_held_ratio"):
        cfg(writeback_max_held_ratio=0.0)

    from radixshmem import cli

    a = cli.parse_args(["--data-bytes", "64M", "--writeback-handler", "test_writeback:make_handler",
                        "--writeback-config", '{"hold": true}', "--writeback-max-held", "0.1"])
    c = cli.build_config(a)
    assert c.writeback and c.writeback_max_held_ratio == 0.1
    h = load_handler(a.writeback_handler, cli.writeback_config(a.writeback_config))
    assert isinstance(h, Recorder) and h.hold
    assert not cli.build_config(cli.parse_args([])).writeback
    with pytest.raises(ValueError, match="module:factory"):
        load_handler("no_colon", {})
    with pytest.raises(TypeError, match="not a WritebackHandler"):
        load_handler("test_writeback:returns_none", {})
