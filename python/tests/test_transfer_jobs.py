# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import radixshmem
from radixshmem import ComponentType, PullJob, PushJob, PushResult, RadixClient, RadixServer, ServerConfig
from radixshmem.client import XFER_IN_FLIGHT, XFER_READ, XFER_WRITE
from radixshmem.server import _Job

_data = getattr(radixshmem, "_data", None)
pytestmark = pytest.mark.skipif(_data is None, reason="radixshmem._data not built")

FULL, SWA, MAMBA = ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA


class FakeXfer:
    def __init__(self, plans):
        self.plans = list(plans)
        self.handles = {}
        self.calls = []

    def _issue(self, op, local, remote):
        plan = self.plans.pop(0)
        if plan is None:
            return _data.INVALID_BATCH, 0
        h = len(self.handles) + 1
        self.handles[h] = list(plan)
        self.calls.append((op, list(local), list(remote)))
        return h, 1

    def write(self, seg, local, remote, kind):
        return self._issue(XFER_WRITE, local, remote)

    def read(self, seg, remote, local, kind):
        return self._issue(XFER_READ, local, remote)

    def poll(self, h):
        state = self.handles[h][0]
        return state, 64 if state == _data.XferState.COMPLETED else 0

    def release(self, h):
        return self.handles[h][1]


def item(op=XFER_WRITE):
    return 1, int(FULL), [5, 6], [10, 11], op


def run(jobs, plans):
    srv = RadixServer(ServerConfig(name="/py_tj"))
    srv._xfer = FakeXfer(plans)
    for ticket, job in enumerate(jobs, start=1):
        job.ticket, job.segs = ticket, ["seg"] * len(job.items)
        srv._jobs[ticket] = job
        srv._queue.append(job)
    srv._running = True
    loop = threading.Thread(target=srv._transfer_loop, daemon=True)
    loop.start()
    deadline = time.monotonic() + 10
    while not all(job.done for job in jobs):
        assert time.monotonic() < deadline, "the transfer loop did not resolve every item"
        time.sleep(0.01)
    srv._running = False
    srv._wake.set()
    loop.join(5)
    return srv


def test_a_transfer_is_in_flight_exactly_while_the_engine_keeps_it():
    S, EBUSY, OK = _data.XferState, _data.XFER_EBUSY, _data.XFER_OK
    plans = [(S.COMPLETED, OK), (S.FAILED, EBUSY), (S.FAILED, OK), (S.PENDING, EBUSY),
             (S.PENDING, OK), (S.TIMEOUT, OK), (S.TIMEOUT, EBUSY), (S.INVALID, OK), None]
    job = _Job([item() for _ in plans], time.monotonic() + 0.3)
    srv = run([job], plans)
    assert [r[0] for r in job.replies] == [True] + [False] * 8
    assert [r[3] for r in job.replies] == ["", XFER_IN_FLIGHT, str(S.FAILED), XFER_IN_FLIGHT, str(S.TIMEOUT),
                                           str(S.TIMEOUT), XFER_IN_FLIGHT, str(S.INVALID), "write rejected"]
    assert sorted(srv._zombies) == [2, 4, 7]
    for h in srv._zombies:
        srv._xfer.handles[h][1] = OK
    srv._reap()
    assert srv._zombies == []


def test_a_job_whose_deadline_passed_in_the_queue_is_not_issued():
    job = _Job([item(), item(XFER_READ)], time.monotonic() - 1)
    srv = run([job], [])
    assert srv._xfer.calls == []
    assert [r[0] for r in job.replies] == [False, False]
    assert all(r[3] == "deadline passed before the transfer was issued" for r in job.replies)


def test_items_go_out_in_their_direction_and_resolved_items_are_skipped():
    S, OK = _data.XferState, _data.XFER_OK
    job = _Job([item(XFER_WRITE), item(XFER_READ), item()], time.monotonic() + 5)
    job.replies[2], job.left = (False, 0, 0, "rank 1 is not in the cluster"), 2
    srv = run([job], [(S.COMPLETED, OK), (S.COMPLETED, OK)])
    assert srv._xfer.calls == [(XFER_WRITE, [10, 11], [5, 6]), (XFER_READ, [10, 11], [5, 6])]
    assert [r[0] for r in job.replies] == [True, True, False]


class StubIndex:
    def __init__(self, hit=0):
        self.hit, self.next = hit, 100
        self.log = []

    def remote_query(self, rank, hashes, lock, update_meta):
        return SimpleNamespace(status=0, total_hit_length=self.hit,
                               finalize=lambda: self.log.append(("unlock", rank)))

    def remote_allocate(self, rank, n, kind):
        slots, self.next = list(range(self.next, self.next + n)), self.next + n
        return SimpleNamespace(slots=slots)

    def remote_insert(self, rank, hashes, start, length, slots, auto_recycle, kind):
        self.log.append(("insert", kind, len(hashes), start, [int(x) for x in slots[:length]]))
        return SimpleNamespace(error=radixshmem.InsertError.OK)

    def remote_recycle(self, rank, slots, kind):
        self.log.append(("recycle", kind, [int(x) for x in slots]))

    def recycle_slots(self, slots, kind):
        self.log.append(("local recycle", kind, [int(x) for x in slots]))


class StubTransport:
    def __init__(self, replies):
        self.replies = replies
        self.items = []

    def submit(self, items, timeout_ms):
        self.items = list(items)
        return 1

    def wait_any(self, tickets, timeout_ms):
        replies = [self.replies.get(ComponentType(it[1]), (True, 64, 1, "")) for it in self.items]
        return {"done": {1: replies}, "unknown": []}

    def close(self):
        pass


def client(index, replies=None):
    c = RadixClient.__new__(RadixClient)
    c._init("/py_tj", StubTransport(replies or {}), 4, False)
    c._index, c._store = index, SimpleNamespace(pool=lambda kind: SimpleNamespace(slot_bytes=10))
    return c


def push(index, replies=None, valid=None):
    c = client(index, replies)
    try:
        return c.push_async(2, np.arange(1, 9, dtype=np.uint64), np.arange(10, 18, dtype=np.int32),
                            [(8, [20, 21, 22, 23])], [(5, 30), (8, 31)], valid=valid).wait(5)
    finally:
        c.close()


def test_push_inserts_every_part_once_all_writes_landed():
    index = StubIndex(hit=3)
    assert push(index) == PushResult(8, 3, 11, 110, True)
    assert index.log == [("insert", FULL, 8, 3, [100, 101, 102, 103, 104]),
                         ("insert", SWA, 8, 0, [105, 106, 107, 108]),
                         ("insert", MAMBA, 5, 0, [109]), ("insert", MAMBA, 8, 0, [110]), ("unlock", 2)]


def test_push_skips_a_full_path_the_target_already_holds():
    index = StubIndex(hit=8)
    assert push(index) == PushResult(8, 8, 6, 60, True)
    assert index.log == [("insert", SWA, 8, 0, [100, 101, 102, 103]),
                         ("insert", MAMBA, 5, 0, [104]), ("insert", MAMBA, 8, 0, [105]), ("unlock", 2)]


def test_push_leaves_only_the_slots_a_write_may_still_reach():
    index = StubIndex(hit=3)
    replies = {FULL: (False, 0, 0, XFER_IN_FLIGHT), SWA: (False, 0, 0, "XferState.FAILED")}
    assert push(index, replies) == PushResult(3, 3, 0, 0, False)
    assert index.log == [("recycle", SWA, [105, 106, 107, 108]), ("recycle", MAMBA, [109, 110]), ("unlock", 2)]


def test_push_recycles_landed_writes_when_the_task_is_no_longer_valid():
    index = StubIndex(hit=3)
    assert push(index, valid=lambda: False) == PushResult(3, 3, 0, 0, False)
    assert index.log == [("recycle", FULL, [100, 101, 102, 103, 104]), ("recycle", SWA, [105, 106, 107, 108]),
                         ("recycle", MAMBA, [109, 110]), ("unlock", 2)]

    index = StubIndex(hit=3)
    assert push(index, {SWA: (False, 0, 0, XFER_IN_FLIGHT)}, valid=lambda: False).complete is False
    assert index.log == [("recycle", FULL, [100, 101, 102, 103, 104]), ("recycle", MAMBA, [109, 110]),
                         ("unlock", 2)]


def test_push_rejects_parts_that_end_outside_the_path():
    c = client(StubIndex())
    h, full = np.arange(1, 9, dtype=np.uint64), np.arange(10, 18, dtype=np.int32)
    try:
        for swa, mamba in (([(9, [20])], []), ([(2, [20, 21, 22])], []), ([], [(0, 30)]), ([], [(9, 30)])):
            with pytest.raises(ValueError, match="ends outside"):
                c.push_async(2, h, full, swa, mamba)
        with pytest.raises(ValueError, match="FULL slots"):
            c.push_async(2, h, full[:4])
        assert c._index.log == []
    finally:
        c.close()


def pull_job(c, replies):
    job = PullJob(1, c)
    job._q, job._hashes = SimpleNamespace(finalize=lambda: None), np.arange(8, dtype=np.uint64)
    job._items = [(1, int(FULL), [5, 6], [10, 11]), (1, int(SWA), [7], [12])]
    job._staged = {FULL: (0, np.array([10, 11], dtype=np.int32)), SWA: (0, np.array([12], dtype=np.int32))}
    c._complete(job, replies)
    return job


def test_pull_keeps_the_staging_slots_a_read_may_still_reach():
    c = client(StubIndex())
    job = pull_job(c, [(False, 0, 0, XFER_IN_FLIGHT), (False, 0, 0, "XferState.FAILED")])
    assert c._index.log == [("local recycle", SWA, [12])]
    assert job.wait(0).common_hit == 0

    c = client(StubIndex())
    pull_job(c, [(False, 0, 0, "XferState.FAILED"), (False, 0, 0, XFER_IN_FLIGHT)])
    assert c._index.log == [("local recycle", FULL, [10, 11])]


def test_close_releases_the_pins_of_unfinished_pushes():
    released = []
    c = client(StubIndex())
    job = PushJob(1, c, 2, np.arange(8, dtype=np.uint64))
    job._q, job._hit = SimpleNamespace(finalize=lambda: released.append(job.job_id)), 3
    c._pending[7] = job
    c.close()
    assert released == [1] and not c._pending
    assert job.wait(0) == PushResult(3, 3, 0, 0, False)
