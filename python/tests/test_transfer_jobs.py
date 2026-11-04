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
    def __init__(self, hit=0, status=None, full_error=None, run_errors=None):
        self.hit, self.next = hit, 100
        self.status, self.full_error, self.run_errors = status, full_error, run_errors or {}
        self.log = []

    def remote_query(self, rank, hashes, lock, update_meta):
        return SimpleNamespace(status=0, total_hit_length=self.hit,
                               finalize=lambda: self.log.append(("unlock", rank)))

    def remote_allocate(self, rank, n, kind):
        slots, self.next = list(range(self.next, self.next + n)), self.next + n
        return SimpleNamespace(slots=slots)

    def remote_insert_path(self, rank, hashes, start, full_slots, runs, auto_recycle):
        self.log.append(("insert", len(hashes), start, [int(x) for x in full_slots],
                         [(kind, end, [int(x) for x in s]) for kind, end, s in runs]))
        ok = radixshmem.InsertError.OK
        return SimpleNamespace(
            remote_status=radixshmem.RemoteStatus.OK if self.status is None else self.status,
            full=SimpleNamespace(error=ok if self.full_error is None else self.full_error),
            runs=[SimpleNamespace(error=self.run_errors.get((kind, end), ok)) for kind, end, _ in runs])

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


RUNS = [(SWA, 8, [105, 106, 107, 108]), (MAMBA, 5, [109]), (MAMBA, 8, [110])]


def test_push_publishes_every_part_in_one_insert_once_all_writes_landed():
    index = StubIndex(hit=3)
    assert push(index) == PushResult(8, 3, 11, 110, True)
    assert index.log == [("insert", 8, 3, [100, 101, 102, 103, 104], RUNS), ("unlock", 2)]


def test_push_skips_a_full_path_the_target_already_holds():
    index = StubIndex(hit=8)
    assert push(index) == PushResult(8, 8, 6, 60, True)
    assert index.log == [("insert", 8, 8, [], [(SWA, 8, [100, 101, 102, 103]), (MAMBA, 5, [104]),
                                                (MAMBA, 8, [105])]), ("unlock", 2)]


def test_push_leaves_only_the_slots_a_write_may_still_reach():
    index = StubIndex(hit=3)
    replies = {FULL: (False, 0, 0, XFER_IN_FLIGHT), SWA: (False, 0, 0, "XferState.FAILED")}
    assert push(index, replies) == PushResult(3, 3, 0, 0, False)
    assert index.log == [("recycle", SWA, [105, 106, 107, 108]), ("recycle", MAMBA, [109, 110]), ("unlock", 2)]


def test_push_publishes_the_parts_whose_writes_landed():
    index = StubIndex(hit=3)
    assert push(index, {SWA: (False, 0, 0, "XferState.FAILED")}) == PushResult(8, 3, 7, 70, False)
    assert index.log == [("recycle", SWA, [105, 106, 107, 108]),
                         ("insert", 8, 3, [100, 101, 102, 103, 104], RUNS[1:]), ("unlock", 2)]


def test_push_recycles_landed_writes_when_the_task_is_no_longer_valid():
    index = StubIndex(hit=3)
    assert push(index, valid=lambda: False) == PushResult(3, 3, 0, 0, False)
    assert index.log == [("recycle", FULL, [100, 101, 102, 103, 104]), ("recycle", SWA, [105, 106, 107, 108]),
                         ("recycle", MAMBA, [109, 110]), ("unlock", 2)]

    index = StubIndex(hit=3)
    assert push(index, {SWA: (False, 0, 0, XFER_IN_FLIGHT)}, valid=lambda: False).complete is False
    assert index.log == [("recycle", FULL, [100, 101, 102, 103, 104]), ("recycle", MAMBA, [109, 110]),
                         ("unlock", 2)]


def test_push_counts_only_the_parts_the_target_published():
    E = radixshmem.InsertError
    index = StubIndex(hit=3, run_errors={(MAMBA, 5): E.FULL_PATH_MISSING})
    assert push(index) == PushResult(8, 3, 10, 100, False)

    index = StubIndex(hit=3, full_error=E.DATA_POOL_FULL,
                      run_errors={(SWA, 8): E.FULL_PATH_MISSING, (MAMBA, 8): E.FULL_PATH_MISSING})
    assert push(index) == PushResult(3, 3, 1, 10, False)
    assert [entry[0] for entry in index.log] == ["insert", "unlock"]


def test_push_keeps_every_slot_when_the_insert_request_failed():
    index = StubIndex(hit=3, status=radixshmem.RemoteStatus.UNAVAILABLE)
    assert push(index) == PushResult(3, 3, 0, 0, False)
    assert index.log == [("insert", 8, 3, [100, 101, 102, 103, 104], RUNS), ("unlock", 2)]


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


def test_close_releases_the_pins_of_unfinished_pushes_concurrently():
    released, c = [], client(StubIndex())
    pushes = []
    for i in range(8):
        job = PushJob(i, c, 2, np.arange(8, dtype=np.uint64))
        job._q, job._hit = SimpleNamespace(finalize=lambda i=i: (time.sleep(0.3), released.append(i))), 3
        c._pending[100 + i] = job
        pushes.append(job)
    pull = PullJob(99, c)
    c._pending[7] = pull
    t0 = time.monotonic()
    c.close()
    assert time.monotonic() - t0 < 1.2
    assert sorted(released) == list(range(8))
    assert list(c._pending.values()) == [pull] and not pull.done()
    assert all(job.wait(0) == PushResult(3, 3, 0, 0, False) for job in pushes)


class ClosingIndex(StubIndex):
    def __init__(self, at, **kw):
        super().__init__(**kw)
        self.at, self.client = at, None

    def remote_allocate(self, rank, n, kind):
        if self.at == "allocate":
            self.at = None
            self.client.close()
        return super().remote_allocate(rank, n, kind)


class ClosingTransport(StubTransport):
    def __init__(self):
        super().__init__({})
        self.client = None

    def submit(self, items, timeout_ms):
        ticket = super().submit(items, timeout_ms)
        self.client.close()
        return ticket


def push_args():
    return (2, np.arange(1, 9, dtype=np.uint64), np.arange(10, 18, dtype=np.int32),
            [(8, [20, 21, 22, 23])], [(5, 30), (8, 31)])


def test_a_push_reaching_submit_after_close_recycles_its_slots_and_releases_its_pin():
    index = ClosingIndex("allocate", hit=3)
    c = index.client = client(index)
    job = c.push_async(*push_args())
    assert c._index is None and c._t.items == []
    assert job.wait(0) == PushResult(3, 3, 0, 0, False)
    assert index.log == [("recycle", FULL, [100, 101, 102, 103, 104]), ("recycle", SWA, [105, 106, 107, 108]),
                         ("recycle", MAMBA, [109, 110]), ("unlock", 2)]
    assert c._pending == {} and c._completer is None and c._slots._value == 4


def test_a_push_submitted_while_closing_releases_its_pin_and_keeps_its_slots():
    index = StubIndex(hit=3)
    c = client(index)
    c._t = t = ClosingTransport()
    t.client = c
    job = c.push_async(*push_args())
    assert len(t.items) == 3 and job.wait(0) == PushResult(3, 3, 0, 0, False)
    assert index.log == [("unlock", 2)]
    assert c._pending == {} and c._completer is None and c._slots._value == 4


def test_close_lets_a_push_waiting_for_a_slot_finish():
    index = StubIndex(hit=3)
    c = client(index)
    c._slots = threading.Semaphore(1)
    held = PushJob(1, c, 2, np.arange(8, dtype=np.uint64))
    held._q = SimpleNamespace(finalize=lambda: index.log.append(("unlock held", 2)))
    c._slots.acquire()
    c._pending[1] = held
    jobs = []
    th = threading.Thread(target=lambda: jobs.append(c.push_async(*push_args())))
    th.start()
    time.sleep(0.2)
    assert th.is_alive() and index.log == []
    c.close()
    th.join(2)
    assert not th.is_alive() and jobs[0].wait(0) == PushResult(3, 3, 0, 0, False)
    assert held.wait(0) == PushResult(0, 0, 0, 0, False)
    assert sorted(index.log, key=str) == sorted([
        ("unlock held", 2), ("recycle", FULL, [100, 101, 102, 103, 104]), ("recycle", SWA, [105, 106, 107, 108]),
        ("recycle", MAMBA, [109, 110]), ("unlock", 2)], key=str)
    assert c._pending == {}


def test_a_push_after_close_takes_no_pin():
    index = StubIndex(hit=3)
    c = client(index)
    c.close()
    with pytest.raises(RuntimeError, match="closed"):
        c.push_async(*push_args())
    assert index.log == [] and c._jobs == {}


def test_a_push_cannot_be_cancelled_and_runs_on():
    index = StubIndex(hit=3)
    c = client(index)
    try:
        job = c.push_async(*push_args())
        for target in (job, job.job_id):
            with pytest.raises(TypeError, match="push"):
                c.cancel(target)
        assert c._jobs[job.job_id] is job and not hasattr(job, "_cancelled")
        assert job.wait(5) == PushResult(8, 3, 11, 110, True)
        assert index.log[-1] == ("unlock", 2)
    finally:
        c.close()


class FailingIndex(StubIndex):
    def __init__(self, fail, **kw):
        super().__init__(**kw)
        self.fail, self.client = fail, None

    def remote_allocate(self, rank, n, kind):
        if self.fail == ("allocate", kind):
            raise RuntimeError("allocate broke")
        return super().remote_allocate(rank, n, kind)

    def remote_insert_path(self, *args):
        if self.fail == "insert":
            raise RuntimeError("insert broke")
        if self.fail == "store":
            self.client._store = None
        return super().remote_insert_path(*args)


def test_a_push_whose_completion_raises_still_completes_and_releases_its_pin():
    index = FailingIndex("insert", hit=3)
    c = index.client = client(index)
    try:
        assert c.push_async(*push_args()).wait(5) == PushResult(3, 3, 0, 0, False)
        assert index.log == [("unlock", 2)]
    finally:
        c.close()


def test_a_push_whose_reservation_raises_returns_what_it_reserved():
    index = FailingIndex(("allocate", SWA), hit=3)
    c = index.client = client(index)
    try:
        job = c.push_async(*push_args())
        assert job.wait(0) == PushResult(3, 3, 0, 0, False) and c._t.items == []
        assert index.log == [("recycle", FULL, [100, 101, 102, 103, 104]), ("unlock", 2)]
    finally:
        c.close()


def test_a_push_counts_its_bytes_with_the_store_it_started_with():
    index = FailingIndex("store", hit=3)
    c = index.client = client(index)
    try:
        assert c.push_async(*push_args()).wait(5) == PushResult(8, 3, 11, 110, True)
    finally:
        c.close()
