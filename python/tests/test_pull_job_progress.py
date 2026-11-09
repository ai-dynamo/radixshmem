# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PullJob progress fields and pull_async range slicing.

No radix-server required: exercises the pure-Python job handle surface.
"""

from __future__ import annotations

import numpy as np
import pytest

from radixshmem.client import PullJob, PullResult, RadixClient, _empty


def test_failed_future_wait_removes_job():
    import threading
    from types import SimpleNamespace
    client = SimpleNamespace(_jobs={}, _mu=threading.Lock())
    job = PullJob(7, client)
    client._jobs[7] = job
    job._f.set_exception(RuntimeError("failed"))
    with pytest.raises(RuntimeError, match="failed"):
        job.wait(0)
    assert client._jobs == {}


def test_quarantine_retains_buffers_and_admission_credit():
    import threading
    from radixshmem.client import _QUARANTINED_CLIENTS
    client = RadixClient.__new__(RadixClient)
    client._mu = threading.Lock()
    client._quarantined = {}
    client.transfer_failures = 0
    client._jobs = {}
    client._pending = {}
    client._slots = threading.Semaphore(0)  # credit already held by this pull
    job = PullJob(9, client)
    client._jobs[9] = job
    job._staged = {"sentinel": (0, np.array([42]))}
    try:
        client._quarantine(job, TimeoutError("response lost"))
        with pytest.raises(RuntimeError, match="quarantined"):
            job.wait(0)
        assert client._jobs == {}
        assert client._quarantined[9] is job
        assert job._staged["sentinel"][1].tolist() == [42]
        assert not client._slots.acquire(blocking=False)
        with pytest.raises(RuntimeError, match="cannot close"):
            client.close()
    finally:
        _QUARANTINED_CLIENTS.discard(client)


@pytest.mark.parametrize("recovers", [True, False])
def test_rpc_outage_retries_or_quarantines_without_recycling(monkeypatch, recovers):
    import threading
    from types import SimpleNamespace
    import radixshmem.client as module
    client = RadixClient.__new__(RadixClient)
    client._mu = threading.Lock()
    client._work = threading.Event()
    client._closed = False
    client._quarantined = {}
    client._slots = threading.Semaphore(0)
    client.transfer_failures = 0
    job = PullJob(1, client)
    job._items = [(1, 0, [2], [3])]
    job._staged = {"sentinel": (0, np.array([3]))}
    client._pending, client._jobs = {5: job}, {1: job}
    ticks = iter([0, 31])
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    calls = []
    def wait_any(*args):
        calls.append(True)
        assert not job.done()
        assert job._staged["sentinel"][1].tolist() == [3]
        if len(calls) == 1 or not recovers:
            raise TimeoutError("control plane unavailable")
        return {"done": {5: [(True, 0, 1, "")]}, "unknown": []}
    client._t = SimpleNamespace(wait_any=wait_any)
    def complete(j, replies):
        j._f.set_result(_result(0))
        client._closed = True
    client._complete = complete
    original = client._quarantine
    def quarantine(j, error):
        original(j, error)
        client._closed = True
    client._quarantine = quarantine
    try:
        client._run()
        assert len(calls) == 2 and not client._pending
        if recovers:
            assert job.wait(0).common_hit == 0
            assert not client._quarantined
            assert client._slots.acquire(blocking=False)
        else:
            with pytest.raises(RuntimeError, match="quarantined"):
                job.wait(0)
            assert client._quarantined[1] is job
            assert not client._slots.acquire(blocking=False)
    finally:
        module._QUARANTINED_CLIENTS.discard(client)


@pytest.mark.parametrize("busy_release", [False, True])
def test_server_deadline_keeps_ownership_until_transport_drains(busy_release):
    import threading
    import time
    from collections import deque
    from types import SimpleNamespace
    from radixshmem import _data
    from radixshmem.server import RadixServer, _Job

    server = RadixServer.__new__(RadixServer)
    server.cfg = SimpleNamespace(max_inflight=1, job_ttl_s=60)
    server._jobs_mu = threading.Lock()
    server._running = True
    server._peers = {}
    server._wake = threading.Event()
    job = _Job([(1, 0, [1], [2])], time.monotonic() - 1)
    job.segs = ["peer"]
    server._queue = deque([job])
    server._jobs = {}
    events = []
    class Xfer:
        polls = 0
        releases = 0
        def read(self, *a, **kw):
            return 10, 1
        def poll(self, h):
            self.polls += 1
            events.append("poll")
            assert not job.done
            return (_data.XferState.PENDING if self.polls == 1
                    else _data.XferState.COMPLETED), 4096
        def release(self, h):
            self.releases += 1
            events.append("release")
            if busy_release and self.releases == 1:
                return _data.XFER_EBUSY
            return _data.XFER_OK
    server._xfer = Xfer()
    def resolve(j, i, reply):
        events.append("resolve")
        assert server._xfer.polls >= 2
        assert not reply[0] and reply[3] == "deadline exceeded"
        j.done = True
        server._running = False
    server._resolve = resolve
    server._transfer_loop()
    assert events[-1] == "resolve"
    assert server._xfer.releases == (2 if busy_release else 1)


def _result(common_hit: int) -> PullResult:
    return PullResult(
        common_hit,
        _empty(),
        0,
        _empty(),
        -1,
        0,
        0,
        -1,
        lambda: None,
    )


def test_published_hit_tracks_local_then_common():
    job = PullJob(1, client=None)  # type: ignore[arg-type]
    job.local_hit = 2
    job.planned_hit = 8
    assert not job.done()
    assert job.published_hit == 2

    job._f.set_result(_result(5))
    assert job.done()
    assert job.published_hit == 5


def test_published_hit_stays_local_on_failed_future():
    job = PullJob(2, client=None)  # type: ignore[arg-type]
    job.local_hit = 3
    job._f.set_exception(RuntimeError("boom"))
    assert job.done()
    assert job.published_hit == 3


def test_pull_async_range_bounds_validation():
    client = RadixClient.__new__(RadixClient)
    client.local_read_only = False
    hashes = np.arange(10, dtype=np.uint64)
    with pytest.raises(ValueError, match="out of bounds"):
        RadixClient.pull_async(client, hashes, start=4, end=3)
    with pytest.raises(ValueError, match="out of bounds"):
        RadixClient.pull_async(client, hashes, start=-1, end=2)
    with pytest.raises(ValueError, match="out of bounds"):
        RadixClient.pull_async(client, hashes, start=0, end=11)
