# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RadixServer + RadixClient over the gRPC channel, standalone (no etcd, no RDMA):
the server starts waiting, a client brings the geometry, clients attach on use."""

import os
import threading
import time
import uuid

import numpy as np
import pytest

import radixshmem
from radixshmem import Geometry, RadixClient, RadixServer, ServerConfig

pytestmark = pytest.mark.skipif(not hasattr(radixshmem, "_data"), reason="radixshmem._data not built")

OK = radixshmem.InsertError.OK
FULL, SWA = radixshmem.ComponentType.FULL, radixshmem.ComponentType.SWA


@pytest.fixture
def cfg(tmp_path):
    """cfg(**server_kw) -> ServerConfig with a unique name and socket."""
    tag = uuid.uuid4().hex[:8]

    def make(**kw):
        kw.setdefault("background_evict_ratio", 0.0)
        return ServerConfig(name=make.name, endpoint=make.endpoint, **kw)

    make.name, make.endpoint = f"/py_rpc_{tag}", f"unix://{tmp_path}/{tag}.sock"
    return make


def data(**kw):
    """Data-mode server settings: a 4 MiB budget, 64 B alignment so small test
    slots keep their size."""
    return {"data_bytes": 4 << 20, "slot_align": 64, "prefault": False, **kw}


def geometry(**kw):
    return Geometry(**{"block_size": 4, "full_slot_bytes": 8192, **kw})


def client(c, geometry=None, **kw):
    return RadixClient(c.name, geometry, endpoint=c.endpoint, **kw)


def up(c, geo, timeout=30):
    """start -> a client brings the geometry -> wait_ready, the way an engine's
    bootstrap does it; the configuring client is closed again."""
    srv = RadixServer(c).start()
    try:
        cl = client(c, geo)
        cl.wait_ready(timeout)
        cl.close()
    except BaseException:
        srv.close()
        raise
    return srv


def hashes(n, base=0):
    return radixshmem.compute_hashes(np.arange(base, base + 4 * n, dtype=np.int64), 4)


def test_client_configures_a_waiting_server_and_attaches_when_ready(cfg):
    c = cfg(**data(swa_ratio=8 * 1024 / (4 << 20)))           # room for 8 SWA slots of 1 KiB
    srv = RadixServer(c).start()
    try:
        assert srv.mode == "waiting" and srv.geometry is None
        # a client exists before the server is ready; it sees the state and cannot use the index yet
        a = client(c)
        assert a.info.mode == "waiting" and a.info.data_plane and a.geometry is None
        assert a.info.last_error == "" and a.info.data_bytes == 4 << 20
        with pytest.raises(radixshmem.ServerNotReady) as ei:
            a.query(hashes(1))
        assert ei.value.mode == "waiting"
        with pytest.raises(radixshmem.ServerNotReady):
            a.store
        with pytest.raises(TimeoutError, match="waiting"):
            a.wait_ready(0.2)

        # the geometry rides with a client; the constructor returns before the regions exist
        geo = geometry(swa_slot_bytes=1024, swa_window_blocks=2)
        b = client(c, geo)
        assert b.info.mode in ("configuring", "ready")
        assert b.info.geometry["pools"]["swa"] == {"slot_bytes": 1024, "num_slots": 8, "window_blocks": 2}
        info = b.wait_ready(30)
        assert info.mode == "ready" and srv.mode == "ready"
        assert a.status().mode == "ready"
        # the same geometry again is a no-op (dict form too); another one is refused
        d = client(c, geo.to_dict())
        assert d.info.mode == "ready"
        d.close()
        assert b.configure(geo).mode == "ready"
        with pytest.raises(radixshmem.GeometryMismatch, match="block_size"):
            client(c, geometry(block_size=8, swa_slot_bytes=1024, swa_window_blocks=2))
        with pytest.raises(ValueError, match="num_slots"):
            client(c, geometry(full_slots=10))

        assert a.geometry == srv.geometry
        assert a.geometry["pools"]["full"]["num_slots"] == ((4 << 20) - 8 * 1024) // 8192
        assert a.geometry["register_chunk_tokens"] == 4096
        assert a.block_size() == 4 and int(a.component_config.swa_window_blocks) == 2   # attaches now
        assert a.store.pool(FULL).slot_bytes == 8192 and a.store.pool(SWA).num_slots == 8
        assert b.geometry == a.geometry and b.info.index_name == c.name

        h = hashes(2)
        slots = a.allocate_slots(2)
        assert a.insert(h, slots).error == OK
        a.slot_view(slots[0])[:4] = b"abcd"
        assert bytes(b.slot_view(slots[0])[:4]) == b"abcd"

        job = b.pull_async(h)
        r = job.wait(5)
        assert job.done() and r.common_hit == 2 and r.remote_blocks == 0
        assert list(r.full_slots) == list(slots)
        assert bool(r.finalize)
        r.finalize()
        assert not bool(r.finalize)
        assert b.check_health() == 0
        a.close()
        b.close()
    finally:
        srv.close()
    assert srv.mode == "closed"
    assert not os.path.exists(f"/dev/shm{c.name}")
    assert not os.path.exists(f"/dev/shm{c.name}_data")
    assert not os.path.exists(c.endpoint[len("unix://"):])


def test_index_only_takes_the_counts_from_the_client(cfg):
    c = cfg()
    srv = up(c, geometry(full_slots=16))            # bytes are ignored, published as 0
    try:
        cl = client(c)
        assert cl.info.mode == "ready" and not cl.info.data_plane
        assert cl.geometry["pools"] == {"full": {"slot_bytes": 0, "num_slots": 16}}
        assert cl.block_size() == 4 and cl.mempool_total() == 16
        h = hashes(2)
        assert cl.insert(h, cl.allocate_slots(2)).error == OK
        assert cl.query(h).common_hit == 2
        with pytest.raises(radixshmem.DataPlaneUnavailable):
            cl.pull_async(h)
        with pytest.raises(radixshmem.DataPlaneUnavailable):
            cl.slot_view(0)
        with pytest.raises(radixshmem.DataPlaneUnavailable):
            srv.submit([(1, 0, [0], [0])], 100)
        with pytest.raises(ValueError, match="index-only"):
            client(c, geometry())                                    # no counts
        with pytest.raises(radixshmem.GeometryMismatch, match="num_slots"):
            cl.configure(geometry(full_slots=32))
        cl.close()
    finally:
        srv.close()
    assert not os.path.exists(f"/dev/shm{c.name}")


def test_failed_configuration_returns_to_waiting_with_the_error(cfg):
    c = cfg(**data())
    srv = RadixServer(c).start()
    try:
        def boom():
            raise RuntimeError("no memory for you")

        srv._bring_up = boom
        cl = client(c, geometry())
        assert cl.info.mode in ("configuring", "waiting")
        with pytest.raises(RuntimeError, match="no memory for you"):
            cl.wait_ready(30)
        info = cl.status()
        assert info.mode == "waiting" and "no memory for you" in info.last_error and info.geometry is None
        with pytest.raises(radixshmem.ServerNotReady, match="no memory for you"):
            cl.query(hashes(1))
        assert not os.path.exists(f"/dev/shm{c.name}_data")
        del srv._bring_up                                   # the next configure starts over
        assert cl.configure(geometry()).mode in ("configuring", "ready")
        assert cl.wait_ready(30).mode == "ready"
        assert cl.status().last_error == ""
        assert cl.query(hashes(1)).common_hit == 0          # attaches now
        cl.close()
    finally:
        srv.close()


def test_cli_builds_the_server_config():
    from radixshmem import cli

    c = cli.build_config(cli.parse_args([
        "--name", "/x", "--data-bytes", "64M", "--swa-ratio", "0.25", "--transfer-dev", "mlx5_0,mlx5_1",
        "--transfer-dev", "mlx5_2", "--no-prefault", "--expected-min-nodes", "2", "--rpc-address", "0.0.0.0"]))
    assert c.name == "/x" and c.data_bytes == 64 << 20 and c.swa_ratio == 0.25
    assert c.transfer_devices == ["mlx5_0", "mlx5_1", "mlx5_2"] and not c.prefault and c.distributed
    assert cli.build_config(cli.parse_args([])).index_only
    with pytest.raises(ValueError, match="--swa-ratio"):
        cli.build_config(cli.parse_args(["--swa-ratio", "0.1"]))
    with pytest.raises(ValueError, match="ratio"):
        cli.build_config(cli.parse_args(["--data-bytes", "1M", "--swa-ratio", "0.6", "--mamba-ratio", "0.5"]))


def test_local_read_only_client_reads_index_and_store_but_never_writes(cfg):
    c = cfg(**data())
    srv = up(c, geometry())
    try:
        w = client(c)
        h = hashes(3)
        slots = w.allocate_slots(3)
        assert w.insert(h, slots).error == OK
        w.slot_view(slots[1])[:4] = b"ro!!"

        ro = client(c, local_read_only=True)
        assert ro.local_read_only and not ro.is_distributed() and ro.rank() == 0
        assert ro.info.world_size == 1 and ro.geometry == w.geometry
        assert bytes(ro.slot_view(slots[1])[:4]) == b"ro!!"
        q = ro.query(h, lock=True)
        assert q.common_hit == 3 and list(q.full_fragments[0][2]) == list(slots)
        q.finalize()
        assert ro.query(h, local_only=False, update_meta=False).common_hit == 3
        for call in (lambda: ro.pull_async(h), lambda: ro.allocate_slots(1),
                     lambda: ro.insert(h, slots), lambda: ro.recycle_slots(slots), ro.reset):
            with pytest.raises(RuntimeError):
                call()
        assert w.query(h, update_meta=False).common_hit == 3     # the refused reset changed nothing
        assert ro.check_health() == 0
        ro.close()

        core = radixshmem._core.RadixClient(c.name, local_read_only=True)     # index-only callers
        assert not core.is_distributed()
        assert core.query(h, update_meta=False).common_hit == 3
        with pytest.raises(RuntimeError):
            core.allocate_slots(1)
        del core
        w.close()
    finally:
        srv.close()


class _Q:
    """Scripted cluster query: local prefix + a rank-7 suffix; `pin` is a real
    locked local query so the prefix is pinned like the C++ query would."""

    status = 0
    swa_source_rank, swa_slots, mamba = -1, np.empty(0, np.int32), None

    def __init__(self, local, remote_off, remote_slots, common_hit):
        self.full_fragments = [(0, 0, local), (7, remote_off, remote_slots)]
        self.common_hit = common_hit
        self.pin = None

    def finalize(self):
        if self.pin is not None:
            self.pin.finalize()


def scripted_index(real, script):
    """Index whose cluster query returns script(hashes) with the real local pin."""

    class Index:
        def __getattr__(self, name):
            return getattr(real, name)

        def query(self, hs, **kw):
            if kw.get("local_only"):
                return real.query(hs, **kw)
            q = script(hs)
            q.pin = real.query(hs, local_only=True, lock=True, update_meta=False)
            return q

    return Index()


class _Transport:
    """Completes the two newest tickets per WaitAny, so completion order is not
    submission order; tickets in `fail` report a failed transfer, tickets in
    `lose` are unknown to the server, `busy` refuses every Submit."""

    def __init__(self):
        self.items, self.submitted, self.n, self.order = {}, [], 0, []
        self.fail, self.lose, self.busy = set(), set(), False
        self.operations, self.results = {}, {}

    def operation(self, command, epoch, session, sequence, **fields):
        key = (session, sequence)
        if command == "reserve":
            return {"state": "RETIRED" if self.busy else "RESERVED"}
        if command == "ack":
            return {"state": "RETIRED"}
        if command == "submit" and key not in self.operations:
            self.operations[key] = self.submit(fields["items"], fields["timeout_ms"])
        ticket = self.operations[key]
        if ticket in self.lose:
            return {"state": "RETIRED"}  # definitive rejection, not UNKNOWN
        if ticket not in self.results:
            self.results.update(self.wait_any([ticket], 0)["done"])
        return {"state": "DONE", "ticket": ticket, "replies": self.results[ticket]}

    def submit(self, items, timeout_ms):
        if self.busy:
            raise radixshmem.ServerBusy("full")
        self.n += 1
        self.items[self.n] = items
        self.submitted.append(items)
        return self.n

    def wait_any(self, tickets, timeout_ms):
        time.sleep(0.002)
        pick = sorted(t for t in tickets if t not in self.lose)[-2:]
        self.order += pick
        done = {t: [(t not in self.fail, 0, 1, "" if t not in self.fail else "boom")]
                * len(self.items.pop(t)) for t in pick}
        return {"done": done, "unknown": [t for t in tickets if t in self.lose]}

    def close(self):
        pass


def test_pull_async_promotes_remote_suffixes_out_of_order(cfg):
    """The cluster query is scripted and the server is a fake; allocation,
    promotion and pinning run on the real index for eight concurrent jobs."""
    c = cfg(**data(data_bytes=1 << 20))
    srv = up(c, geometry())
    try:
        cl = client(c)
        real, scripted = cl.index, {}
        transport = _Transport()
        cl._index, cl._t = scripted_index(real, lambda hs: scripted[int(hs[0])]), transport

        def prefix(k, local_blocks, total=8):
            h = hashes(total, base=1000 * k)
            local = cl.allocate_slots(local_blocks)
            assert cl.insert(h[:local_blocks], local).error == OK
            remote = np.arange(100 * k, 100 * k + total - local_blocks, dtype=np.int32)
            scripted[int(h[0])] = _Q(local, local_blocks, remote, total)
            return h, local, remote

        prefixes = [prefix(k, 3) for k in range(8)]
        nodes = real.total_radix_nodes()
        jobs = [cl.pull_async(h) for h, _, _ in prefixes]
        assert len({j.job_id for j in jobs}) == 8
        done = cl.wait_any(jobs, timeout=10)
        assert done and all(j.done() for j in done)
        for (h, local, remote), job in zip(prefixes, jobs):
            r = job.wait(10)
            assert r.common_hit == 8 and r.remote_blocks == 5 and r.source_rank == 7
            assert list(r.full_slots[:3]) == list(local)
            r.finalize()
            assert real.query(h, local_only=True).common_hit == 8
        assert real.total_radix_nodes() == nodes       # promoted suffixes merged into the unpinned prefixes
        assert len(transport.order) == 8
        assert sorted(items[0][2] for items in transport.submitted) == \
            [list(range(100 * k, 100 * k + 5)) for k in range(8)]
        assert all(items[0][:2] == (7, 0) for items in transport.submitted)
        assert cl.transfer_failures == 0

        h, local, _ = prefix(8, 2)
        transport.fail.add(transport.n + 1)
        r = cl.pull_async(h).wait(10)
        assert r.common_hit == 2 and r.remote_blocks == 0 and list(r.full_slots) == list(local)
        r.finalize()
        h, local, _ = prefix(9, 2)
        transport.lose.add(transport.n + 1)
        r = cl.wait(cl.pull_async(h).job_id, 10)
        assert r.common_hit == 2 and r.remote_blocks == 0
        r.finalize()
        transport.busy = True
        h, local, _ = prefix(10, 2)
        r = cl.pull_async(h).wait(10)
        assert r.common_hit == 2 and r.remote_blocks == 0
        r.finalize()
        assert cl.transfer_failures == 2 and cl.promotion_shortfalls == 0
        assert real.mempool_used() == 8 * 3 + 8 * 5 + 3 * 2
        assert real.check_health() == 0
        cl.close()
        assert cl._jobs == {} and cl._pending == {}
    finally:
        srv.close()


def test_promotion_survives_a_prefix_evicted_in_the_unpinned_window(cfg):
    """The pin drops before the promoted blocks are published; if the prefix is
    evicted in between, the insert lands nothing, its slots return to the pool
    and the job completes with the re-queried (here empty) hit."""
    c = cfg(**data(data_bytes=1 << 20))
    srv = up(c, geometry())
    try:
        cl = client(c)
        real = cl.index
        h = hashes(8)
        local = cl.allocate_slots(3)
        assert cl.insert(h[:3], local).error == OK
        index = scripted_index(real, lambda hs: _Q(local, 3, np.arange(100, 105, dtype=np.int32), 8))

        def insert(hs, slots, **kw):
            hoard = real.allocate_slots(real.mempool_free() + 3)    # only the unpinned prefix can be evicted
            try:
                return real.insert(hs, slots, **kw)
            finally:
                real.recycle_slots(hoard)

        index.insert = insert
        cl._index, cl._t = index, _Transport()
        r = cl.pull_async(h).wait(10)
        assert r.common_hit == 0 and r.remote_blocks == 5 and len(r.full_slots) == 0
        assert cl.promotion_shortfalls == 1 and cl.transfer_failures == 0
        assert real.query(h, local_only=True).common_hit == 0
        assert real.mempool_used() == 0 and real.check_health() == 0
        cl.close()
    finally:
        srv.close()


def test_stage_recycles_a_short_aux_allocation(cfg):
    """A partial SWA allocation (impossible with the all-or-none pool, forced
    here) must go back to the pool instead of leaking."""
    c = cfg(**data(data_bytes=1 << 20, swa_ratio=8 * 1024 / (1 << 20)))
    srv = up(c, geometry(swa_slot_bytes=1024, swa_window_blocks=2))
    try:
        cl = client(c)
        real, recycled = cl.index, []
        h = hashes(4)
        local = cl.allocate_slots(2)
        assert cl.insert(h[:2], local).error == OK

        class Q(_Q):
            swa_source_rank, swa_slots = 7, np.array([5, 6], dtype=np.int32)

        index = scripted_index(real, lambda hs: Q(local, 2, np.array([40, 41], np.int32), 4))
        index.allocate_slots = lambda n, kind=FULL: real.allocate_slots(n - 1 if kind == SWA else n, kind)

        def recycle_slots(slots, kind=FULL):
            recycled.append((kind, list(slots)))
            real.recycle_slots(slots, kind)

        index.recycle_slots = recycle_slots
        cl._index, cl._t = index, _Transport()
        r = cl.pull_async(h).wait(10)
        assert r.common_hit == 4 and r.remote_blocks == 2 and len(r.swa_slots) == 0
        r.finalize()
        assert [k for k, s in recycled if s] == [SWA]
        assert real.swa_mempool_total() == 8 and real.check_health() == 0
        cl.close()
    finally:
        srv.close()


class _Xfer:
    """A SlotTransfer that completes a batch on its third poll; handles in
    `hang` stay pending and refuse release, `reject` fails every read."""

    local_segment = "me"

    def __init__(self, data):
        self.d, self.polls, self.n, self.hang, self.reject = data, {}, 0, set(), False
        self.release_busy = threading.Event()

    def read(self, seg, remote, local, kind):
        if self.reject:
            return self.d.INVALID_BATCH, 0
        self.n += 1
        self.polls[self.n] = 0
        return self.n, 1

    def poll(self, h):
        self.polls[h] += 1
        if h in self.hang or self.polls[h] < 3:
            return self.d.XferState.PENDING, 0
        return self.d.XferState.COMPLETED, 4096

    def release(self, h):
        if h in self.hang:
            self.release_busy.set()
            return self.d.XFER_EBUSY
        return self.d.XFER_OK


def test_server_jobs_complete_timeout_busy_and_expire(cfg):
    from radixshmem import _data

    c = cfg(**data(data_bytes=1 << 20, max_inflight=2, max_pending_jobs=3, job_ttl_s=0.2))
    srv = up(c, geometry())
    try:
        x = _Xfer(_data)
        srv._xfer, srv._peers = x, {7: "peer"}
        item = (7, 0, [1, 2], [3, 4])

        def collect(tickets, timeout=2.0):
            done, deadline = {}, time.monotonic() + timeout
            while len(done) < len(tickets) and time.monotonic() < deadline:
                done.update(srv.wait_any([t for t in tickets if t not in done], 500)["done"])
            return done

        t1, t2 = srv.submit([item, item], 5000), srv.submit([item], 5000)
        done = collect([t1, t2])
        assert done[t1] == [(True, 4096, 1, "")] * 2 and done[t2] == [(True, 4096, 1, "")]
        assert srv.stats()["transfers"] == 3 and srv.stats()["pending_jobs"] == 0

        x.hang.add(x.n + 1)
        t3 = srv.submit([item], 100)
        assert x.release_busy.wait(5)
        assert srv.wait_any([t3], 20) == {"done": {}, "unknown": []}
        assert not srv._jobs[t3].done
        x.hang.clear()
        rep = collect([t3])[t3][0]
        assert rep[0] is False and "TIMEOUT" in rep[3]

        x.reject = True
        t4 = srv.submit([item], 1000)
        assert collect([t4])[t4][0][0] is False
        x.reject = False

        ts = [srv.submit([item], 5000) for _ in range(3)]
        with pytest.raises(radixshmem.ServerBusy):
            srv.submit([item], 5000)
        assert srv.wait_any([999], 10) == {"done": {}, "unknown": [999]}
        time.sleep(1.5)
        assert srv.wait_any(ts, 10)["unknown"] == ts
        srv.submit([item], 5000)
    finally:
        srv._xfer = None            # the fake must not be released as a real SlotTransfer
        srv.close()


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("release_error", ["XFER_EBUSY", "XFER_EBACKEND"])
def test_timed_out_pull_keeps_pins_slots_and_budget_until_release(cfg, cancel, release_error):
    """Delayed I/O writes into real staging slots after the deadline. Neither
    timeout nor cancellation may let the client recycle them or the source pin.
    Unreleased batches must also keep their place in the server's budget."""
    from radixshmem import _data

    c = cfg(**data(data_bytes=8 * 8192, max_inflight=1, max_pending_jobs=2))
    srv = up(c, geometry())
    cl = None
    allow_release, release_attempted = threading.Event(), threading.Event()
    read_started = threading.Event()
    try:
        cl = client(c)
        real = cl.index
        h = hashes(4)
        local = cl.allocate_slots(2)
        assert cl.insert(h[:2], local).error == OK
        q = _Q(local, 2, np.array([100, 101], np.int32), 4)
        cl._index = scripted_index(real, lambda hs: q)

        class DelayedTransfer:
            submitted = 0
            released = 0

            def read(self, seg, remote, destination, kind):
                self.submitted += 1
                self.destination = destination
                read_started.set()
                return self.submitted, 1

            def poll(self, handle):
                if allow_release.is_set():
                    return _data.XferState.COMPLETED, 2 * 8192
                return _data.XferState.PENDING, 0

            def release(self, handle):
                release_attempted.set()
                if not allow_release.is_set():
                    return getattr(_data, release_error)
                # Model a late DMA write just before the backend lets go.
                assert bool(q.pin.finalize)
                assert real.mempool_used() == 4
                for slot in self.destination:
                    cl.slot_view(slot)[:4] = b"late"
                self.released += 1
                return _data.XFER_OK

        x = DelayedTransfer()
        srv._xfer = x
        srv._peer = lambda rank: "peer"

        class Transport:
            submit = staticmethod(srv.submit)
            wait_any = staticmethod(srv.wait_any)

            def operation(self, command, epoch, session, sequence, **fields):
                return srv.operation(dict(command=command, epoch=epoch, session=session,
                                          sequence=sequence, **fields))

            def close(self):
                pass

        cl._t.close()
        cl._t = Transport()
        job = cl.pull_async(h, timeout_ms=5000)
        assert read_started.wait(5)
        deadline = time.monotonic() + 5
        while job._ticket < 0 and time.monotonic() < deadline:
            time.sleep(0.001)
        with srv._jobs_mu:
            srv._jobs[job._ticket].deadline = time.monotonic() - 1
        assert release_attempted.wait(5)
        with pytest.raises(TimeoutError, match="still running"):
            job.wait(0.05)
        assert bool(q.pin.finalize) and real.mempool_used() == 4
        assert not srv._jobs[job._ticket].done
        assert srv.wait_any([job._ticket], 10) == {"done": {}, "unknown": []}

        # Allocation pressure cannot reclaim the pinned prefix or staging.
        free = real.mempool_free()
        pressure = real.allocate_slots(free + len(local))
        try:
            assert len(pressure) == free
            assert real.query(h, local_only=True, update_meta=False).common_hit == 2
        finally:
            real.recycle_slots(pressure)

        queued = srv.submit([(7, 0, [102], [7])], 5000)
        with srv._jobs_mu:
            srv._jobs[queued].deadline = time.monotonic() - 1
        with pytest.raises(radixshmem.ServerBusy):
            srv.submit([(7, 0, [103], [6])], 5000)
        if cancel:
            job.cancel()
        assert x.submitted == 1 and bool(q.pin.finalize)

        allow_release.set()
        result = job._f.result(5) if cancel else job.wait(5)
        assert result.common_hit == 2 and result.remote_blocks == 0
        assert list(result.full_slots) == list(local)
        result.finalize()
        assert not bool(q.pin.finalize)
        assert cl.transfer_failures == 1 and real.mempool_used() == 2
        assert real.query(h, local_only=True, update_meta=False).common_hit == 2
        assert real.check_health() == 0

        # The timed-out batch occupied max_inflight until release. The queued
        # deadline expired meanwhile, so it must not start another transfer.
        reply = srv.wait_any([queued], 5000)["done"][queued][0]
        assert not reply[0] and "TIMEOUT" in reply[3] and reply[2] == 0
        assert x.submitted == 1 and x.released == 1
        cl.close()
        assert srv.stats()["pending_jobs"] == 0
    finally:
        allow_release.set()
        if cl is not None:
            cl.close()
        srv._running = False
        srv._wake.set()
        if srv._thread is not None:
            srv._thread.join(5)
        srv._xfer = None
        srv.close()


@pytest.mark.parametrize("lost_command", ["submit", "query", "ack"])
def test_operation_reply_loss_is_retried_without_releasing_live_memory(cfg, lost_command):
    from radixshmem import _data

    c = cfg(**data(data_bytes=1 << 20))
    srv = up(c, geometry())
    cl = client(c)
    x = _Xfer(_data)
    x.hang.add(1)
    srv._xfer, srv._peer = x, lambda rank: "peer"
    real = cl.index
    h = hashes(4)
    local = cl.allocate_slots(2)
    assert cl.insert(h[:2], local).error == OK
    q = _Q(local, 2, np.array([10, 11], np.int32), 4)
    cl._index = scripted_index(real, lambda hs: q)
    original = cl._t.operation
    dropped = threading.Event()

    def operation(command, *args, **kw):
        reply = original(command, *args, **kw)
        if command == lost_command and not dropped.is_set():
            dropped.set()
            raise TimeoutError("reply lost after server processed the operation")
        return reply

    cl._t.operation = operation
    try:
        job = cl.pull_async(h, timeout_ms=50)
        assert x.release_busy.wait(5)
        assert not job.resources_released and not job.done()
        assert bool(q.pin.finalize) and real.mempool_used() == 4
        # Both close paths must preserve ownership on timeout.
        with pytest.raises(TimeoutError, match="retained"):
            cl.close(timeout_s=0)
        with pytest.raises(TimeoutError, match="retained"):
            srv.close(timeout_s=0)
        assert cl._store is not None and srv._xfer is x
        assert srv.stats()["reserved_bytes"] == 4 * 8192
        x.hang.clear()
        assert dropped.wait(5)
        cl.close(timeout_s=5)
        assert x.n == 1  # lost Submit never starts a second read
        assert job.resources_released and not bool(q.pin.finalize)
        assert real.mempool_used() == 2
        assert srv.stats()["reserved_bytes"] == 0
    finally:
        x.hang.clear()
        cl.close(timeout_s=5)
        srv._xfer = None
        srv.close(timeout_s=5)


def test_node_admission_precedes_peer_query_and_staging(cfg):
    c = cfg(**data(data_bytes=1 << 20, max_reserved_bytes=8192))
    srv = up(c, geometry())
    cl = client(c)
    real = cl.index
    queries = []
    original = real.query

    class Index:
        def __getattr__(self, name):
            return getattr(real, name)

        def query(self, *args, **kw):
            queries.append(kw["local_only"])
            assert kw["local_only"]
            return original(*args, **kw)

    cl._index = Index()
    try:
        result = cl.pull_async(hashes(4), block=False).wait(1)
        assert result.common_hit == 0
        assert queries and all(queries)
        assert real.mempool_used() == 0
        assert srv.stats()["admission_refused"] == 1
    finally:
        cl.close()
        srv.close()


def test_each_component_consumes_backend_admission(cfg):
    from radixshmem import _data

    c = cfg(**data(data_bytes=1 << 20, max_inflight=1))
    srv = up(c, geometry())
    x = _Xfer(_data)
    x.hang.add(1)
    srv._xfer, srv._peer = x, lambda rank: "peer"
    try:
        ticket = srv.submit([(7, 0, [10], [0]), (7, 0, [11], [1]), (7, 0, [12], [2])], 20)
        assert x.release_busy.wait(5)
        assert x.n == 1
        x.hang.clear()
        reply = srv.wait_any([ticket], 5000)["done"][ticket]
        assert len(reply) == 3 and not any(r[0] for r in reply)
        assert x.n == 1
    finally:
        x.hang.clear()
        srv.close()
