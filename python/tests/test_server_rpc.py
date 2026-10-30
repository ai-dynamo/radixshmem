# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RadixServer + RadixClient over the gRPC channel, standalone (no etcd, no RDMA):
the server starts waiting, a client brings the geometry, clients attach on use."""

import os
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
    assert cli.build_config(cli.parse_args([])).transfer_log == "-"
    assert cli.build_config(cli.parse_args(["--transfer-log", ""])).transfer_log == ""
    assert cli.build_config(cli.parse_args([])).role == "prefill"
    assert cli.build_config(cli.parse_args(["--decode"])).role == "decode"
    with pytest.raises(ValueError, match="--swa-ratio"):
        cli.build_config(cli.parse_args(["--swa-ratio", "0.1"]))
    with pytest.raises(ValueError, match="ratio"):
        cli.build_config(cli.parse_args(["--data-bytes", "1M", "--swa-ratio", "0.6", "--mamba-ratio", "0.5"]))


def test_role_in_node_info_and_nodes(cfg):
    with pytest.raises(ValueError, match="role"):
        cfg(role="mixed")
    for role in ("prefill", "decode"):
        c = cfg(role=role) if role == "decode" else cfg()
        srv = RadixServer(c).start()
        try:
            cl = client(c)
            assert cl.info.role == role
            nodes = cl.nodes()
            assert len(nodes) == 1 and nodes[0]["role"] == role
            assert nodes[0]["rank"] == -1 and nodes[0]["mode"] == "waiting"
            assert nodes[0]["endpoint"] == c.endpoint
            cl.close()
        finally:
            srv.close()


def test_grpc_is_the_only_remote_op_transport():
    from radixshmem import cli
    from radixshmem.server import ClusterConfig

    config = cli.build_config(cli.parse_args([
        "--remote-op-transport", "grpc", "--remote-op-listen-port", "31001",
        "--rht-transport", "dc", "--peer-index-transport", "dc"]))
    assert config.cluster.remote_op_transport == "grpc"
    assert config.cluster.remote_op_listen_port == 31001
    assert config.cluster.rht_transport == config.cluster.peer_index_transport == "dc"
    for transport in ("zmq", "dc", "rdma", "xrc", "unknown"):
        with pytest.raises(SystemExit):
            cli.parse_args(["--remote-op-transport", transport])
        with pytest.raises(ValueError, match="only grpc"):
            ClusterConfig(remote_op_transport=transport)
    with pytest.raises(SystemExit):
        cli.parse_args(["--zmq-listen-port", "31001"])
    assert not hasattr(radixshmem._core.RadixServerConfig(), "zmq_listen_port")


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
        assert transport.order != sorted(transport.order)
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
        assert cl.transfer_failures == 3 and cl.promotion_shortfalls == 0
        assert real.mempool_used() == 8 * 3 + 8 * 5 + 3 * 2
        assert real.check_health() == 0
        assert cl._jobs == {} and cl._pending == {}
        cl.close()
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
        return self.d.XFER_EBUSY if h in self.hang else self.d.XFER_OK


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
        rep = collect([t3])[t3][0]
        assert rep[0] is False and "TIMEOUT" in rep[3] and srv._zombies == [x.n]

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


def test_transfer_log_writes_one_line_per_transfer_that_moved_data(cfg, tmp_path):
    import re

    from radixshmem import _data

    log = tmp_path / "xfer.log"
    c = cfg(**data(data_bytes=1 << 20, max_inflight=4, transfer_log=str(log)))
    srv = up(c, geometry())
    try:
        x = _Xfer(_data)
        srv._xfer, srv._peers = x, {7: "peer"}
        item = (7, 0, [1, 2], [3, 4])

        def collect(t, timeout=2.0):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                done = srv.wait_any([t], 500)["done"]
                if done:
                    return done[t]
            raise AssertionError("job did not finish")

        assert collect(srv.submit([item, item], 5000)) == [(True, 4096, 1, "")] * 2
        x.reject = True                       # rejected reads move nothing: not logged
        assert collect(srv.submit([item], 1000))[0][0] is False
        x.reject = False
        x.hang.add(x.n + 1)                   # a timeout with 0 bytes done: not logged
        assert collect(srv.submit([item], 100))[0][0] is False
    finally:
        srv._xfer = None
        srv.close()
    lines = log.read_text().splitlines()
    assert len(lines) == 2
    pat = (r"xfer \d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z peer=7 kind=FULL slots=2 desc=1 bytes=4096 "
           r"us=(\d+\.\d) MBps=(\d+\.\d) wait_us=\d+\.\d inflight=[01]$")
    for line in lines:
        m = re.match(pat, line)
        assert m, line
        assert abs(4096 / float(m.group(1)) - float(m.group(2))) <= 0.1 + 0.01 * float(m.group(2))
