# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""radixshmem._data binding tests — the data plane from Python.

The C++ suite (test_slot_store, test_slot_transfer_e2e) owns the functional
coverage. These tests assert that the binding surface carries it faithfully:
   - geometry and lifetime come through as Python attributes,
   - slot views are windows onto the mapping, not copies,
   - slot lists map to the right C++ parameters, and coalescing is observable
     from Python (read/write's descriptor count) exactly as it is from C++,
   - a real two-process transfer, driven entirely from Python, moves the same
     bytes into the same slots as the C++ end-to-end test does.

Everything past the store tests needs a transfer backend, and the only backend
is Mooncake, so those tests fork: an ibverbs context does not survive fork(), so
every engine is built inside a child and the parent never touches one. They SKIP
when no Mooncake engine can come up on this host.
"""

import os
import struct
import sys

import radixshmem

SLOTS = 64
SLOT_BYTES = 65536


def store_cfg(name, slots=SLOTS, slot_bytes=SLOT_BYTES, prefault=False,
              swa=None, mamba=None):
    return radixshmem._data.SlotStoreConfig(
        name=name, full=radixshmem._data.SlotPoolConfig(slots, slot_bytes),
        swa=swa or radixshmem._data.SlotPoolConfig(),
        mamba=mamba or radixshmem._data.SlotPoolConfig(),
        slot_align=4096, hugepage_path="", prefault=prefault)


FULL, SWA, MAMBA = (radixshmem.ComponentType.FULL, radixshmem.ComponentType.SWA,
                    radixshmem.ComponentType.MAMBA)


def mc_cfg(publish, protocol, devices=()):
    # rpc_port=0: the engine binds what it can and we read the name back off
    # local_segment. Naming the segment ourselves would hand Mooncake a host it
    # cannot resolve.
    return radixshmem._data.TransferConfig(
        backend="mooncake", metadata="P2PHANDSHAKE", ip_or_host="127.0.0.1",
        rpc_port=0, protocol=protocol, device_filter=list(devices),
        publish=publish)


def rdma_devices():
    """HCAs this host exposes to ibverbs. Read from sysfs so listing them costs
    no device context -- opening one before a fork would poison the children."""
    try:
        return sorted(os.listdir("/sys/class/infiniband"))
    except OSError:
        return []


def pattern(slot):
    """Same per-slot byte as the C++ end-to-end test, so a mis-addressed slot
    shows up as the wrong pattern rather than as plausible-looking data."""
    return (slot * 37 + 11) & 0xFF


def fill(store, slot, value):
    store.slot_view(slot)[:] = bytes([value]) * store.slot_bytes


def holds(store, slot, value):
    return store.read_slot(slot) == bytes([value]) * store.slot_bytes


# ==================================================================
def test_store_geometry_and_views():
    """create → geometry attributes, writable views, byte-accurate read back."""
    name = "/py_data_geo"
    radixshmem._data.SlotStore.destroy(name, "")
    s = radixshmem._data.SlotStore.create(store_cfg(name, slots=8, slot_bytes=4096))
    try:
        assert s.valid
        assert s.is_owner
        assert s.name == name
        assert s.num_slots == 8
        assert s.slot_bytes == 4096
        assert s.data_bytes == 8 * 4096
        assert s.slot_offset(3) == 3 * 4096
        assert "8" in repr(s) and "4096" in repr(s)

        # The view is a window: writing through it is what fills the slot.
        v = s.slot_view(5)
        assert len(v) == 4096
        assert not v.readonly
        v[:8] = b"abcdefgh"
        assert s.read_slot(5)[:8] == b"abcdefgh"

        s.write_slot(2, b"xyz")
        assert s.read_slot(2)[:3] == b"xyz"

        # data_view spans the whole registerable range and overlaps the slots.
        d = s.data_view()
        assert len(d) == 8 * 4096
        assert bytes(d[5 * 4096:5 * 4096 + 8]) == b"abcdefgh"

        # Out of range is an exception, not a silent nullptr.
        for bad in (8, 99):
            try:
                s.slot_view(bad)
                assert False, "expected out-of-range slot to raise"
            except IndexError:
                pass
        try:
            s.write_slot(0, b"x" * 5000)
            assert False, "expected oversized payload to raise"
        except ValueError:
            pass
    finally:
        del s
        radixshmem._data.SlotStore.destroy(name, "")
    print("  PASS: store_geometry_and_views")


def test_store_attach_shares_memory():
    """attach reads the geometry off the region and maps the same bytes."""
    name = "/py_data_attach"
    radixshmem._data.SlotStore.destroy(name, "")
    owner = radixshmem._data.SlotStore.create(store_cfg(name, slots=4, slot_bytes=4096))
    try:
        guest = radixshmem._data.SlotStore.attach(name, "", 5000)
        assert guest.valid
        assert not guest.is_owner
        # Geometry is not passed in; it comes from the region.
        assert guest.num_slots == 4
        assert guest.slot_bytes == 4096

        owner.slot_view(1)[:4] = b"ping"
        assert guest.read_slot(1)[:4] == b"ping"
        guest.slot_view(3)[:4] = b"pong"
        assert owner.read_slot(3)[:4] == b"pong"
        del guest
    finally:
        del owner
        radixshmem._data.SlotStore.destroy(name, "")
    print("  PASS: store_attach_shares_memory")


def test_store_pools():
    """Three pools: per-pool geometry, views and addressing from Python."""
    name = "/py_data_pools"
    radixshmem._data.SlotStore.destroy(name, "")
    cfg = store_cfg(name, slots=4, slot_bytes=4096, swa=radixshmem._data.SlotPoolConfig(2, 4096))
    cfg.mamba.num_slots = 1                # in-place edit of a nested pool config
    cfg.mamba.slot_bytes = 100
    assert cfg.pool_mask == 7 and cfg.pool(MAMBA).slot_bytes == 100
    s = radixshmem._data.SlotStore.create(cfg)
    try:
        assert s.pool_mask == 7
        assert s.has_pool(SWA) and s.has_pool(2) and not s.has_pool(FULL) is False
        assert (s.num_slots, s.slot_bytes) == (4, 4096)          # FULL shorthands
        assert s.pool(SWA).num_slots == 2 and s.pool(SWA).base == 4 * 4096
        assert s.pool(MAMBA).slot_bytes == 4096                   # 100 rounded up
        assert s.pool(MAMBA).base == 4 * 4096 + 2 * 4096
        assert s.data_bytes == 7 * 4096
        lay = s.layout()
        assert lay.pool_mask == 7 and lay[MAMBA].base == s.pool(MAMBA).base
        assert lay.same_strides(s.layout())

        # Same id, different pool, different bytes; ints accepted as kinds.
        s.slot_view(1, SWA)[:3] = b"swa"
        s.slot_view(1)[:4] = b"full"
        s.write_slot(0, b"mamba", MAMBA)
        assert s.read_slot(1, SWA)[:3] == b"swa"
        assert s.read_slot(1)[:4] == b"full"
        assert s.read_slot(0, 2)[:5] == b"mamba"
        assert s.slot_offset(1, SWA) == 4 * 4096 + 4096
        d = s.data_view()
        assert bytes(d[s.slot_offset(1, SWA):s.slot_offset(1, SWA) + 3]) == b"swa"
        pv = s.pool_view(SWA)
        assert len(pv) == 2 * 4096 and bytes(pv[4096:4099]) == b"swa"

        for bad in ((2, SWA), (1, MAMBA), (4, FULL)):
            try:
                s.slot_view(*bad)
                assert False, "expected out-of-range slot to raise"
            except IndexError:
                pass
        for bad_kind in (3, "SWA"):
            try:
                s.pool(bad_kind)
                assert False, "expected a bad kind to raise"
            except (ValueError, TypeError):
                pass

        guest = radixshmem._data.SlotStore.attach(name, "", 5000)
        assert guest.pool_mask == 7 and guest.read_slot(1, SWA)[:3] == b"swa"
        del guest
    finally:
        del s
        radixshmem._data.SlotStore.destroy(name, "")

    # A FULL-only store has no other pool.
    radixshmem._data.SlotStore.destroy(name, "")
    s = radixshmem._data.SlotStore.create(store_cfg(name, slots=2, slot_bytes=4096))
    try:
        assert s.pool_mask == 1 and not s.has_pool(SWA)
        try:
            s.pool_view(SWA)
            assert False, "expected missing pool to raise"
        except IndexError:
            pass
    finally:
        del s
        radixshmem._data.SlotStore.destroy(name, "")
    print("  PASS: store_pools")


def test_pools_aligned_with_index():
    """check_pools_aligned: index pools vs store pools, both from one config."""
    if not hasattr(radixshmem, "_core"):
        print("  SKIP: pools_aligned_with_index (no _core)")
        return
    name = "/py_data_align"
    radixshmem._data.SlotStore.destroy(name, "")
    srv = radixshmem._core.RadixServer("/py_data_align_idx", radixshmem._core.ShmConfig(
        max_blocks=8, block_size=1, component_mask=7, swa_window_blocks=2,
        mamba_state_bytes=8192, swa_max_blocks=4, mamba_max_slots=2))
    cl = radixshmem._core.RadixClient(srv)
    assert cl.component_config.component_mask == 7
    assert (cl.mempool_total(), cl.swa_mempool_total(), cl.mamba_mempool_total()) == (8, 4, 2)

    good = radixshmem._data.SlotStore.create(store_cfg(
        name, slots=8, slot_bytes=4096,
        swa=radixshmem._data.SlotPoolConfig(4, 4096), mamba=radixshmem._data.SlotPoolConfig(2, 8192)))
    try:
        radixshmem.check_pools_aligned(cl, good)
        radixshmem.check_pools_aligned(srv, good)
    finally:
        del good
        radixshmem._data.SlotStore.destroy(name, "")

    for label, cfg in (
        ("mask", store_cfg(name, slots=8, slot_bytes=4096)),
        ("swa count", store_cfg(name, slots=8, slot_bytes=4096,
                                swa=radixshmem._data.SlotPoolConfig(3, 4096),
                                mamba=radixshmem._data.SlotPoolConfig(2, 8192))),
        ("mamba bytes", store_cfg(name, slots=8, slot_bytes=4096,
                                  swa=radixshmem._data.SlotPoolConfig(4, 4096),
                                  mamba=radixshmem._data.SlotPoolConfig(2, 4096))),
    ):
        bad = radixshmem._data.SlotStore.create(cfg)
        try:
            radixshmem.check_pools_aligned(cl, bad)
            assert False, "expected misalignment (%s) to raise" % label
        except ValueError:
            pass
        finally:
            del bad
            radixshmem._data.SlotStore.destroy(name, "")
    del cl, srv
    print("  PASS: pools_aligned_with_index")


# ==================================================================
# Two processes over a real engine. Mirrors tests/dataplane/
# test_slot_transfer_e2e.cpp: same pattern, same slot lists, same expected
# descriptor counts — the point being that the Python path lands on the same
# bytes in the same slots.
PUSH_LOCAL = [40, 41, 42, 50]
PUSH_REMOTE = [60, 61, 62, 55]
READ_RUN = 16
SWA_POOL = radixshmem._data.SlotPoolConfig(8, 4096)


def _write_str(fd, s):
    b = s.encode()
    os.write(fd, struct.pack("<I", len(b)))
    os.write(fd, b)


def _read_str(fd):
    head = os.read(fd, 4)
    if len(head) != 4:
        return ""
    (n,) = struct.unpack("<I", head)
    out = b""
    while len(out) < n:
        chunk = os.read(fd, n - len(out))
        if not chunk:
            return ""
        out += chunk
    return out.decode()


def _spawn(fn):
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            code = fn()
        except BaseException:                       # noqa: BLE001
            import traceback
            traceback.print_exc()
            code = 90
        finally:
            sys.stderr.flush()
            os._exit(code if code else 0)
    return pid


def _reap(pid, who):
    _, status = os.waitpid(pid, 0)
    if os.WIFEXITED(status):
        code = os.WEXITSTATUS(status)
        if code == 0:
            return True
        sys.stderr.write("\n    %s child exited with step code %d\n" % (who, code))
    else:
        sys.stderr.write("\n    %s child died on signal %d\n" % (who, os.WTERMSIG(status)))
    return False


def _mooncake_available():
    """Probing costs a fork, but building an engine in this process would poison
    every later fork, which is worse."""
    if _mooncake_available.cached is None:
        pid = _spawn(lambda: 0 if radixshmem._data.create_transfer_backend(
            mc_cfg(True, "rdma")) is not None else 1)
        _, status = os.waitpid(pid, 0)
        _mooncake_available.cached = os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
    return _mooncake_available.cached


_mooncake_available.cached = None


def _publisher(shm, protocol, fds, devices=()):
    name_r, name_w, done_r, done_w = fds
    os.close(name_r)
    os.close(done_w)
    s = radixshmem._data.SlotStore.create(store_cfg(shm, prefault=True, swa=SWA_POOL))
    be = radixshmem._data.create_transfer_backend(mc_cfg(True, protocol, devices))
    if be is None:
        return 2
    x = radixshmem._data.SlotTransfer(s, be)
    if x.register_store("cpu:0") != radixshmem._data.XFER_OK:
        return 3

    for i in range(SLOTS):
        fill(s, i, pattern(i))
    for slot in PUSH_REMOTE:            # landing zone: a no-op would be caught
        fill(s, slot, 0x00)
    for i in range(SWA_POOL.num_slots):
        s.slot_view(i, SWA)[:] = bytes([pattern(i) ^ 0xFF]) * SWA_POOL.slot_bytes

    _write_str(name_w, x.local_segment)
    if os.read(done_r, 1) != b"k":
        return 4

    for local, remote in zip(PUSH_LOCAL, PUSH_REMOTE):
        if not holds(s, remote, pattern(local)):
            return 5
    for i in range(SLOTS):
        if i not in PUSH_REMOTE and not holds(s, i, pattern(i)):
            return 6
    return 0


def _initiator(shm, protocol, fds, devices=()):
    name_r, name_w, done_r, done_w = fds
    os.close(name_w)
    os.close(done_r)
    s = radixshmem._data.SlotStore.create(store_cfg(shm, prefault=True, swa=SWA_POOL))
    be = radixshmem._data.create_transfer_backend(mc_cfg(False, protocol, devices))
    if be is None:
        return 2
    x = radixshmem._data.SlotTransfer(s, be)
    if x.register_store("cpu:0") != radixshmem._data.XFER_OK:
        return 3

    peer = _read_str(name_r)
    if not peer:
        return 4
    # Identical configs: our layout stands in for the registry entry.
    rc, geo = x.attach_peer(peer, s.layout())
    if rc != radixshmem._data.XFER_OK:
        return 5
    if geo[FULL].num_slots != SLOTS or geo[SWA].slot_bytes != SWA_POOL.slot_bytes:
        return 6
    # The stride form is FULL-only; a contradicting layout is refused.
    rc2, _ = x.attach_peer(peer, SLOT_BYTES)
    if rc2 != radixshmem._data.XFER_EINVAL:
        return 30
    other = s.layout()
    other[SWA] = radixshmem._data.PoolGeometry(other[SWA].base, other[SWA].num_slots, 8192)
    rc3, _ = x.attach_peer(peer, other)
    if rc3 != radixshmem._data.XFER_EGEOMETRY:
        return 31
    if x.peer_geometry(peer)[SWA].num_slots != SWA_POOL.num_slots:
        return 32

    # SWA run: peer slots 2..5 -> our 0..3, one request, FULL slot 0 untouched.
    for i in range(4):
        s.slot_view(i, SWA)[:] = b"\x00" * SWA_POOL.slot_bytes
    fill(s, 0, 0xEE)
    hs, ndesc = x.read(peer, [2, 3, 4, 5], [0, 1, 2, 3], kind=SWA)
    if hs == radixshmem._data.INVALID_BATCH or ndesc != 1:
        return 33
    if x.wait(hs, 30000) != radixshmem._data.XferState.COMPLETED:
        return 34
    x.release(hs)
    for i in range(4):
        if s.read_slot(i, SWA) != bytes([pattern(2 + i) ^ 0xFF]) * SWA_POOL.slot_bytes:
            return 35
    if not holds(s, 0, 0xEE):
        return 36

    local = list(range(READ_RUN))
    remote = [8 + i for i in range(READ_RUN)]
    for i in local:
        fill(s, i, 0x00)
    fill(s, READ_RUN, 0xEE)             # sentinel just past the run

    h, ndesc = x.read(peer, remote, local)
    if h == radixshmem._data.INVALID_BATCH:
        return 7
    if ndesc != 1:                      # contiguous both sides: one request
        return 8
    if x.wait(h, 30000) != radixshmem._data.XferState.COMPLETED:
        return 9
    state, done = x.poll(h)
    if x.release(h) != radixshmem._data.XFER_OK:   # COMPLETED batches free outright
        return 19
    if done != READ_RUN * s.slot_bytes:
        return 10
    for lo, re in zip(local, remote):
        if not holds(s, lo, pattern(re)):
            return 11
    if not holds(s, READ_RUN, 0xEE):
        return 12

    for slot in PUSH_LOCAL:
        fill(s, slot, pattern(slot))
    h2, ndesc = x.write(peer, PUSH_LOCAL, PUSH_REMOTE)
    if h2 == radixshmem._data.INVALID_BATCH:
        return 13
    if ndesc != 2:                      # {40,41,42}->{60,61,62} fuses, {50}->{55} does not
        return 14
    if x.wait(h2, 30000) != radixshmem._data.XferState.COMPLETED:
        return 15
    x.release(h2)

    # Pull one back, to confirm the peer really has it.
    fill(s, 30, 0x00)
    h3, _ = x.read(peer, [61], [30])
    if h3 == radixshmem._data.INVALID_BATCH:
        return 16
    if x.wait(h3, 30000) != radixshmem._data.XferState.COMPLETED:
        return 17
    x.release(h3)
    if not holds(s, 30, pattern(41)):
        return 18

    os.write(done_w, b"k")
    return 0


def _run_e2e(protocol, tag, devices=()):
    name_r, name_w = os.pipe()
    done_r, done_w = os.pipe()
    shm_a = "/py_ste_" + tag + "_a"
    shm_b = "/py_ste_" + tag + "_b"
    radixshmem._data.SlotStore.destroy(shm_a, "")
    radixshmem._data.SlotStore.destroy(shm_b, "")

    fds = (name_r, name_w, done_r, done_w)
    pa = _spawn(lambda: _publisher(shm_a, protocol, fds, devices))
    pb = _spawn(lambda: _initiator(shm_b, protocol, fds, devices))
    for fd in fds:
        os.close(fd)

    ok_b = _reap(pb, "initiator")
    ok_a = _reap(pa, "publisher")
    radixshmem._data.SlotStore.destroy(shm_a, "")
    radixshmem._data.SlotStore.destroy(shm_b, "")
    return ok_a and ok_b


def test_e2e_rdma():
    if not _mooncake_available():
        print("  SKIP: e2e_rdma (no Mooncake engine)")
        return
    assert _run_e2e("rdma", "rdma")
    print("  PASS: e2e_rdma")


def test_e2e_tcp():
    if not _mooncake_available():
        print("  SKIP: e2e_tcp (no Mooncake engine)")
        return
    assert _run_e2e("tcp", "tcp")
    print("  PASS: e2e_tcp")


def test_e2e_device_filter():
    """device_filter is how the data plane is kept off the management NIC.
    Pinning both endpoints to one named HCA is the deterministic case: they
    cannot end up on HCAs in different L3 planes, which never reach RTR."""
    if not _mooncake_available():
        print("  SKIP: e2e_device_filter (no Mooncake engine)")
        return
    devs = rdma_devices()
    if not devs:
        print("  SKIP: e2e_device_filter (no RDMA device in sysfs)")
        return
    assert _run_e2e("rdma", "dev1", [devs[0]])
    # An allowlist, intersected with what the host has -- not a list of
    # requirements, so a name that does not exist costs nothing.
    assert _run_e2e("rdma", "dev2", [devs[0], "mlx5_no_such_device"])
    print("  PASS: e2e_device_filter")


# ==================================================================
if __name__ == "__main__":
    if not hasattr(radixshmem, "_data"):
        print("radixshmem._data not built — nothing to test")
        raise SystemExit(0)

    print("=== radixshmem._data binding tests ===\n")
    test_store_geometry_and_views()
    test_store_attach_shares_memory()
    test_store_pools()
    test_pools_aligned_with_index()
    test_e2e_rdma()
    test_e2e_tcp()
    test_e2e_device_filter()
    print("\nAll tests passed.")
