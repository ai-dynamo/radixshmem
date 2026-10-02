# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python binding smoke tests.

The full functional coverage lives in the C++ unit tests (test_tree_*,
test_cluster, etc.). These four tests only assert the binding surface:
   - numpy dtypes thread through correctly,
   - keyword arguments still map to the right C++ parameters,
   - returned objects expose the expected attributes,
   - FinalizeFn is callable and tracks usage state.
"""

import numpy as np
import shmradix


def insert_auto(client, hashes):
    """Allocate the shortfall past the matched prefix and insert. Slots bind to
    hashes[matched:] via start=matched; insert auto-recycles anything unused."""
    matched = int(client.query(hashes, mask=shmradix.COMPONENT_MASK_FULL,
                               update_meta=False).common_hit)
    need = len(hashes) - matched
    slots = client.allocate_slots(need) if need > 0 \
        else np.empty(0, dtype=np.int32)
    return client.insert(hashes, slots, start=matched, auto_recycle=True)


def assert_healthy(client):
    """End-of-test leak audit: no lock held, mempool == tree slots, no pins."""
    rep = client.health_report()
    assert client.check_health() == 0, rep.describe()
    assert rep.ok() and bool(rep)


def test_compute_hashes_returns_numpy():
    """token ids → uint64 numpy hash array (binding surface)."""
    tokens = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    h1 = shmradix.compute_hashes(tokens, 1)
    h2 = shmradix._core.RadixClient.compute_hashes(tokens, 1)
    assert isinstance(h1, np.ndarray)
    assert h1.dtype == np.uint64
    assert len(h1) == 5
    np.testing.assert_array_equal(h1, h2)
    print("  PASS: compute_hashes_returns_numpy")


def test_config_repr():
    cfg = shmradix._core.ShmConfig(max_nodes=100, max_blocks=50)
    s = repr(cfg)
    assert "100" in s
    assert "50" in s
    assert "component_mask=1" in s and "swa_window_blocks=0" in s
    assert cfg.effective_data_pool_bytes() == 50 * 12 * 4   # Full-only: 12 B/block
    assert hasattr(shmradix.InsertError, "REMOTE_FAILED")
    assert shmradix.INVALID_COMPONENT_SLOT == -1
    print("  PASS: config_repr")


def test_insert_query_smoke():
    """End-to-end insert + query through pybind."""
    cfg = shmradix._core.ShmConfig(max_nodes=200000, max_blocks=10000)
    server = shmradix._core.RadixServer("/py_smoke_iq", cfg)
    client = shmradix._core.RadixClient("/py_smoke_iq")

    tokens = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    hashes = shmradix.compute_hashes(tokens, 1)
    ins = insert_auto(client, hashes)
    assert ins.error == shmradix.InsertError.OK
    assert isinstance(ins.unused_slots, np.ndarray)
    assert ins.unused_slots.dtype == np.int32
    assert len(ins.unused_slots) == 0   # fresh insert: all 5 slots landed

    qr = client.query(hashes, mask=shmradix.COMPONENT_MASK_FULL)
    assert qr.common_hit == 5
    assert len(qr.full_fragments) == 1 and len(qr.full_fragments[0][2]) == 5
    assert_healthy(client)
    print("  PASS: insert_query_smoke")


def test_finalize_callable():
    """Query FinalizeFn exposed as callable + __bool__."""
    cfg = shmradix._core.ShmConfig(max_nodes=200000, max_blocks=10000)
    server = shmradix._core.RadixServer("/py_smoke_fin", cfg)
    client = shmradix._core.RadixClient("/py_smoke_fin")

    tokens = np.array([1, 2, 3], dtype=np.int64)
    hashes = shmradix.compute_hashes(tokens, 1)
    insert_auto(client, hashes)

    # lock=True query → finalize is armed (dec_ref on finalize).
    qr = client.query(hashes, mask=shmradix.COMPONENT_MASK_FULL, lock=True)
    assert bool(qr.finalize)
    qr.finalize()
    assert not bool(qr.finalize)
    # Idempotent.
    qr.finalize()

    # lock=False query → finalize is empty no-op.
    qr2 = client.query(hashes, mask=shmradix.COMPONENT_MASK_FULL)
    assert not bool(qr2.finalize)
    qr2.finalize()
    print("  PASS: finalize_callable")


def test_component_query_smoke():
    """Component binding surface: masks, kwargs, result fields, finalize."""
    assert shmradix.COMPONENT_MASK_FULL == 1
    assert shmradix.COMPONENT_MASK_SWA == 2
    assert shmradix.COMPONENT_MASK_MAMBA == 4
    assert hasattr(shmradix.ComponentType, "FULL")

    cfg = shmradix._core.ShmConfig(max_nodes=200000, max_blocks=10000)
    server = shmradix._core.RadixServer("/py_smoke_comp", cfg)
    client = shmradix._core.RadixClient("/py_smoke_comp")

    tokens = np.array([7, 8, 9, 10], dtype=np.int64)
    hashes = shmradix.compute_hashes(tokens, 1)
    insert_auto(client, hashes)

    # ComponentType passed as mask must be rejected outright (pitfall 9).
    try:
        client.query(hashes, mask=shmradix.ComponentType.FULL,
                                local_only=True)
        assert False, "ComponentType accepted as mask"
    except TypeError:
        pass

    # local_only defaults to False (like legacy query); a standalone
    # client serves it from the local source (no remote exists).
    r = client.query(hashes, mask=shmradix.COMPONENT_MASK_FULL)
    assert r.status == shmradix.RemoteStatus.OK and r.common_hit == 4

    # Full-mask hit: source-tagged runs, field shapes and dtypes.
    r = client.query(hashes, mask=shmradix.COMPONENT_MASK_FULL,
                                local_only=True, update_meta=False)
    assert r.status == shmradix.RemoteStatus.OK
    assert r.common_hit == 4
    assert len(r.full_fragments) == 1
    rank, start, slots = r.full_fragments[0]
    assert rank == 0 and start == 0
    assert isinstance(slots, np.ndarray)
    assert slots.dtype == np.int32 and len(slots) == 4
    assert r.swa_source_rank == -1
    assert len(r.swa_slots) == 0 and r.mamba is None
    assert not bool(r.finalize)          # lock=False: nothing armed

    # Miss: OK + empty runs + rank sentinels; finalize no-op.
    miss = shmradix.compute_hashes(
        np.array([99, 98, 97], dtype=np.int64), 1)
    rm = client.query(miss, mask=shmradix.COMPONENT_MASK_FULL,
                                 local_only=True, lock=True)
    assert rm.status == shmradix.RemoteStatus.OK and rm.common_hit == 0
    assert len(rm.full_fragments) == 0 and rm.swa_source_rank == -1
    assert not bool(rm.finalize)
    rm.finalize()

    # lock=True hit arms the one-shot finalize; __bool__ flips exactly once.
    rl = client.query(hashes, mask=shmradix.COMPONENT_MASK_FULL,
                                 local_only=True, lock=True)
    assert bool(rl.finalize)
    rl.finalize()
    assert not bool(rl.finalize)
    rl.finalize()                        # idempotent no-op
    assert_healthy(client)
    print("  PASS: component_query_smoke")


def test_component_insert_smoke():
    all_components = (shmradix.COMPONENT_MASK_FULL |
                      shmradix.COMPONENT_MASK_SWA |
                      shmradix.COMPONENT_MASK_MAMBA)
    cfg = shmradix._core.ShmConfig(max_nodes=1000, max_blocks=1000,
                             component_mask=all_components, swa_window_blocks=4,
                             mamba_state_bytes=256, swa_max_blocks=1000,
                             mamba_max_slots=16)
    server = shmradix._core.RadixServer("/py_smoke_pub", cfg)
    client = shmradix._core.RadixClient("/py_smoke_pub")

    tokens = np.array([11, 12, 13, 14, 15, 16, 17, 18], dtype=np.int64)
    hashes = shmradix.compute_hashes(tokens, 1)
    full = client.allocate_slots(len(hashes))
    r = client.insert(hashes, full, component=shmradix.ComponentType.FULL)
    assert r.error == shmradix.InsertError.OK

    swa = client.allocate_slots(4, component=shmradix.ComponentType.SWA)
    assert len(swa) == 4
    # Allocated but not yet published: the audit must flag a SWA leak here.
    assert client.check_health() & int(shmradix.HealthFlag.SWA_SLOT_LEAK)
    for bad in (0, 1001):
        try:
            client.allocate_slots(bad, component=shmradix.ComponentType.SWA)
            assert False, "allocate_slots accepted a bad count"
        except ValueError:
            pass
    assert len(client.allocate_slots(997, component=shmradix.ComponentType.SWA)) == 0
    # SWA is positional: these 4 slots cover the trailing positions [4, 8).
    r = client.insert(hashes, swa, component=shmradix.ComponentType.SWA)
    assert r.error == shmradix.InsertError.OK and len(r.unused_slots) == 0
    # more slots than positions is a shape error; a non-zero start is FULL-only
    bad = client.allocate_slots(4, component=shmradix.ComponentType.SWA)
    rb = client.insert(hashes[:3], bad, component=shmradix.ComponentType.SWA)
    assert rb.error == shmradix.InsertError.BAD_REQUEST and len(rb.unused_slots) == 4
    bad = client.allocate_slots(4, component=shmradix.ComponentType.SWA)
    rb = client.insert(hashes, bad, start=4, component=shmradix.ComponentType.SWA)
    assert rb.error == shmradix.InsertError.BAD_REQUEST and len(rb.unused_slots) == 4
    # The head of the same path, published later and separately (4 slots on
    # the 4-long prefix): a window shorter than W hits when fully live.
    head = client.allocate_slots(4, component=shmradix.ComponentType.SWA)
    rh = client.insert(hashes[:4], head, component=shmradix.ComponentType.SWA)
    assert rh.error == shmradix.InsertError.OK and len(rh.unused_slots) == 0
    qh = client.query(hashes[:4], mask=shmradix.COMPONENT_MASK_SWA, local_only=True,
                      update_meta=False)
    assert qh.common_hit == 4 and list(qh.swa_slots) == list(head)

    mam = client.allocate_slots(1, component=shmradix.ComponentType.MAMBA)
    r = client.insert(hashes, mam, component=shmradix.ComponentType.MAMBA)
    assert r.error == shmradix.InsertError.OK

    q = client.query(hashes, mask=all_components, local_only=True,
                     update_meta=False)
    assert q.status == shmradix.RemoteStatus.OK and q.common_hit == 8
    assert list(q.swa_slots) == list(swa)
    assert q.mamba is not None and q.mamba[1] == mam[0]

    m2 = client.allocate_slots(1, component=shmradix.ComponentType.MAMBA)
    r2 = client.insert(hashes, m2, component=shmradix.ComponentType.MAMBA)
    assert r2.error == shmradix.InsertError.OK and len(r2.unused_slots) == 1

    # Decode-style growth: one SWA position per publish on a fresh path.
    tokens2 = np.array([21, 22, 23, 24, 25, 26, 27, 28], dtype=np.int64)
    hashes2 = shmradix.compute_hashes(tokens2, 1)
    full2 = client.allocate_slots(len(hashes2))
    assert client.insert(hashes2, full2).error == shmradix.InsertError.OK
    ones = []
    for n in range(1, len(hashes2) + 1):
        s1 = client.allocate_slots(1, component=shmradix.ComponentType.SWA)
        r1 = client.insert(hashes2[:n], s1, component=shmradix.ComponentType.SWA)
        assert r1.error == shmradix.InsertError.OK and len(r1.unused_slots) == 0
        ones.append(int(s1[0]))
    q2 = client.query(hashes2, mask=all_components, local_only=True,
                      update_meta=False)
    assert q2.common_hit == 0   # no mamba checkpoint on this path
    q2 = client.query(hashes2, mask=shmradix.COMPONENT_MASK_FULL |
                      shmradix.COMPONENT_MASK_SWA, local_only=True, update_meta=False)
    assert q2.common_hit == 8 and list(q2.swa_slots) == ones[4:]
    assert_healthy(client)   # every SWA/Mamba slot landed or was auto-recycled
    print("  PASS: component_insert_smoke")




def test_health_check_smoke():
    """check_health / health_report binding surface: flags, report fields,
    and the two leak shapes a test teardown is meant to catch."""
    cfg = shmradix._core.ShmConfig(max_nodes=1000, max_blocks=1000)
    server = shmradix._core.RadixServer("/py_smoke_health", cfg)
    client = shmradix._core.RadixClient("/py_smoke_health")
    assert client.check_health() == 0 and server.check_health() == 0

    hashes = shmradix.compute_hashes(np.arange(1, 9, dtype=np.int64), 1)
    insert_auto(client, hashes)
    assert_healthy(client)

    # allocate without insert/recycle -> FULL slot leak
    stray = client.allocate_slots(3)
    rep = client.health_report()
    assert rep.flags & int(shmradix.HealthFlag.FULL_SLOT_LEAK)
    assert rep.full.leaked == 3 and rep.full.used == rep.full.in_tree + 3
    assert not rep.ok() and "full slots" in rep.describe()
    client.recycle_slots(stray)
    assert_healthy(client)

    # lock=True query without finalize -> pinned nodes (ref leak)
    q = client.query(hashes, mask=shmradix.COMPONENT_MASK_FULL, lock=True)
    rep = client.health_report()
    assert rep.flags & int(shmradix.HealthFlag.REF_LEAK)
    assert rep.nodes_with_refs >= 1 and rep.ref_total >= 1
    q.finalize()
    assert_healthy(client)
    assert rep.locks_inspected and not rep.lru_locked and rep.rdma_rwlock == 0
    assert rep.nodes_reachable == rep.nodes_in_pool == client.total_radix_nodes()
    print("  PASS: health_check_smoke")


if __name__ == "__main__":
    print("=== shmradix Python binding smoke tests ===\n")
    test_compute_hashes_returns_numpy()
    test_config_repr()
    test_insert_query_smoke()
    test_finalize_callable()
    test_component_query_smoke()
    test_component_insert_smoke()
    test_health_check_smoke()
    print("\nAll tests passed.")
