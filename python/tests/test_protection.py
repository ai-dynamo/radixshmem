# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concurrent control-plane fault injection, without a native/RDMA requirement."""
import importlib.util
import json
from pathlib import Path
import sys
import threading

import pytest

spec = importlib.util.spec_from_file_location(
    "_protection_under_test", Path(__file__).parents[1] / "radixshmem" / "protection.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


class MemoryKv:
    """Atomic fake etcd, shared by independent catalog/controller instances."""
    def __init__(self):
        self.data = {}
        self.mu = threading.RLock()
        self.partition = False

    def get(self, key):
        with self.mu:
            if self.partition:
                raise TimeoutError("etcd partition")
            return self.data.get(key)

    def get_prefix(self, prefix):
        with self.mu:
            if self.partition:
                raise TimeoutError("etcd partition")
            return sorted((k, v) for k, v in self.data.items() if k.startswith(prefix))

    def create(self, key, value, lease=0):
        return self.compare_put(json.dumps({key: None}), key, value, lease)

    def compare_put(self, conditions, key, value, lease=0):
        with self.mu:
            if self.partition:
                raise TimeoutError("etcd partition")
            if not all(self.data.get(k) == v for k, v in json.loads(conditions).items()):
                return False
            self.data[key] = value
            return True


@pytest.fixture
def catalog():
    kv = MemoryKv()
    c = m.ProtectedCatalog(kv, "test")
    s = m.PrefixSpec("weights-sha/layout/namespace", (101, 102), mask=3)
    c.create(s)
    for node, domain in (("n1", "rack1"), ("n2", "rack2"), ("n3", "rack3")):
        kv.create(c.member_key(node), m.encode(dict(epoch="e1", domain=domain, rank=int(node[1]))))
        c.publish(s, node, "e1", 2, 3)
    return c, s


def test_concurrent_evictions_cannot_delete_both_required_replicas(catalog):
    c, s = catalog
    barrier, granted = threading.Barrier(2), []

    def evict(node):
        other = m.ProtectedCatalog(c.kv, "test")
        barrier.wait()
        try:
            granted.append((node, other.eviction_intent(s, node, "e1")))
        except m.Conflict:
            pass

    threads = [threading.Thread(target=evict, args=(n,)) for n in ("n1", "n2")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(granted) == 1
    _, members, objects = c.snapshot()
    assert len(c.ready(objects[s.key], members)) == 2


def test_intent_ack_loss_does_not_restore_replica_to_ready(catalog):
    c, s = catalog
    token = c.eviction_intent(s, "n1", "e1")
    assert c.eviction_intent(s, "n1", "e1") == token
    with pytest.raises(m.Conflict, match="ambiguous"):
        c.publish(s, "n1", "e1", 2, 3)
    c.eviction_ack(s, "n1", "e1", token)
    c.eviction_ack(s, "n1", "e1", token)
    assert c.candidates(s) == [2, 3]


def test_partial_components_and_stale_epoch_never_qualify(catalog):
    c, s = catalog
    with pytest.raises(ValueError, match="joint"):
        c.publish(s, "n1", "e1", 2, 1)
    with pytest.raises(ValueError):
        c.publish(s, "n1", "e1", 1, 3)
    c.kv.data[c.member_key("n1")] = m.encode(dict(epoch="e2", domain="rack1", rank=1))
    assert c.candidates(s) == [2, 3]
    with pytest.raises(m.Conflict):
        c.publish(s, "n1", "e1", 2, 3)


def test_partition_blocks_eviction_and_preserves_durable_intent(catalog):
    c, s = catalog
    token = c.eviction_intent(s, "n1", "e1")
    c.kv.partition = True
    with pytest.raises(TimeoutError):
        c.eviction_intent(s, "n2", "e1")
    c.kv.partition = False
    restarted = m.ProtectedCatalog(c.kv, "test")
    assert restarted.eviction_intent(s, "n1", "e1") == token
    assert restarted.candidates(s) == [2, 3]


def test_old_controller_is_fenced_in_same_transaction(catalog):
    c, s = catalog
    del c.kv.data[c.member_key("n2")]
    del c.kv.data[c.member_key("n3")]
    c.kv.data[c.member_key("n4")] = m.encode(dict(epoch="new", domain="rack4", rank=4))
    c.kv.data[c.leader_key] = "new-leader"
    with pytest.raises(m.Conflict):
        c.assign_repair(s, "old-leader")
    assert c.assign_repair(s, "new-leader") == "n4"
    _, members, objects = c.snapshot()
    assert len(c.ready(objects[s.key], members)) == 1  # COPYING is not a replica
    c.publish(s, "n4", "new", 2, 3)
    assert c.candidates(s) == [1, 4]


def test_same_fault_domain_does_not_count_twice(catalog):
    c, s = catalog
    c.kv.data[c.member_key("n2")] = m.encode(dict(epoch="e2", domain="rack1", rank=2))
    c.publish(s, "n2", "e2", 2, 3)
    with pytest.raises(m.Conflict, match="fault-domain"):
        c.eviction_intent(s, "n3", "e1")


def test_manual_manifest_is_bounded_and_identity_is_part_of_key():
    c = m.ProtectedCatalog(MemoryKv(), "test", max_objects=1)
    a, b = m.PrefixSpec("model-a", (1,)), m.PrefixSpec("model-b", (1,))
    assert a.key != b.key
    c.create(a)
    c.create(a)
    with pytest.raises(ValueError, match="quota"):
        c.create(b)
