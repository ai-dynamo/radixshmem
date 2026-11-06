# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The real Go/C++/Python etcd transaction path, with an ephemeral local etcd."""
import json
import shutil
import socket
import subprocess
import time

import pytest

from radixshmem import _core
from radixshmem.protection import Conflict, PrefixSpec, ProtectedCatalog, encode


def port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def kv(tmp_path):
    if not shutil.which("etcd") or not hasattr(_core, "EtcdKv"):
        pytest.skip("requires etcd and native etcd bindings")
    cp, pp = port(), port()
    with open(tmp_path / "etcd.log", "w") as log:
        p = subprocess.Popen(["etcd", "--name", "test", "--data-dir", str(tmp_path / "data"),
            "--listen-client-urls", f"http://127.0.0.1:{cp}",
            "--advertise-client-urls", f"http://127.0.0.1:{cp}",
            "--listen-peer-urls", f"http://127.0.0.1:{pp}",
            "--initial-advertise-peer-urls", f"http://127.0.0.1:{pp}",
            "--initial-cluster", f"test=http://127.0.0.1:{pp}"], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 20
            while True:
                try:
                    connection = _core.EtcdKv(f"127.0.0.1:{cp}")
                    connection.get("health")
                    break
                except Exception:
                    if p.poll() is not None or time.monotonic() > deadline:
                        raise
                    time.sleep(0.1)
            yield connection
        finally:
            p.terminate()
            p.wait(timeout=10)


def test_real_etcd_transaction_fences_old_controller_and_persists_intent(kv):
    catalog = ProtectedCatalog(kv, "transaction-test")
    spec = PrefixSpec("model/layout/namespace", (1, 2))
    catalog.create(spec)
    leases = []
    for rank in range(3):
        lease = kv.grant_lease(30)
        leases.append(lease)
        kv.put(catalog.member_key(str(rank)), encode(dict(epoch="e1", rank=rank, domain=str(rank))), lease)
        catalog.publish(spec, str(rank), "e1", 2, 1)
    token = catalog.eviction_intent(spec, "0", "e1")
    restarted = ProtectedCatalog(kv, "transaction-test")
    assert restarted.eviction_intent(spec, "0", "e1") == token
    with pytest.raises(Conflict):
        restarted.eviction_intent(spec, "1", "e1")
    kv.put(catalog.leader_key, "new-leader")
    with pytest.raises(Conflict):
        catalog.assign_repair(spec, "old-leader")
    assert not kv.compare_put(json.dumps({catalog.leader_key: "old-leader"}), "fenced", "bad")
    assert kv.get("fenced") is None
    kv.revoke(leases[1])
    assert restarted.candidates(spec) == [2]
    catalog.eviction_ack(spec, "0", "e1", token)
    assert json.loads(kv.get(catalog.object_key(spec)))["owners"].get("0") is None
