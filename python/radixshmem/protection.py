# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in, small protected-prefix catalog on an etcd quorum.

This catalog is deliberately separate from the approximate RHT. READY means
an owner holds a complete component query pin for the exact object and epoch.
Every destructive decision is a CAS of the object AND its live membership
witnesses. EVICTING never becomes READY merely because a timer elapsed.

Ordinary prefixes continue using RHT/LRU. This bounded catalog also serves as
an independent recovery directory for the protected set when an RHT holder
loses its soft state. Consumers still validate candidates by walking the tree.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class PrefixSpec:
    identity: str
    hashes: tuple
    mask: int = 1

    def __post_init__(self):
        object.__setattr__(self, "hashes", tuple(int(h) for h in self.hashes))
        if (not self.identity or len(self.identity) > 256 or not 0 < len(self.hashes) <= 8192
                or self.mask < 1 or self.mask > 7 or not self.mask & 1
                or any(h < 0 or h >= 2**64 for h in self.hashes)):
            raise ValueError("invalid protected prefix identity, mask or hashes")

    @property
    def key(self):
        return hashlib.sha256(encode(self.payload()).encode()).hexdigest()

    def payload(self):
        return {"identity": self.identity, "hashes": list(self.hashes), "mask": self.mask}


class Conflict(RuntimeError):
    pass


class ProtectedCatalog:
    def __init__(self, kv, cluster, replicas=2, max_objects=32):
        if replicas < 2 or not cluster or "/" in cluster or max_objects < 1:
            raise ValueError("invalid protected catalog configuration")
        self.kv, self.replicas, self.max_objects = kv, replicas, max_objects
        self.base = f"radix/{cluster}/protected/"
        self.leader_key = self.base + "leader"

    def member_key(self, node):
        if not node or "/" in node:
            raise ValueError("node must be a nonempty path component")
        return self.base + "members/" + node

    def object_key(self, spec):
        return self.base + "objects/" + spec.key

    def snapshot(self):
        # One linearizable etcd range, not independent object/member reads.
        values = dict(self.kv.get_prefix(self.base))
        members = {key.rsplit("/", 1)[-1]: json.loads(value)
                   for key, value in values.items() if key.startswith(self.base + "members/")}
        objects = {key.rsplit("/", 1)[-1]: json.loads(value)
                   for key, value in values.items() if key.startswith(self.base + "objects/")}
        return values, members, objects

    def create(self, spec):
        # Creation is serialized through one small manifest, giving a cluster
        # bound even when several operators add different objects concurrently.
        manifest = self.base + "manifest"
        for _ in range(8):
            raw = self.kv.get(manifest)
            listing = json.loads(raw) if raw else {"version": 0, "keys": []}
            if spec.key not in listing["keys"]:
                if len(listing["keys"]) >= self.max_objects:
                    raise ValueError("protected object quota exceeded")
                listing["keys"].append(spec.key)
                listing["version"] += 1
                if not self.kv.compare_put(encode({manifest: raw}), manifest, encode(listing)):
                    continue
            record = {"version": 1, "spec": spec.payload(), "replicas": self.replicas, "owners": {}}
            self.kv.create(self.object_key(spec), encode(record))
            return
        raise Conflict("protected manifest contention")

    @staticmethod
    def ready(record, members):
        return {node: owner for node, owner in record["owners"].items()
                if owner["state"] == "READY" and node in members
                and owner["epoch"] == members[node]["epoch"]
                and owner["domain"] == members[node]["domain"]}

    def mutate(self, spec, change, leader=None):
        for _ in range(8):
            values, members, objects = self.snapshot()
            record = objects[spec.key]
            conditions = {self.object_key(spec): values[self.object_key(spec)]}
            # Fence membership and object version in the same transaction.
            conditions.update({self.member_key(n): values[self.member_key(n)] for n in members})
            if leader is not None:
                conditions[self.leader_key] = leader
            result = change(record, members)
            record["version"] += 1
            if self.kv.compare_put(encode(conditions), self.object_key(spec), encode(record)):
                return result
        raise Conflict("protected object contention or leadership changed")

    def publish(self, spec, node, epoch, common_hit, component_mask):
        if common_hit != len(spec.hashes) or component_mask & spec.mask != spec.mask:
            raise ValueError("READY requires the complete joint component boundary")

        def change(record, members):
            member = members.get(node)
            if member is None or member["epoch"] != epoch:
                raise Conflict("publication from an expired node epoch")
            old = record["owners"].get(node)
            if old and old["epoch"] == epoch and old["state"] == "EVICTING":
                raise Conflict("ambiguous eviction must be reconciled, never renewed to READY")
            record["owners"][node] = {"epoch": epoch, "domain": member["domain"],
                                      "state": "READY", "rank": member["rank"]}
        self.mutate(spec, change)

    def eviction_intent(self, spec, node, epoch):
        def change(record, members):
            owner = record["owners"].get(node)
            if owner is None or owner["epoch"] != epoch:
                raise Conflict("eviction owner changed")
            if owner["state"] == "EVICTING":
                return owner["intent"]
            remaining = self.ready(record, members)
            remaining.pop(node, None)
            if len({r["domain"] for r in remaining.values()}) < record["replicas"]:
                raise Conflict("eviction would violate the fault-domain replica floor")
            owner["state"], owner["intent"] = "EVICTING", uuid.uuid4().hex
            return owner["intent"]
        return self.mutate(spec, change)

    def eviction_ack(self, spec, node, epoch, intent):
        def change(record, members):
            owner = record["owners"].get(node)
            if owner is None:
                return
            if (owner["epoch"], owner["state"], owner.get("intent")) != (epoch, "EVICTING", intent):
                raise Conflict("eviction token changed")
            del record["owners"][node]
        self.mutate(spec, change)

    def assign_repair(self, spec, leader):
        def change(record, members):
            ready = self.ready(record, members)
            domains = {r["domain"] for r in ready.values()}
            pending = {n for n, r in record["owners"].items() if r["state"] == "COPYING"
                       and n in members and r["epoch"] == members[n]["epoch"]}
            domains.update(members[n]["domain"] for n in pending)
            if len(domains) >= record["replicas"]:
                return None
            for node, member in sorted(members.items()):
                owner = record["owners"].get(node)
                if member["domain"] in domains or (owner and owner["epoch"] == member["epoch"]
                                                     and owner["state"] == "EVICTING"):
                    continue
                record["owners"][node] = {"epoch": member["epoch"], "domain": member["domain"],
                                          "rank": member["rank"], "state": "COPYING"}
                return node
        return self.mutate(spec, change, leader)

    def candidates(self, spec, exclude="", limit=2):
        _, members, objects = self.snapshot()
        record = objects.get(spec.key)
        if record is None:
            return []
        ready = self.ready(record, members)
        return [r["rank"] for n, r in sorted(ready.items()) if n != exclude][:limit]


class ProtectedAgent:
    """One node's bounded pin owner, repair worker and failover controller.

    All nodes run this loop. etcd's leased leader key selects the controller;
    CAS transactions fence the old leader. Copy jobs use the ordinary pull
    protocol, and retain both transfer and protection ownership until release.
    """

    def __init__(self, catalog, client, specs, node, domain, quota_bytes, repair_bytes_per_s):
        if quota_bytes <= 0 or repair_bytes_per_s <= 0 or not domain:
            raise ValueError("protection needs a domain and positive byte quotas")
        self.catalog, self.client, self.specs = catalog, client, list(specs)
        self.node, self.epoch = node, client.info.epoch
        self.quota_bytes, self.rate = quota_bytes, repair_bytes_per_s
        self.pins, self.jobs, self.errors = {}, {}, 0
        self._retired = set()
        self._pin_bytes = {}
        self._next_copy = 0.0
        self._stopping = False
        self._leader = encode({"node": node, "epoch": self.epoch, "controller": uuid.uuid4().hex})
        self.lease = catalog.kv.grant_lease(10)
        catalog.kv.keep_alive(self.lease)
        self.member = encode({"epoch": self.epoch, "domain": domain, "rank": client.info.rank})
        if not catalog.kv.create(catalog.member_key(node), self.member, self.lease):
            catalog.kv.revoke(self.lease)
            raise Conflict("a protection agent already owns this node")
        if sum(self.bytes_for(s) for s in specs) > quota_bytes:
            catalog.kv.revoke(self.lease)
            raise ValueError("manual protection set exceeds per-node quota")
        for spec in specs:
            catalog.create(spec)

    def bytes_for(self, spec):
        pools, size = self.client.geometry["pools"], 0
        for bit, name in ((1, "full"), (2, "swa"), (4, "mamba")):
            if not spec.mask & bit:
                continue
            if name not in pools:
                raise ValueError("protected component absent from geometry")
            count = len(spec.hashes)
            if name == "swa":
                count = min(count, pools[name]["window_blocks"])
            if name == "mamba":
                count = 1
            size += count * pools[name]["slot_bytes"]
        return size

    def _pin(self, spec):
        # Joint query both verifies complete components and pins their actual
        # node/slot coverage. Pin before publishing READY, including retries.
        if spec.key in self.pins:
            q = self.pins[spec.key]
        else:
            q = self.client.query(spec.hashes, mask=spec.mask, local_only=True, lock=True)
            if int(q.status) != 0 or int(q.common_hit) != len(spec.hashes):
                q.finalize()
                return False
            pools = self.client.geometry["pools"]
            charge = sum(int(getattr(q, field)) * pools.get(name, {}).get("slot_bytes", 0)
                         for name, field in (("full", "pinned_full_blocks"),
                                             ("swa", "pinned_swa_slots"),
                                             ("mamba", "pinned_mamba_slots")))
            if sum(self._pin_bytes.values()) + charge > self.quota_bytes:
                q.finalize()
                return False
            self.pins[spec.key] = q
            self._pin_bytes[spec.key] = charge
        self.catalog.publish(spec, self.node, self.epoch, int(q.common_hit), spec.mask)
        return True

    def tick(self):
        if self._stopping:
            return
        kv, catalog = self.catalog.kv, self.catalog
        if kv.get(catalog.member_key(self.node)) != self.member:
            # Never drop pins because a lease expired: old data-plane accesses
            # may still exist. Recovery needs a new agent/pool epoch.
            raise Conflict("protection membership lost; pins retained")
        leader = kv.get(catalog.leader_key)
        if leader is None:
            kv.create(catalog.leader_key, self._leader, self.lease)
            leader = kv.get(catalog.leader_key)
        _, _, objects = catalog.snapshot()
        for spec in self.specs:
            if leader == self._leader:
                catalog.assign_repair(spec, self._leader)
            job = self.jobs.get(spec.key)
            if job is not None:
                if not job.resources_released:
                    continue
                result = job.wait(0)
                result.finalize()
                del self.jobs[spec.key]
            record = objects[spec.key]
            owner = record["owners"].get(self.node)
            if owner and owner["epoch"] == self.epoch and owner["state"] == "EVICTING":
                # Drop only our protection reference. Ordinary query refs still
                # guard readers; LRU will reclaim after the last reference.
                pin = self.pins.pop(spec.key, None)
                if pin is not None:
                    pin.finalize()
                    self._pin_bytes.pop(spec.key, None)
                catalog.eviction_ack(spec, self.node, self.epoch, owner["intent"])
                self._retired.add(spec.key)
                continue
            assigned = owner and owner["epoch"] == self.epoch and owner["state"] == "COPYING"
            if assigned:
                self._retired.discard(spec.key)
            if spec.key not in self._retired:
                if self._pin(spec):
                    continue
                if assigned and not self.jobs and time.monotonic() >= self._next_copy:
                    ranks = catalog.candidates(spec, exclude=self.node)
                    if ranks:
                        self.jobs[spec.key] = self.client.pull_async(spec.hashes, spec.mask,
                            lock=False, block=False, timeout_ms=5000, source_ranks=ranks)
                        self._next_copy = time.monotonic() + self.bytes_for(spec) / self.rate

    def close(self, timeout_s=30):
        # Stopping a node is a membership loss, not an ordinary eviction. Mark
        # it unavailable BEFORE dropping pins; remaining live replicas repair.
        self._stopping = True
        self.client.close(timeout_s=timeout_s)
        self.catalog.kv.revoke(self.lease)
        for q in self.pins.values():
            q.finalize()
        self.pins.clear()
