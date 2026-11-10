# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Node placement stream: the blocks a node's tree stores and drops, published
from its KV-event ring (one stream per node).

Wire (ZMQ PUB, multipart): ``[topic, seq, payload]``

* ``topic`` -- configured bytes, ``b""`` by default.
* ``seq`` -- 8-byte big-endian batch number, +1 per batch from 0. A gap means
  batches were lost (PUB drops under backpressure, or a slow joiner).
* ``payload`` -- msgpack map::

      {"v": 1, "cluster_id": str, "node": int, "node_name": str,
       "incarnation": str, "hash_scheme": str, "block_size": int,
       "events": [["stored" | "removed", [hash, ...]] | ["cleared", []], ...]}

  ``node`` is the node's cluster rank (0 standalone), ``incarnation`` changes
  every time the publisher starts, hashes are signed 64-bit (the same bits the
  clients inserted). A batch with no events is a heartbeat.

A consumer drops what it knows about a node on a seq gap, a new incarnation or
``cleared`` (also sent when the ring lost records), then relearns from later events.
"""

from __future__ import annotations

import logging
import struct
import threading
import uuid
from typing import Any, List

import numpy as np

log = logging.getLogger("radixshmem.kv_events")

KVEVENT_REMOVED, KVEVENT_STORED, KVEVENT_CLEARED = 0, 1, 2
_KIND_NAMES = {KVEVENT_REMOVED: "removed", KVEVENT_STORED: "stored", KVEVENT_CLEARED: "cleared"}
WIRE_VERSION = 1
# Hashes per published batch; the remainder goes out on the next tick.
MAX_HASHES_PER_BATCH = 65536


def records_to_events(records, overflow: bool) -> List[List[Any]]:
    """Ring records -> wire events. A lost record poisons everything before it,
    so overflow leads with ``cleared``."""
    events: List[List[Any]] = [["cleared", []]] if overflow else []
    for kind, hashes in records:
        name = _KIND_NAMES.get(int(kind))
        if name is None:
            continue
        signed = np.asarray(hashes, dtype=np.uint64).view(np.int64)
        events.append([name, signed.tolist()])
    return events


class KvEventPublisher:
    """Drains ``client``'s KV-event ring every ``interval_s`` and publishes
    each non-empty drain as one batch; idle ticks send a heartbeat every
    ``heartbeat_s`` so a new subscriber learns the incarnation and seq."""

    def __init__(self, client, endpoint: str, *, cluster_id: str, node: int, node_name: str,
                 hash_scheme: str, block_size: int, topic: bytes = b"",
                 interval_s: float = 0.01, heartbeat_s: float = 1.0):
        import msgpack  # noqa: F401 - fail at start-up, not on the first batch
        import zmq

        if not client.kvevent_enabled():
            raise ValueError("the region has no KV-event ring; start radix-server with "
                             "--kvevent-ring-capacity > 0")
        self._client = client
        self._header = {"v": WIRE_VERSION, "cluster_id": cluster_id, "node": int(node),
                        "node_name": node_name, "incarnation": uuid.uuid4().hex,
                        "hash_scheme": hash_scheme, "block_size": int(block_size)}
        self._topic = topic
        self._interval_s = interval_s
        self._heartbeat_s = heartbeat_s
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, 100000)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(endpoint)
        self.endpoint = self._sock.getsockopt_string(zmq.LAST_ENDPOINT)
        self._seq = 0
        self._stop = threading.Event()
        client.kvevent_seek_to_head()
        self._thread = threading.Thread(target=self._run, name="radixshmem-kv-events", daemon=True)
        self._thread.start()
        log.info("publishing node placement on %s (incarnation %s)", self.endpoint,
                 self._header["incarnation"])

    @property
    def incarnation(self) -> str:
        return self._header["incarnation"]

    def _send(self, events) -> None:
        import msgpack

        payload = dict(self._header)
        payload["events"] = events
        self._sock.send_multipart([self._topic, struct.pack(">Q", self._seq),
                                   msgpack.packb(payload, use_bin_type=True)])
        self._seq += 1

    def publish_once(self) -> int:
        """One drain -> at most one batch. Returns the number of hashes sent."""
        records, overflow = self._client.kvevent_drain(MAX_HASHES_PER_BATCH)
        if not records and not overflow:
            return 0
        if overflow:
            log.warning("KV-event ring lapped the publisher; announcing cleared")
        events = records_to_events(records, overflow)
        self._send(events)
        return sum(len(e[1]) for e in events)

    def _run(self) -> None:
        since_send = 0.0
        while not self._stop.wait(self._interval_s):
            try:
                if self.publish_once():
                    since_send = 0.0
                    continue
                since_send += self._interval_s
                if since_send >= self._heartbeat_s:
                    self._send([])
                    since_send = 0.0
            except Exception:  # noqa: BLE001 - the stream must outlive one bad drain
                log.exception("KV-event publish failed")

    def close(self) -> None:
        self._stop.set()
        self._thread.join(2)
        self._sock.close(0)


__all__ = ["KvEventPublisher", "records_to_events", "KVEVENT_REMOVED", "KVEVENT_STORED",
           "KVEVENT_CLEARED", "WIRE_VERSION"]
