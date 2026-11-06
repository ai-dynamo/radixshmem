# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, epoch-scoped transfer ownership. No timeout implies DMA completion.

Callers serialize access with the server's jobs lock. Session high-water marks
are retained for the entire server epoch: a retired operation can never be
submitted again, even after its result and reservation have been collected.
Only RESERVED operations expire. SUBMITTED operations require physical
completion followed by an explicit consumer ACK.
"""

import time
from dataclasses import dataclass


@dataclass
class Operation:
    session: str
    sequence: int
    nbytes: int
    created: float
    expires: float
    state: str = "RESERVED"
    fingerprint: str = ""
    ticket: int = -1
    key: str = ""
    leader: tuple = ()


class Operations:
    def __init__(self, epoch, max_jobs, max_bytes, max_sessions=4096, reservation_s=10):
        if min(max_jobs, max_bytes, max_sessions, reservation_s) <= 0:
            raise ValueError("operation limits must be positive")
        self.epoch = epoch
        self.max_jobs, self.max_bytes = max_jobs, max_bytes
        self.max_sessions, self.reservation_s = max_sessions, reservation_s
        self.sessions, self.active = {}, {}
        self.flights = {}
        self.reserved_bytes = 0
        self.refused = 0
        self.accepting = True

    def _check(self, epoch, session, sequence):
        if epoch != self.epoch:
            return "EPOCH_MISMATCH"
        if not isinstance(session, str) or not 1 <= len(session) <= 128 or int(sequence) < 1:
            raise ValueError("invalid session or sequence")
        return ""

    def expire(self):
        now = time.monotonic()
        for key, op in list(self.active.items()):
            if op.state == "RESERVED" and now >= op.expires:
                self.retire(*key)

    def lookup(self, epoch, session, sequence):
        error = self._check(epoch, session, sequence)
        if error:
            return error, None
        self.expire()
        op = self.active.get((session, sequence))
        if op is not None:
            if op.leader and op.leader not in self.active:
                self.retire(session, sequence)
                return "RETIRED", None
            return op.state, op
        # Never use UNKNOWN to authorize recycling after Submit.
        return ("RETIRED" if sequence <= self.sessions.get(session, 0) else "UNKNOWN"), None

    def reserve(self, epoch, session, sequence, nbytes, key=""):
        state, op = self.lookup(epoch, session, sequence)
        if state != "UNKNOWN":
            if op is not None and (op.nbytes != nbytes or op.key != key):
                raise ValueError("operation reservation changed on retry")
            return state, op
        if not isinstance(nbytes, int) or nbytes < 0:
            raise ValueError("invalid reservation size")
        if not isinstance(key, str) or len(key) > 128:
            raise ValueError("invalid flight key")
        if session not in self.sessions and len(self.sessions) >= self.max_sessions:
            self.refused += 1
            return "BUSY", None
        # Advance even on refusal, making delayed retries unable to acquire
        # resources after the consumer already fell back to its local hit.
        self.sessions[session] = sequence
        leader = self.flights.get(key, ()) if key else ()
        if (not self.accepting or len(self.active) >= self.max_jobs
                or self.reserved_bytes + (0 if leader else nbytes) > self.max_bytes):
            self.refused += 1
            return "RETIRED", None
        now = time.monotonic()
        op = Operation(session, sequence, nbytes, now, now + self.reservation_s)
        op.key, op.leader = key, leader
        if leader:
            op.state = "FOLLOWING"
        elif key:
            self.flights[key] = (session, sequence)
        self.active[session, sequence] = op
        self.reserved_bytes += 0 if leader else nbytes
        return op.state, op

    def retire(self, session, sequence):
        op = self.active.pop((session, sequence), None)
        if op is not None:
            self.reserved_bytes -= 0 if op.leader else op.nbytes
            if op.key and self.flights.get(op.key) == (session, sequence):
                del self.flights[op.key]
        return op

    def stats(self):
        self.expire()
        now = time.monotonic()
        return {"reserved_bytes": self.reserved_bytes,
                "reservations": len(self.active), "sessions": len(self.sessions),
                "admission_refused": self.refused,
                "oldest_operation_s": max((now - o.created for o in self.active.values()), default=0)}
