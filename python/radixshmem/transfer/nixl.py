# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL (UCX backend) as a TransferBackend, through NIXL's Python package: ``pip install radixshmem[nixl]``.

The agent is named by its segment, ``host:port``, and listens on that port for metadata. NIXL's metadata does
not carry the peer's registered range, so open_segment asks the peer for it over NIXL notifications:
``BUFREQ`` is answered with ``BUFRESP:<base>:<len>``. The metadata exchange is asynchronous with no completion
signal, so open_segment resends it on a timer until the answer arrives or 30 s pass.
"""

from __future__ import annotations

import importlib
import importlib.util
import socket
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from radixshmem import _core
from radixshmem._core import (INVALID_BATCH, XFER_EBACKEND, XFER_EBUSY, XFER_EGEOMETRY, XFER_EINVAL, XFER_ENOENT,
                              XFER_OK, SegmentInfo, TransferConfig, XferOp, XferState)
from radixshmem.transfer.base import TransferBackend, as_u64

# nixl-cu13 / nixl-cu12 install nixl_cu13 / nixl_cu12; the nixl meta package forwards to one of them.
_MODULES = ("nixl_cu13", "nixl_cu12", "nixl")
_OPEN_TIMEOUT_S = 30.0
_BUFREQ_RETRY_S = 0.2
_MD_RETRY_S = 1.0
_NOTIF_POLL_S = 0.005
BUFREQ = b"BUFREQ"
BUFRESP = b"BUFRESP:"


def available() -> bool:
    return any(importlib.util.find_spec(m) is not None for m in _MODULES)


def _api():
    for m in _MODULES:
        if importlib.util.find_spec(m) is not None:
            return importlib.import_module(f"{m}._api")
    raise ImportError("the nixl transfer backend needs NIXL's Python package: pip install radixshmem[nixl]")


def _warn(msg: str) -> None:
    print(f"radixshmem: NixlBackend: {msg}", file=sys.stderr)


def _split(segment: str) -> Optional[Tuple[str, int]]:
    host, sep, port = segment.rpartition(":")
    if not sep or not host or not port.isdigit() or not 0 < int(port) < 65536:
        return None
    return host, int(port)


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _merge(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for addr, length in sorted(r for r in ranges if r[1]):
        if out and addr <= out[-1][0] + out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][0] + out[-1][1], addr + length) - out[-1][0])
        else:
            out.append((addr, length))
    return out


def _descs(addrs: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """NIXL's (addr, length, device id) rows."""
    d = np.zeros((len(addrs), 3), dtype=np.uint64)
    d[:, 0] = addrs
    d[:, 1] = lengths
    return d


@dataclass
class _Segment:
    name: str
    info: SegmentInfo
    is_self: bool


@dataclass
class _Batch:
    handle: object = None     # nixl_xfer_handle; None for a same-process copy
    nbytes: int = 0
    done: bool = False
    failed: bool = False


class NixlBackend(TransferBackend):
    def __init__(self, cfg: TransferConfig):
        api = _api()
        segment = cfg.local_segment
        if not segment:
            if not cfg.ip_or_host:
                raise ValueError("no local_segment and no ip_or_host: no address to publish")
            port = cfg.rpc_port or _free_port(cfg.ip_or_host)
            segment = f"{cfg.ip_or_host}:{port}"
        hp = _split(segment)
        if hp is None:
            raise ValueError(f"local segment {segment!r} is not host:port")
        self._segment = segment
        self._publish = cfg.publish
        self._agent = api.nixl_agent(segment, api.nixl_agent_config(
            enable_prog_thread=False, enable_listen_thread=True, listen_port=hp[1], backends=[]))
        params = {"device_list": ",".join(cfg.device_filter)} if cfg.device_filter else {}
        self._agent.create_backend("UCX", params)

        self._mu = threading.Lock()
        self._registered: Dict[int, Tuple[int, object]] = {}   # addr -> (length, nixlRegDList)
        self._merged: List[Tuple[int, int]] = []
        self._segments: Dict[int, _Segment] = {}
        self._by_name: Dict[str, int] = {}
        self._batches: Dict[int, _Batch] = {}
        self._next_segment = self._next_batch = 1

        self._notif_cv = threading.Condition()
        self._peer_buf: Dict[str, Tuple[int, int]] = {}
        self._stop = threading.Event()
        self._notif_thread = threading.Thread(target=self._notif_loop, name="radixshmem-nixl-notif", daemon=True)
        self._notif_thread.start()

    @property
    def name(self) -> str:
        return "nixl"

    @property
    def local_segment(self) -> str:
        return self._segment

    # ---------------------------------------------------------------- registration

    def register_memory(self, addr: int, length: int, location: str = "") -> int:
        if not addr or not length:
            return XFER_EINVAL
        if self._agent is None:
            return XFER_EBACKEND
        with self._mu:
            if addr in self._registered:
                return XFER_EINVAL
        try:
            descs = self._agent.register_memory([(addr, length, 0, "")], "DRAM")
        except Exception as e:  # noqa: BLE001
            _warn(f"register_memory({addr:#x}, {length}) failed: {e}")
            return XFER_EBACKEND
        with self._mu:
            self._registered[addr] = (length, descs)
            self._merged = _merge([(a, n) for a, (n, _) in self._registered.items()])
        return XFER_OK

    def unregister_memory(self, addr: int) -> int:
        with self._mu:
            if self._agent is None:
                return XFER_EBACKEND
            entry = self._registered.get(addr)
            if entry is None:
                return XFER_ENOENT
            try:
                self._agent.deregister_memory(entry[1])
            except Exception as e:  # noqa: BLE001
                _warn(f"deregister_memory({addr:#x}) failed: {e}")
                return XFER_EBACKEND
            del self._registered[addr]
            self._merged = _merge([(a, n) for a, (n, _) in self._registered.items()])
        return XFER_OK

    def _own_region(self) -> Optional[Tuple[int, int]]:
        with self._mu:
            return self._merged[0] if len(self._merged) == 1 else None

    # ---------------------------------------------------------------- segments

    def open_segment(self, name: str) -> Tuple[int, Optional[SegmentInfo]]:
        if not name:
            return XFER_EINVAL, None
        if self._agent is None:
            return XFER_EBACKEND, None
        with self._mu:
            sid = self._by_name.get(name)
            if sid is not None:
                return XFER_OK, self._segments[sid].info
        if name == self._segment:
            # Our own segment: same-process copies, no NIXL round trip.
            own = self._own_region()
            if own is None:
                _warn("own segment is not one contiguous registered region")
                return XFER_EGEOMETRY, None
            return XFER_OK, self._add_segment(name, own, True)
        hp = _split(name)
        if hp is None:
            _warn(f"peer segment {name!r} is not host:port")
            return XFER_EINVAL, None
        with self._notif_cv:
            self._peer_buf.pop(name, None)
        deadline = time.monotonic() + _OPEN_TIMEOUT_S
        next_md = next_req = time.monotonic()
        while time.monotonic() < deadline:
            now = time.monotonic()
            try:
                if now >= next_md:
                    self._agent.send_local_metadata(hp[0], hp[1])
                    self._agent.fetch_remote_metadata(name, hp[0], hp[1])
                    next_md = now + _MD_RETRY_S
                if now >= next_req:
                    if self._agent.check_remote_metadata(name):
                        self._agent.send_notif(name, BUFREQ)
                    next_req = now + _BUFREQ_RETRY_S
            except Exception as e:  # noqa: BLE001
                _warn(f"metadata exchange with {name!r}: {e}")
            with self._notif_cv:
                self._notif_cv.wait_for(lambda: name in self._peer_buf, max(0.0, next_req - time.monotonic()))
                region = self._peer_buf.pop(name, None)
            if region is not None:
                return XFER_OK, self._add_segment(name, region, False)
        _warn(f"open_segment({name!r}) timed out waiting for the peer's buffer after {_OPEN_TIMEOUT_S:.0f} s")
        return XFER_ENOENT, None

    def _add_segment(self, name: str, region: Tuple[int, int], is_self: bool) -> SegmentInfo:
        with self._mu:
            sid = self._by_name.get(name)
            if sid is not None:
                return self._segments[sid].info
            info = SegmentInfo(self._next_segment, region[0], region[1])
            self._next_segment += 1
            self._segments[info.id] = _Segment(name, info, is_self)
            self._by_name[name] = info.id
            return info

    def close_segment(self, seg_id: int) -> None:
        with self._mu:
            seg = self._segments.pop(seg_id, None)
            if seg is None:
                return
            self._by_name.pop(seg.name, None)
        if not seg.is_self:
            try:
                self._agent.remove_remote_agent(seg.name)
            except Exception as e:  # noqa: BLE001
                _warn(f"remove_remote_agent({seg.name!r}): {e}")

    def _notif_loop(self) -> None:
        # The only caller of get_new_notifs: answers BUFREQ, files BUFRESP for open_segment.
        while not self._stop.wait(_NOTIF_POLL_S):
            try:
                notifs = self._agent.get_new_notifs()
            except Exception:  # noqa: BLE001
                continue
            for peer, msgs in notifs.items():
                for msg in msgs:
                    try:
                        self._on_notif(peer, msg)
                    except Exception as e:  # noqa: BLE001 - e.g. a BUFREQ before the peer's metadata: it resends
                        _warn(f"notification from {peer!r}: {e}")

    def _on_notif(self, peer: str, msg: bytes) -> None:
        if msg == BUFREQ:
            own = self._own_region()
            if own is not None and self._publish:
                self._agent.send_notif(peer, BUFRESP + f"{own[0]}:{own[1]}".encode())
        elif msg.startswith(BUFRESP):
            base, _, length = msg[len(BUFRESP):].partition(b":")
            if base.isdigit() and length.isdigit():
                with self._notif_cv:
                    self._peer_buf[peer] = (int(base), int(length))
                    self._notif_cv.notify_all()

    # ---------------------------------------------------------------- transfers

    def submit(self, op: XferOp, seg_id: int, local, remote, lengths) -> int:
        local, remote, lengths = (as_u64(local, "addresses"), as_u64(remote, "addresses"),
                                  as_u64(lengths, "lengths"))
        if local is None or remote is None or lengths is None:
            _warn("a negative address or length")
            return INVALID_BATCH
        n = len(local)
        if n == 0 or len(remote) != n or len(lengths) != n:
            return INVALID_BATCH
        with self._mu:
            seg = self._segments.get(seg_id)
            merged = self._merged
        if seg is None or self._agent is None:
            _warn(f"batch targets unknown segment {seg_id}")
            return INVALID_BATCH
        total, why = _core.check_batch(local, remote, lengths, merged, seg.info.base, seg.info.length)
        if why:
            _warn(f"{why} (segment {seg.name!r} [{seg.info.base:#x}, +{seg.info.length}))")
            return INVALID_BATCH

        if seg.is_self:
            _core.copy_ranges(op, local, remote, lengths)
            return self._add_batch(_Batch(None, total, done=True))

        try:
            ldescs = self._agent.get_xfer_descs(_descs(local, lengths), "DRAM")
            rdescs = self._agent.get_xfer_descs(_descs(remote, lengths), "DRAM")
            handle = self._agent.initialize_xfer("READ" if op == XferOp.READ else "WRITE", ldescs, rdescs,
                                                 seg.name)
        except Exception as e:  # noqa: BLE001
            _warn(f"initialize_xfer to {seg.name!r} failed: {e}")
            return INVALID_BATCH
        try:
            state = self._agent.transfer(handle)
        except Exception as e:  # noqa: BLE001
            _warn(f"transfer to {seg.name!r} failed: {e}")
            state = "ERR"
        if state == "ERR":
            try:
                self._agent.release_xfer_handle(handle)
                return INVALID_BATCH
            except Exception:  # noqa: BLE001
                return self._add_batch(_Batch(handle, total, failed=True))
        return self._add_batch(_Batch(handle, total, done=state == "DONE"))

    def _add_batch(self, b: _Batch) -> int:
        with self._mu:
            h = self._next_batch
            self._next_batch += 1
            self._batches[h] = b
            return h

    def poll(self, handle: int) -> Tuple[XferState, int]:
        # Under the lock through the state read: a release() of the same handle frees the request.
        with self._mu:
            b = self._batches.get(handle)
            if b is None:
                return XferState.INVALID, 0
            if b.failed:
                return XferState.FAILED, 0
            if b.done:
                return XferState.COMPLETED, b.nbytes
            state = self._agent.check_xfer_state(b.handle)
            if state == "DONE":
                b.done = True
                return XferState.COMPLETED, b.nbytes
            if state == "PROC":
                return XferState.PENDING, 0
            b.failed = True
            return XferState.FAILED, 0

    def release(self, handle: int) -> int:
        with self._mu:
            b = self._batches.get(handle)
            if b is None:
                return XFER_ENOENT
            if b.handle is not None:
                if not b.done and not b.failed and self._agent.check_xfer_state(b.handle) == "PROC":
                    return XFER_EBUSY
                try:
                    self._agent.release_xfer_handle(b.handle)
                except Exception as e:  # noqa: BLE001
                    _warn(f"release_xfer_handle failed: {e}")
                    return XFER_EBACKEND
            del self._batches[handle]
        return XFER_OK

    def close(self) -> None:
        """Stop the notification thread, release every batch and registration, and drop the agent, which
        closes its listener port. The backend refuses work afterwards."""
        self._stop.set()
        self._notif_thread.join(1)
        with self._mu:
            if self._agent is None:
                return
            for h, b in list(self._batches.items()):
                if b.handle is not None:
                    try:
                        self._agent.release_xfer_handle(b.handle)
                    except Exception:  # noqa: BLE001
                        _warn(f"batch {h} still in flight at close; its handle is left to NIXL")
            self._batches.clear()
            for addr, (_, descs) in list(self._registered.items()):
                try:
                    self._agent.deregister_memory(descs)
                except Exception:  # noqa: BLE001
                    pass
            self._registered.clear()
            self._merged = []
            self._segments.clear()
            self._by_name.clear()
            self._agent = None


def create(cfg: TransferConfig) -> Optional[NixlBackend]:
    try:
        return NixlBackend(cfg)
    except Exception as e:  # noqa: BLE001
        _warn(f"cannot start: {e}")
        return None
