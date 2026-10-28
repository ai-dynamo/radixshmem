# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""gRPC channel between RadixServer and RadixClient: unary methods under
``radixshmem.RadixNode``, JSON bodies, no generated code.

Methods: Describe() -> NodeInfo; WaitReady({"timeout_ms"}) -> NodeInfo;
Configure({"geometry"}) -> NodeInfo; Submit({"items", "timeout_ms"}) ->
{"ticket"}; WaitAny({"tickets", "timeout_ms"}) -> {"done": {ticket: [[ok,
bytes, descriptors, error], ...]}, "unknown": [ticket, ...]}; Stats() -> dict;
Nodes() -> {"nodes": [node record, ...]}.
The slot geometry is what a client configured (or a peer published);
Describe publishes it and clients adopt it.
"""

from __future__ import annotations

import json
import os
from concurrent import futures
from typing import Any, Callable, Dict, Optional

import grpc

SERVICE = "radixshmem.RadixNode"
_OPTIONS = [("grpc.max_receive_message_length", -1), ("grpc.max_send_message_length", -1)]


class DataPlaneUnavailable(RuntimeError):
    """The server runs index-only (no SlotStore) or is not ready."""


class ServerBusy(RuntimeError):
    """The server's job table is full; the client falls back to its local hit."""


class GeometryMismatch(RuntimeError):
    """The server, or a node of its cluster, already serves another geometry."""


class ServerNotReady(RuntimeError):
    """``RadixClient`` attaches only a ``ready`` server; ``mode`` says what it
    is instead and ``last_error`` why the last configuration failed."""

    def __init__(self, mode: str, last_error: str = ""):
        tail = f" (last error: {last_error})" if last_error else ""
        super().__init__(f"server is {mode}{tail}")
        self.mode = mode
        self.last_error = last_error


def _enc(o: Any) -> bytes:
    return json.dumps({} if o is None else o).encode()


def _dec(b: bytes) -> Any:
    return json.loads(b) if b else {}


def endpoint_for(name: str) -> str:
    """Default socket of the server started as --name <name>."""
    return f"unix:///dev/shm/{name.lstrip('/').replace('/', '_')}.sock"


def _abort(ctx, e: BaseException) -> None:
    if isinstance(e, DataPlaneUnavailable):
        ctx.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
    if isinstance(e, ServerBusy):
        ctx.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(e))
    if isinstance(e, GeometryMismatch):
        ctx.abort(grpc.StatusCode.ALREADY_EXISTS, str(e))
    if isinstance(e, ValueError):
        ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
    if isinstance(e, TimeoutError):
        ctx.abort(grpc.StatusCode.DEADLINE_EXCEEDED, str(e))
    ctx.abort(grpc.StatusCode.INTERNAL, f"{type(e).__name__}: {e}")


def _raise(e: grpc.RpcError) -> Exception:
    code, msg = e.code(), e.details()
    if code == grpc.StatusCode.FAILED_PRECONDITION:
        return DataPlaneUnavailable(msg)
    if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
        return ServerBusy(msg)
    if code == grpc.StatusCode.ALREADY_EXISTS:
        return GeometryMismatch(msg)
    if code == grpc.StatusCode.INVALID_ARGUMENT:
        return ValueError(msg)
    if code == grpc.StatusCode.DEADLINE_EXCEEDED:
        return TimeoutError(msg)
    return RuntimeError(f"{code.name}: {msg}")


def serve(handlers: Dict[str, Callable[[Any], Any]], endpoint: str, workers: int) -> grpc.Server:
    """``handlers[method](request)`` returns a JSON-able reply or raises."""
    def wrap(fn):
        def behavior(req, ctx):
            try:
                return fn(req)
            except Exception as e:  # noqa: BLE001
                _abort(ctx, e)
        return grpc.unary_unary_rpc_method_handler(
            behavior, request_deserializer=_dec, response_serializer=_enc)

    if endpoint.startswith("unix://"):
        try:
            os.unlink(endpoint[len("unix://"):])
        except OSError:
            pass
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=workers), options=_OPTIONS)
    server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(
        SERVICE, {m: wrap(fn) for m, fn in handlers.items()}),))
    if server.add_insecure_port(endpoint) == 0:
        raise RuntimeError(f"cannot bind {endpoint}")
    server.start()
    return server


class Stub:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self._ch = grpc.insecure_channel(endpoint, options=_OPTIONS)

    def call(self, method: str, req: Any = None, timeout: Optional[float] = None) -> Any:
        fn = self._ch.unary_unary(f"/{SERVICE}/{method}", request_serializer=_enc,
                                  response_deserializer=_dec)
        try:
            return fn(req, timeout=timeout)
        except grpc.RpcError as e:
            raise _raise(e) from None

    def close(self) -> None:
        self._ch.close()
