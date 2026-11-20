# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""radixshmem - shared-memory radix index and slot store for KV cache sharing.

Public surface: ``RadixServer`` (one per node: index, SlotStore, P2P transfer;
started from a ``ServerConfig`` and configured by a client's ``Geometry``) and
``RadixClient`` (hands the server its geometry, ``status`` / ``wait_ready``,
index ops plus ``pull_async`` / ``push_async``). ``WritebackHandler`` / ``WritebackTask``:
the write-back side of background eviction (``radixshmem.writeback``). The pybind module ``_core`` and
``_data``, its data-plane names (SlotStore, transfer), stay importable for tests and advanced users.
"""

from radixshmem._core import (
    COMPONENT_MASK_FULL,
    COMPONENT_MASK_MAMBA,
    COMPONENT_MASK_SWA,
    INVALID_COMPONENT_SLOT,
    MAX_REMOTE_OP_BLOCKS,
    ComponentQueryResult,
    ComponentType,
    EvictPolicy,
    HealthFlag,
    HealthPoolReport,
    HealthReport,
    InsertError,
    InsertResult,
    QueryResult,
    RemoteStatus,
    compute_hashes,
)

from radixshmem import _data  # noqa: F401
from radixshmem._geometry import Geometry, check_pools_aligned
from radixshmem._version import __version__
from radixshmem.client import NodeInfo, PullJob, PullResult, PushJob, PushResult, RadixClient
from radixshmem.rpc import DataPlaneUnavailable, GeometryMismatch, ServerBusy, ServerNotReady
from radixshmem.server import ClusterConfig, RadixServer, ServerConfig
from radixshmem.writeback import WritebackHandler, WritebackTask


def build_info() -> dict:
    """Package, shm and cluster protocol versions, whether rdma-core loads, and the transfer backends."""
    from radixshmem import _core

    from radixshmem.transfer import available_backends

    backends = available_backends()
    rdma_error = _core.rdma_error()
    info = {"version": __version__, "shm_version": _core.SHM_VERSION,
            "protocol_version": _core.CLUSTER_PROTOCOL_VERSION, "rdma": not rdma_error,
            "transfer_backends": backends}
    if rdma_error:
        info["rdma_error"] = rdma_error
    return info


__all__ = [
    "COMPONENT_MASK_FULL",
    "COMPONENT_MASK_MAMBA",
    "COMPONENT_MASK_SWA",
    "INVALID_COMPONENT_SLOT",
    "MAX_REMOTE_OP_BLOCKS",
    "ClusterConfig",
    "ComponentQueryResult",
    "ComponentType",
    "DataPlaneUnavailable",
    "EvictPolicy",
    "Geometry",
    "GeometryMismatch",
    "HealthFlag",
    "HealthPoolReport",
    "HealthReport",
    "InsertError",
    "InsertResult",
    "NodeInfo",
    "PullJob",
    "PullResult",
    "PushJob",
    "PushResult",
    "QueryResult",
    "RadixClient",
    "RadixServer",
    "RemoteStatus",
    "ServerBusy",
    "ServerConfig",
    "ServerNotReady",
    "WritebackHandler",
    "WritebackTask",
    "build_info",
    "check_pools_aligned",
    "compute_hashes",
]
