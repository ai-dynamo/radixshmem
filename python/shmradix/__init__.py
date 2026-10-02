# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""shmradix - shared-memory radix index and slot store for KV cache sharing.

Public surface: ``RadixServer`` (one per node: index, SlotStore, P2P transfer;
started from a ``ServerConfig`` and configured by a client's ``Geometry``) and
``RadixClient`` (hands the server its geometry, ``status`` / ``wait_ready``,
index ops plus ``pull_async``). The pybind modules ``_core`` (index) and
``_data`` (SlotStore, transfer) stay importable for tests and advanced users.
"""

from shmradix._core import (
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

try:
    from shmradix import _data  # noqa: F401
except ImportError:
    pass

from shmradix._geometry import Geometry, check_pools_aligned
from shmradix.client import NodeInfo, PullJob, PullResult, RadixClient
from shmradix.rpc import DataPlaneUnavailable, GeometryMismatch, ServerBusy, ServerNotReady
from shmradix.server import ClusterConfig, RadixServer, ServerConfig

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
    "QueryResult",
    "RadixClient",
    "RadixServer",
    "RemoteStatus",
    "ServerBusy",
    "ServerConfig",
    "ServerNotReady",
    "check_pools_aligned",
    "compute_hashes",
]
