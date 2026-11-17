# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The data plane in one place: SlotStore, geometry, the data-plane registry (``radixshmem._core``) and the
transfer backends with SlotTransfer (``radixshmem.transfer``)."""

from radixshmem._core import (
    COMPONENT_MASK_FULL,
    COMPONENT_MASK_MAMBA,
    COMPONENT_MASK_SWA,
    INVALID_BATCH,
    PEER_SLOT_BYTES_UNCHECKED,
    XFER_EBACKEND,
    XFER_EBUSY,
    XFER_EGEOMETRY,
    XFER_EINVAL,
    XFER_ENOENT,
    XFER_OK,
    ComponentType,
    DataPlaneEntry,
    DataPlaneRegistry,
    PoolGeometry,
    SegmentInfo,
    SlotGeometry,
    SlotPoolConfig,
    SlotStore,
    SlotStoreConfig,
    StoreGeometry,
    TransferConfig,
    XferOp,
    XferState,
    make_data_plane_entry,
)
from radixshmem.transfer import SlotTransfer, TransferBackend, available_backends, create_transfer_backend

TRANSFER_BACKENDS = available_backends()

__all__ = [
    "COMPONENT_MASK_FULL",
    "COMPONENT_MASK_MAMBA",
    "COMPONENT_MASK_SWA",
    "INVALID_BATCH",
    "PEER_SLOT_BYTES_UNCHECKED",
    "TRANSFER_BACKENDS",
    "XFER_EBACKEND",
    "XFER_EBUSY",
    "XFER_EGEOMETRY",
    "XFER_EINVAL",
    "XFER_ENOENT",
    "XFER_OK",
    "ComponentType",
    "DataPlaneEntry",
    "DataPlaneRegistry",
    "PoolGeometry",
    "SegmentInfo",
    "SlotGeometry",
    "SlotPoolConfig",
    "SlotStore",
    "SlotStoreConfig",
    "SlotTransfer",
    "StoreGeometry",
    "TransferBackend",
    "TransferConfig",
    "XferOp",
    "XferState",
    "create_transfer_backend",
    "make_data_plane_entry",
]
