# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transfer backends behind one interface, TransferBackend, and SlotTransfer on top of them.

``mooncake`` (the default) is the C++ Mooncake plugin next to ``_core``, reached through
``_core.NativeTransferBackend``.
"""

from __future__ import annotations

from typing import List, Optional

from radixshmem import _core
from radixshmem._core import TransferConfig
from radixshmem.transfer.base import TransferBackend
from radixshmem.transfer.slot_transfer import SlotTransfer

TransferBackend.register(_core.NativeTransferBackend)


def available_backends() -> List[str]:
    """The backends this installation can create: those whose plugin is installed."""
    return list(_core.TRANSFER_BACKENDS)


def create_transfer_backend(config: TransferConfig) -> Optional[TransferBackend]:
    """The backend named by ``config.backend``, or None if it is unknown or cannot start (the reason goes to
    stderr)."""
    return _core.create_native_backend(config)


__all__ = ["SlotTransfer", "TransferBackend", "TransferConfig", "available_backends", "create_transfer_backend"]
