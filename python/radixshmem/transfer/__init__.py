# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transfer backends behind one interface, TransferBackend, and SlotTransfer on top of them.

``mooncake`` (the default) is the C++ Mooncake plugin next to ``_core``, reached through
``_core.NativeTransferBackend``. ``nixl`` drives NIXL's Python package (``pip install radixshmem[nixl]``).
"""

from __future__ import annotations

from typing import List, Optional

from radixshmem import _core
from radixshmem._core import TransferConfig
from radixshmem.transfer import nixl
from radixshmem.transfer.base import TransferBackend
from radixshmem.transfer.slot_transfer import SlotTransfer

TransferBackend.register(_core.NativeTransferBackend)


def available_backends() -> List[str]:
    """The backends this installation can create: Mooncake if its plugin is installed, NIXL if its package is."""
    return list(_core.TRANSFER_BACKENDS) + (["nixl"] if nixl.available() else [])


def create_transfer_backend(config: TransferConfig) -> Optional[TransferBackend]:
    """The backend named by ``config.backend``, or None if it is unknown or cannot start (the reason goes to
    stderr)."""
    if config.backend == "nixl":
        return nixl.create(config)
    return _core.create_native_backend(config)


__all__ = ["SlotTransfer", "TransferBackend", "TransferConfig", "available_backends", "create_transfer_backend"]
