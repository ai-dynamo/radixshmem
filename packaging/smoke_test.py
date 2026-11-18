# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test of an installed radixshmem wheel. With rdma-core on the host the Mooncake plugin must
start a backend over TCP; without it radixshmem must still import and the plugin must refuse cleanly."""

import pathlib
import subprocess
import sys

import radixshmem
from radixshmem import _data

info = radixshmem.build_info()
print(info)
assert info["version"] == radixshmem.__version__
assert "mooncake" in info["transfer_backends"], info

script = pathlib.Path(sys.executable).with_name("radix-server")
for cmd in ([sys.executable, "-m", "radixshmem.cli", "--help"], [str(script), "--help"]):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

backend = _data.create_transfer_backend(_data.TransferConfig(
    backend="mooncake", metadata="P2PHANDSHAKE", ip_or_host="127.0.0.1", rpc_port=0, protocol="tcp",
    device_filter=[], publish=False))
if info["rdma"]:
    assert backend is not None, "the Mooncake plugin failed with rdma-core present"
else:
    assert backend is None and info["rdma_error"], info
print(f"smoke test passed ({'with' if info['rdma'] else 'without'} rdma-core)")
