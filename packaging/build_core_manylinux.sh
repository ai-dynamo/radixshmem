#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Part A: build both _core.so variants inside a manylinux_2_28 container.
# Runs INSIDE the manylinux container (see build_wheels.sh for the docker run).
#
# Produces, under /work/packaging/prebuilt/ :
#   _core.rdma.so    (RADIXSHMEM_ENABLE_RDMA=ON  -> needs libibverbs/libmlx5 at runtime)
#   _core.nordma.so  (RADIXSHMEM_ENABLE_RDMA=OFF -> pure single-node shared memory)
set -euo pipefail

SRC=/work                       # radixshmem source (mounted)
PY=/opt/python/cp312-cp312/bin/python
OUT=/work/packaging/prebuilt
mkdir -p "$OUT"

echo "=== install build deps ==="
dnf install -y --enablerepo=powertools xxhash-devel rdma-core-devel libnl3-devel >/dev/null
"$PY" -m pip install -q pybind11 numpy setuptools wheel

build_variant () {
    local rdma="$1" outname="$2"
    echo "=== build variant RDMA=$rdma -> $outname ==="
    # Clean any prior cmake/build state so the toggle takes effect.
    rm -rf "$SRC/build" "$SRC/python/build" \
           "$SRC/python/radixshmem/_core.cpython-"*.so 2>/dev/null || true
    ( cd "$SRC/python" && RADIXSHMEM_ENABLE_RDMA="$rdma" \
        "$PY" setup.py build_ext --inplace )
    cp "$SRC/python/radixshmem/_core.cpython-"*.so "$OUT/$outname"
    echo "  -> $OUT/$outname"
    "$PY" - <<PYEOF
import subprocess, sys
out = subprocess.run(["ldd", "$OUT/$outname"], capture_output=True, text=True).stdout
rdma = [l for l in out.splitlines() if any(x in l for x in ("ibverbs","mlx5"))]
print("  ldd RDMA libs:", rdma if rdma else "(none)")
PYEOF
}

build_variant ON  _core.rdma.so
build_variant OFF _core.nordma.so
echo "=== Part A done ==="
ls -la "$OUT"
