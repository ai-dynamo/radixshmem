#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Part B+C: build source-free wheels for both variants, inside manylinux_2_28.
# Assumes Part A already produced packaging/prebuilt/_core.{rdma,nordma}.so.
# Runs INSIDE the manylinux container.
#
# Output: packaging/dist/*.whl (manylinux-tagged, repaired)
set -euo pipefail

ROOT=/work                      # radixshmem source (mounted)
PKG="$ROOT/packaging"
PY=/opt/python/cp312-cp312/bin/python
PLAT=manylinux_2_28_x86_64

"$PY" -m pip install -q build auditwheel wheel "setuptools>=64"

# Ensure libxxhash is present so auditwheel can find/bundle it.
dnf install -y --enablerepo=powertools xxhash-libs xxhash-devel >/dev/null 2>&1 || true

build_one () {
    local so="$1" variant="$2"; shift 2
    local excludes=("$@")
    echo "======== wheel variant=$variant ========"
    local stage; stage=$(mktemp -d)
    # Stage a clean source-free package tree.
    cp "$PKG/setup.py" "$PKG/pyproject.toml" "$stage/"
    cp "$ROOT/LICENSE" "$ROOT/ATTRIBUTIONS.md" "$stage/"
    mkdir -p "$stage/shmradix"
    cp "$ROOT/python/shmradix/__init__.py" "$stage/shmradix/"
    cp "$ROOT/python/shmradix/server.py"   "$stage/shmradix/"
    : > "$stage/shmradix/py.typed"
    # Swap in this variant's prebuilt .so under the canonical extension name.
    cp "$PKG/prebuilt/$so" \
       "$stage/shmradix/_core.cpython-312-x86_64-linux-gnu.so"

    ( cd "$stage" && SHMRADIX_WHEEL_VARIANT="$variant" \
        "$PY" -m build --wheel --no-isolation )

    local raw; raw=$(ls "$stage"/dist/*.whl)
    echo "  raw wheel: $(basename "$raw")"

    local exargs=()
    for e in "${excludes[@]}"; do exargs+=(--exclude "$e"); done
    # auditwheel bundles libxxhash; RDMA libs are excluded (external system dep).
    auditwheel repair "${exargs[@]}" --plat "$PLAT" -w "$PKG/dist" "$raw" \
        || { echo "  auditwheel manylinux failed; retry linux tag"; \
             auditwheel repair "${exargs[@]}" --plat linux_x86_64 \
               -w "$PKG/dist" "$raw"; }
    rm -rf "$stage"
}

mkdir -p "$PKG/dist"; rm -f "$PKG/dist"/*.whl 2>/dev/null || true

# single-node variant: fully self-contained (bundle libxxhash, no excludes)
build_one _core.nordma.so cpu

# RDMA variant: exclude the driver/hardware-coupled RDMA stack (external dep)
build_one _core.rdma.so rdma \
    libibverbs.so.1 libmlx5.so.1 libnl-route-3.so.200 libnl-3.so.200

echo "======== done ========"
ls -la "$PKG/dist"
