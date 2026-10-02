#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$ROOT/build"
JOBS="$(nproc 2>/dev/null || echo 4)"
# One interpreter for everything below: CMake builds shmradix._core and
# shmradix._data against its pybind11, and pip installs the package into it.
PYTHON="${PYTHON:-python3}"

# ---------- C++ + Python extension modules ----------
echo "=== Building C++ and the Python extension modules (Release) ==="
mkdir -p "$BUILD_DIR"
cmake -S "$ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DPython3_EXECUTABLE="$(command -v "$PYTHON")"
cmake --build "$BUILD_DIR" -j "$JOBS"

# ---------- Python package ----------
echo "=== Installing Python package (pip install -e, no build isolation) ==="
"$PYTHON" -m pip install --no-build-isolation -e "$ROOT/python"

echo "=== Done ==="
