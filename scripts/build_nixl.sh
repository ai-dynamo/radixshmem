#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Builds NIXL (github.com/ai-dynamo/nixl) with the UCX backend and installs it
# where CMakeLists.txt's NIXL_ROOT default expects it, so `cmake ..` finds
# nixl.h / libnixl.so with no further setup.
#
# Pinned to a specific commit rather than tracking a branch: nixl_backend.cpp
# (src/transfer/nixl) codes against NIXL's C++ agent API directly, and NIXL
# has no stability guarantee across commits — an unpinned `git pull` could
# silently change that surface out from under the backend. Bump NIXL_COMMIT
# deliberately (and re-run the dataplane NIXL tests) when picking up a newer
# NIXL, not as a side effect of a routine rebuild.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="${NIXL_SRC_DIR:-$ROOT_DIR/third_party/nixl}"
INSTALL_DIR="${NIXL_INSTALL_DIR:-$ROOT_DIR/third_party/nixl-install}"

# The commit this project's NIXL backend (src/transfer/nixl) was built and
# end-to-end tested against (two-host RDMA READ, byte-verified).
NIXL_COMMIT="483125c2020dfc09ff22e36bc90c4ca0888a1ec9"
NIXL_REPO="https://github.com/ai-dynamo/nixl.git"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RESET='\033[0m'

log() { echo -e "${CYAN}[build_nixl]${RESET} $*"; }
die() { echo -e "${RED}[build_nixl] error:${RESET} $*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight

command -v git >/dev/null   || die "git not found"
command -v meson >/dev/null || die "meson not found (pip install meson)"
command -v ninja >/dev/null || die "ninja not found (pip install ninja)"
pkg-config --exists ucx || die "UCX not found via pkg-config — install/build UCX first"

# NIXL always builds its Python bindings, and meson locates pybind11 through
# pkg-config; point it at the copy installed for the interpreter meson will use.
PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null || die "$PYTHON not found"
"$PYTHON" -c 'import pybind11' 2>/dev/null \
    || die "pybind11 not importable by $PYTHON — pip install pybind11 (NIXL's Python bindings need it)"
export PATH="$(dirname "$(command -v "$PYTHON")"):$PATH"
export PKG_CONFIG_PATH="$("$PYTHON" -m pybind11 --pkgconfigdir)${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

# NIXL's meson build needs nvcc on PATH even for a UCX-only build (core CUDA
# detection, not just the GDS plugin). Common install locations, tried in
# order; override by exporting PATH yourself before running this script.
if ! command -v nvcc >/dev/null; then
    for cuda_bin in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
        if [[ -x "$cuda_bin/nvcc" ]]; then
            export PATH="$cuda_bin:$PATH"
            break
        fi
    done
fi
command -v nvcc >/dev/null || die "nvcc not found — install the CUDA toolkit or add it to PATH"

# ---------------------------------------------------------------- fetch

if [[ ! -d "$SRC_DIR/.git" ]]; then
    log "cloning $NIXL_REPO -> $SRC_DIR"
    git clone "$NIXL_REPO" "$SRC_DIR"
fi

log "checking out pinned commit $NIXL_COMMIT"
git -C "$SRC_DIR" fetch --quiet origin "$NIXL_COMMIT" || true  # already-local commits need no fetch
git -C "$SRC_DIR" checkout --quiet "$NIXL_COMMIT"

# ---------------------------------------------------------------- build

log "meson setup (prefix=$INSTALL_DIR)"
rm -rf "$SRC_DIR/build"
meson setup "$SRC_DIR/build" "$SRC_DIR" --prefix="$INSTALL_DIR" \
    -Denable_plugins=UCX -Dbuild_tests=false -Dbuild_examples=false -Dinstall_headers=true

log "building (ninja -j$(nproc))"
ninja -C "$SRC_DIR/build" -j"$(nproc)"

log "installing to $INSTALL_DIR"
ninja -C "$SRC_DIR/build" install

# ---------------------------------------------------------------- verify

[[ -f "$INSTALL_DIR/include/nixl.h" ]] || die "install finished but include/nixl.h is missing"
find "$INSTALL_DIR" -iname "libnixl.so" | grep -q . || die "install finished but libnixl.so is missing"
find "$INSTALL_DIR" -iname "libplugin_UCX.so" | grep -q . || die "install finished but the UCX plugin is missing"

echo -e "${GREEN}[build_nixl] done${RESET} — NIXL_ROOT=$INSTALL_DIR (matches CMakeLists.txt's default)"

# The UCX plugin is found next to libnixl.so, but it links libucp/libucs from
# wherever UCX lives; outside the loader's default paths that needs an override.
UCX_LIBDIR="$(pkg-config --variable=libdir ucx)"
if ! ldconfig -p 2>/dev/null | grep -q " ${UCX_LIBDIR}/libucp.so"; then
    log "UCX at $UCX_LIBDIR is not in the loader's default path; run NIXL users with:"
    log "    export LD_LIBRARY_PATH=$UCX_LIBDIR\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
fi
