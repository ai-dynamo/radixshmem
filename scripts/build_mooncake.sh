#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="${MOONCAKE_SRC_DIR:-$ROOT_DIR/third_party/mooncake}"
BUILD_DIR="$SRC_DIR/build-te"

MOONCAKE_COMMIT="e5598b0992cc258b06d22f24d875480e8931a28e"
MOONCAKE_REPO="https://github.com/kvcache-ai/Mooncake.git"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RESET='\033[0m'

log() { echo -e "${CYAN}[build_mooncake]${RESET} $*"; }
die() { echo -e "${RED}[build_mooncake] error:${RESET} $*" >&2; exit 1; }

command -v git >/dev/null   || die "git not found"
command -v cmake >/dev/null || die "cmake not found"

if [[ ! -d "$SRC_DIR/.git" ]]; then
    log "cloning $MOONCAKE_REPO -> $SRC_DIR"
    git clone "$MOONCAKE_REPO" "$SRC_DIR"
fi

log "checking out pinned commit $MOONCAKE_COMMIT"
git -C "$SRC_DIR" fetch --quiet origin "$MOONCAKE_COMMIT" || true
git -C "$SRC_DIR" checkout --quiet "$MOONCAKE_COMMIT"
git -C "$SRC_DIR" submodule update --init --recursive --quiet

log "cmake configure ($BUILD_DIR)"
rm -rf "$BUILD_DIR"
cmake -S "$SRC_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DWITH_TE=ON \
    -DWITH_STORE=OFF -DWITH_STORE_RUST=OFF -DWITH_P2P_STORE=OFF \
    -DBUILD_UNIT_TESTS=OFF -DUSE_ETCD=OFF

log "building transfer_engine (-j$(nproc))"
cmake --build "$BUILD_DIR" --target transfer_engine -j"$(nproc)"

[[ -f "$SRC_DIR/mooncake-transfer-engine/include/transfer_engine.h" ]] \
    || die "build finished but transfer_engine.h is missing"
for lib in mooncake-transfer-engine/src/libtransfer_engine.a \
           mooncake-transfer-engine/src/common/base/libbase.a \
           mooncake-common/src/libmooncake_common.a \
           mooncake-common/libasio.so; do
    [[ -f "$BUILD_DIR/$lib" ]] || die "build finished but $lib is missing"
done

echo -e "${GREEN}[build_mooncake] done${RESET} — MOONCAKE_ROOT=$SRC_DIR (matches CMakeLists.txt's default)"
