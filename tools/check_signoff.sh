#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if ! grep -q '^Signed-off-by: ' "$1"; then
    echo "missing Signed-off-by, use git commit -s"
    exit 1
fi
