#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

status=0
for f in "$@"; do
    if ! head -n 5 "$f" | grep -q 'SPDX-License-Identifier: Apache-2.0'; then
        echo "missing SPDX header: $f"
        status=1
    fi
done
exit $status
