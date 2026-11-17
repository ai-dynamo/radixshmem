#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Installs a radixshmem wheel in a fresh venv on a host with no build tools, then runs the smoke
# test and the single-node tests without rdma-core, and the smoke test again with rdma-core.
# Run from the repository root:
#   packaging/test_clean_host.sh <python version> <directory with the wheels>
set -eu
py=$1
wheels=$2

install() {
    if command -v apt-get > /dev/null; then
        apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$@" > /dev/null
    else
        dnf install -y -q "$@" > /dev/null
    fi
}

if ! command -v uv > /dev/null; then
    command -v curl > /dev/null || install curl ca-certificates
    for tool in tar gzip; do command -v $tool > /dev/null || install $tool; done
    mkdir -p /opt/uv
    curl -LsSf https://github.com/astral-sh/uv/releases/download/0.13.6/uv-x86_64-unknown-linux-gnu.tar.gz |
        tar -xz -C /opt/uv --strip-components 1
    PATH=/opt/uv:$PATH
fi
uv venv -q -p "$py" /tmp/venv-"$py"
tag=cp$(echo "$py" | tr -d .)
uv pip install -q -p /tmp/venv-"$py" "$wheels"/radixshmem-*-"$tag"-*.whl pytest
python=/tmp/venv-"$py"/bin/python

$python packaging/smoke_test.py
$python -m pytest -q -p no:cacheprovider python/tests/test_geometry.py python/tests/test_radixshmem.py \
    python/tests/test_monitor.py python/tests/test_server_rpc.py python/tests/test_transfer_jobs.py \
    python/tests/test_writeback.py
$python python/tests/test_radixshmem_data.py

if command -v apt-get > /dev/null; then
    install libibverbs1 ibverbs-providers libnuma1
else
    install libibverbs numactl-libs
fi
$python packaging/smoke_test.py
