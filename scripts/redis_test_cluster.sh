#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Local Redis Cluster for the Redis RHT tests: N masters on 127.0.0.1, no replicas.
#   REDIS_BIN=<dir of redis-server, redis-cli> scripts/redis_test_cluster.sh start [port] [masters]
#   scripts/redis_test_cluster.sh stop [port] [masters]
# Then configure with -DRADIXSHMEM_TEST_REDIS=127.0.0.1:<port>.
set -eu
cmd=${1:-}
port=${2:-7301}
n=${3:-3}
bin=${REDIS_BIN:-$(dirname "$(command -v redis-server)")}
run=/tmp/radixshmem-redis-test

case "$cmd" in
start)
  nodes=""
  for ((i = 0; i < n; i++)); do
    p=$((port + i))
    d=$run/$p
    mkdir -p "$d"
    rm -f "$d/nodes.conf"
    "$bin/redis-server" --port "$p" --bind 127.0.0.1 --cluster-enabled yes \
      --cluster-config-file "$d/nodes.conf" --cluster-node-timeout 5000 \
      --cluster-require-full-coverage no --save "" --appendonly no \
      --maxmemory 2gb --maxmemory-policy allkeys-lru \
      --dir "$d" --logfile "$d/redis.log" --pidfile "$d/redis.pid" --daemonize yes
    nodes="$nodes 127.0.0.1:$p"
  done
  sleep 1
  "$bin/redis-cli" --cluster create $nodes --cluster-replicas 0 --cluster-yes
  ;;
stop)
  for ((i = 0; i < n; i++)); do
    f=$run/$((port + i))/redis.pid
    if [ -f "$f" ]; then kill -CONT "$(cat "$f")" 2>/dev/null || true; kill "$(cat "$f")" || true; fi
  done
  sleep 1
  for ((i = 0; i < n; i++)); do rm -rf "$run/$((port + i))"; done
  ;;
*)
  echo "usage: $0 start|stop [port] [masters]" >&2
  exit 2
  ;;
esac
