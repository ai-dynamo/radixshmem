#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Local Redis Cluster for the Redis RHT tests: N masters on 127.0.0.1, each with R replicas (default 0).
#   REDIS_BIN=<dir of redis-server, redis-cli or their valkey-* names> scripts/redis_test_cluster.sh start [port] [masters] [replicas]
#   scripts/redis_test_cluster.sh stop [port] [masters] [replicas]
# Then configure with -DRADIXSHMEM_TEST_REDIS=127.0.0.1:<port>.
set -eu
cmd=${1:-}
port=${2:-7301}
n=${3:-3}
r=${4:-0}
nodes_total=$((n * (1 + r)))
bin=${REDIS_BIN:-$(dirname "$(command -v redis-server || command -v valkey-server)")}
server=$bin/redis-server
cli=$bin/redis-cli
if [ ! -x "$server" ] && [ -x "$bin/valkey-server" ]; then server=$bin/valkey-server; cli=$bin/valkey-cli; fi
run=/tmp/radixshmem-redis-test

case "$cmd" in
start)
  nodes=""
  for ((i = 0; i < nodes_total; i++)); do
    p=$((port + i))
    d=$run/$p
    mkdir -p "$d"
    rm -f "$d/nodes.conf"
    "$server" --port "$p" --bind 127.0.0.1 --cluster-enabled yes \
      --cluster-config-file "$d/nodes.conf" --cluster-node-timeout 5000 \
      --cluster-require-full-coverage no --save "" --appendonly no \
      --maxmemory 2gb --maxmemory-policy allkeys-lru \
      --dir "$d" --logfile "$d/redis.log" --pidfile "$d/redis.pid" --daemonize yes
    nodes="$nodes 127.0.0.1:$p"
  done
  sleep 1
  "$cli" --cluster create $nodes --cluster-replicas "$r" --cluster-yes
  ;;
stop)
  for ((i = 0; i < nodes_total; i++)); do
    f=$run/$((port + i))/redis.pid
    if [ -f "$f" ]; then kill -CONT "$(cat "$f")" 2>/dev/null || true; kill "$(cat "$f")" || true; fi
  done
  sleep 1
  for ((i = 0; i < nodes_total; i++)); do rm -rf "$run/$((port + i))"; done
  ;;
*)
  echo "usage: $0 start|stop [port] [masters] [replicas]" >&2
  exit 2
  ;;
esac
