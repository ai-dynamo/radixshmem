#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Two-host check of the network device selection (docs/net_device_selection_design_zh.md, step 5).
# Host A runs this script, etcd and one radix server; host B runs the other server over ssh.
# A client gives node A the geometry, a writer on A publishes blocks, a reader on B pulls them.
#
#   scripts/dual_node_net_check.sh probe                 # what B has: devices, addresses, toolchain
#   scripts/dual_node_net_check.sh sync                  # copy this tree to B (build dirs excluded)
#   scripts/dual_node_net_check.sh build_b               # configure + build on B like A's build-mc
#   scripts/dual_node_net_check.sh run [server flags...] # no flags = zero-config; flags = baseline
#   scripts/dual_node_net_check.sh logs | stop
#
# Baseline for this lab:  run --rpc-interface bond0 --index-dev mlx5_bond_0 --gid-idx 3 --transfer-dev mlx5_bond_0
# Index and control on different networks:  run --rpc-interface bond0 --index-dev mlx5_0
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
A_IP=${A_IP:-10.6.131.12}
B_IP=${B_IP:-10.6.131.11}
B_HOST=${B_HOST:-root@$B_IP}
B_SSH_KEY=${B_SSH_KEY:-$HOME/.ssh/id_ed25519_ige}
DIR=${DIR:-$ROOT}                       # same path on both hosts
MOONCAKE_ROOT=${MOONCAKE_ROOT:-/raid/xrg/mooncake-te}
MOONCAKE_BUILD_DIR=${MOONCAKE_BUILD_DIR:-$MOONCAKE_ROOT/build-te}
ETCD_PORT=${ETCD_PORT:-12379}
WORK=${WORK:-/tmp/radixshmem-dual}      # logs, sockets, etcd data, on both hosts
NAME=/py_p2p                            # the client in python/tests/test_p2p_pull.py expects this name
TIMEOUT=${TIMEOUT:-180}

# The etcd Go client and gRPC honour HTTP_PROXY; the lab addresses must not go through a proxy.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
ENV_B="unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; export PYTHONPATH=$DIR/python PYTHONUNBUFFERED=1"
export PYTHONPATH=$DIR/python PYTHONUNBUFFERED=1

b() { ssh -i "$B_SSH_KEY" -o BatchMode=yes -o ConnectTimeout=10 "$B_HOST" "$ENV_B; $*"; }

server_cmd() {   # $1 = node name, $2 = socket path, rest = extra server flags
    local node=$1 sock=$2; shift 2
    echo "python3 -m radixshmem.cli --name $NAME --node-name $node --data-name ${NAME}_data_$node" \
         "--data-bytes 64M --endpoint unix://$sock --registry etcd://$A_IP:$ETCD_PORT" \
         "--cluster-id dual --expected-min-nodes 2 --bootstrap-timeout $TIMEOUT" \
         "--background-evict-ratio 0 --interval 0 $*"
}

wait_for() {     # $1 = needle, $2 = seconds; both server logs must contain it
    local i needle
    printf -v needle '%q' "$1"
    for ((i = 0; i < $2; i++)); do
        grep -q "$1" "$WORK/server_a.log" 2>/dev/null && b "grep -q -- $needle $WORK/server_b.log" && return 0
        if grep -q "Start FAILED\|Traceback" "$WORK/server_a.log" 2>/dev/null; then echo "server A failed"; return 1; fi
        if b "grep -q 'Start FAILED\|Traceback' $WORK/server_b.log 2>/dev/null"; then echo "server B failed"; return 1; fi
        sleep 1
    done
    echo "timeout waiting for: $1"; return 1
}

show_logs() {
    echo "== server A ($A_IP)"; grep -E 'net:|transfer devices|NUMA|registered as|rank=|FAILED|Error|error' "$WORK/server_a.log" | cut -c1-240
    echo "== server B ($B_IP)"; b "grep -E 'net:|transfer devices|NUMA|registered as|rank=|FAILED|Error|error' $WORK/server_b.log" | cut -c1-240
}

stop_all() {   # [r] keeps pkill from matching the shell that carries the pattern
    pkill -f "[r]adixshmem.cli --name $NAME" 2>/dev/null
    b "pkill -f '[r]adixshmem.cli --name $NAME'" 2>/dev/null
    pkill -f "[e]tcd --name dual-net" 2>/dev/null
    sleep 1
}

case "${1:-}" in
probe)
    b "hostname; ls -d $DIR $MOONCAKE_BUILD_DIR 2>&1; ibv_devices | tail -n +3 | awk '{print \$1}' | tr '\n' ' '; echo;
       ip -br addr | grep -v '^lo\|^veth\|^docker\|^br-'; ip route get $A_IP | head -1;
       which python3 cmake g++ go grpc_cpp_plugin; python3 --version;
       for d in /sys/class/infiniband/*; do echo \$(basename \$d) \$(cat \$d/ports/1/state) \$(cat \$d/ports/1/gid_attrs/types/3 2>/dev/null) \$(cat \$d/ports/1/gids/3); done"
    ;;
sync)
    rsync -a -e "ssh -i $B_SSH_KEY" --exclude 'build*/' --exclude '__pycache__/' --exclude '*.so' --exclude '.git/' \
        "$ROOT/" "$B_HOST:$DIR/"
    ;;
build_b)
    b "cmake -S $DIR -B $DIR/build-mc -DCMAKE_BUILD_TYPE=Release -DENABLE_MOONCAKE=ON \
          -DMOONCAKE_ROOT=$MOONCAKE_ROOT -DMOONCAKE_BUILD_DIR=$MOONCAKE_BUILD_DIR && \
       cmake --build $DIR/build-mc -j\$(nproc) && ls $DIR/python/radixshmem/_core*.so $DIR/python/radixshmem/_data*.so"
    ;;
run)
    shift
    b true || { echo "cannot reach $B_HOST with $B_SSH_KEY"; exit 1; }
    stop_all
    rm -rf "$WORK"; mkdir -p "$WORK"; b "rm -rf $WORK; mkdir -p $WORK"
    etcd --name dual-net --data-dir "$WORK/etcd" \
        --listen-client-urls "http://0.0.0.0:$ETCD_PORT" --advertise-client-urls "http://$A_IP:$ETCD_PORT" \
        --listen-peer-urls "http://127.0.0.1:$((ETCD_PORT + 1))" --initial-advertise-peer-urls "http://127.0.0.1:$((ETCD_PORT + 1))" \
        --initial-cluster "dual-net=http://127.0.0.1:$((ETCD_PORT + 1))" > "$WORK/etcd.log" 2>&1 &
    for ((i = 0; i < 50; i++)); do python3 -c "import socket; socket.create_connection(('$A_IP', $ETCD_PORT), 0.5)" 2>/dev/null && break; sleep 0.2; done

    echo "server flags: ${*:-(none: zero-config)}"
    $(server_cmd a "$WORK/node.sock" "$@") > "$WORK/server_a.log" 2>&1 &
    b "cd $DIR || exit 1; nohup $(server_cmd b "$WORK/node.sock" "$@") > $WORK/server_b.log 2>&1 < /dev/null &"
    wait_for "Waiting for a client's geometry" 60 || { show_logs; exit 1; }
    echo "both servers resolved their network; sending the geometry to A"
    python3 -c "
import radixshmem
c = radixshmem.RadixClient('$NAME', radixshmem.Geometry(block_size=1, full_slot_bytes=65536), endpoint='unix://$WORK/node.sock')
c.wait_ready($TIMEOUT); print('cluster:', c.info.mode); c.close()" || { show_logs; exit 1; }
    wait_for "Cluster ready" 30 || { show_logs; exit 1; }
    echo "== writer on A";  python3 "$DIR/python/tests/test_p2p_pull.py" --client writer "unix://$WORK/node.sock" | grep RESULT
    echo "== reader on B";  b "python3 $DIR/python/tests/test_p2p_pull.py --client reader unix://$WORK/node.sock" | grep RESULT
    show_logs
    stop_all
    ;;
logs) show_logs ;;
stop) stop_all ;;
*) sed -n '5,16p' "$0"; exit 2 ;;
esac
