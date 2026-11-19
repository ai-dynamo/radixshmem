# Redis Deployment Guide

With `RADIXSHMEM_RHT_BACKEND=redis`, routes are stored in a Redis Cluster. We recommend three masters with one replica each, six redis-server processes in total.

Recommended versions are Redis 8.10.3 (7.0 or later) and Valkey 9.0.7 (7.2 or later); both are tested.

The example below uses eight radixshmem nodes, n1–n8. Nodes n1–n6 each run one redis-server on port 7001: n1–n3 as masters, n4–n6 as replicas. n7 and n8 run no Redis. Since no node runs more than one redis-server, losing any single node takes down at most one master, and the cluster fails over on its own.

## 1. Redis server configuration

All six nodes use the same `/etc/redis/7001.conf`:

```conf
port 7001
bind <node IP>
requirepass <pw>
masterauth <pw>
dir /var/lib/redis/7001
cluster-enabled yes
cluster-config-file nodes-7001.conf
cluster-node-timeout 5000
cluster-require-full-coverage no
cluster-replica-validity-factor 0
save ""
appendonly no
maxmemory 32gb
maxmemory-policy allkeys-lru
repl-diskless-sync yes
repl-backlog-size 256mb
client-output-buffer-limit replica 1gb 512mb 120
maxclients 20000
timeout 300
```

Also disable transparent huge pages on n1–n6:

```bash
echo never > /sys/kernel/mm/transparent_hugepage/enabled
```

## 2. Start redis-server with systemd

`/etc/systemd/system/redis-7001.service`:

```ini
[Service]
ExecStart=/usr/local/bin/redis-server /etc/redis/7001.conf
Restart=on-failure
RestartSec=5
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

Then on n1–n6:

```bash
mkdir -p /var/lib/redis/7001
systemctl daemon-reload
systemctl enable --now redis-7001
```

## 3. Create the cluster

Once all six redis-servers are running, run on any node:

```bash
redis-cli -a <pw> --cluster create n1:7001 n2:7001 n3:7001 n4:7001 n5:7001 n6:7001 --cluster-replicas 1
redis-cli -a <pw> --cluster check n1:7001
```

The `check` output should list three masters, each with one replica.

## 4. radixshmem configuration

Set these environment variables for radix-server:

```bash
RADIXSHMEM_RHT_BACKEND=redis
RADIXSHMEM_REDIS_ADDRS=n1:7001,n2:7001,n3:7001
RADIXSHMEM_REDIS_PASSWORD=<pw>
RADIXSHMEM_REDIS_TIMEOUT_MS=50
```

Several radixshmem clusters can share one Redis Cluster as long as their `cluster_id`s differ: a new radixshmem cluster deletes every key under its prefix, `rsh:<cluster_id>:`.

Start the Redis Cluster first: rank 0 fails to start unless a node in `RADIXSHMEM_REDIS_ADDRS` reports `cluster_state:ok`. A master that restarts comes back empty, since persistence is off; the radix-servers notice its new `run_id` within a second and send their routes in its slots again.

## 5. Fault testing

`python/tests/redis_chaos.py` checks how a radixshmem cluster rides out Redis failures. It starts its own Redis Cluster on 127.0.0.1 (three masters with one replica each, binaries from `RADIXSHMEM_TEST_REDIS_BIN`) and injects a random fault every `--interval` seconds once the cluster has healed from the previous one: a master or replica killed and restarted at once or after its failover, a master stopped for 3 or 15 seconds, a master killed together with its replica, or every node killed at once. With `--stress` it runs `tests/agentic_stress.py` against that cluster and, at the end, prints the hit rate and throughput around each fault:

```bash
cd python
RADIXSHMEM_TEST_REDIS_BIN=<redis bin dir> python tests/redis_chaos.py --interval 60 \
    --stress "--nodes 4 --duration 600 --monitor-port 9810"
```

That load is synthetic: sessions with random hashes, so they share no prefixes. To replay the AgentX trace instead, which shares long prefixes within an agent session, pass it to `agentic_stress` with `--trace`. At `--trace-speedup 200` the trace takes about 83 minutes, and the run ends when it does:

```bash
RADIXSHMEM_TEST_REDIS_BIN=<redis bin dir> python tests/redis_chaos.py --interval 120 \
    --stress "--nodes 4 --concurrency 32 --executors-per-worker 4 --monitor-port 9810 \
              --trace /path/to/cc-traces-weka-062126-256k.traces.jsonl --trace-speedup 200 \
              --full-slot-bytes 4096 --max-blocks 300000 --duration 36000 --warmup-sec 0"
```

Without `--stress` it prints the `RADIXSHMEM_REDIS_ADDRS` to run against and injects faults until Ctrl-C. The faults are logged in `<workdir>/chaos.jsonl`.
