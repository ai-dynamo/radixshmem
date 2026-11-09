# Redis Deployment Guide

With `RADIXSHMEM_RHT_BACKEND=redis`, routes are stored in a Redis Cluster. We recommend three masters with one replica each, six redis-server processes in total.

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

Several radixshmem clusters can share one Redis Cluster as long as their `cluster_id`s differ.
