# RadixShmem

RadixShmem is a KV cache index built on a distributed radix tree. Each node keeps its own radix tree in POSIX
shared memory, and any process on the node can attach to it and insert or look up KV cache blocks by prefix
hash. Across nodes, a sharded Radix Hash Table (RHT) records which node holds each prefix, and a query
continues the match on that peer's tree, so the KV cache is indexed globally across the cluster. Together with
a transfer engine (Mooncake, or NIXL), RadixShmem also moves the blocks between nodes.

Source, documentation and issues: <https://github.com/ai-dynamo/radixshmem>.

## Install

```bash
pip install radixshmem
```

Wheels are built for Linux x86_64 (glibc 2.28 or newer) and CPython 3.10 to 3.14. They include the Mooncake
transfer backend, the default. For the NIXL backend (`--transfer-backend nixl`), install the `nixl` extra, which
brings `nixl-cu13` and torch (Python 3.13 or older):

```bash
pip install "radixshmem[nixl]"
```

What else the host needs depends on the setup:

| Setup | rdma-core | RDMA NIC |
|---|---|---|
| Single node, index only or with a data plane | no | no |
| Cluster, index only, Redis RHT + gRPC peer index (the default) | no | no |
| `RADIXSHMEM_RHT_BACKEND=rdma` or `RADIXSHMEM_PEER_INDEX_BACKEND=rdma` | yes | yes |
| Cluster with a data plane (Mooncake) | yes, plus libnuma | with `--transfer-protocol rdma` (the default) |
| Cluster with a data plane (NIXL) | as NIXL's UCX needs | for RDMA transports |

rdma-core and libnuma come from the distribution: `apt-get install libibverbs1 ibverbs-providers libnuma1`
on Debian and Ubuntu, `dnf install libibverbs numactl-libs` on RHEL and its rebuilds. A cluster also needs an
etcd for the bootstrap and, with the default RHT, a Redis Cluster
([deployment guide](https://github.com/ai-dynamo/radixshmem/blob/main/docs/redis_deployment.md)).

`python -c "import radixshmem; print(radixshmem.build_info())"` shows the version, whether rdma-core can be
loaded, and the transfer backends available.

## Start a server

Single node:

```bash
radix-server --name /my_kv_cache --data-bytes 1024G
```

Four nodes, coordinated through etcd, with the RHT in a Redis Cluster:

```bash
# Same command on every node except --rpc-address
export RADIXSHMEM_REDIS_ADDRS=10.0.0.1:7001,10.0.0.2:7001
radix-server --name /my_kv_cache --data-bytes 1024G \
    --expected-min-nodes 4 --registry etcd://10.0.0.1:2379 \
    --rpc-address 10.0.0.10 \
    --transfer-dev mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
```

The client API, the server options and the environment variables are described in
[docs/api.md](https://github.com/ai-dynamo/radixshmem/blob/main/docs/api.md).

## Compatibility

All nodes and clients of a cluster must run the same radixshmem release: nodes with a different cluster
protocol or shared-memory layout refuse each other.

## License

Apache License 2.0. The wheels bundle third-party code; their notices are in `ATTRIBUTIONS.md`, installed with
the package metadata.
