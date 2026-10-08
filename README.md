# RadixShmem

A RadixTree-based global KV cache index that provides memory-speed prefix lookups across processes and nodes.

## Architecture

**Within a node.** The index lives in POSIX shared memory (`shm_open` + `mmap`). Processes attach to it and
run prefix queries as direct pointer walks; reads and writes are coordinated by a process-shared read-write
lock, with no serialization or IPC.

**Across nodes.** Servers discover each other and are assigned ranks through etcd, and register their shared
memory as RDMA MRs. A query first matches the local index, then uses the routing hash table (RHT), sharded
across nodes, to find the node with a longer hit, and walks that node's index with RDMA one-sided reads. The
queried node runs no agent thread.

**Data plane (optional).** `radix-server` holds a shared-memory SlotStore that stores KV blocks in FULL / SWA /
Mamba pools, and uses Mooncake or NIXL to pull blocks hit on a peer into local slots and publish them to the
local index. Without the data plane, RadixShmem provides the index only, and slot ids are interpreted by the
caller's own storage.

**Server and client.** `radix-server` starts with deployment parameters only (node identity, cluster, NICs,
SlotStore budget). Tokens per block and slot bytes per pool are supplied by a client through `Geometry`; the
first node in the cluster to receive a `Geometry` publishes it to etcd and the other nodes adopt it. Clients
talk to the local server over gRPC (a unix socket by default) and read and write the index directly in shared
memory.

- `radixshmem.RadixClient`: Python client for query, insert, and KV block pulls (`pull_async`).
- `radix-server` (`radixshmem.RadixServer`): one server process per node, holding the index, the SlotStore, and
  the transfer engine.

## Framework support

The integrations below are complete or in progress.

- sglang: HiCache L3 storage backend (`HiCacheStorage`), in progress.
- vLLM: KV offload backend, in progress.

## Using RadixShmem

### Build

Dependencies: C++17 compiler, CMake, [xxHash](https://github.com/Cyan4973/xxHash), POSIX shared memory
(`librt`), pthreads, pybind11; multi-node deployments also need `libibverbs` and etcd.

```bash
cd python && pip install -e .
```

`pip install` runs CMake to build `radixshmem._core` and `radixshmem._data` under `build/`. RDMA is enabled
when the `libibverbs` headers are found; set `RADIXSHMEM_ENABLE_RDMA=0/1` to choose explicitly.

The data-plane transfer backends are optional: `scripts/build_mooncake.sh` and `scripts/build_nixl.sh` build
Mooncake / NIXL from upstream source at a pinned commit and install them under `third_party/`, where CMake
looks by default.

### Start the server

Each node runs one `radix-server`; its gRPC endpoint defaults to `unix:///dev/shm/<name>.sock`. The server
stays in `waiting` until it receives a `Geometry`, then creates its shared-memory regions, joins the cluster,
and becomes `ready`. All nodes share the same slot shapes; slot counts follow each node's budget.

```bash
# Same command on every node except --node-name and --rpc-address; ranks are assigned by etcd
radix-server --name /my_kv_cache --data-bytes 64G \
    --expected-min-nodes 3 --registry etcd://10.0.0.1:2379 \
    --node-name node0 --rpc-address 10.0.0.10 --transfer-dev mlx5_0
```

`--data-bytes` enables the data plane and requires the selected transfer backend to be built (default
`--transfer-backend mooncake`); without it the server builds the index only, and slot ids are interpreted by
the client's own storage. When neither `--expected-min-nodes` nor `--num-rht-shards` is greater than 1, the
server runs in single-node mode and needs neither etcd nor libibverbs.

Cluster parameters (one-to-one with the fields of Python `ServerConfig.cluster`, i.e. `ClusterConfig`):

| Flag | Default | Description |
|------|---------|-------------|
| `--expected-min-nodes` | `0` | Membership threshold: bootstrap starts once the nodes registered in etcd reach `max(--expected-min-nodes, --num-rht-shards)` and hold for `--settle-ms`; greater than 1 enables cluster mode |
| `--settle-ms` | `500` | Membership settle window (ms) |
| `--registry` | `etcd://127.0.0.1:2379` | etcd shared by the cluster; also holds `geometry/<node>` and `data/<node>` |
| `--cluster-id` | `default` | etcd key prefix `radix/<cluster_id>/...`, separating clusters on the same etcd |
| `--node-name` | auto | Unique node identity in etcd (auto = `node<bind-ip>`); the index shm name and default gRPC socket carry a `_<node_name>` suffix |
| `--rpc-address` / `--rpc-interface` | one required | Bootstrap address, given as an IP or resolved from a NIC name (NIC wins); `0.0.0.0` only for simulating multiple nodes on one host |
| `--rpc-port` | `0` | Bootstrap / XRC listen port; 0 = assigned by the OS |
| `--index-dev` | first available | HCA for the index control plane, independent of the data plane's `--transfer-dev` |
| `--gid-idx` | `3` | RoCE GID index (3 = RoCE v2 over IPv4) |
| `--bootstrap-timeout` | `60` | Bootstrap timeout (s); bootstrap is triggered by `configure` and waits for other nodes in the background |
| `--lease-ttl` | `30` | etcd lease TTL; a crashed node's keys expire after it, and the cluster `/config` and ranks expire after the last node exits |
| `--rht-transport` | `xrc` | RDMA transport from clients to RHT shards, `xrc` or `dc` |
| `--peer-index-transport` | `xrc` | RDMA transport for reading other nodes' index trees, `xrc` or `dc`, configured independently of the RHT |
| `--num-rht-shards` | `0` | Number of RHT shards; 0 = one per node |
| `--rht-shard-holders` | first ranks | Ranks holding shards, comma-separated |
| `--rht-slots` | `1` | RHT slots per bucket: 1 / 2 / 4 / 8 |
| `--enable-remote-insert` / `--enable-remote-query` | off | Enable remote insert / query (`remote_*` API); no effect in single-node mode |
| `--zmq-listen-port` | `0` | TCP port for remote ops; 0 = assigned by the OS |
| `--remote-op-transport` | `zmq` | Remote-op backend, `zmq` or `dc` (`dc` requires mlx5) |

`--rht-transport`, `--peer-index-transport`, `--num-rht-shards`, `--rht-shard-holders`, `--rht-slots`, and
`--remote-op-transport` take effect from rank 0 only and are distributed to the other nodes through etcd
`/config`. After bootstrap, rank and world size are available as `NodeInfo.rank` / `NodeInfo.world_size`.

Index and service parameters:

| Flag | Default | Description |
|------|---------|-------------|
| `--max-nodes` | auto | Radix node pool capacity |
| `--register-chunk-tokens` | `4096` | RHT registration granularity (tokens); a `Geometry` value takes precedence |
| `--data-pool-ratio` | `4.0` | Index DataPool size = FULL slots × ratio × bytes per block (12, or 16 with SWA) |
| `--hugepage-path` | plain shm | hugetlbfs mount shared by the index and the slot store |
| `--endpoint` | `unix:///dev/shm/<name>.sock` | gRPC endpoint; TCP as `host:port` |
| `--rpc-workers` | `32` | gRPC threads; each client with an unfinished job or in `wait_ready` occupies one |

Data-plane parameters (`--data-bytes` enables data mode; the others require it):

| Flag | Default | Description |
|------|---------|-------------|
| `--data-bytes` | none | Slot store budget (e.g. `64G`, `512M`); unset = index only |
| `--swa-ratio` / `--mamba-ratio` | `0` | Share of the budget for the SWA / Mamba pools; the rest goes to FULL |
| `--slot-align` | `4096` | Slot stride alignment; a `Geometry` value takes precedence |
| `--data-name` | `<name>_data` | SlotStore shm name |
| `--no-prefault` | off | Do not pre-touch slot pages at creation |
| `--transfer-backend` | `mooncake` | `mooncake` or `nixl` (`nixl` requires a non-zero `--transfer-port`) |
| `--transfer-dev` | all | HCA allowlist for the data plane, repeatable or comma-separated |
| `--transfer-ip` / `--transfer-port` / `--transfer-protocol` | rpc address / `0` / `rdma` | Data-plane segment address, port, and protocol (`rdma` or `tcp`) |
| `--max-inflight` / `--max-pending-jobs` / `--job-ttl` | `256` / `4096` / `60` | Maximum concurrent transfer batches; maximum pending jobs before `Submit` is rejected; seconds an uncollected job is kept |

Server parameters set the slot store's total bytes and their split across pools; `Geometry` sets tokens per
block and slot bytes per pool. The server computes slot counts: `swa_slots = floor(swa_ratio × data_bytes /
swa_stride)`, likewise for Mamba, with the remaining budget going to FULL. `configure` raises `ValueError`
when the model needs a pool whose ratio is 0, when the budget cannot fit one slot, or when a `Geometry` in
data mode specifies slot counts. It raises `GeometryMismatch` when the `Geometry` differs from the one the
server or cluster has already adopted. In index-only mode, slot counts come from the `Geometry`
(`full_slots`, etc.) and byte sizes are ignored.

Clients adopt the `Geometry` published by the server (`client.geometry`, including the computed slot
counts); an inference engine should check its own page size and bytes per block against it before use. Name
the server after the deployment it serves (e.g. `--name /kv_llama3_tp2`); different models should not share
a server.

The server prints statistics periodically (node count, block count, data pool utilization, mempool usage,
insert success/failure counts) and exits on `Ctrl+C` / `SIGTERM`.

### Connect clients

Clients connect to the local server by name (`RadixClient("/my_kv_cache")`; use `endpoint=` for TCP or
another socket), in any number, including worker processes started independently of the server. The
constructor only issues a gRPC call and returns immediately, so a client can be constructed before the server
is ready. `RadixClient(name, geometry)` also submits the `Geometry` to the server; this is idempotent. Then
call `client.wait_ready(timeout_s)` to block, or poll `client.status()`.

The index and the SlotStore are attached on the first index or data call; if the server is not ready yet,
that call raises `ServerNotReady`. `insert()` writes route updates asynchronously to RHT shards on other
nodes; `query()` issues RDMA one-sided reads when a prefix lives on another node; `pull_async()` pulls blocks
hit on a peer into local slots (the server performs the transfer) and publishes them to the local index. In
index-only mode, `pull_async()` raises `DataPlaneUnavailable`.

```python
import radixshmem

client = radixshmem.RadixClient("/my_kv_cache", radixshmem.Geometry(block_size=16, full_slot_bytes=1179648))
client.wait_ready(timeout_s=600)
assert client.geometry["block_size"] == 16 and client.geometry["pools"]["full"]["slot_bytes"] >= page_bytes
job = client.pull_async(hashes)         # returns immediately; the server performs the transfer
r = job.wait()                         # on return, [0, r.common_hit) is in local slots
for slot in r.full_slots:
    client.slot_view(slot)             # writable memoryview of the slot
r.finalize()                           # release the pin taken by pull_async(lock=True)
```

The full API (methods, parameters, result fields, error codes) is in [`docs/api.md`](docs/api.md).

## License

RadixShmem is released under the Apache License 2.0; the full text is in [LICENSE](LICENSE). Source files
carry an SPDX Apache-2.0 identifier and the NVIDIA copyright notice.
