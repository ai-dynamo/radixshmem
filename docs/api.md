# RadixShmem API Reference

RadixShmem keeps each node's KV cache index, a radix tree of block hashes, in shared memory. The C++ API has two classes: `RadixServer` creates the tree and, in a cluster, joins the other nodes through etcd; `RadixClient` attaches to the tree from any process on the node and runs inserts, queries, slot allocation and operations on other nodes' trees. Headers: `include/radix_server.h`, `include/radix_client.h`, `include/types.h`.

The Python package runs one server process per node, which holds the index, the KV slots and the transfer engine (`radixshmem.RadixServer`, started by `radix-server`), and a client for it (`radixshmem.RadixClient`). `radixshmem._core` exposes the C++ classes.

## Contents

- [1. Configuration](#1-configuration)
  - [1.1 `ShmConfig`](#11-shmconfig)
  - [1.2 `RadixServerConfig`](#12-radixserverconfig)
  - [1.3 `RadixClientConfig`](#13-radixclientconfig)
  - [1.4 Environment variables](#14-environment-variables)
- [2. `RadixServer`](#2-radixserver)
- [3. `RadixClient`](#3-radixclient)
  - [3.1 Construction](#31-construction)
  - [3.2 `insert`](#32-insert)
  - [3.3 `query`](#33-query)
  - [3.4 Slots, `reset` and `flush`](#34-slots-reset-and-flush)
  - [3.5 Remote operations](#35-remote-operations)
  - [3.6 Accessors and `compute_hashes`](#36-accessors-and-compute_hashes)
  - [3.7 `check_health`](#37-check_health)
- [4. Types and enums](#4-types-and-enums)
- [5. Python](#5-python)
  - [5.1 `radixshmem.RadixServer`](#51-radixshmemradixserver)
  - [5.2 `radixshmem.RadixClient`](#52-radixshmemradixclient)
  - [5.3 Write-back](#53-write-back)
  - [5.4 Transfer backends](#54-transfer-backends)
  - [5.5 `radixshmem._core`](#55-radixshmem_core)

## 1. Configuration

### 1.1 `ShmConfig`

The layout of one tree, fixed when the server creates it; processes that attach read it from the shm.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `max_nodes` | `uint32_t` | `0` | Tree node capacity; `0` = derived from `max_blocks` and `block_size`. |
| `max_blocks` | `uint32_t` | `0` | FULL slot pool capacity. `0` = no pool: `allocate_slots` returns 0 and slot ids are the caller's. |
| `block_size` | `int` | `1` | Tokens per block. |
| `register_chunk_tokens` | `int` | `4096` | Node alignment and RHT registration stride, in tokens. In blocks (`effective_register_chunk_size()`: at least 1, `0` = unaligned) it is also the longest node, at most 4096. |
| `background_evict_ratio` | `float` | `0.05` | Free share that background eviction keeps; `0` = off. `allocate_slots` evicts on demand either way. |
| `evict_policy` | `EvictPolicy` | `LRU` | Only `LRU`. |
| `data_pool_ratio` | `float` | `4.0` | Node data pool size: `max_blocks` × ratio × 12 bytes (16 with SWA). |
| `hugepage_path` | `std::string` | `"/dev/hugepages"` | hugetlbfs mount tried first; POSIX shm otherwise. |
| `component_mask` | `uint32_t` | `COMPONENT_MASK_FULL` | Enabled components, an OR of `COMPONENT_MASK_*`. |
| `swa_window_blocks` | `uint32_t` | `0` | SWA window W, in blocks; > 0 when SWA is enabled. |
| `mamba_state_bytes` | `uint64_t` | `0` | Bytes of one Mamba checkpoint, equal to the SlotStore's Mamba slot size. |
| `swa_max_blocks` | `uint32_t` | `0` | SWA slot pool capacity. |
| `mamba_max_slots` | `uint32_t` | `0` | Mamba slot pool capacity. |

### 1.2 `RadixServerConfig`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | `std::string` | `"/radixshmem"` | shm name; in a cluster `_<node_name>` is appended. |
| `shm` | `ShmConfig` | `{}` | See [1.1](#11-shmconfig). |
| `rank` | `int` | `0` | Output: the rank etcd assigned, set by `bootstrap()`. |
| `world_size` | `int` | `1` | Output: the cluster's node count after `bootstrap()`. On input, > 1 selects cluster mode. |
| `registry` | `std::string` | `"etcd://127.0.0.1:2379"` | etcd endpoint. |
| `cluster_id` | `std::string` | `"default"` | etcd key namespace, `radix/<cluster_id>/`. With the Redis RHT it may not contain `{` or `}`. |
| `node_name` | `std::string` | `""` | The node's identity in etcd; empty = `node<bind IP>`. |
| `rpc_address` | `std::string` | `""` | IPv4 address published to peers; `"0.0.0.0"` for tests on one host. |
| `rpc_interface` | `std::string` | `""` | NIC whose IPv4 address to publish; takes precedence over `rpc_address`. |
| `rpc_port` | `uint16_t` | `0` | Port of the gRPC endpoint peers reach (remote operations, the grpc peer index, RDMA connection setup); `0` = chosen by the OS. A single-node server starts no endpoint. |
| `expected_min_nodes` | `int` | `0` | `bootstrap()` waits until this many nodes (with the RDMA RHT, at least `RADIXSHMEM_RHT_SHARDS`) have registered and stayed unchanged for `settle_ms`. > 1 selects cluster mode. |
| `settle_ms` | `int` | `500` | See `expected_min_nodes`. |
| `bootstrap_timeout_sec` | `int` | `60` | Timeout of the etcd rendezvous and connection setup. |
| `lease_ttl_sec` | `int` | `30` | TTL of the node's etcd lease (at least 5): how long the keys of a killed node, or the `/config` of a stopped cluster, outlive it. |
| `register_chunk_tokens` | `int` | `4096` | Copied into `shm.register_chunk_tokens` by `bootstrap()`. |
| `writeback_enabled` | `bool` | `false` | Background eviction holds evicted slots for write-back; see [write-back queue](#write-back-queue). |
| `writeback_max_held_ratio` | `float` | `0.05` | Background eviction pauses while unfinished write-backs hold this share of a pool. |

Cluster mode is `expected_min_nodes > 1 || world_size > 1`. It needs `rpc_address` or `rpc_interface`; without them `bootstrap()` throws `std::runtime_error`. `RADIXSHMEM_REGISTRY`, `RADIXSHMEM_CLUSTER_ID` and `RADIXSHMEM_NODE_NAME` override `registry`, `cluster_id` and `node_name`.

### 1.3 `RadixClientConfig`

C++ only. `RadixClient(const RadixClientConfig&)` finds its node's shm through etcd instead of by name.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `registry` | `std::string` | `"etcd://127.0.0.1:2379"` | As the server's. |
| `cluster_id` | `std::string` | `"default"` | As the server's. |
| `node_name` | `std::string` | `""` | Node whose shm to attach; the client takes its rank. Must be in `/peers` unless `no_local_tree`. |
| `bootstrap_timeout_sec` | `int` | `60` | Timeout of the etcd lookups and connection setup. |
| `rht_only` | `bool` | `false` | Join the RHT only, without the peer index (the server's own clients). |
| `no_local_tree` | `bool` | `false` | Observer: attach no shm; `rank()` is `-1` and queries see remote trees only. |
| `expected_min_nodes` | `int` | `0` | Wait until this many nodes are ready. |
| `settle_ms`, `register_chunk_tokens` | `int` | | Not used; the cluster's values come from etcd. |

Missing `/config`, an unknown `node_name`, a membership timeout or a failed connection throw `std::runtime_error`.

### 1.4 Environment variables

The RHT (radix hash table) maps a prefix to the nodes that hold it; a query that finds more on another node continues there through the peer index. Rank 0's values of the RHT, peer index and RDMA QoS variables (`RADIXSHMEM_RHT_*`, `RADIXSHMEM_REDIS_ADDRS`, `RADIXSHMEM_REDIS_TIMEOUT_MS`, `RADIXSHMEM_PEER_INDEX_BACKEND`, `RADIXSHMEM_PEER_INDEX_RDMA_TRANSPORT`, `RADIXSHMEM_RDMA_SL`, `RADIXSHMEM_RDMA_TC`) apply to the whole cluster: other servers' values are ignored with a warning, and clients need none. A node that joins a running cluster with a different RHT or peer index setting is refused. With the defaults, the Redis RHT and the grpc peer index, the index uses no RDMA device.

| Variable | Default | Meaning |
|---|---|---|
| `RADIXSHMEM_RHT_BACKEND` | `redis` | `redis`: routes in a Redis Cluster ([deployment](redis_deployment.md)). `rdma`: in shards on holder ranks, accessed with one-sided RDMA. |
| `RADIXSHMEM_REDIS_ADDRS` | | redis, required: `host:port,...` of nodes of the Redis Cluster. |
| `RADIXSHMEM_REDIS_PASSWORD` | | redis: `AUTH` password. Every server and client process needs it; it is not stored in etcd. |
| `RADIXSHMEM_REDIS_TIMEOUT_MS` | `50` | redis: deadline of one batch of lookups. A master that misses it is skipped, and writes to it are held, until it answers again. |
| `RADIXSHMEM_RHT_SHARDS` | `0` | rdma: number of shards, `0` = one per node. Bootstrap waits for that many nodes. |
| `RADIXSHMEM_RHT_SHARD_HOLDERS` | ranks `0` to shards − 1 | rdma: ranks that hold the shards, e.g. `0,2,4,6`. |
| `RADIXSHMEM_RHT_SLOTS` | `4` | rdma: slots per bucket: 1, 2, 4 or 8. |
| `RADIXSHMEM_RHT_TRANSPORT` | `xrc` | rdma: `xrc` or `dc`. |
| `RADIXSHMEM_PEER_INDEX_BACKEND` | `grpc` | `grpc`: the peer matches the hashes in one RPC. `rdma`: the caller reads the peer's tree with one-sided RDMA. |
| `RADIXSHMEM_PEER_INDEX_RDMA_TRANSPORT` | `xrc` | rdma peer index: `xrc` or `dc`. |
| `RADIXSHMEM_PEER_INDEX_TIMEOUT_MS` | `500` | grpc peer index: deadline of one query or release; read by each client process. |
| `RADIXSHMEM_INDEX_DEV` | first device | RDMA device of the planes that run on RDMA. A client without it uses the device its server published. |
| `RADIXSHMEM_INDEX_GID_IDX` | `3` | GID index on that device (3 = RoCE v2 over IPv4). |
| `RADIXSHMEM_RDMA_SL`, `RADIXSHMEM_RDMA_TC` | `5`, `160` | Service level and traffic class of the index's RDMA traffic. |
| `RADIXSHMEM_LAZY_CONNECT` | `0` | `1`: connect to a peer over RDMA on first use instead of at startup. |
| `RADIXSHMEM_REMOTE_OP_TIMEOUT_MS` | `5000` | Client deadline of one remote operation. |
| `RADIXSHMEM_REMOTE_OP_MAX_THREADS` | `256` | Server threads for remote operations; a request beyond them gets `RESOURCE_EXHAUSTED`. |
| `RADIXSHMEM_DISABLE_AUTO_DEDUP` | `0` | `1`: clients send no dedup request after a pull, and servers drop nothing on one. |
| `RADIXSHMEM_DEDUP_TAIL_BLOCKS` | `256` | `tail_cap` of the dedup requests a pull sends. |
| `RADIXSHMEM_DEDUP_BRANCH_BLOCKS` | `16` | `branch_cap` of those requests. |
| `RADIXSHMEM_DISABLE_DEDUP_TAIL_BRANCH_MIGRATION` | `0` | `1`: those requests move no blocks to the puller. |
| `RADIXSHMEM_TRANSFER_PLUGIN_DIR` | directory of `_core` | Where the Mooncake transfer plugin is loaded from. |

## 2. `RadixServer`

One per node. It owns the tree and, in a cluster, the node's membership. Inserts and queries go through a `RadixClient`; in the same process, `RadixClient(server)`.

```cpp
explicit RadixServer(const RadixServerConfig& cfg);
RadixServer(const std::string& name, const ShmConfig& cfg = ShmConfig{});
bool bootstrap();
```

The first constructor only stores `cfg`. `bootstrap()` creates the shm, starts background eviction and, in cluster mode, registers the node in etcd, waits for the other nodes (`expected_min_nodes`), takes a rank and starts the endpoint peers reach. It blocks until the node is ready and returns `false` on failure (etcd unreachable, membership timeout, RDMA failure); the server is then unusable. The second constructor creates a single-node server and bootstraps it. All nodes and clients of a cluster must run the same version.

The destructor stops background eviction, leaves the cluster and unlinks the shm; attached processes keep their mapping until they detach.

| Method | Returns |
|---|---|
| `is_distributed()` | `world_size() > 1`. |
| `rank()`, `world_size()` | The node's rank and the cluster's node count. |
| `shm_name()` | Actual shm name; empty before `bootstrap()`. |
| `node_name()` | etcd identity resolved by `bootstrap()`. |
| `lease_id()` | etcd lease of the node's keys; `0` on a single node. |
| `total_radix_nodes()`, `total_blocks()` | Tree nodes in use, blocks in the tree. |
| `data_pool_utilization()` | Node data pool use, `[0, 1]`. |
| `mempool_total()`, `mempool_free()`, `mempool_used()` | FULL pool slots. |
| `swa_mempool_total()`, `swa_mempool_free()`, `mamba_mempool_total()`, `mamba_mempool_free()` | SWA and Mamba pool slots. |
| `component_config()` | `ComponentConfig` of the tree: `component_mask`, `logical_block_tokens`, `swa_window_blocks`, `mamba_state_bytes`. |
| `insert_success_count()` | Successful inserts. |
| `insert_error_node_pool_count()`, `insert_error_data_pool_count()`, `insert_error_ht_pool_count()` | Inserts that ran out of tree nodes, node data, child table entries. |
| `peer_faults()` | Index operations lost to unreachable peers, for the whole node. |
| `check_health(HealthReport* report = nullptr)` | As [`RadixClient::check_health`](#37-check_health), on the server's tree; `HEALTH_NOT_ATTACHED` before `bootstrap()`. |
| `writeback()` | The write-back queue; `nullptr` without `writeback_enabled`. |

The statistics are the node's own, read without locks.

### Write-back queue

```cpp
WritebackQueue* writeback();
std::vector<WritebackTask> WritebackQueue::pop(size_t max_tasks, int timeout_ms);
size_t WritebackQueue::complete(const std::vector<uint64_t>& ids);
void   WritebackQueue::drop_all();
void   WritebackQueue::set_active(bool on);
bool   WritebackQueue::active() const;
WritebackStats WritebackQueue::stats() const;   // queued, popped, produced, completed, dropped, inflight_slots[3]
```

While `active()` is false (the initial state), background eviction recycles what it evicts. While it is true, each evicted chain, a leaf and the ancestors it leaves childless and unpinned, becomes a `WritebackTask{id, HeldEviction h}` for its root path `[0, h.end())`: `hashes`, `full_slots`, `swa_slots` (`INVALID_COMPONENT_SLOT` where none), `mamba_ends` and `mamba_slots`, `evicted_from`, `generation`. `[evicted_from, end)` has left the tree and `[0, evicted_from)` stays pinned; no slot of the path is recycled until `complete()` or `drop_all()` releases the task. Background eviction pauses while held slots reach `writeback_max_held_ratio` of a pool. `reset()` releases every task, popped ones included: after reading a task's slots, compare `server.tree().reset_generation()` with `h.generation` and store the data only if they are equal. Eviction inside `allocate_slots` never produces tasks.

## 3. `RadixClient`

### 3.1 Construction

```cpp
explicit RadixClient(const std::string& name, bool local_read_only = false);
explicit RadixClient(const RadixClientConfig& cfg);
explicit RadixClient(RadixServer& server);
```

- `name`: attaches the shm by name. If the server put it in a cluster, the client joins the cluster too, with the etcd settings stored in the shm.
- `local_read_only = true`: attaches the tree and never joins the cluster, for processes that only read. `query()` is always local, `is_distributed()` is false, `world_size()` is 1, `rank()` is the server's rank, and `insert`, `allocate_slots`, `recycle_slots` and `reset` throw `std::runtime_error`.
- `cfg`: finds the shm through etcd ([1.3](#13-radixclientconfig)).
- `server`: same as `RadixClient(server.shm_name())`.

A failure to join the cluster throws `std::runtime_error`. The destructor detaches; it does not unlink the shm.

### 3.2 `insert`

```cpp
InsertResult insert(const uint64_t* hashes, int n,
                    const int32_t* slots, int num_slots,
                    int start = 0, bool auto_recycle = true,
                    ComponentType component = ComponentType::FULL);
```

Publishes one component of the prefix `hashes[0, n)`. The local tree is updated before `insert` returns; in a cluster the routes go to the RHT asynchronously (see [`flush`](#34-slots-reset-and-flush)). `insert` never evicts.

**FULL** (default): `slots[i]` holds block `hashes[start + i]`. With `matched` the prefix the tree already holds:

| Case | Result |
|---|---|
| `matched < start` | Nothing is inserted and every slot is returned in `unused_slots`; `error` is still `OK`. |
| `matched == start` | The slots are placed on `hashes[start, start + num_slots)`. |
| `matched > start` | The first `matched - start` slots are skipped (another writer inserted those blocks) and returned in `unused_slots`. |

Surplus slots are returned too; with too few slots, the blocks past the last one are not inserted. Typical use:

```cpp
int matched = client.query(hashes, n, COMPONENT_MASK_FULL, /*local_only=*/true).common_hit;
std::vector<int32_t> slots;
if (client.allocate_slots(n - matched, slots) > 0) {
    // write the KV of hashes[matched, matched + slots.size()) into slots
    client.insert(hashes, n, slots.data(), (int)slots.size(), /*start=*/matched);
}
```

**SWA**: `slots` cover the tail `hashes[n - num_slots, n)`, `1 ≤ num_slots ≤ n`, `start = 0`. A block that already has an SWA slot keeps it, and the new slot is returned. The window W does not limit what is published: a query hits at `n` only once every block of `[max(0, n - W), n)` has an SWA slot.

**MAMBA**: one slot, the checkpoint at `n`, `start = 0`. If `n` falls inside a node, the node is split there; an existing checkpoint is kept. A checkpoint stays until the Mamba pool evicts it.

SWA and MAMBA slots must come from the matching pool, and the FULL path `[0, n)` must be in the tree.

| `error` | When |
|---|---|
| `OK` | Published, including when every block was already there. |
| `NODE_POOL_FULL`, `DATA_POOL_FULL`, `HT_POOL_FULL` | The tree ran out of nodes, node data or child table entries; what the call inserted is rolled back. |
| `BAD_REQUEST` | SWA or MAMBA: component not enabled, `num_slots` out of range, or `start != 0`. |
| `FULL_PATH_MISSING` | SWA or MAMBA: the FULL path `[0, n)` is not in the tree. |

Every input slot is either taken by the tree or returned in `unused_slots`. With `auto_recycle` the call recycles them itself; otherwise the caller does.

### 3.3 `query`

```cpp
ComponentQueryResult query(const uint64_t* hashes, int n,
                           uint32_t component_mask = COMPONENT_MASK_FULL,
                           bool local_only  = false,
                           bool lock        = false,
                           bool update_meta = true);
```

Finds `common_hit`, the longest prefix of `hashes[0, n)` that every component in `component_mask` accepts. `n == 0` is an `OK` miss.

| Parameter | Meaning |
|---|---|
| `component_mask` | A non-empty OR of `COMPONENT_MASK_*` within the enabled components, else `BAD_REQUEST`. A `ComponentType` is not a mask. |
| `local_only` | `false`: in a cluster, the query may continue on one other node. |
| `lock` | Pin the hit against eviction until `result.finalize()`. |
| `update_meta` | Refresh the LRU recency of the hit. |

What each component accepts as a boundary `L`:

- FULL: any block boundary of the matched path.
- SWA: `L` when every block of `[max(0, L - W), L)` has an SWA slot.
- MAMBA: a node end with a checkpoint, exactly at `L`.

**Other nodes.** With `local_only = false` the client looks up the node with the longest route in the RHT. If that extends the local hit, the query continues there through the peer index, and the result is the local prefix plus that node's suffix: `full_fragments` then has two entries, each with its `source_rank`, and `swa` and `mamba` come from the remote node. If the remote part fails, the local result stands. An observer has no local prefix.

**Pins.** With `lock`, FULL pins every node that intersects `[0, f)`, where `f` is the matched FULL length, SWA pins the nodes that intersect its window, and MAMBA the node that ends at `common_hit`. A remote suffix is pinned on its node, and only when it extends the local hit. `finalize()` releases all of them; it runs once and later calls do nothing. Without it the nodes stay pinned. Call it before `reset()` or the client's destruction; the client must outlive it.

**LRU.** `update_meta` refreshes FULL recency up to `f` only when the mask includes FULL; SWA refreshes its window only. A remote node's LRU is never touched.

### 3.4 Slots, `reset` and `flush`

```cpp
int  allocate_slots(int num_slots, std::vector<int32_t>& out,
                    ComponentType component = ComponentType::FULL);
void recycle_slots(const int32_t* slots, int num,
                   ComponentType component = ComponentType::FULL);
void reset();
void flush();
```

`allocate_slots` takes free slots from the component's pool and evicts by LRU when there are too few. FULL returns how many it allocated, possibly fewer than `num_slots`. SWA and MAMBA allocate all or none: they return `num_slots`, or `0` with `out` empty. A disabled pool returns `0`; `num_slots ≤ 0` or more than the pool holds returns `-1`. Each pool numbers its slots on its own.

`recycle_slots` returns slots to the component's pool; a slot that is already free is ignored. Do not recycle slots the tree holds or that a transfer may still write.

`reset()` empties the node's tree, returning every slot to its pool, and removes the node's routes from the RHT: with Redis the routes of this rank (retried for up to 30 s), with RDMA the shards this rank holds. Each node resets only itself; to clear a cluster, every node calls it. Calls on one node run one at a time, and a call that waited for another one's reset returns without a second reset. Slot ids, results and `FinalizeFn`s obtained before the call are invalid after it.

`flush()` waits until the RHT writes of earlier inserts are done, so that other nodes' queries see them. On a single node it does nothing.

### 3.5 Remote operations

Remote operations act on another node's tree through its gRPC endpoint. The first argument is the target `rank`; the client keeps one session per rank.

- A bad rank (own, negative, out of range) gives `BAD_TARGET`, in `remote_status` for inserts; nothing throws.
- A rank without a `/peers` entry gives `UNAVAILABLE` without sending. So does a peer marked down after an `UNAVAILABLE` or a timeout, until it registers again or a backoff (1 s, doubling to 30 s) lets one request probe it.
- A request that reaches a restarted target is refused with `STALE_TARGET`.
- `TIMEOUT` and `UNAVAILABLE` do not tell whether the target ran the operation: do not assume that slots it may have allocated are free. `RESOURCE_EXHAUSTED` and `STALE_TARGET` mean it did not run. An allocation or query that waited past its deadline on the target is not run; inserts, recycles, releases and dedup requests always run.
- One operation carries at most `MAX_REMOTE_OP_BLOCKS` (134,217,471) blocks; more is `BAD_REQUEST`.

#### `remote_allocate`

```cpp
AllocateReply remote_allocate(int rank, int n, ComponentType component = ComponentType::FULL);
```

Allocates `n` slots from the target's pool, bound to no prefix: phase 1 of a remote write. `error` is `MEMPOOL_FULL`, with `slots` empty, when the pool cannot supply all `n`. The slots stay allocated until `remote_insert` takes them or `remote_recycle` returns them.

#### `remote_allocate_components`

```cpp
AllocateComponentsReply remote_allocate_components(int rank, const std::vector<ComponentCount>& components);
```

Phase 1 for several pools, all or none: `n` slots from the pool of each `{component, n}`, each component at most once. `slots[i]` belongs to `components[i]`. An empty list returns `OK` without a request.

#### `remote_insert`

```cpp
InsertResult remote_insert(int rank, const uint64_t* hashes, int n,
                           int start, int length, const int32_t* slots,
                           bool auto_recycle = true,
                           ComponentType component = ComponentType::FULL);
```

Phase 2: the target runs [`insert`](#32-insert) with the `length` slots for `hashes[start, start + length)` and publishes the routes; a new prefix uses `start = 0, length = n`. Blocks the target already holds take no slot. With `auto_recycle` the target recycles the unused slots; otherwise return them with `remote_recycle`. A failed request gives `error = REMOTE_FAILED` with the cause in `remote_status`.

#### `remote_insert_path`

```cpp
InsertPathResult remote_insert_path(int rank, const uint64_t* hashes, int n,
                                    int start, int length, const int32_t* full_slots,
                                    const std::vector<ComponentRun>& runs,
                                    bool auto_recycle = true);
```

Phase 2 for a whole path in one request: the FULL slots of `hashes[start, start + length)` (`length = 0`: none), then each run, `SWA` slots for `hashes[end - slots.size(), end)` or the `MAMBA` checkpoint at `end`. `full` and `runs[i]` are the results of the single inserts; a run whose path `[0, end)` the target lacks gets `FULL_PATH_MISSING`. A malformed request gives `remote_status = BAD_REQUEST` and changes nothing.

#### `remote_recycle`

```cpp
RemoteStatus remote_recycle(int rank, const int32_t* slots, int num_slots,
                            ComponentType component = ComponentType::FULL);
```

Returns unpublished slots to the target's pool. Do not repeat it after a failure: the slots may have been recycled and allocated again.

#### `remote_query`

```cpp
QueryResult remote_query(int rank, const uint64_t* hashes, int n,
                         bool lock = false, bool update_meta = true);
```

FULL query of the target's tree alone. With `lock` the target pins the hit, whole nodes, until `finalize()`; with `update_meta = false` its LRU is not touched.

#### `remote_dedup_prefix`

```cpp
DedupReply remote_dedup_prefix(int rank, const uint64_t* hashes, int n, const DedupRequest& req = {});
int dedup_move(int rank, const uint64_t* hashes, int n, int tail_cap, int branch_cap,
               const int landing[NUM_COMPONENT_TYPES]);
```

Tells `rank` that the caller holds `hashes[0, n)`. If the target's match ends at a leaf inside `[0, n)`, the target drops that leaf and its ancestors up to the first one with another child or a pin, and recycles their slots; `freed` counts the FULL blocks dropped. A target started with `RADIXSHMEM_DISABLE_AUTO_DEDUP=1` drops nothing.

With landing slots in `req` (free slots of the caller's pools; `req.rank` is the caller), the target also moves what keeps its copy alive into them: the subtree below the match if it has at most `tail_cap` blocks, the side subtrees of at most `branch_cap` blocks along the path, and the SWA cells and Mamba checkpoints of the dropped part. It stops at a pin, at a larger side subtree, or when the landing slots run out. `moved` lists the paths written, with their slots and ages; `unused` lists the landing slots nothing was written to. Other landing slots not in `moved` may still be written and must not be reused.

`dedup_move` is the caller's side: it takes `landing[k]` free slots of each pool without evicting, sends the request, publishes the moved paths in the local tree with their ages, and recycles the landing slots not used. It returns `freed`, and does nothing when the caller no longer holds `[0, n)`.

### 3.6 Accessors and `compute_hashes`

| Method | Returns |
|---|---|
| `is_distributed()` | Whether the client joined a cluster. |
| `rank()` | The node's rank; `0` on a single node, `-1` for an observer. |
| `world_size()` | The cluster's node count; `1` on a single node. |
| `name()` | Name of the attached shm. |
| `block_size()` | Tokens per block of the tree. |
| `total_radix_nodes()`, `total_blocks()`, `data_pool_utilization()` | As on the server. |
| `mempool_total()`, `mempool_free()`, `mempool_used()` | FULL pool slots. |
| `swa_mempool_total()`, `mamba_mempool_total()` | SWA and Mamba pool capacity; `0` when disabled. |
| `component_config()` | As on the server. |
| `peer_faults()` | As on the server; null without a local tree. |
| `dump_tree(uint32_t max_nodes = 0, uint64_t max_blocks = 0)` | A copy of the local tree for monitoring; `0` = no limit. |
| `print_tree()` | Prints the local tree. |

```cpp
static std::vector<uint64_t> compute_hashes(const int64_t* token_ids, int num_tokens, int tokens_per_block);
```

One cumulative hash per full block of `tokens_per_block` tokens; a partial last block is dropped. All clients of a cluster must use the same `tokens_per_block`.

### 3.7 `check_health`

```cpp
int check_health(HealthReport* report = nullptr) const;
```

Audits the local tree for held locks, slots allocated but neither inserted nor recycled, and pins never released. Returns `HealthFlag` bits, `0` when healthy; `report` receives the details. It takes no lock and writes nothing, so it returns even while another process holds a lock. Meant for a quiet tree, at the end of a test: with concurrent writers it can report false positives. Each node audits only its own tree; a remote reader that did not finish shows as `HEALTH_RDMA_LOCK_HELD` on the node it read. Slots held for write-back count as leaked.

| `HealthFlag` | Set when |
|---|---|
| `HEALTH_RWLOCK_READ_HELD` | A process holds the tree's read lock. |
| `HEALTH_RWLOCK_WRITE_HELD` | A process holds or waits for the write lock. |
| `HEALTH_LRU_LOCK_HELD` | The LRU mutex is locked. |
| `HEALTH_RDMA_LOCK_HELD` | The local writer flag or the remote reader count is set. |
| `HEALTH_FULL_SLOT_LEAK`, `HEALTH_SWA_SLOT_LEAK`, `HEALTH_MAMBA_SLOT_LEAK` | A pool's allocated slots differ from the slots in the tree. |
| `HEALTH_REF_LEAK` | A node still has a pin. |
| `HEALTH_NODE_LEAK` | The nodes reachable from the root differ from the node pool's count. |
| `HEALTH_NOT_ATTACHED` | No local tree: an observer, or a server before `bootstrap()`. |

| `HealthReport` field | Meaning |
|---|---|
| `flags` | The return value; `ok()` is `flags == 0`. |
| `locks_inspected` | `false` on a non-glibc libc, where the three pthread lock flags are not checked. |
| `rwlock_readers`, `rwlock_writers_waiting`, `rwlock_writer_held`, `lru_locked` | Lock state. |
| `rdma_rwlock` | Bit 63: local writer; low bits: remote readers. |
| `full`, `swa`, `mamba` | One `Pool` per slot pool: `enabled`, `total`, `used`, `in_tree`, `leaked` (allocated, not in the tree), `dangling` (in the tree, free or out of range), `duplicate` (in the tree more than once), `held` (held for write-back). |
| `nodes_with_refs`, `ref_total` | Pinned nodes and the sum of their pins. |
| `nodes_reachable`, `nodes_in_pool` | Nodes reachable from the root, and the node pool's count. |
| `describe()` | One line per failure; `"healthy\n"` when healthy. |

## 4. Types and enums

**`InsertResult`**

| Field | Meaning |
|---|---|
| `error` | `InsertError`. |
| `remote_status` | Remote inserts: the `RemoteStatus` behind `REMOTE_FAILED`. `OK` otherwise. |
| `unused_slots` | Slots the tree did not take; already recycled when `auto_recycle`. |
| `topology_changed` | The call split a node. |

**`ComponentQueryResult`**

| Field | Meaning |
|---|---|
| `status` | `RemoteStatus`; the other fields are valid only when `OK`. |
| `common_hit` | Blocks every requested component accepts. |
| `full_fragments` | FULL slots of `[0, common_hit)` as one or two `ComponentSlotFragment`s (local prefix, remote suffix), ordered by `offset`; empty without FULL or on a miss. |
| `swa` | SWA slots of `[max(0, common_hit - W), common_hit)`, from one node; `source_rank` is `-1` when there are none. |
| `mamba` | The checkpoint at `common_hit`, from the node of `swa`; empty when there is none. |
| `rdma_read_count`, `rdma_atomic_count` | One-sided RDMA operations of the remote part; `0` with the grpc peer index. |
| `finalize` | Releases the pins of `lock`; empty when nothing was pinned. |

```cpp
struct ComponentSlotFragment {    // slot ids are valid in source_rank's SlotStore
    int32_t source_rank = -1;
    int offset = 0;               // block position of slot_ids[0]
    std::vector<int32_t> slot_ids;
};
struct ComponentCheckpoint {
    int32_t source_rank = -1;
    int32_t slot_id = INVALID_COMPONENT_SLOT;
};
```

**`QueryResult`**, returned by `remote_query` only:

| Field | Meaning |
|---|---|
| `status` | `RemoteStatus`; the other fields are valid only when `OK`. |
| `total_hit_length`, `remote_hit_length` | FULL blocks of the prefix the target holds. |
| `remote_node_id` | The target rank. |
| `remote_slots` | The target's slot ids of those blocks. |
| `local_hit_length`, `prefix_slots` | Always `0` and empty. |
| `rdma_read_count`, `rdma_atomic_count` | Always `0`. |
| `finalize` | With `lock` and a hit: releases the target's pin. A release refused with `RESOURCE_EXHAUSTED` is retried. |

**`FinalizeFn`**

```cpp
class FinalizeFn {
public:
    void operator()();               // runs once; later calls do nothing
    explicit operator bool() const;  // armed
    bool empty() const;
};
```

Destroying an armed `FinalizeFn` does not run it.

**Other request and reply types**

- `AllocateReply{status, error, slots}`, `AllocateComponentsReply{status, error, slots}` (one vector per requested component).
- `InsertPathResult{remote_status, full, runs}`.
- `ComponentCount{component, n}`, `ComponentRun{component, end, slots}`.
- `DedupRequest{rank, tail_cap, branch_cap, landing[3]}`, `DedupReply{status, freed, moved, unused[3]}`.
- `MovedPath{base, hashes, full_from, full, full_age_ns, cells}`: the request's `[0, base)` plus `hashes`, with landing slots for the FULL blocks `[full_from, base + hashes.size())` and for the SWA cells and Mamba checkpoints in `cells`.

**Enums and constants**

```cpp
enum class InsertError : uint32_t {
    OK                = 0,
    NODE_POOL_FULL    = 1,  // out of tree nodes
    DATA_POOL_FULL    = 2,  // out of node data
    HT_POOL_FULL      = 3,  // out of child table entries
    MEMPOOL_FULL      = 4,  // remote_allocate*: the target's pool cannot supply the slots
    REMOTE_FAILED     = 5,  // remote inserts: see InsertResult::remote_status
    BAD_REQUEST       = 6,  // SWA / MAMBA publish: invalid arguments
    FULL_PATH_MISSING = 7,  // SWA / MAMBA publish: the FULL path [0, n) is missing
};

enum class RemoteStatus : uint16_t {
    OK                 = 0,
    BAD_REQUEST        = 1,   // invalid request
    UNKNOWN_OP         = 2,   // the target does not know the operation
    TIMEOUT            = 4,   // no reply before the deadline
    BAD_REPLY          = 5,   // malformed reply
    BAD_TARGET         = 6,   // own rank or out of range; not sent
    INTERNAL           = 7,   // internal error, or the target's handler failed
    UNAVAILABLE        = 8,   // target unreachable, not serving, or marked down
    RESOURCE_EXHAUSTED = 9,   // the target had no thread for it; not run
    STALE_TARGET       = 10,  // the target restarted; not run
};

enum class EvictPolicy : uint32_t { LRU = 0 };

enum class ComponentType : uint8_t { FULL = 0, SWA = 1, MAMBA = 2 };   // a pool, not a mask
static constexpr uint32_t COMPONENT_MASK_FULL  = 1u << 0;
static constexpr uint32_t COMPONENT_MASK_SWA   = 1u << 1;
static constexpr uint32_t COMPONENT_MASK_MAMBA = 1u << 2;
static constexpr int32_t  INVALID_COMPONENT_SLOT = -1;
```

`RemoteStatus` describes delivery; `InsertError` the result on the target's tree, meaningful only when the status is `OK`. The `HealthFlag` bits are in [3.7](#37-check_health).

## 5. Python

`radixshmem.build_info()` returns the package, shm and protocol versions, whether rdma-core loads, and the available transfer backends.

### 5.1 `radixshmem.RadixServer`

```python
from radixshmem import RadixServer, ServerConfig
server = RadixServer(ServerConfig(name="/rsh", data_bytes=64 << 30)).start()
```

The server process of a node: the index, the SlotStore that holds the KV slots, and the transfer engine, served over gRPC at `endpoint` (`unix:///dev/shm/<name>.sock` by default). `ServerConfig` holds the operator's settings, with the cluster settings in `ServerConfig.cluster` (a `ClusterConfig`); their fields correspond to the `radix-server` flags (`radix-server --help`), with `role="decode"` for `--decode`. `data_bytes = 0` makes the server index-only: it has no SlotStore, and slot ids belong to the client's own storage. `ClusterConfig` has the cluster fields of [`RadixServerConfig`](#12-radixserverconfig); the node joins a cluster when `expected_min_nodes > 1`.

The server goes from `starting` to `waiting` (no geometry yet), `configuring` and `ready`, and ends `closed`. The geometry comes from a client's `configure` or, in a data-mode cluster, from another node's. A failed configuration returns the server to `waiting`, with the reason in `last_error`.

| Member | Meaning |
|---|---|
| `start()` | Serves the endpoint; returns in `waiting`. |
| `configure(geometry)` | Takes a geometry and starts the bring-up; returns in `configuring`. The same geometry again is accepted; a different one raises `GeometryMismatch`, one the server cannot take `ValueError`. |
| `wait_ready(timeout_s)` | Blocks until `ready`. `TimeoutError` while waiting or configuring, `RuntimeError` after a failed configuration. |
| `describe()` | The `NodeInfo` ([5.2](#52-radixshmemradixclient)). |
| `nodes()` | Every live server of the cluster as a dict (`node_name`, `role`, `rank`, `mode`, `data_plane`, `endpoint`), ranked nodes first. |
| `stats()` | A dict of counters: transfers, jobs, the index and its pools, and `writeback` when write-back runs. |
| `close()` | Stops the server. |
| `mode`, `geometry`, `endpoint`, `index`, `writeback_handler` | Properties; `index` is the `_core.RadixServer`. |

### 5.2 `radixshmem.RadixClient`

```python
client = radixshmem.RadixClient("/rsh", radixshmem.Geometry(block_size=64, full_slot_bytes=slot_bytes))
client.wait_ready()
```

`RadixClient(name, geometry=None, *, endpoint=None, max_outstanding=1024, local_read_only=False)` connects to the server started with `name` and returns without waiting for it. With `geometry` it first calls `configure(geometry)`.

| Member | Meaning |
|---|---|
| `info`, `status()` | The server's `NodeInfo`; `status()` fetches it again. |
| `configure(geometry)`, `wait_ready(timeout_s=300)`, `nodes()` | As on the server. |
| `geometry` | The geometry the server published. |
| `index`, `store` | The `_core.RadixClient` and the SlotStore, attached on first use. `ServerNotReady` until the server is ready; `store` raises `DataPlaneUnavailable` on an index-only server. |
| `slot_view(slot, kind=FULL)` | The bytes of one slot. |
| `pull_async(...)`, `push_async(...)` | Below. |
| `wait(job, timeout=None)`, `wait_any(jobs, timeout=None)`, `cancel(job)` | Wait for one job, for the first of several, or give up a pull. |
| `close()` | Closes the client; pushes not finished complete with `complete=False`. |

Other attributes are those of `index`: `client.query(...)`, `client.insert(...)`, `client.allocate_slots(...)` and the rest of [5.5](#55-radixshmem_core). `local_read_only=True` attaches the index read-only, as in [3.1](#31-construction); `pull_async`, `push_async`, `insert`, `allocate_slots`, `recycle_slots` and `reset` then raise `RuntimeError`.

```python
pull_async(hashes, mask=COMPONENT_MASK_FULL, *, lock=True, timeout_ms=30000, block=True) -> PullJob
```

Queries the cluster, has the server copy the remote part into local slots, and publishes them in the local tree. `PullJob.wait(timeout=None)` returns `PullResult(common_hit, full_slots, swa_start, swa_slots, mamba_slot, remote_blocks, remote_bytes, source_rank, finalize)`, with local slots for `[0, common_hit)`. A failed transfer completes the job with the local hit; `wait()` does not raise for it. With `lock` the hit stays pinned until `finalize()`. At `max_outstanding` jobs the call blocks, or with `block=False` completes the job at once with the local hit. `PullJob.local_hit` and `planned_hit` are set when the call returns. `PullJob.cancel()` gives up the result; the transfer still completes and is published. After a pull that landed its planned hit, the client asks the nodes it read from to drop their copies (`dedup_move`, sent in the background; the dedup variables are in [1.4](#14-environment-variables)). Clients of a decode server send none.

```python
push_async(rank, hashes, full_slots, swa=(), mamba=(), *, valid=None, timeout_ms=30000, block=True) -> PushJob
```

Writes the path `hashes` from local slots to node `rank` and publishes it there. `swa`: `(end, slots)` runs covering `[end - len(slots), end)`; `mamba`: `(end, slot)` checkpoints; every `end` at most `len(hashes)`. The target keeps the prefix it already holds; the rest is reserved there with one `remote_allocate_components`, written, and published with one `remote_insert_path` if every write landed and `valid()`, when given, still returns True. `PushJob.wait()` returns `PushResult(common_hit, remote_hit, remote_blocks, remote_bytes, complete)` and never raises; `PushJob.no_space` is True when the target had no room. Keep the local slots until the job is done.

A job that gives up while its transfer is in flight leaves the destination slots allocated, local for a pull and on the target for a push, since the data may still land; `check_health` reports them as leaked.

`Geometry(block_size, full_slot_bytes=0, swa_slot_bytes=0, swa_window_blocks=0, mamba_slot_bytes=0, full_slots=0, swa_slots=0, mamba_slots=0, slot_align=0, register_chunk_tokens=0)` is the client's slot shape. A data-mode server takes the byte sizes and sizes the pools from `data_bytes`; an index-only server takes the counts. A pool exists when its bytes or its count is > 0; `slot_align` and `register_chunk_tokens` `0` mean the server's default. `radixshmem.check_pools_aligned(index, store)` checks that the index pools match the SlotStore pools.

`NodeInfo` fields: `mode`, `data_plane`, `index_name`, `data_name`, `hugepage_path`, `geometry`, `rank`, `world_size`, `node_name`, `segment`, `data_bytes`, `last_error`, `swa_ratio`, `mamba_ratio`, `role`.

Exceptions, all `RuntimeError`: `ServerNotReady` (with `mode` and `last_error`), `GeometryMismatch`, `DataPlaneUnavailable`, `ServerBusy` (the server's job table is full).

### 5.3 Write-back

A prefill server of a data-mode cluster with background eviction writes what background eviction takes to a ready decode node, with `radixshmem.writeback.PeerWriteback` ([write-back queue](#write-back-queue)); other servers run without write-back. `RadixServer.writeback_handler` is the handler, or None. A handler is a `radixshmem.WritebackHandler`:

| Method | Meaning |
|---|---|
| `start(client)` | Called once the server is ready, with an in-process `RadixClient`. |
| `active()` | While False, eviction recycles at once and hands nothing to `write`. |
| `write(tasks, done)` | Handles a batch of `WritebackTask`s; `done(task_ids)` releases them, from any thread. |
| `stop()` | Called first by `RadixServer.close()`, once `write` returns or after 5 s. |
| `stats()` | Counters added to `server.stats()["writeback"]`. |

`WritebackTask` has `task_id`, `hashes`, `full_slots`, `swa_slots`, `mamba_ends`, `mamba_slots`, `evicted_from`, `generation` and `end`. Data read from its slots may be stored only if `task.valid()` is still True after the read. `PeerWriteback` pushes each task with `push_async` to a decode node chosen by its first block hash and counts `pushed`, `incomplete`, `no_space`, `no_target` and `too_long`.

### 5.4 Transfer backends

The server moves slots through a `radixshmem.transfer.TransferBackend` chosen by `ServerConfig.transfer_backend`: `mooncake` (default; the C++ plugin behind `_core.NativeTransferBackend`) or `nixl` (NIXL's Python package, `pip install radixshmem[nixl]`). `available_backends()` lists the ones this installation can create; `create_transfer_backend(config)` creates one.

| `TransferBackend` member | Meaning |
|---|---|
| `name`, `local_segment` | Backend name; the segment name peers open this process's memory by. |
| `register_memory(addr, length, location="")`, `unregister_memory(addr)` | Register memory for transfers. |
| `open_segment(name)` | `(XFER_OK, SegmentInfo(id, base, length))` of a peer, or `(error, None)`. |
| `close_segment(seg_id)` | Closes it. |
| `submit(op, seg_id, local, remote, lengths)` | One batch of ranges, uint64 arrays of absolute addresses; a handle, or `INVALID_BATCH` if any range is refused. |
| `poll(handle)`, `wait(handle, timeout_ms)` | `(XferState, bytes done)`; the final `XferState`. |
| `release(handle)` | `XFER_OK`, or `XFER_EBUSY` while the batch runs. |
| `close()` | Closes the backend. |

`SlotTransfer(store, backend)` works in slots: `register_store()`, `attach_peer(segment, published_geometry)`, `read(peer, remote_slots, local_slots, kind=FULL)` and `write(peer, local_slots, remote_slots, kind=FULL)` returning `(handle, ranges)`, `poll`, `wait`, `release`, and `detach_peer(peer)`, which returns `XFER_EBUSY` while a batch to that peer is unreleased. Slots contiguous on both sides go in one range. `radixshmem._data` has the data-plane names of `_core` and `radixshmem.transfer`.

### 5.5 `radixshmem._core`

The C++ classes, with these differences:

- `RadixServer(config)`, and `RadixServer(name, ShmConfig)`, which bootstraps. `ShmConfig(...)` takes its fields as keywords; `RadixServerConfig()` is set by attribute (`cfg.rpc_port = 31001`). `RadixClientConfig` is not exported: the client constructors are `RadixClient(name, local_read_only=False)` and `RadixClient(server)`.
- Arrays are numpy: `uint64` hashes, `int32` slots, `int64` token ids. Result fields are read-only; slot vectors are `int32` arrays.
- `insert(hashes, slots, start=0, auto_recycle=True, *, component=ComponentType.FULL)`; `len(slots)` is `num_slots`.
- `query(hashes, mask=COMPONENT_MASK_FULL, local_only=False, lock=False, update_meta=True)`; a `ComponentType` as `mask` raises `TypeError`. `ComponentQueryResult` has `full_fragments` as `[(source_rank, offset, slots)]`, `swa_source_rank`, `swa_start`, `swa_slots`, and `mamba` as `None` or `(source_rank, slot_id)`.
- `finalize` is callable; `bool(finalize)` and `finalize.empty()` tell whether it is armed. A result keeps its client alive.
- `allocate_slots(num_slots, component=FULL)` returns an `int32` array, empty on an SWA or Mamba shortfall, and raises `ValueError` where C++ returns -1. `recycle_slots(slots, component=FULL)`.
- `remote_allocate(rank, n, component=FULL)`, `remote_allocate_components(rank, [(component, n), ...])`, `remote_insert(rank, hashes, start, length, slots, auto_recycle=True, component=FULL)`, `remote_insert_path(rank, hashes, start, full_slots, [(component, end, slots), ...], auto_recycle=True)`, `remote_recycle(rank, slots, component=FULL)`, `remote_query(rank, hashes, lock=False, update_meta=True)`, `remote_dedup_prefix(rank, hashes)` (a `DedupReply` with `status` and `freed`), `dedup_move(rank, hashes, tail_cap, branch_cap, landing)`.
- `InsertResult` has `error`, `remote_status` and `unused_slots`.
- `check_health()` and `health_report()` on both classes; `HealthFlag` is an arithmetic enum (`int(HealthFlag.FULL_SLOT_LEAK)`). `peer_fault_counts()` returns a dict, `dump_tree(max_nodes=0, max_blocks=0)` a dict of numpy arrays.
- `RadixServer` adds `pop_writeback`, `complete_writeback`, `drop_writeback`, `set_writeback_active`, `writeback_stats` and `reset_generation` for the write-back queue.
- `compute_hashes(token_ids, tokens_per_block)`, also as `radixshmem.compute_hashes`.
- `EtcdKv(registry)`: `put`, `create` (put if absent), `get`, `get_prefix`, `count`, `delete`, `grant_lease`, `keep_alive`, `keep_alive_stop`, `revoke`.
- Calls release the GIL while they run in C++.
