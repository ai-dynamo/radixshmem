# RadixShmem API Reference

This document is for users who call RadixShmem directly (C++ or Python). It centers on two main classes: `RadixServer` and `RadixClient`. In addition, when the remote-op backend is compiled in, a set of companion result types appears (`AllocateReply`, `RemoteStatus`; see [§3.2 Remote operations](#32-datapath-insert--query--remote-ops)). Remote operations are all rank-first and handle-free, and failures are always reported through `RemoteStatus`. The relationship between the two main classes is as follows:

- `RadixServer`: one instance per node. It holds the entire radix tree in POSIX shared memory and can optionally start a distributed cluster (server-side: registers the RDMA MR, runs **etcd rendezvous** (writes itself into `/peers`, obtains a rank, reads `/config`), starts the XRC accept listener, and stamps the node identity + ready flag into the shm header). `RadixServer` itself does not expose a datapath interface (insert/query, etc.); it is only responsible for lifecycle + statistics.
- `RadixClient`: any process can construct one; it attaches to a shared memory region created by `RadixServer`. `RadixClient` handles the entire datapath (insert/query/...). If the attached shm is marked distributed by the server (`is_distributed` in the header), the client reads `/config` + `/peers` from **etcd**, starts its own RDMA context, and builds XRC / DC endpoints to the RHT holders according to the authoritative `rht_transport` and to the peers according to `peer_index_transport`, so it can independently write the cross-node RHT and independently perform cross-node RDMA reads. The cluster topology is no longer distributed through the shm; the shm header carries only identity (`my_rank` / `node_name` / `registry` / `cluster_id`) and the ready flag.

---



## Contents

- [1. Configuration structs](#1-configuration-structs)
  - [1.1](#11-shmconfig--shm-and-eviction-config) `ShmConfig` [— shm and eviction config](#11-shmconfig--shm-and-eviction-config)
  - [1.2](#12-radixserverconfig--server-startup-config) `RadixServerConfig` [— server startup config](#12-radixserverconfig--server-startup-config)
  - [1.3](#13-radixclientconfig--client-discovery-config) `RadixClientConfig` [— client discovery config](#13-radixclientconfig--client-discovery-config)
- [2.](#2-radixserver) `RadixServer`
  - [2.5 Health check `check_health`](#25-health-check-check_health)
- [3.](#3-radixclient) `RadixClient`
  - [3.1 Construction and lifecycle](#31-construction-and-lifecycle)
  - [3.2 datapath: insert / query / remote ops](#32-datapath-insert--query--remote-ops)
    - `[insert](#insert)`
    - `[query](#query)`
    - `[component semantics of insert](#component-semantics-of-insert)`
    - `[remote_allocate](#remote_allocate)`
    - `[remote_insert](#remote_insert)`
    - `[remote_recycle](#remote_recycle)`
    - `[remote_query](#remote_query)`
    - `[remote_unlock](#remote_unlock)`
  - [3.3 Slot management](#33-slot-management)
    - `[reset](#reset)`
  - [3.4 Reference release](#34-reference-release)
  - [3.5 Distributed helpers](#35-distributed-helpers)
  - [3.6 Utilities and statistics](#36-utilities-and-statistics)
  - [3.7](#37-finalizefn--one-shot-callback) `FinalizeFn`
  - [3.8 Health check `check_health`](#38-health-check-check_health)
- [4. Enums](#4-enums)
- [5. Python binding differences](#5-python-binding-differences)

---



## 1. Configuration structs



### 1.1 `ShmConfig` — shm and eviction config


| Field                   | Type          | Default               | Meaning |
| ----------------------- | ------------- | --------------------- | ------- |
| `max_nodes`             | `uint32_t`    | `0`                   | NodePool capacity limit. `0` = estimated automatically from `max_blocks`. |
| `max_blocks`            | `uint32_t`    | `0`                   | Mempool capacity in blocks. `0` = mempool disabled; every insert does a separate alloc. |
| `block_size`            | `int`         | `1`                   | Tokens per block (used by `compute_hashes`); determines the hash granularity. |
| `register_chunk_tokens` | `int`         | `4096`                | RHT registration granularity, in tokens. `effective_register_chunk_size()` = `register_chunk_tokens / block_size` (at least 1; 0 = no alignment) is the block count cs actually used by the tree. It is also the **per-node block limit** (`min(cs, 4096)`); `create` throws if it exceeds 4096 blocks. cs is written into the shm header at creation, and all attaching processes read the same value. |
| `background_evict_ratio` | `float`      | `0.05`                | Free ratio maintained by the background eviction thread; `0` disables the background thread. Foreground allocation always evicts on demand and is not affected by this value. |
| `evict_policy`          | `EvictPolicy` | `LRU`                 | Eviction policy. |
| `data_pool_ratio`       | `float`       | `4.0`                 | Ratio of DataPool to NodePool. |
| `hugepage_path`         | `std::string` | `"/dev/hugepages"`    | Tries hugepage first, then falls back to `shm_open`. |
| `component_mask`        | `uint32_t`    | `COMPONENT_MASK_FULL` | Bit set of enabled components (bitwise OR of `COMPONENT_MASK_FULL/SWA/MAMBA`). The attach side checks each field; a mismatch fails closed. |
| `swa_window_blocks`     | `uint32_t`    | `0`                   | SWA window W (in blocks). Must be > 0 when SWA is enabled. |
| `mamba_state_bytes`     | `uint64_t`    | `0`                   | Size in bytes of one Mamba checkpoint; should equal the stride of the SlotStore MAMBA pool. The index layer uses it only for cluster ABI validation and the `check_pools_aligned` cross-check; address computation follows SlotStore. |
| `swa_max_blocks`        | `uint32_t`    | `0`                   | SwaMempool capacity (number of SWA slots). |
| `mamba_max_slots`       | `uint32_t`    | `0`                   | MambaMempool capacity (number of checkpoint slots). |


> Note: `RadixServer` provides the convenience overload `RadixServer(name, ShmConfig)`, which bootstraps automatically (for single-node use). Only `RadixServerConfig` lets you specify the etcd registry / node identity / cluster topology.

> **rank and world_size are not here**: they used to be `ShmConfig` fields and have been removed. rank is assigned by etcd, written back into `RadixServerConfig::rank` after bootstrap, and stamped into `my_rank` in the shm header. Whether the deployment is distributed is marked by `is_distributed` in the header (written by the server based on the actually observed `/peers` count); the client reads it at attach time to decide whether to start RDMA.



### 1.2 `RadixServerConfig` — server startup config


| Field                   | Type               | Default                   | Meaning |
| ----------------------- | ------------------ | ------------------------- | ------- |
| `name`                  | `std::string`      | `"/radixshmem"`             | Single node: shm name. Distributed: shm name prefix (`_<node_name>` is appended automatically, **not the rank**). |
| `shm`                   | `ShmConfig`        | `{}`                      | See [1.1](#11-shmconfig--shm-and-eviction-config). |
| `rank`                  | `int`              | `0`                       | Rank of this node. **The input value is ignored**: it is assigned by etcd and written back into this field after a successful bootstrap (`rank()` then reads it). |
| `world_size`            | `int`              | `1`                       | Total number of ranks. As input it is only a compatibility hint that you want to run distributed (see the distributed criterion below); during bootstrap it is overwritten by the **actually observed count** of `/peers`. |
| `registry`              | `std::string`      | `"etcd://127.0.0.1:2379"` | etcd endpoint used for cluster rendezvous. Can be overridden by env `RADIXSHMEM_REGISTRY`. |
| `cluster_id`            | `std::string`      | `"default"`               | etcd key namespace: `radix/<cluster_id>/...`. Multiple clusters on the same etcd are isolated by it. Can be overridden by env `RADIXSHMEM_CLUSTER_ID`. |
| `node_name`             | `std::string`      | `""`                      | Identity of this node in etcd (`/peers/<node_name>`), also the shm name suffix. Empty = automatically `node<bind-ip>`. Can be overridden by env `RADIXSHMEM_NODE_NAME`. **Clients use it to claim their rank**; all clients on the same node that use the same `node_name` get the same rank. |
| `rpc_address`           | `std::string`      | `""`                      | Bootstrap IPv4 published to peers (written into `/peers`), e.g. `"192.168.1.11"`; `"0.0.0.0"` can be used for single-node tests. The resolved IP is also used to generate the default `node_name`. |
| `rpc_interface`         | `std::string`      | `""`                      | NIC name (e.g. `"eth0"`), resolved to that NIC's IPv4; **takes precedence over `rpc_address`**. |
| `rpc_port`              | `uint16_t`         | `0`                       | XRC bootstrap listen port; `0` = assigned by the OS (the actual port is read back and published to `/peers`). |
| `expected_min_nodes`    | `int`              | `0`                       | Lower bound for the membership gate: bootstrap waits until `count(/peers) >= max(num_rht_shards, expected_min_nodes)` and stays stable for `settle_ms` before continuing. `0` = gate only on `num_rht_shards`. |
| `settle_ms`             | `int`              | `500`                     | Stability window of the membership gate (milliseconds): membership is considered settled only when the node count does not change during this period. |
| `index_dev`             | `std::string`      | `""`                      | RDMA device name. Empty = chosen automatically; setting it explicitly is recommended (e.g. `"mlx5_bond_0"`). |
| `gid_idx`               | `int`              | `3`                       | RDMA GID index (3 = RoCE v2 over IPv4). |
| `bootstrap_timeout_sec` | `int`              | `60`                      | Timeout for etcd rendezvous / XRC bootstrap (seconds). |
| `register_chunk_tokens` | `int`              | `4096`                    | Copied into `ShmConfig::register_chunk_tokens`. |
| `rht_transport`         | `std::string`      | `"xrc"`                   | RDMA transport from client to RHT shard holders: `"xrc"` (one QP per holder) or `"dc"` (shared DCI, O(1) QPs). Can be overridden by env `RADIXSHMEM_RHT_TRANSPORT`. **Only rank0 is authoritative** (broadcast via etcd `/config`). |
| `peer_index_transport`  | `std::string`      | `"xrc"`                   | Transport for cross-node remote-walk (one-sided RDMA reads of the peer's tree): `"xrc"` (per-peer full-mesh QPs) or `"dc"`, independent of `rht_transport`; if either is `"dc"` the server creates a DCT shared by both. Can be overridden by env `RADIXSHMEM_PEER_INDEX_TRANSPORT`. **Only rank0 is authoritative**. By default the XRC per-peer QPs are created for all peers at bootstrap (peers that join later are connected at admit time); env `RADIXSHMEM_LAZY_CONNECT=1` changes this to lazy creation on the first remote-walk. DC is not affected. |
| `num_rht_shards`        | `int`              | `0`                       | Number of RHT shards M. `0` = M == world_size (each node holds one shard; old behavior). When `> 0`, only some nodes hold RHT shards, and clients create RHT-plane QPs only to the M holders. When `M > world_size`, `bootstrap()` throws `std::runtime_error` directly (there cannot be more holders than ranks); it is no longer silently truncated. **Only rank0 is authoritative**. |
| `rht_shard_holders`     | `std::vector<int>` | `{}`                      | Candidate holder rank list (e.g. `{0,2,4,6}`). Empty = default ranks `[0, M)`. **Only rank0 is authoritative**. |
| `rht_slots_per_bucket`  | `int`              | `4`                       | Number of slots W per RHT bucket (chaining to handle hash collisions), one of `{1,2,4,8}`. Can be overridden by env `RADIXSHMEM_RHT_SLOTS`. **Only rank0 is authoritative**. |
| `enable_remote_insert`  | `bool`             | `false`                   | Allow peers to insert remotely into this server's tree (two-phase `[remote_allocate](#remote_allocate)` → `[remote_insert](#remote_insert)`). |
| `enable_remote_query`   | `bool`             | `false`                   | Allow peers to query this server's tree remotely (`[remote_query](#remote_query)`; `lock=true` can pin the matched prefix). |
| `remote_op_listen_port` | `uint16_t`         | `0`                       | TCP port of the remote-op endpoint (shared by insert/query; the gRPC server or the zmq REP socket; dc has no TCP endpoint, so this value is always 0). `0` = assigned by the OS (the real port is read back and published to etcd `/peers`). |
| `zmq_listen_port`       | `uint16_t`         | `0`                       | Deprecated alias of `remote_op_listen_port`. When both are set to different nonzero values, `bootstrap()` throws `std::runtime_error`. |
| `remote_op_transport`   | `std::string`      | `"grpc"`                  | remote-op transport: `"grpc"` (gRPC over TCP), `"zmq"` (TCP REQ/REP) or `"dc"` (DCI/DCT). rank0 is authoritative. Can be overridden by env `RADIXSHMEM_REMOTE_OP_TRANSPORT`. Any other value makes `bootstrap()` throw ("rdma"/"xrc" are the removed XRC backend). A backend that is not compiled into the build fails when the endpoint is created; there is no fallback to another backend. |


> **Distributed criterion**: `want_cluster()` = `expected_min_nodes > 1 || num_rht_shards > 1 || world_size > 1`. If any of the three holds, etcd + RDMA bootstrap is used, and in that case **one of** `rpc_address` / `rpc_interface` **must be given**; otherwise `bootstrap()` throws `std::runtime_error`.

> **Lane count**: the remote-op service lane count P (one poller thread per lane; at most 4 for the dc backend, always 1 for zmq) is not a config field. It is controlled only by env `RADIXSHMEM_RC_LANES` (default 1; must be set before bootstrap). The grpc backend has no lanes: it runs at most `RADIXSHMEM_GRPC_MAX_INFLIGHT` ops at once (default 32) and answers the excess with `RESOURCE_EXHAUSTED`. Each grpc op has a deadline of `RADIXSHMEM_REMOTE_OP_TIMEOUT_MS` (default 5000) on the client side.

> **Switching or rolling back the remote-op backend**: set `remote_op_transport` on rank 0 and restart every node. All nodes and clients of a cluster must run the same build: the manifest and the shard map carry a version, and a node of another version is rejected at bootstrap (`bad shard-map blob ... another cluster protocol version`). Mixed-version rolling upgrades are not supported.

> **Note**: `RadixServer::bootstrap()` **writes** `cfg.register_chunk_tokens` into `cfg.shm.register_chunk_tokens`, so you do not need to mirror it manually. `world_size` / `rank` flow in the opposite direction: after etcd determines them during rendezvous they are **written back into `cfg`**, and together with the identity fields they are stamped into the shm header (`my_rank` / `is_distributed` / `node_name` / `registry` / `cluster_id`).

> **rank0 is authoritative**: for `rht_transport` / `peer_index_transport` / `num_rht_shards` / `rht_shard_holders` / `rht_slots_per_bucket` / `remote_op_transport` (and RDMA QoS `RADIXSHMEM_RDMA_SL` / `RADIXSHMEM_RDMA_TC`), only the values on **rank 0** take effect. They are computed at bootstrap and **written into etcd `/config`** (shard map blob), and all other nodes and clients read them from there. Other ranks that pass different values have them ignored (a warning is printed) and run with rank0's authoritative values. This way the operator configures them once on rank0, all nodes + clients align automatically, and the client side needs no configuration. These are all **new fields whose defaults keep the old behavior**; existing deployments that do not set them see no change.



### 1.3 `RadixClientConfig` — client discovery config

Used by `RadixClient(const RadixClientConfig&)`: the shm name does not need to be known in advance. The client resolves the local shm from etcd `/shm/<node_name>` itself, then gets the topology from `/config` + `/peers`. **C++-only, not exported to Python** (Python can only use `radixshmem._core.RadixClient(name)` / `radixshmem._core.RadixClient(server)`).


| Field                   | Type          | Default                   | Meaning |
| ----------------------- | ------------- | ------------------------- | ------- |
| `registry`              | `std::string` | `"etcd://127.0.0.1:2379"` | etcd endpoint; must be the same as the server's. |
| `cluster_id`            | `std::string` | `"default"`               | etcd key namespace; must match the server's. |
| `node_name`             | `std::string` | `""`                      | **Node identity** to attach to: determines which shm `/shm/<node_name>` resolves to, and which `/peers` record `rank()` claims. When not an observer, an exception is thrown if it is not found in `/peers`. |
| `index_dev`             | `std::string` | `""`                      | RDMA device name. Empty = use the device published in this node's own `/peers` record. |
| `gid_idx`               | `int`         | `3`                       | GID index; if it differs from this node's own `/peers` record, the latter is used. |
| `bootstrap_timeout_sec` | `int`         | `60`                      | Timeout (seconds) for waiting on `/config`, `/shm/<node>`, and QP connection setup. |
| `rht_only`              | `bool`        | `false`                   | Create only RHT-plane QPs, no data-plane QPs. The eviction / remote-op service clients embedded in the server use this mode. |
| `no_local_tree`         | `bool`        | `false`                   | **observer mode**: there is no server on this machine; skip the shm attach and join the cluster only as a remote reader. In this mode `rank()` returns `-1` and local statistics are all 0/empty. |
| `expected_min_nodes`    | `int`         | `0`                       | Membership gate: wait until `count(/ready) >= max(expected_min_nodes, 1)` and `/peers` satisfies the same condition before continuing. |
| `settle_ms`             | `int`         | `500`                     | Reserved field; currently not used by the client-side gate (only the server-side field of the same name takes effect). |
| `register_chunk_tokens` | `int`         | `4096`                    | For reference only; the authoritative cs is read from etcd `/config` and is not decided by the client. |


> **Exceptions**: missing `/config`, empty `/peers`, `node_name` not in `/peers` (and not an observer), gate timeout, RDMA failing to start: all throw `std::runtime_error`.
>
> There is also an internal overload `RadixClient(cfg, LocalTreeConnector* borrowed)`: it borrows a tree the caller has already attached (no second attach), implies `rht_only=true`, and is used by services embedded in the server.

---



## 2. `RadixServer`

`RadixServer` is the owner of the node-level server. It creates the shm and runs etcd rendezvous (when distributed). It **does not expose datapath interfaces such as insert/query** — if the same process also needs the datapath, construct a `RadixClient(server)` on top of the server.

### 2.1 Construction and lifecycle

```cpp
RadixServer(const RadixServerConfig& cfg);
RadixServer(const std::string& name, const ShmConfig& cfg = ShmConfig{});
```


| Parameter        | Type                       | Meaning                                                                               |
| ---------------- | -------------------------- | ------------------------------------------------------------------------------------- |
| `cfg`            | `const RadixServerConfig&` | See [1.2](#12-radixserverconfig--server-startup-config). Does **not** bootstrap automatically; `bootstrap()` must be called explicitly. |
| `name`           | `const std::string&`       | shm name for the convenience constructor.                                             |
| `cfg` (overload) | `const ShmConfig&`         | shm config for the convenience constructor.                                           |


The convenience constructor (`(name, ShmConfig)`) is equivalent to: construct with a `RadixServerConfig` that contains only the `name` and `shm` fields, then **bootstrap immediately**.

**The constructor does not touch shm**: it only stores the config (in a `RADIXSHMEM_NO_RDMA` build, it throws here if the config requires distributed mode). shm creation/mmap, the eviction thread, RDMA, and etcd all happen in `bootstrap()` — so on the `RadixServerConfig` path, `bootstrap()` must be called after construction before a usable tree exists.

The destructor automatically stops the eviction thread and destroys the server-side cluster objects, but does **not** call `shm_unlink` — the OS reclaims the shm after all attached processes exit.

### 2.2 `bootstrap`

```cpp
bool bootstrap();
```

Lock: **none** (exclusive startup-time behavior, decoupled from the datapath read/write lock; the shm header uses an atomic release on `cluster_ready_magic` so that same-machine clients synchronize by spin-waiting, while cross-machine clients wait for etcd `/ready`).

Brings the server to the ready state: creates the shm region and starts the eviction thread; when distributed, also runs the full etcd rendezvous.

**Return value**


| Type   | Meaning                                                                     |
| ------ | --------------------------------------------------------------------------- |
| `bool` | `true` means success. On failure (etcd unreachable / membership gate timeout / no rank assigned / RDMA fails to start) returns `false`, and the server is unusable (the shm has been rolled back and destroyed). If distributed but neither `rpc_address` nor `rpc_interface` is given, it **throws an exception** instead of returning false. |


**Behavior**

- **Single machine** (`want_cluster()` is false): creates the shm, starts the eviction thread, and returns `true` directly, without touching etcd or RDMA at all.
- **Distributed**: resolve bind IP → determine `node_name` → create shm (named `name + "_" + node_name`) → etcd rendezvous (write itself into `/peers`, wait for the membership gate to stabilize, obtain a dense rank, rank0 computes and publishes the shard map in `/config`) → holders build RHT shards, start the XRC accept loop / DCT, start the remote-op service → stamp identity + `is_distributed` into the shm header and flip `cluster_ready_magic` → write `/ready`. The whole sequence blocks synchronously until the membership set is stable.
- If background eviction or remote-op is enabled, the end of bootstrap also embeds an `rht_only` `RadixClient` (borrowing the same tree), used to remove the corresponding RHT routing entries during local eviction.



### 2.3 Query interface

> Lock: **all lock-free** (they only read fields fixed at construction).


| Method             | Return type          | Meaning                                |
| ------------------ | -------------------- | -------------------------------------- |
| `is_distributed()` | `bool`               | `true` when `world_size() > 1` (after bootstrap, this means "the actually observed node count > 1"). Always `false` for a client constructed with `local_read_only`. |
| `rank()`           | `int`                | This node's rank. **Before** bootstrap it is the placeholder value from the config (default 0); **after** it is the dense rank assigned by etcd. Clients without a cluster plane (single machine, `local_read_only`) read `my_rank` from the header: 0 for a single-machine region, and the owner's rank when `local_read_only` attaches a distributed region. |
| `world_size()`     | `int`                | Total number of ranks. After bootstrap = the actual observed count in `/peers`. |
| `shm_name()`       | `const std::string&` | The **actual** shm name (with the `_<node_name>` suffix appended when distributed). Empty before bootstrap. |


### 2.4 Statistics interface

All statistics are **for this node only** (not summed across ranks). All return values are instantaneous snapshots, lock-free, and may lag slightly.


| Method                           | Return type | Meaning                                 |
| -------------------------------- | ---------- | --------------------------------------- |
| `total_radix_nodes()`            | `uint32_t` | Number of radix tree nodes currently used in the NodePool (as opposed to physical cluster nodes). |
| `total_blocks()`                 | `uint64_t` | Total number of blocks currently in the tree. |
| `data_pool_utilization()`        | `float`    | DataPool occupancy ratio `[0, 1]`.      |
| `mempool_total()`                | `uint32_t` | Mempool capacity (== `ShmConfig::max_blocks`). |
| `mempool_free()`                 | `uint32_t` | Current number of free Mempool slots.   |
| `mempool_used()`                 | `uint32_t` | Current number of used Mempool slots (`total - free`). |
| `insert_success_count()`         | `uint64_t` | Cumulative number of successful insert calls (one insert counts once as a whole). |
| `insert_error_node_pool_count()` | `uint64_t` | Number of failures because the NodePool was full. |
| `insert_error_data_pool_count()` | `uint64_t` | Number of failures because the DataPool was full. |
| `insert_error_ht_pool_count()`   | `uint64_t` | Number of failures because the hash table entries were full. |
| `insert_error_mempool_count()`   | `uint64_t` | Number of failures because the Mempool slots were full. |


### 2.5 Health check `check_health`

```cpp
int check_health(HealthReport* report = nullptr) const;
```

**The same audit** as [`RadixClient::check_health`](#38-health-check-check_health), but applied to the tree held by the server itself; returns `HEALTH_NOT_ATTACHED` before `bootstrap()`. Checks, return values, `HealthReport` fields, and caveats are all in §3.8. Use this entry point in unit tests that have only a server and no client; when a client exists the two are equivalent (same shm).


---



## 3. `RadixClient`



### 3.1 Construction and lifecycle

```cpp
explicit RadixClient(const std::string& name, bool local_read_only = false);
explicit RadixClient(const RadixClientConfig& cfg);
explicit RadixClient(RadixServer& server);
RadixClient(const RadixClientConfig& cfg, LocalTreeConnector* borrowed);
```


| Parameter         | Type                       | Meaning                                                                                                                                            |
| ----------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`            | `const std::string&`       | POSIX shm name (`/xxx`). After attach, reads `registry` / `cluster_id` / `node_name` from the shm header; if the header is marked `is_distributed`, it automatically runs the full etcd + RDMA bootstrap. **The caller does not need to know the etcd information**. |
| `local_read_only` | `bool`                     | Default `false`. When `true`, only attaches the tree: does not read the header identity and does not run etcd / RDMA / RHT bootstrap (even if the header is marked `is_distributed`); intended for read-only processes (workers, monitoring). In this mode `is_distributed()` is always `false`, `rank()` returns the header's `my_rank`, `world_size()` is 1, and `query()` is always local (the `local_only` parameter is ignored); `insert` / `allocate_slots` / `recycle_slots` / `reset` throw `std::runtime_error` — changing topology or recycling node ids without an RHT plane would leave peers routing to nodes that this process does not withdraw. |
| `cfg`             | `const RadixClientConfig&` | See [1.3](#13-radixclientconfig--client-discovery-config). **The shm name does not need to be known in advance**: it is resolved from etcd `/shm/<node_name>`. With `no_local_tree=true` the client is an observer and does not attach any local shm. |
| `server`          | `RadixServer&`             | Convenience overload, equivalent to `RadixClient(server.shm_name())`; even in the same process as the server, it starts its own independent RDMA context and does not share QPs. |
| `borrowed`        | `LocalTreeConnector*`      | Internal use: borrows a tree the caller has already attached (no second attach, not released on destruction), implies `rht_only`. The eviction / remote-op services embedded in the server use this path. |


**Exceptions**: distributed bootstrap failure throws `std::runtime_error` (etcd unreachable / `/config` missing / `node_name` not in `/peers` / gate timeout / RDMA QP fails to start, etc.). In a non-etcd build, constructing via `RadixClientConfig` also throws.

**Destructor**: releases local attach resources and destroys client-side QPs. Does **not** call `shm_unlink`; the shm is managed by the server side.

### 3.2 datapath: insert / query / remote ops



### `insert`

```cpp
InsertResult insert(const uint64_t* hashes, int n,
                    const int32_t* slots, int num_slots,
                    int start = 0, bool auto_recycle = true,
                    ComponentType component = ComponentType::FULL);
```

Lock: **write lock (**`write_lock`**)**

One call publishes only one `component`; the default is `FULL`, i.e. the classic prefix insert. `FULL` inserts a prefix hash sequence of length `n` into the local tree; in a distributed deployment it **synchronously** inserts into the local tree + **asynchronously** broadcasts routing updates via RDMA to all peers' RHT shards. The write is synchronous: when insert returns, the data can immediately be hit by query; there is no "not ready" intermediate state. For the special publish contract of `SWA` / `MAMBA`, see [component semantics of insert](#component-semantics-of-insert).

The slot IDs of newly inserted blocks are **provided by the caller** (no longer allocated from the mempool inside insert). `slots[i]` holds the KV data of `hashes[start+i]` — slots and hashes **correspond position by position starting at offset `start`** (positional), and are no longer "a pool taken freely from front to back". Typical usage: first `query` to peek the matched length `matched`, call `[allocate_slots](#allocate_slots)` (which triggers LRU eviction only when the mempool is full) to get `n - matched` slots and write the KV into them, then `insert(hashes, n, slots, n-matched, /*start=*/matched)`. insert itself **never evicts and never returns** `MEMPOOL_FULL` — all capacity pressure falls on the `allocate_slots` side.

**`start` contract (three cases)**: let `matched` be the actual match length obtained when insert re-walks internally:

| Case | Behavior |
| --- | --- |
| `matched < start` | The prefix is broken (the parent was evicted/shrunk after the peek), so the sequence cannot be continuous → **the whole call is rejected**, all slots go to `unused_slots`, nothing is published. |
| `matched == start` | Normal: `slots[0:]` are laid onto `hashes[start:]` position by position. |
| `matched > start` | The first `skip = matched - start` blocks were inserted first by another writer → the leading `skip` slots are skipped and only `slots[skip:]` are consumed; the leading `slots[0:skip]` go to `unused_slots`. |

In every case, **slots not registered into the tree** (leading duplicates + trailing slots not placed, merged without distinction) are collected into `unused_slots` of the return value. If there are not enough slots (`num_slots < n - matched`), the trailing blocks are dropped and the corresponding gap is not inserted into the tree.

> **⚠️ `start` default-0 pitfall**: a peek-then-insert caller that allocated only `n - matched` slots **must pass `start=matched` explicitly** and cannot rely on the default 0 (otherwise `skip=matched` goes out of bounds of the short array). The `start=0` default is only for trivial callers that "pass a complete `[0,n)` slot array".

> When protection against eviction is needed, use `query(..., lock=true)` to obtain a pin, and call `finalize()` on the return value after reading to release it. `insert` itself does not create pins.



#### Parameters


| Parameter      | Type              | Default | Meaning                                                                                |
| -------------- | ----------------- | ------ | -------------------------------------------------------------------------------------- |
| `hashes`       | `const uint64_t*` | —      | Cumulative prefix hash array (generated by `compute_hashes`).                          |
| `n`            | `int`             | —      | Array length (number of blocks, **not number of tokens**).                             |
| `slots`        | `const int32_t*`  | —      | Slot IDs for the newly inserted segment; `slots[i]` ↔ `hashes[start+i]` **correspond position by position starting at `start`**. |
| `num_slots`    | `int`             | —      | Length of `slots`. Typically == `n - matched`; may differ (if short, trailing blocks are dropped; if in excess, the leading/trailing surplus goes to `unused_slots`). |
| `start`        | `int`             | `0`    | Starting offset of the slot↔hash correspondence. peek-then-insert **must pass `start=matched`** (see the pitfall above). See the three cases of the `start` contract. |
| `auto_recycle` | `bool`            | `true` | `true`: insert internally calls `recycle_slots` on `unused_slots` automatically (the caller does not need to recycle, but the return value still carries `unused_slots` unchanged); `false`: the caller recycles them. |
| `component`    | `ComponentType`   | `FULL` | The single component published by this call: `FULL` / `SWA` / `MAMBA`; unaccepted slots return to the corresponding component pool. |




#### Return value `InsertResult`

```cpp
struct InsertResult {
    InsertError error = InsertError::OK;
    std::vector<int32_t> unused_slots;   // leading duplicates + trailing not placed, merged
};
```


| Field          | Type                   | Meaning                                                                                                                     |
| -------------- | ---------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `error`        | `[InsertError](#4-enums)` | Operation status code. Slots are provided externally, so it will **not** return `MEMPOOL_FULL`; it may still return `NODE_POOL_FULL` / `DATA_POOL_FULL` / `HT_POOL_FULL`. The rejection path of the three cases (`matched < start`) is **not specially marked**; it is still `OK` and is observed through `unused_slots`. |
| `unused_slots` | `std::vector<int32_t>` | Slot IDs **not registered into the tree** (leading redundant `slots[0:skip)` + trailing not placed `slots[inserted:)`, **flat-merged without distinction**). **Always returned** (regardless of `auto_recycle`); when `auto_recycle=true` these have already been recycled inside insert, and the return value is only for observation/validation. |


> **Number of slots actually registered into the tree** = `num_slots - unused_slots.size()` (the caller computes it if needed). When the whole call is rejected, `unused_slots` == all slots passed in.
>
> When `error != OK`, the part **actually** registered before the failure has been rolled back, and the slots not placed also go to `unused_slots`. With `auto_recycle=false`, the caller can simply call `recycle_slots(r.unused_slots.data(), r.unused_slots.size())` to return all of them (leading + trailing).



#### Notes

Cross-node routing updates are asynchronous; clients on peer ranks typically see them with an observable delay of tens of microseconds — no extra operation is needed on the datapath; subsequent RDMA operations automatically drain the completion events.

---



### `query`

```cpp
ComponentQueryResult query(const uint64_t* hashes, int n,
                           uint32_t component_mask = COMPONENT_MASK_FULL,
                           bool local_only  = false,
                           bool lock        = false,
                           bool update_meta = true);
```

Lock: the local walk holds the **read lock (**`read_lock`**)**; with `update_meta=true` it briefly holds `lru_lock`; with `lock=true` the refs of each component lane use atomics, and remote pins use RDMA FAA. Locks are released before `query` returns; the lifetime of pins is managed by `finalize`.

This is the only regular query interface, and it returns `ComponentQueryResult`. One call can jointly query **Full / SWA / Mamba** through `component_mask`; `common_hit` is the deepest valid boundary accepted by all requested components. Results are labeled by source: `full_fragments` consists of at most two segments, a local prefix and a remote suffix; SWA hits only on a complete node END, and Mamba accepts only an exact END checkpoint.

#### Parameters


| Parameter        | Type              | Default | Meaning                                                                                                       |
| ---------------- | ----------------- | ------- | ------------------------------------------------------------------------------------------------------------- |
| `hashes`         | `const uint64_t*` | —       | Cumulative prefix hash array.                                                                                  |
| `n`              | `int`             | —       | Array length (number of blocks). `n == 0` is an OK-miss, not an error.                                         |
| `component_mask` | `uint32_t`        | `COMPONENT_MASK_FULL` | Bitwise combination of `COMPONENT_MASK_FULL(1)` / `SWA(2)` / `MAMBA(4)`; must be a **non-empty subset** of the **enabled components** (`ShmConfig::component_mask`). `0`, unknown bits, components not enabled, or SWA enabled but `W == 0` → `BAD_REQUEST` fail-closed. **Do not** pass a `ComponentType` enum value as the mask (the numeric values are offset: the `FULL` enum is 0, the Full mask is 1). |
| `local_only`     | `bool`            | `false` | When `false`, a distributed deployment performs at most one remote continuation; when `true`, only the local tree is queried. An observer client (no local tree) supports only `false`. |
| `lock`           | `bool`            | `false` | Whether to add component refs on the hit range. See `lock` semantics below.                                    |
| `update_meta`    | `bool`            | `true`  | Whether to refresh LRU. **Full recency is refreshed only when the mask contains FULL**, and the refresh depth is the Full topology hit `f`, not `common_hit` (consistent with the Full pin range: a remote continuation may hand the caller the Full slots of local `[common_hit, f)`) — aux-only (SWA/Mamba-only) queries do not protect Full data from eviction; a remote continuation never modifies the owner's LRU. SWA only gives new timestamps to blocks in the hit window `[common_hit-W, common_hit)`; blocks before the window keep their old timestamps and are evicted first. |


#### Return value `ComponentQueryResult`

When `status != OK`, the other fields are not usable. On a single machine / with `local_only=true`, `rdma_*_count == 0` and `full_fragments` has at most one segment.

**Match result**


| Field            | Type                                  | Meaning                                                                                                         |
| ---------------- | ------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `status`         | `[RemoteStatus](#remotestatus)`       | Request-level status. Invalid parameters or component configuration → `BAD_REQUEST`; when the remote continuation fails to connect/read, the local incumbent is kept (empty hit for an observer), and the remote walker status does not overwrite the result. |
| `common_hit`     | `int`                                 | The deepest boundary accepted by all requested components (in blocks). When Mamba is included, it can only fall on a **complete node END**; SWA is decided per block and can fall on any boundary; for Full-only it is a regular block-granularity prefix hit. |
| `full_fragments` | `std::vector<ComponentSlotFragment>`  | When Full is requested and the hit is > 0, 1~2 segments, sorted by `offset` ascending, covering `[0, common_hit)` with no gap/overlap; each segment carries its own `source_rank` (distributed fold = local prefix + remote suffix). Empty when Full is not requested or the hit is zero. |
| `swa`            | `ComponentSlotFragment`               | The complete tail window `[max(0, common_hit-W), common_hit)` from exactly one source, slots in ascending order; on miss / not requested, `slot_ids` is empty and `source_rank == -1`. |
| `mamba`          | `std::optional<ComponentCheckpoint>`  | `{source_rank, slot_id}`, checkpoint located at END == `common_hit`; belongs to the same tail source as `swa`. Empty when there is no checkpoint. |


```cpp
struct ComponentSlotFragment {                 // slot IDs are valid only in the SlotStore of source_rank
    int32_t source_rank = -1;
    int offset = 0;                       // block position of the run within [0, common_hit)
    std::vector<int32_t> slot_ids;
};
struct ComponentCheckpoint {
    int32_t source_rank = -1;
    int32_t slot_id = INVALID_COMPONENT_SLOT;
};
```

**RDMA statistics and finalization**


| Field                                   | Type         | Meaning                                                                                                 |
| --------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------- |
| `rdma_read_count` / `rdma_atomic_count` | `int`        | Number of READ / atomic (guard + pin FAA) operations of the remote continuation; operations in the finalize phase are not written back. |
| `finalize`                              | `FinalizeFn` | Set when `lock=true` and at least one pin was actually obtained; encapsulates the one-shot release of both the local and remote halves; empty when `lock=false` or zero pins. See [§3.7](#37-finalizefn--one-shot-callback). |


#### acceptance semantics

- **Full**: a topology match is a hit; block granularity is kept, not quantized to node END by component semantics.
- **SWA (per block)**: boundary `L` is a hit if and only if the number of **consecutive live blocks** ending at `L` is **≥ min(W, L)**, i.e. every block's lane in the tail window `[max(0,L-W), L)` is valid; independent of node boundaries. One missing block resets the consecutive count to zero; after the gap slides out of the window, deeper boundaries hit again. Returns the whole window `[S, L)`; the pin covers **whole nodes** that intersect the window (when `S` falls in the middle of a node, `[A, S)` is a safe over-pin).
- **Mamba**: hits only at an exact node END with a valid checkpoint; a terminal partial match does not use that node's checkpoint.

#### `lock` semantics

- **Full**: inc_ref on **whole nodes** for every node intersecting `[0, f)` (`f` = topology hit length), including the partial terminal in which `f` falls — every returned Full slot is under pin protection. The hashes of the terminal beyond `f` are recorded in the release credential; at release they are concatenated with the caller's hashes to re-walk the full path, so the release is symmetric even if the terminal is later split after `f`.
- **SWA**: `swa_ref_count` +1 on nodes intersecting `[A, common_hit)`; **Mamba**: `mamba_ref_count` +1 on the node with `END == common_hit`.
- When a remote continuation extends the hit, the nodes on the remote suffix are pinned via RDMA FAA (only when the remote `L >` the local hit; on a tie the local result is kept, with zero remote pins).
- **Release**: call `result.finalize()` after reading (one-shot; re-walks the hash path by the internally recorded lane ranges to release). Forgetting to call it leaks the pins obtained. `query` does not expose the internal `PinRecord`; regular callers should not assemble release ranges by hand.

#### Query flow (when `local_only=false`)

1. Authoritative local query (+ optional pin / LRU touch / with `lock=true`, republish RHT according to the Full topology hit).
2. RHT routing: probe the local node start and the deepest cs-multiple position ≤ the local hit; the peer with the deepest hit wins; then follow that peer's routing entries by published length, chaining to the end of the chain (one RHT read per hop). If the chain end does not exceed the local hit, it is a miss directly, without accessing the peer; otherwise the whole chain is passed as a hint to the one-sided RDMA continuation, which batch-reads metadata and payload and verifies hashes, and no longer probes downward from the last node (a missing routing entry is treated as the chain end). Split suffix nodes are published by the inserter together with the path, which keeps the chain complete.
3. The remote result is adopted only when remote `L > local hit`: Full is folded into two segments, local `[0, min(f,L))` + remote `[min(f,L), L)`, and SWA/Mamba are taken entirely from the remote tail source; otherwise the complete local result is kept. An observer is equivalent to a local hit of 0.



### Component semantics of `insert`

Lock: the in-tree publish transaction holds the **write lock (**`write_lock`**)**; the `FULL` branch forwards verbatim to `[insert](#insert)` (including the distributed RHT republish).

`insert` has a single entry point; the trailing `component` parameter selects one publish branch per call. `component` is a `ComponentType`, not a mask: one `slots` payload belongs to exactly one slot pool / lane. The Full suffix, an SWA lane segment, and a single Mamba checkpoint differ in slot count and layout, so they cannot be expressed unambiguously by one mask plus one unpartitioned `slots` array.


| Branch | Semantics |
|---|---|
| `FULL` | Default branch; executes `insert(hashes, n, slots, num_slots, start, auto_recycle)` and keeps the classic prefix insert behavior. |
| `SWA` | **Publishes a tail segment by position**: slots cover the tail of `hashes`, `[n-num_slots, n)`, with `slots[i]` ↔ `hashes[n-num_slots+i]` and `1 <= num_slots <= n`. The start position is derived from `n - num_slots`; `start` is not needed. A call can publish a single position (`num_slots == 1`, per-block append during decode) or fill a long node at once (`num_slots == 32`); publishing a full window is the special case `num_slots == min(W, n)`. The window `W` **does not take part in publishing**; it is only used by `query` to decide whether the tail window is fully live. Blocks in the segment whose lane is INVALID get the slot from this call; blocks that are already live keep their existing slot (**block-level first-writer-wins**), and the corresponding new slot goes into `unused_slots`. Every block in the segment gets the same timestamp (publishing counts as a touch). **Does not split nodes**; `topology_changed` is always false. A successful publish only guarantees that the tail segment is filled; whether position `n` can be hit depends on whether `[max(0,n-W), n)` is fully live, which `query` decides. |
| `MAMBA` | Single-checkpoint publish: `num_slots == 1`; the slot is attached to the END of the node at `n` (first-writer-wins). When `n` falls in the middle of a node, the node is **split once on the right** to create an END. Missing Full path is not built. A published checkpoint is retired only by Mamba pool eviction: when a parent node's END carries a checkpoint, later sequential extension does **not** auto-merge it away. |


**Parameters**


| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `component` | `ComponentType` | — | `FULL` / `SWA` / `MAMBA`. |
| `hashes` / `n` | same as `insert` | — | Cumulative prefix hash array and its length. |
| `slots` / `num_slots` | same as `insert` | — | Positional binding `slots[i]` ↔ `hashes[start+i]` (SWA tail segment: `slots[i]` ↔ `hashes[n-num_slots+i]`, `1 <= num_slots <= n`; MAMBA is always a single slot). Slots must come from `allocate_slots` on the **matching component pool**. |
| `start` | `int` | `0` | Only valid for the `FULL` branch; SWA/MAMBA must pass `0`, otherwise `BAD_REQUEST` (the whole batch of slots goes to `unused_slots` and is reclaimed according to `auto_recycle`). |
| `auto_recycle` | `bool` | `true` | `unused_slots` are automatically returned to the **matching component pool**. |


**Return value `InsertResult`**


| Field | Meaning |
|---|---|
| `error` | See the error table below. |
| `unused_slots` | Input slots that lost (deduplication) or were returned as a whole batch (rejection). Contract: each input slot is either "adopted" or "put in unused", exactly one of the two, with no third state. |
| `topology_changed` | This publish split a node. In distributed mode the client has already republished the RHT chain for `[0,n)` automatically (applied asynchronously, soft state); the caller does not need to act. |


**Errors**


| error | Trigger | Tree changes |
|---|---|---|
| `OK` | Publish succeeded (including the dedup case where the whole segment lost under FWW) | Adopted slots are managed by the tree and later reclaimed by eviction |
| `BAD_REQUEST` | Component not enabled / SWA `num_slots` not in `[1, n]` / MAMBA with multiple slots / SWA or MAMBA passed a non-zero `start` | No change, whole batch unused |
| `FULL_PATH_MISSING` | The `[0,n)` Full path is incomplete (publishing only follows an existing path and does not build one) | No change, whole batch unused |
| `DATA_POOL_FULL` etc. | Not enough resources for the MAMBA right split (mirrors the split-failure handling of `insert`); SWA does not split nodes and never returns this | Whole batch unused |


**Recommended put flow**

```cpp
// SWA: after the Full chain is ready, publish the k new tail blocks of this step (the pages of one node, or 1 page during decode)
std::vector<int32_t> s;
if (client.allocate_slots(k, s, ComponentType::SWA) != k)
    return;                     // all-or-none: on shortage give up, write no data and publish nothing
/* write the SWA KV copies of hashes[n-k, n) into the slots in s */
auto r = client.insert(hashes, n, s.data(), k,
                       /*start=*/0, /*auto_recycle=*/true,
                       ComponentType::SWA);   // new slots for already-live blocks go back to unused_slots
// query(mask=SWA) hits at n only after the whole tail window [max(0,n-W), n) is live

// Mamba: single-slot checkpoint
std::vector<int32_t> m;
if (client.allocate_slots(1, m, ComponentType::MAMBA) != 1) return;
/* write the recurrent state for this depth */
client.insert(hashes, n, m.data(), 1,
              /*start=*/0, /*auto_recycle=*/true,
              ComponentType::MAMBA);
```

---

> **All remote operations are rank-first and handle-free**: the first parameter is the target peer `rank`; the client internally opens and caches one session per rank (created lazily on first use, reused afterwards). An **invalid rank** (own rank / out of range / negative / target has remote-op disabled) is reported in place in the return value as `BAD_TARGET` (query) / `REMOTE_FAILED` (insert) / `false` (recycle, unlock), and **never throws**. A transport `TIMEOUT` / `UNAVAILABLE` recovers by itself and does not invalidate the cached session.

> **Uncertain outcome**: `TIMEOUT` and `UNAVAILABLE` do not mean that the target skipped the op; it may have run. The grpc backend never resends an op (the zmq and dc backends resend the same request up to 3 times on a timeout, and the target does not deduplicate). After such a failure, do not assume the allocated slots are unused on the target, and do not recycle them only because of the failure.

### `remote_allocate`

```cpp
AllocateReply remote_allocate(int rank, int n);
```

**Phase 1** of the two-phase remote write: asks the peer to allocate `n` slots from **its own** mempool as a pure allocation (**no hashes, no server-side dedup, tree unchanged, nothing published, no pin**), and returns the target-local slot ids (so the caller can move KV there). Deduplication has moved to phase-2 `remote_insert` (the local insert only consumes the missed tail segment and the surplus is reclaimed automatically).

#### Parameters


| Parameter | Type    | Meaning                                          |
| ------ | ----- | ------------------------------------------- |
| `rank` | `int` | Target peer (invalid → `status = BAD_TARGET`).        |
| `n`    | `int` | Number of slots to allocate, **≤ 512** (over the limit returns `BAD_REQUEST`; not split). |




#### Return value `AllocateReply`


| Field       | Type                              | Meaning                                                                                                  |
| -------- | ------------------------------- | --------------------------------------------------------------------------------------------------- |
| `status` | `[RemoteStatus](#remotestatus)` | **Check first**; when not `OK`, `slots` is empty.                                                                          |
| `error`  | `InsertError`                   | Business result when `status == OK`: `MEMPOOL_FULL` when the target could not supply all `n` slots (nothing is allocated, `slots` is empty), else `OK`. |
| `slots`  | `std::vector<int32_t>`          | **Target-local** physical slot ids (not addressable from this process; for use by the KV transfer channel), length == `n`. The allocated slots are not bound to any prefix and the evictor cannot touch them; the server is stateless, so **abandoned slots must be returned with** `remote_recycle`. |


---



### `remote_insert`

```cpp
InsertResult remote_insert(int rank, const uint64_t* hashes, int n,
                           int start, int length, const int32_t* slots,
                           bool auto_recycle = true);
```

**Phase 2** of the two-phase remote write: the target runs a real `insert(hashes, slots)` with the phase-1 slots carried in the request, publishing into its tree + RHT (hits are possible immediately after publishing). **Deduplication happens here**: prefixes that the local insert already hits do not consume slots; by default (`auto_recycle=true`) the target reclaims surplus slots in place, and with `auto_recycle=false` they stay allocated and the caller reclaims them explicitly with `remote_recycle`. A new prefix typically passes `start=0, length=n`.

#### Parameters


| Parameter                    | Type                       | Meaning                                                                                    |
| --------------------- | ------------------------ | ------------------------------------------------------------------------------------- |
| `rank`                | `int`                    | Target peer (invalid → `error = REMOTE_FAILED`).                                                 |
| `hashes` / `n`        | `const uint64_t*` / `int`| Cumulative prefix hash array (the server is stateless and re-walks the path from it), `n ≤ 512`.                                                    |
| `start` / `length`    | `int`                    | Publish range; a new prefix uses `start=0, length=n`. The request carries `length` entries of `slots` (slots follow the hashes).                       |
| `slots`               | `const int32_t*`         | The slots returned by phase-1 `remote_allocate`.                                                   |
| `auto_recycle`        | `bool`                   | `true`: the target reclaims the dedup surplus in place; `false`: the surplus stays allocated and the caller reclaims it explicitly with `remote_recycle`.                             |




#### Return value

Same `InsertResult` (`{error, unused_slots}`) as local `[insert](#insert)`: `unused_slots` = slots not consumed due to dedup hits + tail slots that were not applied (if everything is deduplicated, == all `length` slots); the number actually published = `length - unused_slots.size()`; `error` reports the target tree insert result. With `auto_recycle=false`, the caller must return these `unused_slots` with `[remote_recycle](#remote_recycle)`. Invalid rank / transport failure → `error = REMOTE_FAILED`, and `remote_status` holds the [RemoteStatus](#remotestatus) behind it (`BAD_TARGET`, `TIMEOUT`, `UNAVAILABLE`, ...); `remote_status` is `OK` otherwise.

---



### `remote_recycle`

```cpp
RemoteStatus remote_recycle(int rank, const int32_t* slots, int num_slots);
```

Returns slots obtained from `remote_allocate` but **not published** to the target mempool (when the whole operation is aborted, or the surplus under `auto_recycle=false`). The target only runs `recycle_slots` and does not touch the tree.

#### Parameters


| Parameter                    | Type                       | Meaning                                            |
| --------------------- | ------------------------ | --------------------------------------------- |
| `rank`                | `int`                    | Target peer (invalid → `BAD_TARGET`).                   |
| `slots` / `num_slots` | `const int32_t*` / `int` | Slots to return and their count (null pointer / `0` → `BAD_REQUEST`).     |




#### Return value

`[RemoteStatus](#remotestatus)` — `OK` means the target has reclaimed the slots. **A failure must not be retried with a new call** (recycling is not idempotent).

---



### `remote_query`

```cpp
QueryResult remote_query(int rank, const uint64_t* hashes, int n,
                         bool lock = false, bool update_meta = true);
```

Remote-operation query (1 RTT): the peer queries **its own** local tree (this method does not touch the local tree).

#### Parameters


| Parameter             | Type     | Meaning                                                                                       |
| -------------- | ------ | ---------------------------------------------------------------------------------------- |
| `rank`         | `int`  | Target peer (invalid → `status = BAD_TARGET`).                                                     |
| `hashes` / `n` | —      | Cumulative prefix hash array, `n ≤ 512`.                                                                       |
| `lock`         | `bool` | When `true`, the target pins the hit prefix `[0, hit_length)` with inc_ref (it must be released after reading: `result.finalize()` or `remote_unlock`). |
| `update_meta`  | `bool` | When `false`, the target's LRU metadata is not modified.                                                                    |




#### Return value

`QueryResult` is used only by this explicit rank-first remote-op; it is not the return type of the regular `[query](#query)`.

| Field | Meaning |
| --- | --- |
| `status` | Request status; when not `OK`, do not use the other data fields. |
| `total_hit_length` / `remote_hit_length` | Both are the Full prefix hit length on the target peer. |
| `local_hit_length` / `prefix_slots` | Always `0` / empty; this method does not query the local tree. |
| `remote_node_id` | Target peer rank. |
| `remote_slots` | Slot IDs local to the target peer, length equal to the hit length; not directly addressable from this node. |
| `rdma_read_count` / `rdma_atomic_count` | Always `0`; this is a remote-op request/response, not a one-sided RDMA continuation. |
| `finalize` | Valid when `lock=true`, `status==OK`, and the hit is non-empty; internally calls `remote_unlock`. Otherwise empty. |

Blocks in a successful result are readable immediately. Use either `finalize()` or a manual `remote_unlock()`, not both, for the same pin.

---



### `remote_unlock`

```cpp
RemoteStatus remote_unlock(int rank, const uint64_t* hashes, int n, int hit_length);
```

Releases the pin from `remote_query(lock=true)`: the target runs dec_ref on `[0, hit_length)` (this is what `result.finalize()` calls internally).

#### Parameters


| Parameter             | Type    | Meaning                             |
| -------------- | ----- | ------------------------------ |
| `rank`         | `int` | Target peer (invalid → `BAD_TARGET`).     |
| `hashes` / `n` | —     | **The same hash array as the query** (the server is stateless and re-walks the path from it). |
| `hit_length`   | `int` | Length of the pin to release (= hit length from the query result).       |




#### Return value

`[RemoteStatus](#remotestatus)` — `OK` means the target has released the pin. **A failure must not be retried with a new call** (dec_ref is not idempotent).

> **The automatic cross-node walk is now part of** `[query](#query)`: when `query(hashes, n, COMPONENT_MASK_FULL, /*local_only=*/false)` finds too little locally, it routes through the RHT and continues the lookup on at most one peer with one-sided RDMA reads (the target CPU does not participate). Use `rdma_read_count > 0` in the return value to tell whether a remote read was issued; there is no longer a separate `remote_walk` entry point.

---



### 3.3 Slot management

> This interface lets KV cache managers manage slot IDs explicitly. A slot is now a **required** input of `[insert](#insert)` ("slot = the caller's physical block index"). Typical flow: `allocate_slots(n - matched)` → `insert(hashes, n, slots, n-matched, start=matched)` → slots that were not applied or were redundant at the front are returned by insert through `recycle_slots` automatically (`auto_recycle=true` by default; with `auto_recycle=false` the caller reclaims them from `r.unused_slots`). `allocate_slots` is the only LRU eviction point.



#### `allocate_slots`

```cpp
int allocate_slots(int num_slots, std::vector<int32_t>& out,
                   ComponentType component = ComponentType::FULL);
```

Lock: **write lock (**`write_lock`**)** — when capacity is insufficient, triggers LRU eviction for the corresponding component (the only eviction point).

Takes `num_slots` slot IDs from the slot pool selected by `component` (`FULL` main pool / SwaMempool / MambaMempool), **not bound to any prefix**. Used together with `recycle_slots`.

**Parameters**


| Parameter          | Type                      | Meaning                      |
| ----------- | ----------------------- | ----------------------- |
| `num_slots` | `int`                   | Requested number of slots.                 |
| `out`       | `std::vector<int32_t>&` | **Output**. Holds the slot IDs actually allocated. |
| `component` | `ComponentType` (default `FULL`) | Target slot pool. The ID spaces of the three pools are **independent of each other** (values may overlap). |


**Return value**


| Type    | Meaning                             |
| ----- | ------------------------------ |
| `int` | `FULL`: number actually allocated (may be < `num_slots` if still insufficient after eviction). `SWA`/`MAMBA`: **all-or-none** (on shortage, pops exactly the missing count along the component's **slot LRU** starting from the oldest slot; slots on pinned nodes are skipped) — returns `num_slots` when fully satisfied; otherwise hands out none and returns 0 (`out` is empty; once per-component eviction is implemented, this will retry after eviction). Invalid arguments (≤0 or larger than the total pool capacity) return -1. |




#### `recycle_slots`

```cpp
void recycle_slots(const int32_t* slots, int num,
                   ComponentType component = ComponentType::FULL);
```

Lock: **write lock (**`write_lock`**)**

Returns slot IDs taken with `allocate_slots` to the **matching component pool** (idempotent: recycling twice is a no-op). It can be called even if a slot is already attached to the tree, but semantically it returns those slots as "unused" — the caller must ensure no in-flight transfer still references them. Slots adopted by a component `insert` are managed by the tree; do not recycle them.

**Parameters**


| Parameter      | Type               | Meaning               |
| ------- | ---------------- | ---------------- |
| `slots` | `const int32_t*` | Array of slot IDs to recycle. |
| `num`   | `int`            | Array length.            |
| `component` | `ComponentType` (default `FULL`) | Target slot pool to return to; must match the pool used for allocation. |




#### `reset`

```cpp
void reset();
```

Lock: **write lock (**`write_lock`**)** — first drains in-flight remote RDMA readers and makes the background eviction thread yield.

Clears all caches of this client in place: resets this node's radix tree to empty (only a root remains, all slots are returned to the mempool, the `insert_*` diagnostic counters are zeroed), and memsets to zero the **RHT shards held by this rank (**`**holder_rank == my_rank`)**. It does **not** remap and does **not** reconnect RDMA — the shm layout / magic / locks / cluster identity markers / established QPs are all kept as is; only the data inside is erased.

**Key semantics**

- **Per node**: `reset()` only clears this rank. In a distributed deployment the whole cluster is cleared if and only if **every rank calls** `reset()` **once**; if only one rank calls it, the trees of the other ranks and the shards not held by this rank remain.
- **Only clears shards it holds**: the client's `local_rht_shards_` may contain both "shards held by this rank" and "same-host peer shards attached along the way"; `reset()` filters by `holder_rank == my_rank` and only touches the ones it owns, so peer data is not affected.
- **Invalidation**: after the call, all previously obtained slot IDs / `ComponentQueryResult` / `QueryResult` / `InsertResult` / `FinalizeFn` are invalid and must not be used.



### 3.4 Reference release

A pin can only be created by `[query](#query)` with `lock=true`, and can only be released through the one-shot `finalize()` in the return value. `finalize()` uses the per-lane coverage recorded at creation time (internal `PinRecord`, not exposed), re-walks the local tree and the pinned peer tree with the same `hashes`, and decrements references; it tolerates node splits that happened in between. `RadixClient` does not provide `inc_ref` / `dec_ref` / `dec_ref_remote`; C++ and Python are the same here.

- `finalize()` is idempotent: repeated calls are no-ops; `bool(finalize)` is `true` when there is still an unreleased pin.
- The pin lasts as long as the `ComponentQueryResult` (or the moved-out `finalize`) is held; dropping the handle without calling it does **not** release the pin, and the nodes stay non-evictable.
- Embedded integrations that need manual pin/release pairs use the tree-level `RadixShmemTree::inc_ref / dec_ref(PinRecord)` (via `RadixServer::tree()` / `LocalTreeConnector::tree()`).

### 3.5 Distributed helpers

The following methods are all no-ops in a single-node deployment (`is_distributed() == false`).

#### `flush`

```cpp
void flush();
```

Drains all **asynchronous RHT routing writes** triggered by `insert`, acting as a read-write visibility barrier: guarantees that later **cross-rank** `query` calls can see prefixes just inserted by this node. No-op in a single-node deployment.

### 3.6 Utilities and statistics

> Locks: **all lock-free**. Statistics fields are `std::atomic<>` relaxed loads or lock-free size reads; `compute_hashes` is pure CPU computation and does not touch shm.


| Method                    | Return type          | Meaning                                             |
| ------------------------- | -------------------- | --------------------------------------------------- |
| `is_distributed()`        | `bool`               | `true` when the attached shm is marked `is_distributed` by the server and the cluster is up. |
| `rank()`                  | `int`                | **This node's rank, decided by etcd**: the client claims it in `/peers` using its own `node_name` and does not take part in assignment. Single-node (non-distributed header / direct attach by name) returns `0`; observer (`no_local_tree`) returns `-1`. Multiple clients on the same node return the same rank as long as their `node_name` is the same. |
| `world_size()`            | `int`                | Total number of ranks (= number of entries observed in `/peers`); 1 on a single node. |
| `name()`                  | `const std::string&` | Name of the attached shm.                           |
| `total_radix_nodes()`     | `uint32_t`           | Number of radix tree nodes currently used in NodePool. |
| `total_blocks()`          | `uint64_t`           | Total number of blocks currently in the tree.       |
| `block_size()`            | `int`                | Tokens per block (written to the header when the shm is created, read back on attach); available without out-of-band configuration. |
| `data_pool_utilization()` | `float`              | DataPool occupancy ratio `[0, 1]`.                  |
| `mempool_total()`         | `uint32_t`           | Mempool capacity (FULL pool).                       |
| `mempool_free()`          | `uint32_t`           | Current number of free slots in Mempool.            |
| `mempool_used()`          | `uint32_t`           | Current number of used slots in Mempool.            |
| `swa_mempool_total()`     | `uint32_t`           | SWA pool capacity (`swa_max_blocks`); 0 if not enabled. |
| `mamba_mempool_total()`   | `uint32_t`           | Mamba pool capacity (`mamba_max_slots`); 0 if not enabled. |
| `component_config()`      | `ComponentConfig`    | Component ABI written to the header when the shm is created (mask / W / `mamba_state_bytes`). The three pool capacities must equal the three SlotStore pools one to one; the caller configures both sides from the same values. In Python, verify with `radixshmem.check_pools_aligned(client, store)`. |
| `print_tree()`            | `void`               | Prints a readable structure of the local tree (for debugging). |




#### `compute_hashes` (static method)

```cpp
static std::vector<uint64_t> compute_hashes(
    const int64_t* token_ids,
    int num_tokens,
    int tokens_per_block);
```

Folds `token_ids[0..num_tokens)` into blocks, computing one cumulative hash per `tokens_per_block` tokens.

**Parameters**


| Parameter          | Type             | Meaning                |
| ------------------ | ---------------- | ---------------------- |
| `token_ids`        | `const int64_t*` | Array of token IDs.    |
| `num_tokens`       | `int`            | Array length (number of tokens). |
| `tokens_per_block` | `int`            | Number of tokens folded into one block. |


**Return value**


| Type                    | Meaning                                                           |
| ----------------------- | ----------------------------------------------------------------- |
| `std::vector<uint64_t>` | Array of cumulative prefix hashes. Length = `num_tokens / tokens_per_block` (trailing tokens that do not fill a whole block are ignored). |


> All clients in the same deployment must use the same `tokens_per_block`; otherwise prefix keys do not match.



### 3.7 `FinalizeFn` — one-shot callback

```cpp
class FinalizeFn {
public:
    void operator()();               // First call runs the cleanup; later calls are harmless no-ops
    explicit operator bool() const;  // Whether it can still be called
    bool empty() const;
};
```

Both the `ComponentQueryResult` of a plain `query` and the `QueryResult` of an explicit `remote_query` carry `finalize`: it packages the release of the local/remote pins acquired by this read operation into a callable. It is empty when `lock=false` or when no pinnable complete node was obtained, and calling it is still a no-op. Once armed, it must be called explicitly before `reset()`, client destruction, or detach; destroying a `FinalizeFn` does not run it automatically. Calling it again after the first call is harmless.

The C++ callback copies the query hashes but captures a pointer to the owning `RadixClient`, so the client must stay alive until finalize completes. Remote writes (`remote_allocate` / `remote_insert`) carry no finalize; unpublished remote allocations are returned with `[remote_recycle](#remote_recycle)`.

```cpp
// Typical read path
auto qr = client.query(hashes, n, COMPONENT_MASK_FULL,
                       /*local_only=*/false, /*lock=*/true);
// ... use qr.common_hit / qr.full_fragments to pull KV ...
qr.finalize();
```



### 3.8 Health check `check_health`

```cpp
int check_health(HealthReport* report = nullptr) const;
```

Runs a consistency audit of the attached **local** tree **without acquiring any lock**. It is used at the end of unit / end-to-end tests to catch "leaks": who still holds a lock, whether any slot was obtained with `allocate_slots` but neither `insert`ed nor `recycle_slots`ed, and whether any pin from a `lock=true` query was not released with `finalize()`. The return value is a `HealthFlag` bit set; **`0` means all checks passed**. Pass `report` to get per-item details; `report->describe()` prints one line per failure reason (`"healthy\n"` when healthy).

The implementation is separate, in `src/indexer/health_check.cpp`; it only reads the pools as a friend of `RadixShmemTree` and does not change any datapath code path.

#### Checks

| Bit (`HealthFlag`) | What is checked | Data source |
| --- | --- | --- |
| `HEALTH_RWLOCK_READ_HELD` | The inter-process `pthread_rwlock` still has readers | glibc lock word `__readers` of `ShmHeader::rwlock` (reader count from bit 3 up) |
| `HEALTH_RWLOCK_WRITE_HELD` | A writer holds the lock, or a writer is queued | Same as above (`WRLOCKED` bit) + `__writers` |
| `HEALTH_LRU_LOCK_HELD` | `lru_mutex` is in the locked state | `__lock` word of `ShmHeader::lru_mutex` |
| `HEALTH_RDMA_LOCK_HELD` | `rdma_rwlock != 0`: a local writer set the write bit, or a cross-node RDMA reader's FAA +1 has no matching -1 | `ShmHeader::rdma_rwlock` atomic load |
| `HEALTH_FULL_SLOT_LEAK` | Slots allocated in the FULL mempool and slots referenced in the tree **do not correspond one to one** | mempool used-bitmap ⟷ `full_slots` collected by traversing the whole tree from root |
| `HEALTH_SWA_SLOT_LEAK` | Same for the SWA pool (only counts lane entries `!= INVALID_COMPONENT_SLOT`) | SWA mempool ⟷ SWA lane of each node |
| `HEALTH_MAMBA_SLOT_LEAK` | Same for the Mamba pool | Mamba mempool ⟷ `mamba_checkpoint_slot` of each node |
| `HEALTH_REF_LEAK` | `ref_count` / `swa_ref_count` / `mamba_ref_count` of any node is not 0 (pin not released) | Traverse all reachable nodes |
| `HEALTH_NODE_LEAK` | Number of nodes reachable from root ≠ `node_pool_size` (node leak / dangling) | Root traversal count ⟷ `ShmHeader::node_pool_size` |
| `HEALTH_NOT_ATTACHED` | No local tree to check (observer client `no_local_tree`, or server not yet bootstrapped) | — |

The slot audit counts each component pool separately; if any item is non-zero, that pool's leak bit is set:

| `HealthReport::Pool` field | Meaning |
| --- | --- |
| `enabled` / `total` | Whether this pool exists in this shm / capacity |
| `used` | Allocated count from the mempool's view (`total - free`) |
| `in_tree` | Number of slots referenced in the tree |
| `leaked` | Marked allocated in the pool, but not referenced by any node in the tree — **typically allocate followed by a missing insert / recycle** |
| `dangling` | Referenced in the tree, but free or out of range in the pool — means someone recycled a slot that is still in the tree, or inserted a slot that was never allocated |
| `duplicate` | The same slot is referenced by multiple node positions |

#### Return value `HealthReport`

| Field | Type | Meaning |
| --- | --- | --- |
| `flags` | `uint32_t` | Failure bit set, same as the function return value; `ok()` ⇔ `flags == 0` |
| `locks_inspected` | `bool` | Whether the pthread lock words were actually read. `false` on non-glibc libc; in that case the three pthread lock bits are never set (`rdma_rwlock` is still checked) |
| `rwlock_readers` / `rwlock_writers_waiting` / `rwlock_writer_held` | `uint32_t` / `uint32_t` / `bool` | `rwlock` snapshot |
| `lru_locked` | `bool` | `lru_mutex` snapshot |
| `rdma_rwlock` | `int64_t` | Raw value: bit 63 = local writer, low bits = number of remote RDMA readers |
| `full` / `swa` / `mamba` | `Pool` | Slot accounting for each component pool (see table above) |
| `nodes_with_refs` / `ref_total` | `uint32_t` / `int64_t` | Number of nodes still pinned / sum of the three ref kinds |
| `nodes_reachable` / `nodes_in_pool` | `uint32_t` | Number of nodes reachable from root / NodePool accounted count |
| `describe()` | `std::string` | One line per failure; `"healthy\n"` when healthy |

#### Notes

- **Acquires no lock and does not write shm**: lock state is read directly from the internal fields of glibc `pthread_rwlock_t` / `pthread_mutex_t`, `rdma_rwlock` is an atomic load, and tree traversal is plain memory reads. So even if another process is stuck holding a lock, it still returns normally and reports the lock holder.
- Designed for the **quiescent state** (test finished, all client operations have returned). Calling it on a live tree with concurrent writers does not block or crash (the traversal has out-of-range and cycle protection), but it sees a torn snapshot and may report false positives.
- The background eviction thread only briefly takes the write lock with `try_write_lock` when the watermark triggers, and touches no lock when idle, so it does not affect the quiescent-state result.
- Only audits the **local** tree: in distributed mode each rank calls it separately; a remote reader that forgot to release shows up as `HEALTH_RDMA_LOCK_HELD` on the **node being read**.
- Complexity O(number of nodes + total slots across pools); a pool with tens of millions of slots takes about tens of milliseconds. Do not put it on the datapath hot path.
- Test conventions: C++ unit tests call `expect_healthy(client)` before `TEST_PASS()` in each case (`tests/indexer/test_utils.h`; cases that intentionally leave pins / use hand-written slot ids pass the expected bits in `allowed`; set `RADIXSHMEM_HEALTH_SOFT=1` to downgrade the assertion to print-only, so one run lists all leaking cases); in cluster tests, the server process audits before exiting on SIGTERM and reports via `kHealthExitCode` (50), and the parent process collects it with `reap_servers()`; for Python tests see §5.

---



## 4. Enums



### `InsertError`

```cpp
enum class InsertError : uint32_t {
    OK             = 0,
    NODE_POOL_FULL = 1,   // NodePool exhausted
    DATA_POOL_FULL = 2,   // DataPool (buddy allocator) exhausted
    HT_POOL_FULL   = 3,   // Global child hash table entries exhausted
    MEMPOOL_FULL   = 4,   // Mempool slots exhausted (local insert takes slots from the caller, so it is never triggered; reserved)
    REMOTE_FAILED  = 5,   // remote_insert only: transport error / invalid target / request rejected
    BAD_REQUEST       = 6,   // Invalid component publish arguments or configuration
    FULL_PATH_MISSING = 7,   // Full path required by SWA/Mamba publish is incomplete
};
```



### `EvictPolicy`

```cpp
enum class EvictPolicy : uint32_t {
    LRU = 0,    // Only LRU is supported for now
};
```



### `RemoteStatus`

```cpp
enum class RemoteStatus : uint16_t {
    OK            = 0,  // Delivered + processed + reply valid (then check InsertError)
    BAD_REQUEST   = 1,  // Target validation failed: magic/version/length mismatch (detected by target, sent over the wire)
    UNKNOWN_OP    = 2,  // op not recognized by the target (detected by target, sent over the wire)
    // 3 was PEER_DISABLED (removed: when the target has remote-op disabled, peer_for returns nullptr internally,
    //   the op reports BAD_TARGET locally, and there is no corresponding status code anymore; do not reuse this value)
    TIMEOUT       = 4,  // No reply before the deadline (zmq/dc: after their retries) (synthesized locally by the client)
    BAD_REPLY     = 5,  // Reply received but malformed or inconsistent with the request (synthesized locally by the client)
    BAD_TARGET    = 6,  // target of remote_* out of range / == own rank (synthesized locally by the client)
    INTERNAL      = 7,  // Internal consistency / remote pin completion error, or the target's handler failed
    UNAVAILABLE   = 8,  // grpc: target unreachable, not yet serving, or shutting down
    RESOURCE_EXHAUSTED = 9,  // grpc: target refused the op at its in-flight limit (the op did not run)
};
```

The return values of all `remote_*` are judged by this: `QueryResult::status` of `remote_query`, `AllocateReply::status` of `remote_allocate`, `remote_recycle` / `remote_unlock` return `RemoteStatus` directly, and `remote_insert` folds it into `InsertError::REMOTE_FAILED` and keeps it in `InsertResult::remote_status`. Anything other than `OK` means no valid reply was obtained; do not read the other fields. `TIMEOUT` / `UNAVAILABLE` leave the outcome on the target unknown (see the note at the start of the remote operations). It is **orthogonal** to `[InsertError](#inserterror)`: `RemoteStatus` describes "whether the request was delivered / processed / the reply is valid", `InsertError` describes "the insert result on the target tree", and `InsertError` is only meaningful when `RemoteStatus == OK`.

### `HealthFlag`

Return bits of `check_health()`; for all definitions and meanings see [§3.8](#38-health-check-check_health). `HEALTH_OK = 0`, the others are `1u << k` (4 lock bits, 3 slot bits, ref, node), `HEALTH_NOT_ATTACHED = 1u << 31`.

### Component masks

```cpp
static constexpr uint32_t COMPONENT_MASK_FULL  = 1u << 0;
static constexpr uint32_t COMPONENT_MASK_SWA   = 1u << 1;
static constexpr uint32_t COMPONENT_MASK_MAMBA = 1u << 2;
static constexpr int32_t  INVALID_COMPONENT_SLOT = -1;
enum class ComponentType : uint32_t { FULL = 0, SWA = 1, MAMBA = 2 };  // Identifier, not a mask
```

`component_mask` of `query` is a bitwise combination of `COMPONENT_MASK_*`, because one query can read multiple components together; `component` of `insert` uses `ComponentType`, because one call takes only one payload and publishes it to one component pool. `ComponentType` is an identifier enum (its values are offset from the masks) and cannot be passed as a mask; on the Python side a type mismatch raises `TypeError` directly. If atomic publishing of multiple components is needed in the future, it requires a batch request carrying "component → payload", not replacing the current parameter with a mask.

---



## 5. Python binding differences

The public surface of the Python package is two pure-Python classes: `radixshmem.RadixServer` (one process per node, holding the index, SlotStore, transfer engine, and gRPC service; started with the operator parameters `ServerConfig`, it starts in `waiting`, and the geometry is sent by a client) and `radixshmem.RadixClient` (can be constructed before the server is ready; `RadixClient(name, Geometry)` also hands the geometry to the server; `status()` / `configure()` do not block, `wait_ready()` blocks; the index and SlotStore are attached on the first index / data call, raising `ServerNotReady` if not ready; index operations pass through to the underlying `_core.RadixClient`, plus `pull_async()`: query, hand the transfer to the server, publish locally on completion, and `PullJob.wait()` returns local slots). The classes aligned with C++ described below live in the pybind modules `radixshmem._core` (index surface) and `radixshmem._data` (SlotStore / transfer surface), for tests and advanced users to use directly.

`radixshmem._core` matches the C++ API names and signatures nearly one to one. Differences:

- `compute_hashes` is available both at module top level (`radixshmem.compute_hashes(...)`) and as a class static method (`radixshmem.RadixClient.compute_hashes(...)`); the two are equivalent.
- Input/output arrays use `numpy.ndarray` (`dtype=int64` for token ids, `uint64` for hashes, `int32` for slots).
- `RadixServer` has an extra convenience constructor `radixshmem.RadixServer(name, ShmConfig)`, which **calls bootstrap() automatically**. Equivalent to the existing one-line C++ convenience overload.
- Plain `client.query` returns `ComponentQueryResult`; `QueryResult` is only returned by `client.remote_query`. The fields of both, and of `InsertResult`, are readonly; slot vectors are converted to `int32` ndarrays.
- The `finalize` of both query result types is a callable in Python: `qr.finalize()` releases the pins the first time, and later calls are no-ops. `bool(qr.finalize)` / `qr.finalize.empty()` tell whether it is armed.
- `client.allocate_slots(num_slots, component=FULL)` returns an `int32` ndarray directly: the FULL pool may return fewer than `num_slots`, and the SWA/Mamba pools return an empty array if they cannot supply all; invalid arguments for which C++ returns -1 (`num_slots<=0`, exceeding pool capacity, invalid `component`) raise `ValueError` in Python.
- `radixshmem.InsertError` matches the eight C++ values (including `REMOTE_FAILED`); `radixshmem.INVALID_COMPONENT_SLOT` is the empty-slot sentinel.
- `client.reset()` has the same name and meaning as in C++ (clears the tree in place + the RHT shards held by this rank, no remap / no RDMA reconnect), and releases the GIL correctly when called.
- `insert` has only one signature in Python: `client.insert(hashes, slots, start=0, auto_recycle=True, *, component=radixshmem.ComponentType.FULL)`. `component` is keyword-only and defaults to `FULL`; `hashes` is a `uint64` ndarray and `slots` is an `int32` ndarray (**required**; its length is `num_slots`). For Full, `slots[i]` ↔ `hashes[start+i]`; peek-then-insert **must pass `start=matched`**. SWA/Mamba calls pass `component=radixshmem.ComponentType.SWA/MAMBA` respectively, and slots must come from the corresponding component pool. With `auto_recycle=True`, insert recycles `unused_slots` internally; the return value still carries `unused_slots` for observation. `InsertError` includes `BAD_REQUEST` / `FULL_PATH_MISSING`.

**Remote operations ([§3.2](#32-datapath-insert--query--remote-ops)) are all exported to Python** (only when a remote-op backend is compiled in), with signatures matching C++:

- **Always rank-first, no handles**: the client opens and caches sessions per rank internally and lazily. The normal automatic cross-node walk goes through `client.query(hashes, mask=radixshmem.COMPONENT_MASK_FULL, local_only=False)`; when the one-sided RDMA continuation fails, the local incumbent is kept (observer returns an empty hit).
- `client.remote_allocate(rank, n) → AllocateReply` (phase 1, allocation only)
- `client.remote_insert(rank, hashes, start, length, slots, auto_recycle=True) → InsertResult` (phase 2, publish + dedup)
- `client.remote_recycle(rank, slots) → RemoteStatus` (return unpublished allocations)
- `client.remote_query(rank, hashes, lock=False, update_meta=True) → QueryResult` (remote operation, 1 RTT)
- `client.remote_unlock(rank, hashes, hit_length) → RemoteStatus` (release the pin from `lock=True`)
- New types exposed readonly: `AllocateReply` (`status`/`error`/`slots`); enum `radixshmem.RemoteStatus` (9 values); `InsertResult.remote_status`. `RemotePeer` is no longer in Python (sessions are managed inside the client).
- The `finalize` closure of `remote_query(lock=True)` carries `keep_alive`: it keeps the parent `RadixClient` from being GC'd (the reply DCT of a dc session is attached to its process-level receive target).

**Plain query (Python)**: `client.query(...)` is the only entry point, and `mask` defaults to `COMPONENT_MASK_FULL`: `client.query(hashes, mask=radixshmem.COMPONENT_MASK_FULL | radixshmem.COMPONENT_MASK_SWA, local_only=False, lock=False, update_meta=True)`. Key points:

- `mask` must be a bitwise combination of `COMPONENT_MASK_*` constants; passing `radixshmem.ComponentType` by mistake raises `TypeError` directly;
- Returns `ComponentQueryResult` (readonly): `status`, `common_hit`, `full_fragments` (`[(source_rank, offset, int32 ndarray), ...]`), `swa_source_rank` / `swa_start` / `swa_slots`, `mamba` (`None` or a `(source_rank, slot_id)` pair, END == `common_hit`), `rdma_*_count`, `finalize` (see [§3.7](#37-finalizefn--one-shot-callback));
- The binding carries `keep_alive`: the result's `finalize` keeps the parent `RadixClient` from being GC'd.

**Configuration**: `radixshmem.ShmConfig(...)` supports keyword construction; its 13 parameters map one to one to the C++ fields (8 base fields `max_nodes` / `max_blocks` / `block_size` / `register_chunk_tokens` / `background_evict_ratio` / `evict_policy` / `data_pool_ratio` / `hugepage_path`, plus 5 component fields; `effective_register_chunk_size()` gives the converted block count `component_mask` / `swa_window_blocks` / `mamba_state_bytes` / `swa_max_blocks` / `mamba_max_slots`).

`radixshmem.RadixServerConfig` **has only a default constructor**; fields are set by attribute assignment (`cfg.enable_remote_insert = True`, etc.). All are exposed (26): `name` / `shm` / `rank` / `world_size` / `registry` / `cluster_id` / `node_name` / `rpc_address` / `rpc_interface` / `rpc_port` / `expected_min_nodes` / `settle_ms` / `index_dev` / `gid_idx` / `bootstrap_timeout_sec` / `register_chunk_size` / `num_rht_shards` / `rht_shard_holders` / `rht_transport` / `peer_index_transport` / `rht_slots_per_bucket` / `enable_remote_insert` / `enable_remote_query` / `remote_op_transport` / `remote_op_listen_port` / `zmq_listen_port` (the ports are read and written as `int` on the Python side).

**The whole `RadixClientConfig` class is not exported**: on the Python side there are only two constructors, `radixshmem._core.RadixClient(name, local_read_only=False)` and `radixshmem._core.RadixClient(server)`; etcd discovery is done automatically from the identity in the shm header (observer / `rht_only` modes are C++-only); `local_read_only=True` is a lightweight read-only attach that does not join the cluster, see [3.1](#31-construction-and-lifecycle).

**etcd key-value surface**: `radixshmem._core.EtcdKv(registry)` (only in `RADIXSHMEM_HAS_ETCD` builds) exports the key-value operations of the Go etcd client to Python: `put(key, value, lease=0)` / `create(key, value, lease=0) -> bool` (put-if-absent) / `get(key) -> str | None` / `get_prefix(prefix) -> [(key, value)]` / `count(prefix)` / `delete(key)` / `grant_lease(ttl_s)` / `keep_alive(lease)` / `keep_alive_stop(lease)` (stop renewing without revoking) / `revoke(lease)`; values are text. It shares one connection with the `RadixServer` / `DataPlaneRegistry` in the same process. `radixshmem.server` uses it to maintain `radix/<cluster_id>/geometry` (`{"geometry": <fields that must match>, "lease", "node"}`): the first node to accept a geometry creates it with `create`, so of two concurrent configures with different geometries only one wins; every node that accepts it renews the same lease, and the key expires once the last of them stops (a failed bring-up or `close()` only stops renewing). Data-mode nodes in `waiting` adopt it; every other configure is compared against it and raises `GeometryMismatch` on a difference. It also maintains `radix/<cluster_id>/nodes/<node>` on the server's own lease: the node's record (`node_name` / `role` / `rank` / `mode` / `data_plane` / `endpoint`) as JSON, published at `start()` with `rank=-1` and rewritten on every mode change. `role` is `ServerConfig.role` (`"prefill"` by default, `"decode"` with `radix-server --decode`), also in `NodeInfo.role`; `RadixClient.nodes()` (the `Nodes` RPC) returns all records, in any server mode. `radixshmem._core.MAX_REGISTER_CHUNK_SIZE` (= `BuddyHeader::MAX_DATA_BLOCKS`) is the upper limit of `register_chunk_size`.

**Health check ([§3.8](#38-health-check-check_health))**: `client.check_health() -> int` (`HealthFlag` bit set, 0 = healthy) and `client.health_report() -> HealthReport` (readonly fields `flags` / `locks_inspected` / `rwlock_readers` / `rwlock_writers_waiting` / `rwlock_writer_held` / `lru_locked` / `rdma_rwlock` / `full` / `swa` / `mamba` (`HealthPoolReport`) / `nodes_with_refs` / `ref_total` / `nodes_reachable` / `nodes_in_pool`, methods `ok()` / `describe()`; `bool(report)` is `ok()`, and `repr(report)` includes the `describe()` text). `server.check_health()` / `server.health_report()` have the same meaning. The bit constants are in `radixshmem.HealthFlag` (a `py::arithmetic` enum; `int(radixshmem.HealthFlag.FULL_SLOT_LEAK)` gives the integer to bitwise-AND with the return value). Conventions: each case in `python/tests/test_radixshmem.py` ends with `assert client.check_health() == 0`; `test_cluster.py` runs a `health` role on each node at the end; `tests/agentic_stress.py` / `tests/dist_stress_test.py` have each worker audit its own tree at the end, and if there is a leak the whole run returns non-zero.

**C++-only, not exported to Python**: `RadixServer::insert_error_mempool_count()`. (The number of remote-op lanes is controlled by the env `RADIXSHMEM_RC_LANES`, not a config field.)

All datapath methods of the Python module release the GIL correctly when calling the C++ implementation, so they can be used in multithreaded programs.
