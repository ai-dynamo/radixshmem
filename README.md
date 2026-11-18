# RadixShmem

RadixShmem is a KV cache index built on a distributed radix tree. The architecture is decentralized: each node
keeps its own radix tree in POSIX shared memory. Each tree is both a local and a shared index. It tracks the
KV cache blocks held in the node's DRAM, and any process on the node can attach to it and insert or look up
blocks by prefix hash. Other nodes can also query it: a sharded distributed hash table, the Radix Hash Table
(RHT), records which node holds each prefix, and the querying node continues the match on that peer's tree,
by default with one gRPC call to the peer, or with one-sided RDMA reads, so the KV cache is indexed globally
across the cluster. An inference engine integrates with RadixShmem through one client interface, which
handles both local and cross-node lookups. The recommended setup uses RadixShmem as the index together with
an existing transfer engine, such as NIXL or Mooncake, to enable peer-to-peer KV cache sharing.

## Framework support

- sglang: HiCache L3 storage backend, in progress.
- vLLM: KV offload backend, in progress.

## Using RadixShmem

### Build

```bash
pip install .
```

The wheels published to PyPI are built as described in [packaging/README.md](packaging/README.md).

For development, install the build requirements once and install in editable mode. With
`editable.rebuild=true` the C++ modules are rebuilt on import after a change:

```bash
pip install scikit-build-core "pybind11==3.1.0"
pip install --no-build-isolation -Ceditable.rebuild=true -e .
```

The C++ libraries, tools and tests build with CMake. The Python tests in ctest import the modules a plain
CMake build puts in `python/radixshmem`, so run them in an environment without an editable install of
radixshmem; an editable install takes precedence over `PYTHONPATH`.

```bash
cmake -S . -B build && cmake --build build -j && ctest --test-dir build
```

A source build needs the grpc++ and protobuf development packages for the remote ops, hiredis for the
Redis RHT, the rdma-core headers and a Go toolchain for the etcd client, and, for the Mooncake transfer
backend, git and the glog, gflags, jsoncpp, yaml-cpp and libnuma development packages (Ubuntu: `apt-get
install libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc libhiredis-dev libibverbs-dev
libxxhash-dev golang git libgoogle-glog-dev libgflags-dev libjsoncpp-dev libyaml-cpp-dev libnuma-dev`);
configuration fails if one is missing. The first build fetches Mooncake at the commit `MOONCAKE_COMMIT` pins
into `third_party/mooncake` and builds it there (a few minutes, once); an index-only build skips it with
`pip install . -Ccmake.define.ENABLE_MOONCAKE=OFF`. Cross-node remote ops use gRPC exclusively.
The RHT and the peer index each run on one-sided RDMA (XRC or DC) or without it: the RHT on Redis
(`RADIXSHMEM_RHT_BACKEND=redis`), the peer index on the peers' gRPC remote ops
(`RADIXSHMEM_PEER_INDEX_BACKEND=grpc`). Both are the defaults, so by default the index opens no RDMA
device on any node.

The optional data-plane backends are built separately; see [Transfer backends](#transfer-backends).

### Start the server

Each node runs one `radix-server`, which manages `--data-bytes` of host memory; the slot size is configured
by the client. In a distributed setup, `--transfer-dev` selects the devices of the data plane. The index uses
no RDMA by default; an index plane switched to RDMA opens `RADIXSHMEM_INDEX_DEV` (first device if unset).

Single node:

```bash
radix-server --name /my_kv_cache --data-bytes 1024G
```

4 nodes, coordinated through etcd. The RHT lives in a Redis Cluster by default (`RADIXSHMEM_REDIS_ADDRS`;
[`docs/redis_deployment.md`](docs/redis_deployment.md) shows how to set one up);
`RADIXSHMEM_RHT_BACKEND=rdma` keeps it in shards on the nodes instead. A query continues on a peer's tree
with gRPC remote ops by default; `RADIXSHMEM_PEER_INDEX_BACKEND=rdma` walks it with one-sided RDMA instead.
These variables and the RDMA device of such planes (`RADIXSHMEM_INDEX_DEV`, `RADIXSHMEM_INDEX_GID_IDX`) are
listed in [`docs/api.md`](docs/api.md). Peers reach a node on one TCP port, `--rpc-port` (default: picked by
the OS and published in etcd).

```bash
# Same command on every node except --rpc-address
export RADIXSHMEM_REDIS_ADDRS=10.0.0.1:7001,10.0.0.2:7001
radix-server --name /my_kv_cache --data-bytes 1024G \
    --expected-min-nodes 4 --registry etcd://10.0.0.1:2379 \
    --rpc-address 10.0.0.10 \
    --transfer-dev mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
```

Enable the SWA pool:

```bash
radix-server --name /my_kv_cache --data-bytes 1024G --swa-ratio 0.2 \
    --expected-min-nodes 4 --registry etcd://10.0.0.1:2379 \
    --rpc-address 10.0.0.10 \
    --transfer-dev mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
```

Enable the Mamba pool:

```bash
radix-server --name /my_kv_cache --data-bytes 1024G --mamba-ratio 0.2 \
    --expected-min-nodes 4 --registry etcd://10.0.0.1:2379 \
    --rpc-address 10.0.0.10 \
    --transfer-dev mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
```

Prefill/decode disaggregation: decode nodes run the same command with `--decode` and join the same cluster,
so `--expected-min-nodes` counts both roles. A prefill node writes the KV its background eviction takes
(`--background-evict-ratio`, default 0.05) back to a ready decode node instead of dropping it, and pauses
eviction while unfinished write-backs hold `--writeback-max-held` of a pool (default 0.05); with no decode
node ready it recycles the slots at once. Decode nodes write nothing back. 2 prefill + 2 decode nodes:

```bash
# Prefill nodes 10.0.0.10 and 10.0.0.11
radix-server --name /my_kv_cache --data-bytes 1024G \
    --expected-min-nodes 4 --registry etcd://10.0.0.1:2379 \
    --rpc-address 10.0.0.10 \
    --transfer-dev mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7

# Decode nodes 10.0.0.20 and 10.0.0.21
radix-server --name /my_kv_cache --data-bytes 1024G --decode \
    --expected-min-nodes 4 --registry etcd://10.0.0.1:2379 \
    --rpc-address 10.0.0.20 \
    --transfer-dev mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
```

### Client API

A client connects to the local server by name and provides `query`, `pull_async`, `push_async`, and `insert`.

1. **Initialize.** Construct the client with the server name and a `Geometry` (tokens per block and bytes per
   slot), then wait until the server is ready.

```python
import radixshmem

geometry = radixshmem.Geometry(block_size=16, full_slot_bytes=1179648)
client = radixshmem.RadixClient("/my_kv_cache", geometry)
client.wait_ready(timeout_s=600)
```

2. **Query.** `query()` matches the prefix in the local radix tree and, in a cluster, in peer radix trees.
   `common_hit` is the number of blocks hit; `full_fragments` lists the hit as `(source_rank, offset, slots)`
   runs, and the slot ids are valid in the SlotStore of `source_rank`.

```python
hashes = radixshmem.compute_hashes(token_ids, tokens_per_block=16)
r = client.query(hashes)
# 8 blocks, the first 3 cached locally (rank 0) and the first 6 on rank 1:
# r.common_hit == 6
# r.full_fragments == [(0, 0, [12, 40, 41]), (1, 3, [7, 8, 9])]
```

3. **Pull.** `pull_async()` is the data-plane operation: it transfers the KV cache hit on a peer into local
   slots and publishes those blocks to the local radix tree.

```python
r = client.pull_async(hashes).wait()   # [0, r.common_hit) is now in local slots
for slot in r.full_slots:
    client.slot_view(slot)             # memoryview of the slot
r.finalize()                           # release the pin on the hit blocks
```

4. **Push.** `push_async()` is the reverse: it writes local slots to a peer and publishes them in the peer's
   radix tree. A prefill server pushes its background-evicted KV to the decode nodes with it.

```python
q = client.query(hashes, local_only=True, lock=True)
full = np.concatenate([s for _, _, s in q.full_fragments])
r = client.push_async(peer_rank, hashes[:q.common_hit], full).wait()   # the peer holds [0, r.common_hit)
q.finalize()
```

5. **Insert.** To add new KV cache, query the local hit, allocate slots for the remaining blocks, copy the KV
   from GPU straight into the slots through their memoryviews (zero-copy D2H), then insert.

```python
matched = client.query(hashes, local_only=True).common_hit
slots = client.allocate_slots(len(hashes) - matched)
for i, slot in enumerate(slots):
    dst = torch.frombuffer(client.slot_view(slot), dtype=torch.uint8)
    dst.copy_(gpu_blocks[matched + i])  # gpu_blocks[k]: KV of block k as a flat uint8 tensor
client.insert(hashes, slots, start=matched)
```

The full API (methods, parameters, result fields, error codes) is in [`docs/api.md`](docs/api.md).

## Transfer backends

The data plane moves slots through a transfer backend, chosen with `--transfer-backend`. Both backends implement
`radixshmem.transfer.TransferBackend`; `radixshmem.transfer.SlotTransfer` turns slot ids into address ranges for
whichever one is in use.

### 1. Mooncake (default)

The Mooncake backend is C++, built as the plugin `libradixshmem_mooncake.so` next to `radixshmem._core`, which
loads it when a server first needs it. The PyPI wheels include it, and `pip install .` builds it (see Build).
`ENABLE_MOONCAKE` takes `ON` (default: fetch and build Mooncake into `MOONCAKE_ROOT` unless a build is already
there), `AUTO` (use an existing build under `MOONCAKE_ROOT`, never fetch) or `OFF`. C++ programs outside the
package find the plugin through `RADIXSHMEM_TRANSFER_PLUGIN_DIR`.

```bash
radix-server --name /my_kv_cache --data-bytes 1024G \
    --expected-min-nodes 4 --registry etcd://10.0.0.1:2379 \
    --rpc-address 10.0.0.10 \
    --transfer-dev mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
```

### 2. NIXL

The NIXL backend uses NIXL's Python package (UCX backend). Install it with the `nixl` extra, which brings
`nixl-cu13` (and with it torch; no wheels for Python 3.14 yet), or put an existing NIXL installation's Python
package on the path:

```bash
pip install "radixshmem[nixl]"
radix-server --name /my_kv_cache --data-bytes 1024G \
    --transfer-backend nixl \
    --expected-min-nodes 4 --registry etcd://10.0.0.1:2379 \
    --rpc-address 10.0.0.10 \
    --transfer-dev mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
```

`--transfer-port` sets NIXL's metadata port; without it a free port is picked. To check which backends are
available:

```bash
python -c "import radixshmem; print(radixshmem.build_info()['transfer_backends'])"
```

## License

RadixShmem is released under the Apache License 2.0; the full text is in [LICENSE](LICENSE). Source files
carry an SPDX Apache-2.0 identifier and the NVIDIA copyright notice.
