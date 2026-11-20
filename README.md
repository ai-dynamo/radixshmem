# RadixShmem

RadixShmem is a KV cache index built on a distributed radix tree. Each node keeps its own radix tree in POSIX
shared memory, which any process on the node can attach to and insert or look up KV cache blocks by prefix
hash. Across nodes, a sharded Radix Hash Table (RHT) records which node holds each prefix, and a query
continues the match on that peer's tree, so the KV cache is indexed globally across the cluster. With a
transfer engine (Mooncake or NIXL), RadixShmem also moves the blocks between nodes.

## Framework support

- sglang: HiCache L3 storage backend, in progress.
- vLLM: KV offload backend, in progress.

## Build

Install the build dependencies (Ubuntu):

```bash
apt-get install libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc libhiredis-dev \
    libibverbs-dev libxxhash-dev golang git \
    libgoogle-glog-dev libgflags-dev libjsoncpp-dev libyaml-cpp-dev libnuma-dev
```

Then, from the repository root:

```bash
pip install -e .
```

The first build fetches Mooncake at the commit `MOONCAKE_COMMIT` pins into `third_party/mooncake` and builds it
there (a few minutes, once). An index-only build skips it with `-Ccmake.define.ENABLE_MOONCAKE=OFF`. After a
`git pull` or a branch switch, run `pip install -e .` again to rebuild the C++ part.

For development, install the build tools once and let an import rebuild the C++ modules after a change:

```bash
pip install scikit-build-core "pybind11==3.1.0"
pip install --no-build-isolation -Ceditable.rebuild=true -e .
```

The C++ tools and tests build with CMake (run ctest in an environment without an editable install, which takes
precedence over `PYTHONPATH`):

```bash
cmake -S . -B build && cmake --build build -j && ctest --test-dir build
```

The PyPI wheels are built as described in [packaging/README.md](packaging/README.md).

## Start the server

Each node runs one `radix-server`, which manages `--data-bytes` of host memory; without `--data-bytes` it is
index only. The slot size is configured by the client.

Single node:

```bash
radix-server --name /my_kv_cache --data-bytes 1024G
```

A cluster coordinates through etcd. By default the RHT lives in a Redis Cluster
([`docs/redis_deployment.md`](docs/redis_deployment.md)) and queries continue on peers over gRPC, so the index
uses no RDMA; `RADIXSHMEM_RHT_BACKEND=rdma` and `RADIXSHMEM_PEER_INDEX_BACKEND=rdma` move them to one-sided
RDMA. `--transfer-dev` selects the NICs of the data plane.

```bash
# Same command on every node except --rpc-address
export RADIXSHMEM_REDIS_ADDRS=10.0.0.1:7001,10.0.0.2:7001
radix-server --name /my_kv_cache --data-bytes 1024G \
    --expected-min-nodes 4 --registry etcd://10.0.0.1:2379 \
    --rpc-address 10.0.0.10 \
    --transfer-dev mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
```

`--swa-ratio` and `--mamba-ratio` give part of `--data-bytes` to the SWA and Mamba pools. For prefill/decode
disaggregation, decode nodes add `--decode`; prefill nodes then write the KV their background eviction takes
back to a decode node instead of dropping it.

The client API, server options and environment variables are in [`docs/api.md`](docs/api.md).

## Transfer backends

The data plane moves slots through a transfer backend, chosen with `--transfer-backend`:

- `mooncake` (default): the C++ Mooncake plugin, built by `pip install -e .` and included in the PyPI wheels.
- `nixl`: NIXL's Python package. Install it with `pip install -e ".[nixl]"` (brings `nixl-cu13` and torch;
  Python 3.13 or older), or put an existing NIXL installation's Python package on the path. Without
  `--transfer-port`, a free port is picked.

To check which backends are available:

```bash
python -c "import radixshmem; print(radixshmem.build_info()['transfer_backends'])"
```

## License

RadixShmem is released under the Apache License 2.0; the full text is in [LICENSE](LICENSE). Source files
carry an SPDX Apache-2.0 identifier and the NVIDIA copyright notice.
