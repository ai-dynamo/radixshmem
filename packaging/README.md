# Building the wheels

The PyPI wheels are built by [cibuildwheel](https://cibuildwheel.pypa.io) in a manylinux_2_28 image that
carries the dependencies, built once. The configuration is in `[tool.cibuildwheel]` of `pyproject.toml`;
`.github/workflows/wheels.yml` runs the same steps in CI.

| File | Purpose |
|---|---|
| `manylinux/Dockerfile` | The builder image: gRPC and hiredis built static, Mooncake, Go |
| `check_wheel.py` | Checks a repaired wheel: tag, exported symbols, libraries taken from the host, license files |
| `smoke_test.py` | Checks an installed wheel: version, `radix-server`, the Mooncake plugin with and without rdma-core |
| `test_clean_host.sh` | Installs a wheel on a host with no build tools and runs the tests without, then with, rdma-core |

Build the image, then the wheels for CPython 3.10 to 3.14 into `wheelhouse/`:

```bash
docker build -t radixshmem-manylinux packaging/manylinux
pipx run cibuildwheel --platform linux
```

`docker build --build-arg JOBS=<n>` sets the build parallelism (default 32). Behind a proxy, add
`--network host` and the proxy build arguments to `docker build`, and set
`CIBW_CONTAINER_ENGINE="docker; create_args: --network=host"` and
`CIBW_ENVIRONMENT_PASS_LINUX="http_proxy https_proxy no_proxy"` for cibuildwheel.

Each wheel is repaired by auditwheel, which leaves rdma-core, libnuma and libnl to the host, and is checked by
`check_wheel.py`. Inside the image the smoke test and the single-node Python tests run against the installed wheel.

To test a wheel on a clean host, run from the repository root, for example:

```bash
docker run --rm -v "$PWD":/src -w /src ubuntu:24.04 sh packaging/test_clean_host.sh 3.12 wheelhouse
```
