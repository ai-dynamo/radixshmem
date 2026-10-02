# shmradix binary wheels (source-free distribution)

These wheels let others use shmradix **without any C++ source**. All
algorithm code lives inside the compiled `shmradix/_core*.so`; the wheel contains
only that `.so` plus two thin Python files (`__init__.py`, `server.py`).

## Two variants

| wheel | RDMA (cross-node) | runtime system deps |
|-------|-------------------|---------------------|
| `shmradix-0.1.0+cpu-...manylinux_2_28...whl`  | ❌ single-node shared memory only | none (libxxhash bundled) |
| `shmradix-0.1.0+rdma-...manylinux_2_28...whl` | ✅ RDMA one-sided reads       | `libibverbs.so.1`, `libmlx5.so.1` |

Both are `cp312` (CPython 3.12) and `manylinux_2_28` (glibc ≥ 2.28 — works on
Ubuntu 22.04 / most modern distros).

## Install

Single-node (recommended default):
```bash
pip install shmradix-0.1.0+cpu-*.whl
```

RDMA (needs the RDMA user-space libs on the host):
```bash
# Debian/Ubuntu:  apt-get install libibverbs1 libmlx5-1
# RHEL/CentOS:    dnf install libibverbs libmlx5
pip install shmradix-0.1.0+rdma-*.whl
```

## Constraints

- **Python 3.12 only** (wheel is `cp312`). For another Python version, rebuild.
- **glibc ≥ 2.28**.
- RDMA variant requires the host RDMA libs above; otherwise `import shmradix`
  fails with `libibverbs.so.1: cannot open shared object file`.

## Rebuilding (maintainers only — needs the source)

```bash
docker pull quay.io/pypa/manylinux_2_28_x86_64
docker run --rm -v /path/to/RadixShmem:/work \
    quay.io/pypa/manylinux_2_28_x86_64 bash -c \
    "bash /work/packaging/build_core_manylinux.sh && \
     bash /work/packaging/make_wheels_manylinux.sh"
# wheels land in packaging/dist/
```
`build_core_manylinux.sh` compiles both `_core.so` variants; `make_wheels_manylinux.sh`
packages each into a source-free wheel and runs auditwheel (bundling libxxhash,
excluding the RDMA stack for the +rdma variant).
