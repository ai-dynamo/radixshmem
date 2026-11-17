# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks repaired radixshmem wheels: platform tag, exported symbols, the libraries each
shared object takes from the host, and the license files.

    python packaging/check_wheel.py <wheel or directory of wheels> ...
"""

import pathlib
import re
import subprocess
import sys
import tempfile
import zipfile

HOST_LIBS = {"libc.so.6", "libm.so.6", "libdl.so.2", "libpthread.so.0", "librt.so.1", "libresolv.so.2",
             "libgcc_s.so.1", "libstdc++.so.6", "ld-linux-x86-64.so.2"}
# What a transfer plugin may take from the host besides HOST_LIBS: rdma-core and libnuma.
PLUGIN_HOST_LIBS = {"libibverbs.so.1", "libnuma.so.1"}
EXPORTS = {"_core": {"PyInit__core"},
           "libradixshmem_mooncake": {"radixshmem_create_transfer_backend"}}
LICENSES = {"LICENSE", "ATTRIBUTIONS.md"}


def run(*cmd: str) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def check(wheel: pathlib.Path) -> list:
    errors = []
    if "manylinux_2_28_x86_64" not in wheel.name:
        errors.append("not tagged manylinux_2_28_x86_64")
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(wheel) as z:
        z.extractall(tmp)
        licenses = {pathlib.PurePosixPath(n).name for n in z.namelist() if "/licenses/" in n}
        objects = sorted(pathlib.Path(tmp).rglob("*.so*"))
        needs = {so.name: set(re.findall(r"\(NEEDED\).*\[(.+)\]", run("readelf", "-d", str(so))))
                 for so in objects}

        def from_host(name: str, seen: set) -> set:
            seen.add(name)
            libs = set()
            for lib in needs[name] - seen:
                libs |= from_host(lib, seen) if lib in needs else {lib}
            return libs

        # A library auditwheel bundled is checked through the objects that load it.
        for so in (so for so in objects if so.parent.name == "radixshmem"):
            stem = so.name.split(".")[0]
            plugin = stem.startswith("libradixshmem_")
            outside = from_host(so.name, set()) - HOST_LIBS - (PLUGIN_HOST_LIBS if plugin else set())
            if outside:
                errors.append(f"{so.name} takes {sorted(outside)} from the host")
            if stem in EXPORTS:
                exported = {line.split()[-1] for line in
                            run("nm", "-D", "-g", "--defined-only", str(so)).splitlines() if line.strip()}
                if exported != EXPORTS[stem]:
                    errors.append(f"{so.name} exports {sorted(exported ^ EXPORTS[stem])[:5]}")
            if stem == "_core" and re.search(r" (ibv|mlx5dv)_", run("nm", "-D", "--undefined-only", str(so))):
                errors.append(f"{so.name} links rdma-core instead of loading it")
            if plugin:
                versions = sorted(set(re.findall(r"IBVERBS_[0-9.]+", run("objdump", "-T", str(so)))))
                if versions:
                    print(f"  {so.name} needs {', '.join(versions)} from the host's rdma-core")
    missing = LICENSES - licenses
    if missing:
        errors.append(f"license files missing: {sorted(missing)}")
    print(f"{wheel.name}: {wheel.stat().st_size / 2**20:.1f} MiB, {'OK' if not errors else 'FAILED'}")
    return errors


def main(paths: list) -> int:
    wheels = [w for p in map(pathlib.Path, paths) for w in (sorted(p.glob("*.whl")) if p.is_dir() else [p])]
    if not wheels:
        print("no wheels given")
        return 1
    failed = False
    for wheel in wheels:
        for error in check(wheel):
            print(f"  {error}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
