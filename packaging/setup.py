# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-free packaging for radixshmem.

Packages a PRE-COMPILED radixshmem/_core*.so (built separately in manylinux) plus
the thin Python wrappers into a binary wheel. Does NOT compile anything and does
NOT include any C++ source — all algorithm IP lives inside the .so.

Layout expected next to this file at build time:
    radixshmem/__init__.py
    radixshmem/server.py
    radixshmem/_core.cpython-312-x86_64-linux-gnu.so   (swapped in per variant)

The wheel is forced to be a platform wheel (not py3-none-any) via
has_ext_modules, so it carries the linux/cpXXX tag that auditwheel then rewrites
to a manylinux tag.
"""
import os
from setuptools import setup
from setuptools.dist import Distribution

VARIANT = os.environ.get("RADIXSHMEM_WHEEL_VARIANT", "")  # "rdma" | "cpu" | ""
LOCAL = f"+{VARIANT}" if VARIANT else ""


class BinaryDistribution(Distribution):
    """Force a platform-specific wheel (there is a compiled .so inside)."""
    def has_ext_modules(self):  # noqa: D401
        return True


setup(
    name="radixshmem",
    version="0.1.0" + LOCAL,
    description="Shared-memory RadixTree for KV cache prefix matching (binary)",
    license="Apache-2.0",
    license_files=["LICENSE", "ATTRIBUTIONS.md"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.12",
    ],
    packages=["radixshmem"],
    package_data={"radixshmem": ["*.so", "py.typed"]},
    include_package_data=True,
    python_requires="==3.12.*",
    install_requires=["numpy"],
    distclass=BinaryDistribution,
    entry_points={
        "console_scripts": ["radix-server=radixshmem.server:main"],
    },
    zip_safe=False,
)
