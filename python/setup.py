# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(PROJECT_ROOT, "build")
PACKAGE_DIR = os.path.join(PROJECT_ROOT, "python", "radixshmem")

# Extension name -> CMake target. The top-level CMakeLists.txt builds both
# modules from one pybind11 against one interpreter; nothing is compiled here.
CMAKE_TARGETS = {
    "radixshmem._core": "radixshmem_core_py",
    "radixshmem._data": "radixshmem_data_py",
}


def _detect_rdma() -> bool:
    """Best-effort check for libibverbs headers (input to the CMake configure only)."""
    for h in ("/usr/include/infiniband/verbs.h",
              "/usr/local/include/infiniband/verbs.h"):
        if os.path.exists(h):
            return True
    return False


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    """Configure and build the CMake project with this interpreter, then place the modules."""

    def build_extensions(self):
        import pybind11

        env_flag = os.environ.get("RADIXSHMEM_ENABLE_RDMA", "auto").lower()
        if env_flag in ("0", "off", "false", "no"):
            enable_rdma = False
        elif env_flag in ("1", "on", "true", "yes"):
            enable_rdma = True
        else:
            enable_rdma = _detect_rdma()

        os.makedirs(BUILD_DIR, exist_ok=True)
        subprocess.check_call(
            ["cmake", "-S", PROJECT_ROOT, "-B", BUILD_DIR,
             "-DCMAKE_BUILD_TYPE=Release",
             "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
             f"-DENABLE_RDMA={'ON' if enable_rdma else 'OFF'}",
             "-DBUILD_PYTHON_MODULES=ON",
             f"-DPython3_EXECUTABLE={sys.executable}",
             f"-Dpybind11_DIR={pybind11.get_cmake_dir()}"])
        jobs = str(os.cpu_count() or 4)
        for ext in self.extensions:
            subprocess.check_call(
                ["cmake", "--build", BUILD_DIR, "-j", jobs,
                 "--target", CMAKE_TARGETS[ext.name]])

        for ext in self.extensions:
            dst = self.get_ext_fullpath(ext.name)
            built = os.path.join(PACKAGE_DIR, os.path.basename(dst))
            if not os.path.exists(built):
                raise FileNotFoundError(f"CMake did not produce {built}")
            if os.path.abspath(built) != os.path.abspath(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                self.copy_file(built, dst)


setup(
    packages=["radixshmem"],
    ext_modules=[CMakeExtension(name) for name in CMAKE_TARGETS],
    cmdclass={"build_ext": CMakeBuild},
    # console_scripts lives in pyproject.toml ([project.scripts]) — declaring it
    # here too would be dead weight, [project] metadata wins.
    package_data={"radixshmem": ["py.typed"]},
)
