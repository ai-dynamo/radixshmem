# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Wheel builder: manylinux_2_28 with the dependencies a radixshmem build needs, built once.
# gRPC and hiredis are static in /opt/deps; Mooncake is built at the commit CMakeLists.txt pins (MOONCAKE_COMMIT),
# without its HTTP metadata server (radixshmem uses P2PHANDSHAKE), in /opt/mooncake.
#
#   docker build -t radixshmem-manylinux packaging/manylinux     (see packaging/README.md)

FROM quay.io/pypa/manylinux_2_28_x86_64@sha256:d086f64cddcace250ef0f0073583c51b8298159303290a730c9f59bd91edfd2c

ARG GO_VERSION=1.27.4
ARG GRPC_VERSION=v1.85.2
ARG HIREDIS_VERSION=v1.5.0
ARG MOONCAKE_COMMIT=e5598b0992cc258b06d22f24d875480e8931a28e
ARG JOBS=32

ARG CMAKE_ARGS="-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5"

RUN dnf install -y epel-release && dnf config-manager --set-enabled powertools && \
    dnf install -y ninja-build rdma-core-devel numactl-devel xxhash-devel glog-devel gflags-devel jsoncpp-devel && \
    dnf clean all

RUN curl -fsSL https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz | tar -C /usr/local -xz
ENV PATH=/usr/local/go/bin:$PATH

RUN git clone --quiet --depth 1 --branch ${GRPC_VERSION} --recurse-submodules --shallow-submodules \
        https://github.com/grpc/grpc /tmp/grpc && \
    cmake -S /tmp/grpc -B /tmp/grpc/build ${CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=/opt/deps \
        -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_STANDARD=17 -DABSL_PROPAGATE_CXX_STD=ON \
        -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DgRPC_BUILD_CSHARP_EXT=OFF \
        -DgRPC_BUILD_GRPC_CSHARP_PLUGIN=OFF -DgRPC_BUILD_GRPC_NODE_PLUGIN=OFF \
        -DgRPC_BUILD_GRPC_OBJECTIVE_C_PLUGIN=OFF -DgRPC_BUILD_GRPC_PHP_PLUGIN=OFF \
        -DgRPC_BUILD_GRPC_PYTHON_PLUGIN=OFF -DgRPC_BUILD_GRPC_RUBY_PLUGIN=OFF && \
    cmake --build /tmp/grpc/build -j${JOBS} && cmake --install /tmp/grpc/build && rm -rf /tmp/grpc

RUN git clone --quiet --depth 1 --branch ${HIREDIS_VERSION} https://github.com/redis/hiredis /tmp/hiredis && \
    cmake -S /tmp/hiredis -B /tmp/hiredis/build ${CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=/opt/deps \
        -DBUILD_SHARED_LIBS=OFF -DENABLE_SSL=OFF -DDISABLE_TESTS=ON && \
    cmake --build /tmp/hiredis/build -j${JOBS} && cmake --install /tmp/hiredis/build && rm -rf /tmp/hiredis

RUN dnf install -y yaml-cpp-devel && dnf clean all && \
    git init --quiet /opt/mooncake && \
    git -C /opt/mooncake fetch --quiet --depth 1 https://github.com/kvcache-ai/Mooncake ${MOONCAKE_COMMIT} && \
    git -C /opt/mooncake checkout --quiet FETCH_HEAD && \
    git -C /opt/mooncake submodule update --quiet --init --depth 1 extern/pybind11 && \
    cmake -S /opt/mooncake -B /opt/mooncake/build-te ${CMAKE_ARGS} \
        -DWITH_TE=ON -DWITH_STORE=OFF -DWITH_STORE_RUST=OFF -DWITH_P2P_STORE=OFF \
        -DUSE_ETCD=OFF -DUSE_HTTP=OFF -DUSE_CUDA=OFF \
        -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF && \
    cmake --build /opt/mooncake/build-te --target transfer_engine -j${JOBS}

ENV CMAKE_PREFIX_PATH=/opt/deps
