# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/python/radixshmem/_version.py _radixshmem_version
     REGEX "^__version__")
string(REGEX MATCH "[0-9]+(\\.[0-9]+)*" _radixshmem_version "${_radixshmem_version}")
project(RadixShmem VERSION ${_radixshmem_version} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(RADIXSHMEM_TEST_REDIS "" CACHE STRING
    "host:port of a Redis Cluster for the Redis RHT tests; empty skips them")
set(RADIXSHMEM_TEST_REDIS_BIN "" CACHE PATH
    "directory with redis-server and redis-cli for the Redis RHT fault tests; empty skips them")

# The Mooncake transfer backend is a plugin (libradixshmem_mooncake.so) loaded on first use; NIXL is reached
# through its Python package. ENABLE_MOONCAKE: ON (default) builds the plugin, and when MOONCAKE_ROOT holds no
# Mooncake build, fetches Mooncake at MOONCAKE_COMMIT and builds it there first; AUTO builds the plugin only
# from an existing Mooncake build; OFF skips it (an index-only build).
set(ENABLE_MOONCAKE ON CACHE STRING "Build the Mooncake transfer plugin: ON, AUTO or OFF")
set(MOONCAKE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mooncake" CACHE PATH
    "Mooncake source + build root")
set(MOONCAKE_COMMIT e5598b0992cc258b06d22f24d875480e8931a28e CACHE STRING
    "The Mooncake commit fetched when MOONCAKE_ROOT holds no build (packaging/manylinux/Dockerfile pins the same)")

# ==================== radixshmem library ====================
set(_radixshmem_sources
    # ==================== core: shm pools + radix tree engine ====================
    src/indexer/core/hash.cpp
    src/indexer/core/node_pool.cpp
    src/indexer/core/slot_lru.cpp
    src/indexer/core/data_pool.cpp
    src/indexer/core/child_table.cpp
    src/indexer/core/mempool.cpp
    src/indexer/core/radix_tree.cpp
    src/indexer/core/dedup_move.cpp
    # component match semantics (Full/SWA/Mamba), shared by local + RDMA walkers
    src/indexer/component/component_set.cpp
    src/indexer/component/swa_component.cpp
    src/indexer/component/mamba_component.cpp
    # aux chain primitives + per-node retire + touch (eviction lifecycle hooks)
    src/indexer/component/aux_list.cpp
    # RadixServer: shm region + evictor + (distributed) cluster bring-up.
    src/indexer/server.cpp
    # Write-back queue between the background evictor and the write-back side.
    src/indexer/writeback.cpp
    src/indexer/dedup_writer.cpp
    src/indexer/local.cpp
    src/indexer/client.cpp
    # Lock-free leak audit (check_health); friend of RadixShmemTree, no other coupling.
    src/indexer/health_check.cpp
    # Lock-free tree copy for radix-monitor (dump_tree); same coupling as check_health.
    src/indexer/tree_dump.cpp
    # RHT (Radix Hash Table): shard shm + routing engine (core), holder +
    # RdmaRht endpoints.
    src/indexer/rht/rht_core.cpp
    src/indexer/rht/rht_endpoint.cpp
    src/indexer/rht/redis_rht.cpp
    # RDMA transport plane: verbs primitives split by connection model.
    src/indexer/transport/rdma/context.cpp
    src/indexer/transport/rdma/rc_endpoint.cpp
    src/indexer/transport/rdma/xrc.cpp
    src/indexer/transport/rdma/dc_oneside.cpp
    src/indexer/transport/rdma/verbs_loader.cpp
    # XRC SEND-QP setup through the peer's XrcConnect RPC.
    src/indexer/transport/rdma/xrc_connect.cpp
    # The RDMA side of a radix-server (device, XRC SRQ, DCT, tree MR).
    src/indexer/transport/rdma/rdma_host.cpp
    # Peer-facing plane: the peer index backends (RDMA walks, gRPC remote ops)
    # + remote-walk executor + client-side remote-op protocol (RemotePeer).
    src/indexer/peer/peer_index.cpp
    src/indexer/peer/rdma_peer_index.cpp
    src/indexer/peer/grpc_peer_index.cpp
    src/indexer/peer/peer_breaker.cpp
    src/indexer/peer/rdma_remote_walker.cpp
    src/indexer/peer/remote_op.cpp
    # etcd cluster rendezvous; indexer-side (holds a RadixServer&).
    src/indexer/etcd_registry.cpp
    src/indexer/remote_op_service.cpp
    src/indexer/transport/remote_op_factory.cpp)
add_library(radixshmem STATIC ${_radixshmem_sources})
set_property(TARGET radixshmem PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(radixshmem PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    # core internal headers (node_pool.hpp etc.) live under src/indexer/core; radix_tree.h
    # (public) embeds these pools, so consumers must resolve the .hpp names too.
    ${CMAKE_CURRENT_SOURCE_DIR}/src/indexer/core
    # Public cluster headers require the one-sided RDMA transport headers.
    ${CMAKE_CURRENT_SOURCE_DIR}/src/indexer/transport/rdma
    # peer-plane headers (rdma_peer_index.hpp, peer_registry.hpp, net_util.hpp)
    # are private under src/indexer/peer.
    ${CMAKE_CURRENT_SOURCE_DIR}/src/indexer/peer
)

target_include_directories(radixshmem PRIVATE
    # remote_op_service.hpp, shared by the remote-op server backends
    ${CMAKE_CURRENT_SOURCE_DIR}/src/indexer
    # remote_op_factory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/indexer/transport
    ${CMAKE_CURRENT_SOURCE_DIR}/src/indexer/component
    ${CMAKE_CURRENT_SOURCE_DIR}/src/indexer/rht
)

target_compile_definitions(radixshmem PUBLIC XXH_INLINE_ALL)
target_link_libraries(radixshmem PUBLIC rt pthread ${CMAKE_DL_LIBS})

# Redis RHT backend: static hiredis when available, so nothing extra is needed at runtime.
find_path(HIREDIS_INCLUDE_DIR hiredis/hiredis.h)
find_library(HIREDIS_LIBRARY NAMES libhiredis.a hiredis)
if(NOT HIREDIS_INCLUDE_DIR OR NOT HIREDIS_LIBRARY)
    message(FATAL_ERROR "The Redis RHT backend needs hiredis (Ubuntu: apt-get install libhiredis-dev).")
endif()
target_include_directories(radixshmem PRIVATE ${HIREDIS_INCLUDE_DIR})
target_link_libraries(radixshmem PRIVATE ${HIREDIS_LIBRARY})
message(STATUS "Redis RHT backend: ${HIREDIS_LIBRARY}")

# ==================== gRPC (remote-op control plane) ====================
# A gRPC CMake package on CMAKE_PREFIX_PATH (the static build in the wheel image) wins over
# grpc++/protobuf via pkg-config (distro packages); the generated code is a static library only transport/grpc sees.
# gRPC's package does not look up the protobuf it was built with, so find that first.
find_package(protobuf CONFIG QUIET NO_CMAKE_SYSTEM_PATH NO_SYSTEM_ENVIRONMENT_PATH)
find_package(gRPC CONFIG QUIET NO_CMAKE_SYSTEM_PATH NO_SYSTEM_ENVIRONMENT_PATH)
if(gRPC_FOUND)
    set(PROTOC_EXECUTABLE $<TARGET_FILE:protobuf::protoc>)
    set(GRPC_CPP_PLUGIN $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
    set(GRPCPP_LIBRARIES gRPC::grpc++ protobuf::libprotobuf)
    message(STATUS "gRPC found (CMake package ${gRPC_VERSION}, ${gRPC_DIR})")
else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(GRPCPP QUIET grpc++)
        pkg_check_modules(PROTOBUF QUIET protobuf)
    endif()
    find_program(PROTOC_EXECUTABLE protoc)
    find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)
    if(NOT (GRPCPP_FOUND AND PROTOBUF_FOUND AND PROTOC_EXECUTABLE AND GRPC_CPP_PLUGIN))
        message(FATAL_ERROR "Remote ops need gRPC: grpc++ and protobuf via pkg-config, protoc and grpc_cpp_plugin (Ubuntu: apt-get install libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc).")
    endif()
    message(STATUS "gRPC found (grpc++ ${GRPCPP_VERSION}, protobuf ${PROTOBUF_VERSION})")
endif()
set(_grpc_proto_dir ${CMAKE_CURRENT_SOURCE_DIR}/src/indexer/transport/grpc)
set(_grpc_gen_dir   ${CMAKE_BINARY_DIR}/generated/grpc)
set(_grpc_gen_srcs
    ${_grpc_gen_dir}/remote_ops.pb.cc
    ${_grpc_gen_dir}/remote_ops.grpc.pb.cc)
file(MAKE_DIRECTORY ${_grpc_gen_dir})
add_custom_command(
    OUTPUT ${_grpc_gen_srcs}
           ${_grpc_gen_dir}/remote_ops.pb.h
           ${_grpc_gen_dir}/remote_ops.grpc.pb.h
    COMMAND ${PROTOC_EXECUTABLE}
            --proto_path=${_grpc_proto_dir}
            --cpp_out=${_grpc_gen_dir}
            --grpc_out=${_grpc_gen_dir}
            --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
            ${_grpc_proto_dir}/remote_ops.proto
    DEPENDS ${_grpc_proto_dir}/remote_ops.proto
    COMMENT "Generating remote-op gRPC code (remote_ops.proto)"
    VERBATIM)
add_library(radixshmem_grpc_proto STATIC ${_grpc_gen_srcs})
set_property(TARGET radixshmem_grpc_proto PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(radixshmem_grpc_proto PUBLIC
    ${_grpc_gen_dir} ${GRPCPP_INCLUDE_DIRS} ${PROTOBUF_INCLUDE_DIRS})
target_link_directories(radixshmem_grpc_proto PUBLIC
    ${GRPCPP_LIBRARY_DIRS} ${PROTOBUF_LIBRARY_DIRS})
target_link_libraries(radixshmem_grpc_proto PUBLIC
    ${GRPCPP_LIBRARIES} ${PROTOBUF_LIBRARIES})

target_sources(radixshmem PRIVATE
    src/indexer/transport/grpc/grpc_remote_op_client.cpp
    src/indexer/transport/grpc/grpc_remote_op_server.cpp)
target_include_directories(radixshmem PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/indexer/transport/grpc)
target_link_libraries(radixshmem PRIVATE radixshmem_grpc_proto)

# ==================== radixshmem_registry: the etcd plane (Go client wrapper) ====================
# The official Go etcd client compiled to a c-shared lib via `go build -buildmode=c-shared`,
# which also emits libradix_etcd.h (the C surface). grpc/protobuf are statically bundled into
# the .so, so the C++ side gains no heavy deps.
find_program(GO_EXECUTABLE go)
if(NOT GO_EXECUTABLE)
    message(FATAL_ERROR "No `go` toolchain found; the cluster bootstrap registry is the Go etcd client.")
endif()
set(_etcd_go_dir  ${CMAKE_CURRENT_SOURCE_DIR}/src/registry/etcd_go)
set(_etcd_go_lib  ${CMAKE_BINARY_DIR}/lib/libradix_etcd.so)
set(_etcd_go_hdr  ${CMAKE_BINARY_DIR}/include/libradix_etcd.h)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include)
# Build the wrapper into the build tree against the committed go.sum. GOPROXY carries
# a CN mirror fallback so the etcd modules download on restricted networks (Mooncake's pattern).
add_custom_command(
    OUTPUT ${_etcd_go_lib} ${_etcd_go_hdr}
    COMMAND ${CMAKE_COMMAND} -E env
            GOPROXY=https://proxy.golang.org,https://goproxy.cn,direct
            ${GO_EXECUTABLE} build -mod=readonly -trimpath -buildmode=c-shared
            "-ldflags=-s -w -extldflags=-Wl,-soname,libradix_etcd.so"
            -o ${_etcd_go_lib} radix_etcd.go
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_BINARY_DIR}/lib/libradix_etcd.h ${_etcd_go_hdr}
    WORKING_DIRECTORY ${_etcd_go_dir}
    DEPENDS ${_etcd_go_dir}/radix_etcd.go ${_etcd_go_dir}/go.mod ${_etcd_go_dir}/go.sum
    COMMENT "Building Go etcd client wrapper (libradix_etcd.so)"
    VERBATIM)
add_custom_target(radix_etcd_wrapper DEPENDS ${_etcd_go_lib} ${_etcd_go_hdr})
message(STATUS "etcd registry enabled (Go wrapper via ${GO_EXECUTABLE})")

# Its own plane, not the indexer's: both planes key off the same etcd, so both
# may link this.
add_library(radixshmem_registry STATIC src/registry/data_registry.cpp)
target_include_directories(radixshmem_registry PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include                      # libradix_etcd.h
    # etcd_client.hpp / registry_client.hpp — private, included by bare name.
    ${CMAKE_CURRENT_SOURCE_DIR}/src/registry)
target_link_libraries(radixshmem_registry PUBLIC rt pthread)
set_property(TARGET radixshmem_registry PROPERTY POSITION_INDEPENDENT_CODE ON)
add_dependencies(radixshmem_registry radix_etcd_wrapper)
target_link_libraries(radixshmem_registry PUBLIC ${_etcd_go_lib})

target_link_libraries(radixshmem PUBLIC radixshmem_registry)

# ==================== Mooncake transfer engine (data plane) ====================
# Headers and library MUST come from the same Mooncake checkout: TransferMetadata's descriptor structs change
# shape between revisions, and a stale archive against fresh headers links cleanly and then hands back
# garbage (empty buffer lists, in practice). So the library is looked up under MOONCAKE_BUILD_DIR, inside
# MOONCAKE_ROOT, and the headers come from that same tree -- never from a /usr/local install.
set(MOONCAKE_BUILD_DIR "${MOONCAKE_ROOT}/build-te" CACHE PATH
    "Mooncake build tree holding libtransfer_engine (must match MOONCAKE_ROOT's sources)")
set(_mooncake_vars MOONCAKE_INCLUDE_DIR MOONCAKE_LIBRARY MOONCAKE_BASE_LIBRARY MOONCAKE_COMMON_LIBRARY
    MOONCAKE_ASIO_LIBRARY)

macro(radixshmem_find_mooncake)
    find_path(MOONCAKE_INCLUDE_DIR transfer_engine.h
        HINTS ${MOONCAKE_ROOT}/mooncake-transfer-engine/include NO_DEFAULT_PATH)
    find_library(MOONCAKE_LIBRARY NAMES transfer_engine
        HINTS ${MOONCAKE_BUILD_DIR}/mooncake-transfer-engine/src NO_DEFAULT_PATH)
    # The static build splits into several archives and needs its transitive deps spelled out.
    find_library(MOONCAKE_BASE_LIBRARY NAMES base
        HINTS ${MOONCAKE_BUILD_DIR}/mooncake-transfer-engine/src/common/base NO_DEFAULT_PATH)
    find_library(MOONCAKE_COMMON_LIBRARY NAMES mooncake_common
        HINTS ${MOONCAKE_BUILD_DIR}/mooncake-common/src NO_DEFAULT_PATH)
    find_library(MOONCAKE_ASIO_LIBRARY NAMES asio
        HINTS ${MOONCAKE_BUILD_DIR}/mooncake-common NO_DEFAULT_PATH)
endmacro()

function(radixshmem_run what)
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "Mooncake: ${what} failed. It needs git, network access to GitHub and the "
            "development packages of glog, gflags, jsoncpp, yaml-cpp, libnuma and rdma-core (Ubuntu: apt-get "
            "install libgoogle-glog-dev libgflags-dev libjsoncpp-dev libyaml-cpp-dev libnuma-dev "
            "libibverbs-dev). ENABLE_MOONCAKE=OFF builds without the Mooncake transfer backend.")
    endif()
endfunction()

# Fetch Mooncake at MOONCAKE_COMMIT into MOONCAKE_ROOT and build its transfer engine there, as the wheel
# image does: static, without its HTTP metadata server (radixshmem uses P2PHANDSHAKE) and without CUDA.
function(radixshmem_build_mooncake)
    find_package(Git REQUIRED)
    include(ProcessorCount)
    ProcessorCount(_jobs)
    if(_jobs EQUAL 0)
        set(_jobs 8)
    endif()
    message(STATUS "Mooncake: fetching ${MOONCAKE_COMMIT} into ${MOONCAKE_ROOT} and building its transfer engine")
    file(MAKE_DIRECTORY ${MOONCAKE_ROOT})
    if(NOT EXISTS ${MOONCAKE_ROOT}/.git)
        radixshmem_run("git init" ${GIT_EXECUTABLE} init --quiet ${MOONCAKE_ROOT})
    endif()
    radixshmem_run("git fetch" ${GIT_EXECUTABLE} -C ${MOONCAKE_ROOT} fetch --quiet --depth 1
        https://github.com/kvcache-ai/Mooncake ${MOONCAKE_COMMIT})
    radixshmem_run("git checkout" ${GIT_EXECUTABLE} -C ${MOONCAKE_ROOT} checkout --quiet --force FETCH_HEAD)
    radixshmem_run("git submodule update" ${GIT_EXECUTABLE} -C ${MOONCAKE_ROOT} submodule update --quiet
        --init --depth 1 extern/pybind11)
    # The same toolchain as this build: a pip build's ninja is on no PATH, and radixshmem itself enables no C.
    include(CheckLanguage)
    check_language(C)
    set(_toolchain -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
    if(CMAKE_C_COMPILER)
        list(APPEND _toolchain -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER})
    endif()
    if(CMAKE_MAKE_PROGRAM)
        list(APPEND _toolchain -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM})
    endif()
    file(REMOVE_RECURSE ${MOONCAKE_BUILD_DIR})
    radixshmem_run("configure" ${CMAKE_COMMAND} -S ${MOONCAKE_ROOT} -B ${MOONCAKE_BUILD_DIR}
        -G ${CMAKE_GENERATOR} -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ${_toolchain}
        -DWITH_TE=ON -DWITH_STORE=OFF -DWITH_STORE_RUST=OFF -DWITH_P2P_STORE=OFF
        -DUSE_ETCD=OFF -DUSE_HTTP=OFF -DUSE_CUDA=OFF
        -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF)
    radixshmem_run("build" ${CMAKE_COMMAND} --build ${MOONCAKE_BUILD_DIR} --target transfer_engine
        --parallel ${_jobs})
    file(WRITE ${MOONCAKE_BUILD_DIR}/radixshmem_commit ${MOONCAKE_COMMIT})
endfunction()

set(RADIXSHMEM_MOONCAKE_FOUND OFF)
if(ENABLE_MOONCAKE)
    radixshmem_find_mooncake()
    # A build this file made at another commit is rebuilt; one made elsewhere (no stamp) is taken as is.
    set(_mooncake_stale OFF)
    if(EXISTS ${MOONCAKE_BUILD_DIR}/radixshmem_commit)
        file(READ ${MOONCAKE_BUILD_DIR}/radixshmem_commit _mooncake_built)
        if(NOT _mooncake_built STREQUAL MOONCAKE_COMMIT)
            set(_mooncake_stale ON)
        endif()
    endif()
    if(ENABLE_MOONCAKE STREQUAL "ON" AND (NOT (MOONCAKE_INCLUDE_DIR AND MOONCAKE_LIBRARY) OR _mooncake_stale))
        radixshmem_build_mooncake()
        foreach(_v ${_mooncake_vars})
            unset(${_v} CACHE)
        endforeach()
        radixshmem_find_mooncake()
    endif()
    if(MOONCAKE_INCLUDE_DIR AND MOONCAKE_LIBRARY)
        set(RADIXSHMEM_MOONCAKE_FOUND ON)
        set(MOONCAKE_LINK_LIBS
            ${MOONCAKE_LIBRARY}
            ${MOONCAKE_BASE_LIBRARY}
            ${MOONCAKE_COMMON_LIBRARY}
            ${MOONCAKE_ASIO_LIBRARY}
            glog gflags jsoncpp numa ibverbs pthread dl)
        # libcurl only backs Mooncake's HTTP metadata server (USE_HTTP).
        file(STRINGS ${MOONCAKE_BUILD_DIR}/CMakeCache.txt _mooncake_http REGEX "^USE_HTTP:BOOL=ON")
        if(_mooncake_http)
            list(APPEND MOONCAKE_LINK_LIBS curl)
        endif()
        message(STATUS "Mooncake found (${MOONCAKE_LIBRARY}) — transfer plugin enabled")
    elseif(ENABLE_MOONCAKE STREQUAL "ON")
        message(FATAL_ERROR "Mooncake built under MOONCAKE_ROOT=${MOONCAKE_ROOT} but its transfer engine "
                            "is missing (include=${MOONCAKE_INCLUDE_DIR}, lib=${MOONCAKE_LIBRARY})")
    else()
        message(STATUS "Mooncake not found under MOONCAKE_ROOT=${MOONCAKE_ROOT} — transfer plugin disabled "
                       "(ENABLE_MOONCAKE=AUTO)")
    endif()
else()
    message(STATUS "Mooncake disabled (ENABLE_MOONCAKE=OFF)")
endif()

# ==================== radix_server tool (launched by ServerLauncher) ====================
add_executable(radix_server tools/server_main.cpp)
target_link_libraries(radix_server PRIVATE radixshmem)

# ==================== Unit tests ====================
# Keep assert() live in test TUs under Release (trailing -UNDEBUG overrides -DNDEBUG).
function(radixshmem_enable_asserts _tgt)
    target_compile_options(${_tgt} PRIVATE -UNDEBUG)
endfunction()

# ---- core: local shm-tree engine (mirrors src/indexer/core) ----
# These live under tests/indexer/core/ and drive only the local single-node tree
# (allocate/insert/query/evict/state/merge/reset) via RadixClient. The shared
# harness test_utils.h stays at tests/indexer/ (also used by benches), so add it
# to the include path rather than duplicating it.
set(_core_unit_tests
    test_tree_insert
    test_tree_local_query
    test_tree_state
    test_tree_evict
    test_tree_reset
    test_client_concurrency
    test_alloc
    # Consolidated query_components coverage (match/pin/facade and the remote sections).
    test_component_query
    # Aux chain lifecycle (EV1): publish/touch/split/merge/cascade/reuse.
    test_component_evict
    # Background eviction with write-back: held slots, pinned prefixes, queue.
    test_evict_writeback
    test_tree_dedup
    test_dedup_move
    # RhtEngine over a fake one-sided endpoint: scratch-bounded lookup rounds,
    # remote insert/remove plumbing.
    test_rht_engine
    # PeerBreaker: skip/probe/backoff of an unreachable peer.
    test_peer_breaker
)
foreach(_t ${_core_unit_tests})
    add_executable(${_t} tests/indexer/core/${_t}.cpp)
    target_link_libraries(${_t} PRIVATE radixshmem)
    target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/tests/indexer
                                             ${CMAKE_SOURCE_DIR}/src/indexer/rht)
    radixshmem_enable_asserts(${_t})
    add_dependencies(${_t} radix_server)
    add_test(NAME ${_t} COMMAND ${_t})
endforeach()

# XRC shared-state cross-process primitive test (no radixshmem needed).
add_executable(test_xrc_state tests/indexer/test_xrc_state.cpp)
target_link_libraries(test_xrc_state PRIVATE radixshmem)
radixshmem_enable_asserts(test_xrc_state)
add_test(NAME test_xrc_state COMMAND test_xrc_state)


# Cross-process distributed integration test (writer/reader/lockdec/...).
add_executable(test_cluster tests/indexer/test_cluster.cpp)
target_link_libraries(test_cluster PRIVATE radixshmem)
radixshmem_enable_asserts(test_cluster)
add_test(NAME test_cluster COMMAND test_cluster)

# Phase-3 node-join: 2-node cluster + a late data-peer joiner (join_existing),
# existing reader senses it via /peers watcher and remote-walks it lazily.
add_executable(test_cluster_node_join tests/indexer/test_cluster_node_join.cpp)
target_link_libraries(test_cluster_node_join PRIVATE radixshmem)
radixshmem_enable_asserts(test_cluster_node_join)
add_test(NAME test_cluster_node_join COMMAND test_cluster_node_join)

# Observer (no local tree, rank -1): pure client discovers the cluster from
# etcd, remote-walks a writer's prefix, and rejects every local-tree op.
add_executable(test_client_no_server tests/indexer/test_client_no_server.cpp)
target_link_libraries(test_client_no_server PRIVATE radixshmem)
radixshmem_enable_asserts(test_client_no_server)
add_test(NAME test_client_no_server COMMAND test_client_no_server)
add_test(NAME test_client_no_server_dc COMMAND test_client_no_server)
set_tests_properties(test_client_no_server_dc PROPERTIES
    ENVIRONMENT "RADIXSHMEM_RHT_TRANSPORT=dc;RADIXSHMEM_PEER_INDEX_RDMA_TRANSPORT=dc")

# A client borrowing a tree (radix-server's RHT client) that fails to
# bootstrap leaves the tree to its owner.
add_executable(test_client_borrowed_tree tests/indexer/test_client_borrowed_tree.cpp)
target_link_libraries(test_client_borrowed_tree PRIVATE radixshmem)
radixshmem_enable_asserts(test_client_borrowed_tree)
add_test(NAME test_client_borrowed_tree COMMAND test_client_borrowed_tree)

# M < N RHT-shard decoupling test (N=3, M=2, holders={0,1}).
add_executable(test_cluster_shards tests/indexer/test_cluster_shards.cpp)
target_link_libraries(test_cluster_shards PRIVATE radixshmem)
radixshmem_enable_asserts(test_cluster_shards)
add_test(NAME test_cluster_shards COMMAND test_cluster_shards)

# Same binaries re-run over the DC data-plane transport (incl. fetch-add atomics).
add_test(NAME test_cluster_dc COMMAND test_cluster)
set_tests_properties(test_cluster_dc PROPERTIES
    ENVIRONMENT "RADIXSHMEM_RHT_TRANSPORT=dc;RADIXSHMEM_PEER_INDEX_RDMA_TRANSPORT=dc")
# The two planes pick their transport independently: one DC, the other XRC.
add_test(NAME test_cluster_rht_dc_peer_xrc COMMAND test_cluster)
set_tests_properties(test_cluster_rht_dc_peer_xrc PROPERTIES
    ENVIRONMENT "RADIXSHMEM_RHT_TRANSPORT=dc")
add_test(NAME test_cluster_rht_xrc_peer_dc COMMAND test_cluster)
set_tests_properties(test_cluster_rht_xrc_peer_dc PROPERTIES
    ENVIRONMENT "RADIXSHMEM_PEER_INDEX_RDMA_TRANSPORT=dc")
add_test(NAME test_cluster_shards_dc COMMAND test_cluster_shards)
set_tests_properties(test_cluster_shards_dc PROPERTIES
    ENVIRONMENT "RADIXSHMEM_RHT_TRANSPORT=dc;RADIXSHMEM_PEER_INDEX_RDMA_TRANSPORT=dc")

# DC + peer-shard RDMA: a rank only attaches its own shard, so every peer
# shard's RHT access hits the DC one-sided WRITE path (write_async_inline +
# flush) on the wire — no env needed, distinct node_name per rank already
# forces it. W=4 also covers read-modify-write over the collision chain.
add_test(NAME test_cluster_dc_rht_nic COMMAND test_cluster)
set_tests_properties(test_cluster_dc_rht_nic PROPERTIES
    ENVIRONMENT "RADIXSHMEM_RHT_TRANSPORT=dc;RADIXSHMEM_PEER_INDEX_RDMA_TRANSPORT=dc")
add_test(NAME test_cluster_shards_dc_rht_nic_w4 COMMAND test_cluster_shards)
set_tests_properties(test_cluster_shards_dc_rht_nic_w4 PROPERTIES
    ENVIRONMENT "RADIXSHMEM_RHT_TRANSPORT=dc;RADIXSHMEM_PEER_INDEX_RDMA_TRANSPORT=dc;RADIXSHMEM_RHT_SLOTS=4")

# The peer index over the peers' gRPC Query/Unlock instead of RDMA walks.
foreach(_t test_cluster test_cluster_node_join test_client_no_server)
    add_test(NAME ${_t}_grpc_peer COMMAND ${_t})
    set_tests_properties(${_t}_grpc_peer PROPERTIES
        ENVIRONMENT "RADIXSHMEM_PEER_INDEX_BACKEND=grpc")
endforeach()

# Multi-slot RHT re-run with W=4 to exercise collision chaining.
add_test(NAME test_cluster_w4 COMMAND test_cluster)
set_tests_properties(test_cluster_w4 PROPERTIES
    ENVIRONMENT "RADIXSHMEM_RHT_SLOTS=4")
add_test(NAME test_cluster_shards_w4 COMMAND test_cluster_shards)
set_tests_properties(test_cluster_shards_w4 PROPERTIES
    ENVIRONMENT "RADIXSHMEM_RHT_SLOTS=4")

# Redis RHT against a live cluster at RADIXSHMEM_TEST_REDIS; skips without one.
add_executable(test_rht_redis tests/indexer/core/test_rht_redis.cpp)
target_link_libraries(test_rht_redis PRIVATE radixshmem)
target_include_directories(test_rht_redis PRIVATE ${CMAKE_SOURCE_DIR}/tests/indexer)
radixshmem_enable_asserts(test_rht_redis)
add_test(NAME test_rht_redis COMMAND test_rht_redis)
set_tests_properties(test_rht_redis PROPERTIES SKIP_RETURN_CODE 77
    ENVIRONMENT "RADIXSHMEM_TEST_REDIS=${RADIXSHMEM_TEST_REDIS}")

# Starts its own 6-node cluster on free ports; skips without RADIXSHMEM_TEST_REDIS_BIN.
add_executable(test_rht_redis_faults tests/indexer/core/test_rht_redis_faults.cpp)
target_link_libraries(test_rht_redis_faults PRIVATE radixshmem)
target_include_directories(test_rht_redis_faults PRIVATE ${CMAKE_SOURCE_DIR}/tests/indexer)
radixshmem_enable_asserts(test_rht_redis_faults)
add_test(NAME test_rht_redis_faults COMMAND test_rht_redis_faults)
set_tests_properties(test_rht_redis_faults PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120
    ENVIRONMENT "RADIXSHMEM_TEST_REDIS_BIN=${RADIXSHMEM_TEST_REDIS_BIN}")

# ==================== Remote-op e2e gate ====================
# Explicit remote routing API on RadixClient; self-SKIPs (exit 77) without a device.
add_executable(test_remote_api tests/indexer/test_remote_api.cpp)
target_link_libraries(test_remote_api PRIVATE radixshmem)
add_test(NAME test_remote_api_grpc COMMAND test_remote_api)
add_test(NAME test_remote_api_grpc_dcd COMMAND test_remote_api)
set_tests_properties(test_remote_api_grpc_dcd PROPERTIES
    ENVIRONMENT "RADIXSHMEM_RHT_TRANSPORT=dc;RADIXSHMEM_PEER_INDEX_RDMA_TRANSPORT=dc")

# Loopback gRPC tests use a real single-node tree without RDMA or etcd.
add_executable(test_remote_op_grpc tests/indexer/test_remote_op_grpc.cpp)
target_link_libraries(test_remote_op_grpc PRIVATE radixshmem radixshmem_grpc_proto)
target_include_directories(test_remote_op_grpc PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/indexer ${CMAKE_SOURCE_DIR}/src/indexer
    ${CMAKE_SOURCE_DIR}/src/indexer/transport)
radixshmem_enable_asserts(test_remote_op_grpc)
add_test(NAME test_remote_op_grpc COMMAND test_remote_op_grpc)

# Data-plane geometry in the shared etcd registry. Needs an etcd binary but no
# RDMA device, and stays single-process (the Go wrapper deadlocks on a bare
# fork). Links radixshmem_registry ALONE — neither plane's library — which keeps
# the registry plane-neutral.
add_executable(test_data_registry tests/registry/test_data_registry.cpp)
target_link_libraries(test_data_registry PRIVATE radixshmem_registry)
# etcd fixture from tests/indexer; runner from tests/dataplane (no indexer headers).
target_include_directories(test_data_registry PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/indexer
    ${CMAKE_SOURCE_DIR}/tests/dataplane)
radixshmem_enable_asserts(test_data_registry)
add_test(NAME test_data_registry COMMAND test_data_registry)

# Tests that self-SKIP (exit 77) without RDMA device / python ext / etcd binary;
# register code so ctest shows "Skipped". The cluster tests fork a private etcd
# (the sole bootstrap backend) and SKIP when no etcd binary is available.
set(_skippable_tests
    test_remote_api_grpc test_remote_api_grpc_dcd
    test_cluster test_cluster_dc test_cluster_dc_rht_nic test_cluster_w4
    test_cluster_rht_dc_peer_xrc test_cluster_rht_xrc_peer_dc
    test_cluster_node_join
    test_client_no_server test_client_no_server_dc test_client_borrowed_tree
    test_cluster_shards test_cluster_shards_dc
    test_cluster_shards_dc_rht_nic_w4 test_cluster_shards_w4
    test_cluster_grpc_peer test_cluster_node_join_grpc_peer test_client_no_server_grpc_peer
    test_data_registry
)
foreach(_t IN LISTS _skippable_tests)
    if(TEST ${_t})
        set_tests_properties(${_t} PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endforeach()

# The cluster tests run on the RDMA RHT and, but for <test>_grpc_peer, the rdma
# peer index; with RADIXSHMEM_TEST_REDIS the main ones also run on the Redis RHT
# (<test>_redis) and on the defaults, the Redis RHT and the grpc peer index
# (<test>_no_rdma).
foreach(_t IN LISTS _skippable_tests)
    if(TEST ${_t} AND NOT _t STREQUAL "test_data_registry")
        set_property(TEST ${_t} APPEND PROPERTY ENVIRONMENT "RADIXSHMEM_RHT_BACKEND=rdma")
        if(NOT _t MATCHES "_grpc_peer$")
            set_property(TEST ${_t} APPEND PROPERTY ENVIRONMENT
                         "RADIXSHMEM_PEER_INDEX_BACKEND=rdma")
        endif()
    endif()
endforeach()
set(_redis_rht_env "RADIXSHMEM_RHT_BACKEND=redis;RADIXSHMEM_REDIS_ADDRS=${RADIXSHMEM_TEST_REDIS}")
if(RADIXSHMEM_TEST_REDIS)
    foreach(_pair test_cluster:test_cluster test_cluster_node_join:test_cluster_node_join
                  test_client_no_server:test_client_no_server
                  test_remote_api_grpc:test_remote_api)
        string(REPLACE ":" ";" _p ${_pair})
        list(GET _p 0 _name)
        list(GET _p 1 _exe)
        if(TEST ${_name})
            add_test(NAME ${_name}_redis COMMAND ${_exe})
            set_tests_properties(${_name}_redis PROPERTIES SKIP_RETURN_CODE 77
                ENVIRONMENT "${_redis_rht_env};RADIXSHMEM_PEER_INDEX_BACKEND=rdma")
            # The default peer index (grpc) on the Redis RHT uses no RDMA: a device
            # that does not exist fails any plane that would open one.
            add_test(NAME ${_name}_no_rdma COMMAND ${_exe})
            set_tests_properties(${_name}_no_rdma PROPERTIES SKIP_RETURN_CODE 77
                ENVIRONMENT "${_redis_rht_env};RADIXSHMEM_INDEX_DEV=no_rdma_device")
        endif()
    endforeach()
endif()

# ==================== Benchmarks (NOT in ctest) ====================
set(_benches
    bench_insert
    bench_evict
    bench_multiprocess
    bench_query
    bench_query_dist
    bench_cluster
    stress_test_llm
    bench_peer_query
)
foreach(_b ${_benches})
    add_executable(${_b} tests/bench/${_b}.cpp)
    target_link_libraries(${_b} PRIVATE radixshmem)
    # bench/* may include test_utils.h, now under tests/indexer/.
    target_include_directories(${_b} PRIVATE ${CMAKE_SOURCE_DIR}/tests/indexer)
    add_dependencies(${_b} radix_server)
endforeach()

# Remote-op correctness smoke on a 2-rank cluster; manual quick check (needs RDMA device, not in ctest).
add_executable(remote_op_smoke tests/bench/remote_op_smoke.cpp)
target_link_libraries(remote_op_smoke PRIVATE radixshmem)

# ==================== Data plane: radixshmem_data (storage + transfer) ====================
# Deliberately independent of libradixshmem. The ONLY coupling between the control
# plane (indexer) and the data plane is the int32_t slot id, so this target must
# never link radixshmem nor include its headers — the data-plane tests below link
# radixshmem_data alone, which is what keeps that honest. radixshmem_registry is the
# one exception, and not a violation: etcd is plane-neutral, hence src/registry.
set(_radixshmem_data_sources
    # storage: the shm slot pool (slot id -> address, one registerable range).
    src/storage/shm_region.cpp
    src/storage/slot_store.cpp
    # transfer: move bytes between registered ranges; callers never see which
    # backend does it.
    src/transfer/range_merge.cpp
    src/transfer/backend_factory.cpp
    # slot ids -> address ranges, for radixshmem.transfer.SlotTransfer.
    src/transfer/slot_plan.cpp)
add_library(radixshmem_data STATIC ${_radixshmem_data_sources})
target_include_directories(radixshmem_data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(radixshmem_data PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/storage
    ${CMAKE_CURRENT_SOURCE_DIR}/src/transfer)
# radixshmem_registry, not radixshmem: its own geometry under /data/, no indexer.
target_link_libraries(radixshmem_data PUBLIC rt pthread radixshmem_registry ${CMAKE_DL_LIBS})
# The Python extension links this archive into a shared object.
set_property(TARGET radixshmem_data PROPERTY POSITION_INDEPENDENT_CODE ON)

# Transfer plugins: the engine and what it links stay in their own .so, which create_transfer_backend
# loads from the directory of _core (or RADIXSHMEM_TRANSFER_PLUGIN_DIR); only the entry point is exported.
if(SKBUILD)
    set(_radixshmem_plugin_dir ${CMAKE_BINARY_DIR}/lib)
else()
    set(_radixshmem_plugin_dir ${CMAKE_SOURCE_DIR}/python/radixshmem)
endif()
set(_radixshmem_plugin_map ${CMAKE_BINARY_DIR}/radixshmem_plugin.map)
file(WRITE ${_radixshmem_plugin_map} "{ global: radixshmem_create_transfer_backend; local: *; };\n")
function(radixshmem_transfer_plugin name source include_dir)
    add_library(radixshmem_${name} MODULE ${source} src/transfer/range_merge.cpp)
    # The engines' headers require C++20 (<span>) while this project is C++17.
    set_source_files_properties(${source} PROPERTIES COMPILE_OPTIONS "-std=c++20")
    target_include_directories(radixshmem_${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/transfer ${include_dir})
    target_link_libraries(radixshmem_${name} PRIVATE ${ARGN})
    target_link_options(radixshmem_${name} PRIVATE "LINKER:--version-script=${_radixshmem_plugin_map}")
    set_target_properties(radixshmem_${name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${_radixshmem_plugin_dir}
        INSTALL_RPATH "$ORIGIN" INSTALL_RPATH_USE_LINK_PATH ON)
    # Built with anything that links radixshmem_data, so the data plane always finds it.
    add_dependencies(radixshmem_data radixshmem_${name})
    if(SKBUILD)
        install(TARGETS radixshmem_${name} LIBRARY DESTINATION radixshmem)
    endif()
endfunction()
if(RADIXSHMEM_MOONCAKE_FOUND)
    radixshmem_transfer_plugin(mooncake src/transfer/mooncake/mooncake_backend.cpp
                               ${MOONCAKE_INCLUDE_DIR} ${MOONCAKE_LINK_LIBS})
endif()

set(_dataplane_tests test_slot_store)
if(RADIXSHMEM_MOONCAKE_FOUND)
    list(APPEND _dataplane_tests test_mooncake_loopback)
endif()
foreach(_t ${_dataplane_tests})
    add_executable(${_t} tests/dataplane/${_t}.cpp)
    target_link_libraries(${_t} PRIVATE radixshmem_data)   # NOT radixshmem — by design
    target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/tests/dataplane)
    radixshmem_enable_asserts(${_t})
    add_test(NAME ${_t} COMMAND ${_t})
    set_tests_properties(${_t} PROPERTIES
        ENVIRONMENT "RADIXSHMEM_TRANSFER_PLUGIN_DIR=${_radixshmem_plugin_dir}")
endforeach()

# ---------- radixshmem._core: the pybind11 extension module (index and data plane) ----------
option(BUILD_PYTHON_MODULES "Build the radixshmem._core pybind11 extension" ON)
if(BUILD_PYTHON_MODULES)
    find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)
    if(Python3_FOUND)
        execute_process(COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
            OUTPUT_VARIABLE _pybind11_cmakedir OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _pybind11_rc ERROR_QUIET)
        if(_pybind11_rc EQUAL 0)
            find_package(pybind11 CONFIG QUIET
                PATHS ${_pybind11_cmakedir} NO_DEFAULT_PATH)
        endif()
    endif()
    if(SKBUILD AND NOT pybind11_FOUND)
        message(FATAL_ERROR "pybind11 not found for ${Python3_EXECUTABLE}; a wheel needs the extension module")
    endif()
    if(pybind11_FOUND)
        # A pip build installs the module into the wheel; a plain build puts it in the source tree.
        if(SKBUILD)
            set(_radixshmem_py_dir ${CMAKE_BINARY_DIR}/lib)
        else()
            set(_radixshmem_py_dir ${CMAKE_SOURCE_DIR}/python/radixshmem)
        endif()

        pybind11_add_module(radixshmem_core_py python/bindings.cpp python/data_bindings.cpp)
        set_target_properties(radixshmem_core_py PROPERTIES
            OUTPUT_NAME _core
            LIBRARY_OUTPUT_DIRECTORY ${_radixshmem_py_dir}
            # libradix_etcd.so lives in the build tree.
            BUILD_RPATH ${CMAKE_BINARY_DIR}/lib)
        target_link_libraries(radixshmem_core_py PRIVATE radixshmem radixshmem_data)
        # Only PyInit__core stays exported; the installed module finds libradix_etcd.so beside it.
        set(_radixshmem_py_map ${CMAKE_BINARY_DIR}/radixshmem_py.map)
        file(WRITE ${_radixshmem_py_map} "{ global: PyInit_*; local: *; };\n")
        target_link_options(radixshmem_core_py PRIVATE "LINKER:--version-script=${_radixshmem_py_map}")
        set_target_properties(radixshmem_core_py PROPERTIES
            INSTALL_RPATH "$ORIGIN" INSTALL_RPATH_USE_LINK_PATH ON)
        if(SKBUILD)
            install(TARGETS radixshmem_core_py LIBRARY DESTINATION radixshmem)
            install(FILES ${_etcd_go_lib} DESTINATION radixshmem)
        endif()
        # rdma-core is loaded at run time, so neither the module nor radix_server may link it.
        add_test(NAME test_verbs_not_linked
                 COMMAND sh -c "for f in \"$@\"; do nm -D --undefined-only \"$f\"; readelf -d \"$f\"; done | grep -E ' (ibv|mlx5dv)_|lib(ibverbs|mlx5)' && exit 1 || exit 0"
                         sh $<TARGET_FILE:radixshmem_core_py> $<TARGET_FILE:radix_server>)
        # In ctest, unlike test_radixshmem.py: this one needs no server and no
        # numpy, so it can be part of the gate rather than a manual step.
        add_test(NAME test_data_python
                 COMMAND ${Python3_EXECUTABLE} tests/test_radixshmem_data.py
                 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python)
        set_tests_properties(test_data_python PROPERTIES
            DEPENDS radixshmem_core_py TIMEOUT 300
            ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/python")

        execute_process(COMMAND ${Python3_EXECUTABLE} -c "import pytest"
            RESULT_VARIABLE _pytest_rc OUTPUT_QUIET ERROR_QUIET)
        foreach(_t test_geometry test_server_rpc test_radixshmem test_geometry_registry test_writeback
                   test_monitor test_transfer_jobs test_slot_transfer)
            if(_pytest_rc EQUAL 0)
                add_test(NAME py_${_t}
                         COMMAND ${Python3_EXECUTABLE} -m pytest -q tests/${_t}.py)
            else()
                add_test(NAME py_${_t}
                         COMMAND ${Python3_EXECUTABLE} -c "print('SKIP: pytest not installed'); raise SystemExit(77)")
            endif()
        endforeach()
        add_test(NAME py_test_cluster COMMAND ${Python3_EXECUTABLE} tests/test_cluster.py)
        add_test(NAME py_test_p2p_pull COMMAND ${Python3_EXECUTABLE} tests/test_p2p_pull.py)
        add_test(NAME py_test_dedup_move COMMAND ${Python3_EXECUTABLE} tests/test_dedup_move.py)
        add_test(NAME py_test_writeback_pull COMMAND ${Python3_EXECUTABLE} tests/test_writeback_pull.py)
        add_test(NAME py_test_push COMMAND ${Python3_EXECUTABLE} tests/test_push.py)
        set(_stress_args --concurrency 8 --duration 10 --warmup-sec 2 --turns 3,6
            --isl-step 50,100 --osl 10,30 --max-blocks 200000 --max-nodes 200000
            --swa-max-blocks 50000 --mamba-max-slots 10000 --report-sec 5)
        add_test(NAME py_agentic_stress_1node
                 COMMAND ${Python3_EXECUTABLE} tests/agentic_stress.py --nodes 1
                         --base-port 19100 ${_stress_args})
        add_test(NAME py_agentic_stress_2node
                 COMMAND ${Python3_EXECUTABLE} tests/agentic_stress.py --nodes 2
                         --base-port 19200 ${_stress_args})
        add_test(NAME py_agentic_stress_3node_threads
                 COMMAND ${Python3_EXECUTABLE} tests/agentic_stress.py --nodes 3
                         --base-port 19300 ${_stress_args} --concurrency 32
                         --executors-per-worker 8 --rht-slots 4 --register-chunk-size 1
                         --flush-after-insert)
        foreach(_t py_test_geometry py_test_server_rpc py_test_radixshmem py_test_geometry_registry
                   py_test_writeback py_test_monitor py_test_transfer_jobs py_test_slot_transfer
                   py_test_cluster py_test_p2p_pull
                   py_test_dedup_move
                   py_test_writeback_pull
                   py_test_push py_agentic_stress_1node py_agentic_stress_2node py_agentic_stress_3node_threads)
            set_tests_properties(${_t} PROPERTIES
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python
                ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/python;RADIXSHMEM_RHT_BACKEND=rdma;RADIXSHMEM_PEER_INDEX_BACKEND=rdma"
                SKIP_RETURN_CODE 77 TIMEOUT 600)
        endforeach()
        set_tests_properties(py_agentic_stress_3node_threads PROPERTIES TIMEOUT 180)
        # The peer index over gRPC remote ops instead of RDMA walks.
        add_test(NAME py_test_cluster_grpc_peer COMMAND ${Python3_EXECUTABLE} tests/test_cluster.py)
        add_test(NAME py_agentic_stress_2node_grpc_peer
                 COMMAND ${Python3_EXECUTABLE} tests/agentic_stress.py --nodes 2
                         --base-port 19700 ${_stress_args})
        foreach(_t py_test_cluster_grpc_peer py_agentic_stress_2node_grpc_peer)
            set_tests_properties(${_t} PROPERTIES
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python
                ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/python;RADIXSHMEM_RHT_BACKEND=rdma;RADIXSHMEM_PEER_INDEX_BACKEND=grpc"
                SKIP_RETURN_CODE 77 TIMEOUT 600)
        endforeach()
        if(RADIXSHMEM_TEST_REDIS)
            add_test(NAME py_test_cluster_redis COMMAND ${Python3_EXECUTABLE} tests/test_cluster.py)
            add_test(NAME py_test_p2p_pull_redis COMMAND ${Python3_EXECUTABLE} tests/test_p2p_pull.py)
            add_test(NAME py_agentic_stress_2node_redis
                     COMMAND ${Python3_EXECUTABLE} tests/agentic_stress.py --nodes 2
                             --base-port 19500 ${_stress_args})
            add_test(NAME py_agentic_stress_3node_threads_redis
                     COMMAND ${Python3_EXECUTABLE} tests/agentic_stress.py --nodes 3
                             --base-port 19600 ${_stress_args} --concurrency 32
                             --executors-per-worker 8 --register-chunk-size 1
                             --flush-after-insert)
            foreach(_t py_test_cluster_redis py_test_p2p_pull_redis py_agentic_stress_2node_redis
                       py_agentic_stress_3node_threads_redis)
                set_tests_properties(${_t} PROPERTIES
                    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python
                    ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/python;${_redis_rht_env};RADIXSHMEM_PEER_INDEX_BACKEND=rdma"
                    SKIP_RETURN_CODE 77 TIMEOUT 600)
            endforeach()
            # The defaults, Redis RHT + grpc peer index: the index planes use no RDMA.
            add_test(NAME py_agentic_stress_3node_threads_no_rdma
                     COMMAND ${Python3_EXECUTABLE} tests/agentic_stress.py --nodes 3
                             --base-port 19800 ${_stress_args} --concurrency 32
                             --executors-per-worker 8 --register-chunk-size 1
                             --flush-after-insert)
            set_tests_properties(py_agentic_stress_3node_threads_no_rdma PROPERTIES
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python
                ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}/python;${_redis_rht_env};RADIXSHMEM_INDEX_DEV=no_rdma_device"
                SKIP_RETURN_CODE 77 TIMEOUT 600)
        endif()
        message(STATUS "radixshmem._core: building (pybind11 ${pybind11_VERSION}, ${Python3_EXECUTABLE})")
    else()
        message(STATUS "radixshmem._core: skipped (no pybind11 for this Python)")
    endif()
endif()

# Legacy / historical PoCs — kept under tests/legacy/ but not built.
# To build manually: g++ ... tests/legacy/{file}.cpp

enable_testing()

# A pip build, editable rebuilds included, builds the module and what it links, no tools or tests.
if(SKBUILD)
    get_property(_radixshmem_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_t ${_radixshmem_targets})
        if(NOT _t STREQUAL "radixshmem_core_py")
            set_target_properties(${_t} PROPERTIES EXCLUDE_FROM_ALL ON)
        endif()
    endforeach()
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
