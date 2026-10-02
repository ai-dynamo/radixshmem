# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)
project(RadixShmem VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# RDMA support (libibverbs) is optional; disable on hosts without IB headers.
option(ENABLE_RDMA "Build RDMA-dependent sources (requires libibverbs)" ON)

# ZeroMQ (libzmq) is optional — powers the zmq remote-op control plane backend.
option(ENABLE_ZMQ "Build ZeroMQ-dependent remote-insert sources (requires libzmq)" ON)

option(ENABLE_ETCD "Build the etcd cluster-bootstrap registry (requires Go toolchain)" ON)

# Mooncake TransferEngine backs the data plane's transfer backend (src/transfer).
option(ENABLE_MOONCAKE "Build the Mooncake transfer backend (requires Mooncake headers + lib)" ON)
set(MOONCAKE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mooncake" CACHE PATH
    "Mooncake source + build root")

# NIXL (backend="UCX") backs the data plane's other transfer backend (src/transfer/nixl).
# `scripts/build_nixl.sh` builds NIXL from source, pinned to the commit this
# backend was written and end-to-end tested against, and installs it at
# exactly this default path — run it once and NIXL_ROOT needs no override.
option(ENABLE_NIXL "Build the NIXL transfer backend (requires NIXL headers + lib)" ON)
set(NIXL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nixl-install" CACHE PATH
    "NIXL install prefix (meson --prefix), holding include/nixl.h and lib*/libnixl.so")

# ==================== shmradix library ====================
set(_shmradix_sources
    # ==================== core: shm pools + radix tree engine ====================
    src/indexer/core/hash.cpp
    src/indexer/core/node_pool.cpp
    src/indexer/core/slot_lru.cpp
    src/indexer/core/data_pool.cpp
    src/indexer/core/child_table.cpp
    src/indexer/core/mempool.cpp
    src/indexer/core/radix_tree.cpp
    # component match semantics (Full/SWA/Mamba), shared by local + RDMA walkers
    src/indexer/component/component_set.cpp
    src/indexer/component/swa_component.cpp
    src/indexer/component/mamba_component.cpp
    # aux chain primitives + per-node retire + touch (eviction lifecycle hooks)
    src/indexer/component/aux_list.cpp
    # RadixServer: shm region + evictor + (distributed) cluster bring-up. Self-guards
    # RDMA use; must compile in both RDMA and non-RDMA builds.
    src/indexer/server.cpp
    src/indexer/local.cpp
    # Lock-free leak audit (check_health); friend of ShmRadixTree, no other coupling.
    src/indexer/health_check.cpp)
if(ENABLE_RDMA)
    list(APPEND _shmradix_sources
        # RHT (router hash table): shard shm + routing engine (core), holder +
        # consumer endpoints.
        src/indexer/rht/rht_core.cpp
        src/indexer/rht/rht_endpoint.cpp
        # RDMA transport plane: verbs primitives split by connection model.
        src/indexer/transport/rdma/context.cpp
        src/indexer/transport/rdma/rc_endpoint.cpp
        src/indexer/transport/rdma/xrc.cpp
        src/indexer/transport/rdma/dc_oneside.cpp
        src/indexer/client.cpp
        # Peer-facing plane: lazy data-plane endpoints + remote-walk executor +
        # remote-op session factory + client-side remote-op protocol (RemotePeer).
        src/indexer/peer/peer_connector.cpp
        src/indexer/peer/peer_breaker.cpp
        src/indexer/peer/rdma_remote_walker.cpp
        src/indexer/peer/remote_op.cpp
        # etcd cluster rendezvous; indexer-side (holds a RadixServer&).
        src/indexer/etcd_registry.cpp
        # Bilateral (two-sided SEND/RECV) transport backends behind IRemoteOp*Transport,
        # filed by protocol: DC under rdma/ (alongside the one-sided plane), ZMQ under tcp/.
        src/indexer/transport/rdma/dc_bilateral.cpp
        # Server-side remote-op service (RadixServer::rc_* pollers + handlers);
        # needed by the RDMA backend even without ZMQ.
        src/indexer/remote_op_service.cpp
        # TCP backend (libzmq REQ/REP); self-guarded on SHMRADIX_HAS_ZMQ (empty TU when absent).
        src/indexer/transport/tcp/zmq_transport.cpp
    )
endif()
add_library(shmradix STATIC ${_shmradix_sources})
set_property(TARGET shmradix PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(shmradix PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    # core internal headers (node_pool.hpp etc.) live under src/indexer/core; radix_tree.h
    # (public) embeds these pools, so consumers must resolve the .hpp names too.
    ${CMAKE_CURRENT_SOURCE_DIR}/src/indexer/core
    # transport-plane headers (rdma_transport.hpp, dc_oneside/dc_bilateral, xrc ...)
    # live under src/indexer/transport/rdma; public cluster headers embed transport members,
    # so consumers resolve these too.
    ${CMAKE_CURRENT_SOURCE_DIR}/src/indexer/transport/rdma
    # TCP/ZMQ bilateral backend header is private under src/indexer/transport/tcp;
    # consumers include it as "zmq_transport.hpp".
    ${CMAKE_CURRENT_SOURCE_DIR}/src/indexer/transport/tcp
    # peer-plane headers (peer_connector.hpp, peer_registry.hpp, xrc_connect.hpp,
    # net_util.hpp) are private under src/indexer/peer.
    ${CMAKE_CURRENT_SOURCE_DIR}/src/indexer/peer
)

target_include_directories(shmradix PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/indexer/component
    ${CMAKE_CURRENT_SOURCE_DIR}/src/indexer/rht
)

set(_shmradix_link_libs xxhash rt pthread)
if(ENABLE_RDMA)
    list(APPEND _shmradix_link_libs ibverbs mlx5)   # mlx5 for DC (Direct Verbs)
    # CMake-side switches consumed by configure_file() into shmradix/config.h.
    set(SHMRADIX_REMOTE_OP_RDMA ON)
else()
    set(SHMRADIX_NO_RDMA ON)
endif()
target_link_libraries(shmradix PUBLIC ${_shmradix_link_libs})

# ==================== ZeroMQ (remote-insert control plane) ====================
# Probe libzmq via pkg-config; on success the ZMQ backend compiles in. RDMA is a prerequisite.
set(SHMRADIX_ZMQ_FOUND OFF)
if(ENABLE_ZMQ AND ENABLE_RDMA)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(ZMQ QUIET libzmq)
    endif()
    if(ZMQ_FOUND)
        set(SHMRADIX_ZMQ_FOUND ON)
        # ZMQ sources already compiled via _shmradix_sources (RDMA block), self-guarded.
        target_include_directories(shmradix PUBLIC ${ZMQ_INCLUDE_DIRS})
        target_link_libraries(shmradix PUBLIC ${ZMQ_LIBRARIES})
        target_link_directories(shmradix PUBLIC ${ZMQ_LIBRARY_DIRS})
        set(SHMRADIX_HAS_ZMQ ON)
        message(STATUS "ZeroMQ found (${ZMQ_VERSION}) — remote-insert enabled")
    else()
        message(STATUS "ZeroMQ not found — remote-insert (zmq backend) disabled")
    endif()
else()
    message(STATUS "ZeroMQ disabled (ENABLE_ZMQ=${ENABLE_ZMQ}, ENABLE_RDMA=${ENABLE_RDMA})")
endif()

# ==================== shmradix_registry: the etcd plane (Go client wrapper) ====================
# The official Go etcd client compiled to a c-shared lib via `go build -buildmode=c-shared`,
# which also emits libradix_etcd.h (the C surface). grpc/protobuf are statically bundled into
# the .so, so the C++ side gains no heavy deps. RDMA is a prerequisite (no cluster without it).
set(SHMRADIX_ETCD_FOUND OFF)
if(ENABLE_ETCD AND ENABLE_RDMA)
    find_program(GO_EXECUTABLE go)
    if(NOT GO_EXECUTABLE)
        message(FATAL_ERROR
            "ENABLE_ETCD=ON but no `go` toolchain found. The cluster bootstrap "
            "registry requires the Go etcd client. Install Go, or reconfigure with "
            "-DENABLE_ETCD=OFF to build without cluster support.")
    endif()
    set(_etcd_go_dir  ${CMAKE_CURRENT_SOURCE_DIR}/src/registry/etcd_go)
    set(_etcd_go_lib  ${CMAKE_BINARY_DIR}/lib/libradix_etcd.so)
    set(_etcd_go_hdr  ${CMAKE_BINARY_DIR}/include/libradix_etcd.h)
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include)
    # Build the wrapper into the build tree. GOPROXY carries a CN mirror fallback so
    # `go mod tidy` resolves etcd modules on restricted networks (Mooncake's pattern).
    add_custom_command(
        OUTPUT ${_etcd_go_lib} ${_etcd_go_hdr}
        COMMAND ${CMAKE_COMMAND} -E env
                GOPROXY=https://proxy.golang.org,https://goproxy.cn,direct
                ${GO_EXECUTABLE} mod tidy
        COMMAND ${CMAKE_COMMAND} -E env
                GOPROXY=https://proxy.golang.org,https://goproxy.cn,direct
                ${GO_EXECUTABLE} build -buildmode=c-shared -o ${_etcd_go_lib} radix_etcd.go
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_BINARY_DIR}/lib/libradix_etcd.h ${_etcd_go_hdr}
        WORKING_DIRECTORY ${_etcd_go_dir}
        DEPENDS ${_etcd_go_dir}/radix_etcd.go ${_etcd_go_dir}/go.mod
        COMMENT "Building Go etcd client wrapper (libradix_etcd.so)"
        VERBATIM)
    add_custom_target(radix_etcd_wrapper DEPENDS ${_etcd_go_lib} ${_etcd_go_hdr})
    set(SHMRADIX_ETCD_FOUND ON)
    set(SHMRADIX_HAS_ETCD ON)
    message(STATUS "etcd registry enabled (Go wrapper via ${GO_EXECUTABLE})")
else()
    message(STATUS "etcd registry disabled (ENABLE_ETCD=${ENABLE_ETCD}, ENABLE_RDMA=${ENABLE_RDMA}) — no cluster bootstrap")
endif()

# Its own plane, not the indexer's: both planes key off the same etcd, so both
# may link this. Defined unconditionally — every TU under src/registry guards on
# SHMRADIX_HAS_ETCD, so an etcd-less build just yields an empty archive.
add_library(shmradix_registry STATIC src/registry/data_registry.cpp)
target_include_directories(shmradix_registry PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include                      # config.h + libradix_etcd.h
    # etcd_client.hpp / registry_client.hpp — private, included by bare name.
    ${CMAKE_CURRENT_SOURCE_DIR}/src/registry)
target_link_libraries(shmradix_registry PUBLIC rt pthread)
set_property(TARGET shmradix_registry PROPERTY POSITION_INDEPENDENT_CODE ON)
if(SHMRADIX_ETCD_FOUND)
    add_dependencies(shmradix_registry radix_etcd_wrapper)
    target_link_libraries(shmradix_registry PUBLIC ${_etcd_go_lib})
endif()

target_link_libraries(shmradix PUBLIC shmradix_registry)

# ==================== Mooncake transfer engine (data plane) ====================
# Probe only — the backend source lands with src/transfer.
#
# Headers and library MUST come from the same Mooncake checkout: TransferMetadata
# 's descriptor structs change shape between revisions, and a stale archive
# against fresh headers links cleanly and then hands back garbage (empty buffer
# lists, in practice). So the library is looked up under MOONCAKE_BUILD_DIR,
# which lives inside MOONCAKE_ROOT, and the headers come from that same tree --
# never from a /usr/local install of unknown vintage.
#
# Build Mooncake's transfer engine alone with:
#   cmake -S <root> -B <root>/build-te -DCMAKE_BUILD_TYPE=Release -DWITH_TE=ON \
#         -DWITH_STORE=OFF -DWITH_STORE_RUST=OFF -DWITH_P2P_STORE=OFF \
#         -DBUILD_UNIT_TESTS=OFF -DUSE_ETCD=OFF && cmake --build <root>/build-te \
#         --target transfer_engine
set(MOONCAKE_BUILD_DIR "${MOONCAKE_ROOT}/build-te" CACHE PATH
    "Mooncake build tree holding libtransfer_engine (must match MOONCAKE_ROOT's sources)")
set(SHMRADIX_MOONCAKE_FOUND OFF)
if(ENABLE_MOONCAKE)
    find_path(MOONCAKE_INCLUDE_DIR transfer_engine.h
        HINTS ${MOONCAKE_ROOT}/mooncake-transfer-engine/include NO_DEFAULT_PATH)
    find_library(MOONCAKE_LIBRARY NAMES transfer_engine
        HINTS ${MOONCAKE_BUILD_DIR}/mooncake-transfer-engine/src NO_DEFAULT_PATH)
    # The static build splits into several archives and needs its transitive deps
    # spelled out; a shared build (-DBUILD_SHARED_LIBS=ON) would carry its own.
    find_library(MOONCAKE_BASE_LIBRARY NAMES base
        HINTS ${MOONCAKE_BUILD_DIR}/mooncake-transfer-engine/src/common/base NO_DEFAULT_PATH)
    find_library(MOONCAKE_COMMON_LIBRARY NAMES mooncake_common
        HINTS ${MOONCAKE_BUILD_DIR}/mooncake-common/src NO_DEFAULT_PATH)
    find_library(MOONCAKE_ASIO_LIBRARY NAMES asio
        HINTS ${MOONCAKE_BUILD_DIR}/mooncake-common NO_DEFAULT_PATH)
    if(MOONCAKE_INCLUDE_DIR AND MOONCAKE_LIBRARY)
        set(SHMRADIX_MOONCAKE_FOUND ON)
        set(SHMRADIX_HAS_MOONCAKE ON)
        set(MOONCAKE_LINK_LIBS
            ${MOONCAKE_LIBRARY}
            ${MOONCAKE_BASE_LIBRARY}
            ${MOONCAKE_COMMON_LIBRARY}
            ${MOONCAKE_ASIO_LIBRARY}
            glog gflags jsoncpp numa ibverbs curl pthread dl)
        message(STATUS "Mooncake found (${MOONCAKE_LIBRARY}) — transfer backend enabled")
    else()
        message(STATUS "Mooncake not found (include=${MOONCAKE_INCLUDE_DIR}, "
                       "lib=${MOONCAKE_LIBRARY}) — transfer backend disabled")
    endif()
else()
    message(STATUS "Mooncake disabled (ENABLE_MOONCAKE=OFF)")
endif()

# ==================== NIXL transfer engine (data plane) ====================
# Probe only — the backend source lands with src/transfer. Same headers/lib
# provenance rule as Mooncake above: both must come from the one NIXL_ROOT
# install (meson --prefix=NIXL_ROOT install), never a stray /usr/local one —
# run scripts/build_nixl.sh to produce one at the default NIXL_ROOT.
set(SHMRADIX_NIXL_FOUND OFF)
if(ENABLE_NIXL)
    find_path(NIXL_INCLUDE_DIR nixl.h
        HINTS ${NIXL_ROOT}/include NO_DEFAULT_PATH)
    find_library(NIXL_LIBRARY NAMES nixl
        HINTS ${NIXL_ROOT}/lib/x86_64-linux-gnu ${NIXL_ROOT}/lib64 ${NIXL_ROOT}/lib
        NO_DEFAULT_PATH)
    find_library(NIXL_BUILD_LIBRARY NAMES nixl_build
        HINTS ${NIXL_ROOT}/lib/x86_64-linux-gnu ${NIXL_ROOT}/lib64 ${NIXL_ROOT}/lib
        NO_DEFAULT_PATH)
    find_library(NIXL_COMMON_LIBRARY NAMES nixl_common
        HINTS ${NIXL_ROOT}/lib/x86_64-linux-gnu ${NIXL_ROOT}/lib64 ${NIXL_ROOT}/lib
        NO_DEFAULT_PATH)
    if(NIXL_INCLUDE_DIR AND NIXL_LIBRARY)
        set(SHMRADIX_NIXL_FOUND ON)
        set(SHMRADIX_HAS_NIXL ON)
        # The UCX plugin (plugins/libplugin_UCX.so) is dlopen'd by libnixl at
        # runtime via NIXL_PLUGIN_DIR / its compiled-in default — not linked here.
        set(NIXL_LINK_LIBS ${NIXL_LIBRARY})
        foreach(_lib NIXL_BUILD_LIBRARY NIXL_COMMON_LIBRARY)
            if(${_lib})
                list(APPEND NIXL_LINK_LIBS ${${_lib}})
            endif()
        endforeach()
        get_filename_component(NIXL_LIBRARY_DIR ${NIXL_LIBRARY} DIRECTORY)
        message(STATUS "NIXL found (${NIXL_LIBRARY}) — transfer backend enabled")
    else()
        message(STATUS "NIXL not found (include=${NIXL_INCLUDE_DIR}, "
                       "lib=${NIXL_LIBRARY}) — transfer backend disabled")
    endif()
else()
    message(STATUS "NIXL disabled (ENABLE_NIXL=OFF)")
endif()

# ==================== Generated feature-config header ====================
# Feature macros baked into one generated header every consumer includes.
# Generated into the BUILD tree so parallel build dirs keep separate configs.
configure_file(include/shmradix/config.h.in
               ${CMAKE_BINARY_DIR}/include/shmradix/config.h)
target_include_directories(shmradix PUBLIC ${CMAKE_BINARY_DIR}/include)

# ==================== radix_server tool (launched by ServerLauncher) ====================
if(ENABLE_RDMA)
    add_executable(radix_server tools/server_main.cpp)
    target_link_libraries(radix_server PRIVATE shmradix)
endif()

# ==================== Unit tests ====================
# Keep assert() live in test TUs under Release (trailing -UNDEBUG overrides -DNDEBUG).
function(shmradix_enable_asserts _tgt)
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
    # Consolidated query_components coverage (match/pin/facade; the remote
    # sections self-gate on SHMRADIX_NO_RDMA inside the file).
    test_component_query
    # Aux chain lifecycle (EV1): publish/touch/split/merge/cascade/reuse.
    test_component_evict
    # RhtEngine over a fake one-sided endpoint: scratch-bounded lookup rounds,
    # remote insert/remove plumbing (self-gates on SHMRADIX_NO_RDMA).
    test_rht_engine
    # PeerBreaker: skip/probe/backoff of an unreachable peer.
    test_peer_breaker
)
foreach(_t ${_core_unit_tests})
    add_executable(${_t} tests/indexer/core/${_t}.cpp)
    target_link_libraries(${_t} PRIVATE shmradix)
    target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/tests/indexer
                                             ${CMAKE_SOURCE_DIR}/src/indexer/rht)
    shmradix_enable_asserts(${_t})
    if(ENABLE_RDMA)
        add_dependencies(${_t} radix_server)
    endif()
    add_test(NAME ${_t} COMMAND ${_t})
endforeach()

if(ENABLE_RDMA)
    # XRC shared-state cross-process primitive test (no shmradix needed).
    add_executable(test_xrc_state tests/indexer/test_xrc_state.cpp)
    target_link_libraries(test_xrc_state PRIVATE shmradix)
    shmradix_enable_asserts(test_xrc_state)
    add_test(NAME test_xrc_state COMMAND test_xrc_state)


    # Cross-process distributed integration test (writer/reader/lockdec/...).
    add_executable(test_cluster tests/indexer/test_cluster.cpp)
    target_link_libraries(test_cluster PRIVATE shmradix)
    shmradix_enable_asserts(test_cluster)
    add_test(NAME test_cluster COMMAND test_cluster)

    # Phase-3 node-join: 2-node cluster + a late data-peer joiner (join_existing),
    # existing reader senses it via /peers watcher and remote-walks it lazily.
    add_executable(test_cluster_node_join tests/indexer/test_cluster_node_join.cpp)
    target_link_libraries(test_cluster_node_join PRIVATE shmradix)
    shmradix_enable_asserts(test_cluster_node_join)
    add_test(NAME test_cluster_node_join COMMAND test_cluster_node_join)

    # Observer (no local tree, rank -1): pure client discovers the cluster from
    # etcd, remote-walks a writer's prefix, and rejects every local-tree op.
    add_executable(test_client_no_server tests/indexer/test_client_no_server.cpp)
    target_link_libraries(test_client_no_server PRIVATE shmradix)
    shmradix_enable_asserts(test_client_no_server)
    add_test(NAME test_client_no_server COMMAND test_client_no_server)
    add_test(NAME test_client_no_server_dc COMMAND test_client_no_server)
    set_tests_properties(test_client_no_server_dc PROPERTIES
        ENVIRONMENT "SHMRADIX_RHT_TRANSPORT=dc;SHMRADIX_PEER_INDEX_TRANSPORT=dc")

    # M < N RHT-shard decoupling test (N=3, M=2, holders={0,1}).
    add_executable(test_cluster_shards tests/indexer/test_cluster_shards.cpp)
    target_link_libraries(test_cluster_shards PRIVATE shmradix)
    shmradix_enable_asserts(test_cluster_shards)
    add_test(NAME test_cluster_shards COMMAND test_cluster_shards)

    # Same binaries re-run over the DC data-plane transport (incl. fetch-add atomics).
    add_test(NAME test_cluster_dc COMMAND test_cluster)
    set_tests_properties(test_cluster_dc PROPERTIES
        ENVIRONMENT "SHMRADIX_RHT_TRANSPORT=dc;SHMRADIX_PEER_INDEX_TRANSPORT=dc")
    # The two planes pick their transport independently: one DC, the other XRC.
    add_test(NAME test_cluster_rht_dc_peer_xrc COMMAND test_cluster)
    set_tests_properties(test_cluster_rht_dc_peer_xrc PROPERTIES
        ENVIRONMENT "SHMRADIX_RHT_TRANSPORT=dc")
    add_test(NAME test_cluster_rht_xrc_peer_dc COMMAND test_cluster)
    set_tests_properties(test_cluster_rht_xrc_peer_dc PROPERTIES
        ENVIRONMENT "SHMRADIX_PEER_INDEX_TRANSPORT=dc")
    add_test(NAME test_cluster_shards_dc COMMAND test_cluster_shards)
    set_tests_properties(test_cluster_shards_dc PROPERTIES
        ENVIRONMENT "SHMRADIX_RHT_TRANSPORT=dc;SHMRADIX_PEER_INDEX_TRANSPORT=dc")

    # DC + peer-shard RDMA: a rank only attaches its own shard, so every peer
    # shard's RHT access hits the DC one-sided WRITE path (write_async_inline +
    # flush) on the wire — no env needed, distinct node_name per rank already
    # forces it. W=4 also covers read-modify-write over the collision chain.
    add_test(NAME test_cluster_dc_rht_nic COMMAND test_cluster)
    set_tests_properties(test_cluster_dc_rht_nic PROPERTIES
        ENVIRONMENT "SHMRADIX_RHT_TRANSPORT=dc;SHMRADIX_PEER_INDEX_TRANSPORT=dc")
    add_test(NAME test_cluster_shards_dc_rht_nic_w4 COMMAND test_cluster_shards)
    set_tests_properties(test_cluster_shards_dc_rht_nic_w4 PROPERTIES
        ENVIRONMENT "SHMRADIX_RHT_TRANSPORT=dc;SHMRADIX_PEER_INDEX_TRANSPORT=dc;SHMRADIX_RHT_SLOTS=4")

    # Multi-slot RHT re-run with W=4 to exercise collision chaining.
    add_test(NAME test_cluster_w4 COMMAND test_cluster)
    set_tests_properties(test_cluster_w4 PROPERTIES
        ENVIRONMENT "SHMRADIX_RHT_SLOTS=4")
    add_test(NAME test_cluster_shards_w4 COMMAND test_cluster_shards)
    set_tests_properties(test_cluster_shards_w4 PROPERTIES
        ENVIRONMENT "SHMRADIX_RHT_SLOTS=4")
endif()

# ==================== Remote-op e2e gate ====================
# Explicit remote routing API on RadixClient; built whenever ENABLE_RDMA
# (remote-op is always compiled in RDMA builds), self-SKIPs (exit 77) without a device.
if(ENABLE_RDMA)
    add_executable(test_remote_api tests/indexer/test_remote_api.cpp)
    target_link_libraries(test_remote_api PRIVATE shmradix)

    # Two-variant matrix: base = zmq control × xrc data (zmq-gated); _dcd = both planes DC (superset).
    if(SHMRADIX_ZMQ_FOUND)
        add_test(NAME test_remote_api COMMAND test_remote_api)
    endif()
    add_test(NAME test_remote_api_dcd COMMAND test_remote_api)
    set_tests_properties(test_remote_api_dcd PROPERTIES
        ENVIRONMENT "SHMRADIX_RHT_TRANSPORT=dc;SHMRADIX_PEER_INDEX_TRANSPORT=dc;SHMRADIX_REMOTE_OP_TRANSPORT=dc")
endif()

# Data-plane geometry in the shared etcd registry. Needs an etcd binary but no
# RDMA device, and stays single-process (the Go wrapper deadlocks on a bare
# fork), so it is gated on the registry alone. Links shmradix_registry ALONE —
# neither plane's library — which keeps the registry plane-neutral.
if(SHMRADIX_HAS_ETCD)
    add_executable(test_data_registry tests/registry/test_data_registry.cpp)
    target_link_libraries(test_data_registry PRIVATE shmradix_registry)
    # etcd fixture from tests/indexer; runner from tests/dataplane (no indexer headers).
    target_include_directories(test_data_registry PRIVATE
        ${CMAKE_SOURCE_DIR}/tests/indexer
        ${CMAKE_SOURCE_DIR}/tests/dataplane)
    shmradix_enable_asserts(test_data_registry)
    add_test(NAME test_data_registry COMMAND test_data_registry)
endif()

# Tests that self-SKIP (exit 77) without RDMA device / python ext / etcd binary;
# register code so ctest shows "Skipped". The cluster tests fork a private etcd
# (the sole bootstrap backend) and SKIP when no etcd binary is available.
set(_skippable_tests
    test_remote_api test_remote_api_dcd
    test_cluster test_cluster_dc test_cluster_dc_rht_nic test_cluster_w4
    test_cluster_rht_dc_peer_xrc test_cluster_rht_xrc_peer_dc
    test_cluster_node_join
    test_client_no_server test_client_no_server_dc
    test_cluster_shards test_cluster_shards_dc
    test_cluster_shards_dc_rht_nic_w4 test_cluster_shards_w4
    test_data_registry
)
foreach(_t IN LISTS _skippable_tests)
    if(TEST ${_t})
        set_tests_properties(${_t} PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endforeach()

# ==================== Benchmarks (NOT in ctest) ====================
set(_benches
    bench_insert
    bench_evict
    bench_multiprocess
    bench_query
    bench_query_dist
    bench_cluster
    stress_test_llm
)
foreach(_b ${_benches})
    add_executable(${_b} tests/bench/${_b}.cpp)
    target_link_libraries(${_b} PRIVATE shmradix)
    # bench/* may include test_utils.h, now under tests/indexer/.
    target_include_directories(${_b} PRIVATE ${CMAKE_SOURCE_DIR}/tests/indexer)
    if(ENABLE_RDMA)
        add_dependencies(${_b} radix_server)
    endif()
endforeach()

# Remote-op correctness smoke on a 2-rank cluster; manual quick check (needs RDMA device, not in ctest).
if(ENABLE_RDMA)
    add_executable(remote_op_smoke tests/bench/remote_op_smoke.cpp)
    target_link_libraries(remote_op_smoke PRIVATE shmradix)
endif()

# ==================== Data plane: shmradix_data (storage + transfer) ====================
# Deliberately independent of libshmradix. The ONLY coupling between the control
# plane (indexer) and the data plane is the int32_t slot id, so this target must
# never link shmradix nor include its headers — the data-plane tests below link
# shmradix_data alone, which is what keeps that honest. shmradix_registry is the
# one exception, and not a violation: etcd is plane-neutral, hence src/registry.
set(_shmradix_data_sources
    # storage: the shm slot pool (slot id -> address, one registerable range).
    src/storage/shm_region.cpp
    src/storage/slot_store.cpp
    # transfer: move bytes between registered ranges; callers never see which
    # backend does it.
    src/transfer/range_merge.cpp
    src/transfer/backend_factory.cpp
    # slot-granular convenience layer over the two above.
    src/transfer/slot_transfer.cpp)
if(SHMRADIX_MOONCAKE_FOUND)
    list(APPEND _shmradix_data_sources src/transfer/mooncake/mooncake_backend.cpp)
endif()
if(SHMRADIX_NIXL_FOUND)
    list(APPEND _shmradix_data_sources src/transfer/nixl/nixl_backend.cpp)
endif()
add_library(shmradix_data STATIC ${_shmradix_data_sources})
if(SHMRADIX_MOONCAKE_FOUND)
    # Mooncake's headers require C++20 (<span>) while this project is C++17.
    # Only this one translation unit is raised, and no Mooncake type crosses its
    # boundary — the backend is reached solely through create_mooncake_backend.
    set_source_files_properties(src/transfer/mooncake/mooncake_backend.cpp
        PROPERTIES COMPILE_OPTIONS "-std=c++20")
    target_include_directories(shmradix_data PRIVATE ${MOONCAKE_INCLUDE_DIR})
    target_link_libraries(shmradix_data PUBLIC ${MOONCAKE_LINK_LIBS})
endif()
if(SHMRADIX_NIXL_FOUND)
    # NIXL's headers require C++20 (<span>) too — same treatment as Mooncake's TU.
    set_source_files_properties(src/transfer/nixl/nixl_backend.cpp
        PROPERTIES COMPILE_OPTIONS "-std=c++20")
    target_include_directories(shmradix_data PRIVATE ${NIXL_INCLUDE_DIR})
    target_link_libraries(shmradix_data PUBLIC ${NIXL_LINK_LIBS})
endif()
target_include_directories(shmradix_data PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include)                       # generated config.h
target_include_directories(shmradix_data PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/storage
    ${CMAKE_CURRENT_SOURCE_DIR}/src/transfer)
# shmradix_registry, not shmradix: its own geometry under /data/, no indexer.
target_link_libraries(shmradix_data PUBLIC rt pthread shmradix_registry)
# The Python extension links this archive into a shared object.
set_property(TARGET shmradix_data PROPERTY POSITION_INDEPENDENT_CODE ON)

set(_dataplane_tests
    test_slot_store
    # Geometry contract against a stub backend — no engine, so it runs
    # everywhere, Mooncake or not.
    test_slot_geometry
)
if(SHMRADIX_MOONCAKE_FOUND)
    list(APPEND _dataplane_tests test_mooncake_loopback test_slot_transfer_e2e)
endif()
if(SHMRADIX_NIXL_FOUND)
    list(APPEND _dataplane_tests test_nixl_loopback)
endif()
foreach(_t ${_dataplane_tests})
    add_executable(${_t} tests/dataplane/${_t}.cpp)
    target_link_libraries(${_t} PRIVATE shmradix_data)   # NOT shmradix — by design
    target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR}/tests/dataplane)
    shmradix_enable_asserts(${_t})
    add_test(NAME ${_t} COMMAND ${_t})
endforeach()

# Data-plane benchmark (NOT in ctest). Links shmradix_data only, like the tests.
add_executable(bench_slot_transfer tests/bench/bench_slot_transfer.cpp)
target_link_libraries(bench_slot_transfer PRIVATE shmradix_data)

# Manual real-hardware NIXL check (NOT in ctest — needs two hosts on one
# fabric with the peer already running). See the file header for usage.
if(SHMRADIX_NIXL_FOUND)
    add_executable(test_nixl_two_node tests/dataplane/test_nixl_two_node.cpp)
    target_link_libraries(test_nixl_two_node PRIVATE shmradix_data)
endif()

# ---------- shmradix._core + shmradix._data: pybind11 extension modules ----------
# Both from one pybind11 and one Python: pybind11 shares types across modules
# only when they were built alike, and _data reuses _core's ComponentType.
# _data stays its own module so the data plane never links the indexer.
option(BUILD_PYTHON_MODULES "Build the shmradix._core and shmradix._data pybind11 extensions" ON)
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
    if(pybind11_FOUND)
        set(_shmradix_py_dir ${CMAKE_SOURCE_DIR}/python/shmradix)

        pybind11_add_module(shmradix_core_py python/bindings.cpp)
        set_target_properties(shmradix_core_py PROPERTIES
            OUTPUT_NAME _core
            LIBRARY_OUTPUT_DIRECTORY ${_shmradix_py_dir}
            # libradix_etcd.so lives in the build tree.
            BUILD_RPATH ${CMAKE_BINARY_DIR}/lib)
        target_link_libraries(shmradix_core_py PRIVATE shmradix)

        # libradix_etcd.so lives in the build tree; libnixl.so, and next to it the
        # plugins it dlopens, in NIXL_LIBRARY_DIR.
        set(_shmradix_data_py_rpath ${CMAKE_BINARY_DIR}/lib)
        if(SHMRADIX_NIXL_FOUND)
            list(APPEND _shmradix_data_py_rpath ${NIXL_LIBRARY_DIR})
        endif()
        pybind11_add_module(shmradix_data_py python/data_bindings.cpp)
        set_target_properties(shmradix_data_py PROPERTIES
            OUTPUT_NAME _data
            LIBRARY_OUTPUT_DIRECTORY ${_shmradix_py_dir}
            BUILD_RPATH "${_shmradix_data_py_rpath}")
        target_link_libraries(shmradix_data_py PRIVATE shmradix_data)
        # In ctest, unlike test_shmradix.py: this one needs no server and no
        # numpy, so it can be part of the gate rather than a manual step.
        add_test(NAME test_data_python
                 COMMAND ${Python3_EXECUTABLE} tests/test_shmradix_data.py
                 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python)
        set_tests_properties(test_data_python PROPERTIES
            DEPENDS shmradix_data_py TIMEOUT 300)
        message(STATUS "shmradix._core + shmradix._data: building "
                       "(pybind11 ${pybind11_VERSION}, ${Python3_EXECUTABLE})")
    else()
        message(STATUS "shmradix._core + shmradix._data: skipped (no pybind11 for this Python)")
    endif()
endif()

# Legacy / historical PoCs — kept under tests/legacy/ but not built.
# To build manually: g++ ... tests/legacy/{file}.cpp

enable_testing()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
