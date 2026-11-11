// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RadixServer — public server surface. Owns the shm region + radix tree, runs the
// background evictor, and (distributed) joins the cluster through etcd and brings
// up its planes: the RDMA side (RdmaHost), the RHT shard, and the remote-op
// endpoint. The embedded eviction service borrows LocalTreeConnector into an
// rht_only RadixClient.

#include "radix_server.h"

#include "local.h"
#include "radix_tree.h"
#include "net_util.hpp"

#ifndef RADIXSHMEM_NO_RDMA
#include "radix_client.h"
#include "cluster_manifest.h"
#include "cluster_protocol.h"
#include "cluster_registry.h"
#include "rdma_host.hpp"
#include "rht_endpoint.h"
#include "rht_redis.h"
#include "remote_op_service.hpp"
#include "etcd_registry.hpp"   // EtcdRegistry (sole bootstrap path; self-guards on RADIXSHMEM_HAS_ETCD)
#endif

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace radixshmem {

// =====================================================================
// tree accessors + stats
// =====================================================================

RadixShmemTree&       RadixServer::tree()       { return *tree_; }
const RadixShmemTree& RadixServer::tree() const { return *tree_; }

const std::string& RadixServer::shm_name() const { return shm_name_; }

int64_t RadixServer::lease_id() const {
#if !defined(RADIXSHMEM_NO_RDMA) && defined(RADIXSHMEM_HAS_ETCD)
    return registry_ ? registry_->lease_id() : 0;
#else
    return 0;
#endif
}

uint32_t RadixServer::total_radix_nodes() const  { return tree_->total_radix_nodes(); }
uint64_t RadixServer::total_blocks() const { return tree_->total_blocks(); }
float    RadixServer::data_pool_utilization() const { return tree_->data_pool_utilization(); }
uint32_t RadixServer::mempool_total() const { return tree_->mempool_total(); }
uint32_t RadixServer::mempool_free()  const { return tree_->mempool_free();  }
uint32_t RadixServer::mempool_used()  const { return tree_->mempool_used();  }
uint32_t RadixServer::swa_mempool_total()   const { return tree_->swa_mempool_total();   }
uint32_t RadixServer::mamba_mempool_total() const { return tree_->mamba_mempool_total(); }
uint32_t RadixServer::swa_mempool_free()    const { return tree_->swa_mempool_free();    }
uint32_t RadixServer::mamba_mempool_free()  const { return tree_->mamba_mempool_free();  }
ComponentConfig RadixServer::component_config() const { return tree_->component_config(); }

uint64_t RadixServer::insert_success_count() const { return tree_->insert_success_count(); }
PeerFaultCounters* RadixServer::peer_faults() const { return tree_->peer_faults(); }
uint64_t RadixServer::insert_error_node_pool_count() const { return tree_->insert_error_node_pool_count(); }
uint64_t RadixServer::insert_error_data_pool_count() const { return tree_->insert_error_data_pool_count(); }
uint64_t RadixServer::insert_error_ht_pool_count()   const { return tree_->insert_error_ht_pool_count();   }
uint64_t RadixServer::insert_error_mempool_count()   const { return tree_->insert_error_mempool_count();   }

int RadixServer::check_health(HealthReport* report) const {
    HealthReport r;
    if (tree_) r = radixshmem::check_health(*tree_);
    else       r.flags = HEALTH_NOT_ATTACHED;
    if (report) *report = r;
    return (int)r.flags;
}

// =====================================================================
// background evictor
// =====================================================================

void RadixServer::start_evictor() {
    if (tree_->background_evict_ratio() <= 0.0f)
        return;
    evictor_running_.store(true, std::memory_order_release);
    evictor_thread_ = std::thread(&RadixServer::evictor_loop, this);
}

void RadixServer::stop_evictor() {
    evictor_running_.store(false, std::memory_order_release);
    if (evictor_thread_.joinable())
        evictor_thread_.join();
}

void RadixServer::evictor_loop() {
    const float ratio = tree_->background_evict_ratio();
    const uint32_t mp_total = tree_->mempool_total();
    const uint32_t mp_threshold = mp_total > 0 ? (uint32_t)(mp_total * ratio) : 0;
    const uint32_t np_capacity = tree_->node_pool_capacity();
    const uint32_t np_threshold = (uint32_t)(np_capacity * ratio);
    static constexpr float DP_HIGH_WATERMARK = 0.90f;
    static constexpr int LOCK_BATCH = 50;

    uint64_t wb_limit[NUM_COMPONENT_TYPES] = {};   // 0 = pool absent
    if (writeback_) {
        const uint32_t totals[NUM_COMPONENT_TYPES] = {
            mp_total, tree_->swa_mempool_total(), tree_->mamba_mempool_total()};
        for (int k = 0; k < NUM_COMPONENT_TYPES; k++)
            if (totals[k] > 0)
                wb_limit[k] = std::max<uint64_t>(
                    1, (uint64_t)(totals[k] * cfg_.writeback_max_held_ratio));
    }
    std::vector<HeldEviction> held;

    while (evictor_running_.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));

        bool need_evict = false;
        if (mp_total > 0 && tree_->mempool_free() < mp_threshold)
            need_evict = true;
        if (tree_->data_pool_utilization() >= DP_HIGH_WATERMARK)
            need_evict = true;
        if (tree_->node_pool_free() < np_threshold)
            need_evict = true;
        if (!need_evict)
            continue;
        if (!writeback_ || !writeback_->active()) {
            tree_->try_evict_batch(LOCK_BATCH);
            continue;
        }

        // Held slots are not free: the in-flight watermark bounds the eviction.
        uint64_t inflight[NUM_COMPONENT_TYPES];
        writeback_->inflight_slots(inflight);
        bool paused = false;
        uint64_t budget[NUM_COMPONENT_TYPES];
        for (int k = 0; k < NUM_COMPONENT_TYPES; k++) {
            paused |= wb_limit[k] > 0 && inflight[k] >= wb_limit[k];
            budget[k] = wb_limit[k] > 0 ? wb_limit[k] - inflight[k] : UINT64_MAX;
        }
        if (paused)
            continue;
        held.clear();
        tree_->try_evict_batch(LOCK_BATCH, &held, budget);
        if (!held.empty())
            writeback_->push(std::move(held));
    }
}

#ifndef RADIXSHMEM_NO_RDMA

// =====================================================================
// cluster bring-up — shard-map helpers
// =====================================================================

namespace {

int env_int(const char* name, int fallback) {
    const char* v = ::getenv(name);
    return v ? std::atoi(v) : fallback;
}

} // namespace

int RadixServer::register_chunk_size() const {
    return tree_->register_chunk_size();
}

// The RDMA fields come later, from setup_rdma().
void RadixServer::fill_local_entry(PeerEntry& e) const {
    memset(&e, 0, sizeof(e));
    e.rank = cfg_.rank;
    e.rht_shard_id = -1;   // overwritten by setup_local_rht_shard on holders

    const char* host = bind_ip_.empty() ? "127.0.0.1" : bind_ip_.c_str();
    std::strncpy(e.host, host, sizeof(e.host) - 1);
    e.shm_size = (uint64_t)tree_->shm_size();

    auto* hdr = static_cast<ShmHeader*>(tree_->shm_base());
    e.node_pool_offset    = hdr->node_pool_offset;
    e.node_pool_capacity  = hdr->node_pool_capacity;
    e.data_pool_offset    = hdr->data_pool_offset;
    e.global_ht_offset    = hdr->global_ht_offset;
    e.ht_bucket_count     = hdr->global_ht_bucket_count;
    e.ht_entry_capacity   = hdr->global_ht_entry_capacity;
    // RHT shard fields filled later by setup_local_rht_shard() on holder ranks only.
    e.rdma_rwlock_offset  = (uint64_t)(
        reinterpret_cast<const char*>(&hdr->rdma_write_flag) -
        reinterpret_cast<const char*>(hdr));
    e.root_node_id        = hdr->root_node_id;
    e.component_config       = hdr->component_config;
    e.component_state_offset = hdr->component_state_offset;

    std::strncpy(e.index_dev, cfg_.index_dev.c_str(), sizeof(e.index_dev) - 1);
}

void RadixServer::read_rht_env() {
    auto env = [](const char* name) {
        const char* v = ::getenv(name);
        return std::string(v ? v : "");
    };
    auto bad = [](const char* name, const std::string& v) {
        return std::runtime_error(std::string("RadixServer: invalid ") + name + "=" + v);
    };
    RhtEnv e;
    std::string v = env("RADIXSHMEM_RHT_BACKEND");
    if (v == "rdma") e.backend = CLUSTER_RHT_RDMA;
    else if (!v.empty() && v != "redis") throw bad("RADIXSHMEM_RHT_BACKEND", v);
    if (e.backend == CLUSTER_RHT_REDIS) {
        e.redis = env("RADIXSHMEM_REDIS_ADDRS");
        v = env("RADIXSHMEM_REDIS_TIMEOUT_MS");
        if (!v.empty() && (e.redis_timeout_ms = std::atoi(v.c_str())) <= 0)
            throw bad("RADIXSHMEM_REDIS_TIMEOUT_MS", v);
    } else {
        v = env("RADIXSHMEM_RHT_SHARDS");
        if (!v.empty() && (e.rdma.shards = std::atoi(v.c_str())) < 0) throw bad("RADIXSHMEM_RHT_SHARDS", v);
        v = env("RADIXSHMEM_RHT_SHARD_HOLDERS");
        for (const char* p = v.c_str(); *p;) {
            char* end = nullptr;
            long r = std::strtol(p, &end, 10);
            if (end == p || r < 0 || r > INT_MAX) throw bad("RADIXSHMEM_RHT_SHARD_HOLDERS", v);
            e.rdma.holders.push_back((int)r);
            for (p = end; *p == ',' || *p == ' '; p++) {}
        }
        v = env("RADIXSHMEM_RHT_SLOTS");
        if (!v.empty()) e.rdma.slots = std::atoi(v.c_str());
        if (e.rdma.slots != 1 && e.rdma.slots != 2 && e.rdma.slots != 4 && e.rdma.slots != 8)
            throw bad("RADIXSHMEM_RHT_SLOTS", v);
        v = env("RADIXSHMEM_RHT_TRANSPORT");
        if (v == "dc") e.rdma.transport = CLUSTER_TRANSPORT_DC;
        else if (!v.empty() && v != "xrc") throw bad("RADIXSHMEM_RHT_TRANSPORT", v);
    }
    rht_env_ = e;
}

void RadixServer::resolve_shard_map(int ranks) {
    // Authoritative on the leader; other ranks overwrite from the broadcast map.
    rdma_sl_ = env_int("RADIXSHMEM_RDMA_SL", 5);
    rdma_tc_ = env_int("RADIXSHMEM_RDMA_TC", 160);
    peer_index_transport_ = (cfg_.peer_index_transport == "dc")
                                ? CLUSTER_TRANSPORT_DC : CLUSTER_TRANSPORT_XRC;
    dc_access_key_ = 0xD0C0ACCE5500F00DULL;   // every plane that runs DC shares it
    rht_backend_ = rht_env_.backend;
    num_shards_ = 0;
    holder_of_shard_.clear();
    my_local_shard_ = -1;
    rht_capacity_ = 0;
    rht_slots_ = 0;
    rht_transport_ = CLUSTER_TRANSPORT_XRC;

    if (rht_backend_ == CLUSTER_RHT_REDIS) {
        if (cfg_.cluster_id.find_first_of("{}") != std::string::npos)
            throw std::runtime_error("RadixServer: the redis RHT needs a cluster id without { "
                                     "or }, not '" + cfg_.cluster_id + "'");
        rht_redis_ = rht_env_.redis;
        rht_redis_timeout_ms_ = (uint32_t)rht_env_.redis_timeout_ms;
        if (rht_redis_.empty() || rht_redis_.size() >= CLUSTER_RHT_SEEDS_LEN)
            throw std::runtime_error(
                "RadixServer: the redis RHT needs RADIXSHMEM_REDIS_ADDRS=host:port,... "
                "(under 256 chars); RADIXSHMEM_RHT_BACKEND=rdma selects the RDMA RHT");
        std::string why = redis_rht_check({rht_redis_, "", rht_env_.redis_timeout_ms});
        if (!why.empty())
            throw std::runtime_error("RadixServer: Redis RHT at " + rht_redis_ + ": " + why);
        return;
    }

    int M = rht_env_.rdma.shards;
    if (M <= 0) M = ranks;                           // M == N
    if (M > (int)CLUSTER_MAX_SHARDS) M = CLUSTER_MAX_SHARDS;
    if (M > ranks)                                   // can't have more holders than nodes
        throw std::runtime_error("RadixServer: RADIXSHMEM_RHT_SHARDS (" + std::to_string(M) +
                                 ") exceeds node count (" + std::to_string(ranks) + ")");
    num_shards_ = M;

    const std::vector<int>& holders = rht_env_.rdma.holders;
    for (int h : holders)
        if (h >= ranks)
            throw std::runtime_error("RadixServer: RADIXSHMEM_RHT_SHARD_HOLDERS names rank " +
                                     std::to_string(h) + ", but the ranks are 0.." +
                                     std::to_string(ranks - 1));
    holder_of_shard_.assign(M, 0);
    for (int s = 0; s < M; s++)   // round-robin over the candidates, default ranks [0, M)
        holder_of_shard_[s] = holders.empty() ? s : holders[s % holders.size()];

    // Per-shard capacity REQUEST: global buckets ~= 2*R*N (load ~0.5), so each
    // shard asks for 2*R*N/M. Rounding (power of two, floor) is the RHT
    // module's business; it applies the same rule on holder and consumer.
    auto* hdr = static_cast<ShmHeader*>(tree_->shm_base());
    uint64_t R = hdr->node_pool_capacity;
    uint64_t per_shard = (2 * R * (uint64_t)ranks) / (uint64_t)M;
    rht_capacity_ = per_shard > UINT32_MAX ? UINT32_MAX : (uint32_t)per_shard;

    for (int s = 0; s < M; s++) {
        if (holder_of_shard_[s] == cfg_.rank) { my_local_shard_ = s; break; }
    }
    rht_transport_ = rht_env_.rdma.transport;
    rht_slots_ = (uint32_t)rht_env_.rdma.slots;
}

void RadixServer::adopt_shard_map(const ShardMapMsg& sm) {
    // Follower path: install the leader's authoritative shard map + QoS. Mirrors
    // what resolve_shard_map() computes on the leader, but from the wire message.
#ifndef RADIXSHMEM_HAS_REDIS
    if (sm.rht_backend == CLUSTER_RHT_REDIS)
        throw std::runtime_error("RadixServer: the cluster keeps its RHT in Redis, and this "
                                 "radixshmem was built without hiredis");
#endif
    rht_backend_  = sm.rht_backend;
    rht_redis_.assign(sm.rht_redis_seeds, strnlen(sm.rht_redis_seeds, sizeof(sm.rht_redis_seeds)));
    rht_redis_timeout_ms_ = sm.rht_redis_timeout_ms;
    num_shards_   = (int)sm.num_shards;
    rht_capacity_ = sm.rht_capacity;
    rdma_sl_      = sm.rdma_sl;
    rdma_tc_      = sm.rdma_tc;
    rht_transport_        = sm.rht_transport;
    peer_index_transport_ = sm.peer_index_transport;
    rht_slots_     = sm.rht_slots;
    dc_access_key_ = sm.dc_access_key;
    holder_of_shard_.assign(num_shards_, 0);
    my_local_shard_ = -1;
    for (int s = 0; s < num_shards_; s++) {
        holder_of_shard_[s] = (int)sm.holder_rank[s];
        if (holder_of_shard_[s] == cfg_.rank && my_local_shard_ < 0)
            my_local_shard_ = s;
    }
    if (rht_env_.backend != rht_backend_)
        fprintf(stderr, "[RadixServer rank %d] WARN: local RADIXSHMEM_RHT_BACKEND ignored, "
                "the cluster uses the %s RHT\n", cfg_.rank,
                rht_backend_ == CLUSTER_RHT_REDIS ? "redis" : "rdma");
}

void RadixServer::setup_planes(PeerEntry& e) {
    setup_rdma(e);
    setup_local_rht_shard(e);
    setup_remote_op(e);
}

void RadixServer::setup_rdma(PeerEntry& e) {
    RdmaHostConfig hc;
    hc.dev     = cfg_.index_dev;
    hc.gid_idx = cfg_.gid_idx;
    hc.sl      = rdma_sl_;
    hc.tc      = rdma_tc_;
    hc.port    = cfg_.rpc_port;
    hc.dc      = rht_transport_ == CLUSTER_TRANSPORT_DC ||
                 peer_index_transport_ == CLUSTER_TRANSPORT_DC;
    hc.dc_access_key = dc_access_key_;
    hc.rank    = cfg_.rank;
    rdma_ = std::make_unique<RdmaHost>(hc);
    rdma_->expose_tree(tree_->shm_base(), tree_->shm_size());
    rdma_->publish_to(e);
}

void RadixServer::setup_local_rht_shard(PeerEntry& e) {
    if (my_local_shard_ < 0) return;   // not a holder

    rht_holder_ = std::make_unique<RhtHolder>(
        rdma_->pd(), shm_name_, my_local_shard_, cfg_.rank,
        rht_capacity_, rht_slots_);
    const RhtShardAddr& m = rht_holder_->addr();
    m.publish_to(e);

    fprintf(stderr,
            "[RadixServer rank %d] holds RHT shard %d: %s cap=%u rkey=0x%x\n",
            cfg_.rank, my_local_shard_, m.shm_name,
            m.capacity, m.rkey);
}

void RadixServer::setup_remote_op(PeerEntry& e) {
    std::random_device rd;
    instance_id_ = (((uint64_t)rd() << 32) | rd()) | 1;
    e.instance_id = instance_id_;
    e.remote_op_listen_port = rc_build_endpoint().port;
    fprintf(stderr, "[RadixServer rank %d] remote-op endpoint up: grpc (port=%u)\n",
            cfg_.rank, e.remote_op_listen_port);
}

std::string RadixServer::shard_map_mismatch(const ShardMapMsg& sm) const {
    auto diff = [](const char* what, long long have, long long want) {
        return std::string(what) + " " + std::to_string(have) + " != local " +
               std::to_string(want);
    };
    if (sm.rht_backend != rht_env_.backend)
        return diff("RHT backend", sm.rht_backend, rht_env_.backend);
    if (sm.rht_backend == CLUSTER_RHT_RDMA) {
        const auto& e = rht_env_.rdma;
        if (e.shards > 0 && (int)sm.num_shards != e.shards)
            return diff("RADIXSHMEM_RHT_SHARDS", sm.num_shards, e.shards);
        for (int s = 0; !e.holders.empty() && s < (int)sm.num_shards; s++)
            if ((int)sm.holder_rank[s] != e.holders[s % e.holders.size()])
                return diff(("holder of shard " + std::to_string(s)).c_str(),
                            sm.holder_rank[s], e.holders[s % e.holders.size()]);
        if ((int)sm.rht_slots != e.slots) return diff("RADIXSHMEM_RHT_SLOTS", sm.rht_slots, e.slots);
        if (sm.rht_transport != e.transport)
            return diff("RADIXSHMEM_RHT_TRANSPORT", sm.rht_transport, e.transport);
    }
    uint32_t peer_tp = cfg_.peer_index_transport == "dc" ? CLUSTER_TRANSPORT_DC
                                                         : CLUSTER_TRANSPORT_XRC;
    if (sm.peer_index_transport != peer_tp)
        return "peer_index_transport differs from local " + cfg_.peer_index_transport;
    int cs = register_chunk_size();
    if (sm.register_chunk_size && cs > 0 && (int)sm.register_chunk_size != cs)
        return diff("register_chunk_size", sm.register_chunk_size, cs);
    return "";
}

void RadixServer::drop_stale_routes(bool whole_cluster) {
    if (rht_backend_ != CLUSTER_RHT_REDIS) return;
    const RedisRhtConfig rc{rht_redis_, redis_rht_prefix(cfg_.cluster_id),
                            (int)rht_redis_timeout_ms_};
    auto t0 = std::chrono::steady_clock::now();
    const bool clean = whole_cluster ? redis_rht_clear(rc) : redis_rht_drop_rank(rc, cfg_.rank);
    fprintf(stderr, "[RadixServer] stale routes of %s %s %.1f s\n",
            whole_cluster ? "the cluster" : ("rank " + std::to_string(cfg_.rank)).c_str(),
            clean ? "dropped in" : "not all dropped after",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}

void RadixServer::stamp_shm_ready(int assigned_rank, int n_obs) {
    auto* hdr = static_cast<ShmHeader*>(tree_->shm_base());
    hdr->my_rank        = (uint32_t)assigned_rank;
    hdr->is_distributed = (n_obs > 1) ? 1u : 0u;
    std::strncpy(hdr->registry, cfg_.registry.c_str(), sizeof(hdr->registry) - 1);
    std::strncpy(hdr->cluster_id, cfg_.cluster_id.c_str(), sizeof(hdr->cluster_id) - 1);
    std::strncpy(hdr->node_name, cfg_.node_name.c_str(), sizeof(hdr->node_name) - 1);
    hdr->cluster_ready_magic.store(CLUSTER_MANIFEST_MAGIC,
                                    std::memory_order_release);
}

// ==================== cluster bootstrap ====================

bool RadixServer::cluster_bootstrap() {
    if (cfg_.registry.empty()) return true;   // standalone — caller bypasses

    // etcd rendezvous: discover peers + order bringup via etcd (sole path); the
    // planes come up in its round 2, once the shard map is known.
#ifdef RADIXSHMEM_HAS_ETCD
    // registry uri = "etcd://host:port" → strip scheme to "host:port".
    std::string ep = cfg_.registry;
    auto pos = ep.find("://");
    if (pos != std::string::npos) ep = ep.substr(pos + 3);
    if (cfg_.node_name.empty()) {
        fprintf(stderr, "[RadixServer] node_name required for cluster bootstrap\n");
        return false;
    }
    registry_ = std::make_unique<EtcdRegistry>(*this, ep, cfg_.cluster_id,
                                               cfg_.node_name, shm_name_);
    std::vector<PeerEntry> entries;
    if (registry_->run_rendezvous(entries)) return true;
    registry_.reset();
    return false;
#else
    // No etcd built in → no cluster bootstrap. CMake FATAL_ERRORs before this in
    // RDMA builds; this guard only covers a hand-forced ENABLE_ETCD=OFF RDMA build.
    fprintf(stderr, "[RadixServer] built without etcd — no cluster bootstrap "
                    "available\n");
    return false;
#endif
}

#endif  // !RADIXSHMEM_NO_RDMA

// =====================================================================
// RadixServer lifecycle
// =====================================================================

namespace {
// Distributed when the caller asks for a multi-node cluster: a min-node gate
// above 1, or a legacy world_size hint.
bool want_cluster(const RadixServerConfig& c) {
    return c.expected_min_nodes > 1 || c.world_size > 1;
}
} // namespace

RadixServer::RadixServer(const RadixServerConfig& cfg) : cfg_(cfg) {
#ifdef RADIXSHMEM_NO_RDMA
    if (want_cluster(cfg_))
        throw std::runtime_error(
            "RadixServer: built without RDMA — distributed deployments "
            "require libibverbs.");
#endif
}

RadixServer::RadixServer(const std::string& name, const ShmConfig& shm) {
    cfg_.name = name;
    cfg_.shm  = shm;
    if (!bootstrap())
        throw std::runtime_error("RadixServer: bootstrap failed for shm " + name);
}

RadixServer::~RadixServer() {
    dedup_writes_->set_active(false);
#ifndef RADIXSHMEM_NO_RDMA
    // Stop remote-op serving before tearing down svc_*.
    rc_stop();
#endif
    // Stop the evictor before anything that touches tree_ is torn down.
    stop_evictor();
    writeback_.reset();
#ifndef RADIXSHMEM_NO_RDMA
    // Release the eviction-service client (its evict callback touches tree_).
    svc_client_.reset();   // releases (not frees) the borrowed svc_tree_
    svc_tree_.reset();
#ifdef RADIXSHMEM_HAS_ETCD
    registry_.reset();     // revokes the lease: peers see this node leave at once
#endif
    rht_holder_.reset();   // its MR lives on rdma_'s PD
    rdma_.reset();
#endif
    // Finally drop the tree shm region.
    if (tree_) { tree_->destroy(); tree_.reset(); }
}

bool RadixServer::bootstrap() {
    const bool distributed = want_cluster(cfg_);
    const float held_ratio = cfg_.writeback_max_held_ratio;
    if (cfg_.writeback_enabled && !(held_ratio > 0.0f && held_ratio <= 1.0f))
        throw std::runtime_error("RadixServer: writeback_max_held_ratio must be in (0, 1]");
#ifndef RADIXSHMEM_NO_RDMA
    if (distributed) read_rht_env();
#endif

    ShmConfig shm = cfg_.shm;
    shm.register_chunk_tokens = cfg_.register_chunk_tokens;
    shm.max_nodes = shm.effective_max_nodes();

    // Bootstrap IP peers connect to: interface (wins) or explicit address.
    // Required in distributed mode; the resolved IP also seeds node_name.
    std::string bind_ip = resolve_bind_ip(cfg_.rpc_address, cfg_.rpc_interface);
    if (distributed && bind_ip.empty())
        throw std::runtime_error(
            "RadixServer: distributed mode requires --rpc-address <ip> or "
            "--rpc-interface <nic> (use 0.0.0.0 for a single-node test)");

    // Node identity: explicit node_name, else derived from the bind IP (node<ip>).
    std::string node_name = cfg_.node_name;
    if (node_name.empty())
        node_name = "node" + bind_ip;
    cfg_.node_name = node_name;

    // Shm name is identity-based; multiple nodes on one host disambiguate by identity.
    std::string shm_name = cfg_.name;
    if (distributed && !cfg_.node_name.empty())
        shm_name = cfg_.name + "_" + cfg_.node_name;

    // Bring up the shm region + evictor.
    tree_ = std::make_unique<RadixShmemTree>(RadixShmemTree::create(shm_name, shm));
    shm_name_        = shm_name;
    if (cfg_.writeback_enabled)
        writeback_ = std::make_unique<WritebackQueue>(*tree_);
    start_evictor();

#ifndef RADIXSHMEM_NO_RDMA
    if (distributed) {
        bind_ip_ = bind_ip;   // published as PeerEntry.host

        // Deployment-level env overrides.
        if (const char* t = ::getenv("RADIXSHMEM_PEER_INDEX_TRANSPORT"))
            cfg_.peer_index_transport = t;
        if (const char* r = ::getenv("RADIXSHMEM_REGISTRY"))
            cfg_.registry = r;
        if (const char* c = ::getenv("RADIXSHMEM_CLUSTER_ID"))
            cfg_.cluster_id = c;
        if (const char* n = ::getenv("RADIXSHMEM_NODE_NAME"))
            cfg_.node_name = n;

        if (!cluster_bootstrap()) {
            rc_stop();
            rht_holder_.reset();
            rdma_.reset();       // deregisters the tree MR before the shm goes
            stop_evictor();
            writeback_.reset();
            if (tree_) { tree_->destroy(); tree_.reset(); }
            return false;
        }
        // rank + observed world size were written by the rendezvous into cfg_.
        const bool want_evict = cfg_.shm.background_evict_ratio > 0.0f;
        svc_tree_ = std::make_unique<LocalTreeConnector>(shm_name);
        RadixClientConfig cc;
        cc.index_dev = cfg_.index_dev;
        cc.gid_idx  = cfg_.gid_idx;
        cc.bootstrap_timeout_sec = cfg_.bootstrap_timeout_sec;
        cc.rht_only = true;
        cc.registry           = cfg_.registry;
        cc.cluster_id         = cfg_.cluster_id;
        cc.node_name          = cfg_.node_name;
        cc.expected_min_nodes = cfg_.expected_min_nodes;
        cc.settle_ms          = cfg_.settle_ms;
        try {
            // Borrows svc_tree_ (no second attach); builds the RHT plane only.
            svc_client_ = std::make_unique<RadixClient>(cc, svc_tree_.get());
        } catch (const std::exception& e) {
            fprintf(stderr, "[RadixServer] svc RadixClient bootstrap failed: %s\n",
                    e.what());
            svc_client_.reset();
            svc_tree_.reset();
            rc_stop();
            rht_holder_.reset();
            rdma_.reset();
            stop_evictor();
            writeback_.reset();
            if (tree_) { tree_->destroy(); tree_.reset(); }
            return false;
        }
        if (want_evict) {
            RadixClient*  ec = svc_client_.get();
            std::mutex*    mu = &svc_client_mu_;
            tree_->set_evict_callback(
                [ec, mu](const std::vector<NodeHead>& evicted) {
                    std::vector<RadixClient::EvictedKey> keys;
                    keys.reserve(evicted.size());
                    for (const auto& e : evicted)
                        keys.push_back({e.first_hash, e.tree_node_id});
                    std::lock_guard<std::mutex> g(*mu);
                    ec->remove_evicted_from_rht(keys.data(), (int)keys.size());
                });
        }
        // RemoteOpService::allocate holds svc_client_mu_; relocking here would self-deadlock.
        RadixClient* ec = svc_client_.get();
        svc_tree_->tree().set_evict_callback(
            [ec](const std::vector<NodeHead>& evicted) {
                std::vector<RadixClient::EvictedKey> keys;
                keys.reserve(evicted.size());
                for (const auto& e : evicted)
                    keys.push_back({e.first_hash, e.tree_node_id});
                ec->remove_evicted_from_rht(keys.data(), (int)keys.size());
            });
    }
#endif
#ifndef RADIXSHMEM_NO_RDMA
    rc_spawn();
#endif
    return true;
}

} // namespace radixshmem
