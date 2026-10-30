// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RadixServer — public server surface. Owns the shm region + radix tree, runs the
// background evictor, and (distributed) brings up the RDMA/XRC/RHT cluster plane +
// remote-op control plane. The embedded eviction service borrows LocalTreeConnector
// into an rht_only RadixClient. This TU is the sole home of the heavy RDMA headers;
// radix_server.h stays verbs-free for the python bindings + tests.

#include "radix_server.h"

#include "local.h"
#include "radix_tree.h"
#include "net_util.hpp"

#ifndef RADIXSHMEM_NO_RDMA
#include "radix_client.h"
#include "cluster_manifest.h"
#include "cluster_protocol.h"
#include "cluster_registry.h"
#include "rdma_transport.hpp"
#include "rht_endpoint.h"
#include "remote_op_service.hpp"
#include "etcd_registry.hpp"   // EtcdRegistry (sole bootstrap path; self-guards on RADIXSHMEM_HAS_ETCD)

#include <arpa/inet.h>
#include <cerrno>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
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

    while (evictor_running_.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));

        bool need_evict = false;
        if (mp_total > 0 && tree_->mempool_free() < mp_threshold)
            need_evict = true;
        if (tree_->data_pool_utilization() >= DP_HIGH_WATERMARK)
            need_evict = true;
        if (tree_->node_pool_free() < np_threshold)
            need_evict = true;

        if (need_evict)
            tree_->try_evict_batch(LOCK_BATCH);
    }
}

#ifndef RADIXSHMEM_NO_RDMA

// =====================================================================
// cluster bring-up — TCP + shard-map helpers
// =====================================================================

namespace {

bool send_all(int fd, const void* data, size_t len) {
    const char* p = static_cast<const char*>(data);
    while (len > 0) {
        ssize_t n = ::send(fd, p, len, MSG_NOSIGNAL);
        if (n <= 0) return false;
        p += n; len -= (size_t)n;
    }
    return true;
}

bool recv_all(int fd, void* data, size_t len) {
    char* p = static_cast<char*>(data);
    while (len > 0) {
        ssize_t n = ::recv(fd, p, len, 0);
        if (n <= 0) return false;
        p += n; len -= (size_t)n;
    }
    return true;
}

// Bind+listen on `port` (0 = OS-assigned); returns fd, writes bound port to *out_port.
int tcp_listen(uint16_t port, uint16_t* out_port) {
    constexpr int kMaxAttempts = 12;
    constexpr auto kBackoff = std::chrono::milliseconds(500);
    int last_errno = 0;
    for (int attempt = 0; attempt < kMaxAttempts; attempt++) {
        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) { last_errno = errno; continue; }
        int opt = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port);
        if (::bind(fd, (sockaddr*)&addr, sizeof(addr)) == 0 &&
            ::listen(fd, 64) == 0) {
            socklen_t alen = sizeof(addr);
            ::getsockname(fd, (sockaddr*)&addr, &alen);
            if (out_port) *out_port = ntohs(addr.sin_port);
            return fd;
        }
        last_errno = errno;
        ::close(fd);
        std::this_thread::sleep_for(kBackoff);
    }
    fprintf(stderr, "[RadixServer] tcp_listen(:%u) gave up: %s\n",
            port, strerror(last_errno));
    errno = last_errno;
    return -1;
}

} // namespace

int RadixServer::register_chunk_size() const {
    return tree_->register_chunk_size();
}

void RadixServer::fill_local_entry(PeerEntry& e) const {
    memset(&e, 0, sizeof(e));
    e.rank = cfg_.rank;
    e.rht_shard_id = -1;   // overwritten by setup_local_rht_shard on holders

    const char* host = bind_ip_.empty() ? "127.0.0.1" : bind_ip_.c_str();
    std::strncpy(e.host, host, sizeof(e.host) - 1);
    e.xrc_listen_port = actual_listen_port_;

    e.shm_base_addr = (uint64_t)shm_mr_->addr();
    e.shm_size      = (uint64_t)shm_mr_->length();
    e.shm_rkey      = shm_mr_->rkey();

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

    memcpy(e.gid_raw, &ctx_->gid(), 16);
    e.gid_index  = ctx_->gid_index();

    ibv_port_attr pa{};
    if (ibv_query_port(ctx_->ctx(), ctx_->port(), &pa) == 0) {
        e.lid        = pa.lid;
        e.link_layer = (uint8_t)pa.link_layer;
    }
    e.srq_num = srq_ ? srq_->srq_num() : 0;

    std::strncpy(e.index_dev, cfg_.index_dev.c_str(), sizeof(e.index_dev) - 1);
}

void RadixServer::resolve_shard_map(int n_obs) {
    // Authoritative on the leader; other ranks overwrite from the broadcast map.
    int M = cfg_.num_rht_shards;
    if (M <= 0) M = n_obs;                           // M == N
    if (M > (int)CLUSTER_MAX_SHARDS) M = CLUSTER_MAX_SHARDS;
    if (M > n_obs) {                                 // can't have more holders than nodes
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "RadixServer: num_rht_shards (%d) exceeds node count (%d)",
                 M, n_obs);
        throw std::runtime_error(msg);
    }
    num_shards_ = M;

    holder_of_shard_.assign(M, 0);
    if (!cfg_.rht_shard_holders.empty()) {
        // Round-robin over the candidate holder list.
        int nh = (int)cfg_.rht_shard_holders.size();
        for (int s = 0; s < M; s++)
            holder_of_shard_[s] = cfg_.rht_shard_holders[s % nh];
    } else {
        // Default: holders are ranks [0, M).
        for (int s = 0; s < M; s++)
            holder_of_shard_[s] = s;
    }

    // Per-shard capacity REQUEST: global buckets ~= 2*R*N (load ~0.5), so each
    // shard asks for 2*R*N/M. Rounding (power of two, floor) is the RHT
    // module's business; it applies the same rule on holder and consumer.
    auto* hdr = static_cast<ShmHeader*>(tree_->shm_base());
    uint64_t R = hdr->node_pool_capacity;
    uint64_t per_shard = (2 * R * (uint64_t)n_obs) / (uint64_t)M;
    rht_capacity_ = per_shard > UINT32_MAX ? UINT32_MAX : (uint32_t)per_shard;

    // Which shard (if any) does this rank hold? (first match)
    my_local_shard_ = -1;
    for (int s = 0; s < M; s++) {
        if (holder_of_shard_[s] == cfg_.rank) { my_local_shard_ = s; break; }
    }

    // Authoritative QoS = rank 0's ctx values (seeded from its env).
    rdma_sl_ = ctx_->sl();
    rdma_tc_ = ctx_->traffic_class();

    // Authoritative RHT / peer-index transports (rank 0's cfg). DC access key
    // is unconditional: every plane that runs DC shares it.
    rht_transport_ = (cfg_.rht_transport == "dc") ? CLUSTER_TRANSPORT_DC
                                                   : CLUSTER_TRANSPORT_XRC;
    peer_index_transport_ = (cfg_.peer_index_transport == "dc")
                                ? CLUSTER_TRANSPORT_DC : CLUSTER_TRANSPORT_XRC;
    dc_access_key_ = 0xD0C0ACCE5500F00DULL;

    // Authoritative RHT slots-per-bucket (W). Validate ∈ {1,2,4,8}; default 4.
    int W = cfg_.rht_slots_per_bucket > 0 ? cfg_.rht_slots_per_bucket : 4;
    if (W != 1 && W != 2 && W != 4 && W != 8) {
        fprintf(stderr,
                "[RadixServer rank 0] WARN: rht_slots=%d not in {1,2,4,8}, "
                "using 4\n", W);
        W = 4;
    }
    rht_slots_ = (uint32_t)W;
}

void RadixServer::adopt_shard_map(const ShardMapMsg& sm) {
    if (sm.remote_op_transport != REMOTE_OP_TRANSPORT_GRPC)
        throw std::runtime_error("RadixServer: cluster remote-op transport is not grpc; redeploy the cluster with grpc");
    // Follower path: install the leader's authoritative shard map + QoS. Mirrors
    // what resolve_shard_map() computes on the leader, but from the wire message.
    num_shards_   = (int)sm.num_shards;
    rht_capacity_ = sm.rht_capacity;
    rdma_sl_      = sm.rdma_sl;
    rdma_tc_      = sm.rdma_tc;
    rht_transport_        = sm.rht_transport;
    peer_index_transport_ = sm.peer_index_transport;
    rht_slots_     = sm.rht_slots ? sm.rht_slots : 4;
    dc_access_key_ = sm.dc_access_key;
    ctx_->set_qos(rdma_sl_, rdma_tc_);   // adopt authoritative QoS before QPs connect
    holder_of_shard_.assign(num_shards_, 0);
    my_local_shard_ = -1;
    for (int s = 0; s < num_shards_; s++) {
        holder_of_shard_[s] = (int)sm.holder_rank[s];
        if (holder_of_shard_[s] == cfg_.rank && my_local_shard_ < 0)
            my_local_shard_ = s;
    }
    // Warn if local cfg disagreed with the authoritative map (non-fatal).
    if (cfg_.num_rht_shards > 0 && cfg_.num_rht_shards != num_shards_) {
        fprintf(stderr,
                "[RadixServer rank %d] WARN: local --num-rht-shards=%d ignored, "
                "cluster authoritative M=%d\n",
                cfg_.rank, cfg_.num_rht_shards, num_shards_);
    }
}

void RadixServer::setup_local_rht_shard(PeerEntry& e) {
    if (my_local_shard_ < 0) return;   // not a holder

    rht_holder_ = std::make_unique<RhtHolder>(
        ctx_->pd(), shm_name_, my_local_shard_, cfg_.rank,
        rht_capacity_, rht_slots_);
    const RhtShardAddr& m = rht_holder_->addr();
    m.publish_to(e);

    fprintf(stderr,
            "[RadixServer rank %d] holds RHT shard %d: %s cap=%u rkey=0x%x\n",
            cfg_.rank, my_local_shard_, m.shm_name,
            m.capacity, m.rkey);
}

void RadixServer::setup_dct(PeerEntry& e) {
    if (rht_transport_ != CLUSTER_TRANSPORT_DC &&
        peer_index_transport_ != CLUSTER_TRANSPORT_DC)
        return;   // both planes XRC: no DCT
    dct_ = std::make_unique<DcTarget>(*ctx_, dc_access_key_, /*srq_max_wr=*/1);
    e.dct_num = dct_->open_dct();
    fprintf(stderr, "[RadixServer rank %d] DC target up: dctn=%u\n",
            cfg_.rank, e.dct_num);
}

void RadixServer::setup_remote_op(PeerEntry& e) {
    // Remote ops off for this rank unless the control plane was requested.
    if (!(cfg_.enable_remote_insert || cfg_.enable_remote_query)) return;

    e.remote_op_listen_port = rc_build_endpoint().port;
    fprintf(stderr, "[RadixServer rank %d] remote-op endpoint up: grpc (port=%u)\n",
            cfg_.rank, e.remote_op_listen_port);
}

std::string RadixServer::shard_map_mismatch(const ShardMapMsg& sm) const {
    auto diff = [](const char* what, long long have, long long want) {
        return std::string(what) + " " + std::to_string(have) + " != local " +
               std::to_string(want);
    };
    if (cfg_.num_rht_shards > 0 && (int)sm.num_shards != cfg_.num_rht_shards)
        return diff("num_rht_shards", sm.num_shards, cfg_.num_rht_shards);
    if (!cfg_.rht_shard_holders.empty()) {
        int nh = (int)cfg_.rht_shard_holders.size();
        for (int s = 0; s < (int)sm.num_shards; s++)
            if ((int)sm.holder_rank[s] != cfg_.rht_shard_holders[s % nh])
                return diff(("holder of shard " + std::to_string(s)).c_str(),
                            sm.holder_rank[s], cfg_.rht_shard_holders[s % nh]);
    }
    int W = cfg_.rht_slots_per_bucket > 0 ? cfg_.rht_slots_per_bucket : 4;
    if (W != 1 && W != 2 && W != 4 && W != 8) W = 4;
    if ((int)sm.rht_slots != W) return diff("rht_slots", sm.rht_slots, W);
    auto tp = [](const std::string& s) {
        return s == "dc" ? CLUSTER_TRANSPORT_DC : CLUSTER_TRANSPORT_XRC;
    };
    if (sm.rht_transport != (uint32_t)tp(cfg_.rht_transport))
        return "rht_transport differs from local " + cfg_.rht_transport;
    if (sm.peer_index_transport != (uint32_t)tp(cfg_.peer_index_transport))
        return "peer_index_transport differs from local " + cfg_.peer_index_transport;
    int cs = register_chunk_size();
    if (sm.register_chunk_size && cs > 0 && (int)sm.register_chunk_size != cs)
        return diff("register_chunk_size", sm.register_chunk_size, cs);
    return "";
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

// ==================== Listener / hello-reply ====================

void RadixServer::xrc_listener_loop() {
    while (!stopping_.load()) {
        int conn = ::accept(xrc_listen_fd_, nullptr, nullptr);
        if (conn < 0) {
            if (stopping_.load()) break;
            if (errno == EINTR) continue;
            fprintf(stderr, "[RadixServer] accept failed: %s\n", strerror(errno));
            break;
        }
        int one = 1;
        setsockopt(conn, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        try {
            handle_client(conn);
        } catch (const std::exception& e) {
            fprintf(stderr, "[RadixServer] handle_client: %s\n", e.what());
        }
        ::close(conn);
    }
}

void RadixServer::handle_client(int conn_fd) {
    ClusterHelloMsg hello{};
    if (!recv_all(conn_fd, &hello, sizeof(hello)))
        throw std::runtime_error("hello recv short");
    if (hello.magic != CLUSTER_HELLO_MAGIC)
        throw std::runtime_error("bad hello magic");
    if (hello.version != CLUSTER_PROTOCOL_VERSION)
        throw std::runtime_error("hello version mismatch");

    auto recv_qp = std::make_unique<XrcRecvQp>(*ctx_, *xrcd_, *srq_, cq_);

    XrcPeerInfo peer{};
    peer.recv_qp_num = 0;
    peer.remote_srqn = 0;
    peer.gid         = hello.gid;
    peer.gid_index   = hello.gid_index;
    peer.lid         = hello.lid;
    peer.link_layer  = hello.link_layer;
    recv_qp->connect(hello.client_send_qp_num, peer);

    uint32_t my_recv = recv_qp->qp_num();

    ibv_port_attr pa{};
    if (ibv_query_port(ctx_->ctx(), ctx_->port(), &pa) != 0)
        throw std::runtime_error("ibv_query_port failed");

    ClusterReplyMsg reply{};
    reply.magic              = CLUSTER_REPLY_MAGIC;
    reply.version            = CLUSTER_PROTOCOL_VERSION;
    reply.server_rank        = cfg_.rank;
    reply.server_recv_qp_num = my_recv;
    reply.server_srq_num     = srq_->srq_num();
    reply.link_layer         = (uint8_t)pa.link_layer;
    reply.lid                = pa.lid;
    reply.gid_index          = ctx_->gid_index();
    reply.gid                = ctx_->gid();
    if (!send_all(conn_fd, &reply, sizeof(reply)))
        throw std::runtime_error("reply send short");

    auto slot = std::make_unique<ClientSlot>();
    slot->recv_qp           = std::move(recv_qp);
    slot->client_rank       = hello.client_rank;
    slot->client_send_qp_num = hello.client_send_qp_num;
    slot->recv_qp_num       = my_recv;
    {
        std::lock_guard<std::mutex> g(clients_mu_);
        clients_.emplace(next_client_id_++, std::move(slot));
    }
}

// ==================== cluster bootstrap ====================

bool RadixServer::cluster_bootstrap() {
    if (cfg_.registry.empty()) return true;   // standalone — caller bypasses

    const char* dev = cfg_.index_dev.empty() ? nullptr : cfg_.index_dev.c_str();
    ctx_  = std::make_unique<RdmaContext>(dev, 1, cfg_.gid_idx);
    xrcd_ = std::make_unique<RdmaXrcd>(*ctx_);
    // cq_ dormant under one-sided traffic; exists only because XRC SRQ/RECV QPs need one.
    cq_   = ibv_create_cq(ctx_->ctx(), 256, nullptr, nullptr, 0);
    if (!cq_) throw std::runtime_error("RadixServer: ibv_create_cq failed");
    srq_  = std::make_unique<XrcSrq>(*ctx_, *xrcd_, cq_);

    // Tree shm MR: READ + ATOMIC only (no REMOTE_WRITE). RHT writes use the separate RHT-MR.
    shm_mr_ = std::make_unique<RdmaMemoryRegion>(
        ctx_->pd(), tree_->shm_base(), tree_->shm_size(),
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
        IBV_ACCESS_REMOTE_ATOMIC);

    // XRC accept-loop listener
    xrc_listen_fd_ = tcp_listen(cfg_.rpc_port, &actual_listen_port_);
    if (xrc_listen_fd_ < 0) return false;
    listener_thread_ = std::thread(&RadixServer::xrc_listener_loop, this);
    fprintf(stderr, "[RadixServer rank %d] XRC accept loop on :%u\n",
            cfg_.rank, actual_listen_port_);

    // etcd rendezvous: discover peers + order bringup via etcd (sole path).
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
// Distributed when the caller asks for a multi-node cluster: a min-node gate or
// shard count above 1, or a legacy world_size hint.
bool want_cluster(const RadixServerConfig& c) {
    return c.expected_min_nodes > 1 || c.num_rht_shards > 1 || c.world_size > 1;
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
#ifndef RADIXSHMEM_NO_RDMA
    // Stop remote-op serving before tearing down svc_*.
    rc_stop();
#endif
    // Stop the evictor before anything that touches tree_ is torn down.
    stop_evictor();
#ifndef RADIXSHMEM_NO_RDMA
    // Release the eviction-service client (its evict callback touches tree_).
    svc_client_.reset();   // releases (not frees) the borrowed svc_tree_
    svc_tree_.reset();
#ifdef RADIXSHMEM_HAS_ETCD
    registry_.reset();     // revokes the lease: peers see this node leave at once
#endif

    // Ordered RDMA teardown (all bound to ctx_ / tree_ shm).
    stopping_.store(true);
    if (xrc_listen_fd_ >= 0) {
        ::shutdown(xrc_listen_fd_, SHUT_RDWR);
        ::close(xrc_listen_fd_);
        xrc_listen_fd_ = -1;
    }
    if (listener_thread_.joinable()) listener_thread_.join();
    {
        std::lock_guard<std::mutex> g(clients_mu_);
        clients_.clear();
    }
    dct_.reset();            // destroy DCT before PD/ctx teardown
    shm_mr_.reset();
    rht_holder_.reset();     // MR + shard shm released before PD teardown
    srq_.reset();
    if (cq_) { ibv_destroy_cq(cq_); cq_ = nullptr; }
    xrcd_.reset();
    ctx_.reset();
#endif
    // Finally drop the tree shm region.
    if (tree_) { tree_->destroy(); tree_.reset(); }
}

bool RadixServer::bootstrap() {
    const bool distributed = want_cluster(cfg_);
    if (distributed) {
        if (const char* transport = ::getenv("RADIXSHMEM_REMOTE_OP_TRANSPORT"))
            cfg_.remote_op_transport = transport;
    }
    if (cfg_.remote_op_transport != "grpc")
        throw std::runtime_error("RadixServer: remote_op_transport '" +
                                 cfg_.remote_op_transport + "' is unsupported; only grpc is supported");

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
    start_evictor();

#ifndef RADIXSHMEM_NO_RDMA
    if (distributed) {
        bind_ip_ = bind_ip;   // published as PeerEntry.host

        // Deployment-level env overrides.
        if (const char* t = ::getenv("RADIXSHMEM_RHT_TRANSPORT"))
            cfg_.rht_transport = t;
        if (const char* t = ::getenv("RADIXSHMEM_PEER_INDEX_TRANSPORT"))
            cfg_.peer_index_transport = t;
        if (const char* w = ::getenv("RADIXSHMEM_RHT_SLOTS"))
            cfg_.rht_slots_per_bucket = atoi(w);
        if (const char* r = ::getenv("RADIXSHMEM_REGISTRY"))
            cfg_.registry = r;
        if (const char* c = ::getenv("RADIXSHMEM_CLUSTER_ID"))
            cfg_.cluster_id = c;
        if (const char* n = ::getenv("RADIXSHMEM_NODE_NAME"))
            cfg_.node_name = n;

        if (!cluster_bootstrap()) {
            rc_stop();   // endpoint lives on ctx_ — destroy BEFORE tree_/ctx_
            stop_evictor();
            if (tree_) { tree_->destroy(); tree_.reset(); }
            return false;
        }
        // rank + observed world size were written by the rendezvous into cfg_.
        const bool want_evict = cfg_.shm.background_evict_ratio > 0.0f;
        const bool want_svc   = want_evict || cfg_.enable_remote_insert
                                           || cfg_.enable_remote_query;
        if (want_svc) {
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
                stop_evictor();
                if (tree_) { tree_->destroy(); tree_.reset(); }
                return false;
            }
            if (want_evict) {
                RadixClient*  ec = svc_client_.get();
                std::mutex*    mu = &svc_client_mu_;
                tree_->set_evict_callback(
                    [ec, mu](const std::vector<EvictedNode>& evicted) {
                        std::vector<RadixClient::EvictedKey> keys;
                        keys.reserve(evicted.size());
                        for (const auto& e : evicted)
                            keys.push_back({e.first_hash, e.tree_node_id});
                        std::lock_guard<std::mutex> g(*mu);
                        ec->remove_evicted_from_rht(keys.data(), (int)keys.size());
                    });
            }
            if (cfg_.enable_remote_insert) {
                // RemoteOpService::allocate holds svc_client_mu_; relocking here would self-deadlock.
                RadixClient* ec = svc_client_.get();
                svc_tree_->tree().set_evict_callback(
                    [ec](const std::vector<EvictedNode>& evicted) {
                        std::vector<RadixClient::EvictedKey> keys;
                        keys.reserve(evicted.size());
                        for (const auto& e : evicted)
                            keys.push_back({e.first_hash, e.tree_node_id});
                        ec->remove_evicted_from_rht(keys.data(), (int)keys.size());
                    });
            }
        }
    }
#endif
#ifndef RADIXSHMEM_NO_RDMA
    rc_spawn();
#endif
    return true;
}

} // namespace radixshmem
