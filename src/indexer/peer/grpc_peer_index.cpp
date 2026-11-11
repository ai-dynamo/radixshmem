// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "grpc_peer_index.hpp"

#include "peer_registry.hpp"
#include "remote_op_factory.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <shared_mutex>
#include <string>
#include <thread>
#include <utility>

namespace radixshmem {

namespace {

using Clock = std::chrono::steady_clock;

double since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

int peer_index_timeout_ms() {
    if (const char* e = ::getenv("RADIXSHMEM_PEER_INDEX_TIMEOUT_MS")) {
        int v = std::atoi(e);
        if (v > 0) return v;
    }
    return 500;
}

// The peer did not answer in time or at all: whether it ran the op is unknown.
bool unreachable(RemoteStatus st) {
    return st == RemoteStatus::UNAVAILABLE || st == RemoteStatus::TIMEOUT;
}

}  // namespace

GrpcPeerIndex::GrpcPeerIndex(PeerRegistry& peers, const PeerIndexConfig& cfg)
    : peers_(peers), cfg_(cfg), timeout_ms_(peer_index_timeout_ms()),
      breaker_("[RadixClient rank " + std::to_string(peers.my_rank()) + "] peer index",
               cfg.faults) {}

std::shared_ptr<GrpcPeerIndex::Session> GrpcPeerIndex::session(int rank) {
    std::lock_guard<std::mutex> g(mu_);
    auto it = sessions_.find(rank);
    if (it != sessions_.end()) return it->second;
    PeerEntry e;
    {
        std::shared_lock<std::shared_mutex> lk(peers_.mutex());
        if (rank < 0 || rank >= peers_.size_locked() || rank == peers_.my_rank())
            return nullptr;
        e = peers_.peers_locked()[rank];
    }
    if (e.rank != rank) return nullptr;   // node down
    auto s = std::make_shared<Session>();
    try {
        std::string endpoint;
        s->ops = make_remote_op_client(e, endpoint, timeout_ms_);
    } catch (const std::exception& ex) {
        fprintf(stderr, "[GrpcPeerIndex rank %d] peer rank %d: %s\n", peers_.my_rank(), rank,
                ex.what());
        return nullptr;
    }
    s->instance = e.instance_id;
    s->usable = cfg_.has_expected && peer_component_abi_valid(e, cfg_.expected);
    if (!s->usable)
        fprintf(stderr,
                "[GrpcPeerIndex rank %d] queries to peer rank %d disabled (%s): no "
                "remote hits from this peer\n", peers_.my_rank(), rank,
                cfg_.has_expected ? "peer config/ABI mismatch"
                                  : "no expected config (observer without consensus)");
    breaker_.name(rank, e.host);
    sessions_.emplace(rank, s);
    return s;
}

void GrpcPeerIndex::drop(int rank) {
    std::lock_guard<std::mutex> g(mu_);
    sessions_.erase(rank);
}

bool GrpcPeerIndex::preconnect(int rank) {
    auto s = session(rank);
    if (!s) return false;
    s->ops->connect();
    return true;
}

void GrpcPeerIndex::peer_changed(const PeerEntry& e) {
    drop(e.rank);
    breaker_.retry_now(e.rank);
}

void GrpcPeerIndex::peer_left(int rank) {
    drop(rank);
    breaker_.left(rank, "left the cluster (etcd entry removed)");
}

QueryReply GrpcPeerIndex::query(int rank, const uint64_t* hashes, int n, const RemoteQuery& q,
                                const std::vector<RemoteHit>&) {
    QueryReply out;
    out.status = RemoteStatus::UNAVAILABLE;
    if (breaker_.gate(rank) == PeerBreaker::Gate::SKIP) {
        breaker_.skipped(rank, PeerBreaker::Op::WALK);
        return out;
    }
    std::shared_ptr<Session> s = session(rank);
    if (!s) return out;
    if (!s->usable || s->stale) {
        out.status = s->stale ? RemoteStatus::STALE_TARGET : RemoteStatus::BAD_REQUEST;
        return out;
    }
    const auto t0 = Clock::now();
    out = s->ops->query(hashes, n, q);
    if (out.status == RemoteStatus::OK) {
        breaker_.succeeded(rank);
        out.pin.source_rank = (int32_t)rank;
        out.pin.source_instance = s->instance;
    } else if (unreachable(out.status)) {
        breaker_.failed(rank, "query",
                        out.status == RemoteStatus::TIMEOUT ? "timed out" : "unavailable",
                        since(t0));
        breaker_.skipped(rank, PeerBreaker::Op::WALK);
    } else if (out.status == RemoteStatus::STALE_TARGET) {
        s->stale = true;   // the peer restarted; peer_changed() brings its new entry
    }
    return out;
}

void GrpcPeerIndex::unlock(const uint64_t* hashes, int n, PinRecord&& pin) {
    const int rank = pin.source_rank;
    std::shared_ptr<Session> s = session(rank);
    // The run that took the pin is gone, and its pins with it.
    if (!s || s->stale || s->instance != pin.source_instance) return;
    if (breaker_.gate(rank) == PeerBreaker::Gate::SKIP) {
        breaker_.skipped(rank, PeerBreaker::Op::RELEASE);
        return;
    }
    const auto t0 = Clock::now();
    RemoteStatus st = s->ops->unlock(hashes, n, pin);
    // RESOURCE_EXHAUSTED is the one refusal that leaves the op unrun, so the one retried.
    for (int i = 0; i < 5 && st == RemoteStatus::RESOURCE_EXHAUSTED; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10 << i));
        st = s->ops->unlock(hashes, n, pin);
    }
    if (st == RemoteStatus::OK || st == RemoteStatus::STALE_TARGET) return;
    if (unreachable(st))
        breaker_.failed(rank, "unlock",
                        st == RemoteStatus::TIMEOUT ? "timed out" : "unavailable", since(t0));
    fprintf(stderr, "[GrpcPeerIndex rank %d] unlock on peer rank %d failed (status %d); "
            "its pin may stay\n", peers_.my_rank(), rank, (int)st);
}

} // namespace radixshmem
