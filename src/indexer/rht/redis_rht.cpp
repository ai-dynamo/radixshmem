// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "rht_redis.h"

#include <stdexcept>

#ifdef RADIXSHMEM_HAS_REDIS

#include "peer_breaker.hpp"

#include <hiredis/hiredis.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <vector>

namespace radixshmem {

namespace {

using Clock = std::chrono::steady_clock;
using Op = PeerBreaker::Op;

const char* kRemoveScript =
    "local v = redis.call('HGET', KEYS[1], ARGV[1]) "
    "if v and string.sub(v, 1, 4) == ARGV[2] then "
    "return redis.call('HDEL', KEYS[1], ARGV[1]) end return 0";

struct Endpoint {
    std::string host;
    int port = 0;
    std::string str() const { return host + ":" + std::to_string(port); }
    bool operator==(const Endpoint& o) const { return host == o.host && port == o.port; }
};

std::vector<Endpoint> parse_seeds(const std::string& s) {
    std::vector<Endpoint> out;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t end = s.find(',', pos);
        if (end == std::string::npos) end = s.size();
        std::string e = s.substr(pos, end - pos);
        e.erase(0, e.find_first_not_of(" \t"));
        e.erase(e.find_last_not_of(" \t") + 1);
        size_t colon = e.rfind(':');
        if (colon != std::string::npos && colon > 0 && std::atoi(e.c_str() + colon + 1) > 0)
            out.push_back({e.substr(0, colon), std::atoi(e.c_str() + colon + 1)});
        pos = end + 1;
    }
    return out;
}

// Redis Cluster key slot: CRC16/XMODEM; our keys carry no hash tag.
int key_slot(const char* p, size_t n) {
    static const std::array<uint16_t, 256> tab = [] {
        std::array<uint16_t, 256> t{};
        for (int i = 0; i < 256; i++) {
            uint16_t c = (uint16_t)(i << 8);
            for (int j = 0; j < 8; j++)
                c = (c & 0x8000) ? (uint16_t)((c << 1) ^ 0x1021) : (uint16_t)(c << 1);
            t[i] = c;
        }
        return t;
    }();
    uint16_t crc = 0;
    for (size_t i = 0; i < n; i++)
        crc = (uint16_t)((crc << 8) ^ tab[((crc >> 8) ^ (uint8_t)p[i]) & 0xff]);
    return crc & 16383;
}

timeval tv_ms(int ms) { return timeval{ms / 1000, (ms % 1000) * 1000}; }

double secs_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

void put_u32(char* p, uint32_t v) {
    for (int i = 0; i < 4; i++) p[i] = (char)(v >> (8 * i));
}

uint32_t get_u32(const char* p) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) v |= (uint32_t)(uint8_t)p[i] << (8 * i);
    return v;
}

std::string reply_error(redisReply* r, const char* fallback) {
    return r && r->type == REDIS_REPLY_ERROR ? std::string(r->str, r->len) : fallback;
}

// Connected and authenticated, or null with the reason in err.
redisContext* open_conn(const Endpoint& ep, int timeout_ms, std::string& err) {
    redisContext* c = redisConnectWithTimeout(ep.host.c_str(), ep.port, tv_ms(timeout_ms));
    if (!c || c->err) {
        err = c ? c->errstr : "out of memory";
        if (c) redisFree(c);
        return nullptr;
    }
    redisSetTimeout(c, tv_ms(timeout_ms));
    int one = 1;
    setsockopt(c->fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    const char* pw = std::getenv("RADIXSHMEM_RHT_REDIS_PASSWORD");
    if (pw && *pw) {
        redisReply* r = (redisReply*)redisCommand(c, "AUTH %s", pw);
        bool ok = r && r->type == REDIS_REPLY_STATUS;
        if (!ok) err = reply_error(r, c->err ? c->errstr : "AUTH failed");
        if (r) freeReplyObject(r);
        if (!ok) {
            redisFree(c);
            return nullptr;
        }
    }
    return c;
}

// A cluster node: its connection, this batch's commands and the replies its
// async writes still owe.
struct Node {
    Endpoint ep;
    redisContext* c = nullptr;
    std::vector<int> tags;
    size_t next = 0;
    std::deque<Op> owed;
};

// Slot routing over hiredis: one pipeline per node per batch, every node's
// replies read with one deadline. Not thread-safe.
class RedisCluster {
public:
    using OnReply = std::function<void(int tag, redisReply* r)>;

    RedisCluster(const RedisRhtConfig& cfg, PeerBreaker& breaker)
        : seeds_(parse_seeds(cfg.seeds)), timeout_ms_(cfg.timeout_ms), breaker_(breaker) {
        refresh();
    }

    ~RedisCluster() {
        for (Node& n : nodes_)
            if (n.c) redisFree(n.c);
    }

    RedisCluster(const RedisCluster&) = delete;
    RedisCluster& operator=(const RedisCluster&) = delete;

    // The connected node serving key, or -1 with the op counted as dropped.
    int route(const char* key, size_t len, Op op) {
        if (stale_ && Clock::now() - refreshed_ > std::chrono::milliseconds(200)) refresh();
        int m = owner_[key_slot(key, len)];
        if (m < 0) {
            stale_ = true;
            breaker_.skipped(-1, op);
            return -1;
        }
        return usable(m, op) ? m : -1;
    }

    bool usable(int m, Op op) {
        if (nodes_[m].c) return true;
        if (breaker_.gate(m) != PeerBreaker::Gate::SKIP && connect(m)) return true;
        stale_ = true;   // a failover moves the slots: look again soon
        breaker_.skipped(m, op);
        return false;
    }

    void append(int m, int argc, const char** argv, const size_t* lens, int tag) {
        Node& n = nodes_[m];
        if (n.tags.empty()) touched_.push_back(m);
        redisAppendCommandArgv(n.c, argc, argv, lens);
        n.tags.push_back(tag);
    }

    // Sends this batch; with wait its replies go to on_reply, else they are
    // owed and read before the node's next batch.
    void send(Op op, bool wait, const OnReply& on_reply) {
        t0_ = Clock::now();
        std::vector<int> live;
        for (int m : touched_) {
            Node& n = nodes_[m];
            int done = 0;
            while (!done && redisBufferWrite(n.c, &done) == REDIS_OK) {}
            if (!done) {
                fail(m, "write", op);
            } else if (wait) {
                live.push_back(m);
            } else {
                n.owed.insert(n.owed.end(), n.tags.size(), op);
                n.tags.clear();
            }
        }
        touched_.clear();
        read(live, op, on_reply);
    }

    void drain() {
        t0_ = Clock::now();
        std::vector<int> live;
        for (size_t m = 0; m < nodes_.size(); m++)
            if (nodes_[m].c && !nodes_[m].owed.empty()) live.push_back((int)m);
        read(live, Op::RHT_WRITE, nullptr);
    }

    std::vector<int> masters() const {
        std::vector<int> v(owner_.begin(), owner_.end());
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        v.erase(std::remove(v.begin(), v.end(), -1), v.end());
        return v;
    }

    const std::string& sha() const { return sha_; }

private:
    bool connect(int m) {
        Node& n = nodes_[m];
        auto t0 = Clock::now();
        std::string err;
        n.c = open_conn(n.ep, timeout_ms_, err);
        if (n.c) {
            redisReply* r = (redisReply*)redisCommand(n.c, "SCRIPT LOAD %s", kRemoveScript);
            if (r && r->type == REDIS_REPLY_STRING) {
                sha_.assign(r->str, r->len);
            } else {
                err = reply_error(r, n.c->err ? n.c->errstr : "SCRIPT LOAD failed");
                redisFree(n.c);
                n.c = nullptr;
            }
            if (r) freeReplyObject(r);
        }
        if (!n.c) {
            breaker_.failed(m, "connect", err.c_str(), secs_since(t0));
            return false;
        }
        breaker_.succeeded(m);
        return true;
    }

    void fail(int m, const char* what, Op op) {
        Node& n = nodes_[m];
        std::string err = n.c && n.c->err ? n.c->errstr : what;
        for (Op o : n.owed) breaker_.skipped(m, o);
        if (n.next < n.tags.size()) breaker_.skipped(m, op, n.tags.size() - n.next);
        n.owed.clear();
        n.tags.clear();
        n.next = 0;
        if (n.c) redisFree(n.c);
        n.c = nullptr;
        breaker_.failed(m, what, err.c_str(), secs_since(t0_));
        stale_ = true;
    }

    // 1: every reply read, 0: more to come, -1: the node failed.
    int consume(int m, Op op, const OnReply& on_reply) {
        Node& n = nodes_[m];
        if (!n.c) return -1;
        while (!n.owed.empty() || n.next < n.tags.size()) {
            void* raw = nullptr;
            if (redisGetReplyFromReader(n.c, &raw) != REDIS_OK) {
                fail(m, "protocol", op);
                return -1;
            }
            if (!raw) return 0;
            redisReply* r = (redisReply*)raw;
            bool err = r->type == REDIS_REPLY_ERROR;
            if (err && (!std::strncmp(r->str, "MOVED", 5) || !std::strncmp(r->str, "ASK", 3)))
                stale_ = true;
            if (!n.owed.empty()) {
                if (err) breaker_.skipped(m, n.owed.front());
                n.owed.pop_front();
            } else {
                if (err) breaker_.skipped(m, op);
                else if (on_reply) on_reply(n.tags[n.next], r);
                n.next++;
            }
            freeReplyObject(r);
        }
        n.tags.clear();
        n.next = 0;
        breaker_.succeeded(m);
        return 1;
    }

    void read(std::vector<int> live, Op op, const OnReply& on_reply) {
        auto deadline = t0_ + std::chrono::milliseconds(timeout_ms_);
        std::vector<pollfd> fds;
        while (true) {
            live.erase(std::remove_if(live.begin(), live.end(),
                                      [&](int m) { return consume(m, op, on_reply) != 0; }),
                       live.end());
            if (live.empty()) return;
            auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - Clock::now()).count();
            if (left <= 0) {
                for (int m : live) fail(m, "timeout", op);
                return;
            }
            fds.clear();
            for (int m : live) fds.push_back({nodes_[m].c->fd, POLLIN, 0});
            int rc = ::poll(fds.data(), fds.size(), (int)left);
            if (rc < 0 && errno != EINTR) {
                for (int m : live) fail(m, "poll", op);
                return;
            }
            for (size_t i = 0; i < fds.size(); i++)
                if (fds[i].revents && redisBufferRead(nodes_[live[i]].c) != REDIS_OK)
                    fail(live[i], "read", op);
        }
    }

    // Reloads the slot map from any node that answers; nodes are only added.
    void refresh() {
        refreshed_ = Clock::now();
        std::vector<Endpoint> cand;
        for (const Node& n : nodes_) cand.push_back(n.ep);
        cand.insert(cand.end(), seeds_.begin(), seeds_.end());
        for (const Endpoint& ep : cand) {
            std::string err;
            redisContext* c = open_conn(ep, timeout_ms_, err);
            if (!c) continue;
            redisReply* r = (redisReply*)redisCommand(c, "CLUSTER SLOTS");
            bool ok = r && r->type == REDIS_REPLY_ARRAY && r->elements > 0;
            if (ok) {
                std::vector<int> owner(16384, -1);
                for (size_t i = 0; i < r->elements; i++) {
                    redisReply* e = r->element[i];
                    if (e->type != REDIS_REPLY_ARRAY || e->elements < 3) continue;
                    redisReply* a = e->element[2];
                    if (a->type != REDIS_REPLY_ARRAY || a->elements < 2 ||
                        a->element[0]->type != REDIS_REPLY_STRING || !a->element[0]->len)
                        continue;
                    int idx = node_of({std::string(a->element[0]->str, a->element[0]->len),
                                       (int)a->element[1]->integer});
                    long long lo = std::max(0LL, e->element[0]->integer);
                    long long hi = std::min(16383LL, e->element[1]->integer);
                    for (long long s = lo; s <= hi; s++) owner[s] = idx;
                }
                owner_.swap(owner);
                stale_ = false;
            }
            if (r) freeReplyObject(r);
            redisFree(c);
            if (ok) return;
        }
    }

    int node_of(const Endpoint& ep) {
        for (size_t i = 0; i < nodes_.size(); i++)
            if (nodes_[i].ep == ep) return (int)i;
        nodes_.push_back(Node{ep});
        breaker_.name((int)nodes_.size() - 1, ep.str());
        return (int)nodes_.size() - 1;
    }

    std::vector<Endpoint> seeds_;
    int timeout_ms_;
    PeerBreaker& breaker_;
    std::deque<Node> nodes_;   // index = breaker id; references stay valid
    std::vector<int> owner_ = std::vector<int>(16384, -1);
    std::vector<int> touched_;
    std::string sha_;
    bool stale_ = true;
    Clock::time_point refreshed_{};
    Clock::time_point t0_{};
};

// A cluster client plus the buffer a key's name is written into.
struct Conn {
    RedisCluster cluster;
    std::string key;

    Conn(const RedisRhtConfig& cfg, PeerBreaker& breaker)
        : cluster(cfg, breaker), key(cfg.key_prefix + std::string(16, '0')) {}

    int route(uint64_t k, Op op) {
        static const char hex[] = "0123456789abcdef";
        char* p = &key[key.size() - 16];
        for (int i = 15; i >= 0; i--, k >>= 4) p[i] = hex[k & 15];
        return cluster.route(key.data(), key.size(), op);
    }
};

// Writes share one client so flush() drains them all; lookups take an idle
// client, so concurrent callers do not wait on each other's round trips.
class RedisRht final : public RhtBackend {
public:
    RedisRht(const RedisRhtConfig& cfg, int my_rank, PeerFaultCounters* faults)
        : cfg_(cfg), my_rank_(my_rank),
          breaker_("[RadixClient rank " + std::to_string(my_rank) + "] Redis RHT", faults),
          writer_(cfg, breaker_) {}

    void insert_batch(const RhtInsertEntry* e, int count) override {
        std::lock_guard<std::mutex> g(write_mu_);
        Conn& w = writer_;
        for (int i = 0; i < count; i++) {
            int m = w.route(e[i].key, Op::RHT_WRITE);
            if (m < 0) continue;
            std::string field = std::to_string(e[i].node_id);
            char val[8];
            put_u32(val, e[i].tree_node_id);
            put_u32(val + 4, e[i].num_blocks);
            const char* argv[] = {"HSET", w.key.data(), field.c_str(), val};
            const size_t lens[] = {4, w.key.size(), field.size(), sizeof(val)};
            w.cluster.append(m, 4, argv, lens, i);
        }
        w.cluster.send(Op::RHT_WRITE, false, nullptr);
    }

    void remove_batch(const RhtRemoveEntry* e, int count) override {
        std::lock_guard<std::mutex> g(write_mu_);
        Conn& w = writer_;
        for (int i = 0; i < count; i++) {
            int m = w.route(e[i].key, Op::RHT_REMOVE);
            if (m < 0) continue;
            std::string field = std::to_string(e[i].node_id);
            char tnid[4];
            put_u32(tnid, e[i].tree_node_id);
            const std::string& sha = w.cluster.sha();
            const char* argv[] = {"EVALSHA", sha.c_str(), "1", w.key.data(), field.c_str(), tnid};
            const size_t lens[] = {7, sha.size(), 1, w.key.size(), field.size(), sizeof(tnid)};
            w.cluster.append(m, 6, argv, lens, i);
        }
        w.cluster.send(Op::RHT_REMOVE, false, nullptr);
    }

    void lookup_batch(const uint64_t* keys, int count,
                      std::vector<RHTValue>* results) override {
        std::unique_ptr<Conn> c = acquire();
        for (int i = 0; i < count; i++) {
            results[i].clear();
            int m = c->route(keys[i], Op::RHT_LOOKUP);
            if (m < 0) continue;
            const char* argv[] = {"HGETALL", c->key.data()};
            const size_t lens[] = {7, c->key.size()};
            c->cluster.append(m, 2, argv, lens, i);
        }
        c->cluster.send(Op::RHT_LOOKUP, true, [&](int i, redisReply* r) {
            if (r->type != REDIS_REPLY_ARRAY) return;
            for (size_t j = 0; j + 1 < r->elements; j += 2) {
                redisReply* f = r->element[j];
                redisReply* v = r->element[j + 1];
                if (f->type != REDIS_REPLY_STRING || v->type != REDIS_REPLY_STRING || v->len != 8)
                    continue;
                results[i].push_back({std::atoi(f->str), get_u32(v->str), get_u32(v->str + 4)});
            }
        });
        release(std::move(c));
    }

    void flush() override {
        std::lock_guard<std::mutex> g(write_mu_);
        writer_.cluster.drain();
    }

    // Drops this rank's field from every key under the prefix.
    void reset() override {
        std::lock_guard<std::mutex> g(write_mu_);
        RedisCluster& w = writer_.cluster;
        w.drain();
        const std::string field = std::to_string(my_rank_);
        const std::string match = cfg_.key_prefix + "*";
        for (int m : w.masters()) {
            std::string cursor = "0";
            do {
                if (!w.usable(m, Op::RHT_REMOVE)) break;
                const char* argv[] = {"SCAN", cursor.c_str(), "MATCH", match.c_str(),
                                      "COUNT", "1000"};
                const size_t lens[] = {4, cursor.size(), 5, match.size(), 5, 4};
                w.append(m, 6, argv, lens, 0);
                std::vector<std::string> found;
                bool scanned = false;
                w.send(Op::RHT_REMOVE, true, [&](int, redisReply* r) {
                    if (r->type != REDIS_REPLY_ARRAY || r->elements != 2) return;
                    cursor.assign(r->element[0]->str, r->element[0]->len);
                    for (size_t j = 0; j < r->element[1]->elements; j++)
                        found.emplace_back(r->element[1]->element[j]->str,
                                           r->element[1]->element[j]->len);
                    scanned = true;
                });
                if (!scanned) break;
                for (const std::string& k : found) {
                    const char* a[] = {"HDEL", k.c_str(), field.c_str()};
                    const size_t l[] = {4, k.size(), field.size()};
                    w.append(m, 3, a, l, 0);
                }
                w.send(Op::RHT_REMOVE, false, nullptr);
            } while (cursor != "0");
        }
        w.drain();
    }

    void dump(uint64_t key, std::vector<std::string>& lines) override {
        lines.clear();
        std::unique_ptr<Conn> c = acquire();
        int m = c->route(key, Op::RHT_LOOKUP);
        if (m >= 0) {
            const char* argv[] = {"HGETALL", c->key.data()};
            const size_t lens[] = {7, c->key.size()};
            c->cluster.append(m, 2, argv, lens, 0);
            c->cluster.send(Op::RHT_LOOKUP, true, [&](int, redisReply* r) {
                if (r->type != REDIS_REPLY_ARRAY) return;
                for (size_t j = 0; j + 1 < r->elements; j += 2) {
                    redisReply* v = r->element[j + 1];
                    char buf[96];
                    if (v->len == 8)
                        snprintf(buf, sizeof(buf), "r%s:t%u/%u", r->element[j]->str,
                                 get_u32(v->str), get_u32(v->str + 4));
                    else
                        snprintf(buf, sizeof(buf), "r%s:%zu bytes", r->element[j]->str,
                                 (size_t)v->len);
                    lines.emplace_back(buf);
                }
            });
        }
        release(std::move(c));
    }

private:
    std::unique_ptr<Conn> acquire() {
        {
            std::lock_guard<std::mutex> g(pool_mu_);
            if (!idle_.empty()) {
                std::unique_ptr<Conn> c = std::move(idle_.back());
                idle_.pop_back();
                return c;
            }
        }
        return std::make_unique<Conn>(cfg_, breaker_);
    }

    void release(std::unique_ptr<Conn> c) {
        std::lock_guard<std::mutex> g(pool_mu_);
        idle_.push_back(std::move(c));
    }

    const RedisRhtConfig cfg_;
    const int my_rank_;
    PeerBreaker breaker_;
    std::mutex write_mu_;
    Conn writer_;
    std::mutex pool_mu_;
    std::vector<std::unique_ptr<Conn>> idle_;
};

// The settings the RHT relies on; CONFIG may be disabled, then none is checked.
void warn_config(redisContext* c, const Endpoint& ep) {
    auto get = [&](const char* name) {
        std::string v;
        redisReply* r = (redisReply*)redisCommand(c, "CONFIG GET %s", name);
        if (r && r->type == REDIS_REPLY_ARRAY && r->elements == 2 &&
            r->element[1]->type == REDIS_REPLY_STRING)
            v.assign(r->element[1]->str, r->element[1]->len);
        if (r) freeReplyObject(r);
        return v;
    };
    std::string policy = get("maxmemory-policy");
    if (!policy.empty() && policy.rfind("allkeys", 0) != 0)
        fprintf(stderr, "[Redis RHT] %s: maxmemory-policy %s; stale routes are evicted "
                "only under allkeys-*\n", ep.str().c_str(), policy.c_str());
    if (get("cluster-require-full-coverage") == "yes")
        fprintf(stderr, "[Redis RHT] %s: cluster-require-full-coverage yes; one failed "
                "master stops every lookup\n", ep.str().c_str());
}

} // namespace

std::unique_ptr<RhtBackend> make_redis_rht(const RedisRhtConfig& cfg, int my_rank,
                                           PeerFaultCounters* faults) {
    if (parse_seeds(cfg.seeds).empty())
        throw std::runtime_error("Redis RHT: no host:port in '" + cfg.seeds + "'");
    if (cfg.key_prefix.find_first_of("{}") != std::string::npos)
        throw std::runtime_error("Redis RHT: key prefix '" + cfg.key_prefix +
                                 "' must not contain { or }");
    if (cfg.timeout_ms <= 0) throw std::runtime_error("Redis RHT: timeout_ms must be > 0");
    return std::make_unique<RedisRht>(cfg, my_rank, faults);
}

std::string redis_rht_check(const RedisRhtConfig& cfg) {
    std::vector<Endpoint> seeds = parse_seeds(cfg.seeds);
    if (seeds.empty()) return "no host:port in '" + cfg.seeds + "'";
    std::string why;
    for (const Endpoint& ep : seeds) {
        std::string err;
        redisContext* c = open_conn(ep, std::max(cfg.timeout_ms, 1000), err);
        if (!c) {
            why += ep.str() + ": " + err + "; ";
            continue;
        }
        redisReply* r = (redisReply*)redisCommand(c, "CLUSTER INFO");
        bool ok = r && r->type == REDIS_REPLY_STRING && std::strstr(r->str, "cluster_state:ok");
        if (!ok) why += ep.str() + ": " + reply_error(r, "cluster_state is not ok") + "; ";
        if (r) freeReplyObject(r);
        if (ok) warn_config(c, ep);
        redisFree(c);
        if (ok) return "";
    }
    return why;
}

} // namespace radixshmem

#else

namespace radixshmem {

std::unique_ptr<RhtBackend> make_redis_rht(const RedisRhtConfig&, int, PeerFaultCounters*) {
    throw std::runtime_error("Redis RHT: radixshmem was built without hiredis");
}

std::string redis_rht_check(const RedisRhtConfig&) {
    return "radixshmem was built without hiredis";
}

} // namespace radixshmem

#endif
