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
#include <signal.h>
#include <sys/socket.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
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

// While alive, a write to a closed connection fails with EPIPE on this thread
// instead of raising SIGPIPE, which would end the process.
class SigpipeBlock {
public:
    SigpipeBlock() {
        sigset_t pipe = pipe_set();
        pthread_sigmask(SIG_BLOCK, &pipe, &old_);
        had_ = pending();
    }
    ~SigpipeBlock() {
        if (!had_ && pending()) {
            sigset_t pipe = pipe_set();
            const timespec now{0, 0};
            while (sigtimedwait(&pipe, nullptr, &now) < 0 && errno == EINTR) {}
        }
        pthread_sigmask(SIG_SETMASK, &old_, nullptr);
    }
    SigpipeBlock(const SigpipeBlock&) = delete;
    SigpipeBlock& operator=(const SigpipeBlock&) = delete;

private:
    static sigset_t pipe_set() {
        sigset_t s;
        sigemptyset(&s);
        sigaddset(&s, SIGPIPE);
        return s;
    }
    static bool pending() {
        sigset_t s;
        sigpending(&s);
        return sigismember(&s, SIGPIPE) == 1;
    }
    sigset_t old_;
    bool had_ = false;
};

// Connected and authenticated, or null with the reason in err.
redisContext* open_conn(const Endpoint& ep, int timeout_ms, std::string& err) {
    SigpipeBlock pipe;
    redisContext* c = redisConnectWithTimeout(ep.host.c_str(), ep.port, tv_ms(timeout_ms));
    if (!c || c->err) {
        err = c ? c->errstr : "out of memory";
        if (c) redisFree(c);
        return nullptr;
    }
    redisSetTimeout(c, tv_ms(timeout_ms));
    int one = 1;
    setsockopt(c->fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    const char* pw = std::getenv("RADIXSHMEM_REDIS_PASSWORD");
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

constexpr int kMaxMasters = 4096;    // endpoint ids are never reused
constexpr size_t kMaxOwed = 16384;   // async replies a connection may leave unread

struct Master {
    Endpoint ep;
    std::atomic<bool> down{false};
    std::atomic<uint32_t> gen{0};    // bumped each time the master goes down
};

// Masters, slot owners and health, shared by every connection of one RedisRht.
// A maintenance thread refreshes the slot map and probes the masters that are
// down; requests only read.
class ClusterView {
public:
    using SlotMap = std::vector<int16_t>;

    ClusterView(const RedisRhtConfig& cfg, PeerBreaker& breaker)
        : seeds_(parse_seeds(cfg.seeds)), timeout_ms_(std::max(cfg.timeout_ms, 200)),
          breaker_(breaker), masters_(new Master[kMaxMasters]),
          slots_(std::make_shared<SlotMap>(16384, -1)) {
        stale_ = !refresh();
        thread_ = std::thread([this] { run(); });
    }

    ~ClusterView() {
        {
            std::lock_guard<std::mutex> g(mu_);
            stop_ = true;
        }
        cv_.notify_one();
        thread_.join();
    }

    ClusterView(const ClusterView&) = delete;
    ClusterView& operator=(const ClusterView&) = delete;

    std::shared_ptr<const SlotMap> slots() const {
        std::lock_guard<std::mutex> g(mu_);
        return slots_;
    }

    const Endpoint& endpoint(int id) const { return masters_[id].ep; }
    uint64_t version() const { return version_.load(std::memory_order_acquire); }
    // Bumped when the slot map changes or a down master answers again.
    uint64_t epoch() const {
        return version_.load(std::memory_order_acquire) + ups_.load(std::memory_order_acquire);
    }
    bool down(int id) const { return masters_[id].down.load(std::memory_order_acquire); }
    uint32_t generation(int id) const { return masters_[id].gen.load(std::memory_order_acquire); }

    // A request failed on master id: the first failure marks it down; only
    // failed probes lengthen the backoff.
    void failed(int id, const char* what, const std::string& err, double secs) {
        if (!masters_[id].down.exchange(true, std::memory_order_acq_rel)) {
            masters_[id].gen.fetch_add(1, std::memory_order_acq_rel);
            breaker_.failed(id, what, err.c_str(), secs);
        }
        mark_stale();   // a failover moves the slots
    }

    void mark_stale() {
        if (!stale_.exchange(true)) cv_.notify_one();
    }

    // No master that owns slots is down.
    bool settled() const { return down_serving().empty(); }

private:
    void run() {
        auto next_refresh = Clock::now();
        int backoff_ms = 200;
        std::unique_lock<std::mutex> lk(mu_);
        while (!stop_) {
            cv_.wait_for(lk, std::chrono::milliseconds(100));
            if (stop_) break;
            lk.unlock();
            std::vector<int> down = down_serving();
            // While a master that owns slots is down, look for its failover every second.
            if ((stale_.load() || !down.empty()) && Clock::now() >= next_refresh) {
                stale_ = false;
                bool ok = refresh();
                if (!ok) stale_ = true;
                if (ok) down = down_serving();
                backoff_ms = !ok ? std::min(backoff_ms * 2, 5000) : down.empty() ? 200 : 1000;
                next_refresh = Clock::now() + std::chrono::milliseconds(backoff_ms);
            }
            for (int id : down)
                if (breaker_.gate(id) == PeerBreaker::Gate::PROBE) probe(id);
            lk.lock();
        }
    }

    // Loads the slot map from the first node that answers: masters that are
    // up, then the seeds.
    bool refresh() {
        SigpipeBlock pipe;
        std::vector<Endpoint> cand;
        int n = count_.load(std::memory_order_acquire);
        for (int id = 0; id < n; id++)
            if (!down(id)) cand.push_back(masters_[id].ep);
        for (const Endpoint& s : seeds_)
            if (std::find(cand.begin(), cand.end(), s) == cand.end()) cand.push_back(s);
        for (const Endpoint& ep : cand) {
            std::string err;
            redisContext* c = open_conn(ep, timeout_ms_, err);
            if (!c) continue;
            redisReply* r = (redisReply*)redisCommand(c, "CLUSTER SLOTS");
            bool ok = r && r->type == REDIS_REPLY_ARRAY && r->elements > 0;
            if (ok) {
                auto map = std::make_shared<SlotMap>(16384, -1);
                for (size_t i = 0; i < r->elements; i++) {
                    redisReply* e = r->element[i];
                    if (e->type != REDIS_REPLY_ARRAY || e->elements < 3) continue;
                    redisReply* a = e->element[2];
                    if (a->type != REDIS_REPLY_ARRAY || a->elements < 2 ||
                        a->element[0]->type != REDIS_REPLY_STRING || !a->element[0]->len)
                        continue;
                    int id = id_of({std::string(a->element[0]->str, a->element[0]->len),
                                    (int)a->element[1]->integer});
                    if (id < 0) continue;
                    long long lo = std::max(0LL, e->element[0]->integer);
                    long long hi = std::min(16383LL, e->element[1]->integer);
                    for (long long s = lo; s <= hi; s++) (*map)[s] = (int16_t)id;
                }
                std::lock_guard<std::mutex> g(mu_);
                if (*map != *slots_) version_.fetch_add(1, std::memory_order_acq_rel);
                slots_ = std::move(map);
            }
            if (r) freeReplyObject(r);
            redisFree(c);
            if (ok) return true;
        }
        return false;
    }

    // Masters that are down and still own slots.
    std::vector<int> down_serving() const {
        int n = count_.load(std::memory_order_acquire);
        std::vector<int> ids;
        for (int id = 0; id < n; id++)
            if (down(id)) ids.push_back(id);
        if (ids.empty()) return ids;
        const std::shared_ptr<const SlotMap> map = slots();
        std::vector<char> serving(kMaxMasters, 0);
        for (int16_t m : *map) if (m >= 0 && m < kMaxMasters) serving[m] = 1;
        ids.erase(std::remove_if(ids.begin(), ids.end(), [&](int id) { return !serving[id]; }),
                  ids.end());
        return ids;
    }

    void probe(int id) {
        SigpipeBlock pipe;
        auto t0 = Clock::now();
        std::string err;
        bool ok = false;
        if (redisContext* c = open_conn(masters_[id].ep, timeout_ms_, err)) {
            redisReply* r = (redisReply*)redisCommand(c, "PING");
            ok = r && r->type == REDIS_REPLY_STATUS;
            if (!ok) err = reply_error(r, c->err ? c->errstr : "PING failed");
            if (r) freeReplyObject(r);
            redisFree(c);
        }
        if (!ok) {
            breaker_.failed(id, "probe", err.c_str(), secs_since(t0));
            return;
        }
        breaker_.succeeded(id);
        masters_[id].down.store(false, std::memory_order_release);
        ups_.fetch_add(1, std::memory_order_acq_rel);
    }

    // Only the maintenance thread (and the constructor before it) adds masters.
    int id_of(const Endpoint& ep) {
        int n = count_.load(std::memory_order_relaxed);
        for (int i = 0; i < n; i++)
            if (masters_[i].ep == ep) return i;
        if (n == kMaxMasters) return -1;
        masters_[n].ep = ep;
        breaker_.name(n, ep.str());
        count_.store(n + 1, std::memory_order_release);
        return n;
    }

    const std::vector<Endpoint> seeds_;
    const int timeout_ms_;   // background connects and commands
    PeerBreaker& breaker_;
    std::unique_ptr<Master[]> masters_;
    std::atomic<int> count_{0};
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::shared_ptr<const SlotMap> slots_;
    std::atomic<uint64_t> version_{0};   // bumped when slot owners change
    std::atomic<uint64_t> ups_{0};       // probes that found a master up
    std::atomic<bool> stale_{false};
    bool stop_ = false;
    std::thread thread_;
};

// A master's connection, this batch's commands and the replies its async
// writes still owe.
struct Node {
    redisContext* c = nullptr;
    uint32_t gen = 0;
    bool in_batch = false;   // in touched_: this batch has commands on c
    std::vector<int> tags;
    size_t next = 0;
    std::deque<Op> owed;
};

// One caller's connections: one pipeline per master per batch, every master's
// replies read under one deadline. A patient Conn (the writer) waits as long
// as replies keep coming, so a busy master is not taken for a dead one.
// Not thread-safe.
class Conn {
public:
    using OnReply = std::function<void(int tag, redisReply* r)>;

    Conn(ClusterView& view, const RedisRhtConfig& cfg, PeerBreaker& breaker, bool patient)
        : view_(view), breaker_(breaker), timeout_ms_(cfg.timeout_ms), patient_(patient),
          key_(cfg.key_prefix + std::string(16, '0')), slots_(view.slots()) {}

    ~Conn() {
        for (Node& n : nodes_)
            if (n.c) redisFree(n.c);
    }

    Conn(const Conn&) = delete;
    Conn& operator=(const Conn&) = delete;

    // Takes the current slot map for the next batch.
    void begin() { slots_ = view_.slots(); }

    const std::string& key() const { return key_; }

    // The master serving k, whose Redis key is left in key(), or -1 with the
    // op counted as dropped.
    int route(uint64_t k, Op op) {
        int m = owner(k);
        if (m < 0) {
            view_.mark_stale();
            breaker_.skipped(-1, op);
            return -1;
        }
        return usable(m, op) ? m : -1;
    }

    // The master serving k in this batch's map, or -1; counts nothing.
    int owner(uint64_t k) {
        static const char hex[] = "0123456789abcdef";
        char* p = &key_[key_.size() - 16];
        for (int i = 15; i >= 0; i--, k >>= 4) p[i] = hex[k & 15];
        return (*slots_)[key_slot(key_.data(), key_.size())];
    }

    bool down(int m) const { return view_.down(m); }

    // Error replies read so far.
    uint64_t errors() const { return errors_; }

    // While quiet, failed ops are not counted as dropped: a reset retries them.
    void set_quiet(bool q) { quiet_ = q; }

    // Master m is up and connected; else the op is counted as dropped. A
    // connection that already carries commands of this batch is kept.
    bool usable(int m, Op op) {
        if (view_.down(m)) {
            breaker_.skipped(m, op);
            return false;
        }
        Node& n = node(m);
        if (n.c && n.in_batch) return true;
        uint32_t gen = view_.generation(m);
        if (n.c && (n.gen != gen || hung_up(m)) && n.c) close(m, op);   // stale or closed by the server
        if (!n.c) {
            auto t0 = Clock::now();
            std::string err;
            n.c = open_conn(view_.endpoint(m), timeout_ms_, err);
            if (!n.c) {
                view_.failed(m, "connect", err, secs_since(t0));
                breaker_.skipped(m, op);
                return false;
            }
            n.gen = gen;
        }
        return true;
    }

    void append(int m, int argc, const char** argv, const size_t* lens, int tag) {
        Node& n = nodes_[m];
        if (!n.in_batch) {
            n.in_batch = true;
            touched_.push_back(m);
        }
        redisAppendCommandArgv(n.c, argc, argv, lens);
        n.tags.push_back(tag);
    }

    // Sends this batch; with wait its replies go to on_reply, else they are
    // owed and read as they arrive, waiting only once kMaxOwed pile up.
    void send(Op op, bool wait, const OnReply& on_reply) {
        t0_ = Clock::now();
        std::vector<int> live, backlog;
        {
            SigpipeBlock pipe;
            for (int m : touched_) {
                Node& n = nodes_[m];
                n.in_batch = false;
                if (!n.c) {
                    if (!quiet_) breaker_.skipped(m, op, n.tags.size());
                    n.tags.clear();
                    continue;
                }
                int done = 0;
                while (!done && redisBufferWrite(n.c, &done) == REDIS_OK) {}
                if (!done) {
                    fail(m, "write", op);
                } else if (wait) {
                    live.push_back(m);
                } else {
                    n.owed.insert(n.owed.end(), n.tags.size(), op);
                    n.tags.clear();
                    absorb(m, op);
                    if (n.c && n.owed.size() > kMaxOwed) backlog.push_back(m);
                }
            }
        }
        touched_.clear();
        t0_ = Clock::now();   // replies get the whole timeout, however long the writes took
        read(live, op, on_reply);
        if (!backlog.empty()) {
            t0_ = Clock::now();
            read(backlog, op, nullptr);
        }
    }

    void drain() {
        t0_ = Clock::now();
        std::vector<int> live;
        for (size_t m = 0; m < nodes_.size(); m++)
            if (nodes_[m].c && !nodes_[m].owed.empty()) live.push_back((int)m);
        read(live, Op::RHT_WRITE, nullptr);
    }

    std::vector<int> masters() const {
        std::vector<int> v(slots_->begin(), slots_->end());
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        v.erase(std::remove(v.begin(), v.end(), -1), v.end());
        return v;
    }

private:
    Node& node(int m) {
        while ((int)nodes_.size() <= m) nodes_.emplace_back();
        return nodes_[m];
    }

    // Drops m's connection; the replies it still owes count as dropped ops.
    void close(int m, Op op) {
        Node& n = nodes_[m];
        if (!quiet_) {
            for (Op o : n.owed) breaker_.skipped(m, o);
            if (n.next < n.tags.size()) breaker_.skipped(m, op, n.tags.size() - n.next);
        }
        n.owed.clear();
        n.tags.clear();
        n.next = 0;
        redisFree(n.c);
        n.c = nullptr;
    }

    void fail(int m, const char* what, Op op) {
        Node& n = nodes_[m];
        std::string err = n.c && n.c->err ? n.c->errstr : what;
        if (n.c) close(m, op);
        view_.failed(m, what, err, secs_since(t0_));
    }

    // The server closed m's connection (a restart, its idle timeout, CLIENT
    // KILL); the replies that arrived before are read first.
    bool hung_up(int m) {
        Node& n = nodes_[m];
        pollfd p{n.c->fd, POLLIN | POLLRDHUP, 0};
        if (::poll(&p, 1, 0) <= 0 || !(p.revents & (POLLRDHUP | POLLHUP | POLLERR))) return false;
        while (!n.owed.empty() && redisBufferRead(n.c) == REDIS_OK &&
               consume(m, n.owed.front(), nullptr) == 0) {}
        return true;
    }

    // Reads the replies that already arrived, without waiting.
    void absorb(int m, Op op) {
        pollfd p{nodes_[m].c->fd, POLLIN, 0};
        if (::poll(&p, 1, 0) <= 0) return;
        if (redisBufferRead(nodes_[m].c) != REDIS_OK) fail(m, "read", op);
        else consume(m, op, nullptr);
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
            replies_++;
            errors_ += err;
            if (err && (!std::strncmp(r->str, "MOVED", 5) || !std::strncmp(r->str, "ASK", 3)))
                view_.mark_stale();
            if (!n.owed.empty()) {
                if (err && !quiet_) breaker_.skipped(m, n.owed.front());
                n.owed.pop_front();
            } else {
                if (!err && on_reply) on_reply(n.tags[n.next], r);
                if (err && !quiet_) breaker_.skipped(m, op);
                n.next++;
            }
            freeReplyObject(r);
        }
        n.tags.clear();
        n.next = 0;
        return 1;
    }

    void read(std::vector<int> live, Op op, const OnReply& on_reply) {
        auto deadline = t0_ + std::chrono::milliseconds(timeout_ms_);
        uint64_t seen = replies_;
        std::vector<pollfd> fds;
        while (true) {
            live.erase(std::remove_if(live.begin(), live.end(),
                                      [&](int m) { return consume(m, op, on_reply) != 0; }),
                       live.end());
            if (live.empty()) return;
            auto now = Clock::now();
            if (patient_ && replies_ != seen) {
                seen = replies_;
                deadline = now + std::chrono::milliseconds(timeout_ms_);
            }
            auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
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

    ClusterView& view_;
    PeerBreaker& breaker_;
    const int timeout_ms_;
    const bool patient_;
    bool quiet_ = false;
    std::string key_;
    std::shared_ptr<const ClusterView::SlotMap> slots_;
    std::deque<Node> nodes_;   // by master id; references stay valid
    std::vector<int> touched_;
    uint64_t replies_ = 0, errors_ = 0;
    Clock::time_point t0_{};
};

// Writes share one connection set so flush() drains them all; lookups take an
// idle one, so concurrent callers do not wait on each other's round trips.
class RedisRht final : public RhtBackend {
public:
    RedisRht(const RedisRhtConfig& cfg, int my_rank, PeerFaultCounters* faults)
        : cfg_(cfg), my_rank_(my_rank),
          breaker_("[RadixClient rank " + std::to_string(my_rank) + "] Redis RHT", faults,
                   200, 2000),
          view_(cfg, breaker_), writer_(view_, cfg, breaker_, true) {}

    void insert_batch(const RhtInsertEntry* e, int count) override {
        std::lock_guard<std::mutex> g(write_mu_);
        Conn& w = writer_;
        w.begin();
        replay(w);
        for (int i = 0; i < count; i++) {
            const Held h{true, e[i].node_id, e[i].tree_node_id, e[i].num_blocks};
            int m = writable(w, e[i].key, Op::RHT_WRITE);
            if (m < 0) {
                hold(e[i].key, h, Op::RHT_WRITE);
                continue;
            }
            if (!held_.empty()) held_.erase(e[i].key);
            append(w, m, h);
        }
        w.send(Op::RHT_WRITE, false, nullptr);
    }

    void remove_batch(const RhtRemoveEntry* e, int count) override {
        std::lock_guard<std::mutex> g(write_mu_);
        Conn& w = writer_;
        w.begin();
        replay(w);
        for (int i = 0; i < count; i++) {
            const Held h{false, e[i].node_id, e[i].tree_node_id, 0};
            int m = writable(w, e[i].key, Op::RHT_REMOVE);
            if (m < 0) {
                hold(e[i].key, h, Op::RHT_REMOVE);
                continue;
            }
            if (!held_.empty()) held_.erase(e[i].key);
            append(w, m, h);
        }
        w.send(Op::RHT_REMOVE, false, nullptr);
    }

    void lookup_batch(const uint64_t* keys, int count,
                      std::vector<RHTValue>* results) override {
        std::unique_ptr<Conn> c = acquire();
        c->begin();
        for (int i = 0; i < count; i++) {
            results[i].clear();
            int m = c->route(keys[i], Op::RHT_LOOKUP);
            if (m < 0) continue;
            const char* argv[] = {"HGETALL", c->key().data()};
            const size_t lens[] = {7, c->key().size()};
            c->append(m, 2, argv, lens, i);
        }
        c->send(Op::RHT_LOOKUP, true, [&](int i, redisReply* r) {
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
        writer_.begin();
        replay(writer_);
        writer_.drain();
    }

    // Drops this rank's routes: the listed ones, or with count < 0 this rank's
    // field in every key under the prefix.
    void reset(const RhtRemoveEntry* own, int count) override {
        std::lock_guard<std::mutex> g(write_mu_);
        held_.clear();
        writer_.begin();
        writer_.drain();
        writer_.set_quiet(true);
        bool clean = count >= 0 ? remove_listed(own, count) : scan_remove(std::to_string(my_rank_));
        writer_.set_quiet(false);
        if (!clean)
            fprintf(stderr, "[RadixClient rank %d] Redis RHT: reset gave up after %d s; routes of "
                    "this rank remain on masters it could not reach\n", my_rank_, kRetrySecs);
    }

    // Deletes every key under the prefix, i.e. the routes of all ranks.
    bool clear() {
        std::lock_guard<std::mutex> g(write_mu_);
        writer_.begin();
        writer_.drain();
        writer_.set_quiet(true);
        bool clean = scan_remove("");
        writer_.set_quiet(false);
        return clean;
    }

    void dump(uint64_t key, std::vector<std::string>& lines) override {
        lines.clear();
        std::unique_ptr<Conn> c = acquire();
        c->begin();
        int m = c->route(key, Op::RHT_LOOKUP);
        if (m >= 0) {
            const char* argv[] = {"HGETALL", c->key().data()};
            const size_t lens[] = {7, c->key().size()};
            c->append(m, 2, argv, lens, 0);
            c->send(Op::RHT_LOOKUP, true, [&](int, redisReply* r) {
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
    // A write that a down master could not take, sent once the key's owner is up.
    struct Held {
        bool insert;
        int32_t node_id;
        uint32_t tree_node_id, num_blocks;
    };
    static constexpr size_t kMaxHeld = 200000;

    // The master to write key to, or -1 when it is down or unknown.
    int writable(Conn& w, uint64_t key, Op op) {
        int m = w.owner(key);
        if (m < 0) view_.mark_stale();
        return m >= 0 && !w.down(m) && w.usable(m, op) ? m : -1;
    }

    // Appends the write to the key whose name owner() or writable() left in w.
    void append(Conn& w, int m, const Held& h) {
        std::string field = std::to_string(h.node_id);
        if (h.insert) {
            char val[8];
            put_u32(val, h.tree_node_id);
            put_u32(val + 4, h.num_blocks);
            const char* argv[] = {"HSET", w.key().data(), field.c_str(), val};
            const size_t lens[] = {4, w.key().size(), field.size(), sizeof(val)};
            w.append(m, 4, argv, lens, 0);
        } else {
            char tnid[4];
            put_u32(tnid, h.tree_node_id);
            const char* argv[] = {"EVAL", kRemoveScript, "1", w.key().data(), field.c_str(), tnid};
            const size_t lens[] = {4, std::strlen(kRemoveScript), 1, w.key().size(),
                                   field.size(), sizeof(tnid)};
            w.append(m, 6, argv, lens, 0);
        }
    }

    // Keeps the latest write per key; past kMaxHeld the write is dropped.
    void hold(uint64_t key, const Held& h, Op op) {
        if (held_.size() >= kMaxHeld && !held_.count(key)) {
            breaker_.skipped(-1, op);
            return;
        }
        held_[key] = h;
    }

    // Sends the held writes whose master is up, once a master is back or the
    // slot map changed. The tree decides whether each still holds: an insert
    // goes out only while its node is live, a remove only while it is not, so
    // the rank's processes agree in whatever order they send theirs.
    void replay(Conn& w) {
        if (held_.empty() || view_.epoch() == replayed_) return;
        replayed_ = view_.epoch();
        bool any = false;
        for (auto it = held_.begin(); it != held_.end();) {
            const Held& h = it->second;
            int m = writable(w, it->first, h.insert ? Op::RHT_WRITE : Op::RHT_REMOVE);
            if (m < 0) {
                ++it;
                continue;
            }
            if (!cfg_.live || cfg_.live(it->first, h.tree_node_id) == h.insert) {
                append(w, m, h);
                any = true;
            }
            it = held_.erase(it);
        }
        if (any) w.send(Op::RHT_WRITE, false, nullptr);
    }

    static constexpr int kRetrySecs = 30;
    static constexpr size_t kChunk = 16384;

    // Waits until no master that owns slots is down, or the retries run out.
    bool wait_settled(Clock::time_point give_up) {
        do {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            if (view_.settled()) return true;
        } while (Clock::now() < give_up);
        return false;
    }

    // Drops this rank's field from the listed keys, whatever node it names (the
    // rank's routes are all stale once its tree is reset), and retries those a
    // down master, a slot move or an error reply kept from landing.
    bool remove_listed(const RhtRemoveEntry* own, int count) {
        Conn& w = writer_;
        const auto give_up = Clock::now() + std::chrono::seconds(kRetrySecs);
        std::vector<RhtRemoveEntry> todo(own, own + count);
        while (true) {
            w.begin();
            std::vector<RhtRemoveEntry> left;
            for (size_t b = 0; b < todo.size(); b += kChunk) {
                const size_t n = std::min(kChunk, todo.size() - b);
                std::vector<char> landed(n, 0);
                for (size_t i = 0; i < n; i++) {
                    const RhtRemoveEntry& e = todo[b + i];
                    int m = w.owner(e.key);
                    if (m < 0) view_.mark_stale();
                    if (m < 0 || w.down(m) || !w.usable(m, Op::RHT_REMOVE)) continue;
                    std::string field = std::to_string(e.node_id);
                    const char* argv[] = {"HDEL", w.key().data(), field.c_str()};
                    const size_t lens[] = {4, w.key().size(), field.size()};
                    w.append(m, 3, argv, lens, (int)i);
                }
                w.send(Op::RHT_REMOVE, true, [&](int i, redisReply*) { landed[i] = 1; });
                for (size_t i = 0; i < n; i++)
                    if (!landed[i]) left.push_back(todo[b + i]);
            }
            todo.swap(left);
            if (todo.empty() || !wait_settled(give_up)) break;
        }
        if (!todo.empty()) breaker_.skipped(-1, Op::RHT_REMOVE, todo.size());
        return todo.empty();
    }

    // Drops `field` from every key under the prefix, or the keys when it is
    // empty. A pass that a down master, an error reply or a slot move may have
    // left incomplete is repeated once the masters are up again.
    bool scan_remove(const std::string& field) {
        const auto give_up = Clock::now() + std::chrono::seconds(kRetrySecs);
        while (true) {
            const uint64_t version = view_.version(), errors = writer_.errors();
            writer_.begin();
            bool complete = scan_pass(field);
            writer_.drain();
            if (complete && writer_.errors() == errors && view_.version() == version) return true;
            if (!wait_settled(give_up)) return false;
        }
    }

    // One pass: each round sends a SCAN to every master still scanning, then the
    // removals for the keys that came back, so the masters work at the same time.
    bool scan_pass(const std::string& field) {
        Conn& w = writer_;
        const std::string match = cfg_.key_prefix + "*";
        const std::vector<int> masters = w.masters();
        const size_t n = masters.size();
        std::vector<std::string> cursor(n, "0");
        std::vector<char> done(n, 0);
        bool complete = true;
        while (true) {
            int scanning = 0;
            for (size_t i = 0; i < n; i++) {
                if (done[i]) continue;
                if (w.down(masters[i]) || !w.usable(masters[i], Op::RHT_REMOVE)) {
                    done[i] = 1;
                    complete = false;
                    continue;
                }
                const char* argv[] = {"SCAN", cursor[i].c_str(), "MATCH", match.c_str(),
                                      "COUNT", "1000"};
                const size_t lens[] = {4, cursor[i].size(), 5, match.size(), 5, 4};
                w.append(masters[i], 6, argv, lens, (int)i);
                scanning++;
            }
            if (!scanning) return complete;
            std::vector<std::vector<std::string>> found(n);
            std::vector<char> scanned(n, 0);
            w.send(Op::RHT_REMOVE, true, [&](int i, redisReply* r) {
                if (r->type != REDIS_REPLY_ARRAY || r->elements != 2) return;
                cursor[i].assign(r->element[0]->str, r->element[0]->len);
                for (size_t j = 0; j < r->element[1]->elements; j++)
                    found[i].emplace_back(r->element[1]->element[j]->str,
                                          r->element[1]->element[j]->len);
                scanned[i] = 1;
            });
            for (size_t i = 0; i < n; i++) {
                if (done[i]) continue;
                if (!scanned[i]) complete = false;
                if (!scanned[i] || cursor[i] == "0") done[i] = 1;
                for (const std::string& k : found[i]) {
                    const char* a[] = {field.empty() ? "UNLINK" : "HDEL", k.c_str(), field.c_str()};
                    const size_t l[] = {field.empty() ? 6u : 4u, k.size(), field.size()};
                    w.append(masters[i], field.empty() ? 2 : 3, a, l, 0);
                }
            }
            w.send(Op::RHT_REMOVE, false, nullptr);
        }
    }

    std::unique_ptr<Conn> acquire() {
        {
            std::lock_guard<std::mutex> g(pool_mu_);
            if (!idle_.empty()) {
                std::unique_ptr<Conn> c = std::move(idle_.back());
                idle_.pop_back();
                return c;
            }
        }
        return std::make_unique<Conn>(view_, cfg_, breaker_, false);
    }

    void release(std::unique_ptr<Conn> c) {
        std::lock_guard<std::mutex> g(pool_mu_);
        idle_.push_back(std::move(c));
    }

    const RedisRhtConfig cfg_;
    const int my_rank_;
    PeerBreaker breaker_;
    ClusterView view_;
    std::mutex write_mu_;
    Conn writer_;
    std::unordered_map<uint64_t, Held> held_;   // under write_mu_
    uint64_t replayed_ = 0;
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

bool redis_rht_clear(const RedisRhtConfig& cfg) {
    if (parse_seeds(cfg.seeds).empty() || cfg.timeout_ms <= 0) return false;
    return RedisRht(cfg, -1, nullptr).clear();
}

std::string redis_rht_check(const RedisRhtConfig& cfg) {
    std::vector<Endpoint> seeds = parse_seeds(cfg.seeds);
    if (seeds.empty()) return "no host:port in '" + cfg.seeds + "'";
    SigpipeBlock pipe;
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

bool redis_rht_clear(const RedisRhtConfig&) { return false; }

std::string redis_rht_check(const RedisRhtConfig&) {
    return "radixshmem was built without hiredis";
}

} // namespace radixshmem

#endif
