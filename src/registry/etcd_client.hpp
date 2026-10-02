// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared etcd access surface: base64 codec, PeerEntry/ShardMapMsg wire encoding,
// and a thin RAII wrapper over the Go-wrapper C API. Used by EtcdRegistry
// (server) and RadixClient (client), both keyed under radix/<cluster_id>/.

#pragma once

#include "shmradix/config.h"

#ifdef SHMRADIX_HAS_ETCD

#include "shmradix/cluster_manifest.h"
#include "shmradix/cluster_protocol.h"

#include "libradix_etcd.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace shmradix {
namespace etcd {

// ==================== base64 ====================

inline const char* b64_alphabet() {
    return "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
}

inline std::string b64_encode(const void* data, size_t len) {
    const char* kB64 = b64_alphabet();
    const uint8_t* p = static_cast<const uint8_t*>(data);
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        uint32_t n = p[i] << 16;
        if (i + 1 < len) n |= p[i + 1] << 8;
        if (i + 2 < len) n |= p[i + 2];
        out.push_back(kB64[(n >> 18) & 63]);
        out.push_back(kB64[(n >> 12) & 63]);
        out.push_back(i + 1 < len ? kB64[(n >> 6) & 63] : '=');
        out.push_back(i + 2 < len ? kB64[n & 63] : '=');
    }
    return out;
}

// Decode into buf (up to buf_cap bytes). Returns decoded length, or -1 on bad
// input / overflow. Skips whitespace.
inline int b64_decode(const char* s, size_t slen, void* buf, size_t buf_cap) {
    const char* kB64 = b64_alphabet();
    int8_t rev[256];
    for (int i = 0; i < 256; i++) rev[i] = -1;
    for (int i = 0; i < 64; i++) rev[(uint8_t)kB64[i]] = (int8_t)i;
    uint8_t* out = static_cast<uint8_t*>(buf);
    size_t olen = 0;
    uint32_t acc = 0;
    int bits = 0;
    for (size_t i = 0; i < slen; i++) {
        char c = s[i];
        if (c == '=' || c == '\n' || c == '\r' || c == ' ') continue;
        int8_t v = rev[(uint8_t)c];
        if (v < 0) return -1;
        acc = (acc << 6) | (uint32_t)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            if (olen >= buf_cap) return -1;
            out[olen++] = (uint8_t)((acc >> bits) & 0xFF);
        }
    }
    return (int)olen;
}

// ==================== PeerEntry / ShardMapMsg wire codec ====================

// PeerEntry blob framed with magic + manifest version so a stale/foreign value
// fails loudly rather than corrupting the peer set.
constexpr uint32_t kPeerBlobMagic = 0x52585045;   // "RXPE"

inline std::string encode_peer(const PeerEntry& e) {
    std::string raw;
    raw.resize(8 + sizeof(PeerEntry));
    uint32_t magic = kPeerBlobMagic, ver = CLUSTER_MANIFEST_VERSION;
    memcpy(&raw[0], &magic, 4);
    memcpy(&raw[4], &ver, 4);
    memcpy(&raw[8], &e, sizeof(PeerEntry));
    return b64_encode(raw.data(), raw.size());
}

inline bool decode_peer(const std::string& b64, PeerEntry& out) {
    std::vector<char> raw(8 + sizeof(PeerEntry));
    int n = b64_decode(b64.data(), b64.size(), raw.data(), raw.size());
    if (n != (int)raw.size()) return false;
    uint32_t magic = 0, ver = 0;
    memcpy(&magic, &raw[0], 4);
    memcpy(&ver, &raw[4], 4);
    if (magic != kPeerBlobMagic || ver != CLUSTER_MANIFEST_VERSION) return false;
    memcpy(&out, &raw[8], sizeof(PeerEntry));
    return true;
}

inline std::string encode_shardmap(const ShardMapMsg& sm) {
    return b64_encode(&sm, sizeof(sm));
}
inline bool decode_shardmap(const std::string& b64, ShardMapMsg& out) {
    int n = b64_decode(b64.data(), b64.size(), &out, sizeof(out));
    return n == (int)sizeof(out) &&
           out.magic == CLUSTER_SHARDMAP_MAGIC &&
           out.version == CLUSTER_PROTOCOL_VERSION;
}

// ==================== Client ====================

// Throw with the wrapper's malloc'd errMsg (freed here). rc<0 → error.
inline void check(int rc, char* err, const char* where) {
    if (rc >= 0) return;
    std::string msg = where;
    if (err) { msg += ": "; msg += err; ::free(err); }
    throw std::runtime_error(msg);
}

// The Go wrapper's etcd client is a process-global singleton (init idempotent).
// Refcount init/close so the last user tears it down; earlier users keep it live.
inline std::atomic<int>& global_refcount() {
    static std::atomic<int> rc{0};
    return rc;
}

// Thin RAII over the Go wrapper C surface. One per component; all share the
// process-global connection.
struct Client {
    bool inited = false;

    void init(const std::string& endpoint) {
        char* err = nullptr;
        std::string ep = endpoint;
        check(RadixEtcdInit(const_cast<char*>(ep.c_str()), &err), err,
              "RadixEtcdInit");
        inited = true;
        global_refcount().fetch_add(1, std::memory_order_acq_rel);
    }
    ~Client() {
        if (inited &&
            global_refcount().fetch_sub(1, std::memory_order_acq_rel) == 1) {
            RadixEtcdClose();
        }
    }

    int64_t grant_lease(int64_t ttl) {
        char* err = nullptr;
        long long id = 0;
        check(RadixEtcdGrantLease((long long)ttl, &id, &err), err,
              "RadixEtcdGrantLease");
        return (int64_t)id;
    }

    void keep_alive(int64_t lease) {
        char* err = nullptr;
        check(RadixEtcdKeepAlive((long long)lease, &err), err, "RadixEtcdKeepAlive");
    }

    void keep_alive_stop(int64_t lease) { RadixEtcdKeepAliveStop((long long)lease); }

    void revoke(int64_t lease) {
        char* err = nullptr;
        check(RadixEtcdRevoke((long long)lease, &err), err, "RadixEtcdRevoke");
    }

    void put(const std::string& k, const std::string& v, int64_t lease) {
        char* err = nullptr;
        check(RadixEtcdPut(const_cast<char*>(k.c_str()),
                           const_cast<char*>(v.data()), (int)v.size(),
                           (long long)lease, &err), err, "RadixEtcdPut");
    }

    // returns true if we created it (leader / first-claimer).
    bool create(const std::string& k, const std::string& v, int64_t lease) {
        char* err = nullptr;
        int created = 0;
        check(RadixEtcdCreate(const_cast<char*>(k.c_str()),
                              const_cast<char*>(v.data()), (int)v.size(),
                              (long long)lease, &created, &err), err,
              "RadixEtcdCreate");
        return created != 0;
    }

    // returns false if key absent; else fills out.
    bool get(const std::string& k, std::string& out) {
        char* err = nullptr;
        int found = 0;
        char* val = nullptr;
        int vlen = 0;
        check(RadixEtcdGet(const_cast<char*>(k.c_str()), &found, &val,
                           &vlen, &err), err, "RadixEtcdGet");
        if (!found) return false;
        out.assign(val, (size_t)vlen);
        ::free(val);
        return true;
    }

    // get() plus the key's lease id (0 = no lease).
    bool get_with_lease(const std::string& k, std::string& out, int64_t& lease) {
        char* err = nullptr;
        int found = 0;
        char* val = nullptr;
        int vlen = 0;
        long long id = 0;
        check(RadixEtcdGetWithLease(const_cast<char*>(k.c_str()), &found, &val,
                                    &vlen, &id, &err), err, "RadixEtcdGetWithLease");
        lease = (int64_t)id;
        if (!found) return false;
        out.assign(val, (size_t)vlen);
        ::free(val);
        return true;
    }

    int64_t count(const std::string& prefix) {
        char* err = nullptr;
        long long c = 0;
        check(RadixEtcdCount(const_cast<char*>(prefix.c_str()), &c, &err),
              err, "RadixEtcdCount");
        return (int64_t)c;
    }

    // Key/value pair from a prefix scan; keys returned sorted ascending.
    struct KV { std::string key, value; };

    // out_rev (optional) receives the snapshot revision, for anchoring a watch.
    std::vector<KV> get_prefix(const std::string& prefix,
                               int64_t* out_rev = nullptr) {
        char* err = nullptr;
        int cnt = 0;
        long long rev = 0;
        char** keys = nullptr; int* ksz = nullptr;
        char** vals = nullptr; int* vsz = nullptr;
        check(RadixEtcdGetPrefix(const_cast<char*>(prefix.c_str()), &cnt,
                                 &keys, &ksz, &vals, &vsz, &rev, &err), err,
              "RadixEtcdGetPrefix");
        if (out_rev) *out_rev = (int64_t)rev;
        std::vector<KV> out;
        out.reserve((size_t)cnt);
        for (int i = 0; i < cnt; i++)
            out.push_back({std::string(keys[i], (size_t)ksz[i]),
                           std::string(vals[i], (size_t)vsz[i])});
        for (int i = 0; i < cnt; i++) { ::free(keys[i]); ::free(vals[i]); }
        if (cnt > 0) { ::free(keys); ::free(ksz); ::free(vals); ::free(vsz); }
        return out;
    }

    std::vector<std::string> get_prefix_values(const std::string& prefix) {
        std::vector<std::string> out;
        for (auto& kv : get_prefix(prefix)) out.push_back(std::move(kv.value));
        return out;
    }

    // Exactly one key — not its prefix-siblings. Absent key is not an error.
    void del(const std::string& k) {
        char* err = nullptr;
        check(RadixEtcdDelete(const_cast<char*>(k.c_str()), &err), err,
              "RadixEtcdDelete");
    }

    void delete_prefix(const std::string& prefix) {
        char* err = nullptr;
        check(RadixEtcdDeletePrefix(const_cast<char*>(prefix.c_str()), &err),
              err, "RadixEtcdDeletePrefix");
    }
};

// ==================== Watch ====================

// One decoded etcd event pulled off a Watch's queue.
struct WatchEvent {
    enum Type { kPut = 0, kDelete = 1 };
    Type        type = kPut;
    std::string key;
    std::string value;   // empty for deletes
};

// RAII over a single prefix watch. The Go side runs a goroutine that streams
// events into a buffered channel; next() drains one. Non-copyable, non-movable
// (the id is owned); stop in the dtor. Blocks only the caller's thread — put
// this behind a dedicated watcher thread so the main plane never stalls.
struct Watch {
    int64_t id = -1;

    Watch() = default;
    Watch(const Watch&) = delete;
    Watch& operator=(const Watch&) = delete;
    ~Watch() { stop(); }

    // start_rev == 0 watches from the current revision; > 0 from that revision.
    void start(const std::string& prefix, int64_t start_rev = 0) {
        char* err = nullptr;
        long long wid = 0;
        check(RadixEtcdWatchStart(const_cast<char*>(prefix.c_str()),
                                  (long long)start_rev, &wid, &err),
              err, "RadixEtcdWatchStart");
        id = (int64_t)wid;
    }

    // timeout_ms < 0 blocks, == 0 polls, > 0 waits up to the deadline.
    // Returns true and fills ev when an event arrived, false on timeout/poll-miss.
    bool next(int timeout_ms, WatchEvent& ev) {
        char* err = nullptr;
        int has = 0, ty = 0, ksz = 0, vsz = 0;
        char* k = nullptr; char* v = nullptr;
        check(RadixEtcdWatchNext((long long)id, timeout_ms, &has, &ty,
                                 &k, &ksz, &v, &vsz, &err), err,
              "RadixEtcdWatchNext");
        if (!has) return false;
        ev.type = ty == WatchEvent::kDelete ? WatchEvent::kDelete
                                            : WatchEvent::kPut;
        ev.key.assign(k, (size_t)ksz);
        ev.value.assign(v, (size_t)vsz);
        ::free(k);
        ::free(v);
        return true;
    }

    // Read-and-clear the dropped-event flag: true means the channel overflowed
    // and the consumer should reconcile via a full prefix scan.
    bool overflowed() {
        char* err = nullptr;
        int flag = 0;
        check(RadixEtcdWatchOverflow((long long)id, &flag, &err), err,
              "RadixEtcdWatchOverflow");
        return flag != 0;
    }

    void stop() {
        if (id < 0) return;
        char* err = nullptr;
        check(RadixEtcdWatchStop((long long)id, &err), err,
              "RadixEtcdWatchStop");
        id = -1;
    }
};

// ==================== helpers ====================

// Poll a predicate until it holds or the deadline passes.
template <typename F>
inline bool poll_until(int timeout_sec, F&& pred) {
    auto deadline = std::chrono::steady_clock::now()
                  + std::chrono::seconds(timeout_sec);
    for (;;) {
        if (pred()) return true;
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

// Strip "scheme://" from a registry URI, leaving "host:port".
inline std::string strip_scheme(const std::string& uri) {
    auto pos = uri.find("://");
    return pos == std::string::npos ? uri : uri.substr(pos + 3);
}

// etcd key builders under radix/<cluster_id>/.
struct Keys {
    std::string cluster_id;
    std::string leader()  const { return base() + "leader"; }
    std::string config()  const { return base() + "config"; }
    std::string cluster_lease() const { return base() + "cluster_lease"; }
    std::string peer(const std::string& node)  const { return base() + "peers/" + node; }
    std::string ready(const std::string& node) const { return base() + "ready/" + node; }
    std::string rank_slot(int i) const {
        char b[16]; snprintf(b, sizeof(b), "%04d", i);
        return base() + "rank/slot/" + b;
    }
    std::string shm(const std::string& node) const { return base() + "shm/" + node; }
    // Data-plane slot geometry, published per node (see data_registry.h).
    std::string data(const std::string& node) const { return base() + "data/" + node; }
    std::string data_prefix()      const { return base() + "data/"; }
    std::string peers_prefix()     const { return base() + "peers/"; }
    std::string ready_prefix()     const { return base() + "ready/"; }
    std::string rank_slot_prefix() const { return base() + "rank/slot/"; }
private:
    std::string base() const { return "radix/" + cluster_id + "/"; }
};

} // namespace etcd
} // namespace shmradix

#endif // SHMRADIX_HAS_ETCD
