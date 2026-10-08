// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared etcd fixture for the cross-process cluster ctests.
//
// Each cluster test forks a private, single-node etcd (temp data-dir, per-pid
// ports) before spawning the RadixServer processes, and tears it down on scope
// exit. This is the sole cluster bootstrap backend now that the TCP rendezvous
// path is gone, so the fixture is mandatory — but if no etcd binary is present
// the test SKIPs (exit 77) rather than failing, so environments without etcd
// (e.g. CI not yet provisioned) don't red the whole suite.
//
// etcd binary discovery: $RADIXSHMEM_ETCD_BIN, else "etcd" on $PATH.
// Endpoint handed to servers: "etcd://127.0.0.1:<client_port>".
// cluster_id namespace: "radix-test-<pid>" — unique per test run, so a stale
// key prefix from an aborted run never leaks into a fresh etcd.

#pragma once

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace radixshmem_test {

// Exit code ctest maps to "Skipped" via SKIP_RETURN_CODE (see CMakeLists.txt).
constexpr int kSkipExitCode = 77;

// Locate the etcd binary: env override first, then PATH.
inline std::string find_etcd_binary() {
    if (const char* p = ::getenv("RADIXSHMEM_ETCD_BIN")) {
        if (::access(p, X_OK) == 0) return p;
    }
    // Probe PATH via `command -v` semantics without spawning a shell: scan $PATH.
    const char* path = ::getenv("PATH");
    if (!path) path = "/usr/local/bin:/usr/bin:/bin";
    std::string acc;
    for (const char* c = path; ; ++c) {
        if (*c == ':' || *c == '\0') {
            if (!acc.empty()) {
                std::string cand = acc + "/etcd";
                if (::access(cand.c_str(), X_OK) == 0) return cand;
            }
            acc.clear();
            if (*c == '\0') break;
        } else {
            acc.push_back(*c);
        }
    }
    return {};
}

// Reserve an even client port whose odd neighbor (peer port) is also free, by
// actually bind()ing both. On a busy host (--network host) the ephemeral range
// is heavily occupied, so a blind pid-hash collides ~1/3 of the time; probing
// for a bindable pair removes that flakiness. Returns the even base, or 0.
inline uint16_t find_free_port_pair() {
    for (int attempt = 0; attempt < 200; ++attempt) {
        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return 0;
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port   = 0;   // let the kernel pick a free port
        ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
        if (::bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0) {
            ::close(fd);
            continue;
        }
        socklen_t alen = sizeof(a);
        ::getsockname(fd, reinterpret_cast<sockaddr*>(&a), &alen);
        uint16_t got = ntohs(a.sin_port);
        ::close(fd);
        uint16_t base = static_cast<uint16_t>(got & ~1u);   // even
        if (base < 2000) continue;
        // Verify the (base, base+1) pair is bindable now that the kernel port is freed.
        int f0 = ::socket(AF_INET, SOCK_STREAM, 0);
        int f1 = ::socket(AF_INET, SOCK_STREAM, 0);
        bool ok = (f0 >= 0 && f1 >= 0);
        if (ok) {
            int one = 1;
            ::setsockopt(f0, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
            ::setsockopt(f1, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
            sockaddr_in b0{}, b1{};
            b0.sin_family = b1.sin_family = AF_INET;
            ::inet_pton(AF_INET, "127.0.0.1", &b0.sin_addr);
            ::inet_pton(AF_INET, "127.0.0.1", &b1.sin_addr);
            b0.sin_port = htons(base);
            b1.sin_port = htons(static_cast<uint16_t>(base + 1));
            ok = ::bind(f0, reinterpret_cast<sockaddr*>(&b0), sizeof(b0)) == 0 &&
                 ::bind(f1, reinterpret_cast<sockaddr*>(&b1), sizeof(b1)) == 0;
        }
        if (f0 >= 0) ::close(f0);
        if (f1 >= 0) ::close(f1);
        if (ok) return base;
    }
    return 0;
}

inline bool tcp_reachable(uint16_t port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return false;
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port   = htons(port);
    ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
    bool ok = ::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) == 0;
    ::close(fd);
    return ok;
}

// RAII: fork+exec a private etcd, wait until its client port answers, and on
// destruction SIGTERM the process + rm -rf the temp data-dir. usable()==false
// means no etcd binary was found — the caller should exit(kSkipExitCode).
class EtcdHarness {
public:
    EtcdHarness() {
        bin_ = find_etcd_binary();
        if (bin_.empty()) return;   // usable_ stays false → SKIP

        const int pid = static_cast<int>(::getpid());
        // Two distinct ports per instance. Probe for a bindable (even, even+1)
        // pair rather than a blind pid-hash — the host ephemeral range is busy
        // (--network host) and a fixed hash collides ~1/3 of the time → SKIP.
        uint16_t base = find_free_port_pair();
        if (base == 0) base = static_cast<uint16_t>(20000 + (pid % 10000) * 2);
        client_port_ = base;
        peer_port_   = static_cast<uint16_t>(base + 1);
        data_dir_    = "/tmp/radix_etcd_test_" + std::to_string(pid) + ".d";

        // Fresh data-dir.
        rm_rf(data_dir_);

        const std::string curl  = "http://127.0.0.1:" + std::to_string(client_port_);
        const std::string purl  = "http://127.0.0.1:" + std::to_string(peer_port_);

        pid_ = ::fork();
        if (pid_ < 0) return;
        if (pid_ == 0) {
            // Child: silence etcd's chatter, exec the server.
            int devnull = ::open("/dev/null", O_WRONLY);
            if (devnull >= 0) { ::dup2(devnull, 1); ::dup2(devnull, 2); }
            ::execl(bin_.c_str(), bin_.c_str(),
                    "--name", "radix-test",
                    "--data-dir", data_dir_.c_str(),
                    "--listen-client-urls", curl.c_str(),
                    "--advertise-client-urls", curl.c_str(),
                    "--listen-peer-urls", purl.c_str(),
                    "--initial-advertise-peer-urls", purl.c_str(),
                    "--initial-cluster", ("radix-test=" + purl).c_str(),
                    "--initial-cluster-state", "new",
                    "--log-level", "error",
                    (char*)nullptr);
            _exit(127);   // exec failed
        }

        // Parent: wait for the client port to accept connections.
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        while (std::chrono::steady_clock::now() < deadline) {
            if (tcp_reachable(client_port_)) { usable_ = true; break; }
            // Bail early if etcd died on exec.
            int st = 0;
            if (::waitpid(pid_, &st, WNOHANG) == pid_) { pid_ = -1; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (!usable_) {
            fprintf(stderr, "[etcd-harness] etcd at %s never came up\n",
                    curl.c_str());
            stop();
        }
    }

    ~EtcdHarness() { stop(); }

    EtcdHarness(const EtcdHarness&) = delete;
    EtcdHarness& operator=(const EtcdHarness&) = delete;

    bool usable() const { return usable_; }

    // "etcd://host:port" endpoint for RadixServerConfig::registry.
    std::string registry() const {
        return "etcd://127.0.0.1:" + std::to_string(client_port_);
    }
    // Per-run key namespace.
    std::string cluster_id() const {
        return "radix-test-" + std::to_string(static_cast<int>(::getpid()));
    }

private:
    void stop() {
        if (pid_ > 0) {
            ::kill(pid_, SIGTERM);
            ::waitpid(pid_, nullptr, 0);
            pid_ = -1;
        }
        if (!data_dir_.empty()) rm_rf(data_dir_);
    }

    static void rm_rf(const std::string& dir) {
        // fork+exec rm -rf; the dir is our own temp path.
        pid_t p = ::fork();
        if (p == 0) {
            ::execl("/bin/rm", "rm", "-rf", dir.c_str(), (char*)nullptr);
            _exit(127);
        }
        if (p > 0) ::waitpid(p, nullptr, 0);
    }

    std::string bin_;
    std::string data_dir_;
    uint16_t    client_port_ = 0;
    uint16_t    peer_port_   = 0;
    pid_t       pid_ = -1;
    bool        usable_ = false;
};

} // namespace radixshmem_test
