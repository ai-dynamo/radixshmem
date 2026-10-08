// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "health_check.h"
#include "radix_client.h"
#include "radix_server.h"
#include "radix_tree.h"
#include "types.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <signal.h>
#include <string>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

// ==================== Tiny test runner ====================
//
// Each test_*.cpp file should:
//   - include this header
//   - define static void tests via TEST_BEGIN(...) ... TEST_PASS()
//   - in main(), call each test and then `return tests_report();`
//
// All counters and the runner are in this header so every binary is
// self-contained — no extra shared.cpp to link.

namespace radixshmem_test {
inline int& tests_run()    { static int v = 0; return v; }
inline int& tests_passed() { static int v = 0; return v; }
} // namespace radixshmem_test

#define TEST_BEGIN(name)                                                       \
    do {                                                                       \
        radixshmem_test::tests_run()++;                                          \
        printf("  [%d] %-50s",                                                 \
               radixshmem_test::tests_run(), name);                              \
        fflush(stdout);                                                        \
    } while (0)

#define TEST_PASS()                                                            \
    do {                                                                       \
        radixshmem_test::tests_passed()++;                                       \
        printf("PASS\n");                                                      \
    } while (0)

inline int tests_report() {
    printf("\n%d/%d tests passed.\n",
           radixshmem_test::tests_passed(), radixshmem_test::tests_run());
    return (radixshmem_test::tests_passed() == radixshmem_test::tests_run()) ? 0 : 1;
}

inline radixshmem::ShmConfig default_cfg() {
    return radixshmem::ShmConfig{1000, 10000};
}

// Force LRU eviction of at least `n` slots, restoring free count when done.
// Returns mempool_free() after the eviction (matches the old reserve() semantics).
inline int force_evict(radixshmem::RadixClient& client, int n) {
    std::vector<int32_t> tmp;
    client.allocate_slots(n, tmp);
    int got = (int)tmp.size();
    if (got > 0) client.recycle_slots(tmp.data(), got);
    return (int)client.mempool_free();
}

// End-of-test leak audit (no locks taken): prints the report and asserts when
// any HealthFlag outside `allowed` is set. Tests that deliberately leave pins
// or slots behind pass the expected bits in `allowed`.
// RADIXSHMEM_HEALTH_SOFT=1 downgrades the assert to a report (lists every leaking
// test in one run instead of aborting at the first).
inline void expect_health_report(const radixshmem::HealthReport& r, uint32_t allowed) {
    if (!(r.flags & ~allowed)) return;
    fprintf(stderr, "\nhealth check failed (flags=0x%x allowed=0x%x):\n%s",
            r.flags, allowed, r.describe().c_str());
    if (getenv("RADIXSHMEM_HEALTH_SOFT")) return;
    assert((r.flags & ~allowed) == 0);
}

inline void expect_healthy(const radixshmem::RadixClient& client,
                           uint32_t allowed = 0) {
    radixshmem::HealthReport r;
    client.check_health(&r);
    expect_health_report(r, allowed);
}

inline void expect_healthy(const radixshmem::RadixShmemTree& tree,
                           uint32_t allowed = 0) {
    expect_health_report(radixshmem::check_health(tree), allowed);
}

// Cluster tests: a server process audits its tree on SIGTERM and exits with
// kHealthExitCode when anything leaked; the parent reaps with reap_servers().
constexpr int kHealthExitCode = 50;

inline bool server_healthy(const radixshmem::RadixServer& srv, const char* tag) {
    radixshmem::HealthReport r;
    srv.check_health(&r);
    fprintf(stderr, "[server %s] check_health: %s", tag, r.describe().c_str());
    return r.ok();
}

// waitpid() every live pid; returns how many exited with kHealthExitCode.
inline int reap_servers(std::vector<pid_t>& pids) {
    int unhealthy = 0;
    for (pid_t& p : pids) {
        if (p <= 0) continue;
        int st = 0;
        if (::waitpid(p, &st, 0) == p && WIFEXITED(st) &&
            WEXITSTATUS(st) == kHealthExitCode)
            unhealthy++;
        p = -1;
    }
    return unhealthy;
}

inline int32_t remote_src(const radixshmem::ComponentQueryResult& r) {
    return r.full_fragments.empty() ? -1 : r.full_fragments.back().source_rank;
}

inline const std::vector<int32_t>&
hit_slots(const radixshmem::ComponentQueryResult& r) {
    static const std::vector<int32_t> kEmpty;
    return r.full_fragments.empty() ? kEmpty : r.full_fragments[0].slot_ids;
}

inline int query_hit(radixshmem::RadixClient& client, const uint64_t* h, int n,
                     bool local_only = false, bool update_meta = false) {
    return client.query(h, n, radixshmem::COMPONENT_MASK_FULL, local_only,
                        /*lock=*/false, update_meta).common_hit;
}

// Allocate the shortfall then insert. Peeks the matched prefix (update_meta=
// false, no LRU touch), allocates only n - matched, and binds those slots to
// hashes[matched:] via start=matched. insert auto-recycles any unused slots
// (leading redundant + unlanded suffix), so no manual tail recycle here.
inline radixshmem::InsertResult
insert_auto(radixshmem::RadixClient& client, const uint64_t* h, int n) {
    int matched = query_hit(client, h, n);
    int need = n - matched;
    std::vector<int32_t> slots;
    if (need > 0) client.allocate_slots(need, slots);
    return client.insert(h, n, slots.data(), (int)slots.size(),
                         /*start=*/matched, /*auto_recycle=*/true);
}

// Pin [0, n) of `h` via query(lock=true); call the returned handle to release.
// Dropping the handle without calling it leaves the pin in place.
inline radixshmem::FinalizeFn pin_path(radixshmem::RadixClient& client,
                                     const uint64_t* h, int n) {
    auto r = client.query(h, n, radixshmem::COMPONENT_MASK_FULL,
                          /*local_only=*/false, /*lock=*/true,
                          /*update_meta=*/false);
    return std::move(r.finalize);
}

inline radixshmem::InsertResult
insert_pinned(radixshmem::RadixClient& client, const uint64_t* h, int n) {
    // Re-peek matched so a same-thread rejection (matched < peeked) is visible.
    int matched = query_hit(client, h, n);
    int need = n - matched;
    std::vector<int32_t> slots;
    if (need > 0) client.allocate_slots(need, slots);
    auto r = client.insert(h, n, slots.data(), (int)slots.size(),
                           /*start=*/matched, /*auto_recycle=*/true);
    // Reject leaves every slot unused and nothing on the path — don't pin.
    bool rejected = need > 0 && (int)r.unused_slots.size() == (int)slots.size();
    if (!rejected) (void)pin_path(client, h, n);   // held, never released
    return r;
}

class ServerLauncher {
public:
    ServerLauncher(const std::string& name,
                   const radixshmem::ShmConfig& config = radixshmem::ShmConfig{})
        : name_(name), pid_(-1) {
        std::string bin = server_bin_path();

        char nodes_str[32];
        snprintf(nodes_str, sizeof(nodes_str), "%u", config.max_nodes);

        char blocks_str[32];
        snprintf(blocks_str, sizeof(blocks_str), "%u", config.max_blocks);

        char evict_str[32];
        snprintf(evict_str, sizeof(evict_str), "%.4f", config.background_evict_ratio);

        char cs_str[32];
        snprintf(cs_str, sizeof(cs_str), "%d", config.register_chunk_tokens);

        char dpr_str[32];
        snprintf(dpr_str, sizeof(dpr_str), "%.4f", config.data_pool_ratio);

        shm_unlink(name.c_str());

        pid_ = fork();
        if (pid_ == 0) {
            freopen("/dev/null", "w", stdout);
            std::string logfile = "/tmp/radix_server_" + name.substr(1) + ".log";
            freopen(logfile.c_str(), "w", stderr);
            if (config.max_blocks > 0) {
                execl(bin.c_str(), "radix_server",
                      "--name", name.c_str(),
                      "--max-nodes", nodes_str,
                      "--max-blocks", blocks_str,
                      "--background-evict-ratio", evict_str,
                      "--register-chunk-tokens", cs_str,
                      "--data-pool-ratio", dpr_str,
                      "--interval", "86400",
                      nullptr);
            } else {
                execl(bin.c_str(), "radix_server",
                      "--name", name.c_str(),
                      "--max-nodes", nodes_str,
                      "--background-evict-ratio", evict_str,
                      "--register-chunk-tokens", cs_str,
                      "--interval", "86400",
                      nullptr);
            }
            _exit(127);
        }
        assert(pid_ > 0);
        wait_for_ready();
    }

    ~ServerLauncher() {
        if (pid_ > 0) {
            kill(pid_, SIGTERM);
            int status;
            waitpid(pid_, &status, 0);
            pid_ = -1;
        }
    }

    ServerLauncher(const ServerLauncher&) = delete;
    ServerLauncher& operator=(const ServerLauncher&) = delete;

    pid_t pid() const { return pid_; }

    // Non-blocking check: returns true if server is still running.
    // If dead, fills exit_status/signal_num with cause.
    bool check_alive(int& exit_status, int& signal_num) {
        if (pid_ <= 0) { exit_status = -1; signal_num = 0; return false; }
        int status;
        pid_t r = waitpid(pid_, &status, WNOHANG);
        if (r == 0) return true;  // still running
        if (r == pid_) {
            if (WIFEXITED(status)) {
                exit_status = WEXITSTATUS(status);
                signal_num = 0;
            } else if (WIFSIGNALED(status)) {
                exit_status = -1;
                signal_num = WTERMSIG(status);
            } else {
                exit_status = -1;
                signal_num = 0;
            }
            pid_ = -1;
            return false;
        }
        return true;  // waitpid error, assume alive
    }

private:
    void wait_for_ready() {
        for (int i = 0; i < 300; i++) {
            int status;
            pid_t r = waitpid(pid_, &status, WNOHANG);
            if (r == pid_) {
                fprintf(stderr,
                        "ServerLauncher: radix_server exited early (status %d)\n",
                        WIFEXITED(status) ? WEXITSTATUS(status) : -1);
                pid_ = -1;
                abort();
            }
            try {
                auto tree = radixshmem::RadixShmemTree::attach(name_);
                tree.detach();
                return;
            } catch (...) {}
            usleep(10000);
        }
        fprintf(stderr, "ServerLauncher: timeout waiting for '%s'\n",
                name_.c_str());
        kill(pid_, SIGTERM);
        waitpid(pid_, nullptr, 0);
        pid_ = -1;
        abort();
    }

    static std::string server_bin_path() {
        char buf[4096];
        ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
        if (len > 0) {
            buf[len] = '\0';
            std::string exe(buf);
            auto pos = exe.rfind('/');
            if (pos != std::string::npos)
                return exe.substr(0, pos) + "/radix_server";
        }
        return "./bin/radix_server";
    }

    std::string name_;
    pid_t pid_;
};
