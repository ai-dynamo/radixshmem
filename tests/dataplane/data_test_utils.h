// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Tiny self-contained runner for the data-plane tests. Mirrors
// tests/indexer/test_utils.h, minus every radixshmem indexer include — these
// binaries link radixshmem_data ONLY, which is what proves the data plane carries
// no dependency on the indexer.

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>

#include <sys/wait.h>
#include <unistd.h>

namespace radixshmem_test {
inline int& tests_run()    { static int v = 0; return v; }
inline int& tests_passed() { static int v = 0; return v; }
} // namespace radixshmem_test

#define TEST_BEGIN(name)                                                       \
    do {                                                                       \
        radixshmem_test::tests_run()++;                                          \
        printf("  [%d] %-58s",                                                 \
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

// Run `body` in a forked child and assert it exited cleanly. An assert() firing
// inside the child aborts it, which surfaces here as a non-zero status.
inline void run_child(const std::function<void()>& body) {
    pid_t pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        body();
        _exit(0);
    }
    int status = 0;
    assert(waitpid(pid, &status, 0) == pid);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

// Fork `body` into a child and return its pid without waiting. Needed when two
// children must run concurrently and talk to each other, which run_child's
// wait-immediately shape cannot express.
inline pid_t spawn_child(const std::function<void()>& body) {
    pid_t pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        body();
        _exit(0);
    }
    return pid;
}

// True when the child exited cleanly.
inline bool reap_child(pid_t pid) {
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return false;
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

// Run `body` in a child and report its exit status instead of asserting on it.
// Used to probe for capabilities (RDMA present? engine initialises?) without
// dragging the probe's state back into the parent.
inline bool child_succeeds(const std::function<void()>& body) {
    return reap_child(spawn_child(body));
}

#define TEST_SKIP(why)                                                         \
    do {                                                                       \
        radixshmem_test::tests_passed()++;                                       \
        printf("SKIP (%s)\n", why);                                            \
    } while (0)

// True when `fn` throws anything.
inline bool throws(const std::function<void()>& fn) {
    try { fn(); } catch (...) { return true; }
    return false;
}
