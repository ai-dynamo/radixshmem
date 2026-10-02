// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Verifies that XrcSendQpSharedState (pthread_mutex with PROCESS_SHARED+ROBUST
// and atomic<int> counters) actually synchronizes correctly across processes
// when placed in shm.
//
// Topology:
//   parent: shm_open + ftruncate + mmap; init_shared() on the region;
//           fork() N children.
//   each child: munmap (drop inherited mapping), shm_open + mmap again
//               (proves a *fresh* process attach works), then loop M times:
//                  lock(mu); ++inflight_unsignaled; unlock(mu);   // mutexed
//                  outstanding_signaled.fetch_add(1);             // atomic
//   parent: waitpid all; verify both counters == N*M.

#include "rdma_transport.hpp"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

using shmradix::XrcSendQpSharedState;

constexpr const char* SHM_NAME = "/test_xrc_shared_state";
constexpr int  N_PROCS = 8;
constexpr int  ITERS   = 50000;

static XrcSendQpSharedState* open_state_in_shm(bool creator) {
    int fd = shm_open(SHM_NAME,
                      creator ? (O_CREAT | O_EXCL | O_RDWR) : O_RDWR,
                      0600);
    if (fd < 0) {
        fprintf(stderr, "shm_open(%s, creator=%d) failed: %s\n",
                SHM_NAME, creator, strerror(errno));
        _exit(1);
    }
    size_t sz = sizeof(XrcSendQpSharedState);
    if (creator && ftruncate(fd, sz) != 0) {
        fprintf(stderr, "ftruncate failed\n"); _exit(1);
    }
    void* p = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); _exit(1); }
    return reinterpret_cast<XrcSendQpSharedState*>(p);
}

static void mutex_lock_recover(pthread_mutex_t* m) {
    int rc = pthread_mutex_lock(m);
    if (rc == EOWNERDEAD) {
        pthread_mutex_consistent(m);
    } else if (rc != 0) {
        fprintf(stderr, "pthread_mutex_lock failed: %d\n", rc);
        _exit(2);
    }
}

int main() {
    shm_unlink(SHM_NAME);   // best effort

    XrcSendQpSharedState* parent_state = open_state_in_shm(/*creator=*/true);
    XrcSendQpSharedState::init_shared(parent_state);

    pid_t kids[N_PROCS];
    for (int i = 0; i < N_PROCS; i++) {
        pid_t pid = fork();
        if (pid < 0) { perror("fork"); return 1; }
        if (pid == 0) {
            // Drop the inherited mapping and reopen the shm from scratch.
            munmap(parent_state, sizeof(XrcSendQpSharedState));
            XrcSendQpSharedState* st = open_state_in_shm(false);

            for (int j = 0; j < ITERS; j++) {
                mutex_lock_recover(&st->mu);
                int v = st->inflight_unsignaled.load(std::memory_order_relaxed);
                st->inflight_unsignaled.store(v + 1, std::memory_order_relaxed);
                pthread_mutex_unlock(&st->mu);

                st->outstanding_signaled.fetch_add(1, std::memory_order_relaxed);
            }
            munmap(st, sizeof(XrcSendQpSharedState));
            _exit(0);
        }
        kids[i] = pid;
    }

    int fails = 0;
    for (int i = 0; i < N_PROCS; i++) {
        int status = 0;
        if (waitpid(kids[i], &status, 0) < 0) { perror("waitpid"); fails++; continue; }
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr, "child %d exit status=%d\n", i, status);
            fails++;
        }
    }

    int expected = N_PROCS * ITERS;
    int inflight = parent_state->inflight_unsignaled.load(std::memory_order_relaxed);
    int outst    = parent_state->outstanding_signaled.load(std::memory_order_relaxed);

    printf("expected           = %d\n", expected);
    printf("inflight (mutexed) = %d  %s\n",
           inflight, inflight == expected ? "OK" : "FAIL");
    printf("outst    (atomic)  = %d  %s\n",
           outst,    outst    == expected ? "OK" : "FAIL");

    XrcSendQpSharedState::destroy(parent_state);
    munmap(parent_state, sizeof(XrcSendQpSharedState));
    shm_unlink(SHM_NAME);

    bool ok = (fails == 0) && (inflight == expected) && (outst == expected);
    return ok ? 0 : 1;
}
