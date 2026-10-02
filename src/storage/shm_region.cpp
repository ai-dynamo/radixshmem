// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "shm_region.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23   // Linux 5.14+
#endif

namespace shmradix {

namespace {

constexpr uint64_t kPopulateChunkBytes = 256ull << 20;

// Not mmap(MAP_POPULATE): that holds the process mmap_lock for the whole
// populate, and the server's XRC listener (libmlx5 maps QP buffers per
// handshake) stalls behind it. Chunks bound the stall; EINVAL = no
// MADV_POPULATE_WRITE, touch a byte per page.
void populate(void* base, uint64_t bytes, uint64_t page) {
    char* p = static_cast<char*>(base);
    for (uint64_t off = 0; off < bytes; off += kPopulateChunkBytes) {
        const uint64_t len = std::min(kPopulateChunkBytes, bytes - off);
        if (::madvise(p + off, len, MADV_POPULATE_WRITE) == 0 || errno != EINVAL)
            continue;
        for (uint64_t i = off; i < off + len; i += page) {
            volatile char* q = p + i;
            *q = *q;
        }
    }
}

// POSIX shm names carry a leading '/'; a hugetlbfs file path must not.
std::string hugepage_file(const std::string& dir, const std::string& name) {
    const std::string base = (!name.empty() && name[0] == '/') ? name.substr(1) : name;
    return dir + "/" + base;
}

// Both backings, always: creating one kind over a stale object of the other
// kind leaves an attacher that resolves the other way silently mapping old data.
void unlink_both(const std::string& name, const std::string& hugepage_path) {
    const std::string dir =
        hugepage_path.empty() ? SHM_REGION_DEFAULT_HUGEPAGE_DIR : hugepage_path;
    ::unlink(hugepage_file(dir, name).c_str());
    ::shm_unlink(name.c_str());
}

uint64_t file_size(int fd) {
    struct stat st {};
    if (::fstat(fd, &st) < 0) return 0;
    return (uint64_t)st.st_size;
}

} // namespace

ShmRegion::~ShmRegion() { release(); }

ShmRegion::ShmRegion(ShmRegion&& o) noexcept
    : base_(o.base_), bytes_(o.bytes_), fd_(o.fd_), is_owner_(o.is_owner_),
      is_hugepage_(o.is_hugepage_), name_(std::move(o.name_)),
      path_(std::move(o.path_)) {
    o.base_ = nullptr;
    o.bytes_ = 0;
    o.fd_ = -1;
    o.is_owner_ = false;
    o.is_hugepage_ = false;
}

ShmRegion& ShmRegion::operator=(ShmRegion&& o) noexcept {
    if (this != &o) {
        release();
        base_ = o.base_;
        bytes_ = o.bytes_;
        fd_ = o.fd_;
        is_owner_ = o.is_owner_;
        is_hugepage_ = o.is_hugepage_;
        name_ = std::move(o.name_);
        path_ = std::move(o.path_);
        o.base_ = nullptr;
        o.bytes_ = 0;
        o.fd_ = -1;
        o.is_owner_ = false;
        o.is_hugepage_ = false;
    }
    return *this;
}

void ShmRegion::release() {
    if (base_) ::munmap(base_, bytes_);
    if (fd_ >= 0) ::close(fd_);
    if (is_owner_) {
        if (is_hugepage_) ::unlink(path_.c_str());
        else              ::shm_unlink(name_.c_str());
    }
    base_ = nullptr;
    bytes_ = 0;
    fd_ = -1;
    is_owner_ = false;
    is_hugepage_ = false;
}

ShmRegion ShmRegion::create(const std::string& name, uint64_t bytes,
                            const std::string& hugepage_path, bool prefault) {
    if (name.empty() || bytes == 0)
        throw std::runtime_error("ShmRegion::create: empty name or zero size");

    ShmRegion r;
    r.name_ = name;
    r.is_owner_ = true;
    const int map_flags = MAP_SHARED;

    unlink_both(name, hugepage_path);

    uint64_t hp_bytes = 0;
    if (!hugepage_path.empty() &&
        region_align_up_checked(bytes, SHM_REGION_HUGEPAGE_BYTES, &hp_bytes)) {
        const std::string path = hugepage_file(hugepage_path, name);
        int fd = ::open(path.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd >= 0) {
            if (::ftruncate(fd, (off_t)hp_bytes) == 0) {
                void* p = ::mmap(nullptr, hp_bytes, PROT_READ | PROT_WRITE,
                                 map_flags, fd, 0);
                if (p != MAP_FAILED) {
                    if (prefault) populate(p, hp_bytes, SHM_REGION_HUGEPAGE_BYTES);
                    r.fd_ = fd;
                    r.base_ = p;
                    r.bytes_ = hp_bytes;
                    r.is_hugepage_ = true;
                    r.path_ = path;
                    fprintf(stderr, "[shmradix] ShmRegion: hugepages %s (%.1f MB)\n",
                            path.c_str(), hp_bytes / (1024.0 * 1024.0));
                    return r;
                }
            }
            ::close(fd);
            ::unlink(path.c_str());
        }
        fprintf(stderr,
                "[shmradix] ShmRegion: hugepage alloc failed for %s, falling back to shm_open\n",
                name.c_str());
    }

    int fd = ::shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd < 0)
        throw std::runtime_error("ShmRegion::create: shm_open failed: " + name);
    if (::ftruncate(fd, (off_t)bytes) < 0) {
        ::close(fd);
        ::shm_unlink(name.c_str());
        throw std::runtime_error("ShmRegion::create: ftruncate failed: " + name);
    }
    void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, map_flags, fd, 0);
    if (p == MAP_FAILED) {
        ::close(fd);
        ::shm_unlink(name.c_str());
        throw std::runtime_error("ShmRegion::create: mmap failed: " + name);
    }
    if (prefault) populate(p, bytes, (uint64_t)::sysconf(_SC_PAGESIZE));
    r.fd_ = fd;
    r.base_ = p;
    r.bytes_ = bytes;
    return r;
}

ShmRegion ShmRegion::attach(const std::string& name, uint64_t min_bytes,
                            const std::string& hugepage_path) {
    ShmRegion r;
    r.name_ = name;

    int fd = -1;
    if (!hugepage_path.empty()) {
        const std::string path = hugepage_file(hugepage_path, name);
        fd = ::open(path.c_str(), O_RDWR);
        if (fd >= 0) {
            r.is_hugepage_ = true;
            r.path_ = path;
        }
    }
    if (fd < 0) fd = ::shm_open(name.c_str(), O_RDWR, 0666);
    if (fd < 0)
        throw std::runtime_error("ShmRegion::attach: no such region: " + name);

    const uint64_t size = file_size(fd);
    if (size == 0 || (min_bytes > 0 && size < min_bytes)) {
        ::close(fd);
        throw std::runtime_error("ShmRegion::attach: region too small: " + name);
    }
    // No MAP_POPULATE: an attacher usually touches a small subset, and MR
    // registration faults in whatever the transfer plane needs anyway.
    void* p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
        ::close(fd);
        throw std::runtime_error("ShmRegion::attach: mmap failed: " + name);
    }
    r.fd_ = fd;
    r.base_ = p;
    r.bytes_ = size;
    return r;
}

void ShmRegion::unlink(const std::string& name, const std::string& hugepage_path) {
    unlink_both(name, hugepage_path);
}

} // namespace shmradix
