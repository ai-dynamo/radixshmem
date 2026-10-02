// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// ShmRegion — a named shared-memory mapping: hugepage file first (a file under
// hugepage_path), falling back to POSIX shm_open. The owner creates and unlinks;
// attachers only map. Same strategy the indexer's ShmRadixTree uses, kept as a
// standalone unit here so the data plane never pulls in an indexer header.

#pragma once

#include <cstdint>
#include <string>

namespace shmradix {

inline constexpr uint64_t SHM_REGION_HUGEPAGE_BYTES = 2ULL * 1024 * 1024;

// Used when a caller opts out of hugepages: cleanup still has to reach a file
// left behind by a run that opted in.
inline constexpr const char* SHM_REGION_DEFAULT_HUGEPAGE_DIR = "/dev/hugepages";

inline uint64_t region_align_up(uint64_t v, uint64_t a) {
    return (v + a - 1) & ~(a - 1);
}

// Same, but false instead of a wrapped-to-zero result. `a` must be a power of two.
inline bool region_align_up_checked(uint64_t v, uint64_t a, uint64_t* out) {
    if (v > UINT64_MAX - (a - 1)) return false;
    *out = (v + a - 1) & ~(a - 1);
    return true;
}

class ShmRegion {
public:
    ShmRegion() = default;
    ~ShmRegion();

    ShmRegion(ShmRegion&&) noexcept;
    ShmRegion& operator=(ShmRegion&&) noexcept;
    ShmRegion(const ShmRegion&) = delete;
    ShmRegion& operator=(const ShmRegion&) = delete;

    // Owner. `bytes` is rounded up to the hugepage size when hugepages are used.
    // Throws std::runtime_error if neither backing store works.
    static ShmRegion create(const std::string& name, uint64_t bytes,
                            const std::string& hugepage_path, bool prefault);

    // Attacher. Size comes from the backing object; `min_bytes` > 0 asserts it
    // is at least that large. Throws when the object is absent or too small.
    static ShmRegion attach(const std::string& name, uint64_t min_bytes,
                            const std::string& hugepage_path);

    // Best-effort removal of both possible backings. Never throws.
    static void unlink(const std::string& name, const std::string& hugepage_path);

    void*    base()  const { return base_; }
    uint64_t bytes() const { return bytes_; }
    bool     is_owner() const { return is_owner_; }
    bool     is_hugepage() const { return is_hugepage_; }
    const std::string& name() const { return name_; }

private:
    void release();

    void*       base_        = nullptr;
    uint64_t    bytes_       = 0;
    int         fd_          = -1;
    bool        is_owner_    = false;
    bool        is_hugepage_ = false;
    std::string name_;   // logical name, as passed in
    std::string path_;   // hugepage file path, empty when backed by shm_open
};

} // namespace shmradix
