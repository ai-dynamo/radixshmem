// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <xxhash.h>

namespace radixshmem {

using HashType = uint64_t;

class Hasher {
public:
    Hasher();
    ~Hasher();

    Hasher(const Hasher&) = delete;
    Hasher& operator=(const Hasher&) = delete;

    void reset();
    Hasher& update(const void* data, size_t size);
    HashType digest();

private:
    XXH64_state_t* state_;
};

void gen_hashes(const int64_t* token_ids, int num_tokens, int tokens_per_block,
                HashType* out_hashes);

} // namespace radixshmem
