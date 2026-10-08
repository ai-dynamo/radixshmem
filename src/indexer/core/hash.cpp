// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "hash.h"
#include <stdexcept>

namespace radixshmem {

Hasher::Hasher() {
    state_ = XXH64_createState();
    if (!state_) throw std::runtime_error("Failed to create XXH64 state");
    reset();
}

Hasher::~Hasher() {
    if (state_) XXH64_freeState(state_);
}

void Hasher::reset() { XXH64_reset(state_, 0); }

Hasher& Hasher::update(const void* data, size_t size) {
    XXH64_update(state_, data, size);
    return *this;
}

HashType Hasher::digest() { return XXH64_digest(state_); }

void gen_hashes(const int64_t* token_ids, int num_tokens, int tokens_per_block,
                HashType* out_hashes) {
    Hasher hasher;
    int num_blocks = num_tokens / tokens_per_block;
    for (int i = 0; i < num_blocks; i++) {
        hasher.update(token_ids + i * tokens_per_block,
                      tokens_per_block * sizeof(int64_t));
        out_hashes[i] = hasher.digest();
    }
}

} // namespace radixshmem
