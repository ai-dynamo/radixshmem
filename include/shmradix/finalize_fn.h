// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <utility>

namespace shmradix {

class FinalizeFn {
public:
    FinalizeFn() = default;
    explicit FinalizeFn(std::function<void()> fn) : fn_(std::move(fn)) {}

    FinalizeFn(FinalizeFn&&) noexcept = default;
    FinalizeFn& operator=(FinalizeFn&&) noexcept = default;

    void operator()() {
        if (used_ || !fn_) return;
        used_ = true;
        fn_();
        fn_ = nullptr;
    }

    bool empty() const { return !fn_ || used_; }
    explicit operator bool() const { return !empty(); }

private:
    std::function<void()> fn_;
    bool used_ = false;
};

} // namespace shmradix
