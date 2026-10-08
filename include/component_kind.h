// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace radixshmem {

enum class ComponentType : uint8_t { FULL = 0, SWA = 1, MAMBA = 2 };

inline constexpr int NUM_COMPONENT_TYPES = 3;

static constexpr uint32_t COMPONENT_MASK_FULL  = 1u << 0;
static constexpr uint32_t COMPONENT_MASK_SWA   = 1u << 1;
static constexpr uint32_t COMPONENT_MASK_MAMBA = 1u << 2;
static constexpr uint32_t COMPONENT_MASK_ALL =
    COMPONENT_MASK_FULL | COMPONENT_MASK_SWA | COMPONENT_MASK_MAMBA;

inline constexpr uint32_t component_mask_of(ComponentType t) noexcept {
    return 1u << static_cast<uint32_t>(t);
}

inline constexpr bool component_type_valid(ComponentType t) noexcept {
    return static_cast<uint32_t>(t) < static_cast<uint32_t>(NUM_COMPONENT_TYPES);
}

inline constexpr const char* component_type_name(ComponentType t) noexcept {
    switch (t) {
        case ComponentType::FULL:  return "FULL";
        case ComponentType::SWA:   return "SWA";
        case ComponentType::MAMBA: return "MAMBA";
    }
    return "?";
}

inline constexpr bool component_has_swa(uint32_t mask) noexcept {
    return (mask & COMPONENT_MASK_SWA) != 0;
}
inline constexpr bool component_has_mamba(uint32_t mask) noexcept {
    return (mask & COMPONENT_MASK_MAMBA) != 0;
}
inline constexpr bool component_has_aux(uint32_t mask) noexcept {
    return component_has_swa(mask) || component_has_mamba(mask);
}

static constexpr int32_t INVALID_COMPONENT_SLOT = -1;

} // namespace radixshmem
