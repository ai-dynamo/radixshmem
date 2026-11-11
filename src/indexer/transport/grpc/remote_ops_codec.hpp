// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// PinRecord <-> pb::Pin, shared by the gRPC client and server.
#pragma once

#include "component_types.h"
#include "remote_ops.pb.h"

namespace radixshmem {

inline void pin_to_pb(const PinRecord& in, ::radixshmem::remoteops::v1::Pin* out) {
    out->set_full_start(in.full.start);
    out->set_full_end(in.full.end);
    out->set_swa_start(in.swa.start);
    out->set_swa_end(in.swa.end);
    out->set_mamba_end(in.mamba_end > 0 ? (uint32_t)in.mamba_end : 0);
    out->mutable_tail()->Add(in.tail.begin(), in.tail.end());
}

inline PinRecord pin_from_pb(const ::radixshmem::remoteops::v1::Pin& in) {
    PinRecord out;
    out.full = {in.full_start(), in.full_end()};
    out.swa = {in.swa_start(), in.swa_end()};
    if (in.mamba_end() > 0) out.mamba_end = (int32_t)in.mamba_end();
    out.tail.assign(in.tail().begin(), in.tail().end());
    return out;
}

}  // namespace radixshmem
