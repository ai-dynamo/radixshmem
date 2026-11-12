// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// PinRecord <-> pb::Pin and XrcConnInfo <-> pb::XrcEnd, shared by the gRPC client and server.
#pragma once

#include "bilateral_transport.h"
#include "component_types.h"
#include "remote_ops.pb.h"

#include <cstring>

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

inline void xrc_to_pb(const XrcConnInfo& in, ::radixshmem::remoteops::v1::XrcEnd* out) {
    out->set_rank(in.rank);
    out->set_qp_num(in.qp_num);
    out->set_srq_num(in.srq_num);
    out->set_link_layer(in.link_layer);
    out->set_lid(in.lid);
    out->set_gid_index(in.gid_index);
    out->set_gid(in.gid, sizeof(in.gid));
}

// false if the GID is not 16 bytes.
inline bool xrc_from_pb(const ::radixshmem::remoteops::v1::XrcEnd& in, XrcConnInfo& out) {
    if (in.gid().size() != sizeof(out.gid)) return false;
    out.rank = in.rank();
    out.qp_num = in.qp_num();
    out.srq_num = in.srq_num();
    out.link_layer = in.link_layer();
    out.lid = in.lid();
    out.gid_index = in.gid_index();
    std::memcpy(out.gid, in.gid().data(), sizeof(out.gid));
    return true;
}

}  // namespace radixshmem
