# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from radixshmem import Geometry
from radixshmem import _geometry as G
from radixshmem.server import ServerConfig


def cfg(**kw):
    kw.setdefault("name", "/geo")
    return ServerConfig(**kw)


def test_data_mode_plans_counts_from_budget_and_ratios():
    spec = Geometry(block_size=4, full_slot_bytes=4096, swa_slot_bytes=1024, swa_window_blocks=2,
                    mamba_slot_bytes=512).to_dict()
    g = G.plan(spec, cfg(data_bytes=1 << 20, swa_ratio=1 / 16, mamba_ratio=1 / 128, slot_align=64))
    p = g["pools"]
    assert p["swa"]["num_slots"] == 64                # 1/16 of 1 MiB in 1 KiB slots
    assert p["mamba"]["num_slots"] == 16              # 1/128 of 1 MiB in 512 B slots
    assert p["full"]["num_slots"] == ((1 << 20) - 64 * 1024 - 16 * 512) // 4096
    assert p["swa"]["window_blocks"] == 2 and p["full"]["slot_bytes"] == 4096
    assert G.geometry_bytes(g) <= 1 << 20
    assert g["max_nodes"] == 2 * p["full"]["num_slots"] + 1024
    assert g["register_chunk_tokens"] == 4096 and g["slot_align"] == 64
    assert g["data_name"] == "/geo_data" and g["index_name"] == "/geo"
    assert G.describe(g) == "FULL 238 slots x 4096 B, SWA 64 slots x 1024 B (window 2), MAMBA 16 slots x 512 B"


def test_data_mode_budgets_with_the_aligned_stride():
    g = G.plan(Geometry(block_size=1, full_slot_bytes=5000).to_dict(), cfg(data_bytes=1 << 20))
    assert g["pools"]["full"]["num_slots"] == (1 << 20) // 8192     # default slot_align 4096
    assert g["pools"]["full"]["slot_bytes"] == 5000                  # raw bytes stay; the store rounds
    g = G.plan(Geometry(block_size=1, full_slot_bytes=5000, slot_align=512, register_chunk_tokens=7).to_dict(),
               cfg(data_bytes=1 << 20))
    assert g["pools"]["full"]["num_slots"] == (1 << 20) // 5120 and g["slot_align"] == 512
    assert g["register_chunk_tokens"] == 7
    assert G.shm_config(g, background_evict_ratio=0.0, data_pool_ratio=4.0).effective_register_chunk_size() == 7


def test_data_mode_refusals():
    c = cfg(data_bytes=1 << 20, swa_ratio=0.1)
    with pytest.raises(ValueError, match="num_slots must not be given"):
        G.plan(Geometry(block_size=4, full_slot_bytes=4096, full_slots=10).to_dict(), c)
    with pytest.raises(ValueError, match="slot_bytes > 0"):
        G.plan(Geometry(block_size=4).to_dict(), c)
    with pytest.raises(ValueError, match="mamba_ratio is 0"):
        G.plan(Geometry(block_size=4, full_slot_bytes=4096, mamba_slot_bytes=512).to_dict(), c)
    with pytest.raises(ValueError, match="holds no swa slot"):
        G.plan(Geometry(block_size=4, full_slot_bytes=4096, swa_slot_bytes=1 << 19,
                        swa_window_blocks=1).to_dict(), c)
    with pytest.raises(ValueError, match="leaving no FULL slot"):
        G.plan(Geometry(block_size=4, full_slot_bytes=1 << 20, swa_slot_bytes=1024,
                        swa_window_blocks=1).to_dict(), c)
    with pytest.raises(ValueError, match="window_blocks"):
        G.plan(Geometry(block_size=4, full_slot_bytes=4096, swa_slot_bytes=1024).to_dict(), c)
    with pytest.raises(ValueError, match="without an SWA pool"):
        G.plan(Geometry(block_size=4, full_slot_bytes=4096, swa_window_blocks=2).to_dict(), c)
    with pytest.raises(ValueError, match="register_chunk_tokens"):
        G.plan(Geometry(block_size=4, full_slot_bytes=4096,
                        register_chunk_tokens=4 * (G.MAX_REGISTER_CHUNK_SIZE + 1)).to_dict(), c)
    with pytest.raises(ValueError, match="block_size"):
        G.plan(Geometry(block_size=0, full_slot_bytes=4096).to_dict(), c)
    with pytest.raises(ValueError, match="unknown pool"):
        G.plan({"block_size": 4, "pools": {"full": {"slot_bytes": 4096}, "kv": {"slot_bytes": 1}}}, c)


def test_index_only_takes_counts_and_publishes_no_bytes():
    spec = Geometry(block_size=4, full_slot_bytes=999, full_slots=16, swa_slots=8,
                    swa_window_blocks=2).to_dict()
    g = G.plan(spec, cfg())
    assert g["pools"] == {"full": {"slot_bytes": 0, "num_slots": 16},
                          "swa": {"slot_bytes": 0, "num_slots": 8, "window_blocks": 2}}
    assert g["data_name"] == "" and g["slot_align"] == 0
    assert G.describe(g) == "FULL 16 slots, SWA 8 slots (window 2)"
    with pytest.raises(ValueError, match="index-only"):
        G.plan(Geometry(block_size=4, full_slot_bytes=4096).to_dict(), cfg())
    with pytest.raises(ValueError, match="index-only"):
        G.plan(Geometry(block_size=4, full_slots=16, swa_slot_bytes=1024, swa_window_blocks=2).to_dict(),
               cfg())
    shm = G.shm_config(g, background_evict_ratio=0.0, data_pool_ratio=4.0)
    assert (shm.max_blocks, shm.swa_max_blocks, shm.swa_window_blocks) == (16, 8, 2)
    assert shm.register_chunk_tokens == 4096 and shm.effective_register_chunk_size() == 4096 // 4


def test_uniform_fields_and_diff():
    c = cfg(data_bytes=1 << 20, swa_ratio=0.25)
    geo = Geometry(block_size=4, full_slot_bytes=4096, swa_slot_bytes=1024, swa_window_blocks=2)
    a = G.plan(geo.to_dict(), c)
    b = G.plan(geo.to_dict(), cfg(data_bytes=2 << 20, swa_ratio=0.5))
    assert G.diff_geometry(a, b, counts=False) == []
    assert G.diff_geometry(a, b, counts=True)          # counts are per node
    u = G.uniform_fields(a)
    assert "num_slots" not in u["pools"]["full"] and u["pools"]["swa"]["window_blocks"] == 2
    assert G.diff_geometry(G.plan(u, c), a, counts=True) == []    # a published geometry replans to the same
    other = G.plan(Geometry(block_size=8, full_slot_bytes=4096).to_dict(), c)
    d = G.diff_geometry(a, other, counts=False)
    assert any(x.startswith("block_size") for x in d) and any(x.startswith("swa pool") for x in d)


def test_server_config_validation():
    with pytest.raises(ValueError, match="ratio"):
        cfg(data_bytes=1, swa_ratio=0.6, mamba_ratio=0.5)
    with pytest.raises(ValueError, match="need data_bytes"):
        cfg(swa_ratio=0.1)
    with pytest.raises(ValueError, match="slot_align"):
        cfg(data_bytes=1, slot_align=3000)
    with pytest.raises(ValueError, match=r"in \[0, 1\)"):
        cfg(data_bytes=1, mamba_ratio=1.0)
    assert cfg().index_only and not cfg(data_bytes=1).index_only
    assert cfg(data_bytes=1).resolved_data_name == "/geo_data" and cfg().resolved_data_name == ""
