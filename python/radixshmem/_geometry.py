# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Slot geometry: what a slot looks like and how many there are. One dict,
JSON on the wire::

    {"version": 1, "index_name": "/x", "data_name": "/x_data", "hugepage_path": "",
     "block_size": 64, "slot_align": 4096, "register_chunk_tokens": 4096, "max_nodes": 0,
     "hash_scheme": "xxh64-v1",
     "pools": {"full":  {"slot_bytes": B, "num_slots": N},
               "swa":   {"slot_bytes": B, "num_slots": N, "window_blocks": W},
               "mamba": {"slot_bytes": B, "num_slots": N}}}

Only ``full`` is mandatory. A client states its side with ``Geometry`` (tokens
per block, bytes per slot, SWA window) and sends ``to_dict()`` in Configure;
``plan()`` resolves that against the server's ``ServerConfig``: in data mode
the slot counts come from ``data_bytes`` and the pool ratios, in index-only
mode from the client (the slots are its own storage; ``slot_bytes`` is 0).
The result is what the server publishes in Describe and clients adopt as
``RadixClient.geometry``. ``uniform_fields()`` is the part every node of a
cluster must share (counts and names may differ); it is what goes to etcd.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List

from radixshmem import _core

PROTOCOL_VERSION = 1
POOL_KINDS = ("full", "swa", "mamba")
# How clients derive block hashes; every client of a cluster must agree.
DEFAULT_HASH_SCHEME = "xxh64-v1"
MAX_REGISTER_CHUNK_SIZE = int(getattr(_core, "MAX_REGISTER_CHUNK_SIZE", 4096))


@dataclass
class Geometry:
    """A client's side of the geometry. Data-mode servers take the byte sizes
    and plan the counts; index-only servers take the counts (``*_slots``) and
    ignore the bytes. A pool exists when its bytes or its count is > 0.
    ``slot_align`` / ``register_chunk_tokens`` 0 = the server's default.
    ``hash_scheme`` names how the client derives block hashes ("" = the
    built-in ``compute_hashes``, ``DEFAULT_HASH_SCHEME``)."""

    block_size: int
    full_slot_bytes: int = 0
    swa_slot_bytes: int = 0
    swa_window_blocks: int = 0
    mamba_slot_bytes: int = 0
    full_slots: int = 0
    swa_slots: int = 0
    mamba_slots: int = 0
    slot_align: int = 0
    register_chunk_tokens: int = 0
    hash_scheme: str = ""

    def to_dict(self) -> Dict[str, Any]:
        pools: Dict[str, Dict[str, int]] = {}
        for kind in POOL_KINDS:
            slot_bytes = int(getattr(self, f"{kind}_slot_bytes"))
            num_slots = int(getattr(self, f"{kind}_slots"))
            if kind == "full" or slot_bytes > 0 or num_slots > 0:
                pools[kind] = {"slot_bytes": slot_bytes, "num_slots": num_slots}
        if "swa" in pools or self.swa_window_blocks:
            pools.setdefault("swa", {"slot_bytes": 0, "num_slots": 0})
            pools["swa"]["window_blocks"] = int(self.swa_window_blocks)
        out = {"version": PROTOCOL_VERSION, "block_size": int(self.block_size),
               "slot_align": int(self.slot_align),
               "register_chunk_tokens": int(self.register_chunk_tokens), "pools": pools}
        if self.hash_scheme:
            out["hash_scheme"] = str(self.hash_scheme)
        return out


def hash_scheme(g: Dict[str, Any]) -> str:
    return str(g.get("hash_scheme") or DEFAULT_HASH_SCHEME)


def as_spec(geometry: Any) -> Dict[str, Any]:
    return geometry.to_dict() if isinstance(geometry, Geometry) else dict(geometry)


def validate_spec(spec: Dict[str, Any]) -> None:
    """Shape of a client's geometry, mode-independent. ValueError otherwise."""
    if int(spec.get("version", PROTOCOL_VERSION)) != PROTOCOL_VERSION:
        raise ValueError(f"geometry protocol version {spec.get('version')} != {PROTOCOL_VERSION}")
    if int(spec.get("block_size", 0)) <= 0:
        raise ValueError("geometry needs block_size > 0")
    pools = spec.get("pools")
    if not isinstance(pools, dict) or "full" not in pools:
        raise ValueError("geometry needs a full pool")
    for kind, pool in pools.items():
        if kind not in POOL_KINDS:
            raise ValueError(f"unknown pool kind {kind!r}")
        if int(pool.get("slot_bytes", 0)) < 0 or int(pool.get("num_slots", 0)) < 0:
            raise ValueError(f"{kind} pool has a negative size")
    if "swa" in pools:
        swa = pools["swa"]
        if int(swa.get("window_blocks", 0)) <= 0:
            raise ValueError("an SWA pool needs window_blocks > 0")
        if int(swa.get("slot_bytes", 0)) <= 0 and int(swa.get("num_slots", 0)) <= 0:
            raise ValueError("swa_window_blocks given without an SWA pool")
    align = int(spec.get("slot_align", 0))
    if align < 0 or (align and align & (align - 1)):
        raise ValueError("slot_align must be 0 or a power of two")
    if int(spec.get("register_chunk_tokens", 0)) < 0:
        raise ValueError("register_chunk_tokens must be >= 0")


def _stride(slot_bytes: int, align: int) -> int:
    return -(-int(slot_bytes) // align) * align


def _plan_budget(pools: Dict[str, Any], budget: int, align: int,
                 ratios: Dict[str, Any], what: str,
                 ignore_counts: bool = False) -> Dict[str, Dict[str, int]]:
    """Slot counts for ``budget`` bytes: each aux pool gets ``ratio x budget``
    (``ratios``: kind -> (setting name, value)), FULL what is left."""
    out: Dict[str, Dict[str, int]] = {}
    for kind, pool in pools.items():
        if int(pool.get("num_slots", 0)) > 0 and not ignore_counts:
            raise ValueError(f"{what}: the server plans the {kind} pool from its budget; "
                             f"num_slots must not be given")
        if int(pool.get("slot_bytes", 0)) <= 0:
            raise ValueError(f"{what}: the {kind} pool needs slot_bytes > 0")
    fixed = 0
    for kind in ("swa", "mamba"):
        if kind not in pools:
            continue
        setting, ratio = ratios.get(kind, (f"{kind}_ratio", 0.0))
        ratio = float(ratio)
        if ratio <= 0:
            raise ValueError(f"the model needs a {kind} pool but the server's {setting} is 0")
        stride = _stride(pools[kind]["slot_bytes"], align)
        n = math.floor(ratio * budget / stride + 1e-6)
        if n <= 0:
            raise ValueError(f"{kind} ratio {ratio} of {budget} B holds no {kind} slot of {stride} B")
        fixed += n * stride
        out[kind] = {"slot_bytes": int(pools[kind]["slot_bytes"]), "num_slots": n}
    full_stride = _stride(pools["full"]["slot_bytes"], align)
    n_full = (budget - fixed) // full_stride
    if n_full <= 0:
        raise ValueError(f"the aux pools take {fixed} B of the {budget} B {what} budget, leaving no "
                         f"FULL slot of {full_stride} B")
    out["full"] = {"slot_bytes": int(pools["full"]["slot_bytes"]), "num_slots": int(n_full)}
    return out


def _ssd_files(cfg: Any, kind: str, pool: Dict[str, int]) -> Dict[str, Any]:
    """Lay ``pool``'s slots out over the SSD directories the way FlexKV
    addresses them: slot s is on directory s % D, in file (s // D) % F at
    block (s // D) // F. Rounds num_slots down to D x F x blocks_per_file."""
    dirs, slot_bytes = list(cfg.ssd_dirs), int(pool["slot_bytes"])
    per_dir = int(pool["num_slots"]) // len(dirs)
    per_file = per_dir
    if int(cfg.ssd_file_bytes) > 0:
        per_file = min(per_dir, int(cfg.ssd_file_bytes) // slot_bytes)
    if per_file <= 0:
        raise ValueError(f"SSD {kind} pool: {pool['num_slots']} slots of {slot_bytes} B do not fill "
                         f"one block per directory ({len(dirs)} directories)")
    files_per_dir = per_dir // per_file
    pool["num_slots"] = len(dirs) * files_per_dir * per_file
    stem = cfg.name.lstrip("/").replace("/", "_")
    return {"blocks_per_file": per_file,
            "files": [[os.path.join(d, f"{stem}_{kind}_{i}_{j}.bin") for j in range(files_per_dir)]
                      for i, d in enumerate(dirs)]}


def plan(spec: Dict[str, Any], cfg: Any) -> Dict[str, Any]:
    """Resolve a client's geometry against the server (``ServerConfig``:
    name, data_bytes, swa_ratio, mamba_ratio, slot_align, register_chunk_tokens,
    max_nodes, hugepage_path, data_name, ssd_*). Data mode plans every count from the
    budget: SWA and Mamba get ``ratio x data_bytes`` worth of slots, FULL what
    is left. Index-only takes the counts from the spec, unless the server owns
    SSD files (``ssd_bytes``): then it plans them from that budget the same
    way (ignoring any counts the spec brings) and publishes the files under
    ``ssd``. ValueError on a malformed
    spec, one for the other mode, or a budget that fits no slot."""
    validate_spec(spec)
    pools = spec["pools"]
    block_size = int(spec["block_size"])
    tokens = int(spec.get("register_chunk_tokens", 0)) or int(cfg.register_chunk_tokens)
    blocks = _core.ShmConfig(block_size=block_size, register_chunk_tokens=tokens).effective_register_chunk_size()
    if blocks > MAX_REGISTER_CHUNK_SIZE:
        raise ValueError(f"register_chunk_tokens {tokens} is {blocks} blocks of {block_size} tokens; "
                         f"the index allows at most {MAX_REGISTER_CHUNK_SIZE}")
    out: Dict[str, Any] = {
        "version": PROTOCOL_VERSION, "index_name": cfg.name, "data_name": "",
        "hugepage_path": cfg.hugepage_path, "block_size": block_size, "slot_align": 0,
        "register_chunk_tokens": tokens, "max_nodes": int(cfg.max_nodes),
        "hash_scheme": hash_scheme(spec), "pools": {}}
    data_bytes = int(cfg.data_bytes)
    ssd_bytes = int(getattr(cfg, "ssd_bytes", 0))
    if data_bytes <= 0 and ssd_bytes > 0:
        if "mamba" in pools:
            raise ValueError("the SSD files hold no Mamba pool")
        out["pools"] = _plan_budget(pools, ssd_bytes, 1, {"swa": ("ssd_swa_ratio", cfg.ssd_swa_ratio)}, "SSD",
                                    ignore_counts=True)
        out["ssd"] = {kind: _ssd_files(cfg, kind, pool) for kind, pool in out["pools"].items()}
    elif data_bytes <= 0:
        for kind, pool in pools.items():
            n = int(pool.get("num_slots", 0))
            if n <= 0:
                raise ValueError(f"index-only server: the {kind} pool needs num_slots > 0 "
                                 "(the slots are the client's own storage)")
            out["pools"][kind] = {"slot_bytes": 0, "num_slots": n}
    else:
        align = int(spec.get("slot_align", 0)) or int(cfg.slot_align)
        out["slot_align"] = align
        out["data_name"] = cfg.data_name or f"{cfg.name}_data"
        out["pools"] = _plan_budget(pools, data_bytes, align,
                                    {"swa": ("swa_ratio", cfg.swa_ratio),
                                     "mamba": ("mamba_ratio", cfg.mamba_ratio)}, "data mode")
    if "swa" in pools:
        out["pools"]["swa"]["window_blocks"] = int(pools["swa"]["window_blocks"])
    if out["max_nodes"] <= 0:
        out["max_nodes"] = 2 * int(out["pools"]["full"]["num_slots"]) + 1024
    return out


def uniform_fields(g: Dict[str, Any]) -> Dict[str, Any]:
    """What every node of a cluster must share: hash granularity, alignment,
    RHT chunk, the pool set and its byte sizes. Counts and names are per node."""
    return {"version": PROTOCOL_VERSION, "block_size": int(g["block_size"]),
            "slot_align": int(g.get("slot_align", 0)),
            "register_chunk_tokens": int(g.get("register_chunk_tokens", 0)),
            "hash_scheme": hash_scheme(g),
            "pools": {kind: {k: int(v) for k, v in pool.items() if k != "num_slots"}
                      for kind, pool in g["pools"].items()}}


def diff_geometry(a: Dict[str, Any], b: Dict[str, Any], *, counts: bool) -> List[str]:
    """Field-by-field differences of two planned or published geometries;
    ``counts`` also compares num_slots. Empty = same."""
    diffs = []
    for key in ("block_size", "slot_align", "register_chunk_tokens"):
        if int(a.get(key, 0)) != int(b.get(key, 0)):
            diffs.append(f"{key}: {a.get(key, 0)} != {b.get(key, 0)}")
    if hash_scheme(a) != hash_scheme(b):
        diffs.append(f"hash_scheme: {hash_scheme(a)} != {hash_scheme(b)}")
    pa, pb = a["pools"], b["pools"]
    fields = ("slot_bytes", "window_blocks") + (("num_slots",) if counts else ())
    for kind in POOL_KINDS:
        if (kind in pa) != (kind in pb):
            diffs.append(f"{kind} pool: {'present' if kind in pa else 'absent'} != "
                         f"{'present' if kind in pb else 'absent'}")
        elif kind in pa:
            diffs += [f"{kind}.{f}: {pa[kind].get(f, 0)} != {pb[kind].get(f, 0)}"
                      for f in fields if int(pa[kind].get(f, 0)) != int(pb[kind].get(f, 0))]
    return diffs


def geometry_bytes(g: Dict[str, Any]) -> int:
    return sum(int(p["slot_bytes"]) * int(p["num_slots"]) for p in g["pools"].values())


def shm_config(g: Dict[str, Any], *, background_evict_ratio: float, data_pool_ratio: float,
               kvevent_ring_capacity: int = 0):
    pools = g["pools"]
    cfg = _core.ShmConfig()
    cfg.max_nodes = int(g.get("max_nodes", 0)) or (2 * int(pools["full"]["num_slots"]) + 1024)
    cfg.max_blocks = int(pools["full"]["num_slots"])
    cfg.block_size = int(g["block_size"])
    cfg.hugepage_path = g.get("hugepage_path", "")
    cfg.register_chunk_tokens = int(g["register_chunk_tokens"])
    cfg.background_evict_ratio = float(background_evict_ratio)
    cfg.data_pool_ratio = float(data_pool_ratio)
    mask = int(_core.COMPONENT_MASK_FULL)
    if "swa" in pools:
        mask |= int(_core.COMPONENT_MASK_SWA)
        cfg.swa_max_blocks = int(pools["swa"]["num_slots"])
        cfg.swa_window_blocks = int(pools["swa"]["window_blocks"])
    if "mamba" in pools:
        mask |= int(_core.COMPONENT_MASK_MAMBA)
        cfg.mamba_max_slots = int(pools["mamba"]["num_slots"])
        cfg.mamba_state_bytes = int(pools["mamba"].get("slot_bytes", 0))   # 0 = unstamped
    cfg.component_mask = mask
    cfg.kvevent_ring_capacity = int(kvevent_ring_capacity)
    return cfg


def store_config(g: Dict[str, Any], *, prefault: bool = True):
    from radixshmem import _data

    pools = g["pools"]

    def pool(kind: str):
        p = pools.get(kind)
        if p is None or int(p["num_slots"]) <= 0:
            return _data.SlotPoolConfig()
        return _data.SlotPoolConfig(int(p["num_slots"]), int(p["slot_bytes"]))

    return _data.SlotStoreConfig(
        name=g["data_name"], full=pool("full"), swa=pool("swa"), mamba=pool("mamba"),
        slot_align=int(g["slot_align"]), hugepage_path=g.get("hugepage_path", ""),
        prefault=prefault)


def check_pools_aligned(client, store) -> None:
    """Every index slot pool must equal the store pool of the same kind."""
    ct = _core.ComponentType
    cc = client.component_config
    diffs = []
    if int(cc.component_mask) != int(store.pool_mask):
        diffs.append(f"component_mask: index={cc.component_mask} store={store.pool_mask}")
    for kind, total in ((ct.FULL, client.mempool_total()), (ct.SWA, client.swa_mempool_total()),
                        (ct.MAMBA, client.mamba_mempool_total())):
        have = store.pool(kind).num_slots if store.has_pool(kind) else 0
        if int(total) != int(have):
            diffs.append(f"{kind.name} slots: index={total} store={have}")
    if store.has_pool(ct.MAMBA) and int(cc.mamba_state_bytes) != int(store.pool(ct.MAMBA).slot_bytes):
        diffs.append(f"mamba_state_bytes: index={cc.mamba_state_bytes} "
                     f"store stride={store.pool(ct.MAMBA).slot_bytes}")
    if diffs:
        raise ValueError("index/store pools misaligned: " + "; ".join(diffs))


def check_attached(g: Dict[str, Any], client, store) -> None:
    """The index, and the store when there is one (``store=None`` for an
    index-only server), must match the geometry pool by pool. An index without
    a Mamba byte stamp is accepted."""
    ct = _core.ComponentType
    kinds = {"full": ct.FULL, "swa": ct.SWA, "mamba": ct.MAMBA}
    totals = {"full": client.mempool_total(), "swa": client.swa_mempool_total(),
              "mamba": client.mamba_mempool_total()}
    cc = client.component_config
    mask = int(cc.component_mask)
    want = {"full": True, "swa": "swa" in g["pools"], "mamba": "mamba" in g["pools"]}
    have = {"full": True, "swa": bool(mask & int(_core.COMPONENT_MASK_SWA)),
            "mamba": bool(mask & int(_core.COMPONENT_MASK_MAMBA))}
    diffs: list[str] = []
    for kind in POOL_KINDS:
        if want[kind] != have[kind]:
            diffs.append(f"{kind}: index has={have[kind]} needed={want[kind]}")
            continue
        if not want[kind]:
            continue
        p = g["pools"][kind]
        if int(totals[kind]) != int(p["num_slots"]):
            diffs.append(f"{kind}.num_slots: index={totals[kind]} geometry={p['num_slots']}")
        if store is None:
            continue
        if not store.has_pool(kinds[kind]):
            diffs.append(f"{kind}: store has no pool")
            continue
        sp = store.pool(kinds[kind])
        if int(sp.num_slots) != int(p["num_slots"]):
            diffs.append(f"{kind}.num_slots: store={sp.num_slots} geometry={p['num_slots']}")
        if int(sp.slot_bytes) < int(p["slot_bytes"]):
            diffs.append(f"{kind}.slot_bytes: store={sp.slot_bytes} geometry={p['slot_bytes']}")
    if want["swa"] and have["swa"] and int(cc.swa_window_blocks) != int(g["pools"]["swa"]["window_blocks"]):
        diffs.append(f"swa.window_blocks: index={cc.swa_window_blocks} "
                     f"geometry={g['pools']['swa']['window_blocks']}")
    if (store is not None and want["mamba"] and have["mamba"]
            and int(cc.mamba_state_bytes) > int(store.pool(ct.MAMBA).slot_bytes)):
        diffs.append(f"mamba.slot_bytes: index stamp={cc.mamba_state_bytes} "
                     f"store={store.pool(ct.MAMBA).slot_bytes}")
    if int(client.block_size()) != int(g["block_size"]):
        diffs.append(f"block_size: index={client.block_size()} geometry={g['block_size']}")
    if store is not None and g.get("hugepage_path") and not store.is_hugepage:
        diffs.append("hugepage_path requested but the store is plain shm")
    if diffs:
        raise ValueError("index/store/geometry mismatch: " + "; ".join(diffs))


def describe(g: Dict[str, Any]) -> str:
    parts = []
    for kind in POOL_KINDS:
        p = g["pools"].get(kind)
        if p is None:
            continue
        s = f"{kind.upper()} {p['num_slots']} slots"
        if int(p.get("slot_bytes", 0)) > 0:
            s += f" x {p['slot_bytes']} B"
        if kind == "swa":
            s += f" (window {p['window_blocks']})"
        parts.append(s)
    return ", ".join(parts)
