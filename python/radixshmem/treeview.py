# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tree algorithms of the monitoring endpoint (``radixshmem.monitor``), numpy only.

A tree is a set of parallel arrays with the root at index 0: ``parent`` (-1 for
the root), ``weight`` (blocks the node holds itself) and ``key`` (uint64, the
prefix hash at the node's end; 0 for the root).

Block hashes are prefix hashes (``gen_hashes`` never resets the hasher), so the
hash at block depth d names the whole prefix [0, d) on every node of the
cluster. ``anchors`` cuts a node's tree at every ``stride``-th depth and at its
leaf ends; ``MergedTree`` unions the anchors of many trees by hash into one
tree whose nodes carry how many trees hold them.
"""

from __future__ import annotations

import heapq
import io
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

ANCHOR_FIELDS = ("hash", "parent", "depth", "blocks", "age_ms", "pinned")


def offsets(num_blocks: np.ndarray) -> np.ndarray:
    off = np.zeros(len(num_blocks) + 1, np.int64)
    np.cumsum(num_blocks, out=off[1:])
    return off


def levels(parent: np.ndarray) -> np.ndarray:
    """Edges from the root by pointer jumping; parents need not come first."""
    anc = np.where(parent >= 0, parent, 0).astype(np.int64)
    dist = (parent >= 0).astype(np.int64)
    while np.any(anc != 0):
        dist = dist + dist[anc]
        anc = anc[anc]
    return dist


def subtree_sums(parent: np.ndarray, weight: np.ndarray, level: np.ndarray) -> np.ndarray:
    n = len(parent)
    sub = weight.astype(np.float64)
    if n <= 1:
        return sub
    order = np.argsort(level, kind="stable")
    lv = level[order]
    bounds = np.searchsorted(lv, np.arange(int(lv[-1]) + 2))
    for lvl in range(int(lv[-1]), 0, -1):
        idx = order[bounds[lvl]:bounds[lvl + 1]]
        sub += np.bincount(parent[idx], weights=sub[idx], minlength=n)
    return sub


def hex_key(k) -> str:
    return format(int(k), "016x")


class TreeIndex:
    """A tree prepared for ``view``: subtree weights and children sorted
    heaviest first. Built once per snapshot; ``view`` is cheap."""

    def __init__(self, parent: np.ndarray, weight: np.ndarray, key: np.ndarray,
                 attrs: Optional[Dict[str, np.ndarray]] = None, level: Optional[np.ndarray] = None):
        self.parent = parent.astype(np.int64)
        self.weight = weight.astype(np.int64)
        self.key = key.astype(np.uint64)
        self.attrs = attrs or {}
        self.n = len(self.parent)
        self.level = levels(self.parent) if level is None else level.astype(np.int64)
        self.sub = subtree_sums(self.parent, self.weight, self.level)
        kids = np.arange(1, self.n, dtype=np.int64)
        kp = self.parent[1:]
        order = np.lexsort((-self.sub[1:], kp))
        self.kids = kids[order]
        self.kid_ptr = np.searchsorted(kp[order], np.arange(self.n + 1))

    @property
    def total(self) -> float:
        return float(self.sub[0]) if self.n else 0.0

    def find(self, key: str) -> Optional[int]:
        try:
            k = np.uint64(int(key, 16))
        except ValueError:
            return None
        hit = np.flatnonzero(self.key == k)
        return int(hit[0]) if len(hit) else None

    def view(self, root: int = 0, budget: int = 2000, min_frac: Optional[float] = None,
             extra: Optional[Callable[[np.ndarray], Dict[str, list]]] = None) -> Dict:
        """The heaviest ``budget`` nodes under ``root``, chosen greedily so they
        form a tree. Children left out are summed per node into ``rest_w`` /
        ``rest_n``; subtrees lighter than ``min_frac`` of the root (default
        0.25 / budget) never show."""
        sub, kid_ptr, kids = self.sub, self.kid_ptr, self.kids
        thresh = sub[root] * (0.25 / max(budget, 1) if min_frac is None else min_frac)
        selected = [root]
        heap: List[tuple] = []

        def push(p: int, r: int) -> None:
            s = kid_ptr[p] + r
            if s < kid_ptr[p + 1]:
                c = int(kids[s])
                if sub[c] > 0 and sub[c] >= thresh:
                    heapq.heappush(heap, (-sub[c], c, p, r))

        push(root, 0)
        while heap and len(selected) < budget:
            _, c, p, r = heapq.heappop(heap)
            selected.append(c)
            push(p, r + 1)
            push(c, 0)

        sel = np.asarray(selected, np.int64)
        local = {int(g): i for i, g in enumerate(selected)}
        par = np.array([-1] + [local[int(self.parent[g])] for g in selected[1:]], np.int64)
        shown_w = np.bincount(par[1:], weights=sub[sel[1:]], minlength=len(sel))
        shown_n = np.bincount(par[1:], minlength=len(sel))
        nkids = kid_ptr[sel + 1] - kid_ptr[sel]
        out = {
            "root_key": hex_key(self.key[root]),
            "total": float(sub[root]),
            "nodes_total": int(self.n),
            "id": list(range(len(sel))),
            "parent": par.tolist(),
            "key": [hex_key(k) for k in self.key[sel]],
            "w": sub[sel].tolist(),
            "self": self.weight[sel].tolist(),
            "rest_w": np.maximum(sub[sel] - self.weight[sel] - shown_w, 0).tolist(),
            "rest_n": (nkids - shown_n).tolist(),
            "level": (self.level[sel] - self.level[root]).tolist(),
        }
        for name, arr in self.attrs.items():
            v = arr[sel]
            out[name] = np.round(v, 1).tolist() if v.dtype.kind == "f" else v.tolist()
        if extra is not None:
            out.update(extra(sel))
        crumbs, g = [], int(self.parent[root])
        while g >= 0 and len(crumbs) < 64:
            crumbs.append({"key": hex_key(self.key[g]), "w": float(sub[g])})
            g = int(self.parent[g])
        out["crumbs"] = crumbs[::-1]
        return out


# ------------------------------------------------------------ one node's tree

def node_tree(dump: Dict) -> TreeIndex:
    """The radix tree of one ``RadixClient.dump_tree()`` as it is: one entry
    per radix node, keyed by the prefix hash at its end."""
    nb = dump["num_blocks"].astype(np.int64)
    off = offsets(nb)
    hashes = dump["hashes"]
    key = np.zeros(len(nb), np.uint64)
    has = nb > 0
    key[has] = hashes[off[1:][has] - 1]
    age = (np.int64(dump["now"]) - dump["last_access"].astype(np.int64)) / 1e6
    age = np.clip(age, 0, None)
    age[0] = 0
    attrs = {
        "age_ms": age.astype(np.float64),
        "pinned": (dump["ref_count"] > 0).astype(np.int64),
        "swa": dump["swa_blocks"].astype(np.int64),
        "mamba": dump["mamba"].astype(np.int64),
        "depth": dump["depth"].astype(np.int64),
    }
    return TreeIndex(dump["parent"], nb, key, attrs, level=dump["level"])


def anchors(dump: Dict, stride: int, max_anchors: int = 0) -> Dict[str, np.ndarray]:
    """Cut points of one dump: every block depth that is a multiple of
    ``stride`` plus each leaf's end. ``parent`` is the anchor one cut up the
    path (0 = the root), the same on every node holding the prefix. Over
    ``max_anchors`` the deepest are dropped, which keeps every parent."""
    s = max(int(stride), 1)
    parent = dump["parent"].astype(np.int64)
    nb = dump["num_blocks"].astype(np.int64)
    start = dump["depth"].astype(np.int64)
    end = start + nb
    off = offsets(nb)
    hashes = dump["hashes"]
    n = len(nb)
    level = dump["level"].astype(np.int64)

    first = (start // s + 1) * s
    cnt = np.where(first <= end, (end - first) // s + 1, 0)
    last_cut = (end // s) * s
    own_last = np.zeros(n, np.uint64)
    m = cnt > 0
    own_last[m] = hashes[off[:-1][m] + (last_cut[m] - start[m] - 1)]

    # inc[i]: the last cut above node i's first block; out[i]: the last cut at
    # or above its last block. BFS order keeps levels contiguous.
    inc = np.zeros(n, np.uint64)
    out = np.zeros(n, np.uint64)
    bounds = np.searchsorted(level, np.arange(int(level[-1]) + 2)) if n else [0]
    for lvl in range(1, len(bounds) - 1):
        idx = np.arange(bounds[lvl], bounds[lvl + 1])
        inc[idx] = out[parent[idx]]
        out[idx] = np.where(m[idx], own_last[idx], inc[idx])

    node_of = np.repeat(np.arange(n), cnt)
    k = np.arange(len(node_of)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    d = first[node_of] + k * s
    h = hashes[off[:-1][node_of] + (d - start[node_of] - 1)]
    hp = np.empty_like(h)
    if len(h):
        hp[1:] = h[:-1]
        hp[0] = 0
    p = np.where(k == 0, inc[node_of], hp)
    blocks = np.full(len(h), s, np.int64)

    has_kid = np.zeros(n, bool)
    has_kid[parent[1:]] = True
    tail = (~has_kid) & (nb > 0) & (end % s != 0)
    t_idx = np.flatnonzero(tail)
    h = np.concatenate([h, hashes[off[1:][t_idx] - 1]])
    p = np.concatenate([p, out[t_idx]])
    d = np.concatenate([d, end[t_idx]])
    blocks = np.concatenate([blocks, end[t_idx] - last_cut[t_idx]])
    node_of = np.concatenate([node_of, t_idx])

    age = np.clip((np.int64(dump["now"]) - dump["last_access"].astype(np.int64)) / 1e6, 0, None)
    res = {"hash": h.astype(np.uint64), "parent": p.astype(np.uint64), "depth": d.astype(np.uint32),
           "blocks": blocks.astype(np.uint32), "age_ms": age[node_of].astype(np.float32),
           "pinned": (dump["ref_count"][node_of] > 0).astype(np.uint8)}
    if max_anchors and len(h) > max_anchors:
        keep = np.argsort(res["depth"], kind="stable")[:max_anchors]
        res = {f: v[keep] for f, v in res.items()}
    return res


def pack(arrays: Dict[str, np.ndarray], **meta) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays, **{f"meta_{k}": np.asarray(v) for k, v in meta.items()})
    return buf.getvalue()


def unpack(data: bytes):
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        arrays = {f: z[f] for f in z.files if not f.startswith("meta_")}
        meta = {f[5:]: z[f].item() for f in z.files if f.startswith("meta_")}
    return arrays, meta


# ------------------------------------------------------------ the cluster's tree

class MergedTree:
    """Anchors of many trees unioned by hash, with unary chains held by the
    same set of trees folded into one node. ``index.attrs['dup']`` is the
    number of trees holding a node; ``owners(i)`` lists them."""

    def __init__(self, parts: Sequence[Dict[str, np.ndarray]]):
        self.ntrees = len(parts)
        sizes = [len(p["hash"]) for p in parts]
        cat = {f: np.concatenate([p[f] for p in parts]) if parts else np.zeros(0)
               for f in ANCHOR_FIELDS}
        tree = np.repeat(np.arange(len(parts), dtype=np.int64), sizes)
        self.total_blocks = float(cat["blocks"].astype(np.float64).sum()) if parts else 0.0

        uniq, first, inv = np.unique(cat["hash"].astype(np.uint64), return_index=True, return_inverse=True)
        nu = len(uniq)
        pair = np.unique(inv.astype(np.int64) * max(len(parts), 1) + tree)
        pu, pt = pair // max(len(parts), 1), pair % max(len(parts), 1)
        dup = np.bincount(pu, minlength=nu)
        rng = np.random.default_rng(12345)
        tsig = rng.integers(1, 2 ** 63, size=max(len(parts), 1), dtype=np.int64).astype(np.uint64)
        sig = np.zeros(nu, np.uint64)
        np.add.at(sig, pu, tsig[pt])
        self._own_ptr = np.searchsorted(pu, np.arange(nu + 1))
        self._own = pt

        age = np.full(nu, np.inf)
        np.minimum.at(age, inv, cat["age_ms"].astype(np.float64))
        pinned = np.zeros(nu, np.int64)
        np.maximum.at(pinned, inv, cat["pinned"].astype(np.int64))

        # Anchor tree: index 0 = root, anchor j at j + 1.
        ph = cat["parent"][first].astype(np.uint64)
        pos = np.searchsorted(uniq, ph)
        pos_c = np.minimum(pos, max(nu - 1, 0))
        found = (ph != 0) & (pos < nu) & (uniq[pos_c] == ph) if nu else np.zeros(0, bool)
        parent = np.concatenate([[-1], np.where(found, pos_c + 1, 0)]).astype(np.int64)
        weight = np.concatenate([[0], cat["blocks"][first].astype(np.int64)])
        key = np.concatenate([np.zeros(1, np.uint64), uniq.astype(np.uint64)])
        dup = np.concatenate([[len(parts)], dup])
        sig = np.concatenate([np.zeros(1, np.uint64), sig])
        age = np.concatenate([[0.0], age])
        pinned = np.concatenate([[0], pinned])
        depth = np.concatenate([[0], cat["depth"][first].astype(np.int64)])
        self.unique_blocks = float(weight.sum())

        # Fold chains: a node whose parent has it as the only child and the same owners.
        n = len(parent)
        nkids = np.bincount(parent[1:], minlength=n)
        merge = np.zeros(n, bool)
        merge[1:] = (parent[1:] > 0) & (nkids[parent[1:]] == 1) & (sig[1:] == sig[parent[1:]])
        head = np.arange(n)
        head[merge] = parent[merge]
        while True:
            nxt = head[head]
            if np.array_equal(nxt, head):
                break
            head = nxt
        heads = np.flatnonzero(head == np.arange(n))
        gid = np.searchsorted(heads, head)
        g_parent = np.where(parent[heads] >= 0, gid[np.maximum(parent[heads], 0)], -1)
        g_weight = np.bincount(gid, weights=weight, minlength=len(heads)).astype(np.int64)
        g_age = np.full(len(heads), np.inf)
        np.minimum.at(g_age, gid, age)
        g_pin = np.zeros(len(heads), np.int64)
        np.maximum.at(g_pin, gid, pinned)
        g_end = np.zeros(len(heads), np.int64)
        np.maximum.at(g_end, gid, depth)
        self._anchor_of = heads - 1          # group -> its head anchor (-1 = root)
        self.index = TreeIndex(g_parent, g_weight, key[heads], {
            "dup": dup[heads].astype(np.int64),
            "age_ms": np.where(np.isfinite(g_age), g_age, 0.0),
            "pinned": g_pin,
            "depth": g_end,
        })

    def owners(self, sel: np.ndarray, limit: int = 16) -> Dict[str, list]:
        out = []
        for g in sel:
            a = int(self._anchor_of[g])
            if a < 0:
                out.append([])
                continue
            out.append(self._own[self._own_ptr[a]:self._own_ptr[a + 1]][:limit].tolist())
        return {"owners": out}


def pick_stride(max_blocks: int, per_tree_budget: int) -> int:
    """Smallest power of two that keeps a tree of ``max_blocks`` under budget."""
    need = -(-max(int(max_blocks), 1) // max(int(per_tree_budget), 1))
    s = 1
    while s < need:
        s <<= 1
    return s


# ------------------------------------------------------------ radix nodes shared across servers

def node_heads(dump: Dict, index: TreeIndex, cap: int) -> Dict[str, np.ndarray]:
    """First block hash, own blocks and start depth of the ``cap`` heaviest
    radix nodes of one dump (``index`` = ``node_tree(dump)``)."""
    nb = dump["num_blocks"].astype(np.int64)
    off = offsets(nb)
    ids = np.flatnonzero(nb > 0)
    if cap and len(ids) > cap:
        ids = ids[np.argsort(-index.sub[ids], kind="stable")[:cap]]
    return {"first": dump["hashes"][off[ids]].astype(np.uint64), "blocks": nb[ids].astype(np.uint32),
            "depth": dump["depth"][ids].astype(np.uint32)}


def top_shared(parts: Sequence[Dict[str, np.ndarray]], k: int = 20) -> List[tuple]:
    """(first hash, servers, own blocks, start depth) of the ``k`` radix nodes
    whose first hash starts a node on the most servers, more than one;
    ties go to the larger node."""
    firsts, blocks, depth = [], [], []
    for p in parts:
        f, i = np.unique(p["first"], return_index=True)    # once per server
        firsts.append(f)
        blocks.append(p["blocks"][i])
        depth.append(p["depth"][i])
    if not firsts:
        return []
    f, b, d = np.concatenate(firsts), np.concatenate(blocks), np.concatenate(depth)
    u, first, cnt = np.unique(f, return_index=True, return_counts=True)
    keep = cnt > 1
    u, first, cnt = u[keep], first[keep], cnt[keep]
    order = np.lexsort((-b[first].astype(np.int64), -cnt))[:k]
    return [(int(u[i]), int(cnt[i]), int(b[first[i]]), int(d[first[i]])) for i in order]
