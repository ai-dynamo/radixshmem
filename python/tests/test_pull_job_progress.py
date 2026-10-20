# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PullJob progress fields and pull_async range slicing.

No radix-server required: exercises the pure-Python job handle surface.
"""

from __future__ import annotations

import numpy as np
import pytest

from radixshmem.client import PullJob, PullResult, RadixClient, _empty


def _result(common_hit: int) -> PullResult:
    return PullResult(
        common_hit,
        _empty(),
        0,
        _empty(),
        -1,
        0,
        0,
        -1,
        lambda: None,
    )


def test_published_hit_tracks_local_then_common():
    job = PullJob(1, client=None)  # type: ignore[arg-type]
    job.local_hit = 2
    job.planned_hit = 8
    assert not job.done()
    assert job.published_hit == 2

    job._f.set_result(_result(5))
    assert job.done()
    assert job.published_hit == 5


def test_published_hit_stays_local_on_failed_future():
    job = PullJob(2, client=None)  # type: ignore[arg-type]
    job.local_hit = 3
    job._f.set_exception(RuntimeError("boom"))
    assert job.done()
    assert job.published_hit == 3


def test_pull_async_range_bounds_validation():
    client = RadixClient.__new__(RadixClient)
    client.local_read_only = False
    hashes = np.arange(10, dtype=np.uint64)
    with pytest.raises(ValueError, match="out of bounds"):
        RadixClient.pull_async(client, hashes, start=4, end=3)
    with pytest.raises(ValueError, match="out of bounds"):
        RadixClient.pull_async(client, hashes, start=-1, end=2)
    with pytest.raises(ValueError, match="out of bounds"):
        RadixClient.pull_async(client, hashes, start=0, end=11)
