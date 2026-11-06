# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol admission tests also run on hosts without the native extension."""

import importlib.util
from pathlib import Path
import sys
import time

import pytest

spec = importlib.util.spec_from_file_location(
    "_operations_under_test", Path(__file__).parents[1] / "radixshmem" / "_operations.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Operations = module.Operations


def test_duplicate_and_retired_operation_cannot_be_resubmitted():
    ledger = Operations("epoch", 2, 100)
    state, first = ledger.reserve("epoch", "client", 1, 40)
    assert state == "RESERVED"
    assert ledger.reserve("epoch", "client", 1, 40)[1] is first
    assert ledger.reserved_bytes == 40
    ledger.retire("client", 1)
    assert ledger.reserve("epoch", "client", 1, 40) == ("RETIRED", None)
    assert ledger.reserved_bytes == 0
    # An older delayed request cannot pass a later admission watermark.
    ledger.reserve("epoch", "client", 3, 40)
    assert ledger.reserve("epoch", "client", 2, 40) == ("RETIRED", None)


def test_cancelled_or_done_but_unacknowledged_transfer_keeps_node_budget():
    ledger = Operations("epoch", 8, 100)
    _, op = ledger.reserve("epoch", "a", 1, 70)
    op.state = "SUBMITTED"
    op.expires = time.monotonic() - 1
    assert ledger.reserve("epoch", "b", 1, 40) == ("RETIRED", None)
    ledger.expire()
    assert ledger.reserved_bytes == 70
    ledger.retire("a", 1)
    assert ledger.reserve("epoch", "b", 2, 40)[0] == "RESERVED"


def test_lost_reservation_can_expire_but_late_submit_stays_rejected():
    ledger = Operations("epoch", 1, 100)
    _, op = ledger.reserve("epoch", "a", 1, 100)
    op.expires = time.monotonic() - 1
    assert ledger.lookup("epoch", "a", 1) == ("RETIRED", None)
    assert ledger.reserve("epoch", "b", 1, 100)[0] == "RESERVED"
    assert ledger.lookup("epoch", "a", 1) == ("RETIRED", None)


def test_epoch_change_and_unknown_do_not_mean_not_accepted():
    ledger = Operations("new", 1, 100)
    assert ledger.lookup("old", "a", 1) == ("EPOCH_MISMATCH", None)
    assert ledger.lookup("new", "a", 1) == ("UNKNOWN", None)
    assert not ledger.active


def test_session_and_result_records_are_bounded_without_unsafe_gc():
    ledger = Operations("epoch", 1, 100, max_sessions=1)
    ledger.reserve("epoch", "a", 1, 100)
    ledger.retire("a", 1)
    assert ledger.reserve("epoch", "b", 1, 100) == ("BUSY", None)
    assert ledger.lookup("epoch", "a", 1) == ("RETIRED", None)
    with pytest.raises(ValueError):
        Operations("epoch", 0, 100)


def test_retry_cannot_change_its_byte_reservation():
    ledger = Operations("epoch", 2, 100)
    ledger.reserve("epoch", "a", 1, 50)
    with pytest.raises(ValueError, match="changed"):
        ledger.reserve("epoch", "a", 1, 100)


def test_node_wide_singleflight_waits_for_publish_ack_and_cancels_independently():
    ledger = Operations("epoch", 4, 100)
    _, owner = ledger.reserve("epoch", "a", 1, 100, key="exact-prefix-and-mask")
    owner.state = "HELD"
    owner.expires = time.monotonic() - 1
    state, follower = ledger.reserve("epoch", "b", 1, 100, key="exact-prefix-and-mask")
    assert state == "FOLLOWING" and ledger.reserved_bytes == 100
    ledger.retire("b", 1)
    assert ledger.lookup("epoch", "a", 1)[1] is owner
    ledger.reserve("epoch", "c", 1, 100, key="exact-prefix-and-mask")
    ledger.retire("a", 1)
    assert ledger.lookup("epoch", "c", 1) == ("RETIRED", None)
    assert not ledger.active and ledger.reserved_bytes == 0
