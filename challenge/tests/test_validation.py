from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from challenge.techjam_recsys.validation import fidelity_can_qualify, last_days_holdout


def test_last_three_days_split_is_strict_deterministic_and_nonoverlapping() -> None:
    frame = pd.DataFrame(
        {
            "date": np.repeat([1, 2, 3, 4, 5], 2),
            "time_ms": np.arange(10, dtype=np.int64) * 100,
            "long_view": [0, 1] * 5,
        }
    )
    train_a, holdout_a, manifest_a = last_days_holdout(frame)
    train_b, holdout_b, manifest_b = last_days_holdout(frame)
    assert np.array_equal(train_a, train_b)
    assert np.array_equal(holdout_a, holdout_b)
    assert not np.any(train_a & holdout_a)
    assert np.all(train_a | holdout_a)
    assert tuple(frame.loc[holdout_a, "date"].unique()) == (3, 4, 5)
    assert manifest_a.manifest_sha256 == manifest_b.manifest_sha256
    assert manifest_a.max_train_time_ms < manifest_a.min_holdout_time_ms


def test_equal_boundary_timestamps_are_all_assigned_to_holdout() -> None:
    frame = pd.DataFrame({"date": [1, 2, 3, 4], "time_ms": [10, 20, 20, 30]})
    train, holdout, manifest = last_days_holdout(frame, holdout_days=2)
    assert train.tolist() == [True, False, False, False]
    assert holdout.tolist() == [False, True, True, True]
    assert manifest.max_train_time_ms < manifest.min_holdout_time_ms


def test_only_full_fidelity_can_qualify() -> None:
    assert fidelity_can_qualify("screen") is False
    assert fidelity_can_qualify("full") is True
    with pytest.raises(ValueError):
        fidelity_can_qualify("unknown")
