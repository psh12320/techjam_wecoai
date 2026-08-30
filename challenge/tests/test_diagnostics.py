from __future__ import annotations

import json

import numpy as np
import pandas as pd

from challenge.techjam_recsys.diagnostics import (
    aggregate_validation_diagnostics,
    diagnostics_prompt_summary,
)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "video_id": [10, 11, 10, 12, 13, 14],
            "author_id": [100, 101, 100, 102, 103, 104],
            "primary_tag": [5, 6, 5, 7, 8, 9],
        }
    )
    valid = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2, 4, 4],
            "video_id": [10, 15, 16, 10, 12, 17, 18, 19],
            "author_id": [100, 105, 106, 100, 102, 107, 108, 109],
            "primary_tag": [5, 10, 11, 5, 7, 12, 13, 14],
            "duration_ms": [5000, 10000, 20000, 30000, 50000, 80000, 10000, 90000],
            "user_active_degree": [1, 1, 1, 2, 2, 2, 3, 3],
            # Trusted diagnostics must strip these before segment construction.
            "long_view": [1, 0, 1, 0, 1, 0, 1, 0],
            "play_time_ms": [1] * 8,
        }
    )
    return train, valid


def test_trusted_diagnostics_are_aggregate_only_and_deterministic() -> None:
    train, valid = _frames()
    users = valid["user_id"].to_numpy()
    labels = np.asarray([1, 0, 1, 0, 1, 0, 1, 0])
    predictions = {
        "baseline": np.asarray([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4]),
        "diverse": np.asarray([0.1, 0.9, 0.8, 0.7, 0.2, 0.3, 0.4, 0.6]),
    }
    first = aggregate_validation_diagnostics(
        train, valid, users, labels, predictions, min_segment_rows=1
    )
    second = aggregate_validation_diagnostics(
        train, valid, users, labels, predictions, min_segment_rows=1
    )
    assert first == second
    encoded = json.dumps(first, sort_keys=True)
    assert "aggregate_only_no_row_labels_or_predictions" in encoded
    assert '"labels"' not in encoded
    assert '"predictions"' not in encoded
    assert first["models"]["baseline"]["overall"]["primary"] > 0.9
    assert "baseline__diverse" in first["model_diversity"]


def test_diagnostic_prompt_summary_is_bounded() -> None:
    train, valid = _frames()
    labels = np.asarray([1, 0, 1, 0, 1, 0, 1, 0])
    report = aggregate_validation_diagnostics(
        train,
        valid,
        valid["user_id"].to_numpy(),
        labels,
        {"baseline": np.linspace(0, 1, len(valid))},
        min_segment_rows=1,
    )
    summary = diagnostics_prompt_summary(report, max_chars=512)
    assert len(summary) <= 512
    assert summary.startswith("Trusted aggregate validation diagnostics")


def test_trusted_diagnostics_reject_row_misalignment() -> None:
    train, valid = _frames()
    try:
        aggregate_validation_diagnostics(
            train,
            valid,
            valid["user_id"].to_numpy(),
            np.asarray([0, 1]),
            {"baseline": np.asarray([0.1, 0.2])},
        )
    except ValueError as exc:
        assert "must align" in str(exc)
    else:
        raise AssertionError("misaligned diagnostics unexpectedly accepted")
