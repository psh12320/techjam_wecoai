from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from challenge.techjam_recsys.data import DatasetSplits
from challenge.techjam_recsys.eda import (
    OUTCOME_COLUMNS,
    analyze_splits,
    validation_static_frame,
    write_eda_artifacts,
)


def _splits(valid_labels: np.ndarray | None = None) -> DatasetSplits:
    train = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 2, 3],
            "video_id": [10, 11, 10, 12, 12, 13],
            "author_id": [100, 101, 100, 102, 102, 103],
            "primary_tag": [5, 6, 5, 7, 7, 8],
            "music_id": [20, 21, 20, 22, 22, 23],
            "date": [20220408, 20220408, 20220409, 20220409, 20220410, 20220410],
            "hourmin": [900, 1000, 1100, 1200, 1300, 1400],
            "hour": [9, 10, 11, 12, 13, 14],
            "weekday": [4, 4, 5, 5, 6, 6],
            "duration_ms": [10000, 20000, 10000, 30000, 30000, 40000],
            "play_time_ms": [5000, 21000, 10000, 5000, 40000, 1000],
            "is_click": [1, 1, 1, 0, 1, 0],
            "is_like": [0, 1, 0, 0, 1, 0],
            "is_follow": [0, 0, 0, 0, 0, 0],
            "is_comment": [0, 0, 0, 0, 0, 0],
            "is_forward": [0, 0, 0, 0, 0, 0],
            "is_hate": [0, 0, 0, 1, 0, 0],
            "long_view": [0, 1, 1, 0, 1, 0],
        }
    )
    valid = pd.DataFrame(
        {
            "user_id": [1, 2, 4, 4],
            "video_id": [10, 14, 10, 15],
            "author_id": [100, 104, 100, 105],
            "primary_tag": [5, 9, 5, 10],
            "music_id": [20, 24, 20, 25],
            "date": [20220422] * 4,
            "hourmin": [900, 1800, 900, 1800],
            "hour": [9, 18, 9, 18],
            "weekday": [4] * 4,
            "duration_ms": [10000, 50000, 10000, 60000],
            "play_time_ms": [999999, 999999, 999999, 999999],
            "is_click": [1, 1, 1, 1],
            "long_view": [1, 1, 1, 1],
        }
    )
    labels = np.asarray([1, 1, 1, 1]) if valid_labels is None else valid_labels
    return DatasetSplits(
        train=train,
        valid=valid,
        valid_users=valid["user_id"].to_numpy(),
        valid_labels=labels,
    )


def test_validation_static_frame_excludes_all_outcomes() -> None:
    frame = validation_static_frame(_splits().valid)
    assert not OUTCOME_COLUMNS.intersection(frame.columns)
    assert {"user_id", "video_id", "duration_ms"}.issubset(frame.columns)


def test_eda_is_deterministic_and_independent_of_validation_labels() -> None:
    first_splits = _splits(np.asarray([0, 0, 0, 0]))
    second_splits = _splits(np.asarray([1, 1, 1, 1]))
    second_splits.valid.loc[:, "long_view"] = [0, 0, 0, 0]
    second_splits.valid.loc[:, "is_click"] = [0, 0, 0, 0]
    second_splits.valid.loc[:, "play_time_ms"] = [0, 0, 0, 0]
    first = analyze_splits(first_splits)
    second = analyze_splits(second_splits)
    assert first.stable_hash == second.stable_hash
    assert first.report == second.report
    assert first.report["label_imbalance"]["source"] == "train_only"
    assert first.report["overlap"]["user_id"]["validation_cold_row_rate"] == 0.5
    assert (
        first.report["candidate_aware_support"]["user_video"][
            "validation_prior_pair_support_rate"
        ]
        == 0.25
    )


def test_eda_writes_bounded_machine_and_human_artifacts(tmp_path: Path) -> None:
    report = analyze_splits(_splits())
    paths = write_eda_artifacts(report, tmp_path, max_prompt_chars=512)
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["report_sha256"] == report.stable_hash
    assert len(payload["prompt_summary"]) <= 512
    assert "KuaiRand Autonomous EDA" in paths["markdown"].read_text(encoding="utf-8")
    assert paths["prompt"].read_text(encoding="utf-8").startswith("EDA evidence")
