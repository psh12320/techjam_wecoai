"""Create a development-only AIDE input directory with no hidden-test dates."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VALID_END = 20220428
TRAIN_COLUMNS = [
    "user_id",
    "video_id",
    "date",
    "hourmin",
    "time_ms",
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "play_time_ms",
    "duration_ms",
    "profile_stay_time",
    "comment_stay_time",
    "is_profile_enter",
    "tab",
]
VALID_COLUMNS = [
    "user_id",
    "video_id",
    "date",
    "hourmin",
    "time_ms",
    "duration_ms",
    "tab",
    "long_view",
]


def _read_rows_through_date(
    path: Path, columns: list[str], max_date: int
) -> tuple[pd.DataFrame, int]:
    """Read outcomes only for public rows selected by a date-only prepass."""

    dates = pd.read_csv(path, usecols=["date"], dtype={"date": "int32"})["date"]
    keep_data_rows = np.flatnonzero(dates.to_numpy() <= max_date)
    keep_file_lines = set((keep_data_rows + 1).tolist())
    frame = pd.read_csv(
        path,
        usecols=columns,
        skiprows=lambda line: line > 0 and line not in keep_file_lines,
    )
    return frame, len(dates)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "kuairand-starter-kit" / "KuaiRand-Pure" / "data",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "challenge" / "agent_data"
    )
    parser.add_argument(
        "--evaluator-dir",
        type=Path,
        default=ROOT / "challenge" / "private" / "evaluator",
        help="Private evaluator-only artifacts; never copied into an AIDE workspace.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.evaluator_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(
        args.data_dir / "log_standard_4_08_to_4_21_pure.csv",
        usecols=TRAIN_COLUMNS,
    )
    valid, later_rows = _read_rows_through_date(
        args.data_dir / "log_standard_4_22_to_5_08_pure.csv",
        VALID_COLUMNS,
        VALID_END,
    )
    if len(train) != 1_141_112 or len(valid) != 124_909:
        raise RuntimeError("Official development split sizes changed")
    if int(valid["date"].max()) > VALID_END:
        raise RuntimeError("Hidden-test date entered AIDE input")

    train_path = args.output_dir / "train.csv"
    valid_path = args.output_dir / "valid.csv"
    train.to_csv(train_path, index=False)
    valid.to_csv(valid_path, index=False)
    for source, alias in (
        (train_path, args.output_dir / "log_standard_4_08_to_4_21_pure.csv"),
        (valid_path, args.output_dir / "log_standard_4_22_to_5_08_pure.csv"),
    ):
        alias.unlink(missing_ok=True)
        os.link(source, alias)
    for name in ("video_features_basic_pure.csv", "user_features_pure.csv"):
        shutil.copy2(args.data_dir / name, args.output_dir / name)
    for name in ("baseline.py", "data.py", "evaluate.py"):
        shutil.copy2(ROOT / "kuairand-starter-kit" / name, args.output_dir / name)
    np.savez_compressed(
        args.evaluator_dir / "validation_index.npz",
        row_id=np.arange(len(valid), dtype=np.int32),
        user_id=valid["user_id"].to_numpy(dtype=np.int32),
        video_id=valid["video_id"].to_numpy(dtype=np.int32),
        long_view=valid["long_view"].to_numpy(dtype=np.int8),
    )
    # Remove the legacy location if an older preparation run placed evaluator
    # labels inside the directory copied to generated candidates.
    (args.output_dir / "validation_index.npz").unlink(missing_ok=True)
    shutil.rmtree(args.output_dir / "__pycache__", ignore_errors=True)
    manifest = {
        "train_rows": len(train),
        "valid_rows": len(valid),
        "last_allowed_date": VALID_END,
        "hidden_test_present": False,
        "later_file_rows_seen_by_date_only_prepass": later_rows,
        "hidden_outcome_rows_materialized": 0,
        "train_columns": TRAIN_COLUMNS,
        "valid_columns": VALID_COLUMNS,
        "excluded_for_leakage_risk": ["video_features_statistic_pure.csv"],
        "evaluator_artifacts_in_candidate_input": 0,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
