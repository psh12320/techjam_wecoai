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
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(
        args.data_dir / "log_standard_4_08_to_4_21_pure.csv",
        usecols=TRAIN_COLUMNS,
    )
    later = pd.read_csv(
        args.data_dir / "log_standard_4_22_to_5_08_pure.csv",
        usecols=VALID_COLUMNS,
    )
    valid = later.loc[later["date"] <= VALID_END].copy()
    del later
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
        args.output_dir / "validation_index.npz",
        row_id=np.arange(len(valid), dtype=np.int32),
        user_id=valid["user_id"].to_numpy(dtype=np.int32),
        video_id=valid["video_id"].to_numpy(dtype=np.int32),
        long_view=valid["long_view"].to_numpy(dtype=np.int8),
    )
    manifest = {
        "train_rows": len(train),
        "valid_rows": len(valid),
        "last_allowed_date": VALID_END,
        "hidden_test_present": False,
        "excluded_for_leakage_risk": ["video_features_statistic_pure.csv"],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
