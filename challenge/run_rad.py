"""Relative Advantage Debiasing auxiliaries for KuaiRand-Pure."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from challenge.techjam_recsys.data import load_development_splits
from challenge.techjam_recsys.metrics import evaluate

CATEGORICAL = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_bucket",
    "duration_rad_bin",
    "hour",
    "weekday",
    "music_id",
    "music_type",
    "video_type",
    "upload_type",
    "primary_tag",
]
NUMERIC = ["log_duration", "upload_age_days", "aspect_ratio"]


def midrank_cdf(frame, group_columns: list[str], value_column: str) -> np.ndarray:
    grouped = frame.groupby(group_columns, sort=False, observed=True)[value_column]
    rank = grouped.rank(method="average").to_numpy(dtype=np.float32)
    size = grouped.transform("size").to_numpy(dtype=np.float32)
    return ((rank - 0.5) / size).astype(np.float32)


def add_duration_bins(train, valid, bins: int) -> None:
    values = train["duration_ms"].to_numpy(dtype=np.float64)
    edges = np.unique(
        np.quantile(values[np.isfinite(values)], np.linspace(0, 1, bins + 1)[1:-1])
    )
    for frame in (train, valid):
        raw = frame["duration_ms"].to_numpy(dtype=np.float64)
        frame["duration_rad_bin"] = np.searchsorted(
            edges, np.nan_to_num(raw, nan=-1), side="right"
        ).astype(np.int16)


def train_model(train, valid, target: np.ndarray, seed: int, rounds: int, jobs: int):
    features = CATEGORICAL + NUMERIC
    fit_rows = train["date"].to_numpy() <= 20220418
    tune_rows = ~fit_rows
    model = lgb.LGBMRegressor(
        objective="regression_l2",
        n_estimators=rounds,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=100,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=jobs,
        verbosity=-1,
    )
    model.fit(
        train.loc[fit_rows, features],
        target[fit_rows],
        categorical_feature=CATEGORICAL,
        eval_set=[(train.loc[tune_rows, features], target[tune_rows])],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)],
    )
    best_iteration = int(model.best_iteration_ or rounds)
    final = lgb.LGBMRegressor(**{**model.get_params(), "n_estimators": best_iteration})
    final.fit(
        train[features],
        target,
        categorical_feature=CATEGORICAL,
        callbacks=[lgb.log_evaluation(0)],
    )
    return final.predict(valid[features]), best_iteration


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-bins", type=int, choices=(4, 8), default=4)
    parser.add_argument("--rounds", type=int, default=250)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--models",
        default="v,u",
        help="Comma-separated subset of v (video), u (user-duration), d (duration)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("kuairand-starter-kit/KuaiRand-Pure/data"),
    )
    parser.add_argument("--run-dir", type=Path, default=Path("challenge/runs/rad"))
    args = parser.parse_args()
    started = time.perf_counter()
    splits = load_development_splits(args.data_dir)
    add_duration_bins(splits.train, splits.valid, args.duration_bins)
    requested = {value.strip() for value in args.models.split(",") if value.strip()}
    unknown = requested.difference({"v", "u", "d"})
    if unknown:
        raise ValueError(f"Unknown RAD model names: {sorted(unknown)}")
    target_specs = {
        "v": (["video_id"], args.seed),
        "u": (["user_id", "duration_rad_bin"], args.seed + 1),
        "d": (["duration_rad_bin"], args.seed + 2),
    }
    targets = {}
    predictions = {}
    iterations = {}
    metrics = {}
    for name in sorted(requested):
        groups, seed = target_specs[name]
        targets[name] = midrank_cdf(splits.train, groups, "play_time_ms")
        predictions[name], iterations[name] = train_model(
            splits.train,
            splits.valid,
            targets[name],
            seed,
            args.rounds,
            args.jobs,
        )
        metrics[f"rad_{name}"] = evaluate(
            splits.valid_users, splits.valid_labels, predictions[name]
        )
    if {"u", "v"}.issubset(predictions):
        prediction_uv = 0.5 * predictions["v"] + 0.5 * predictions["u"]
        metrics["rad_uv"] = evaluate(
            splits.valid_users, splits.valid_labels, prediction_uv
        )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    stem = f"rad_d{args.duration_bins}_seed{args.seed}"
    for name, prediction in predictions.items():
        np.save(args.run_dir / f"{stem}_{name}_valid.npy", prediction)
    if {"u", "v"}.issubset(predictions):
        np.save(args.run_dir / f"{stem}_uv_valid.npy", prediction_uv)
    result = {
        "method": "relative_advantage_debiasing",
        "duration_bins": args.duration_bins,
        "seed": args.seed,
        "models": sorted(requested),
        "best_iterations": {
            f"rad_{name}": iteration for name, iteration in iterations.items()
        },
        "target_summary": {
            f"rad_{name}_mean": float(target.mean()) for name, target in targets.items()
        },
        "metrics": metrics,
        "wall_seconds": time.perf_counter() - started,
    }
    (args.run_dir / f"{stem}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("TECHJAM_RESULT=" + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
