"""Reproduce the organizer FM baseline without reading the held-out test labels.

This script intentionally reuses the starter kit's FM and exact evaluator while
loading only the official train and public-validation date ranges.  The hidden
test rows are not loaded, evaluated, or exposed to the experiment loop.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "kuairand-starter-kit"
sys.path.insert(0, str(STARTER))

import baseline as organizer_baseline  # noqa: E402
from data import SPLITS, encode  # noqa: E402
from evaluate import evaluate  # noqa: E402

EXPECTED_ROWS = {"train": 1_141_112, "valid": 124_909}
EXPECTED_VALID = {"GAUC": 0.6674, "nDCG@5": 0.5357, "primary": 0.6016}


def load_train_valid_only(data_dir: Path) -> dict[str, list[tuple]]:
    """Load only dates allowed during development.

    The starter kit's ``data.load`` also parses the nominal test dates because
    it is a self-contained reference implementation.  This loader preserves
    its tuple schema while refusing dates after the public validation window.
    """

    authors: dict[str, str] = {}
    with (data_dir / "video_features_basic_pure.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            authors[row["video_id"]] = row["author_id"]

    output = {"train": [], "valid": [], "test": []}
    valid_last_day = SPLITS["valid"][1]
    log_names = (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    )
    for name in log_names:
        with (data_dir / name).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                date = int(row["date"])
                if date > valid_last_day:
                    continue
                record = (
                    date,
                    row["user_id"],
                    row["video_id"],
                    authors.get(row["video_id"], "UNK"),
                    row["tab"],
                    float(row["duration_ms"]),
                    1 if row["long_view"] != "0" else 0,
                )
                for split_name in ("train", "valid"):
                    lo, hi = SPLITS[split_name]
                    if lo <= date <= hi:
                        output[split_name].append(record)
                        break

    actual = {name: len(output[name]) for name in EXPECTED_ROWS}
    if actual != EXPECTED_ROWS:
        raise RuntimeError(f"Unexpected official split sizes: {actual}")
    return output


def run_seed(
    splits: dict[str, list[tuple]],
    *,
    seed: int,
    k: int,
    learning_rate: float,
    max_epochs: int,
    batch_size: int,
    patience: int,
) -> dict[str, float | int]:
    encoded, dimension = encode(splits)
    train_x, train_y, _ = encoded["train"]
    valid_x, valid_y, valid_users = encoded["valid"]
    model = organizer_baseline.FM(dimension, k=k, lr=learning_rate, seed=seed)
    rng = np.random.default_rng(seed)
    best_primary = -1.0
    best_state = None
    bad_epochs = 0
    best_epoch = 0

    for epoch in range(1, max_epochs + 1):
        started = time.perf_counter()
        indices = rng.permutation(len(train_y))
        losses = []
        for offset in range(0, len(indices), batch_size):
            batch = indices[offset : offset + batch_size]
            losses.append(model.step(train_x[batch], train_y[batch]))
        metrics = evaluate(valid_users, valid_y, model.predict(valid_x))
        print(
            f"seed={seed} epoch={epoch:02d} loss={np.mean(losses):.5f} "
            f"GAUC={metrics['GAUC']:.6f} nDCG@5={metrics['nDCG@5']:.6f} "
            f"primary={metrics['primary']:.6f} "
            f"seconds={time.perf_counter() - started:.2f}",
            flush=True,
        )
        if metrics["primary"] > best_primary + 1e-5:
            best_primary = metrics["primary"]
            best_epoch = epoch
            bad_epochs = 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("FM training produced no checkpoint")
    model.V, model.W, model.b = best_state
    final = evaluate(valid_users, valid_y, model.predict(valid_x))
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "GAUC": float(final["GAUC"]),
        "nDCG@5": float(final["nDCG@5"]),
        "primary": float(final["primary"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=STARTER / "KuaiRand-Pure" / "data",
    )
    parser.add_argument("--seeds", default="0", help="Comma-separated integer seeds")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.002,
        help="Allowed absolute difference from the published validation primary",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    started = time.perf_counter()
    splits = load_train_valid_only(args.data_dir)
    runs = [
        run_seed(
            splits,
            seed=seed,
            k=args.k,
            learning_rate=args.learning_rate,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            patience=args.patience,
        )
        for seed in seeds
    ]
    mean_metrics = {
        name: statistics.mean(float(run[name]) for run in runs)
        for name in ("GAUC", "nDCG@5", "primary")
    }
    reproduced = (
        abs(mean_metrics["primary"] - EXPECTED_VALID["primary"]) <= args.tolerance
    )
    result = {
        "status": "reproduced" if reproduced else "outside_tolerance",
        "dataset": "KuaiRand-Pure",
        "split": "valid",
        "test_labels_accessed": False,
        "config": {
            "model": "FM",
            "k": args.k,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "seeds": seeds,
        },
        "published": EXPECTED_VALID,
        "mean": mean_metrics,
        "runs": runs,
        "wall_seconds": time.perf_counter() - started,
    }
    print("TECHJAM_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    return 0 if reproduced else 2


if __name__ == "__main__":
    raise SystemExit(main())
