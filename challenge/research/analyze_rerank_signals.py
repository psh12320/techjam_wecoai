"""Measure train-derived reranking signals against an AIDE validation node.

This is prompt-development tooling, not candidate code.  It reads one legal
AIDE prediction artifact, builds every statistic from the public training
split, and uses the public validation split only for deterministic feedback.
The resulting aggregate findings may be added to experiment memory; the
prediction artifact itself must never be exposed to a generated candidate.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "challenge" / "agent_data"
sys.path.insert(0, str(DATA_DIR))

from evaluate import evaluate  # noqa: E402


def within_user_zscore(values: np.ndarray, users: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"user_id": users, "value": values})
    grouped = frame.groupby("user_id", sort=False)["value"]
    mean = grouped.transform("mean").to_numpy(dtype=np.float64)
    std = grouped.transform("std").fillna(0.0).to_numpy(dtype=np.float64)
    return (values - mean) / np.maximum(std, 1e-6)


def fast_evaluate(
    users: np.ndarray, labels: np.ndarray, scores: np.ndarray, k: int = 5
) -> dict[str, float]:
    """Vectorized sort plus exact per-user reductions for bounded weight probes."""

    original = np.arange(len(scores), dtype=np.int64)
    order = np.lexsort((original, -scores, users))
    sorted_users = users[order]
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    stops = np.r_[starts[1:], len(sorted_users)]
    discounts = 1.0 / np.log2(np.arange(k, dtype=np.float64) + 2.0)
    ideal_prefix = np.cumsum(discounts)
    gnum = 0.0
    gden = 0.0
    ndcg_sum = 0.0
    for start, stop in zip(starts, stops):
        group_labels = sorted_labels[start:stop]
        group_scores = sorted_scores[start:stop]
        npos = int(group_labels.sum())
        nneg = len(group_labels) - npos
        if 0 < npos < len(group_labels):
            wins = 0.0
            neg_remaining = nneg
            cursor = 0
            while cursor < len(group_labels):
                end = cursor + 1
                while end < len(group_labels) and group_scores[end] == group_scores[cursor]:
                    end += 1
                block = group_labels[cursor:end]
                block_pos = int(block.sum())
                block_neg = len(block) - block_pos
                wins += block_pos * (neg_remaining - block_neg)
                wins += 0.5 * block_pos * block_neg
                neg_remaining -= block_neg
                cursor = end
            gnum += npos * wins / (npos * nneg)
            gden += npos
        if npos:
            top = group_labels[:k]
            dcg = float(np.dot(top, discounts[: len(top)]))
            idcg = float(ideal_prefix[min(npos, k) - 1])
            ndcg_sum += dcg / idcg
    gauc = gnum / gden
    ndcg = ndcg_sum / len(starts)
    return {"GAUC": gauc, "nDCG@5": ndcg, "primary": (gauc + ndcg) / 2.0}


def smoothed_rate(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    keys: list[str],
    *,
    prior: float,
    fallback: float,
) -> np.ndarray:
    stats = train.groupby(keys, sort=False)["long_view"].agg(["sum", "count"])
    stats["rate"] = (stats["sum"] + prior * fallback) / (
        stats["count"] + prior
    )
    if len(keys) == 1:
        return (
            valid[keys[0]]
            .map(stats["rate"])
            .fillna(fallback)
            .to_numpy(dtype=np.float64)
        )
    index = pd.MultiIndex.from_frame(valid[keys])
    return stats["rate"].reindex(index).fillna(fallback).to_numpy(dtype=np.float64)


def add_features(frame: pd.DataFrame, video: pd.DataFrame) -> pd.DataFrame:
    result = frame.merge(video, on="video_id", how="left", sort=False)
    duration = pd.to_numeric(result["duration_ms"], errors="coerce").fillna(0)
    result["duration_bucket"] = np.clip(
        np.floor(np.log1p(duration.to_numpy(dtype=np.float64)) / 0.65), 0, 31
    ).astype(np.int16)
    result["tab_duration"] = (
        result["tab"].fillna(-1).astype(str)
        + "|"
        + result["duration_bucket"].astype(str)
    )
    tag = result["tag"].fillna("").astype(str).str.split(",").str[0]
    result["primary_tag"] = tag.replace("", "-1")
    return result


def load_predictions(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float64, copy=False)
    frame = pd.read_csv(path)
    return frame["score"].to_numpy(dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction", type=Path)
    args = parser.parse_args()

    train = pd.read_csv(
        DATA_DIR / "train.csv",
        usecols=["user_id", "video_id", "duration_ms", "tab", "long_view"],
    )
    valid = pd.read_csv(
        DATA_DIR / "valid.csv",
        usecols=["user_id", "video_id", "duration_ms", "tab", "long_view"],
    )
    video = pd.read_csv(
        DATA_DIR / "video_features_basic_pure.csv",
        usecols=["video_id", "author_id", "music_id", "tag"],
    )
    train = add_features(train, video)
    valid = add_features(valid, video)

    scores = load_predictions(args.prediction)
    if len(scores) != len(valid) or not np.isfinite(scores).all():
        raise ValueError("prediction length/finite check failed")
    users = valid["user_id"].to_numpy()
    labels = valid["long_view"].to_numpy(dtype=np.int8)
    base = np.log(np.clip(scores, 1e-6, 1 - 1e-6)) - np.log(
        np.clip(1 - scores, 1e-6, 1 - 1e-6)
    )
    base_z = within_user_zscore(base, users)
    base_metric = evaluate(users, labels, base_z)
    fast_base_metric = fast_evaluate(users, labels, base_z)
    if any(abs(base_metric[key] - fast_base_metric[key]) > 1e-12 for key in ("GAUC", "nDCG@5", "primary")):
        raise AssertionError((base_metric, fast_base_metric))
    print("base", base_metric)

    global_rate = float(train["long_view"].mean())
    user_rate = smoothed_rate(
        train, valid, ["user_id"], prior=20.0, fallback=global_rate
    )
    signals = {
        "video_rate": smoothed_rate(
            train, valid, ["video_id"], prior=30.0, fallback=global_rate
        ),
        "author_rate": smoothed_rate(
            train, valid, ["author_id"], prior=50.0, fallback=global_rate
        ),
        "tag_rate": smoothed_rate(
            train, valid, ["primary_tag"], prior=100.0, fallback=global_rate
        ),
        "music_rate": smoothed_rate(
            train, valid, ["music_id"], prior=50.0, fallback=global_rate
        ),
        "duration_rate": smoothed_rate(
            train, valid, ["duration_bucket"], prior=200.0, fallback=global_rate
        ),
        "tab_duration_rate": smoothed_rate(
            train, valid, ["tab_duration"], prior=100.0, fallback=global_rate
        ),
        "user_duration_residual": smoothed_rate(
            train,
            valid,
            ["user_id", "duration_bucket"],
            prior=10.0,
            fallback=global_rate,
        )
        - user_rate,
        "user_author_residual": smoothed_rate(
            train,
            valid,
            ["user_id", "author_id"],
            prior=8.0,
            fallback=global_rate,
        )
        - user_rate,
    }
    weights = (-0.20, -0.10, -0.05, 0.025, 0.05, 0.10, 0.20, 0.30)
    normalized_signals: dict[str, np.ndarray] = {}
    for name, signal in signals.items():
        signal_z = within_user_zscore(signal, users)
        normalized_signals[name] = signal_z
        best = None
        for weight in weights:
            metric = evaluate(users, labels, base_z + weight * signal_z)
            row = (metric["primary"], weight, metric)
            if best is None or row[0] > best[0]:
                best = row
        assert best is not None
        print(name, "weight", best[1], best[2])

    # A deliberately tiny two-dimensional probe: the tag prior can help top-5
    # ranking while the sparse user-author residual has shown complementary
    # GAUC movement.  This remains public-validation prompt research, not a
    # candidate-side weight search.
    tag_weights = (-0.015, -0.020, -0.025, -0.030, -0.035, -0.040)
    author_weights = (-0.025, -0.050, -0.075, -0.100, -0.150)
    feasible = []
    best_combo = None
    for tag_weight in tag_weights:
        for author_weight in author_weights:
            combined = (
                base_z
                + tag_weight * normalized_signals["tag_rate"]
                + author_weight * normalized_signals["user_author_residual"]
            )
            metric = evaluate(users, labels, combined)
            row = (metric["primary"], tag_weight, author_weight, metric)
            if best_combo is None or row[0] > best_combo[0]:
                best_combo = row
            if metric["GAUC"] > 0.6710518008586268 and metric["nDCG@5"] > 0.5380142516919405:
                feasible.append(row)
    print("best_tag_author_combo", best_combo)
    print("champion_feasible_tag_author_combos", sorted(feasible, reverse=True)[:10])

    # Fixed-seed prompt-development search around the conservative two-signal
    # region.  All component values remain train-derived; validation labels
    # are used only by the public evaluator.
    search_names = (
        "tag_rate",
        "user_author_residual",
        "video_rate",
        "duration_rate",
        "music_rate",
        "user_duration_residual",
        "tab_duration_rate",
    )
    center = np.array([-0.02, -0.075, 0.0, 0.0, 0.0, 0.0, 0.0])
    scale = np.array([0.025, 0.06, 0.04, 0.04, 0.04, 0.04, 0.04])
    matrix = np.stack([normalized_signals[name] for name in search_names], axis=1)
    rng = np.random.default_rng(20260830)
    candidates = [center]
    candidates.extend(center + rng.normal(size=(600, len(center))) * scale)
    random_best = []
    for weights_row in candidates:
        combined = base_z + matrix @ weights_row
        metric = fast_evaluate(users, labels, combined)
        if (
            metric["GAUC"] > 0.6710518008586268
            and metric["nDCG@5"] > 0.5380142516919405
        ):
            random_best.append((metric["primary"], weights_row.tolist(), metric))
    random_best.sort(reverse=True, key=lambda row: row[0])
    checked = []
    for _, weights_row, _ in random_best[:10]:
        combined = base_z + matrix @ np.asarray(weights_row)
        metric = evaluate(users, labels, combined)
        checked.append((metric["primary"], weights_row, metric))
    checked.sort(reverse=True, key=lambda row: row[0])
    print("best_multisignal_combos", checked)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
