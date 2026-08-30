"""Trusted aggregate-only validation diagnostics for research planning.

This code belongs on the evaluator side of the boundary.  It may transiently
receive validation labels, but it returns no row-level labels, predictions, or
segment assignments.  Candidate programs should receive only the bounded text
summary produced here.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .eda import OUTCOME_COLUMNS, validation_static_frame
from .metrics import evaluate, rank_normalize_within_user

DIAGNOSTICS_VERSION = "kuairand-diagnostics-v1"


def _float(value: object) -> float:
    number = float(value)
    return round(number, 8) if np.isfinite(number) else 0.0


def _bucket_counts(values: pd.Series, edges: Sequence[float]) -> pd.Series:
    return (
        pd.cut(
            values,
            bins=[-np.inf, *edges, np.inf],
            labels=False,
            include_lowest=True,
        )
        .fillna(-1)
        .astype(int)
    )


def _fixed_segments(train: pd.DataFrame, valid: pd.DataFrame) -> dict[str, pd.Series]:
    user_counts = train["user_id"].value_counts(sort=False)
    video_counts = train["video_id"].value_counts(sort=False)
    history = valid["user_id"].map(user_counts).fillna(0)
    popularity = valid["video_id"].map(video_counts).fillna(0)
    segments: dict[str, pd.Series] = {
        "history_length": _bucket_counts(history, [0, 5, 20, 100]).map(
            {-1: "missing", 0: "0", 1: "1-5", 2: "6-20", 3: "21-100", 4: "101+"}
        ),
        "video_popularity": _bucket_counts(popularity, [0, 5, 20, 100]).map(
            {-1: "missing", 0: "unseen", 1: "1-5", 2: "6-20", 3: "21-100", 4: "101+"}
        ),
        "cold_user": (history == 0).map({True: "cold", False: "warm"}),
        "cold_video": (popularity == 0).map({True: "cold", False: "warm"}),
    }
    if "duration_ms" in valid:
        duration_seconds = pd.to_numeric(valid["duration_ms"], errors="coerce") / 1000
        segments["duration"] = _bucket_counts(duration_seconds, [7, 18, 32, 69]).map(
            {
                -1: "missing",
                0: "<=7s",
                1: "7-18s",
                2: "18-32s",
                3: "32-69s",
                4: ">69s",
            }
        )
    if "user_active_degree" in valid:
        segments["user_activity"] = valid["user_active_degree"].map(
            lambda value: "missing" if pd.isna(value) else f"level_{value}"
        )
    for name, column in (("author", "author_id"), ("tag", "primary_tag")):
        if column in train and column in valid:
            seen = valid[column].isin(pd.Index(train[column].dropna().unique()))
            segments[f"seen_{name}"] = seen.map({True: "seen", False: "unseen"})
    return segments


def _top5_summary(
    users: np.ndarray, labels: np.ndarray, scores: np.ndarray
) -> dict[str, float | int]:
    frame = pd.DataFrame({"user": users, "label": labels, "score": scores})
    captured = 0
    positives = 0
    top_rows = 0
    top_positives = 0
    users_with_positive = 0
    users_with_top5_hit = 0
    for _, group in frame.groupby("user", sort=False):
        ordered = group.sort_values("score", ascending=False, kind="stable")
        user_positives = int(group["label"].sum())
        top = ordered.iloc[:5]
        hits = int(top["label"].sum())
        positives += user_positives
        captured += hits
        top_rows += len(top)
        top_positives += hits
        if user_positives:
            users_with_positive += 1
            users_with_top5_hit += int(hits > 0)
    return {
        "positive_capture_rate_at_5": _float(captured / positives if positives else 0),
        "precision_at_5": _float(top_positives / top_rows if top_rows else 0),
        "positive_rows_missed_at_5": int(positives - captured),
        "users_with_positive": int(users_with_positive),
        "user_hit_rate_at_5": _float(
            users_with_top5_hit / users_with_positive if users_with_positive else 0
        ),
    }


def _top5_sets(users: np.ndarray, scores: np.ndarray) -> dict[object, set[int]]:
    frame = pd.DataFrame({"user": users, "score": scores, "row": np.arange(len(users))})
    return {
        user: set(
            group.sort_values("score", ascending=False, kind="stable")
            .iloc[:5]["row"]
            .astype(int)
        )
        for user, group in frame.groupby("user", sort=False)
    }


def _model_diversity(
    users: np.ndarray, predictions: Mapping[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    names = sorted(predictions)
    ranked = {
        name: rank_normalize_within_user(users, prediction)
        for name, prediction in predictions.items()
    }
    top_sets = {name: _top5_sets(users, value) for name, value in predictions.items()}
    output: dict[str, dict[str, float]] = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            left_values = predictions[left]
            right_values = predictions[right]
            raw_std = float(left_values.std() * right_values.std())
            rank_std = float(ranked[left].std() * ranked[right].std())
            raw_correlation = (
                float(np.corrcoef(left_values, right_values)[0, 1])
                if raw_std > 0
                else 0.0
            )
            rank_correlation = (
                float(np.corrcoef(ranked[left], ranked[right])[0, 1])
                if rank_std > 0
                else 0.0
            )
            jaccards = []
            for user in top_sets[left]:
                first = top_sets[left][user]
                second = top_sets[right][user]
                union = first | second
                jaccards.append(len(first & second) / len(union) if union else 1.0)
            output[f"{left}__{right}"] = {
                "pearson": _float(raw_correlation),
                "within_user_rank_correlation": _float(rank_correlation),
                "mean_absolute_rank_difference": _float(
                    np.abs(ranked[left] - ranked[right]).mean()
                ),
                "mean_top5_jaccard": _float(np.mean(jaccards) if jaccards else 0),
            }
    return output


def aggregate_validation_diagnostics(
    train: pd.DataFrame,
    validation_features: pd.DataFrame,
    user_ids: Sequence[object],
    labels: Sequence[int | float],
    predictions: Mapping[str, Sequence[int | float]],
    *,
    min_segment_rows: int = 250,
) -> dict[str, Any]:
    """Return trusted aggregate metrics without exposing row-level outcomes.

    Segment definitions use only training history and static validation fields.
    Validation labels are consumed solely by the immutable metric calculation
    and the aggregate top-five reducer.
    """

    valid = validation_static_frame(validation_features).reset_index(drop=True)
    if any(column in valid for column in OUTCOME_COLUMNS):
        raise RuntimeError("Validation outcome entered diagnostic segment features")
    users = np.asarray(user_ids)
    target = np.asarray(labels)
    if len(valid) != len(users) or len(users) != len(target):
        raise ValueError("validation features, users, and labels must align")
    if not np.isin(target, (0, 1)).all():
        raise ValueError("labels must be binary")
    if not predictions:
        raise ValueError("at least one prediction exit is required")
    arrays: dict[str, np.ndarray] = {}
    for name, values in predictions.items():
        array = np.asarray(values, dtype=np.float64)
        if len(array) != len(target) or not np.isfinite(array).all():
            raise ValueError(f"prediction exit {name!r} is unaligned or non-finite")
        arrays[str(name)] = array

    segments = _fixed_segments(train, valid)
    models: dict[str, Any] = {}
    for name in sorted(arrays):
        scores = arrays[name]
        by_segment: dict[str, dict[str, Any]] = {}
        for dimension, assignments in segments.items():
            groups: dict[str, Any] = {}
            for group in sorted(assignments.dropna().astype(str).unique()):
                mask = assignments.astype(str).to_numpy() == group
                if int(mask.sum()) < min_segment_rows:
                    continue
                groups[group] = evaluate(users[mask], target[mask], scores[mask])
            if groups:
                by_segment[dimension] = groups
        models[name] = {
            "overall": evaluate(users, target, scores),
            "segments": by_segment,
            "top5_errors": _top5_summary(users, target, scores),
        }

    report: dict[str, Any] = {
        "diagnostics_version": DIAGNOSTICS_VERSION,
        "privacy_boundary": "aggregate_only_no_row_labels_or_predictions",
        "models": models,
        "model_diversity": _model_diversity(users, arrays),
    }
    canonical = json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    report["diagnostics_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return report


def diagnostics_prompt_summary(
    diagnostics: Mapping[str, Any], max_chars: int = 3000
) -> str:
    """Create a bounded aggregate-only summary for research prompts."""

    if max_chars < 256:
        raise ValueError("max_chars must be at least 256")
    lines = ["Trusted aggregate validation diagnostics:"]
    for name, model in sorted(diagnostics.get("models", {}).items()):
        overall = model["overall"]
        top5 = model["top5_errors"]
        lines.append(
            f"- {name}: GAUC={overall['GAUC']:.6f}, nDCG@5={overall['nDCG@5']:.6f}, primary={overall['primary']:.6f}, top5 positive capture={top5['positive_capture_rate_at_5']:.4f}."
        )
        for dimension, groups in sorted(model.get("segments", {}).items()):
            weakest = min(groups.items(), key=lambda item: item[1]["primary"])
            lines.append(
                f"  Weakest {dimension}: {weakest[0]} primary={weakest[1]['primary']:.6f} (rows={weakest[1]['rows']})."
            )
    for pair, values in sorted(diagnostics.get("model_diversity", {}).items()):
        lines.append(
            f"- Diversity {pair}: within-user rank corr={values['within_user_rank_correlation']:.4f}, top5 Jaccard={values['mean_top5_jaccard']:.4f}."
        )
    lines.append(
        "Tie the next atomic hypothesis to one EDA observation and one aggregate weakness; do not request row-level labels or predictions."
    )
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    marker = "\n[diagnostic summary truncated at configured bound]"
    return text[: max(0, max_chars - len(marker))].rstrip() + marker
