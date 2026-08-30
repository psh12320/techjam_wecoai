"""Deterministic, leakage-safe EDA for the KuaiRand development split.

The public validation frame loaded by :mod:`data` contains outcomes for the
private evaluator.  This module deliberately removes every serving-time
outcome before inspecting validation data and never reads ``valid_labels``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .data import DatasetSplits

REPORT_VERSION = "kuairand-eda-v1"

# These values are observed after an impression and are unavailable when a
# candidate must be scored.  Validation EDA is restricted to the complement.
OUTCOME_COLUMNS = frozenset(
    {
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "long_view",
        "play_time_ms",
        "profile_stay_time",
        "comment_stay_time",
        "is_profile_enter",
    }
)

ENTITY_COLUMNS = ("user_id", "video_id", "author_id", "primary_tag", "music_id")
FEEDBACK_COLUMNS = (
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
)


def _float(value: object, digits: int = 8) -> float:
    number = float(value)
    if not np.isfinite(number):
        return 0.0
    return round(number, digits)


def _rate(mask: pd.Series | np.ndarray) -> float:
    values = np.asarray(mask, dtype=bool)
    return _float(values.mean()) if len(values) else 0.0


def _numeric_summary(values: pd.Series | np.ndarray) -> dict[str, float | int]:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0}
    quantiles = np.quantile(array, [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    return {
        "count": int(len(array)),
        "mean": _float(array.mean()),
        "std": _float(array.std()),
        "min": _float(quantiles[0]),
        "p25": _float(quantiles[1]),
        "p50": _float(quantiles[2]),
        "p75": _float(quantiles[3]),
        "p90": _float(quantiles[4]),
        "p95": _float(quantiles[5]),
        "p99": _float(quantiles[6]),
        "max": _float(quantiles[7]),
    }


def _frequency_summary(series: pd.Series) -> dict[str, float | int]:
    counts = series.value_counts(dropna=False, sort=False)
    if not len(counts):
        return {"entities": 0, "rows": 0}
    descending = counts.sort_values(ascending=False, kind="stable")
    total = int(descending.sum())
    summary = _numeric_summary(descending.to_numpy())
    return {
        "entities": int(len(descending)),
        "rows": total,
        "singleton_entity_rate": _rate(descending.to_numpy() == 1),
        "singleton_row_rate": _float((descending.to_numpy() == 1).sum() / total),
        "top1_row_share": _float(descending.iloc[0] / total),
        "top10_row_share": _float(descending.iloc[:10].sum() / total),
        "count_distribution": summary,
    }


def _schema(frame: pd.DataFrame) -> dict[str, dict[str, float | int | str]]:
    rows = max(len(frame), 1)
    return {
        str(column): {
            "dtype": str(frame[column].dtype),
            "missing_count": int(frame[column].isna().sum()),
            "missing_rate": _float(frame[column].isna().sum() / rows),
            "cardinality": int(frame[column].nunique(dropna=True)),
        }
        for column in frame.columns
    }


def _overlap(train: pd.Series, valid: pd.Series) -> dict[str, float | int]:
    train_values = pd.Index(train.dropna().unique())
    valid_values = pd.Index(valid.dropna().unique())
    valid_seen = valid.notna() & valid.isin(train_values)
    seen_entities = valid_values.isin(train_values)
    return {
        "train_entities": int(len(train_values)),
        "validation_entities": int(len(valid_values)),
        "shared_entities": int(seen_entities.sum()),
        "validation_seen_row_rate": _rate(valid_seen),
        "validation_cold_row_rate": _rate(~valid_seen),
        "validation_cold_entity_rate": _float(
            1.0 - seen_entities.mean() if len(seen_entities) else 0.0
        ),
    }


def _total_variation(train: pd.Series, valid: pd.Series) -> float:
    left = train.value_counts(normalize=True, dropna=False)
    right = valid.value_counts(normalize=True, dropna=False)
    index = left.index.union(right.index)
    return _float(
        0.5
        * np.abs(
            left.reindex(index, fill_value=0).to_numpy()
            - right.reindex(index, fill_value=0).to_numpy()
        ).sum()
    )


def _pair_support(
    train: pd.DataFrame, valid: pd.DataFrame, entity: str
) -> dict[str, float | int]:
    required = ["user_id", entity]
    if any(column not in train or column not in valid for column in required):
        return {"available": False}
    train_pairs = pd.MultiIndex.from_frame(train[required].drop_duplicates())
    valid_pairs = pd.MultiIndex.from_frame(valid[required])
    support = valid_pairs.isin(train_pairs)
    train_duplicate_rows = train[required].duplicated(keep="first")
    train_repeated_rows = train[required].duplicated(keep=False)
    return {
        "available": True,
        "unique_train_pairs": int(len(train_pairs)),
        "validation_prior_pair_support_rate": _rate(support),
        "train_repeat_after_first_row_rate": _rate(train_duplicate_rows),
        "train_rows_in_repeated_pairs_rate": _rate(train_repeated_rows),
    }


def _date_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if "date" not in frame or frame.empty:
        return {"available": False}
    counts = frame["date"].value_counts(sort=False).sort_index()
    return {
        "available": True,
        "min": int(frame["date"].min()),
        "max": int(frame["date"].max()),
        "days": int(frame["date"].nunique()),
        "rows_per_day": {str(key): int(value) for key, value in counts.items()},
    }


def validation_static_frame(valid: pd.DataFrame) -> pd.DataFrame:
    """Return a view containing only columns legal at validation scoring time."""

    permitted = [column for column in valid.columns if column not in OUTCOME_COLUMNS]
    return valid.loc[:, permitted]


def _build_observations(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    imbalance = report["label_imbalance"]["long_view_positive_rate"]
    observations.append(
        {
            "id": "label_imbalance",
            "finding": f"Training long-view positive rate is {imbalance:.4f}.",
            "implication": "Use calibrated pointwise supervision and evaluate ranking-aware refinements rather than accuracy.",
            "suggested_families": [
                "rich_fm",
                "duration_auxiliary",
                "ranknet_lambdaloss",
            ],
        }
    )

    overlap = report["overlap"]
    cold_user = overlap.get("user_id", {}).get("validation_cold_row_rate", 0.0)
    cold_video = overlap.get("video_id", {}).get("validation_cold_row_rate", 0.0)
    observations.append(
        {
            "id": "cold_start",
            "finding": (
                f"Validation cold-user row rate is {cold_user:.4f}; "
                f"cold-video row rate is {cold_video:.4f}."
            ),
            "implication": "Balance ID memorization with metadata and popularity-aware fallback features.",
            "suggested_families": ["rich_fm", "fwfm", "catboost_residual"],
        }
    )

    candidate = report["candidate_aware_support"]
    author = candidate.get("user_author", {}).get(
        "validation_prior_pair_support_rate", 0.0
    )
    tag = candidate.get("user_primary_tag", {}).get(
        "validation_prior_pair_support_rate", 0.0
    )
    video = candidate.get("user_video", {}).get(
        "validation_prior_pair_support_rate", 0.0
    )
    adequate = max(author, tag) >= 0.20
    observations.append(
        {
            "id": "candidate_aware_history_support",
            "finding": (
                f"Prior user-video/author/tag support is {video:.4f}/{author:.4f}/{tag:.4f}."
            ),
            "implication": (
                "Candidate-aware history has enough coarse entity support for a bounded residual experiment."
                if adequate
                else "Sparse pair support makes a full sequential model risky; screen cheap history aggregates first."
            ),
            "suggested_families": (
                ["history_residual", "din_lite"] if adequate else ["history_residual"]
            ),
        }
    )

    duration = report.get("watch_ratio", {})
    observations.append(
        {
            "id": "duration_relative_behavior",
            "finding": (
                "Training median watch ratio is "
                f"{duration.get('ratio_distribution', {}).get('p50', 0.0):.4f}; "
                f"the share at or above completion is {duration.get('share_ge_1', 0.0):.4f}."
            ),
            "implication": "Duration-relative auxiliary targets are testable without exposing serving-time outcomes.",
            "suggested_families": ["duration_auxiliary", "multi_task_engagement"],
        }
    )

    temporal = report["temporal"]
    observations.append(
        {
            "id": "temporal_drift",
            "finding": (
                f"Train/validation hour total-variation distance is "
                f"{temporal.get('hour_total_variation', 0.0):.4f}."
            ),
            "implication": "Preserve chronological validation and include compact time context when drift is material.",
            "suggested_families": ["rich_fm", "dcn_v2", "history_residual"],
        }
    )
    return observations


def analyze_splits(splits: DatasetSplits) -> "EDAReport":
    """Analyze train outcomes and label-free validation/static distributions.

    ``splits.valid_labels`` is intentionally never touched.  Outcome columns are
    retained in the training frame for EDA but are stripped from validation
    before any schema, distribution, overlap, or support calculation.
    """

    train = splits.train
    valid = validation_static_frame(splits.valid)
    if "long_view" not in train:
        raise ValueError("Training frame must contain long_view")
    if "long_view" in valid or any(column in valid for column in OUTCOME_COLUMNS):
        raise RuntimeError("Validation outcome entered the label-free EDA frame")

    entities = [
        column for column in ENTITY_COLUMNS if column in train and column in valid
    ]
    overlap = {column: _overlap(train[column], valid[column]) for column in entities}
    train_user_counts = train["user_id"].value_counts(sort=False)
    validation_history = valid["user_id"].map(train_user_counts).fillna(0)
    validation_candidates = valid.groupby("user_id", sort=False).size()

    support_entities = {
        "user_video": "video_id",
        "user_author": "author_id",
        "user_primary_tag": "primary_tag",
        "user_music": "music_id",
    }
    pair_support = {
        name: _pair_support(train, valid, entity)
        for name, entity in support_entities.items()
        if entity in train and entity in valid
    }
    pair_support["validation_history_count"] = _numeric_summary(validation_history)
    pair_support["validation_warm_history_rate"] = _rate(validation_history > 0)

    duration: dict[str, Any] = {"available": False}
    if "duration_ms" in train and "duration_ms" in valid:
        duration = {
            "available": True,
            "train_ms": _numeric_summary(train["duration_ms"]),
            "validation_ms": _numeric_summary(valid["duration_ms"]),
            "train_validation_log_duration_total_variation": _total_variation(
                np.floor(np.log1p(train["duration_ms"].clip(lower=0)) * 2),
                np.floor(np.log1p(valid["duration_ms"].clip(lower=0)) * 2),
            ),
        }

    play_time: dict[str, Any] = {"available": False}
    watch_ratio: dict[str, Any] = {"available": False}
    if "play_time_ms" in train:
        play_time = {
            "available": True,
            "train_ms": _numeric_summary(train["play_time_ms"]),
        }
    if "play_time_ms" in train and "duration_ms" in train:
        duration_values = pd.to_numeric(train["duration_ms"], errors="coerce")
        ratio = pd.to_numeric(
            train["play_time_ms"], errors="coerce"
        ) / duration_values.where(duration_values > 0)
        finite = ratio[np.isfinite(ratio)].clip(lower=0, upper=5)
        positive = train["long_view"].to_numpy() == 1
        ratio_values = ratio.to_numpy(dtype=np.float64)
        valid_ratio = np.isfinite(ratio_values)
        watch_ratio = {
            "available": True,
            "ratio_distribution": _numeric_summary(finite),
            "share_ge_0_5": _rate(valid_ratio & (ratio_values >= 0.5)),
            "share_ge_1": _rate(valid_ratio & (ratio_values >= 1.0)),
            "positive_mean_ratio_clipped": _float(
                np.clip(ratio_values[valid_ratio & positive], 0, 5).mean()
                if np.any(valid_ratio & positive)
                else 0
            ),
            "negative_mean_ratio_clipped": _float(
                np.clip(ratio_values[valid_ratio & ~positive], 0, 5).mean()
                if np.any(valid_ratio & ~positive)
                else 0
            ),
        }

    feedback = {
        column: _float(pd.to_numeric(train[column], errors="coerce").mean())
        for column in FEEDBACK_COLUMNS
        if column in train
    }

    temporal: dict[str, Any] = {
        "train_dates": _date_summary(train),
        "validation_dates": _date_summary(valid),
    }
    if "hour" in train and "hour" in valid:
        temporal["hour_total_variation"] = _total_variation(
            train["hour"], valid["hour"]
        )
    elif "hourmin" in train and "hourmin" in valid:
        temporal["hour_total_variation"] = _total_variation(
            train["hourmin"] // 100, valid["hourmin"] // 100
        )
    if "weekday" in train and "weekday" in valid:
        temporal["weekday_total_variation"] = _total_variation(
            train["weekday"], valid["weekday"]
        )

    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "rows": {"train": int(len(train)), "validation": int(len(valid))},
        "schema": {"train": _schema(train), "validation_static": _schema(valid)},
        "validation_excluded_outcomes": sorted(
            column for column in splits.valid.columns if column in OUTCOME_COLUMNS
        ),
        "label_imbalance": {
            "source": "train_only",
            "long_view_positive_rate": _float(train["long_view"].mean()),
            "long_view_positive_rows": int((train["long_view"] == 1).sum()),
            "long_view_negative_rows": int((train["long_view"] == 0).sum()),
        },
        "entity_frequencies": {
            column: _frequency_summary(train[column]) for column in entities
        },
        "overlap": overlap,
        "cold_start": {
            column: {
                "validation_cold_row_rate": value["validation_cold_row_rate"],
                "validation_cold_entity_rate": value["validation_cold_entity_rate"],
            }
            for column, value in overlap.items()
        },
        "history_length": {
            "train_rows_per_user": _numeric_summary(train_user_counts),
            "validation_rows_prior_train_history": _numeric_summary(validation_history),
            "validation_rows_with_no_train_history_rate": _rate(
                validation_history == 0
            ),
        },
        "candidate_set_size": {
            "validation_rows_per_user": _numeric_summary(validation_candidates),
            "validation_users": int(len(validation_candidates)),
        },
        "temporal": temporal,
        "duration": duration,
        "play_time": play_time,
        "watch_ratio": watch_ratio,
        "feedback_rates": {"source": "train_only", **feedback},
        "repeat_exposure": {
            key: value for key, value in pair_support.items() if key.startswith("user_")
        },
        "candidate_aware_support": pair_support,
    }
    report["observations"] = _build_observations(report)
    return EDAReport(report=report)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _prompt_summary(report: Mapping[str, Any], max_chars: int) -> str:
    rows = report["rows"]
    overlap = report["overlap"]
    history = report["history_length"]
    candidates = report["candidate_set_size"]["validation_rows_per_user"]
    lines = [
        "EDA evidence (deterministic; train outcomes + label-free validation features only):",
        f"- Rows: train={rows['train']}, validation={rows['validation']}; train long_view rate={report['label_imbalance']['long_view_positive_rate']:.4f}.",
        (
            "- Validation cold-row rates: "
            + ", ".join(
                f"{key}={value['validation_cold_row_rate']:.4f}"
                for key, value in overlap.items()
            )
            + "."
        ),
        f"- Prior train history per validation row: p50={history['validation_rows_prior_train_history'].get('p50', 0):.1f}, p90={history['validation_rows_prior_train_history'].get('p90', 0):.1f}; no-history rate={history['validation_rows_with_no_train_history_rate']:.4f}.",
        f"- Validation candidates/user: p50={candidates.get('p50', 0):.1f}, p90={candidates.get('p90', 0):.1f}, max={candidates.get('max', 0):.0f}.",
    ]
    lines.extend(
        f"- [{item['id']}] {item['finding']} Implication: {item['implication']}"
        for item in report["observations"]
    )
    lines.append(
        "Every proposed experiment must cite at least one observation ID and change one scientific variable."
    )
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    marker = "\n[summary truncated at configured bound]"
    return text[: max(0, max_chars - len(marker))].rstrip() + marker


@dataclass(frozen=True)
class EDAReport:
    report: dict[str, Any]

    @property
    def stable_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self.report).encode("utf-8")).hexdigest()

    def prompt_summary(self, max_chars: int = 4000) -> str:
        if max_chars < 256:
            raise ValueError("max_chars must be at least 256")
        return _prompt_summary(self.report, max_chars)

    def to_json_dict(self, max_prompt_chars: int = 4000) -> dict[str, Any]:
        return {
            "report_sha256": self.stable_hash,
            "prompt_summary": self.prompt_summary(max_prompt_chars),
            "report": self.report,
        }

    def to_markdown(self) -> str:
        rows = self.report["rows"]
        labels = self.report["label_imbalance"]
        lines = [
            "# KuaiRand Autonomous EDA",
            "",
            f"Stable report SHA-256: `{self.stable_hash}`",
            "",
            "This report uses training outcomes and validation features available at scoring time. Validation outcomes are excluded.",
            "",
            "## Overview",
            "",
            "| Split | Rows |",
            "|---|---:|",
            f"| Train | {rows['train']} |",
            f"| Validation (static only) | {rows['validation']} |",
            "",
            f"Training long-view positive rate: `{labels['long_view_positive_rate']:.6f}`.",
            "",
            "## Overlap and cold start",
            "",
            "| Entity | Train entities | Validation entities | Cold row rate | Cold entity rate |",
            "|---|---:|---:|---:|---:|",
        ]
        for entity, values in self.report["overlap"].items():
            lines.append(
                f"| {entity} | {values['train_entities']} | {values['validation_entities']} | {values['validation_cold_row_rate']:.6f} | {values['validation_cold_entity_rate']:.6f} |"
            )
        lines.extend(["", "## Research observations", ""])
        for item in self.report["observations"]:
            families = ", ".join(item["suggested_families"])
            lines.extend(
                [
                    f"### {item['id']}",
                    "",
                    item["finding"],
                    "",
                    f"Implication: {item['implication']}",
                    "",
                    f"Relevant families: {families}.",
                    "",
                ]
            )
        lines.extend(
            [
                "## Prompt summary",
                "",
                "```text",
                self.prompt_summary(),
                "```",
                "",
            ]
        )
        return "\n".join(lines)


def write_eda_artifacts(
    report: EDAReport,
    output_dir: Path,
    *,
    basename: str = "eda_report",
    max_prompt_chars: int = 4000,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / f"{basename}.json",
        "markdown": output_dir / f"{basename}.md",
        "prompt": output_dir / f"{basename}.prompt.txt",
    }
    paths["json"].write_text(
        json.dumps(
            report.to_json_dict(max_prompt_chars),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["markdown"].write_text(report.to_markdown(), encoding="utf-8")
    paths["prompt"].write_text(
        report.prompt_summary(max_prompt_chars) + "\n", encoding="utf-8"
    )
    return paths
