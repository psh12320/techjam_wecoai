"""Deterministic AIDE node review for the KuaiRand prediction contract."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from aide.agent import ReviewResult
from aide.interpreter import ExecutionResult
from aide.journal import Node

from .metrics import evaluate
from .protocol import BASELINE_VALID, CHAMPION_VALID
from .diagnostics import (
    aggregate_validation_diagnostics,
    diagnostics_prompt_summary,
)


class KuaiRandPredictionReviewer:
    def __init__(
        self,
        workspace_dir: Path,
        validation_index_path: Path,
        artifact_dir: Path,
    ):
        self.workspace_dir = Path(workspace_dir)
        self.prediction_path = (
            self.workspace_dir / "working" / "validation_predictions.csv"
        )
        self.change_decision_path = (
            self.workspace_dir / "working" / "change_decision.json"
        )
        index = np.load(validation_index_path)
        self.row_ids = index["row_id"]
        self.users = index["user_id"]
        self.video_ids = index["video_id"]
        self.labels = index["long_view"]
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.diagnostics_dir = self.artifact_dir.parent / "diagnostics"
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        input_dir = self.workspace_dir / "input"
        diagnostic_columns = ["user_id", "video_id", "date", "time_ms", "duration_ms"]
        self.diagnostic_train = pd.read_csv(
            input_dir / "train.csv", usecols=diagnostic_columns
        )
        self.diagnostic_valid = pd.read_csv(
            input_dir / "valid.csv", usecols=diagnostic_columns
        )
        video = pd.read_csv(
            input_dir / "video_features_basic_pure.csv",
            usecols=["video_id", "author_id", "tag"],
        )
        video["primary_tag"] = (
            video["tag"].fillna("UNK").astype(str).str.split(",").str[0]
        )
        video = video.drop(columns="tag")
        users = pd.read_csv(
            input_dir / "user_features_pure.csv",
            usecols=["user_id", "user_active_degree"],
        )
        for name in ("diagnostic_train", "diagnostic_valid"):
            frame = getattr(self, name)
            frame = frame.merge(
                video, on="video_id", how="left", validate="many_to_one"
            )
            frame = frame.merge(users, on="user_id", how="left", validate="many_to_one")
            setattr(self, name, frame)
        self.previous_predictions: dict[str, np.ndarray] = {}
        self.diagnostics_by_node: dict[str, dict[str, object]] = {}

    def clear_candidate_output(self) -> None:
        self.prediction_path.unlink(missing_ok=True)
        self.change_decision_path.unlink(missing_ok=True)

    def _load_change_decision(self, node: Node) -> dict[str, object]:
        if node.parent is None:
            return {
                "accepted_change": True,
                "reason": "immutable organizer ancestry root",
            }
        if not self.change_decision_path.exists():
            raise ValueError("candidate did not write working/change_decision.json")
        value = json.loads(self.change_decision_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("change_decision.json must contain one JSON object")
        if not isinstance(value.get("accepted_change"), bool):
            raise ValueError("change_decision.accepted_change must be boolean")
        if not str(value.get("reason") or "").strip():
            raise ValueError("change_decision.reason must be nonempty")
        for key in ("internal_parent_metrics", "internal_candidate_metrics"):
            if not isinstance(value.get(key), dict):
                raise ValueError(f"change_decision.{key} must be an object")
        return value

    def _nearest_metric_parent_scores(
        self, node: Node
    ) -> tuple[Node, np.ndarray]:
        parent = node.parent
        while parent is not None:
            if parent.metric is not None and not parent.metric.is_worst:
                scores = self.previous_predictions.get(parent.id)
                if scores is None:
                    artifact = self.artifact_dir / f"{parent.id}.npy"
                    if artifact.exists():
                        scores = np.load(artifact)
                if scores is not None:
                    return parent, np.asarray(scores, dtype=np.float64)
            parent = parent.parent
        raise ValueError("no metric-bearing parent prediction is available for fallback")

    @staticmethod
    def _metric_summary(metrics: dict[str, float]) -> dict[str, object]:
        deltas = {
            key: float(metrics[key] - BASELINE_VALID[key])
            for key in ("GAUC", "nDCG@5", "primary")
        }
        champion_deltas = {
            key: float(metrics[key] - CHAMPION_VALID[key])
            for key in ("GAUC", "nDCG@5", "primary")
        }
        return {
            "metrics": {
                key: metrics[key] for key in ("GAUC", "nDCG@5", "primary")
            },
            "baseline_deltas": deltas,
            "champion_deltas": champion_deltas,
            "beats_both_components": (
                deltas["GAUC"] > 0 and deltas["nDCG@5"] > 0
            ),
            "beats_champion": all(
                champion_deltas[key] > 0
                for key in ("GAUC", "nDCG@5", "primary")
            ),
        }

    def __call__(self, node: Node, result: ExecutionResult) -> ReviewResult:
        if result.exc_type is not None:
            return ReviewResult(
                is_bug=True,
                summary=f"Execution failed with {result.exc_type}.",
                metric=None,
                lower_is_better=False,
            )
        if not self.prediction_path.exists():
            return ReviewResult(
                is_bug=True,
                summary=("Candidate did not write working/validation_predictions.csv."),
                metric=None,
                lower_is_better=False,
            )
        try:
            decision = self._load_change_decision(node)
            node.candidate_spec["runtime_change_accepted"] = bool(
                decision["accepted_change"]
            )
            node.candidate_spec["change_decision_reason"] = str(decision["reason"])
            if not decision["accepted_change"]:
                parent, scores = self._nearest_metric_parent_scores(node)
                if scores.shape != self.labels.shape or not np.isfinite(scores).all():
                    raise ValueError("trusted parent fallback predictions are invalid")
                metrics = evaluate(self.users, self.labels, scores)
                node.candidate_spec["fallback_parent_node_id"] = parent.id
                diagnostic_record: dict[str, object] = {
                    "accepted_change": False,
                    "fallback_parent_node_id": parent.id,
                    "max_frontier_prediction_correlation": 1.0,
                    "prompt_summary": (
                        "The candidate's internal gate rejected its scientific change; "
                        "the trusted harness retained the exact metric-bearing parent."
                    ),
                }
                self.diagnostics_by_node[node.id] = diagnostic_record
                summary = {
                    **self._metric_summary(metrics),
                    "scientific_change_accepted": False,
                    "fallback_parent_node_id": parent.id,
                    "change_decision_reason": str(decision["reason"]),
                }
                return ReviewResult(
                    is_bug=False,
                    summary=json.dumps(summary, sort_keys=True),
                    metric=float(metrics["primary"]),
                    lower_is_better=False,
                )
            frame = pd.read_csv(self.prediction_path)
            if list(frame.columns) != ["row_id", "score"]:
                raise ValueError("header must be exactly row_id,score")
            if len(frame) != len(self.row_ids):
                raise ValueError(
                    f"expected {len(self.row_ids)} rows, received {len(frame)}"
                )
            row_ids = frame["row_id"].to_numpy()
            if not np.array_equal(row_ids, self.row_ids):
                raise ValueError("row_id must be zero-based, contiguous, and aligned")
            scores = frame["score"].to_numpy(dtype=np.float64)
            if not np.isfinite(scores).all():
                raise ValueError("scores contain NaN or infinity")
            # Reject the most obvious label-copy failure. This is a safety check,
            # not a substitute for the hidden-test boundary or code review.
            if np.array_equal(scores, self.labels.astype(np.float64)):
                raise ValueError("predictions exactly copy validation labels")
            metrics = evaluate(self.users, self.labels, scores)
            metric_summary = self._metric_summary(metrics)
            np.save(self.artifact_dir / f"{node.id}.npy", scores)
            shutil.copy2(
                self.prediction_path,
                self.artifact_dir / f"{node.id}.csv",
            )
            comparison_predictions = {
                **self.previous_predictions,
                node.id: scores,
            }
            diagnostics = aggregate_validation_diagnostics(
                self.diagnostic_train,
                self.diagnostic_valid,
                self.users,
                self.labels,
                comparison_predictions,
            )
            diagnostics_path = self.diagnostics_dir / f"{node.id}.json"
            diagnostics_path.write_text(
                json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
            )
            correlations = []
            for pair, values in diagnostics.get("model_diversity", {}).items():
                if node.id in pair.split("__"):
                    correlations.append(
                        float(values.get("within_user_rank_correlation", 0.0))
                    )
            diagnostic_record: dict[str, object] = {
                "accepted_change": True,
                "diagnostics_sha256": diagnostics["diagnostics_sha256"],
                "max_frontier_prediction_correlation": (
                    max(correlations) if correlations else None
                ),
                "prompt_summary": diagnostics_prompt_summary(diagnostics),
            }
            self.diagnostics_by_node[node.id] = diagnostic_record
            self.previous_predictions[node.id] = scores.copy()
            while len(self.previous_predictions) > 4:
                self.previous_predictions.pop(next(iter(self.previous_predictions)))
            summary = json.dumps(
                {
                    **metric_summary,
                    "scientific_change_accepted": True,
                    "diagnostics_sha256": diagnostics["diagnostics_sha256"],
                    "diagnostics_summary": diagnostic_record["prompt_summary"],
                    "max_frontier_prediction_correlation": diagnostic_record[
                        "max_frontier_prediction_correlation"
                    ],
                },
                sort_keys=True,
            )
            return ReviewResult(
                is_bug=False,
                summary=summary,
                metric=float(metrics["primary"]),
                lower_is_better=False,
            )
        except Exception as exc:
            return ReviewResult(
                is_bug=True,
                summary=f"Prediction contract failed: {type(exc).__name__}: {exc}",
                metric=None,
                lower_is_better=False,
            )
