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
            deltas = {
                key: float(metrics[key] - BASELINE_VALID[key])
                for key in ("GAUC", "nDCG@5", "primary")
            }
            champion_deltas = {
                key: float(metrics[key] - CHAMPION_VALID[key])
                for key in ("GAUC", "nDCG@5", "primary")
            }
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
