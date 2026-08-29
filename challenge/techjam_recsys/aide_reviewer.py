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
from .protocol import BASELINE_VALID


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
            np.save(self.artifact_dir / f"{node.id}.npy", scores)
            shutil.copy2(
                self.prediction_path,
                self.artifact_dir / f"{node.id}.csv",
            )
            summary = json.dumps(
                {
                    "metrics": {
                        key: metrics[key] for key in ("GAUC", "nDCG@5", "primary")
                    },
                    "baseline_deltas": deltas,
                    "beats_both_components": (
                        deltas["GAUC"] > 0 and deltas["nDCG@5"] > 0
                    ),
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
