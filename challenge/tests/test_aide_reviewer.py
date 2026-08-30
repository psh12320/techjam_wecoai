from __future__ import annotations

import json

import numpy as np
import pandas as pd

from aide.interpreter import ExecutionResult
from aide.journal import Node
from aide.utils.metric import MetricValue
from challenge.techjam_recsys.aide_reviewer import KuaiRandPredictionReviewer
from challenge.techjam_recsys.metrics import evaluate


def _reviewer(tmp_path) -> KuaiRandPredictionReviewer:
    reviewer = object.__new__(KuaiRandPredictionReviewer)
    reviewer.workspace_dir = tmp_path
    reviewer.prediction_path = tmp_path / "working" / "validation_predictions.csv"
    reviewer.change_decision_path = tmp_path / "working" / "change_decision.json"
    reviewer.prediction_path.parent.mkdir(parents=True)
    reviewer.row_ids = np.arange(4)
    reviewer.users = np.asarray(["a", "a", "b", "b"])
    reviewer.video_ids = np.arange(4)
    reviewer.labels = np.asarray([1, 0, 0, 1])
    reviewer.artifact_dir = tmp_path / "artifacts"
    reviewer.artifact_dir.mkdir()
    reviewer.diagnostics_dir = tmp_path / "diagnostics"
    reviewer.diagnostics_dir.mkdir()
    reviewer.previous_predictions = {}
    reviewer.diagnostics_by_node = {}
    return reviewer


def test_internal_rejection_retains_exact_parent_predictions(tmp_path) -> None:
    reviewer = _reviewer(tmp_path)
    parent_scores = np.asarray([0.9, 0.1, 0.2, 0.8])
    parent_metrics = evaluate(reviewer.users, reviewer.labels, parent_scores)
    parent = Node(
        id="parent",
        code="pass",
        plan="parent",
        candidate_spec={"model_family": "official_fm_seed"},
        metric=MetricValue(parent_metrics["primary"], maximize=True),
        analysis=json.dumps({"metrics": parent_metrics}),
        is_buggy=False,
    )
    reviewer.previous_predictions[parent.id] = parent_scores
    candidate = Node(
        id="candidate",
        code="pass",
        plan="rejected residual",
        parent=parent,
        candidate_spec={"model_family": "history_residual"},
    )
    pd.DataFrame(
        {"row_id": np.arange(4), "score": [0.1, 0.9, 0.8, 0.2]}
    ).to_csv(reviewer.prediction_path, index=False)
    reviewer.change_decision_path.write_text(
        json.dumps(
            {
                "accepted_change": False,
                "reason": "joint gate failed",
                "internal_parent_metrics": {"primary": 0.6},
                "internal_candidate_metrics": {"primary": 0.5},
                "selected_configuration": {"scale": 0.0},
            }
        ),
        encoding="utf-8",
    )

    review = reviewer(
        candidate,
        ExecutionResult(term_out=[], exec_time=1.0, exc_type=None),
    )
    payload = json.loads(review.summary)
    assert review.is_bug is False
    assert review.metric == parent_metrics["primary"]
    assert payload["scientific_change_accepted"] is False
    assert payload["fallback_parent_node_id"] == parent.id
    assert candidate.candidate_spec["runtime_change_accepted"] is False
    assert not (reviewer.artifact_dir / "candidate.npy").exists()


def test_generated_candidate_requires_change_decision(tmp_path) -> None:
    reviewer = _reviewer(tmp_path)
    parent = Node(
        code="pass",
        plan="parent",
        metric=MetricValue(0.6, maximize=True),
        analysis=json.dumps(
            {"metrics": {"GAUC": 0.66, "nDCG@5": 0.54, "primary": 0.6}}
        ),
        is_buggy=False,
    )
    candidate = Node(code="pass", plan="candidate", parent=parent)
    pd.DataFrame(
        {"row_id": np.arange(4), "score": [0.9, 0.1, 0.2, 0.8]}
    ).to_csv(reviewer.prediction_path, index=False)

    review = reviewer(
        candidate,
        ExecutionResult(term_out=[], exec_time=1.0, exc_type=None),
    )
    assert review.is_bug is True
    assert "change_decision.json" in review.summary
