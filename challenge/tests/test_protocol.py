from pathlib import Path

import numpy as np
import pytest

import aide.backend as backend
from challenge.run_aide_research import estimate_uncached_cost
from challenge.techjam_recsys.metrics import evaluate, rank_normalize_within_user
from challenge.techjam_recsys.protocol import (
    ConvergenceTracker,
    ExperimentLedger,
    TrialRecord,
    select_champion,
)


def test_exact_metric_smoke() -> None:
    result = evaluate(
        ["a", "a", "b", "b"],
        [1, 0, 0, 0],
        [0.9, 0.1, 0.2, 0.1],
    )
    assert result["GAUC"] == 1.0
    assert result["nDCG@5"] == 0.5
    assert result["primary"] == 0.75


def test_rank_normalization_is_per_user() -> None:
    output = rank_normalize_within_user([1, 1, 1, 2, 2], [10.0, 30.0, 20.0, -1.0, -2.0])
    np.testing.assert_allclose(output, [0.0, 1.0, 0.5, 1.0, 0.0])


def test_rank_normalization_preserves_ties() -> None:
    output = rank_normalize_within_user([1, 1, 1], [2.0, 2.0, 3.0])
    np.testing.assert_allclose(output, [0.25, 0.25, 1.0])


def test_convergence_uses_organizer_epsilon_and_patience() -> None:
    tracker = ConvergenceTracker()
    assert tracker.observe(0.6016) is False
    assert tracker.observe(0.6020) is False
    assert tracker.observe(0.6030) is False
    assert tracker.observe(0.6035) is True
    assert tracker.stop_reason == "converged"


def test_ledger_and_champion_quality_gate(tmp_path: Path) -> None:
    ledger = ExperimentLedger(tmp_path / "run.jsonl")
    primary_only = TrialRecord(
        iteration=0,
        hypothesis="raise one metric",
        model_family="x",
        status="success",
        metrics={"GAUC": 0.6800, "nDCG@5": 0.5350, "primary": 0.6075},
    )
    balanced = TrialRecord(
        iteration=1,
        hypothesis="raise both metrics",
        model_family="y",
        status="success",
        metrics={"GAUC": 0.6680, "nDCG@5": 0.5360, "primary": 0.6020},
    )
    ledger.append(primary_only)
    ledger.append(balanced)
    loaded = ledger.read()
    assert len(loaded) == 2
    assert select_champion(loaded).trial_id == balanced.trial_id


def test_uncached_cost_envelope_matches_published_rate_math() -> None:
    assert estimate_uncached_cost(75_000, 18_000, 2.50, 15.00) == pytest.approx(0.4575)


def test_durable_cost_tracker_emits_each_crossed_boundary(tmp_path, capsys) -> None:
    previous = backend._cost_tracking
    try:
        path = tmp_path / "cost.json"
        backend.configure_cost_tracking(path, 1.0, 0.0, notification_step_usd=10.0)
        first = backend._record_cost_event("test", 6_000_000, 0)
        second = backend._record_cost_event("test", 15_000_000, 0)
        state = backend.get_cost_tracking_totals()

        assert first["crossed_notification_thresholds_usd"] == []
        assert second["crossed_notification_thresholds_usd"] == [10.0, 20.0]
        assert state["total_estimated_cost_usd"] == pytest.approx(21.0)
        assert state["next_notification_usd"] == 30.0
        output = capsys.readouterr().out
        assert '"threshold_usd": 10.0' in output
        assert '"threshold_usd": 20.0' in output
    finally:
        backend._cost_tracking = previous
