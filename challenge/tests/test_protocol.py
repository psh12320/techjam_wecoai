import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

import aide.backend as backend
from challenge.run_aide_research import (
    bounded_candidate_exec_seconds,
    candidate_metrics_pass,
    campaign_final_designation,
    compact_cost_totals,
    estimate_uncached_cost,
    parse_args,
    single_run_candidate_evidence,
)
from challenge.techjam_recsys.metrics import evaluate, rank_normalize_within_user
from challenge.techjam_recsys.protocol import (
    CHAMPION_VALID,
    ChallengeMetric,
    ConvergenceTracker,
    ExperimentLedger,
    TrialRecord,
    count_manual_interventions,
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


def test_compact_cost_totals_keeps_only_latest_event() -> None:
    state = {
        "total_estimated_cost_usd": 9.4,
        "total_input_tokens": 100,
        "total_output_tokens": 50,
        "total_requests": 2,
        "next_notification_usd": 10.0,
        "events": [{"model": "old"}, {"model": "gpt-5.6-sol"}],
    }
    compact = compact_cost_totals(state)
    assert compact["latest_event"] == {"model": "gpt-5.6-sol"}
    assert "events" not in compact


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


def test_failed_execution_counts_toward_attempt_cap_not_metric_patience() -> None:
    tracker = ConvergenceTracker(max_iterations=4)
    assert tracker.observe(0.6016) is False
    assert tracker.observe_failure() is False
    assert tracker.insignificant_iterations == 0
    assert tracker.observe(0.6020) is False
    assert tracker.observe_failure() is True
    assert tracker.stop_reason == "iteration_cap"


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
    assert loaded[0].record_sha256
    assert loaded[1].previous_record_sha256 == loaded[0].record_sha256
    assert select_champion(loaded).trial_id == balanced.trial_id


def test_tampered_ledger_record_fails_hash_validation(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    ledger = ExperimentLedger(path)
    ledger.append(
        TrialRecord(
            iteration=0,
            hypothesis="original",
            model_family="rich_fm",
            status="success",
        )
    )
    path.write_text(
        path.read_text(encoding="utf-8").replace("original", "tampered"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="record hash"):
        ledger.read()


def test_trial_record_separates_candidate_runtime_from_end_to_end_wall(
    tmp_path: Path,
) -> None:
    ledger = ExperimentLedger(tmp_path / "run.jsonl")
    ledger.append(
        TrialRecord(
            iteration=0,
            hypothesis="bounded candidate",
            model_family="rich_fm",
            status="success",
            wall_seconds=975.0,
            candidate_exec_seconds=900.0,
        )
    )

    record = ledger.read()[0]
    assert record.wall_seconds == 975.0
    assert record.candidate_exec_seconds == 900.0


def test_candidate_runtime_accounting_is_bounded_by_interpreter_timeout() -> None:
    assert bounded_candidate_exec_seconds(901.25, 900) == 900.0
    assert bounded_candidate_exec_seconds(12.5, 900) == 12.5
    assert bounded_candidate_exec_seconds(None, 900) is None


def test_legacy_chained_record_without_candidate_runtime_remains_readable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.jsonl"
    payload = asdict(
        TrialRecord(
            iteration=0,
            hypothesis="legacy",
            model_family="official_fm",
            status="success",
        )
    )
    payload.pop("candidate_exec_seconds")
    payload["record_sha256"] = None
    canonical = json.dumps(
        payload, sort_keys=True, allow_nan=False, separators=(",", ":")
    )
    payload["record_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    record = ExperimentLedger(path).read()[0]
    assert record.wall_seconds == 0.0
    assert record.candidate_exec_seconds is None


def test_uncached_cost_envelope_matches_published_rate_math() -> None:
    assert estimate_uncached_cost(75_000, 18_000, 2.50, 15.00) == pytest.approx(0.4575)


def test_champion_gate_requires_both_component_improvements() -> None:
    winner = ChallengeMetric(
        gauc=CHAMPION_VALID["GAUC"] + 0.0001,
        ndcg5=CHAMPION_VALID["nDCG@5"] + 0.0001,
    )
    tradeoff = ChallengeMetric(
        gauc=CHAMPION_VALID["GAUC"] + 0.01,
        ndcg5=CHAMPION_VALID["nDCG@5"] - 0.0001,
    )
    assert winner.beats_champion is True
    assert tradeoff.beats_champion is False


def test_single_run_gate_is_evaluated_against_all_champion_components() -> None:
    winner = {
        "GAUC": CHAMPION_VALID["GAUC"] + 0.0001,
        "nDCG@5": CHAMPION_VALID["nDCG@5"] + 0.0001,
        "primary": CHAMPION_VALID["primary"] + 0.0001,
    }
    ndcg_regression = dict(winner, **{"nDCG@5": CHAMPION_VALID["nDCG@5"] - 0.0001})
    assert candidate_metrics_pass(winner) is True
    assert candidate_metrics_pass(ndcg_regression) is False
    assert candidate_metrics_pass(None) is False


def qualifying_single_run_record(**overrides) -> TrialRecord:
    digest = "a" * 64
    values = {
        "iteration": 1,
        "hypothesis": "one deterministic full-data candidate",
        "model_family": "rich_fm",
        "status": "success",
        "config": {"card_complete": True},
        "metrics": {
            "GAUC": CHAMPION_VALID["GAUC"] + 0.0001,
            "nDCG@5": CHAMPION_VALID["nDCG@5"] + 0.0001,
            "primary": CHAMPION_VALID["primary"] + 0.0001,
        },
        "source": "aide_generated",
        "node_id": "winning-node",
        "seed": 0,
        "evaluation_fidelity": "full",
        "exit_status": "success",
        "record_sha256": digest,
        "code_sha256": digest,
        "prompt_sha256": digest,
        "campaign_manifest_sha256": digest,
        "source_sha256": digest,
        "input_sha256": digest,
        "dependency_sha256": digest,
        "evaluator_sha256": digest,
        "diagnostics_sha256": digest,
        "internal_validation_sha256": digest,
        "assignment_compliant": True,
        "artifact_ids": [
            "predictions/winning-node.npy",
            "diagnostics/winning-node.json",
        ],
        "artifact_sha256": {
            "predictions/winning-node.npy": digest,
            "diagnostics/winning-node.json": digest,
        },
    }
    values.update(overrides)
    return TrialRecord(**values)


def test_single_run_evidence_requires_seed_zero_full_fidelity_and_hashes() -> None:
    assert single_run_candidate_evidence(qualifying_single_run_record())["valid"]
    assert not single_run_candidate_evidence(qualifying_single_run_record(seed=1))[
        "valid"
    ]
    assert not single_run_candidate_evidence(
        qualifying_single_run_record(evaluation_fidelity="screen")
    )["valid"]
    assert not single_run_candidate_evidence(
        qualifying_single_run_record(evaluator_sha256=None)
    )["valid"]
    assert not single_run_candidate_evidence(
        qualifying_single_run_record(artifact_sha256={})
    )["valid"]


def test_final_designation_requires_single_run_and_clean_evidence() -> None:
    evidence = {"valid": True}
    candidate = single_run_candidate_evidence(qualifying_single_run_record())
    assert campaign_final_designation("clean", {"valid": False}, 0, evidence) == (
        "rejected",
        False,
        False,
    )
    assert campaign_final_designation("development", candidate, 0, evidence) == (
        "single_run_development_candidate",
        True,
        False,
    )
    assert campaign_final_designation("clean", candidate, 0, evidence) == (
        "competition_ready",
        True,
        True,
    )
    assert campaign_final_designation("clean", candidate, 1, evidence) == (
        "single_run_development_candidate",
        True,
        False,
    )


def test_confirmation_cli_is_removed_and_49_generated_steps_parse() -> None:
    assert parse_args(["--steps", "49"]).steps == 49
    assert parse_args(["--baseline-only"]).baseline_only is True
    with pytest.raises(SystemExit):
        parse_args(["--confirm-on-breakthrough"])


def test_manual_interventions_are_derived_from_event_log(tmp_path: Path) -> None:
    path = tmp_path / "interventions.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"actor": "agent", "event_type": "recovery"}),
                json.dumps({"actor": "human", "event_type": "intervention"}),
                json.dumps({"actor": "human", "event_type": "note"}),
            ]
        ),
        encoding="utf-8",
    )
    assert count_manual_interventions(path) == 1


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
