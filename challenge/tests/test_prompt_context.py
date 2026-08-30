from __future__ import annotations

import json
from pathlib import Path

import pytest

from challenge.build_experiment_memory import build_memory
from challenge.techjam_recsys.prompt_context import load_prompt_context
from challenge.techjam_recsys.protocol import ExperimentLedger, TrialRecord


def test_prompt_sections_are_separate_and_bounded(tmp_path: Path) -> None:
    root = tmp_path
    (root / "challenge/prompts").mkdir(parents=True)
    (root / "challenge/research_memory").mkdir(parents=True)
    (root / "challenge/prompts/hard_constraints.md").write_text(
        "hard", encoding="utf-8"
    )
    (root / "challenge/prompts/research_menu.md").write_text("menu", encoding="utf-8")
    (root / "challenge/research_memory/experiment_memory.json").write_text(
        json.dumps({"entries": [{"model_family": "rich_fm"}]}), encoding="utf-8"
    )
    (root / "challenge/research_memory/literature_manifest.json").write_text(
        json.dumps({"notes": []}), encoding="utf-8"
    )

    sections = load_prompt_context(root).sections()
    assert list(sections) == [
        "Hard constraints",
        "Experiment memory",
        "Optional research menu",
        "EDA evidence",
        "Literature evidence",
    ]
    assert "rich_fm" in sections["Experiment memory"]


def test_prompt_memory_rejects_reusable_artifacts(tmp_path: Path) -> None:
    (tmp_path / "challenge/prompts").mkdir(parents=True)
    (tmp_path / "challenge/research_memory").mkdir(parents=True)
    (tmp_path / "challenge/prompts/hard_constraints.md").write_text(
        "hard", encoding="utf-8"
    )
    (tmp_path / "challenge/prompts/research_menu.md").write_text(
        "menu", encoding="utf-8"
    )
    (tmp_path / "challenge/research_memory/experiment_memory.json").write_text(
        json.dumps({"entries": [{"predictions": [0.1]}]}), encoding="utf-8"
    )
    (tmp_path / "challenge/research_memory/literature_manifest.json").write_text(
        "{}", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Unsafe reusable artifact"):
        load_prompt_context(tmp_path)


def test_memory_builder_validates_ledger_and_omits_code_and_artifacts(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "iterations.jsonl"
    ledger = ExperimentLedger(ledger_path)
    ledger.append(
        TrialRecord(
            iteration=1,
            hypothesis="bounded residual",
            model_family="history_residual",
            status="success",
            config={
                "scientific_change": "add strict history residual",
                "features": ["prior_count"],
                "losses": {"bce": 1.0},
                "artifact_ids": ["predictions/x.npy"],
            },
            metrics={"GAUC": 0.672, "nDCG@5": 0.539, "primary": 0.6055},
            code_diff="secret implementation",
            artifact_ids=["predictions/x.npy"],
            code_sha256="a" * 64,
        )
    )

    memory = build_memory([ledger_path])
    rendered = json.dumps(memory)
    assert "secret implementation" not in rendered
    assert "predictions/x.npy" not in rendered
    assert memory["entries"][0]["configuration"]["scientific_change"]
    assert memory["entries"][0]["outcome"] == "evaluated"
    assert memory["content_sha256"] != "unbuilt"


def test_memory_builder_reserves_latest_evaluated_descendants(tmp_path: Path) -> None:
    ledger = ExperimentLedger(tmp_path / "iterations.jsonl")
    root = TrialRecord(
        iteration=0,
        hypothesis="organizer",
        model_family="official_fm_seed",
        status="success",
        metrics={"GAUC": 0.667, "nDCG@5": 0.535, "primary": 0.601},
    )
    ledger.append(root)
    older = TrialRecord(
        iteration=1,
        hypothesis="older strong candidate",
        model_family="duration_auxiliary",
        status="success",
        parent_trial_id=root.trial_id,
        metrics={"GAUC": 0.672, "nDCG@5": 0.539, "primary": 0.6055},
    )
    ledger.append(older)
    recent_tradeoff = TrialRecord(
        iteration=2,
        hypothesis="recent component tradeoff",
        model_family="rich_fm",
        status="success",
        parent_trial_id=root.trial_id,
        metrics={"GAUC": 0.6669, "nDCG@5": 0.5352, "primary": 0.60105},
    )
    ledger.append(recent_tradeoff)
    recent_fallback = TrialRecord(
        iteration=3,
        hypothesis="recent gated fallback",
        model_family="history_residual",
        status="success",
        parent_trial_id=recent_tradeoff.trial_id,
        metrics=dict(recent_tradeoff.metrics or {}),
    )
    ledger.append(recent_fallback)

    memory = build_memory([ledger.path], max_entries=3)
    ids = {entry["trial_id"] for entry in memory["entries"]}
    assert recent_tradeoff.trial_id in ids
    assert recent_fallback.trial_id in ids


def test_prompt_projection_keeps_all_sixteen_compact_entries(tmp_path: Path) -> None:
    (tmp_path / "challenge/prompts").mkdir(parents=True)
    (tmp_path / "challenge/research_memory").mkdir(parents=True)
    (tmp_path / "challenge/prompts/hard_constraints.md").write_text(
        "hard", encoding="utf-8"
    )
    (tmp_path / "challenge/prompts/research_menu.md").write_text(
        "menu", encoding="utf-8"
    )
    entries = []
    for index in range(16):
        entries.append(
            {
                "node_id": f"node-{index}",
                "model_family": f"family-{index}",
                "status": "success",
                "outcome": "component_tradeoff",
                "metrics": {"GAUC": 0.66, "nDCG@5": 0.53, "primary": 0.595},
                "configuration": {
                    "scientific_change": f"bounded change {index}",
                    "features": [f"feature-{value}" for value in range(12)],
                    "losses": {"bce": 1.0},
                },
            }
        )
    (tmp_path / "challenge/research_memory/experiment_memory.json").write_text(
        json.dumps({"entries": entries}), encoding="utf-8"
    )
    (tmp_path / "challenge/research_memory/literature_manifest.json").write_text(
        "{}", encoding="utf-8"
    )

    prompt = json.loads(load_prompt_context(tmp_path).experiment_memory)
    assert prompt["included_entries"] == 16
    assert prompt["available_entries"] == 16
    assert {entry["node_id"] for entry in prompt["entries"]} == {
        f"node-{index}" for index in range(16)
    }


def test_repair_outcome_uses_nearest_metric_parent(tmp_path: Path) -> None:
    ledger = ExperimentLedger(tmp_path / "repair.jsonl")
    root = TrialRecord(
        iteration=0,
        hypothesis="organizer",
        model_family="official_fm_seed",
        status="success",
        metrics={"GAUC": 0.667, "nDCG@5": 0.536, "primary": 0.6015},
    )
    ledger.append(root)
    failed = TrialRecord(
        iteration=1,
        hypothesis="duration branch",
        model_family="duration_auxiliary",
        status="failed",
        parent_trial_id=root.trial_id,
    )
    ledger.append(failed)
    repaired = TrialRecord(
        iteration=2,
        hypothesis="same duration branch repaired",
        model_family="duration_auxiliary",
        status="success",
        parent_trial_id=failed.trial_id,
        metrics={"GAUC": 0.666, "nDCG@5": 0.535, "primary": 0.6005},
    )
    ledger.append(repaired)

    memory = build_memory([ledger.path], max_entries=3)
    repair_entry = next(
        entry for entry in memory["entries"] if entry["trial_id"] == repaired.trial_id
    )
    assert repair_entry["outcome"] == "parent_dominated"


def test_internal_rejection_is_recorded_as_falsified(tmp_path: Path) -> None:
    ledger = ExperimentLedger(tmp_path / "falsified.jsonl")
    root = TrialRecord(
        iteration=0,
        hypothesis="organizer",
        model_family="official_fm_seed",
        status="success",
        metrics={"GAUC": 0.667, "nDCG@5": 0.536, "primary": 0.6015},
    )
    ledger.append(root)
    rejected = TrialRecord(
        iteration=1,
        hypothesis="rejected residual",
        model_family="history_residual",
        status="success",
        parent_trial_id=root.trial_id,
        config={"runtime_change_accepted": False},
        metrics=dict(root.metrics or {}),
    )
    ledger.append(rejected)

    memory = build_memory([ledger.path], max_entries=2)
    entry = next(
        item for item in memory["entries"] if item["trial_id"] == rejected.trial_id
    )
    assert entry["outcome"] == "falsified_internal"
