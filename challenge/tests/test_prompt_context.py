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
    assert memory["content_sha256"] != "unbuilt"
