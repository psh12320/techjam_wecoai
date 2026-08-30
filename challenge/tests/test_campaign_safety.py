from __future__ import annotations

from pathlib import Path

import pytest

from aide.utils import copytree
from challenge.techjam_recsys.campaign_safety import (
    CANDIDATE_INPUT_ALLOWLIST,
    CampaignEventLedger,
    validate_candidate_input,
)


def _allowed_input(path: Path) -> None:
    path.mkdir()
    for name in CANDIDATE_INPUT_ALLOWLIST:
        (path / name).write_text(name, encoding="utf-8")


def test_candidate_workspace_allowlist_excludes_evaluator_and_champion_artifacts(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    _allowed_input(input_dir)
    hashes = validate_candidate_input(input_dir)
    assert set(hashes) == CANDIDATE_INPUT_ALLOWLIST

    (input_dir / "validation_index.npz").write_bytes(b"private")
    with pytest.raises(RuntimeError, match="validation_index"):
        validate_candidate_input(input_dir)


def test_clean_evidence_requires_start_completion_and_no_human_event(
    tmp_path: Path,
) -> None:
    ledger = CampaignEventLedger(tmp_path / "events.jsonl", "run-1")
    assert ledger.clean_evidence("manifest")["valid"] is False
    ledger.append("run_started", details={"manifest_sha256": "manifest"})
    assert ledger.clean_evidence("manifest")["valid"] is False
    ledger.append("run_completed", details={"manifest_sha256": "manifest"})
    assert ledger.clean_evidence("manifest")["valid"] is True

    ledger.append("intervention", actor="human")
    evidence = ledger.clean_evidence("manifest")
    assert evidence["valid"] is False
    assert evidence["human_events"] == 1


def test_workspace_copy_ignores_python_caches(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "public.csv").write_text("x\n1\n", encoding="utf-8")
    cache = source / "__pycache__"
    cache.mkdir()
    (cache / "data.cpython-311.pyc").write_bytes(b"cache")

    copytree(source, destination, use_symlinks=False)

    assert (destination / "public.csv").exists()
    assert not (destination / "__pycache__").exists()
