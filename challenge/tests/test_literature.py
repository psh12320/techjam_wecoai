from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from challenge.techjam_recsys.literature import (
    LiteratureBounds,
    LiteratureValidationError,
    build_literature_queries,
    freeze_manifest,
    load_manifest,
    new_manifest,
    resolve_candidate_citations,
    run_literature_research,
    write_manifest,
)


def test_repository_starter_manifest_is_valid_and_frozen() -> None:
    starter = (
        Path(__file__).resolve().parents[1]
        / "research_memory"
        / "literature_manifest.json"
    )
    manifest = load_manifest(starter)
    assert manifest["mode"] == "frozen"
    assert len(manifest["notes"]) == 9
    assert all(note["citation_id"].startswith("lit-") for note in manifest["notes"])


def _note(index: int, *, padding: str = "") -> dict[str, Any]:
    return {
        "title": f"Original Recommender Paper {index}",
        "authors": ["Ada Researcher", "Lin Scientist"],
        "year": 2020 + index,
        "url": f"https://example.org/papers/{index}",
        "technique": f"bounded ranking method {index}{padding}",
        "data": "implicit user-item interactions and content fields",
        "effect": "expected to improve the weak top-k ranking component",
        "cost": "one small CPU model",
        "risk": "may trade broad ordering for top-k quality",
        "applicability": "uses only fields present at KuaiRand serving time",
    }


class FakeProvider:
    def __init__(self, results: list[dict[str, Any]]):
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, limit: int):
        self.calls.append((query, limit))
        yield from self.results


class FailingProvider:
    def search(self, query: str, *, limit: int):
        raise AssertionError("frozen mode must never invoke its provider")


def test_query_builder_is_deterministic_and_bounded() -> None:
    eda = {
        "cold_start_rate": 0.2,
        "candidate_aware_history": "adequate support",
        "duration_drift": True,
    }
    weaknesses = ["nDCG top-five ranking error", "model correlation"]
    first = build_literature_queries(eda, weaknesses, max_queries=3)
    second = build_literature_queries(eda, weaknesses, max_queries=3)
    assert first == second
    assert len(first) == 3
    assert "LambdaLoss" in first[0]


def test_online_cache_hit_avoids_a_second_provider_call(tmp_path: Path) -> None:
    manifest_path = tmp_path / "literature.json"
    provider = FakeProvider([_note(1)])
    kwargs = {
        "manifest_path": manifest_path,
        "eda_findings": {"candidate-aware history": True},
        "weaknesses": ["nDCG"],
        "mode": "online",
        "provider": provider,
        "bounds": LiteratureBounds(max_queries=2),
    }
    first = run_literature_research(**kwargs)
    assert first.cache_misses == 2
    assert len(provider.calls) == 2

    second = run_literature_research(**kwargs)
    assert second.cache_hits == 2
    assert second.cache_misses == 0
    assert len(provider.calls) == 2
    assert second.citation_ids == first.citation_ids
    assert load_manifest(manifest_path)["manifest_sha256"] == second.manifest_sha256


def test_provider_result_count_and_bytes_are_bounded(tmp_path: Path) -> None:
    provider = FakeProvider([_note(index) for index in range(8)])
    bounds = LiteratureBounds(
        max_queries=1,
        max_results_per_query=2,
        max_total_notes=1,
        max_provider_bytes=4_000,
        max_note_bytes=2_000,
    )
    result = run_literature_research(
        manifest_path=tmp_path / "bounded.json",
        eda_findings={},
        weaknesses="nDCG",
        mode="online",
        provider=provider,
        bounds=bounds,
    )
    manifest = load_manifest(tmp_path / "bounded.json")
    assert provider.calls[0][1] == 2
    assert len(manifest["notes"]) == 1
    assert result.provider_bytes <= bounds.max_provider_bytes


def test_oversized_note_is_rejected(tmp_path: Path) -> None:
    provider = FakeProvider([_note(1, padding="x" * 3_000), _note(2)])
    result = run_literature_research(
        manifest_path=tmp_path / "oversized.json",
        eda_findings={},
        weaknesses="nDCG",
        mode="online",
        provider=provider,
        bounds=LiteratureBounds(
            max_queries=1,
            max_note_bytes=1_000,
            max_provider_bytes=4_000,
        ),
    )
    assert result.rejected_results == 1
    assert len(result.citation_ids) == 1


def test_frozen_mode_never_invokes_provider(tmp_path: Path) -> None:
    development = tmp_path / "development.json"
    frozen = tmp_path / "frozen.json"
    provider = FakeProvider([_note(1)])
    run_literature_research(
        manifest_path=development,
        eda_findings={},
        weaknesses="nDCG",
        mode="online",
        provider=provider,
        bounds=LiteratureBounds(max_queries=1),
    )
    freeze_manifest(development, frozen)

    result = run_literature_research(
        manifest_path=frozen,
        eda_findings={},
        weaknesses="nDCG",
        mode="frozen",
        provider=FailingProvider(),
        bounds=LiteratureBounds(max_queries=1),
    )
    assert result.cache_hits == 1
    assert result.provider_bytes == 0


def test_candidate_citations_resolve_in_order_and_reject_unknown(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = new_manifest(mode="development")
    manifest["notes"] = [_note(2), _note(1)]
    write_manifest(manifest_path, manifest)
    checked = load_manifest(manifest_path)
    ids = [note["citation_id"] for note in checked["notes"]]

    resolved = resolve_candidate_citations(
        {"literature_citations": [ids[1], ids[0], ids[1]]}, checked
    )
    assert [note["citation_id"] for note in resolved] == [ids[1], ids[0]]
    with pytest.raises(LiteratureValidationError, match="unknown"):
        resolve_candidate_citations(
            {"literature_citations": ["lit-0000000000000000"]}, checked
        )


def test_manifest_hash_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_manifest(path, new_manifest(mode="frozen"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cache_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LiteratureValidationError, match="cache_sha256"):
        load_manifest(path)
