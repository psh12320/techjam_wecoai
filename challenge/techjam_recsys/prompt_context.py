"""Bounded, auditable prompt context for KuaiRand research campaigns."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FORBIDDEN_MEMORY_KEYS = {
    "artifact_ids",
    "artifact_path",
    "checkpoint",
    "code",
    "code_diff",
    "prediction",
    "predictions",
    "weights",
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_text(path: Path, *, max_chars: int) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if len(text) > max_chars:
        raise ValueError(f"Prompt section exceeds {max_chars} characters: {path}")
    return text


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _reject_unsafe_memory(value: Any, *, path: str = "memory") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_MEMORY_KEYS:
                raise ValueError(f"Unsafe reusable artifact field at {path}.{key}")
            _reject_unsafe_memory(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_unsafe_memory(child, path=f"{path}[{index}]")


def _bounded_json(value: dict[str, Any], *, max_chars: int) -> str:
    _reject_unsafe_memory(value)
    rendered = json.dumps(value, sort_keys=True, ensure_ascii=True, indent=2)
    if len(rendered) > max_chars:
        raise ValueError(f"Prompt JSON section exceeds {max_chars} characters")
    return rendered


def _experiment_memory_prompt(value: dict[str, Any], *, max_chars: int) -> str:
    """Project durable memory to a bounded prompt without artifact reuse."""

    _reject_unsafe_memory(value)
    projected: dict[str, Any] = {
        "schema_version": value.get("schema_version"),
        "content_sha256": value.get("content_sha256"),
        "entries": [],
    }
    entries = value.get("entries", [])
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        config = entry.get("configuration") or {}
        compact = {
            "node_id": entry.get("node_id"),
            "parent_trial_id": entry.get("parent_trial_id"),
            "model_family": entry.get("model_family"),
            "status": entry.get("status"),
            "decision": entry.get("decision"),
            "recovery_outcome": entry.get("recovery_outcome"),
            "outcome": entry.get("outcome"),
            "metrics": entry.get("metrics"),
            "candidate_exec_seconds": entry.get("candidate_exec_seconds"),
            "error_type": entry.get("error_type"),
            "scientific_change": config.get("scientific_change"),
            "change_scope": config.get("change_scope"),
            "preserved_parent_components": list(
                config.get("preserved_parent_components") or []
            )[:4],
            "features": list(config.get("features") or [])[:8],
            "losses": config.get("losses"),
            "target_metric": config.get("target_metric"),
            "role": config.get("role"),
            "external_role_gate_passed": config.get("external_role_gate_passed"),
            "runtime_change_accepted": config.get("runtime_change_accepted"),
            "change_decision_reason": config.get("change_decision_reason"),
            "fallback_parent_node_id": config.get("fallback_parent_node_id"),
        }
        compact = {
            key: child
            for key, child in compact.items()
            if child is not None and child != [] and child != {}
        }
        candidate = dict(projected)
        candidate["entries"] = [*projected["entries"], compact]
        rendered = json.dumps(
            candidate, sort_keys=True, ensure_ascii=True, separators=(",", ":")
        )
        if len(rendered) > max_chars:
            break
        projected = candidate
    projected["included_entries"] = len(projected["entries"])
    projected["available_entries"] = len(entries)
    return json.dumps(
        projected, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    )


def _literature_prompt(value: dict[str, Any], *, max_chars: int) -> str:
    """Project the largest bounded set of sanitized literature notes."""

    _reject_unsafe_memory(value)
    notes = value.get("notes", [])
    projected: dict[str, Any] = {
        "mode": value.get("mode"),
        "manifest_sha256": value.get("manifest_sha256"),
        "notes": [],
    }
    for note in notes if isinstance(notes, list) else []:
        if not isinstance(note, dict):
            continue
        compact = {
            key: note.get(key)
            for key in (
                "citation_id",
                "title",
                "year",
                "url",
                "technique",
                "effect",
                "cost",
                "risk",
                "applicability",
            )
            if note.get(key) is not None
        }
        candidate = {**projected, "notes": [*projected["notes"], compact]}
        rendered = json.dumps(candidate, sort_keys=True, ensure_ascii=True)
        if len(rendered) > max_chars:
            break
        projected = candidate
    projected["included_notes"] = len(projected["notes"])
    projected["available_notes"] = len(notes) if isinstance(notes, list) else 0
    return json.dumps(
        projected, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    )


@dataclass(frozen=True)
class PromptContext:
    hard_constraints: str
    experiment_memory: str
    research_menu: str
    eda_evidence: str
    literature_evidence: str

    def sections(self) -> dict[str, str]:
        """Return deliberately separated top-level prompt sections."""

        return {
            "Hard constraints": self.hard_constraints,
            "Experiment memory": self.experiment_memory,
            "Optional research menu": self.research_menu,
            "EDA evidence": self.eda_evidence,
            "Literature evidence": self.literature_evidence,
        }


def load_prompt_context(
    root: Path,
    *,
    experiment_memory_path: Path | None = None,
    eda_summary_path: Path | None = None,
    literature_manifest_path: Path | None = None,
) -> PromptContext:
    root = Path(root)
    hard = _load_text(root / "challenge/prompts/hard_constraints.md", max_chars=12_000)
    menu = _load_text(root / "challenge/prompts/research_menu.md", max_chars=12_000)
    memory = _load_json(
        experiment_memory_path
        or root / "challenge/research_memory/experiment_memory.json"
    )
    eda = _load_json(
        eda_summary_path or root / "challenge/research_memory/eda_summary.json"
    )
    literature = _load_json(
        literature_manifest_path
        or root / "challenge/research_memory/literature_manifest.json"
    )
    return PromptContext(
        hard_constraints=hard,
        experiment_memory=_experiment_memory_prompt(memory, max_chars=16_000),
        research_menu=menu,
        eda_evidence=_bounded_json(eda, max_chars=8_000),
        literature_evidence=_literature_prompt(literature, max_chars=18_000),
    )
