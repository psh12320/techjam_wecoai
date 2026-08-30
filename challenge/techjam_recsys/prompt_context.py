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
    for entry in value.get("entries", []):
        if not isinstance(entry, dict):
            continue
        config = entry.get("configuration") or {}
        compact = {
            "node_id": entry.get("node_id"),
            "parent_trial_id": entry.get("parent_trial_id"),
            "code_sha256": entry.get("code_sha256"),
            "model_family": entry.get("model_family"),
            "status": entry.get("status"),
            "decision": entry.get("decision"),
            "recovery_outcome": entry.get("recovery_outcome"),
            "outcome": entry.get("outcome"),
            "metrics": entry.get("metrics"),
            "champion_deltas": entry.get("champion_deltas"),
            "candidate_exec_seconds": entry.get("candidate_exec_seconds"),
            "error_type": entry.get("error_type"),
            "scientific_change": config.get("scientific_change"),
            "change_scope": config.get("change_scope"),
            "preserved_parent_components": list(
                config.get("preserved_parent_components") or []
            )[:8],
            "features": list(config.get("features") or [])[:12],
            "losses": config.get("losses"),
            "target_metric": config.get("target_metric"),
            "expected_metric_effects": config.get("expected_metric_effects"),
            "risks": list(config.get("risks") or [])[:2],
        }
        candidate = dict(projected)
        candidate["entries"] = [*projected["entries"], compact]
        rendered = json.dumps(candidate, sort_keys=True, ensure_ascii=True, indent=2)
        if len(rendered) > max_chars:
            break
        projected = candidate
    projected["included_entries"] = len(projected["entries"])
    projected["available_entries"] = len(value.get("entries", []))
    return json.dumps(projected, sort_keys=True, ensure_ascii=True, indent=2)


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
        literature_evidence=_bounded_json(literature, max_chars=12_000),
    )
