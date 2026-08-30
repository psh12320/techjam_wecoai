"""Build safe, frozen experiment memory from hash-validated AIDE ledgers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from challenge.techjam_recsys.prompt_context import canonical_sha256  # noqa: E402
from challenge.techjam_recsys.protocol import (  # noqa: E402
    ChallengeMetric,
    ExperimentLedger,
    TrialRecord,
)


SAFE_CONFIG_KEYS = {
    "abort_criteria",
    "eda_observation_ids",
    "estimated_memory_mb",
    "estimated_runtime_seconds",
    "expected_metric_effects",
    "falsification_condition",
    "features",
    "fidelity",
    "hyperparameters",
    "internal_validation",
    "literature_citation_ids",
    "losses",
    "model_family",
    "risks",
    "scientific_change",
    "change_scope",
    "preserved_parent_components",
    "target_metric",
}


def _safe_config(record: TrialRecord) -> dict[str, Any]:
    return {
        key: value
        for key, value in (record.config or {}).items()
        if key in SAFE_CONFIG_KEYS
    }


def _outcome(record: TrialRecord, records_by_trial_id: dict[str, TrialRecord]) -> str:
    if not record.metrics:
        return "execution_failure"
    parent = records_by_trial_id.get(str(record.parent_trial_id or ""))
    if parent is None or not parent.metrics:
        return "evaluated"
    keys = ("GAUC", "nDCG@5", "primary")
    child_metrics = {key: float(record.metrics[key]) for key in keys}
    parent_metrics = {key: float(parent.metrics[key]) for key in keys}
    if all(child_metrics[key] > parent_metrics[key] for key in ("GAUC", "nDCG@5")):
        return "improved_both_components"
    if all(parent_metrics[key] >= child_metrics[key] for key in keys) and any(
        parent_metrics[key] > child_metrics[key] for key in keys
    ):
        return "parent_dominated"
    return "component_tradeoff"


def _entry(
    record: TrialRecord, records_by_trial_id: dict[str, TrialRecord]
) -> dict[str, Any]:
    metrics = dict(record.metrics) if record.metrics else None
    component_deltas = None
    if metrics:
        component_deltas = ChallengeMetric.from_mapping(metrics).champion_deltas
    return {
        "trial_id": record.trial_id,
        "parent_trial_id": record.parent_trial_id,
        "node_id": record.node_id,
        "source_record_sha256": record.record_sha256,
        "code_sha256": record.code_sha256,
        "model_family": record.model_family,
        "assignment_family": record.assignment_family,
        "status": record.status,
        "decision": record.decision,
        "recovery_outcome": record.recovery_outcome,
        "outcome": _outcome(record, records_by_trial_id),
        "metrics": metrics,
        "champion_deltas": component_deltas,
        "wall_seconds": record.wall_seconds,
        "candidate_exec_seconds": record.candidate_exec_seconds,
        "error_type": record.error_type,
        "configuration": _safe_config(record),
    }


def build_memory(ledger_paths: list[Path], *, max_entries: int = 16) -> dict[str, Any]:
    records: list[TrialRecord] = []
    source_hashes: list[str] = []
    for path in ledger_paths:
        path = Path(path)
        loaded = ExperimentLedger(path).read(validate_chain=True)
        records.extend(loaded)
        source_hashes.extend(
            record.record_sha256 for record in loaded if record.record_sha256
        )

    records_by_trial_id = {record.trial_id: record for record in records}

    # Keep the strongest positive evidence, metric-bearing dead ends, and hard
    # failures. The sort is deterministic and contains no code, predictions, or
    # artifact paths.
    def rank(record: TrialRecord) -> tuple[int, float, float]:
        primary = float((record.metrics or {}).get("primary", float("-inf")))
        return (1 if record.metrics else 0, primary, record.created_at_unix)

    evaluated = [record for record in records if record.metrics]
    successes = sorted(
        (
            record
            for record in evaluated
            if _outcome(record, records_by_trial_id) != "parent_dominated"
        ),
        key=rank,
        reverse=True,
    )
    regressions = sorted(
        (
            record
            for record in evaluated
            if _outcome(record, records_by_trial_id) == "parent_dominated"
        ),
        key=rank,
        reverse=True,
    )
    failures = sorted(
        (record for record in records if not record.metrics),
        key=lambda record: (record.created_at_unix, record.trial_id),
        reverse=True,
    )
    # Always reserve two slots for the most recent evaluated descendants. A
    # pure score/family ranking can otherwise omit the newest component
    # trade-off and a gated fallback, causing the next paid campaign to repeat
    # the experiment it just completed.
    recent_slots = min(2, max_entries)
    recent = sorted(
        (
            record
            for record in evaluated
            if record.parent_trial_id and record.model_family != "official_fm_seed"
        ),
        key=lambda record: (record.created_at_unix, record.trial_id),
        reverse=True,
    )[:recent_slots]
    failure_slots = min(4, max_entries // 4)
    regression_slots = min(4, max_entries // 4)
    success_slots = max(
        0, max_entries - recent_slots - failure_slots - regression_slots
    )

    def diverse(values: list[TrialRecord], limit: int, key) -> list[TrialRecord]:
        if limit <= 0:
            return []
        chosen: list[TrialRecord] = []
        seen: set[Any] = set()
        for record in values:
            identity = key(record)
            if identity in seen:
                continue
            chosen.append(record)
            seen.add(identity)
            if len(chosen) >= limit:
                return chosen
        for record in values:
            if record not in chosen:
                chosen.append(record)
                if len(chosen) >= limit:
                    break
        return chosen

    selected = list(recent)
    selected += diverse(successes, success_slots, lambda record: record.model_family)
    selected += diverse(
        regressions,
        regression_slots,
        lambda record: (record.model_family, _safe_config(record).get("change_scope")),
    )
    selected += diverse(
        failures,
        failure_slots,
        lambda record: (record.model_family, record.error_type),
    )
    deduplicated: list[TrialRecord] = []
    seen_trials: set[str] = set()
    for record in selected:
        if record.trial_id in seen_trials:
            continue
        deduplicated.append(record)
        seen_trials.add(record.trial_id)
    # Fill any quota lost to overlap while retaining the explicit priority:
    # newest descendants, strongest/diverse successes, regressions, failures.
    fallback = [*successes, *regressions, *failures]
    for record in fallback:
        if len(deduplicated) >= max_entries:
            break
        if record.trial_id not in seen_trials:
            deduplicated.append(record)
            seen_trials.add(record.trial_id)
    selected = deduplicated[:max_entries]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "source": "hash-validated AIDE experiment ledgers",
        "source_record_hashes": sorted(source_hashes),
        "entries": [_entry(record, records_by_trial_id) for record in selected],
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledgers", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-entries", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    memory = build_memory(args.ledgers, max_entries=args.max_entries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(memory, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(args.output), "content_sha256": memory["content_sha256"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
