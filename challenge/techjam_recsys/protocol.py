"""Challenge limits, convergence, champion selection, and audit logging."""

from __future__ import annotations

import json
import hashlib
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

BASELINE_VALID = {"GAUC": 0.6674, "nDCG@5": 0.5357, "primary": 0.6016}
CHAMPION_VALID = {
    "GAUC": 0.6710518008586268,
    "nDCG@5": 0.5380142516919405,
    "primary": 0.6045330262752837,
}
MAX_ITERATIONS = 50
MAX_WALL_SECONDS = 6 * 60 * 60
CONVERGENCE_EPSILON = 0.002
CONVERGENCE_PATIENCE = 3


@dataclass(frozen=True)
class ChallengeMetric:
    gauc: float
    ndcg5: float

    @property
    def primary(self) -> float:
        return (self.gauc + self.ndcg5) / 2.0

    @property
    def baseline_deltas(self) -> dict[str, float]:
        return {
            "GAUC": self.gauc - BASELINE_VALID["GAUC"],
            "nDCG@5": self.ndcg5 - BASELINE_VALID["nDCG@5"],
            "primary": self.primary - BASELINE_VALID["primary"],
        }

    @property
    def clears_quality_gate(self) -> bool:
        deltas = self.baseline_deltas
        return deltas["GAUC"] > 0 and deltas["nDCG@5"] > 0

    @property
    def champion_deltas(self) -> dict[str, float]:
        return {
            "GAUC": self.gauc - CHAMPION_VALID["GAUC"],
            "nDCG@5": self.ndcg5 - CHAMPION_VALID["nDCG@5"],
            "primary": self.primary - CHAMPION_VALID["primary"],
        }

    @property
    def beats_champion(self) -> bool:
        deltas = self.champion_deltas
        return deltas["GAUC"] > 0 and deltas["nDCG@5"] > 0 and deltas["primary"] > 0

    @classmethod
    def from_mapping(cls, value: dict[str, float]) -> "ChallengeMetric":
        return cls(gauc=float(value["GAUC"]), ndcg5=float(value["nDCG@5"]))


@dataclass
class ConvergenceTracker:
    epsilon: float = CONVERGENCE_EPSILON
    patience: int = CONVERGENCE_PATIENCE
    max_iterations: int = MAX_ITERATIONS
    max_wall_seconds: float = MAX_WALL_SECONDS
    # Epoch time deliberately includes host suspension in the six-hour wall cap.
    started_at: float = field(default_factory=time.time)
    best_primary: float = float("-inf")
    best_iteration: int | None = None
    insignificant_iterations: int = 0
    observations: list[float | None] = field(default_factory=list)

    def observe(self, primary: float) -> bool:
        """Record a completed iteration and return whether the run must stop."""

        iteration = len(self.observations)
        self.observations.append(float(primary))
        improvement = primary - self.best_primary
        if improvement > self.epsilon:
            self.best_primary = float(primary)
            self.best_iteration = iteration
            self.insignificant_iterations = 0
        else:
            if primary > self.best_primary:
                self.best_primary = float(primary)
                self.best_iteration = iteration
            self.insignificant_iterations += 1
        return self.should_stop

    def observe_failure(self) -> bool:
        """Count a failed iteration against both the hard cap and patience."""

        self.observations.append(None)
        self.insignificant_iterations += 1
        return self.should_stop

    @property
    def should_stop(self) -> bool:
        return (
            self.insignificant_iterations >= self.patience
            or len(self.observations) >= self.max_iterations
            or time.time() - self.started_at >= self.max_wall_seconds
        )

    @property
    def stop_reason(self) -> str | None:
        if self.insignificant_iterations >= self.patience:
            return "converged"
        if len(self.observations) >= self.max_iterations:
            return "iteration_cap"
        if time.time() - self.started_at >= self.max_wall_seconds:
            return "wall_clock_cap"
        return None


@dataclass
class TrialRecord:
    iteration: int
    hypothesis: str
    model_family: str
    status: str
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] | None = None
    parent_trial_id: str | None = None
    code_diff: str | None = None
    error: str | None = None
    recovery: str | None = None
    # End-to-end trial latency, including LLM generation and host suspension.
    wall_seconds: float = 0.0
    # Interpreter-reported candidate runtime, bounded by the per-trial timeout.
    # None keeps ledgers written before this field backwards-compatible.
    candidate_exec_seconds: float | None = None
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    gpu_hours: float = 0.0
    manual_interventions: int = 0
    source: str = "unknown"
    node_id: str | None = None
    code_sha256: str | None = None
    prompt_sha256: str | None = None
    prompt_version: str | None = None
    assignment_family: str | None = None
    seed: int | None = None
    evaluation_fidelity: str = "unknown"
    evaluator_sha256: str | None = None
    artifact_ids: list[str] = field(default_factory=list)
    artifact_sha256: dict[str, str] = field(default_factory=dict)
    run_id: str | None = None
    campaign_manifest_sha256: str | None = None
    declared_model_family: str | None = None
    assignment_compliant: bool | None = None
    scheduler_action: str | None = None
    scheduler_reason: str | None = None
    scheduler_utility: float | None = None
    scheduler_alternatives: list[dict[str, Any]] = field(default_factory=list)
    scheduler_feature_vector: dict[str, float] = field(default_factory=dict)
    pareto_frontier_member: bool | None = None
    expected_api_cost_usd: float | None = None
    actual_api_cost_usd: float | None = None
    internal_validation_sha256: str | None = None
    diagnostics_sha256: str | None = None
    decision: str | None = None
    recovery_outcome: str | None = None
    error_type: str | None = None
    error_info: dict[str, Any] | None = None
    exit_status: str | None = None
    source_sha256: str | None = None
    input_sha256: str | None = None
    dependency_sha256: str | None = None
    previous_record_sha256: str | None = None
    record_sha256: str | None = None
    trial_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at_unix: float = field(default_factory=time.time)


class ExperimentLedger:
    """Append-only JSONL ledger matching the organizer's run-log requirements."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: TrialRecord) -> None:
        previous = None
        existing = self.read(validate_chain=True)
        if existing:
            previous = existing[-1].record_sha256
        record.previous_record_sha256 = previous
        unsigned = asdict(record)
        unsigned["record_sha256"] = None
        canonical = json.dumps(
            unsigned, sort_keys=True, allow_nan=False, separators=(",", ":")
        )
        record.record_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        payload = json.dumps(asdict(record), sort_keys=True, allow_nan=False)
        descriptor = os.open(
            self.path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            os.write(descriptor, (payload + "\n").encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def read(self, *, validate_chain: bool = True) -> list[TrialRecord]:
        if not self.path.exists():
            return []
        records = []
        previous = None
        chain_started = False
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    record = TrialRecord(**payload)
                    if validate_chain and record.record_sha256 is not None:
                        chain_started = True
                        if record.previous_record_sha256 != previous:
                            raise ValueError("Experiment ledger hash chain is broken")
                        # Verify exactly the schema that was serialized. Rebuilding
                        # from ``asdict(record)`` would inject defaults for fields
                        # added later and invalidate otherwise sound legacy chains.
                        unsigned = dict(payload)
                        claimed = unsigned.pop("record_sha256")
                        unsigned["record_sha256"] = None
                        canonical = json.dumps(
                            unsigned,
                            sort_keys=True,
                            allow_nan=False,
                            separators=(",", ":"),
                        )
                        actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                        if actual != claimed:
                            raise ValueError("Experiment ledger record hash is invalid")
                        previous = claimed
                    elif validate_chain and chain_started:
                        raise ValueError(
                            "Experiment ledger mixes chained and unchained records"
                        )
                    records.append(record)
        return records


def count_manual_interventions(path: Path) -> int:
    """Derive intervention count from append-only events instead of a constant."""

    path = Path(path)
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid intervention event on line {line_number}: {exc}"
                ) from exc
            if (
                event.get("actor") == "human"
                and event.get("event_type") == "intervention"
            ):
                count += 1
    return count


def select_champion(records: Iterable[TrialRecord]) -> TrialRecord:
    """Select validation best, preferring candidates that beat both component metrics."""

    successful = [record for record in records if record.metrics is not None]
    if not successful:
        raise ValueError("No successful trials are available")

    def rank(record: TrialRecord) -> tuple[bool, bool, float, float]:
        metric = ChallengeMetric.from_mapping(record.metrics or {})
        minimum_component_delta = min(
            metric.baseline_deltas["GAUC"], metric.baseline_deltas["nDCG@5"]
        )
        return (
            metric.beats_champion,
            metric.clears_quality_gate,
            metric.primary,
            minimum_component_delta,
        )

    return max(successful, key=rank)
