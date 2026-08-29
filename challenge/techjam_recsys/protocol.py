"""Challenge limits, convergence, champion selection, and audit logging."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

BASELINE_VALID = {"GAUC": 0.6674, "nDCG@5": 0.5357, "primary": 0.6016}
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

    @classmethod
    def from_mapping(cls, value: dict[str, float]) -> "ChallengeMetric":
        return cls(gauc=float(value["GAUC"]), ndcg5=float(value["nDCG@5"]))


@dataclass
class ConvergenceTracker:
    epsilon: float = CONVERGENCE_EPSILON
    patience: int = CONVERGENCE_PATIENCE
    max_iterations: int = MAX_ITERATIONS
    max_wall_seconds: float = MAX_WALL_SECONDS
    started_at: float = field(default_factory=time.monotonic)
    best_primary: float = float("-inf")
    best_iteration: int | None = None
    insignificant_iterations: int = 0
    observations: list[float] = field(default_factory=list)

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

    @property
    def should_stop(self) -> bool:
        return (
            self.insignificant_iterations >= self.patience
            or len(self.observations) >= self.max_iterations
            or time.monotonic() - self.started_at >= self.max_wall_seconds
        )

    @property
    def stop_reason(self) -> str | None:
        if self.insignificant_iterations >= self.patience:
            return "converged"
        if len(self.observations) >= self.max_iterations:
            return "iteration_cap"
        if time.monotonic() - self.started_at >= self.max_wall_seconds:
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
    wall_seconds: float = 0.0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    gpu_hours: float = 0.0
    manual_interventions: int = 0
    trial_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at_unix: float = field(default_factory=time.time)


class ExperimentLedger:
    """Append-only JSONL ledger matching the organizer's run-log requirements."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: TrialRecord) -> None:
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

    def read(self) -> list[TrialRecord]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(TrialRecord(**json.loads(line)))
        return records


def select_champion(records: Iterable[TrialRecord]) -> TrialRecord:
    """Select validation best, preferring candidates that beat both component metrics."""

    successful = [record for record in records if record.metrics is not None]
    if not successful:
        raise ValueError("No successful trials are available")

    def rank(record: TrialRecord) -> tuple[bool, float, float]:
        metric = ChallengeMetric.from_mapping(record.metrics or {})
        minimum_component_delta = min(
            metric.baseline_deltas["GAUC"], metric.baseline_deltas["nDCG@5"]
        )
        return metric.clears_quality_gate, metric.primary, minimum_component_delta

    return max(successful, key=rank)
