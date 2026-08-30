"""Deterministic chronological internal validation and fidelity contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

Fidelity = Literal["screen", "full"]


@dataclass(frozen=True)
class ChronologicalSplitManifest:
    protocol: str
    train_rows: int
    holdout_rows: int
    train_dates: tuple[int, ...]
    holdout_dates: tuple[int, ...]
    boundary_date: int
    max_train_time_ms: int
    min_holdout_time_ms: int
    index_sha256: str
    manifest_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hash_indices(train_indices: np.ndarray, holdout_indices: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(train_indices, dtype="<i8").tobytes())
    digest.update(b"\0")
    digest.update(np.asarray(holdout_indices, dtype="<i8").tobytes())
    return digest.hexdigest()


def _seal_manifest(manifest: ChronologicalSplitManifest) -> ChronologicalSplitManifest:
    payload = manifest.to_dict()
    payload["manifest_sha256"] = ""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return ChronologicalSplitManifest(
        **{**payload, "manifest_sha256": hashlib.sha256(canonical.encode()).hexdigest()}
    )


def last_days_holdout(
    frame: pd.DataFrame,
    *,
    holdout_days: int = 3,
) -> tuple[np.ndarray, np.ndarray, ChronologicalSplitManifest]:
    """Split on whole days so equal timestamps can never cross the boundary."""

    required = {"date", "time_ms"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("Missing chronological columns: " + ", ".join(sorted(missing)))
    if holdout_days < 1:
        raise ValueError("holdout_days must be positive")
    dates = pd.to_numeric(frame["date"], errors="raise").to_numpy(dtype=np.int64)
    times = pd.to_numeric(frame["time_ms"], errors="raise").to_numpy(dtype=np.int64)
    unique_dates = np.unique(dates)
    if len(unique_dates) <= holdout_days:
        raise ValueError("Need at least one training day before the holdout")
    boundary_date = int(unique_dates[-holdout_days])
    # KuaiRand's calendar-date labels overlap slightly in epoch time around the
    # day boundary. Anchor on the first timestamp from the last N labeled days,
    # then move every equal or later timestamp into holdout. This preserves a
    # strict event-time boundary even when some previous-date rows spill over.
    boundary_time = int(times[dates >= boundary_date].min())
    train_mask = times < boundary_time
    holdout_mask = times >= boundary_time
    train_indices = np.flatnonzero(train_mask)
    holdout_indices = np.flatnonzero(holdout_mask)
    if not len(train_indices) or not len(holdout_indices):
        raise ValueError("Chronological split produced an empty partition")
    max_train_time = int(times[train_mask].max())
    min_holdout_time = int(times[holdout_mask].min())
    if max_train_time >= min_holdout_time:
        raise ValueError(
            "time_ms is not strictly ordered across whole-day split; equal/future time leakage risk"
        )
    manifest = _seal_manifest(
        ChronologicalSplitManifest(
            protocol=f"last_{holdout_days}_train_days",
            train_rows=int(train_mask.sum()),
            holdout_rows=int(holdout_mask.sum()),
            train_dates=tuple(int(value) for value in np.unique(dates[train_mask])),
            holdout_dates=tuple(int(value) for value in np.unique(dates[holdout_mask])),
            boundary_date=boundary_date,
            max_train_time_ms=max_train_time,
            min_holdout_time_ms=min_holdout_time,
            index_sha256=_hash_indices(train_indices, holdout_indices),
        )
    )
    return train_mask, holdout_mask, manifest


def fidelity_can_qualify(fidelity: str) -> bool:
    """Only full-data, full-runtime executions may pass the final score gate."""

    if fidelity not in {"screen", "full"}:
        raise ValueError("fidelity must be 'screen' or 'full'")
    return fidelity == "full"
