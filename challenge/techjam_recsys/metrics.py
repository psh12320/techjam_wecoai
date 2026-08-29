"""Thin, tested access to the organizer's immutable evaluation implementation."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ORGANIZER_EVALUATOR = ROOT / "kuairand-starter-kit" / "evaluate.py"


@lru_cache(maxsize=1)
def _organizer_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "techjam_organizer_evaluate", ORGANIZER_EVALUATOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load evaluator: {ORGANIZER_EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(
    user_ids: Iterable[object],
    labels: Iterable[int | float],
    scores: Iterable[int | float],
) -> dict[str, float | int]:
    """Run the exact organizer evaluator after validating prediction safety."""

    users = list(user_ids)
    target = np.asarray(list(labels))
    prediction = np.asarray(list(scores), dtype=np.float64)
    if len(users) != len(target) or len(target) != len(prediction):
        raise ValueError("users, labels, and scores must have identical lengths")
    if not np.isfinite(prediction).all():
        raise ValueError("scores contain NaN or infinity")
    if not np.isin(target, (0, 1)).all():
        raise ValueError("long_view labels must be binary")
    result = _organizer_module().evaluate(users, target.tolist(), prediction.tolist())
    return {
        "GAUC": float(result["GAUC"]),
        "nDCG@5": float(result["nDCG@5"]),
        "primary": float(result["primary"]),
        "users": int(result["users"]),
        "rows": int(result["rows"]),
    }


def rank_normalize_within_user(
    user_ids: Iterable[object], scores: Iterable[int | float]
) -> np.ndarray:
    """Map scores to deterministic within-user percentile ranks for blending."""

    users = np.asarray(list(user_ids))
    values = np.asarray(list(scores), dtype=np.float64)
    if len(users) != len(values):
        raise ValueError("users and scores must have identical lengths")
    output = np.empty(len(values), dtype=np.float64)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    for start, end in zip(starts, ends):
        positions = order[start:end]
        local_order = np.argsort(values[positions], kind="stable")
        ranks = np.empty(len(positions), dtype=np.float64)
        sorted_values = values[positions][local_order]
        tie_starts = np.r_[
            0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1
        ]
        tie_ends = np.r_[tie_starts[1:], len(positions)]
        for tie_start, tie_end in zip(tie_starts, tie_ends):
            average_rank = (tie_start + tie_end - 1) / 2.0
            ranks[local_order[tie_start:tie_end]] = average_rank
        if len(positions) > 1:
            ranks /= len(positions) - 1
        else:
            ranks[0] = 0.5
        output[positions] = ranks
    return output
