"""Exact-metric simplex search over complementary validation predictions."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from challenge.blend_existing import transform  # noqa: E402
from challenge.techjam_recsys.metrics import evaluate  # noqa: E402


def simplex_weights(model_count: int, step: float):
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("step must divide 1.0 exactly")
    for cuts in itertools.combinations_with_replacement(
        range(units + 1), model_count - 1
    ):
        points = (0,) + cuts + (units,)
        yield np.diff(points).astype(np.float64) / units


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--method", choices=("rank", "zscore"), default="zscore")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument(
        "--validation-index",
        type=Path,
        default=ROOT / "challenge" / "agent_data" / "validation_index.npz",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "challenge" / "runs" / "best_blend.npy"
    )
    args = parser.parse_args()
    if not 2 <= len(args.model) <= 5:
        raise ValueError("Use between two and five models")
    index = np.load(args.validation_index)
    users = index["user_id"]
    labels = index["long_view"]
    raw = [np.load(path) for path in args.model]
    if any(len(scores) != len(labels) for scores in raw):
        raise ValueError("Every prediction must align with validation_index.npz")
    scores = [transform(args.method, users, values) for values in raw]

    champion = None
    evaluated = 0
    for weights in simplex_weights(len(scores), args.step):
        blend = sum(weight * values for weight, values in zip(weights, scores))
        metric = evaluate(users, labels, blend)
        candidate = (float(metric["primary"]), weights, metric, blend)
        if champion is None or candidate[0] > champion[0]:
            champion = candidate
        evaluated += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, champion[3])
    result = {
        "models": [str(path) for path in args.model],
        "method": args.method,
        "step": args.step,
        "evaluated": evaluated,
        "weights": champion[1].tolist(),
        "metrics": champion[2],
        "output": str(args.output),
    }
    print("TECHJAM_RESULT=" + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
