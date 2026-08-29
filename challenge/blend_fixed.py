"""Evaluate a predeclared fixed blend without retuning validation weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from challenge.blend_existing import transform  # noqa: E402
from challenge.techjam_recsys.metrics import evaluate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--weight", type=float, action="append", required=True)
    parser.add_argument("--method", choices=("rank", "zscore"), default="zscore")
    parser.add_argument(
        "--validation-index",
        type=Path,
        default=ROOT / "challenge" / "agent_data" / "validation_index.npz",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(args.model) != len(args.weight):
        raise ValueError("Each model needs one weight")
    if not np.isclose(sum(args.weight), 1.0):
        raise ValueError("Weights must sum to 1")
    index = np.load(args.validation_index)
    users = index["user_id"]
    labels = index["long_view"]
    blend = np.zeros(len(labels), dtype=np.float64)
    for path, weight in zip(args.model, args.weight):
        values = np.load(path)
        blend += weight * transform(args.method, users, values)
    metric = evaluate(users, labels, blend)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, blend)
    result = {
        "models": [str(path) for path in args.model],
        "weights": args.weight,
        "method": args.method,
        "metrics": metric,
        "output": str(args.output) if args.output else None,
    }
    print("TECHJAM_RESULT=" + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
