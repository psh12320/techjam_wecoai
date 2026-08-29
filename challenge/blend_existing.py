"""Search leakage-safe blends among persisted validation predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "kuairand-starter-kit"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STARTER))

from challenge.reproduce_baseline import load_train_valid_only  # noqa: E402
from challenge.techjam_recsys.metrics import (  # noqa: E402
    evaluate,
    rank_normalize_within_user,
)


def user_zscore(users, scores):
    users = np.asarray(users)
    scores = np.asarray(scores, dtype=np.float64)
    output = np.empty_like(scores)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    for start, end in zip(starts, ends):
        positions = order[start:end]
        local = scores[positions]
        std = float(local.std())
        output[positions] = (local - local.mean()) / (std if std > 1e-8 else 1.0)
    return output


def transform(method, users, scores):
    if method == "rank":
        return rank_normalize_within_user(users, scores)
    if method == "zscore":
        return user_zscore(users, scores)
    raise ValueError(method)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fm", type=Path, required=True)
    parser.add_argument("--aux", type=Path, action="append", required=True)
    parser.add_argument("--method", choices=["rank", "zscore"], default="rank")
    parser.add_argument("--max-aux-weight", type=float, default=0.5)
    parser.add_argument("--step", type=float, default=0.025)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=STARTER / "KuaiRand-Pure" / "data",
    )
    args = parser.parse_args()

    splits = load_train_valid_only(args.data_dir)
    valid_rows = splits["valid"]
    users = np.asarray([row[1] for row in valid_rows])
    labels = np.asarray([row[6] for row in valid_rows], dtype=np.int8)
    fm_raw = np.load(args.fm)
    fm = transform(args.method, users, fm_raw)
    baseline = evaluate(users, labels, fm_raw)
    print("fm_raw=" + json.dumps(baseline, sort_keys=True))

    champion = (float(baseline["primary"]), "fm_raw", 0.0, baseline, fm_raw)
    for auxiliary_path in args.aux:
        auxiliary_raw = np.load(auxiliary_path)
        auxiliary = transform(args.method, users, auxiliary_raw)
        auxiliary_metric = evaluate(users, labels, auxiliary_raw)
        print(
            f"aux[{auxiliary_path.name}]="
            + json.dumps(auxiliary_metric, sort_keys=True)
        )
        weights = np.arange(0.0, args.max_aux_weight + args.step / 2.0, args.step)
        local = []
        for weight in weights:
            scores = (1.0 - weight) * fm + weight * auxiliary
            metric = evaluate(users, labels, scores)
            local.append(
                (
                    float(metric["primary"]),
                    auxiliary_path.name,
                    float(weight),
                    metric,
                    scores,
                )
            )
        best = max(local, key=lambda value: value[0])
        print(
            f"best[{auxiliary_path.name}]="
            + json.dumps(
                {
                    "aux_weight": best[2],
                    **{key: best[3][key] for key in ("GAUC", "nDCG@5", "primary")},
                },
                sort_keys=True,
            )
        )
        if best[0] > champion[0]:
            champion = best

    print(
        "TECHJAM_RESULT="
        + json.dumps(
            {
                "auxiliary": champion[1],
                "aux_weight": champion[2],
                "method": args.method,
                **{key: champion[3][key] for key in ("GAUC", "nDCG@5", "primary")},
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
