"""Rank individual engineered signals and their residual lift over FM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from challenge.blend_existing import user_zscore  # noqa: E402
from challenge.techjam_recsys.data import load_development_splits  # noqa: E402
from challenge.techjam_recsys.features import build_features  # noqa: E402
from challenge.techjam_recsys.metrics import evaluate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fm", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "kuairand-starter-kit" / "KuaiRand-Pure" / "data",
    )
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    splits = load_development_splits(args.data_dir)
    matrices = build_features(splits)
    fm_raw = np.load(args.fm)
    fm = user_zscore(matrices.valid_users, fm_raw)
    baseline = evaluate(matrices.valid_users, matrices.valid_y, fm_raw)
    rows = []
    for column in matrices.feature_names:
        if column in matrices.categorical_features:
            continue
        raw = matrices.valid_x[column].to_numpy(dtype=np.float64)
        direct = evaluate(matrices.valid_users, matrices.valid_y, raw)
        rows.append((float(direct["primary"]), column, direct, raw))
    rows.sort(reverse=True, key=lambda row: row[0])

    results = []
    for _, column, direct, raw in rows[: args.top]:
        auxiliary = user_zscore(matrices.valid_users, raw)
        blends = []
        for weight in np.linspace(-0.5, 0.5, 21):
            scores = fm + weight * auxiliary
            metric = evaluate(matrices.valid_users, matrices.valid_y, scores)
            blends.append((float(metric["primary"]), float(weight), metric))
        best = max(blends, key=lambda row: row[0])
        result = {
            "feature": column,
            "direct_primary": float(direct["primary"]),
            "best_weight": best[1],
            "GAUC": float(best[2]["GAUC"]),
            "nDCG@5": float(best[2]["nDCG@5"]),
            "primary": float(best[2]["primary"]),
            "lift_over_fm": float(best[2]["primary"] - baseline["primary"]),
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    print(
        "TECHJAM_RESULT="
        + json.dumps(
            {
                "fm": {key: baseline[key] for key in ("GAUC", "nDCG@5", "primary")},
                "features": results,
                "test_labels_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
