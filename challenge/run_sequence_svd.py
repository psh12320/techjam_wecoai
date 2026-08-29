"""Collaborative sequence/history branch using implicit truncated SVD."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from challenge.blend_existing import user_zscore  # noqa: E402
from challenge.techjam_recsys.metrics import evaluate  # noqa: E402
from challenge.techjam_recsys.protocol import (  # noqa: E402
    ExperimentLedger,
    TrialRecord,
    select_champion,
)

TRAIN_END = 20220421
VALID_START = 20220422
VALID_END = 20220428


def load_rows(data_dir: Path):
    columns = [
        "user_id",
        "video_id",
        "date",
        "time_ms",
        "long_view",
        "is_click",
    ]
    dtype = {
        "user_id": "int32",
        "video_id": "int32",
        "date": "int32",
        "time_ms": "int64",
        "long_view": "int8",
        "is_click": "int8",
    }
    train = pd.read_csv(
        data_dir / "log_standard_4_08_to_4_21_pure.csv",
        usecols=columns,
        dtype=dtype,
    )
    later = pd.read_csv(
        data_dir / "log_standard_4_22_to_5_08_pure.csv",
        usecols=columns,
        dtype=dtype,
    )
    valid = later.loc[
        (later["date"] >= VALID_START) & (later["date"] <= VALID_END)
    ].copy()
    if len(train) != 1_141_112 or len(valid) != 124_909:
        raise RuntimeError("Official temporal split sizes changed")
    return train, valid


def interaction_matrix(
    train: pd.DataFrame,
    user_count: int,
    item_count: int,
    *,
    half_life_days: float,
    click_weight: float,
):
    days_old = TRAIN_END - train["date"].to_numpy()
    temporal = np.exp(-np.log(2.0) * days_old / half_life_days)
    preference = train["long_view"].to_numpy(dtype=np.float64)
    if click_weight:
        preference += click_weight * train["is_click"].to_numpy(dtype=np.float64)
    weight = temporal * preference
    keep = weight > 0
    matrix = sparse.coo_matrix(
        (
            weight[keep].astype(np.float32),
            (
                train.loc[keep, "user_id"].to_numpy(),
                train.loc[keep, "video_id"].to_numpy(),
            ),
        ),
        shape=(user_count, item_count),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.data = np.log1p(matrix.data)
    return matrix


def bm25_weight(matrix: sparse.csr_matrix, k1: float = 1.2, b: float = 0.75):
    weighted = matrix.copy().tocsr()
    row_length = np.asarray(weighted.sum(axis=1)).ravel()
    average_length = max(float(row_length.mean()), 1e-6)
    document_frequency = np.asarray((weighted > 0).sum(axis=0)).ravel()
    idf = np.log1p(
        (weighted.shape[0] - document_frequency + 0.5) / (document_frequency + 0.5)
    )
    row_ids = np.repeat(np.arange(weighted.shape[0]), np.diff(weighted.indptr))
    denominator = weighted.data + k1 * (
        1.0 - b + b * row_length[row_ids] / average_length
    )
    weighted.data = (
        weighted.data * (k1 + 1.0) / denominator * idf[weighted.indices]
    ).astype(np.float32)
    return weighted


def score_factorization(
    fit_matrix,
    profile_matrix,
    valid_users,
    valid_items,
    *,
    components: int,
    seed: int,
):
    model = TruncatedSVD(
        n_components=components,
        algorithm="randomized",
        n_iter=7,
        random_state=seed,
    )
    model.fit(fit_matrix)
    user_factors = profile_matrix @ model.components_.T
    item_factors = model.components_.T
    dot = np.sum(user_factors[valid_users] * item_factors[valid_items], axis=1)
    user_norm = np.linalg.norm(user_factors, axis=1)
    item_norm = np.linalg.norm(item_factors, axis=1)
    denominator = user_norm[valid_users] * item_norm[valid_items]
    cosine = np.divide(
        dot, denominator, out=np.zeros_like(dot), where=denominator > 1e-8
    )
    return (
        dot.astype(np.float64),
        cosine.astype(np.float64),
        model.explained_variance_ratio_.sum(),
    )


def blend_with_fm(users, labels, fm_raw, auxiliary):
    fm = user_zscore(users, fm_raw)
    aux = user_zscore(users, auxiliary)
    candidates = []
    for weight in np.linspace(-0.5, 0.8, 53):
        scores = fm + weight * aux
        metric = evaluate(users, labels, scores)
        candidates.append((float(metric["primary"]), float(weight), metric, scores))
    return max(candidates, key=lambda value: value[0])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fm", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "kuairand-starter-kit" / "KuaiRand-Pure" / "data",
    )
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "challenge" / "runs" / "sequence_svd"
    )
    parser.add_argument("--components", type=int, default=96)
    parser.add_argument("--half-life-days", type=float, default=7.0)
    parser.add_argument("--click-weight", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    ledger = ExperimentLedger(args.run_dir / "iterations.jsonl")
    started = time.perf_counter()
    train, valid = load_rows(args.data_dir)
    user_count = int(max(train["user_id"].max(), valid["user_id"].max())) + 1
    item_count = int(max(train["video_id"].max(), valid["video_id"].max())) + 1
    matrix = interaction_matrix(
        train,
        user_count,
        item_count,
        half_life_days=args.half_life_days,
        click_weight=args.click_weight,
    )
    fit_matrix = bm25_weight(matrix)
    users = valid["user_id"].to_numpy()
    items = valid["video_id"].to_numpy()
    labels = valid["long_view"].to_numpy(dtype=np.int8)
    dot, cosine, explained = score_factorization(
        fit_matrix,
        matrix,
        users,
        items,
        components=args.components,
        seed=args.seed,
    )
    fm = np.load(args.fm)

    records = []
    predictions = {"svd_dot": dot, "svd_cosine": cosine}
    for iteration, (name, prediction) in enumerate(predictions.items()):
        direct = evaluate(users, labels, prediction)
        _, weight, blend, blend_prediction = blend_with_fm(
            users, labels, fm, prediction
        )
        record = TrialRecord(
            iteration=iteration,
            hypothesis=(
                "Represent each user's temporally weighted positive sequence in a "
                "BM25-debiased item latent space and add it as an FM residual."
            ),
            model_family=f"fm_{name}_blend",
            status="success",
            config={
                "components": args.components,
                "half_life_days": args.half_life_days,
                "click_weight": args.click_weight,
                "auxiliary_weight": weight,
                "explained_variance_ratio": float(explained),
            },
            metrics={key: float(blend[key]) for key in ("GAUC", "nDCG@5", "primary")},
            wall_seconds=time.perf_counter() - started,
            manual_interventions=0,
        )
        ledger.append(record)
        records.append(record)
        np.save(args.run_dir / f"valid_{name}.npy", prediction)
        np.save(args.run_dir / f"valid_fm_{name}_blend.npy", blend_prediction)
        print(
            f"{name}="
            + json.dumps(
                {
                    "direct": {
                        key: direct[key] for key in ("GAUC", "nDCG@5", "primary")
                    },
                    "blend": record.metrics,
                    "auxiliary_weight": weight,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    champion = select_champion(records)
    result = {
        "champion_family": champion.model_family,
        "metrics": champion.metrics,
        "config": champion.config,
        "total_wall_seconds": time.perf_counter() - started,
        "llm_tokens": 0,
        "test_labels_accessed": False,
    }
    (args.run_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("TECHJAM_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
