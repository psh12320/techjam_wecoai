"""Out-of-time LambdaRank residual stacked on the official FM."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "kuairand-starter-kit"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STARTER))

import baseline as organizer_baseline  # noqa: E402
from data import encode  # noqa: E402

from challenge.blend_existing import user_zscore  # noqa: E402
from challenge.reproduce_baseline import load_train_valid_only  # noqa: E402
from challenge.techjam_recsys.data import (  # noqa: E402
    DatasetSplits,
    load_development_splits,
)
from challenge.techjam_recsys.features import build_features  # noqa: E402
from challenge.techjam_recsys.metrics import evaluate  # noqa: E402
from challenge.techjam_recsys.protocol import (  # noqa: E402
    ExperimentLedger,
    TrialRecord,
)


def train_early_fm(raw_splits, cutoff, seed, max_epochs=30, batch_size=8192):
    all_train = raw_splits["train"]
    early = [row for row in all_train if row[0] <= cutoff]
    meta = [row for row in all_train if row[0] > cutoff]
    meta_row_ids = np.asarray(
        [index for index, row in enumerate(all_train) if row[0] > cutoff],
        dtype=np.int32,
    )
    encoded, dimension = encode({"train": early, "valid": meta, "test": []})
    train_x, train_y, _ = encoded["train"]
    meta_x, meta_y, meta_users = encoded["valid"]
    model = organizer_baseline.FM(dimension, k=16, lr=0.001, seed=seed)
    rng = np.random.default_rng(seed)
    best = -1.0
    best_state = None
    bad_epochs = 0
    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(len(train_y))
        for offset in range(0, len(order), batch_size):
            batch = order[offset : offset + batch_size]
            model.step(train_x[batch], train_y[batch])
        prediction = model.predict(meta_x)
        metric = evaluate(meta_users, meta_y, prediction)
        print(
            f"early_fm epoch={epoch:02d} meta_GAUC={metric['GAUC']:.6f} "
            f"meta_nDCG@5={metric['nDCG@5']:.6f} "
            f"meta_primary={metric['primary']:.6f}",
            flush=True,
        )
        if metric["primary"] > best + 1e-5:
            best = float(metric["primary"])
            bad_epochs = 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad_epochs += 1
            if bad_epochs >= 4:
                break
    if best_state is None:
        raise RuntimeError("Early FM produced no checkpoint")
    model.V, model.W, model.b = best_state
    return meta_row_ids, model.predict(meta_x)


def group_sizes(users):
    return np.unique(users, return_counts=True)[1].astype(np.int32)


def fit_residual_ranker(meta_features, meta_dates, seed, estimators):
    train_mask = meta_dates < meta_dates.max()
    eval_mask = ~train_mask
    train_indices = np.flatnonzero(train_mask)
    eval_indices = np.flatnonzero(eval_mask)
    train_indices = train_indices[
        np.argsort(meta_features.valid_users[train_indices], kind="stable")
    ]
    eval_indices = eval_indices[
        np.argsort(meta_features.valid_users[eval_indices], kind="stable")
    ]
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[5],
        lambdarank_truncation_level=10,
        label_gain=[0, 1],
        n_estimators=estimators,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=100,
        max_bin=255,
        colsample_bytree=0.9,
        reg_alpha=0.5,
        reg_lambda=8.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        meta_features.valid_x.iloc[train_indices],
        meta_features.valid_y[train_indices],
        group=group_sizes(meta_features.valid_users[train_indices]),
        categorical_feature=meta_features.categorical_features,
        eval_set=[
            (
                meta_features.valid_x.iloc[eval_indices],
                meta_features.valid_y[eval_indices],
            )
        ],
        eval_group=[group_sizes(meta_features.valid_users[eval_indices])],
        callbacks=[lgb.early_stopping(35, verbose=True)],
    )
    return model


def search_blend(users, labels, fm_raw, residual_raw):
    fm = user_zscore(users, fm_raw)
    residual = user_zscore(users, residual_raw)
    candidates = []
    for residual_weight in np.linspace(-0.25, 1.0, 51):
        scores = fm + residual_weight * residual
        metric = evaluate(users, labels, scores)
        candidates.append(
            (
                float(metric["primary"]),
                float(residual_weight),
                metric,
                scores,
            )
        )
    return max(candidates, key=lambda value: value[0])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fm", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=STARTER / "KuaiRand-Pure" / "data",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "challenge" / "runs" / "residual_ranker",
    )
    parser.add_argument("--cutoff", type=int, default=20220417)
    parser.add_argument("--estimators", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    ledger = ExperimentLedger(args.run_dir / "iterations.jsonl")
    started = time.perf_counter()

    raw_splits = load_train_valid_only(args.data_dir)
    meta_row_ids, meta_fm = train_early_fm(raw_splits, args.cutoff, args.seed)
    del raw_splits
    gc.collect()

    full = load_development_splits(args.data_dir)
    early_frame = full.train.loc[full.train["date"] <= args.cutoff].copy()
    meta_frame = (
        full.train.loc[full.train["date"] > args.cutoff]
        .copy()
        .sort_values("_row_id", kind="stable")
        .reset_index(drop=True)
    )
    prediction_by_row = np.full(len(full.train), np.nan, dtype=np.float64)
    prediction_by_row[meta_row_ids] = meta_fm
    aligned_meta_fm = prediction_by_row[meta_frame["_row_id"].to_numpy()]
    if not np.isfinite(aligned_meta_fm).all():
        raise RuntimeError("Could not align out-of-time FM predictions")
    meta_split = DatasetSplits(
        train=early_frame,
        valid=meta_frame,
        valid_users=meta_frame["user_id"].to_numpy(copy=True),
        valid_labels=meta_frame["long_view"].to_numpy(dtype=np.int8, copy=True),
    )
    meta_features = build_features(meta_split)
    meta_features.valid_x["fm_score"] = aligned_meta_fm.astype(np.float32)
    meta_features.valid_x["fm_user_zscore"] = user_zscore(
        meta_features.valid_users, aligned_meta_fm
    ).astype(np.float32)
    meta_features.feature_names.extend(["fm_score", "fm_user_zscore"])
    meta_dates = meta_frame["date"].to_numpy(copy=True)
    model = fit_residual_ranker(meta_features, meta_dates, args.seed, args.estimators)
    del meta_features, meta_split, meta_frame, early_frame
    gc.collect()

    final_features = build_features(full)
    full_fm = np.load(args.fm)
    final_features.valid_x["fm_score"] = full_fm.astype(np.float32)
    final_features.valid_x["fm_user_zscore"] = user_zscore(
        final_features.valid_users, full_fm
    ).astype(np.float32)
    residual = model.predict(final_features.valid_x)
    direct = evaluate(final_features.valid_users, final_features.valid_y, residual)
    _, weight, blend, blend_prediction = search_blend(
        final_features.valid_users,
        final_features.valid_y,
        full_fm,
        residual,
    )
    record = TrialRecord(
        iteration=0,
        hypothesis=(
            "Train a LambdaRank correction on truly out-of-time FM predictions "
            "and apply the learned error pattern to the full-train FM."
        ),
        model_family="fm_out_of_time_lambdarank_residual",
        status="success",
        config={
            "cutoff": args.cutoff,
            "estimators": args.estimators,
            "best_iteration": int(model.best_iteration_),
            "residual_weight": weight,
        },
        metrics={key: float(blend[key]) for key in ("GAUC", "nDCG@5", "primary")},
        wall_seconds=time.perf_counter() - started,
        manual_interventions=0,
    )
    ledger.append(record)
    np.save(args.run_dir / "valid_residual.npy", residual)
    np.save(args.run_dir / "valid_blend.npy", blend_prediction)
    result = {
        "direct_residual": {key: direct[key] for key in ("GAUC", "nDCG@5", "primary")},
        "blend": record.metrics,
        "config": record.config,
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
