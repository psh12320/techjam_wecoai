"""Fit the frozen validation champion and write a hidden-label-free submission."""

from __future__ import annotations

import argparse
import csv
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

from challenge.blend_existing import user_zscore  # noqa: E402
from challenge.run_enriched_fm import (  # noqa: E402
    VARIANTS,
    FieldGatedFM,
    add_train_derived_buckets,
    encode_fields,
)
from challenge.run_portfolio import _group_sizes, ranker_model  # noqa: E402
from challenge.run_rad import (  # noqa: E402
    CATEGORICAL as RAD_CATEGORICAL,
    NUMERIC as RAD_NUMERIC,
    add_duration_bins,
    midrank_cdf,
)
from challenge.techjam_recsys.data import (  # noqa: E402
    DatasetSplits,
    SubmissionSplits,
    load_submission_splits,
)
from challenge.techjam_recsys.features import build_features  # noqa: E402

COMPONENT_WEIGHTS = {"rich": 0.8, "ranker": 0.1, "rad": 0.1}
FORBIDDEN_SCORE_COLUMNS = {
    "long_view",
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "play_time_ms",
}


def train_rich(
    splits: SubmissionSplits,
    *,
    seeds: list[int],
    epochs: int,
    batch_size: int,
) -> np.ndarray:
    add_train_derived_buckets(splits.train, splits.score)
    fields = VARIANTS["rich_lite"]
    train_x, score_x, dimension, _ = encode_fields(splits.train, splits.score, fields)
    labels = splits.train["long_view"].to_numpy(dtype=np.float32)
    ensemble = np.zeros(len(splits.score), dtype=np.float64)
    for seed in seeds:
        model = FieldGatedFM(
            dimension,
            fields=len(fields),
            core_fields=5,
            k=16,
            lr=0.001,
            l2=1e-6,
            seed=seed,
            extra_gate_init=0.1,
            gate_l2=0.001,
        )
        rng = np.random.default_rng(seed)
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(labels))
            losses = []
            for offset in range(0, len(order), batch_size):
                batch = order[offset : offset + batch_size]
                losses.append(model.step(train_x[batch], labels[batch]))
            print(
                f"rich seed={seed} epoch={epoch} loss={np.mean(losses):.6f}",
                flush=True,
            )
        prediction = model.predict(score_x)
        ensemble += user_zscore(splits.score_users, prediction) / len(seeds)
        del model, prediction
        gc.collect()
    del train_x, score_x, labels
    gc.collect()
    return ensemble


def train_rad(splits: SubmissionSplits, *, estimators: int, jobs: int) -> np.ndarray:
    add_duration_bins(splits.train, splits.score, bins=4)
    target = midrank_cdf(splits.train, ["user_id", "duration_rad_bin"], "play_time_ms")
    features = RAD_CATEGORICAL + RAD_NUMERIC
    model = lgb.LGBMRegressor(
        objective="regression_l2",
        n_estimators=estimators,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=100,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=1,
        n_jobs=jobs,
        verbosity=-1,
    )
    model.fit(
        splits.train[features],
        target,
        categorical_feature=RAD_CATEGORICAL,
        callbacks=[lgb.log_evaluation(0)],
    )
    prediction = model.predict(splits.score[features])
    del model, target
    gc.collect()
    return user_zscore(splits.score_users, prediction)


def train_ranker(splits: SubmissionSplits, *, estimators: int, jobs: int) -> np.ndarray:
    feature_splits = DatasetSplits(
        train=splits.train,
        valid=splits.score,
        valid_users=splits.score_users,
        valid_labels=np.zeros(len(splits.score), dtype=np.int8),
    )
    matrices = build_features(feature_splits)
    train_users = splits.train["user_id"].to_numpy(copy=True)
    train_order = np.argsort(train_users, kind="stable")
    model = ranker_model(seed=2026, estimators=estimators)
    model.set_params(n_jobs=jobs)
    model.fit(
        matrices.train_x.iloc[train_order],
        matrices.train_y[train_order],
        group=_group_sizes(train_users[train_order]),
        categorical_feature=matrices.categorical_features,
        callbacks=[lgb.log_evaluation(0)],
    )
    prediction = model.predict(matrices.valid_x)
    del model, matrices, train_order, train_users
    gc.collect()
    return user_zscore(splits.score_users, prediction)


def write_submission(
    path: Path,
    users: np.ndarray,
    videos: np.ndarray,
    scores: np.ndarray,
) -> None:
    if not (len(users) == len(videos) == len(scores)):
        raise RuntimeError("Submission component lengths do not match")
    if not np.isfinite(scores).all():
        raise RuntimeError("Submission contains NaN or infinite scores")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, (user, video, score) in enumerate(zip(users, videos, scores)):
            writer.writerow([row_id, int(user), int(video), f"{float(score):.9g}"])


def verify_submission(
    path: Path, users: np.ndarray, videos: np.ndarray
) -> dict[str, int | bool]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != ["row_id", "user_id", "video_id", "score"]:
            raise RuntimeError(f"Unexpected submission header: {header}")
        count = 0
        for count, record in enumerate(reader, start=1):
            row_id = count - 1
            if len(record) != 4 or int(record[0]) != row_id:
                raise RuntimeError(f"Invalid row ID or width at row {count + 1}")
            if int(record[1]) != int(users[row_id]) or int(record[2]) != int(
                videos[row_id]
            ):
                raise RuntimeError(f"ID alignment failed at row {count + 1}")
            if not np.isfinite(float(record[3])):
                raise RuntimeError(f"Non-finite score at row {count + 1}")
    if count != len(users):
        raise RuntimeError(f"Expected {len(users)} score rows, found {count}")
    return {"rows": count, "finite_scores": True, "aligned": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=STARTER / "KuaiRand-Pure" / "data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "challenge" / "runs" / "submission" / "submission.csv",
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--rich-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--ranker-estimators", type=int, default=300)
    parser.add_argument("--rad-estimators", type=int, default=119)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    started = time.perf_counter()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("At least one rich-FM seed is required")

    splits = load_submission_splits(args.data_dir)
    leaked = FORBIDDEN_SCORE_COLUMNS.intersection(splits.score.columns)
    if leaked:
        raise RuntimeError(f"Hidden outcome columns entered score frame: {leaked}")
    print(
        f"loaded final_train={len(splits.train):,} hidden_context={len(splits.score):,}",
        flush=True,
    )

    components = {
        "rich": train_rich(
            splits,
            seeds=seeds,
            epochs=args.rich_epochs,
            batch_size=args.batch_size,
        ),
        "rad": train_rad(splits, estimators=args.rad_estimators, jobs=args.jobs),
        "ranker": train_ranker(
            splits, estimators=args.ranker_estimators, jobs=args.jobs
        ),
    }
    score = sum(COMPONENT_WEIGHTS[name] * values for name, values in components.items())
    write_submission(args.output, splits.score_users, splits.score_videos, score)
    verification = verify_submission(
        args.output, splits.score_users, splits.score_videos
    )
    for name, values in components.items():
        np.save(args.output.parent / f"{name}_test.npy", values)
    np.save(args.output.parent / "champion_test.npy", score)

    result = {
        "status": "submission_ready",
        "output": str(args.output),
        "rows": len(score),
        "weights": COMPONENT_WEIGHTS,
        "rich_seeds": seeds,
        "rich_epochs": args.rich_epochs,
        "ranker_estimators": args.ranker_estimators,
        "rad_estimators": args.rad_estimators,
        "test_labels_accessed": False,
        "finite_scores": bool(np.isfinite(score).all()),
        "verification": verification,
        "wall_seconds": time.perf_counter() - started,
    }
    (args.output.parent / "submission_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("TECHJAM_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
