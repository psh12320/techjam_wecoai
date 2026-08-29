"""Train ranking-aligned CPU models and select a validation-only hybrid.

No LLM is called by this script.  It is the deterministic model/evaluation
substrate that the AIDE research agent will modify and branch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from challenge.techjam_recsys.data import load_development_splits  # noqa: E402
from challenge.techjam_recsys.features import build_features  # noqa: E402
from challenge.techjam_recsys.metrics import (  # noqa: E402
    evaluate,
    rank_normalize_within_user,
)
from challenge.techjam_recsys.protocol import (  # noqa: E402
    ExperimentLedger,
    TrialRecord,
    select_champion,
)

REPRODUCED_BASELINE = {
    "GAUC": 0.6671333909034729,
    "nDCG@5": 0.5358057022094727,
    "primary": 0.6014695167541504,
}


def classifier_model(seed: int, estimators: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=estimators,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=200,
        max_bin=255,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=5.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def ranker_model(seed: int, estimators: int) -> lgb.LGBMRanker:
    return lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[5],
        lambdarank_truncation_level=10,
        label_gain=[0, 1],
        n_estimators=estimators,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=200,
        max_bin=255,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=5.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def _group_sizes(users: np.ndarray) -> np.ndarray:
    _, counts = np.unique(users, return_counts=True)
    return counts.astype(np.int32)


def train_classifier(features, seed: int, estimators: int) -> np.ndarray:
    model = classifier_model(seed, estimators)
    model.fit(
        features.train_x,
        features.train_y,
        categorical_feature=features.categorical_features,
        eval_set=[(features.valid_x, features.valid_y)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(40, verbose=True)],
    )
    return model.predict_proba(features.valid_x)[:, 1]


def train_ranker(features, train_users: np.ndarray, seed: int, estimators: int):
    train_order = np.argsort(train_users, kind="stable")
    valid_order = np.argsort(features.valid_users, kind="stable")
    inverse_valid = np.empty(len(valid_order), dtype=np.int64)
    inverse_valid[valid_order] = np.arange(len(valid_order))
    sorted_train_users = train_users[train_order]
    sorted_valid_users = features.valid_users[valid_order]
    model = ranker_model(seed, estimators)
    model.fit(
        features.train_x.iloc[train_order],
        features.train_y[train_order],
        group=_group_sizes(sorted_train_users),
        categorical_feature=features.categorical_features,
        eval_set=[(features.valid_x.iloc[valid_order], features.valid_y[valid_order])],
        eval_group=[_group_sizes(sorted_valid_users)],
        eval_at=[5],
        callbacks=[lgb.early_stopping(40, verbose=True)],
    )
    sorted_prediction = model.predict(features.valid_x.iloc[valid_order])
    return sorted_prediction[inverse_valid]


def search_blend(users, labels, left, right):
    left_rank = rank_normalize_within_user(users, left)
    right_rank = rank_normalize_within_user(users, right)
    candidates = []
    for alpha in np.linspace(0.0, 1.0, 21):
        scores = alpha * left_rank + (1.0 - alpha) * right_rank
        metric = evaluate(users, labels, scores)
        candidates.append((metric["primary"], float(alpha), metric, scores))
    return max(candidates, key=lambda candidate: candidate[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "kuairand-starter-kit" / "KuaiRand-Pure" / "data",
    )
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "challenge" / "runs" / "portfolio"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--classifier-estimators", type=int, default=500)
    parser.add_argument("--ranker-estimators", type=int, default=500)
    parser.add_argument(
        "--models",
        default="classifier,ranker,hybrid",
        help="Comma-separated subset of classifier,ranker,hybrid",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requested = {name.strip() for name in args.models.split(",") if name.strip()}
    args.run_dir.mkdir(parents=True, exist_ok=True)
    ledger = ExperimentLedger(args.run_dir / "iterations.jsonl")
    baseline = TrialRecord(
        iteration=0,
        hypothesis="Reproduce the immutable organizer FM baseline.",
        model_family="official_fm",
        status="success",
        config={"k": 16, "learning_rate": 0.001, "seed": 0},
        metrics=REPRODUCED_BASELINE,
        wall_seconds=84.746,
        manual_interventions=0,
    )
    if not ledger.read():
        ledger.append(baseline)

    started = time.perf_counter()
    splits = load_development_splits(args.data_dir)
    train_users = splits.train["user_id"].to_numpy(copy=True)
    features = build_features(splits)
    print(
        f"features_ready train={len(features.train_y):,} "
        f"valid={len(features.valid_y):,} columns={len(features.feature_names)} "
        f"seconds={time.perf_counter() - started:.1f}",
        flush=True,
    )

    predictions: dict[str, np.ndarray] = {}
    next_iteration = len(ledger.read())
    model_specs = [
        (
            "classifier",
            "Use ordered historical affinity features with binary logloss to improve GAUC.",
        ),
        (
            "ranker",
            "Use LambdaRank grouped by user to align training with nDCG@5.",
        ),
    ]
    for model_name, hypothesis in model_specs:
        if model_name not in requested and "hybrid" not in requested:
            continue
        trial_started = time.perf_counter()
        try:
            if model_name == "classifier":
                prediction = train_classifier(
                    features, args.seed, args.classifier_estimators
                )
            else:
                prediction = train_ranker(
                    features, train_users, args.seed, args.ranker_estimators
                )
            predictions[model_name] = prediction
            metric = evaluate(features.valid_users, features.valid_y, prediction)
            record = TrialRecord(
                iteration=next_iteration,
                hypothesis=hypothesis,
                model_family=f"lightgbm_{model_name}",
                status="success",
                config={
                    "seed": args.seed,
                    "estimators": (
                        args.classifier_estimators
                        if model_name == "classifier"
                        else args.ranker_estimators
                    ),
                    "feature_count": len(features.feature_names),
                },
                metrics={
                    key: float(metric[key]) for key in ("GAUC", "nDCG@5", "primary")
                },
                wall_seconds=time.perf_counter() - trial_started,
                manual_interventions=0,
            )
            ledger.append(record)
            np.save(args.run_dir / f"valid_{model_name}.npy", prediction)
            print(f"{model_name}=" + json.dumps(record.metrics, sort_keys=True))
        except Exception as exc:
            ledger.append(
                TrialRecord(
                    iteration=next_iteration,
                    hypothesis=hypothesis,
                    model_family=f"lightgbm_{model_name}",
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    recovery="Log the failure and continue with the surviving model branch.",
                    wall_seconds=time.perf_counter() - trial_started,
                    manual_interventions=0,
                )
            )
            raise
        finally:
            next_iteration += 1

    if "hybrid" in requested:
        if "classifier" not in predictions:
            predictions["classifier"] = np.load(args.run_dir / "valid_classifier.npy")
        if "ranker" not in predictions:
            predictions["ranker"] = np.load(args.run_dir / "valid_ranker.npy")
        _, alpha, metric, scores = search_blend(
            features.valid_users,
            features.valid_y,
            predictions["classifier"],
            predictions["ranker"],
        )
        record = TrialRecord(
            iteration=next_iteration,
            hypothesis=(
                "Rank-normalize within user and blend the GAUC-oriented classifier "
                "with the nDCG-oriented ranker using validation-only weight search."
            ),
            model_family="rank_blend",
            status="success",
            config={"classifier_weight": alpha, "grid_step": 0.05},
            metrics={key: float(metric[key]) for key in ("GAUC", "nDCG@5", "primary")},
            parent_trial_id=None,
            wall_seconds=0.0,
            manual_interventions=0,
        )
        ledger.append(record)
        np.save(args.run_dir / "valid_hybrid.npy", scores)
        print(
            "hybrid="
            + json.dumps({**record.metrics, "classifier_weight": alpha}, sort_keys=True)
        )

    champion = select_champion(ledger.read())
    summary = {
        "champion_trial_id": champion.trial_id,
        "champion_family": champion.model_family,
        "metrics": champion.metrics,
        "total_wall_seconds": time.perf_counter() - started,
        "llm_tokens": 0,
        "test_labels_accessed": False,
    }
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("TECHJAM_RESULT=" + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
