"""Fine-tune the official FM with within-user hard-negative BPR loss."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "kuairand-starter-kit"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STARTER))

import baseline as organizer_baseline  # noqa: E402
from data import encode  # noqa: E402

from challenge.reproduce_baseline import load_train_valid_only  # noqa: E402
from challenge.techjam_recsys.metrics import (  # noqa: E402
    evaluate,
    rank_normalize_within_user,
)
from challenge.techjam_recsys.protocol import (  # noqa: E402
    ExperimentLedger,
    TrialRecord,
    select_champion,
)


class PairwiseFM(organizer_baseline.FM):
    """Organizer FM parameters and Adam optimizer with a BPR update."""

    def step_pairwise(self, positive_x: np.ndarray, negative_x: np.ndarray) -> float:
        batch_size = len(positive_x)
        combined = np.concatenate([positive_x, negative_x], axis=0)
        logits, embeddings, summed = self.logits(combined)
        difference = logits[:batch_size] - logits[batch_size:]
        magnitude = organizer_baseline.sigmoid(-difference).astype(np.float32)
        magnitude /= batch_size
        logit_gradient = np.concatenate([-magnitude, magnitude])

        gradient_v = np.zeros_like(self.V)
        gradient_w = np.zeros_like(self.W)
        np.add.at(gradient_w, combined, logit_gradient[:, None])
        np.add.at(
            gradient_v,
            combined,
            logit_gradient[:, None, None] * (summed[:, None, :] - embeddings),
        )
        gradient_v += self.l2 * self.V
        gradient_w += self.l2 * self.W
        self.t += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for parameter, gradient, moment, variance in (
            (self.V, gradient_v, self.mV, self.vV),
            (self.W, gradient_w, self.mW, self.vW),
        ):
            moment *= beta1
            moment += (1.0 - beta1) * gradient
            variance *= beta2
            variance += (1.0 - beta2) * gradient * gradient
            corrected_moment = moment / (1.0 - beta1**self.t)
            corrected_variance = variance / (1.0 - beta2**self.t)
            parameter -= (
                self.lr * corrected_moment / (np.sqrt(corrected_variance) + epsilon)
            )
        # The global bias cancels from every pair and is intentionally unchanged.
        return float(np.mean(np.logaddexp(0.0, -difference)))


def train_pointwise(
    model: PairwiseFM,
    train_x,
    train_y,
    valid_x,
    valid_y,
    valid_users,
    *,
    seed: int,
    max_epochs: int,
    batch_size: int,
    patience: int,
):
    rng = np.random.default_rng(seed)
    best_primary = -1.0
    state = None
    bad_epochs = 0
    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(len(train_y))
        losses = []
        for offset in range(0, len(order), batch_size):
            batch = order[offset : offset + batch_size]
            losses.append(model.step(train_x[batch], train_y[batch]))
        prediction = model.predict(valid_x)
        metric = evaluate(valid_users, valid_y, prediction)
        print(
            f"pointwise epoch={epoch:02d} loss={np.mean(losses):.5f} "
            f"GAUC={metric['GAUC']:.6f} nDCG@5={metric['nDCG@5']:.6f} "
            f"primary={metric['primary']:.6f}",
            flush=True,
        )
        if metric["primary"] > best_primary + 1e-5:
            best_primary = float(metric["primary"])
            bad_epochs = 0
            state = (
                model.V.copy(),
                model.W.copy(),
                np.float32(model.b),
                model.mV.copy(),
                model.vV.copy(),
                model.mW.copy(),
                model.vW.copy(),
                model.t,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if state is None:
        raise RuntimeError("Pointwise pretraining produced no checkpoint")
    (
        model.V,
        model.W,
        model.b,
        model.mV,
        model.vV,
        model.mW,
        model.vW,
        model.t,
    ) = state
    prediction = model.predict(valid_x)
    return prediction, evaluate(valid_users, valid_y, prediction)


def build_pair_index(train_rows):
    negative_by_user: dict[str, list[int]] = defaultdict(list)
    positive_by_user: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(train_rows):
        if row[6] == 1:
            positive_by_user[row[1]].append(index)
        else:
            negative_by_user[row[1]].append(index)

    pools: list[np.ndarray] = []
    positive_indices: list[int] = []
    positive_positions_by_pool: list[np.ndarray] = []
    for user in sorted(positive_by_user):
        negatives = negative_by_user.get(user)
        if not negatives:
            continue
        pool_id = len(pools)
        pools.append(np.asarray(negatives, dtype=np.int32))
        start = len(positive_indices)
        positive_indices.extend(positive_by_user[user])
        positive_positions_by_pool.append(
            np.arange(start, len(positive_indices), dtype=np.int32)
        )
        if pool_id != len(positive_positions_by_pool) - 1:
            raise AssertionError("Pair-pool indexing drifted")
    return (
        np.asarray(positive_indices, dtype=np.int32),
        pools,
        positive_positions_by_pool,
    )


def sample_negative_matrix(
    rng: np.random.Generator,
    pools: list[np.ndarray],
    positions_by_pool: list[np.ndarray],
    positive_count: int,
    candidates: int,
) -> np.ndarray:
    sampled = np.empty((candidates, positive_count), dtype=np.int32)
    for pool, positions in zip(pools, positions_by_pool):
        for candidate in range(candidates):
            sampled[candidate, positions] = rng.choice(
                pool, size=len(positions), replace=True
            )
    return sampled


def choose_hard_negatives(
    model: PairwiseFM,
    train_x: np.ndarray,
    negative_candidates: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    local = negative_candidates[:, positions]
    flat = local.reshape(-1)
    scores = model.predict(train_x[flat]).reshape(local.shape)
    hardest = np.argmax(scores, axis=0)
    return local[hardest, np.arange(local.shape[1])]


def fine_tune_pairwise(
    model: PairwiseFM,
    train_x,
    positive_indices,
    pools,
    positions_by_pool,
    valid_x,
    valid_y,
    valid_users,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    hard_negatives: int,
    learning_rate: float,
    reset_optimizer: bool,
):
    rng = np.random.default_rng(seed + 10_000)
    model.lr = learning_rate
    if reset_optimizer:
        model.mV.fill(0)
        model.vV.fill(0)
        model.mW.fill(0)
        model.vW.fill(0)
        model.t = 0
    initial_prediction = model.predict(valid_x)
    best_metric = evaluate(valid_users, valid_y, initial_prediction)
    best_prediction = initial_prediction
    best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        candidates = sample_negative_matrix(
            rng,
            pools,
            positions_by_pool,
            len(positive_indices),
            hard_negatives,
        )
        order = rng.permutation(len(positive_indices))
        losses = []
        for offset in range(0, len(order), batch_size):
            positions = order[offset : offset + batch_size]
            negative_indices = choose_hard_negatives(
                model, train_x, candidates, positions
            )
            losses.append(
                model.step_pairwise(
                    train_x[positive_indices[positions]],
                    train_x[negative_indices],
                )
            )
        prediction = model.predict(valid_x)
        metric = evaluate(valid_users, valid_y, prediction)
        print(
            f"pairwise epoch={epoch:02d} loss={np.mean(losses):.5f} "
            f"GAUC={metric['GAUC']:.6f} nDCG@5={metric['nDCG@5']:.6f} "
            f"primary={metric['primary']:.6f} "
            f"seconds={time.perf_counter() - started:.1f}",
            flush=True,
        )
        if metric["primary"] > best_metric["primary"]:
            best_metric = metric
            best_prediction = prediction
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
    model.V, model.W, model.b = best_state
    return best_prediction, best_metric


def search_blend(users, labels, pointwise, pairwise):
    left = rank_normalize_within_user(users, pointwise)
    right = rank_normalize_within_user(users, pairwise)
    candidates = []
    for pairwise_weight in np.linspace(0.0, 1.0, 21):
        scores = (1.0 - pairwise_weight) * left + pairwise_weight * right
        metric = evaluate(users, labels, scores)
        candidates.append((metric["primary"], float(pairwise_weight), metric, scores))
    return max(candidates, key=lambda value: value[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=STARTER / "KuaiRand-Pure" / "data",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "challenge" / "runs" / "pairwise_fm",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--pointwise-learning-rate", type=float, default=0.001)
    parser.add_argument("--pairwise-learning-rate", type=float, default=0.00005)
    parser.add_argument("--pointwise-epochs", type=int, default=40)
    parser.add_argument("--pairwise-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hard-negatives", type=int, default=1)
    parser.add_argument(
        "--keep-optimizer-state",
        action="store_true",
        help="Continue pointwise Adam moments instead of resetting for BPR",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    ledger = ExperimentLedger(args.run_dir / "iterations.jsonl")
    started = time.perf_counter()
    splits = load_train_valid_only(args.data_dir)
    encoded, dimension = encode(splits)
    train_x, train_y, _ = encoded["train"]
    valid_x, valid_y, valid_users = encoded["valid"]
    model = PairwiseFM(
        dimension,
        k=args.k,
        lr=args.pointwise_learning_rate,
        seed=args.seed,
    )

    point_started = time.perf_counter()
    point_prediction, point_metric = train_pointwise(
        model,
        train_x,
        train_y,
        valid_x,
        valid_y,
        valid_users,
        seed=args.seed,
        max_epochs=args.pointwise_epochs,
        batch_size=args.batch_size,
        patience=4,
    )
    point_record = TrialRecord(
        iteration=0,
        hypothesis="Reproduce the official pointwise FM checkpoint.",
        model_family="official_fm",
        status="success",
        config={"k": args.k, "learning_rate": args.pointwise_learning_rate},
        metrics={
            key: float(point_metric[key]) for key in ("GAUC", "nDCG@5", "primary")
        },
        wall_seconds=time.perf_counter() - point_started,
        manual_interventions=0,
    )
    ledger.append(point_record)
    np.save(args.run_dir / "valid_pointwise.npy", point_prediction)

    positive_indices, pools, positions_by_pool = build_pair_index(splits["train"])
    print(
        f"pair_index positives={len(positive_indices):,} users={len(pools):,} "
        f"hard_negatives={args.hard_negatives}",
        flush=True,
    )
    pair_started = time.perf_counter()
    pair_prediction, pair_metric = fine_tune_pairwise(
        model,
        train_x,
        positive_indices,
        pools,
        positions_by_pool,
        valid_x,
        valid_y,
        valid_users,
        seed=args.seed,
        epochs=args.pairwise_epochs,
        batch_size=args.batch_size,
        hard_negatives=args.hard_negatives,
        learning_rate=args.pairwise_learning_rate,
        reset_optimizer=not args.keep_optimizer_state,
    )
    pair_record = TrialRecord(
        iteration=1,
        hypothesis=(
            "Fine-tune the official FM using hard-negative BPR pairs sampled "
            "within each user to align optimization with GAUC and ranking."
        ),
        model_family="fm_bpr_hard_negative",
        status="success",
        config={
            "k": args.k,
            "learning_rate": args.pairwise_learning_rate,
            "epochs": args.pairwise_epochs,
            "hard_negatives": args.hard_negatives,
            "reset_optimizer": not args.keep_optimizer_state,
        },
        metrics={key: float(pair_metric[key]) for key in ("GAUC", "nDCG@5", "primary")},
        parent_trial_id=point_record.trial_id,
        wall_seconds=time.perf_counter() - pair_started,
        manual_interventions=0,
    )
    ledger.append(pair_record)
    np.save(args.run_dir / "valid_pairwise.npy", pair_prediction)

    _, pairwise_weight, blend_metric, blend_prediction = search_blend(
        valid_users, valid_y, point_prediction, pair_prediction
    )
    blend_record = TrialRecord(
        iteration=2,
        hypothesis=(
            "Blend pointwise calibration and pairwise ordering after within-user "
            "rank normalization to improve both official component metrics."
        ),
        model_family="fm_pointwise_bpr_blend",
        status="success",
        config={"pairwise_weight": pairwise_weight, "grid_step": 0.05},
        metrics={
            key: float(blend_metric[key]) for key in ("GAUC", "nDCG@5", "primary")
        },
        parent_trial_id=pair_record.trial_id,
        wall_seconds=0.0,
        manual_interventions=0,
    )
    ledger.append(blend_record)
    np.save(args.run_dir / "valid_blend.npy", blend_prediction)

    champion = select_champion(ledger.read())
    result = {
        "champion_family": champion.model_family,
        "metrics": champion.metrics,
        "pairwise_weight": pairwise_weight,
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
