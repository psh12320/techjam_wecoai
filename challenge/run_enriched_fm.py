"""Train leakage-safe enriched factorization machines on the official split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "kuairand-starter-kit"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STARTER))

import baseline as official  # noqa: E402

from challenge.techjam_recsys.data import (  # noqa: E402
    TRAIN_END,
    VALID_END,
    VALID_START,
    load_temporal_splits,
)
from challenge.techjam_recsys.metrics import evaluate  # noqa: E402

VARIANTS = {
    "core": ["user_id", "video_id", "author_id", "tab", "duration_bucket"],
    "temporal": [
        "user_id",
        "video_id",
        "author_id",
        "tab",
        "duration_bucket",
        "hour",
        "weekday",
    ],
    "content": [
        "user_id",
        "video_id",
        "author_id",
        "tab",
        "duration_bucket",
        "music_id",
        "music_type",
        "video_type",
        "upload_type",
        "primary_tag",
        "upload_age_bucket",
        "aspect_bucket",
    ],
    "all": [
        "user_id",
        "video_id",
        "author_id",
        "tab",
        "duration_bucket",
        "hour",
        "weekday",
        "music_id",
        "music_type",
        "video_type",
        "upload_type",
        "primary_tag",
        "upload_age_bucket",
        "aspect_bucket",
    ],
    "rich": [
        "user_id",
        "video_id",
        "author_id",
        "tab",
        "duration_bucket",
        "duration_rule_band",
        "tab_duration_cross",
        "hour",
        "weekday",
        "primary_tag",
        "tag_2",
        "tag_3",
        "user_active_degree",
        "is_lowactive_period",
        "is_live_streamer",
        "is_video_author",
        "follow_user_num_range",
        "fans_user_num_range",
        "friend_user_num_range",
        "register_days_range",
        "onehot_feat0",
        "onehot_feat1",
        "onehot_feat2",
        "onehot_feat3",
        "onehot_feat4",
        "onehot_feat5",
        "onehot_feat6",
        "onehot_feat7",
        "onehot_feat8",
        "onehot_feat9",
        "onehot_feat10",
        "onehot_feat11",
    ],
    "rich_lite": [
        "user_id",
        "video_id",
        "author_id",
        "tab",
        "duration_bucket",
        "duration_rule_band",
        "tab_duration_cross",
        "hour",
        "primary_tag",
        "tag_2",
        "user_active_degree",
        "follow_user_num_range",
        "register_days_range",
    ],
}


class FieldGatedFM(official.FM):
    """FM with learned global field gates initialized around the core model."""

    def __init__(
        self,
        dim: int,
        fields: int,
        core_fields: int,
        k: int,
        lr: float,
        l2: float,
        seed: int,
        extra_gate_init: float,
        gate_l2: float,
    ):
        super().__init__(dim, k=k, lr=lr, l2=l2, seed=seed)
        self.gates = np.full(fields, extra_gate_init, dtype=np.float32)
        self.gates[:core_fields] = 1.0
        self.gate_prior = self.gates.copy()
        self.gate_l2 = gate_l2
        self.mG = np.zeros_like(self.gates)
        self.vG = np.zeros_like(self.gates)

    def logits(self, X):
        raw = self.V[X]
        embedded = raw * self.gates[None, :, None]
        summed = embedded.sum(1)
        interaction = 0.5 * ((summed**2).sum(1) - (embedded**2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + interaction, raw, embedded, summed

    def step(self, X, y):
        batch_size = len(y)
        logits, raw, embedded, summed = self.logits(X)
        probability = official.sigmoid(logits)
        gradient = ((probability - y) / batch_size).astype(np.float32)
        grad_v = np.zeros_like(self.V)
        grad_w = np.zeros_like(self.W)
        np.add.at(grad_w, X, gradient[:, None])
        np.add.at(
            grad_v,
            X,
            gradient[:, None, None]
            * (summed[:, None, :] - embedded)
            * self.gates[None, :, None],
        )
        grad_gate = np.sum(
            gradient[:, None] * np.sum(raw * (summed[:, None, :] - embedded), axis=2),
            axis=0,
        )
        grad_v += self.l2 * self.V
        grad_w += self.l2 * self.W
        grad_gate += self.gate_l2 * (self.gates - self.gate_prior)
        self.t += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for parameter, grad, mean, variance in (
            (self.V, grad_v, self.mV, self.vV),
            (self.W, grad_w, self.mW, self.vW),
            (self.gates, grad_gate, self.mG, self.vG),
        ):
            mean *= beta1
            mean += (1 - beta1) * grad
            variance *= beta2
            variance += (1 - beta2) * (grad * grad)
            parameter -= (
                self.lr
                * (mean / (1 - beta1**self.t))
                / (np.sqrt(variance / (1 - beta2**self.t)) + epsilon)
            )
        self.gates[:] = np.clip(self.gates, -2.0, 2.0)
        self.b -= self.lr * gradient.sum()
        return float(
            -np.mean(
                y * np.log(probability + 1e-9)
                + (1 - y) * np.log(1 - probability + 1e-9)
            )
        )

    def predict(self, X, bs=200_000):
        return np.concatenate(
            [self.logits(X[index : index + bs])[0] for index in range(0, len(X), bs)]
        )


def add_train_derived_buckets(train, valid) -> None:
    for source, target, quantiles in (
        ("upload_age_days", "upload_age_bucket", 20),
        ("aspect_ratio", "aspect_bucket", 10),
    ):
        values = train[source].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        edges = np.unique(
            np.quantile(finite, np.linspace(0.0, 1.0, quantiles + 1)[1:-1])
        )
        for frame in (train, valid):
            raw = frame[source].to_numpy(dtype=np.float64)
            raw = np.nan_to_num(raw, nan=-1.0, posinf=1e9, neginf=-1.0)
            frame[target] = np.searchsorted(edges, raw, side="right").astype(np.int16)


def encode_fields(train, valid, fields: list[str]):
    train_x = np.empty((len(train), len(fields)), dtype=np.int32)
    valid_x = np.empty((len(valid), len(fields)), dtype=np.int32)
    offset = 0
    cardinalities = {}
    for index, field in enumerate(fields):
        train_values = train[field].fillna("__MISSING__").astype(str)
        valid_values = valid[field].fillna("__MISSING__").astype(str)
        vocabulary = np.unique(train_values.to_numpy())
        mapping = {value: code for code, value in enumerate(vocabulary)}
        unknown = len(vocabulary)
        train_codes = train_values.map(mapping).to_numpy(dtype=np.int32)
        valid_codes = valid_values.map(mapping).fillna(unknown).to_numpy(dtype=np.int32)
        cardinality = unknown + 1
        train_x[:, index] = train_codes + offset
        valid_x[:, index] = valid_codes + offset
        cardinalities[field] = cardinality
        offset += cardinality
    return train_x, valid_x, offset, cardinalities


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="all")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--l2", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--gated", action="store_true")
    parser.add_argument("--extra-gate-init", type=float, default=0.1)
    parser.add_argument("--gate-l2", type=float, default=1e-3)
    parser.add_argument("--train-end", type=int, default=TRAIN_END)
    parser.add_argument("--valid-start", type=int, default=VALID_START)
    parser.add_argument("--valid-end", type=int, default=VALID_END)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=STARTER / "KuaiRand-Pure" / "data",
    )
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "challenge" / "runs" / "enriched_fm"
    )
    args = parser.parse_args()

    splits = load_temporal_splits(
        args.data_dir,
        train_end=args.train_end,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
    )
    add_train_derived_buckets(splits.train, splits.valid)
    fields = VARIANTS[args.variant]
    train_x, valid_x, dimension, cardinalities = encode_fields(
        splits.train, splits.valid, fields
    )
    train_y = splits.train["long_view"].to_numpy(dtype=np.float32)
    if args.gated:
        model = FieldGatedFM(
            dimension,
            fields=len(fields),
            core_fields=5,
            k=args.k,
            lr=args.lr,
            l2=args.l2,
            seed=args.seed,
            extra_gate_init=args.extra_gate_init,
            gate_l2=args.gate_l2,
        )
    else:
        model = official.FM(dimension, k=args.k, lr=args.lr, l2=args.l2, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    best_primary = -np.inf
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(train_y))
        losses = []
        for offset in range(0, len(order), args.batch_size):
            batch = order[offset : offset + args.batch_size]
            losses.append(model.step(train_x[batch], train_y[batch]))
        prediction = model.predict(valid_x)
        metric = evaluate(splits.valid_users, splits.valid_labels, prediction)
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            **metric,
        }
        history.append(record)
        print("epoch=" + json.dumps(record, sort_keys=True), flush=True)
        if float(metric["primary"]) > best_primary + 1e-5:
            best_primary = float(metric["primary"])
            best_epoch = epoch
            best_state = (
                model.V.copy(),
                model.W.copy(),
                np.float32(model.b),
                model.gates.copy() if args.gated else None,
            )
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    model.V, model.W, model.b, best_gates = best_state
    if args.gated:
        model.gates = best_gates
    prediction = model.predict(valid_x)
    metric = evaluate(splits.valid_users, splits.valid_labels, prediction)
    split_suffix = (
        ""
        if (args.train_end, args.valid_start, args.valid_end)
        == (TRAIN_END, VALID_START, VALID_END)
        else f"_tr{args.train_end}_va{args.valid_start}-{args.valid_end}"
    )
    run_name = (
        f"{args.variant}_k{args.k}_lr{args.lr:g}_l2{args.l2:g}_seed{args.seed}"
        f"{split_suffix}"
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.run_dir / f"{run_name}_valid.npy", prediction)
    result = {
        "variant": args.variant,
        "fields": fields,
        "cardinalities": cardinalities,
        "dimension": dimension,
        "k": args.k,
        "lr": args.lr,
        "l2": args.l2,
        "seed": args.seed,
        "split": {
            "train_end": args.train_end,
            "valid_start": args.valid_start,
            "valid_end": args.valid_end,
        },
        "gated": args.gated,
        "gates": (
            {field: float(gate) for field, gate in zip(fields, model.gates)}
            if args.gated
            else None
        ),
        "best_epoch": best_epoch,
        "wall_seconds": time.perf_counter() - started,
        "metrics": metric,
        "history": history,
    }
    (args.run_dir / f"{run_name}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("TECHJAM_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
