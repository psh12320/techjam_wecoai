"""API-free prompt-development probe for one AIDE-generated multi-exit node.

The probe re-executes the exact generated source with one diagnostic array save
inserted after inference.  It never changes the recorded campaign, and its
arrays stay outside future candidate workspaces.  Only aggregate findings may
be copied into prompt experiment memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aide.interpreter import Interpreter  # noqa: E402
from challenge.research.analyze_rerank_signals import fast_evaluate  # noqa: E402
from challenge.run_aide_research import candidate_execution_policy  # noqa: E402
from challenge.techjam_recsys.aide_portfolio import (  # noqa: E402
    validate_candidate_source,
)
from challenge.techjam_recsys.metrics import rank_normalize_within_user  # noqa: E402
from challenge.techjam_recsys.protocol import CHAMPION_VALID  # noqa: E402


INFERENCE_MARKER = """    final_logits, dcn_logits, rad_scores = predict_all(
        model, X_valid, X_valid_hist, batch_size=VALID_BATCH_SIZE
    )

    order = np.argsort(valid_users, kind=\"mergesort\")
"""

INSTRUMENTED_INFERENCE = """    final_logits, dcn_logits, rad_scores = predict_all(
        model, X_valid, X_valid_hist, batch_size=VALID_BATCH_SIZE
    )
    np.savez_compressed(
        \"./working/prompt_development_components.npz\",
        history=final_logits,
        dcn=dcn_logits,
        rad=rad_scores,
        users=valid_users,
        labels=y_valid,
    )

    order = np.argsort(valid_users, kind=\"mergesort\")
"""


def load_generated_source(journal_path: Path, node_id: str) -> str:
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    matches = [node for node in journal["nodes"] if node.get("id") == node_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one node {node_id!r}, found {len(matches)}")
    source = str(matches[0]["code"])
    if source.count(INFERENCE_MARKER) != 1:
        raise ValueError("Generated source does not have the expected inference marker")
    return source.replace(INFERENCE_MARKER, INSTRUMENTED_INFERENCE)


def beats_champion(metric: dict[str, float]) -> bool:
    return all(metric[name] > CHAMPION_VALID[name] for name in CHAMPION_VALID)


def run_seed(
    source: str,
    workspace: Path,
    seed: int,
    timeout_seconds: int,
    output_dir: Path,
) -> dict[str, np.ndarray]:
    component_path = workspace / "working" / "prompt_development_components.npz"
    component_path.unlink(missing_ok=True)
    interpreter = Interpreter(
        workspace,
        timeout=timeout_seconds,
        execution_policy=candidate_execution_policy(workspace, seed),
        max_memory_bytes=3 * 1024**3,
    )
    try:
        result = interpreter.run(source, reset_session=True)
    finally:
        interpreter.cleanup_session()
    if result.exc_type is not None:
        tail = "".join(result.term_out[-30:])[-4000:]
        raise RuntimeError(f"Seed {seed} failed: {result.exc_type}\n{tail}")
    if not component_path.exists():
        raise RuntimeError(f"Seed {seed} did not write diagnostic components")

    with np.load(component_path) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    if any(len(value) != 124_909 for value in arrays.values()):
        raise ValueError(f"Seed {seed} component length check failed")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError(f"Seed {seed} component finite check failed")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_path = output_dir / f"components_seed{seed}.npz"
    np.savez_compressed(saved_path, **arrays)
    print(
        json.dumps(
            {
                "seed": seed,
                "candidate_exec_seconds": result.exec_time,
                "component_sha256": hashlib.sha256(saved_path.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--reuse-components", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    workspace = args.workspace.resolve()
    output_dir = run_dir / "prompt_development_diagnostics"
    source = load_generated_source(run_dir / "journal.json", args.node_id)
    violations = validate_candidate_source(source)
    if violations:
        raise ValueError(f"Instrumented generated source violates policy: {violations}")
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()

    seeds: dict[int, dict[str, np.ndarray]] = {}
    for seed in (0, 1, 2):
        saved_path = output_dir / f"components_seed{seed}.npz"
        if args.reuse_components and saved_path.exists():
            with np.load(saved_path) as payload:
                seeds[seed] = {name: payload[name].copy() for name in payload.files}
        else:
            seeds[seed] = run_seed(
                source, workspace, seed, args.timeout, output_dir
            )

    ranked: dict[int, dict[str, np.ndarray]] = {}
    exits: dict[str, list[dict[str, float]]] = {
        "history": [],
        "dcn": [],
        "rad": [],
    }
    for seed, arrays in seeds.items():
        users = arrays["users"]
        labels = arrays["labels"].astype(np.int8, copy=False)
        ranked[seed] = {}
        for name in exits:
            values = rank_normalize_within_user(users, arrays[name])
            ranked[seed][name] = values
            exits[name].append(fast_evaluate(users, labels, values))

    candidates = []
    history_weights = np.arange(0.20, 0.651, 0.025)
    rad_weights = np.arange(0.0, 0.151, 0.025)
    for history_weight in history_weights:
        for rad_weight in rad_weights:
            dcn_weight = 1.0 - history_weight - rad_weight
            if dcn_weight < 0.20:
                continue
            metrics = []
            for seed, arrays in seeds.items():
                score = (
                    history_weight * ranked[seed]["history"]
                    + dcn_weight * ranked[seed]["dcn"]
                    + rad_weight * ranked[seed]["rad"]
                )
                metrics.append(
                    fast_evaluate(
                        arrays["users"], arrays["labels"].astype(np.int8), score
                    )
                )
            means = {
                name: float(np.mean([metric[name] for metric in metrics]))
                for name in CHAMPION_VALID
            }
            candidates.append(
                {
                    "weights": {
                        "history": float(history_weight),
                        "dcn": float(dcn_weight),
                        "rad": float(rad_weight),
                    },
                    "seeds": metrics,
                    "means": means,
                    "all_seeds_beat_champion": all(
                        beats_champion(metric) for metric in metrics
                    ),
                    # Historical diagnostic only. Autonomous acceptance is one
                    # deterministic seed-0 execution, never this aggregate.
                    "all_seed_diagnostic_pass": all(
                        beats_champion(metric) for metric in metrics
                    ),
                }
            )

    candidates.sort(
        key=lambda item: (
            item["all_seed_diagnostic_pass"],
            item["all_seeds_beat_champion"],
            item["means"]["primary"],
        ),
        reverse=True,
    )
    report = {
        "diagnostic_only": True,
        "generated_node_id": args.node_id,
        "instrumented_source_sha256": source_sha256,
        "champion": CHAMPION_VALID,
        "acceptance_policy": "diagnostic_only; not used by autonomous seed-0 gate",
        "exit_metrics": exits,
        "top_blends": candidates[:20],
        "all_seed_diagnostic_pass_count": sum(
            item["all_seed_diagnostic_pass"] for item in candidates
        ),
    }
    report_path = output_dir / "blend_grid.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("PROMPT_DEVELOPMENT_RESULT=" + json.dumps(report, sort_keys=True))
    print(f"Saved diagnostic report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
