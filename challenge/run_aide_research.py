"""Budget-gated AIDE tree search with deterministic KuaiRand evaluation."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aide import backend  # noqa: E402
from aide.agent import Agent  # noqa: E402
from aide.interpreter import Interpreter  # noqa: E402
from aide.journal import Journal, Node  # noqa: E402
from aide.utils.config import (  # noqa: E402
    _load_cfg,
    load_task_desc,
    prep_agent_workspace,
    prep_cfg,
    save_run,
)
from challenge.techjam_recsys.aide_reviewer import (  # noqa: E402
    KuaiRandPredictionReviewer,
)
from challenge.techjam_recsys.protocol import (  # noqa: E402
    BASELINE_VALID,
    ConvergenceTracker,
    ExperimentLedger,
    TrialRecord,
)


class KuaiRandAgent(Agent):
    @property
    def _prompt_impl_guideline(self):
        return {
            "Implementation guideline": [
                "Implement the complete proposed solution as one self-contained Python program.",
                "Read only files under ./input and write temporary/output files under ./working.",
                "Use long_view only as a training or validation label, never as a model input.",
                "Never use current-row engagement outcomes as validation features.",
                "Derive model features from the train/validation column intersection; training-only outcome columns must never enter X.",
                "Write ./working/validation_predictions.csv with exactly the header row_id,score and 124909 aligned rows.",
                "Use a fixed random seed and finish within the execution timeout.",
                "Print useful training progress, but the external deterministic evaluator is authoritative.",
            ]
        }


def code_diff(node: Node) -> str:
    before = node.parent.code.splitlines() if node.parent is not None else []
    after = node.code.splitlines()
    return "\n".join(
        difflib.unified_diff(before, after, fromfile="parent.py", tofile="candidate.py")
    )


def parse_metrics(analysis: str):
    try:
        value = json.loads(analysis)
        return value.get("metrics"), None
    except Exception as exc:
        return None, f"Could not parse deterministic review: {exc}"


def estimate_uncached_cost(
    input_tokens: int,
    output_tokens: int,
    input_usd_per_million: float,
    output_usd_per_million: float,
) -> float:
    """Conservative cost estimate using uncached published token rates."""

    return (
        input_tokens * input_usd_per_million + output_tokens * output_usd_per_million
    ) / 1_000_000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--max-output-tokens-per-call", type=int, default=6000)
    parser.add_argument("--per-trial-timeout", type=int, default=900)
    parser.add_argument("--approval-id")
    parser.add_argument("--approved-max-usd", type=float)
    parser.add_argument("--approved-max-input-tokens", type=int)
    parser.add_argument("--approved-max-output-tokens", type=int)
    parser.add_argument("--input-usd-per-million", type=float)
    parser.add_argument("--output-usd-per-million", type=float)
    parser.add_argument(
        "--agent-data", type=Path, default=ROOT / "challenge" / "agent_data"
    )
    parser.add_argument(
        "--run-root", type=Path, default=ROOT / "challenge" / "runs" / "aide"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.steps <= 49:
        raise ValueError("steps must be between 1 and 49 (the seed is iteration 0)")
    dry_run = {
        "paid_execution": bool(args.execute),
        "model": args.model,
        "paid_iterations": args.steps,
        "api_calls_per_iteration": "1 normally, up to 3 if code extraction retries",
        "deterministic_review_api_calls": 0,
        "max_output_tokens_per_call": args.max_output_tokens_per_call,
        "worst_case_output_tokens": (args.steps * 3 * args.max_output_tokens_per_call),
        "approval_required_every_run": True,
    }
    if not args.execute:
        print("DRY_RUN=" + json.dumps(dry_run, sort_keys=True))
        return 0
    required = {
        "approval_id": args.approval_id,
        "approved_max_usd": args.approved_max_usd,
        "approved_max_input_tokens": args.approved_max_input_tokens,
        "approved_max_output_tokens": args.approved_max_output_tokens,
        "input_usd_per_million": args.input_usd_per_million,
        "output_usd_per_million": args.output_usd_per_million,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise RuntimeError(
            "Paid run blocked; missing one-run approval fields: " + ", ".join(missing)
        )
    estimated_ceiling = estimate_uncached_cost(
        args.approved_max_input_tokens,
        args.approved_max_output_tokens,
        args.input_usd_per_million,
        args.output_usd_per_million,
    )
    if estimated_ceiling > args.approved_max_usd:
        raise RuntimeError(
            f"Paid run blocked: token envelope estimates ${estimated_ceiling:.4f}, "
            f"above the approved ${args.approved_max_usd:.4f} ceiling"
        )
    if not args.agent_data.exists():
        raise RuntimeError("Run challenge/prepare_agent_data.py first")
    load_dotenv(ROOT / ".env.local")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not available")

    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(args.agent_data)
    cfg.desc_file = str(ROOT / "challenge" / "task.md")
    cfg.goal = None
    cfg.eval = None
    cfg.log_dir = str(args.run_root / "aide_logs")
    cfg.workspace_dir = str(args.run_root / "workspaces")
    cfg.preprocess_data = False
    # Windows developer sessions commonly lack symlink privilege. The prepared
    # agent bundle is development-only and small enough to copy safely.
    cfg.copy_data = True
    cfg.generate_report = False
    cfg.exec.timeout = args.per_trial_timeout
    cfg.agent.steps = args.steps
    cfg.agent.k_fold_validation = 1
    cfg.agent.metric_maximize = True
    # The immutable FM seed is the single draft. Every paid iteration must
    # improve or debug an evaluated node rather than spend money restarting.
    cfg.agent.search.num_drafts = 1
    cfg.agent.code.model = args.model
    # Current GPT-5 generation models do not need AIDE's legacy sampling value.
    # Keeping this unset also avoids spending an approved call on a rejected
    # request when a model does not support a custom temperature.
    cfg.agent.code.temp = None
    cfg.agent.code.max_tokens = args.max_output_tokens_per_call
    cfg = prep_cfg(cfg)
    task_desc = load_task_desc(cfg)
    prep_agent_workspace(cfg)

    journal = Journal(metric_maximize=True)
    reviewer = KuaiRandPredictionReviewer(
        cfg.workspace_dir,
        Path(args.agent_data) / "validation_index.npz",
        cfg.log_dir / "predictions",
    )
    agent = KuaiRandAgent(
        task_desc=task_desc,
        cfg=cfg,
        journal=journal,
        result_reviewer=reviewer,
    )
    interpreter = Interpreter(
        cfg.workspace_dir,
        **OmegaConf.to_container(cfg.exec),
    )
    ledger = ExperimentLedger(cfg.log_dir / "iterations.jsonl")
    tracker = ConvergenceTracker(max_iterations=min(args.steps + 1, 50))
    backend.reset_usage_totals()
    run_started = time.perf_counter()

    def execute(code: str, reset_session: bool = True):
        reviewer.clear_candidate_output()
        return interpreter.run(code, reset_session)

    seed = Node(
        plan="Reproduce the immutable organizer FM baseline before research iterations.",
        code=(ROOT / "challenge" / "agent_seed.py").read_text(encoding="utf-8"),
    )
    seed_started = time.perf_counter()
    agent.parse_exec_result(seed, execute(seed.code, True))
    journal.append(seed)
    seed_metrics, seed_error = parse_metrics(seed.analysis)
    ledger.append(
        TrialRecord(
            iteration=0,
            hypothesis=seed.plan,
            model_family="official_fm_seed",
            status="success" if seed_metrics else "failed",
            metrics=seed_metrics,
            error=seed_error,
            code_diff=code_diff(seed),
            wall_seconds=time.perf_counter() - seed_started,
            manual_interventions=0,
        )
    )
    save_run(cfg, journal)
    if not seed_metrics:
        raise RuntimeError(f"Seed baseline failed: {seed.analysis}")
    tracker.observe(float(seed_metrics["primary"]))

    for iteration in range(1, args.steps + 1):
        usage_before = backend.get_usage_totals()
        remaining_output = (
            args.approved_max_output_tokens - usage_before["output_tokens"]
        )
        retry_reserve = 3 * args.max_output_tokens_per_call
        if remaining_output < retry_reserve:
            print(
                "Stopping before the next API call: the remaining approved "
                "output-token budget does not cover the three-retry reserve."
            )
            break
        trial_started = time.perf_counter()
        agent.step(exec_callback=execute)
        node = journal.nodes[-1]
        usage_after = backend.get_usage_totals()
        metrics, parse_error = parse_metrics(node.analysis)
        ledger.append(
            TrialRecord(
                iteration=iteration,
                hypothesis=node.plan or "",
                model_family=node.stage_name,
                status="failed" if node.is_buggy else "success",
                metrics=metrics,
                parent_trial_id=node.parent.id if node.parent else None,
                code_diff=code_diff(node),
                error=(
                    parse_error
                    if parse_error
                    else (node.analysis if node.is_buggy else None)
                ),
                recovery=(
                    "AIDE will debug this leaf within max_debug_depth=3."
                    if node.is_buggy
                    else None
                ),
                wall_seconds=time.perf_counter() - trial_started,
                llm_input_tokens=(
                    usage_after["input_tokens"] - usage_before["input_tokens"]
                ),
                llm_output_tokens=(
                    usage_after["output_tokens"] - usage_before["output_tokens"]
                ),
                manual_interventions=0,
            )
        )
        save_run(cfg, journal)
        observed = float(metrics["primary"]) if metrics else 0.0
        if tracker.observe(observed):
            break
        if usage_after["input_tokens"] >= args.approved_max_input_tokens:
            print("Stopping: approved input-token budget reached.")
            break

    interpreter.cleanup_session()
    usage = backend.get_usage_totals()
    best = journal.get_best_node()
    result = {
        "approval_id": args.approval_id,
        "approved_max_usd": args.approved_max_usd,
        "estimated_uncached_cost_ceiling": estimated_ceiling,
        "pricing_usd_per_million": {
            "input": args.input_usd_per_million,
            "output": args.output_usd_per_million,
        },
        "best_primary": float(best.metric.value) if best else None,
        "published_baseline": BASELINE_VALID,
        "iterations": len(journal.nodes),
        "stop_reason": tracker.stop_reason,
        "wall_seconds": time.perf_counter() - run_started,
        "usage": usage,
        "manual_interventions": 0,
    }
    (cfg.log_dir / "resource_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("TECHJAM_RESULT=" + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
