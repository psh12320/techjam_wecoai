"""Budget-gated AIDE tree search with deterministic KuaiRand evaluation."""

from __future__ import annotations

import argparse
import atexit
import difflib
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import tempfile
import time
import uuid
from dataclasses import replace
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "techjam-wecoai-matplotlib")
)

from aide import backend  # noqa: E402
from aide.agent import Agent  # noqa: E402
from aide.interpreter import (  # noqa: E402
    ExecutionPolicy,
    ExecutionResult,
    Interpreter,
)
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
from challenge.techjam_recsys.campaign_safety import (  # noqa: E402
    CANDIDATE_INPUT_ALLOWLIST,
    CampaignEventLedger,
    CampaignManifest,
    fingerprint_sources,
    fingerprint_tree,
    sha256_file,
    validate_candidate_input,
)
from challenge.techjam_recsys.aide_portfolio import (  # noqa: E402
    PROMPT_VERSION,
    PortfolioAssignment,
    PortfolioScheduler,
    candidate_code_sha256,
    clean_candidate_plan,
    pareto_frontier,
    parse_candidate_spec,
    validate_candidate_spec,
    validate_candidate_source,
)
from challenge.techjam_recsys.literature import load_manifest  # noqa: E402
from challenge.techjam_recsys.prompt_context import (  # noqa: E402
    PromptContext,
    canonical_sha256,
    load_prompt_context,
)
from challenge.techjam_recsys.protocol import (  # noqa: E402
    BASELINE_VALID,
    CHAMPION_VALID,
    MAX_ITERATIONS,
    ChallengeMetric,
    ConvergenceTracker,
    ExperimentLedger,
    TrialRecord,
    count_manual_interventions,
)
from challenge.techjam_recsys.validation import last_days_holdout  # noqa: E402


class KuaiRandAgent(Agent):
    def __init__(
        self,
        *args,
        scheduler: PortfolioScheduler | None = None,
        prompt_context: PromptContext | None = None,
        allowed_eda_observation_ids: set[str] | None = None,
        allowed_literature_citation_ids: set[str] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scheduler = scheduler or PortfolioScheduler(
            max_debug_depth=self.acfg.search.max_debug_depth
        )
        self.current_assignment = PortfolioAssignment(
            family="rich_fm",
            action="improve",
            reason="Establish the required rich-FM milestone.",
        )
        self.prompt_context = prompt_context
        self.allowed_eda_observation_ids = allowed_eda_observation_ids or set()
        self.allowed_literature_citation_ids = allowed_literature_citation_ids or set()

    @property
    def _prompt_environment(self):
        distributions = {
            "numpy": "numpy",
            "pandas": "pandas",
            "scikit-learn": "sklearn",
            "scipy": "scipy",
            "lightgbm": "lightgbm",
            "torch": "torch",
            "catboost": "catboost",
        }
        installed = []
        missing = []
        for distribution, module in distributions.items():
            if importlib.util.find_spec(module) is None:
                missing.append(distribution)
                continue
            try:
                version = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                version = "installed"
            installed.append(f"{distribution}=={version}")
        return {
            "Verified runtime": {
                "Installed packages": installed,
                "Unavailable packages": missing,
                "CPU and memory": "Use exactly 1-4 CPU threads and less than 3 GB RAM.",
                "Filesystem": "Enforced: read public files under ./input only; write under ./working only. Parent traversal is blocked.",
                "Network and processes": "Enforced: network access and child-process creation are disabled, and credentials are removed from the candidate environment.",
            }
        }

    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "Start with exactly one <candidate_spec> tag containing valid JSON "
                "with keys parent_node_id, parent_code_sha256, model_family, "
                "eda_observation_ids, literature_citation_ids, scientific_change, "
                "change_scope, preserved_parent_components, hypothesis, features, "
                "losses, hyperparameters, target_metric, "
                "expected_metric_effects (with GAUC, nDCG@5, and primary), "
                "estimated_runtime_seconds, estimated_memory_mb, risks, abort_criteria, "
                "falsification_condition, fidelity, and internal_validation. Use fidelity "
                "'full' and internal_validation split 'last_3_train_days_strict_time', "
                "boundary_time_ms 1650295266482, train_rows 1079102, and holdout_rows 62010. "
                "The parent fields "
                "must exactly match the selected portfolio parent. model_family must equal "
                "the assigned scientific-change family, not merely the parent's base family. "
                "change_scope must be exactly one of features, architecture, loss, training, "
                "or reranking; list the parent mechanisms retained verbatim in "
                "preserved_parent_components. "
                "After </candidate_spec>, write a 3-5 sentence hypothesis, then exactly one complete Python code block. Do not add headings or "
                "text after the code block."
            )
        }

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
                "Use the fixed campaign seed 0 and finish within the execution timeout.",
                "Read the base seed from int(os.environ.get('AIDE_SEED', '0')) and use it for every random generator; the runner always supplies 0.",
                "Train and evaluate exactly once with AIDE_SEED=0; do not train, rerun, or ensemble multiple seeds inside a candidate.",
                "Use vectorized library operations for million-row training; avoid per-example Python loops and repeated numpy.add.at sparse updates.",
                "When importing the organizer evaluator, insert ./input into sys.path before from evaluate import evaluate; do not implement a substitute metric.",
                "For chronological pandas loops, do not access leading-underscore helper columns through named itertuples attributes; use arrays, positional tuples, or non-underscore helper names.",
                "The organizer baseline.FM API expects a dense integer matrix of feature indices shaped (rows, fields); never pass a scipy sparse/CSR one-hot matrix directly to baseline.FM.step or baseline.FM.predict.",
                "Implement the canonical internal split by computing boundary_time_ms=min(time_ms where date>=20220419), then use time_ms<boundary for prefix and time_ms>=boundary for holdout. Assert boundary_time_ms=1650295266482, prefix rows=1079102, and holdout rows=62010; a date-only April 19-21 mask is invalid.",
                "Use at most four CPU threads and keep peak memory below 3 GB; never derive thread count from os.cpu_count().",
                "Print useful training progress, but the external deterministic evaluator is authoritative.",
            ],
            "Portfolio assignment": self.current_assignment.as_prompt(),
        }

    def search_policy(self) -> Node | None:
        parent, assignment = self.scheduler.choose(self.journal)
        if parent is not None:
            assignment = replace(
                assignment,
                parent_node_id=parent.id,
                parent_code_sha256=candidate_code_sha256(parent.code),
            )
        self.current_assignment = assignment
        return parent

    def plan_and_code_query(self, prompt, retries=1) -> tuple[str, str]:
        # One request per iteration keeps the paid budget predictable. A malformed
        # response becomes a normal debuggable node instead of triggering hidden
        # retry spend.
        prompt = dict(prompt)
        if "Memory" in prompt:
            prompt["Current campaign journal"] = prompt.pop("Memory")
        if self.prompt_context is not None:
            prompt.update(self.prompt_context.sections())
        return super().plan_and_code_query(prompt, retries=retries)

    def update_data_preview(self) -> None:
        # Generic CSV previews are large, unstable, and duplicate the audited EDA.
        self.data_preview = "Disabled; use the bounded EDA evidence section."

    def _attach_candidate_spec(self, node: Node) -> Node:
        node.candidate_spec = parse_candidate_spec(
            node.plan or "", fallback_family=self.current_assignment.family
        )
        node.plan = clean_candidate_plan(node.plan or "")
        parent = node.parent
        errors = validate_candidate_spec(
            node.candidate_spec,
            expected_parent_node_id=parent.id if parent is not None else None,
            expected_parent_code_sha256=(
                candidate_code_sha256(parent.code) if parent is not None else None
            ),
            allowed_eda_observation_ids=self.allowed_eda_observation_ids,
            allowed_literature_citation_ids=self.allowed_literature_citation_ids,
        )
        if (
            self.current_assignment.action == "debug"
            and parent is not None
            and parent.candidate_spec
        ):
            if node.candidate_spec.get("model_family") != parent.candidate_spec.get(
                "model_family"
            ):
                errors.append("debug repair changed the scientific model family")
            if node.candidate_spec.get(
                "scientific_change"
            ) != parent.candidate_spec.get("scientific_change"):
                errors.append("debug repair changed the scientific hypothesis")
            if node.candidate_spec.get("hypothesis") != parent.candidate_spec.get(
                "hypothesis"
            ):
                errors.append("debug repair rephrased the locked hypothesis")
            if node.candidate_spec.get("change_scope") != parent.candidate_spec.get(
                "change_scope"
            ):
                errors.append("debug repair changed the locked change scope")
        node.candidate_spec["validation_errors"] = sorted(set(errors))
        node.candidate_spec["card_complete"] = not errors
        return node

    def _draft(self) -> Node:
        return self._attach_candidate_spec(super()._draft())

    def _improve(self, parent_node: Node) -> Node:
        return self._attach_candidate_spec(super()._improve(parent_node))

    def _debug(self, parent_node: Node) -> Node:
        return self._attach_candidate_spec(super()._debug(parent_node))

    def step(self, exec_callback):
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()
        parent_node = self.search_policy()
        if parent_node is None:
            result_node = self._draft()
        elif parent_node.is_buggy:
            result_node = self._debug(parent_node)
        else:
            result_node = self._improve(parent_node)
        card_errors = result_node.candidate_spec.get("validation_errors") or []
        result = (
            policy_rejection(["invalid candidate card: " + "; ".join(card_errors)])
            if card_errors
            else exec_callback(result_node.code, True)
        )
        self.parse_exec_result(result_node, result)
        self.journal.append(result_node)


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


def prompt_fingerprint(extra_paths: tuple[Path, ...] = ()) -> str:
    """Hash every source that materially determines the generated prompt."""

    digest = hashlib.sha256()
    paths = {Path(path).resolve() for path in (*campaign_source_paths(), *extra_paths)}
    for path in sorted(paths, key=str):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def campaign_source_paths() -> tuple[Path, ...]:
    return (
        ROOT / "challenge" / "task.md",
        ROOT / "challenge" / "prompts" / "hard_constraints.md",
        ROOT / "challenge" / "prompts" / "research_menu.md",
        ROOT / "challenge" / "research_memory" / "experiment_memory.json",
        ROOT / "challenge" / "research_memory" / "eda_summary.json",
        ROOT / "challenge" / "research_memory" / "literature_manifest.json",
        ROOT / "challenge" / "prepare_agent_data.py",
        ROOT / "challenge" / "run_aide_research.py",
        ROOT / "challenge" / "requirements-agent.txt",
        ROOT / "challenge" / "techjam_recsys" / "aide_portfolio.py",
        ROOT / "challenge" / "techjam_recsys" / "aide_reviewer.py",
        ROOT / "challenge" / "techjam_recsys" / "campaign_safety.py",
        ROOT / "challenge" / "techjam_recsys" / "diagnostics.py",
        ROOT / "challenge" / "techjam_recsys" / "eda.py",
        ROOT / "challenge" / "techjam_recsys" / "literature.py",
        ROOT / "challenge" / "techjam_recsys" / "metrics.py",
        ROOT / "challenge" / "techjam_recsys" / "protocol.py",
        ROOT / "challenge" / "techjam_recsys" / "prompt_context.py",
        ROOT / "challenge" / "techjam_recsys" / "validation.py",
        ROOT / "aide" / "agent.py",
        ROOT / "aide" / "interpreter.py",
        ROOT / "aide" / "utils" / "__init__.py",
        ROOT / "aide" / "utils" / "config.py",
    )


def dependency_fingerprint() -> str:
    distributions = (
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "lightgbm",
        "torch",
        "catboost",
        "openai",
        "psutil",
    )
    versions = {}
    for name in distributions:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    payload = json.dumps(versions, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def candidate_execution_policy(workspace_dir: Path, seed: int) -> ExecutionPolicy:
    workspace_dir = Path(workspace_dir).resolve()
    working = workspace_dir / "working"
    safe_tmp = working / "tmp"
    safe_home = working / "home"
    safe_tmp.mkdir(parents=True, exist_ok=True)
    safe_home.mkdir(parents=True, exist_ok=True)
    retained = {
        key: os.environ[key]
        for key in ("PATH", "SystemRoot", "WINDIR")
        if key in os.environ
    }
    retained.update(
        {
            "AIDE_SEED": str(seed),
            "HOME": str(safe_home),
            "USERPROFILE": str(safe_home),
            "TEMP": str(safe_tmp),
            "TMP": str(safe_tmp),
            "MPLCONFIGDIR": str(safe_tmp / "matplotlib"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUTF8": "1",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
        }
    )
    runtime_roots = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
    system_root = retained.get("SystemRoot") or retained.get("WINDIR")
    if system_root:
        runtime_roots.add(Path(system_root).resolve())
    return ExecutionPolicy(
        read_roots=tuple(
            str(path)
            for path in (
                workspace_dir,
                working,
                *sorted(runtime_roots, key=str),
            )
        ),
        write_roots=(str(working),),
        write_files=(str(workspace_dir / "runfile.py"),),
        environment=retained,
        deny_network=True,
        deny_process_creation=True,
    )


def artifact_hashes(log_dir: Path, node_id: str) -> dict[str, str]:
    output = {}
    for suffix in ("npy", "csv"):
        relative = f"predictions/{node_id}.{suffix}"
        path = Path(log_dir) / relative
        if path.exists():
            output[relative] = sha256_file(path)
    diagnostics_relative = f"diagnostics/{node_id}.json"
    diagnostics_path = Path(log_dir) / diagnostics_relative
    if diagnostics_path.exists():
        output[diagnostics_relative] = sha256_file(diagnostics_path)
    return output


def format_node_error(node: Node, parse_error: str | None) -> str | None:
    if not node.is_buggy and parse_error is None:
        return None
    details = []
    if node.exc_type:
        details.append(node.exc_type)
    if node.exc_info:
        details.append(json.dumps(node.exc_info, sort_keys=True))
    if node.analysis:
        details.append(node.analysis)
    tail = "".join(node.term_out[-30:]).strip()
    if tail:
        details.append(tail[-6000:])
    if parse_error:
        details.append(parse_error)
    return "\n".join(details) or "Unknown candidate failure"


def policy_rejection(violations: list[str]) -> ExecutionResult:
    message = "Candidate policy rejected generated source: " + "; ".join(violations)
    return ExecutionResult(
        term_out=[message + "\n"],
        exec_time=0.0,
        exc_type="CandidatePolicyViolation",
        exc_info={"violations": violations},
        exc_stack=[],
    )


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


def compact_cost_totals(state: dict[str, object] | None) -> dict[str, object]:
    """Summarize durable spend without echoing the complete request history."""

    state = dict(state or {})
    events = state.get("events")
    latest_event = events[-1] if isinstance(events, list) and events else None
    return {
        "total_estimated_cost_usd": state.get("total_estimated_cost_usd", 0.0),
        "total_input_tokens": state.get("total_input_tokens", 0),
        "total_output_tokens": state.get("total_output_tokens", 0),
        "total_requests": state.get("total_requests", 0),
        "next_notification_usd": state.get("next_notification_usd"),
        "latest_event": latest_event,
    }


def bounded_candidate_exec_seconds(
    exec_time: float | None, timeout_seconds: float
) -> float | None:
    """Normalize interpreter accounting to the configured per-trial cap."""

    if exec_time is None:
        return None
    return min(max(float(exec_time), 0.0), float(timeout_seconds))


def wall_elapsed_seconds(started_at_unix: float) -> float:
    """Return elapsed wall time, including time spent in host suspension."""

    return max(0.0, time.time() - started_at_unix)


def candidate_metrics_pass(metrics: dict[str, float] | None) -> bool:
    """Pass only when one evaluated prediction strictly beats every champion metric."""

    if metrics is None:
        return False
    try:
        return all(
            float(metrics[key]) > CHAMPION_VALID[key]
            for key in ("GAUC", "nDCG@5", "primary")
        )
    except (KeyError, TypeError, ValueError):
        return False


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def single_run_candidate_evidence(record: TrialRecord | None) -> dict[str, object]:
    """Build fail-closed evidence for one full-data, deterministic seed-0 result."""

    if record is None:
        return {"valid": False, "reason": "no qualifying candidate"}
    prediction_key = next(
        (
            artifact_id
            for artifact_id in record.artifact_ids
            if artifact_id.endswith(".npy")
        ),
        None,
    )
    prediction_sha256 = (
        record.artifact_sha256.get(prediction_key) if prediction_key else None
    )
    diagnostics_key = next(
        (
            artifact_id
            for artifact_id in record.artifact_ids
            if artifact_id.startswith("diagnostics/") and artifact_id.endswith(".json")
        ),
        None,
    )
    diagnostics_artifact_sha256 = (
        record.artifact_sha256.get(diagnostics_key) if diagnostics_key else None
    )
    required_hashes = {
        "record_sha256": record.record_sha256,
        "code_sha256": record.code_sha256,
        "prompt_sha256": record.prompt_sha256,
        "campaign_manifest_sha256": record.campaign_manifest_sha256,
        "source_sha256": record.source_sha256,
        "input_sha256": record.input_sha256,
        "dependency_sha256": record.dependency_sha256,
        "evaluator_sha256": record.evaluator_sha256,
        "prediction_sha256": prediction_sha256,
        "diagnostics_sha256": diagnostics_artifact_sha256,
        "internal_validation_sha256": record.internal_validation_sha256,
    }
    quality_gate_passed = candidate_metrics_pass(record.metrics)
    hashes_valid = all(_is_sha256(value) for value in required_hashes.values())
    valid = bool(
        record.source == "aide_generated"
        and record.status == "success"
        and record.exit_status == "success"
        and record.seed == 0
        and record.evaluation_fidelity == "full"
        and record.assignment_compliant is True
        and bool((record.config or {}).get("card_complete"))
        and record.diagnostics_sha256 == diagnostics_artifact_sha256
        and quality_gate_passed
        and hashes_valid
    )
    return {
        "valid": valid,
        "candidate_node_id": record.node_id,
        "trial_id": record.trial_id,
        "seed": record.seed,
        "evaluation_fidelity": record.evaluation_fidelity,
        "metrics": record.metrics,
        "quality_gate_passed": quality_gate_passed,
        **required_hashes,
    }


def campaign_final_designation(
    campaign_mode: str,
    candidate_evidence: dict[str, object] | None,
    manual_interventions: int,
    clean_evidence: dict[str, object],
) -> tuple[str, bool, bool]:
    """Return designation, single-run acceptance, and clean acceptance."""

    single_run_candidate_accepted = bool(
        candidate_evidence and candidate_evidence.get("valid")
    )
    clean_campaign_accepted = bool(
        campaign_mode == "clean"
        and single_run_candidate_accepted
        and manual_interventions == 0
        and clean_evidence.get("valid")
    )
    if clean_campaign_accepted:
        designation = "competition_ready"
    elif single_run_candidate_accepted:
        designation = "single_run_development_candidate"
    else:
        designation = "rejected"
    return designation, single_run_candidate_accepted, clean_campaign_accepted


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run the deterministic organizer seed and diagnostics without any API call.",
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--max-output-tokens-per-call", type=int, default=10_000)
    parser.add_argument("--per-trial-timeout", type=int, default=900)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-run-usd", type=float, default=2.0)
    parser.add_argument("--max-input-tokens", type=int, default=200_000)
    parser.add_argument("--max-output-tokens", type=int, default=100_000)
    parser.add_argument("--input-usd-per-million", type=float, required=False)
    parser.add_argument("--output-usd-per-million", type=float, required=False)
    parser.add_argument("--cost-notification-step-usd", type=float, default=10.0)
    parser.add_argument(
        "--campaign-mode", choices=("development", "clean"), default="development"
    )
    parser.add_argument(
        "--intervention-log",
        type=Path,
        default=None,
        help="Optional append-only JSONL of human intervention events.",
    )
    parser.add_argument(
        "--agent-data", type=Path, default=ROOT / "challenge" / "agent_data"
    )
    parser.add_argument(
        "--validation-index",
        type=Path,
        default=ROOT / "challenge" / "private" / "evaluator" / "validation_index.npz",
        help="Evaluator-only public-validation index outside candidate input.",
    )
    parser.add_argument(
        "--run-root", type=Path, default=ROOT / "challenge" / "runs" / "aide"
    )
    parser.add_argument(
        "--cumulative-cost-file",
        type=Path,
        default=ROOT / "challenge" / "private" / "api_spend.json",
    )
    parser.add_argument(
        "--experiment-memory",
        type=Path,
        default=ROOT / "challenge" / "research_memory" / "experiment_memory.json",
    )
    parser.add_argument(
        "--eda-summary",
        type=Path,
        default=ROOT / "challenge" / "research_memory" / "eda_summary.json",
    )
    parser.add_argument(
        "--literature-manifest",
        type=Path,
        default=ROOT / "challenge" / "research_memory" / "literature_manifest.json",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    max_search_steps = MAX_ITERATIONS - 1
    if not 1 <= args.steps <= max_search_steps:
        raise ValueError(
            f"steps must be between 1 and {max_search_steps}; the organizer seed and "
            f"generated attempts must stay within {MAX_ITERATIONS} total iterations"
        )
    context_paths = (
        args.experiment_memory.resolve(),
        args.eda_summary.resolve(),
        args.literature_manifest.resolve(),
    )
    for context_path in context_paths:
        if not context_path.exists():
            raise RuntimeError(f"Frozen prompt context is missing: {context_path}")
    experiment_memory = json.loads(args.experiment_memory.read_text(encoding="utf-8"))
    memory_claim = experiment_memory.get("content_sha256")
    memory_unsigned = dict(experiment_memory)
    memory_unsigned.pop("content_sha256", None)
    if memory_claim != canonical_sha256(memory_unsigned):
        raise RuntimeError("Experiment-memory content hash is invalid")
    eda_summary = json.loads(args.eda_summary.read_text(encoding="utf-8"))
    literature_manifest = load_manifest(args.literature_manifest)
    if args.campaign_mode == "clean" and literature_manifest.get("mode") != "frozen":
        raise RuntimeError(
            "Clean campaigns require a frozen offline literature manifest"
        )
    prompt_context = load_prompt_context(
        ROOT,
        experiment_memory_path=args.experiment_memory,
        eda_summary_path=args.eda_summary,
        literature_manifest_path=args.literature_manifest,
    )
    prompt_hash = prompt_fingerprint(context_paths)
    experiment_memory_hash = str(memory_claim)
    eda_hash = str(eda_summary.get("report_sha256") or canonical_sha256(eda_summary))
    literature_hash = str(literature_manifest["manifest_sha256"])
    scheduler_hash = canonical_sha256(
        {"scheduler": "component-aware-pareto-v2", "max_debug_depth": 3}
    )
    dry_run = {
        "paid_execution": bool(args.execute and not args.baseline_only),
        "baseline_only": bool(args.baseline_only),
        "model": args.model,
        "paid_iterations": args.steps,
        "api_calls_per_iteration": 1,
        "deterministic_review_api_calls": 0,
        "max_output_tokens_per_call": args.max_output_tokens_per_call,
        "worst_case_output_tokens": (args.steps * args.max_output_tokens_per_call),
        "per_run_approval_required": False,
        "cost_notifications_usd": args.cost_notification_step_usd,
        "campaign_mode": args.campaign_mode,
        "campaign_seed": 0,
        "evaluation_fidelity": "full",
        "max_total_iterations": MAX_ITERATIONS,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash,
        "experiment_memory_sha256": experiment_memory_hash,
        "eda_sha256": eda_hash,
        "literature_sha256": literature_hash,
        "scheduler_sha256": scheduler_hash,
        "internal_validation_sha256": "computed from train.csv on --execute",
    }
    if not args.execute and not args.baseline_only:
        print("DRY_RUN=" + json.dumps(dry_run, sort_keys=True))
        return 0
    if args.baseline_only:
        input_price = 0.0
        output_price = 0.0
        estimated_ceiling = 0.0
    else:
        required = {
            "input_usd_per_million": args.input_usd_per_million,
            "output_usd_per_million": args.output_usd_per_million,
        }
        missing = [key for key, value in required.items() if value is None]
        if missing:
            raise RuntimeError(
                "Paid run blocked; current pricing must be supplied: "
                + ", ".join(missing)
            )
        input_price = float(args.input_usd_per_million)
        output_price = float(args.output_usd_per_million)
        estimated_ceiling = estimate_uncached_cost(
            args.max_input_tokens,
            args.max_output_tokens,
            input_price,
            output_price,
        )
        if estimated_ceiling > args.max_run_usd:
            raise RuntimeError(
                f"Paid run blocked: token envelope estimates ${estimated_ceiling:.4f}, "
                f"above the configured ${args.max_run_usd:.4f} run ceiling"
            )
    if not args.agent_data.exists():
        raise RuntimeError("Run challenge/prepare_agent_data.py first")
    if not args.validation_index.exists():
        raise RuntimeError(
            "Evaluator index is missing; rerun challenge/prepare_agent_data.py so it "
            "is created outside the candidate input directory"
        )
    if not args.baseline_only:
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
    cfg.agent.data_preview = False
    # The immutable FM seed is the single draft. Every paid iteration must
    # improve or debug an evaluated node rather than spend money restarting.
    cfg.agent.search.num_drafts = 1
    cfg.agent.code.model = args.model
    # Current GPT-5 generation models do not need AIDE's legacy sampling value.
    # Keeping this unset also avoids spending a configured call on a rejected
    # request when a model does not support a custom temperature.
    cfg.agent.code.temp = None
    cfg.agent.code.max_tokens = args.max_output_tokens_per_call
    cfg = prep_cfg(cfg)
    run_id = args.run_id or f"techjam-aide-{uuid.uuid4().hex[:8]}"
    event_path = args.intervention_log or (cfg.log_dir / "campaign_events.jsonl")
    task_desc = load_task_desc(cfg)
    prep_agent_workspace(cfg)
    input_file_hashes = validate_candidate_input(cfg.workspace_dir / "input")
    input_hash = fingerprint_tree(
        cfg.workspace_dir / "input", names=CANDIDATE_INPUT_ALLOWLIST
    )
    source_paths = tuple(
        sorted(
            {
                Path(path).resolve()
                for path in (*campaign_source_paths(), *context_paths)
            },
            key=str,
        )
    )
    source_hash = fingerprint_sources(source_paths)
    dependency_hash = dependency_fingerprint()
    evaluator_hash = sha256_file(args.validation_index)
    split_frame = pd.read_csv(
        args.agent_data / "train.csv", usecols=["date", "time_ms"]
    )
    _, _, internal_split = last_days_holdout(split_frame)
    del split_frame
    internal_validation_hash = internal_split.manifest_sha256
    scheduler_hash = canonical_sha256(
        {
            "scheduler": "component-aware-pareto-v2",
            "max_debug_depth": int(cfg.agent.search.max_debug_depth),
        }
    )
    manifest = CampaignManifest(
        run_id=run_id,
        campaign_mode=args.campaign_mode,
        prompt_sha256=prompt_hash,
        source_sha256=source_hash,
        input_sha256=input_hash,
        dependency_sha256=dependency_hash,
        evaluator_sha256=evaluator_hash,
        experiment_memory_sha256=experiment_memory_hash,
        eda_sha256=eda_hash,
        literature_sha256=literature_hash,
        scheduler_sha256=scheduler_hash,
        validation_sha256=internal_validation_hash,
    )
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    (cfg.log_dir / "campaign_manifest.json").write_text(
        json.dumps(
            {
                **manifest.__dict__,
                "manifest_sha256": manifest.sha256,
                "candidate_input_files": input_file_hashes,
                "internal_validation": internal_split.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    events = CampaignEventLedger(event_path, run_id)
    if event_path.exists() and event_path.stat().st_size:
        raise RuntimeError("Campaign event ledger must be empty at run start")
    events.append(
        "run_started",
        details={
            "manifest_sha256": manifest.sha256,
            "prompt_sha256": prompt_hash,
        },
    )
    manual_interventions = count_manual_interventions(event_path)

    journal = Journal(metric_maximize=True)
    reviewer = KuaiRandPredictionReviewer(
        cfg.workspace_dir,
        args.validation_index,
        cfg.log_dir / "predictions",
    )
    agent = KuaiRandAgent(
        task_desc=task_desc,
        cfg=cfg,
        journal=journal,
        result_reviewer=reviewer,
        scheduler=PortfolioScheduler(max_debug_depth=cfg.agent.search.max_debug_depth),
        prompt_context=prompt_context,
        allowed_eda_observation_ids={
            str(item["id"]) for item in eda_summary.get("observations", [])
        },
        allowed_literature_citation_ids={
            str(item["citation_id"]) for item in literature_manifest.get("notes", [])
        },
    )
    interpreter = Interpreter(
        cfg.workspace_dir,
        **OmegaConf.to_container(cfg.exec),
        execution_policy=candidate_execution_policy(cfg.workspace_dir, seed=0),
        max_memory_bytes=3 * 1024**3,
    )
    atexit.register(interpreter.cleanup_session)
    ledger = ExperimentLedger(cfg.log_dir / "iterations.jsonl")
    tracker = ConvergenceTracker(max_iterations=min(args.steps + 1, MAX_ITERATIONS))
    backend.reset_usage_totals()
    backend.configure_cost_tracking(
        args.cumulative_cost_file,
        input_price,
        output_price,
        args.cost_notification_step_usd,
    )
    run_started = time.time()
    node_trial_ids: dict[str, str] = {}

    def assert_campaign_unchanged() -> None:
        current = {
            "source": fingerprint_sources(source_paths),
            "input": fingerprint_tree(
                cfg.workspace_dir / "input", names=CANDIDATE_INPUT_ALLOWLIST
            ),
            "dependency": dependency_fingerprint(),
            "evaluator": sha256_file(args.validation_index),
        }
        expected = {
            "source": manifest.source_sha256,
            "input": manifest.input_sha256,
            "dependency": manifest.dependency_sha256,
            "evaluator": manifest.evaluator_sha256,
        }
        if current != expected:
            raise RuntimeError(
                f"Campaign drift detected; expected={expected}, current={current}"
            )

    def execute(code: str, reset_session: bool = True):
        reviewer.clear_candidate_output()
        violations = validate_candidate_source(code)
        if violations:
            return policy_rejection(violations)
        # Seed zero is a runner invariant, not a caller-controlled campaign option.
        interpreter.execution_policy = candidate_execution_policy(
            cfg.workspace_dir, seed=0
        )
        return interpreter.run(code, reset_session)

    seed = Node(
        plan="Reproduce the immutable organizer FM baseline before research iterations.",
        code=(ROOT / "challenge" / "agent_seed.py").read_text(encoding="utf-8"),
        candidate_spec={"model_family": "official_fm_seed", "structured": True},
    )
    seed_started = time.time()
    agent.parse_exec_result(seed, execute(seed.code, True))
    journal.append(seed)
    seed_metrics, seed_error = parse_metrics(seed.analysis)
    seed_artifact_hashes = artifact_hashes(cfg.log_dir, seed.id)
    seed_record = TrialRecord(
        iteration=0,
        hypothesis=seed.plan,
        model_family="official_fm_seed",
        status="success" if seed_metrics else "failed",
        config=seed.candidate_spec,
        metrics=seed_metrics,
        error=seed_error,
        code_diff=code_diff(seed),
        wall_seconds=wall_elapsed_seconds(seed_started),
        candidate_exec_seconds=bounded_candidate_exec_seconds(
            seed.exec_time, args.per_trial_timeout
        ),
        manual_interventions=manual_interventions,
        source="organizer_seed",
        node_id=seed.id,
        code_sha256=candidate_code_sha256(seed.code),
        prompt_sha256=prompt_hash,
        prompt_version=PROMPT_VERSION,
        assignment_family="official_fm_seed",
        seed=0,
        evaluation_fidelity="full",
        evaluator_sha256=evaluator_hash,
        artifact_ids=(
            [
                f"predictions/{seed.id}.npy",
                f"predictions/{seed.id}.csv",
                f"diagnostics/{seed.id}.json",
            ]
            if seed_metrics
            else []
        ),
        artifact_sha256=seed_artifact_hashes,
        run_id=run_id,
        campaign_manifest_sha256=manifest.sha256,
        declared_model_family="official_fm_seed",
        assignment_compliant=True,
        scheduler_action="seed",
        scheduler_reason="Immutable organizer FM ancestry root.",
        scheduler_feature_vector={"organizer_root": 1.0},
        pareto_frontier_member=True,
        internal_validation_sha256=internal_validation_hash,
        diagnostics_sha256=(seed_artifact_hashes.get(f"diagnostics/{seed.id}.json")),
        decision="accepted_as_ancestry_root" if seed_metrics else "failed",
        error_type=seed.exc_type,
        error_info=seed.exc_info,
        exit_status="success" if seed_metrics else "failed",
        source_sha256=source_hash,
        input_sha256=input_hash,
        dependency_sha256=dependency_hash,
    )
    ledger.append(seed_record)
    agent.scheduler.observe(seed_record, reviewer.diagnostics_by_node.get(seed.id, {}))
    node_trial_ids[seed.id] = seed_record.trial_id
    save_run(cfg, journal)
    if not seed_metrics:
        raise RuntimeError(f"Seed baseline failed: {seed.analysis}")
    tracker.observe(float(seed_metrics["primary"]))
    assert_campaign_unchanged()

    search_iterations = () if args.baseline_only else range(1, args.steps + 1)
    for iteration in search_iterations:
        usage_before = backend.get_usage_totals()
        remaining_output = args.max_output_tokens - usage_before["output_tokens"]
        retry_reserve = args.max_output_tokens_per_call
        if remaining_output < retry_reserve:
            print(
                "Stopping before the next API call: the remaining configured "
                "output-token budget does not cover one bounded request."
            )
            break
        trial_started = time.time()
        agent.step(exec_callback=execute)
        node = journal.nodes[-1]
        usage_after = backend.get_usage_totals()
        metrics, parse_error = parse_metrics(node.analysis)
        spec = node.candidate_spec or parse_candidate_spec(
            node.plan or "", fallback_family=agent.current_assignment.family
        )
        assignment = agent.current_assignment
        diagnostics_record = reviewer.diagnostics_by_node.get(node.id, {})
        candidate_artifact_hashes = artifact_hashes(cfg.log_dir, node.id)
        frontier_ids = {
            candidate.id for candidate in pareto_frontier(journal.good_nodes)
        }
        input_tokens = usage_after["input_tokens"] - usage_before["input_tokens"]
        output_tokens = usage_after["output_tokens"] - usage_before["output_tokens"]
        actual_api_cost = estimate_uncached_cost(
            input_tokens,
            output_tokens,
            input_price,
            output_price,
        )
        can_repair = (
            node.is_buggy and node.debug_depth < agent.scheduler.max_debug_depth
        )
        if node.is_buggy:
            decision = "repair_pending" if can_repair else "abandoned"
        elif assignment.action == "debug":
            decision = "promoted_after_repair"
        else:
            decision = "evaluated"
        recovery_outcome = None
        if assignment.action == "debug":
            if node.is_buggy:
                recovery_outcome = "repair_failed" if can_repair else "abandoned"
            else:
                recovery_outcome = "repaired"
        candidate_record = TrialRecord(
            iteration=iteration,
            hypothesis=node.plan or "",
            model_family=spec["model_family"],
            status="failed" if node.is_buggy else "success",
            config=spec,
            metrics=metrics,
            parent_trial_id=(
                node_trial_ids.get(node.parent.id) if node.parent else None
            ),
            code_diff=code_diff(node),
            error=format_node_error(
                node,
                parse_error or "; ".join(spec.get("validation_errors") or []) or None,
            ),
            recovery=None,
            wall_seconds=wall_elapsed_seconds(trial_started),
            candidate_exec_seconds=bounded_candidate_exec_seconds(
                node.exec_time, args.per_trial_timeout
            ),
            llm_input_tokens=input_tokens,
            llm_output_tokens=output_tokens,
            manual_interventions=count_manual_interventions(event_path),
            source="aide_generated",
            node_id=node.id,
            code_sha256=candidate_code_sha256(node.code),
            prompt_sha256=prompt_hash,
            prompt_version=PROMPT_VERSION,
            assignment_family=agent.current_assignment.family,
            seed=0,
            evaluation_fidelity="full",
            evaluator_sha256=evaluator_hash,
            artifact_ids=(
                [
                    f"predictions/{node.id}.npy",
                    f"predictions/{node.id}.csv",
                    f"diagnostics/{node.id}.json",
                ]
                if metrics
                else []
            ),
            artifact_sha256=candidate_artifact_hashes,
            run_id=run_id,
            campaign_manifest_sha256=manifest.sha256,
            declared_model_family=spec.get("declared_model_family"),
            assignment_compliant=(
                spec.get("model_family") == agent.current_assignment.family
            ),
            scheduler_action=assignment.action,
            scheduler_reason=assignment.reason,
            scheduler_utility=assignment.utility,
            scheduler_alternatives=[
                {"family": family, "utility": utility}
                for family, utility in assignment.alternatives
            ],
            scheduler_feature_vector=dict(assignment.feature_vector),
            pareto_frontier_member=node.id in frontier_ids,
            expected_api_cost_usd=(args.max_run_usd / max(1, args.steps)),
            actual_api_cost_usd=actual_api_cost,
            internal_validation_sha256=internal_validation_hash,
            diagnostics_sha256=candidate_artifact_hashes.get(
                f"diagnostics/{node.id}.json"
            ),
            decision=decision,
            recovery_outcome=recovery_outcome,
            error_type=node.exc_type or ("MetricParseError" if parse_error else None),
            error_info=node.exc_info,
            exit_status="failed" if node.is_buggy else "success",
            source_sha256=source_hash,
            input_sha256=input_hash,
            dependency_sha256=dependency_hash,
        )
        ledger.append(candidate_record)
        agent.scheduler.observe(candidate_record, diagnostics_record)
        node_trial_ids[node.id] = candidate_record.trial_id
        save_run(cfg, journal)
        assert_campaign_unchanged()
        should_stop = (
            tracker.observe(float(metrics["primary"]))
            if metrics
            else tracker.observe_failure()
        )
        if should_stop:
            break
        if usage_after["input_tokens"] >= args.max_input_tokens:
            print("Stopping: configured input-token budget reached.")
            break

    generated: list[tuple[Node, dict[str, float], ChallengeMetric]] = []
    for node in journal.good_nodes:
        if node.parent is None:
            continue
        metrics, _ = parse_metrics(node.analysis)
        if metrics:
            generated.append((node, metrics, ChallengeMetric.from_mapping(metrics)))
    best_generated = max(
        generated, key=lambda item: float(item[1]["primary"]), default=None
    )
    breakthroughs = [item for item in generated if candidate_metrics_pass(item[1])]
    breakthrough = max(
        breakthroughs, key=lambda item: float(item[1]["primary"]), default=None
    )
    winning_record = None
    if breakthrough is not None:
        winning_node = breakthrough[0]
        winning_record = next(
            (
                record
                for record in reversed(ledger.read(validate_chain=True))
                if record.node_id == winning_node.id
            ),
            None,
        )
    candidate_evidence = single_run_candidate_evidence(winning_record)

    interpreter.cleanup_session()
    usage = backend.get_usage_totals()
    run_estimated_cost_usd = estimate_uncached_cost(
        int(usage.get("input_tokens", 0)),
        int(usage.get("output_tokens", 0)),
        input_price,
        output_price,
    )
    candidate_exec_seconds_total = sum(
        float(record.candidate_exec_seconds or 0.0)
        for record in ledger.read(validate_chain=True)
    )
    best = journal.get_best_node()
    best_metrics, _ = parse_metrics(best.analysis) if best else (None, None)
    best_aide_metrics = best_generated[1] if best_generated else None
    assert_campaign_unchanged()
    events.append(
        "run_completed",
        details={
            "manifest_sha256": manifest.sha256,
            "single_run_candidate_accepted": bool(candidate_evidence.get("valid")),
            "candidate_node_id": candidate_evidence.get("candidate_node_id"),
        },
    )
    final_manual_interventions = count_manual_interventions(event_path)
    clean_evidence = events.clean_evidence(manifest.sha256)
    final_designation, single_run_candidate_accepted, clean_campaign_accepted = (
        campaign_final_designation(
            args.campaign_mode,
            candidate_evidence,
            final_manual_interventions,
            clean_evidence,
        )
    )
    candidate_node_id = candidate_evidence.get("candidate_node_id")
    result = {
        "run_id": run_id,
        "max_run_usd": args.max_run_usd,
        "estimated_uncached_cost_ceiling": estimated_ceiling,
        "pricing_usd_per_million": {
            "input": input_price,
            "output": output_price,
        },
        "baseline_only": args.baseline_only,
        "campaign_mode": args.campaign_mode,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash,
        "experiment_memory_sha256": experiment_memory_hash,
        "eda_sha256": eda_hash,
        "literature_sha256": literature_hash,
        "scheduler_sha256": scheduler_hash,
        "internal_validation_sha256": internal_validation_hash,
        "best_primary": float(best.metric.value) if best else None,
        "best_metrics": best_metrics,
        "best_aide_generated_metrics": best_aide_metrics,
        "published_baseline": BASELINE_VALID,
        "reference_champion": CHAMPION_VALID,
        "single_run_breakthrough": single_run_candidate_accepted,
        "breakthrough": single_run_candidate_accepted,
        "candidate_evidence": candidate_evidence,
        "final_designation": final_designation,
        "single_run_candidate_accepted": single_run_candidate_accepted,
        "single_run_candidate_node_id": (
            candidate_node_id if single_run_candidate_accepted else None
        ),
        "clean_campaign_accepted": clean_campaign_accepted,
        "accepted_candidate_node_id": (
            candidate_node_id if clean_campaign_accepted else None
        ),
        "best_solution_designation": (
            "accepted_competition_solution"
            if clean_campaign_accepted
            else "research_best_only"
        ),
        "clean_campaign_evidence": clean_evidence,
        "campaign_manifest_sha256": manifest.sha256,
        "iterations": len(journal.nodes),
        "stop_reason": tracker.stop_reason,
        "wall_seconds": wall_elapsed_seconds(run_started),
        "candidate_exec_seconds_total": candidate_exec_seconds_total,
        "per_trial_timeout_seconds": args.per_trial_timeout,
        "usage": usage,
        "run_estimated_cost_usd": run_estimated_cost_usd,
        "cumulative_cost": compact_cost_totals(backend.get_cost_tracking_totals()),
        "manual_interventions": final_manual_interventions,
    }
    (cfg.log_dir / "resource_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (cfg.log_dir / "acceptance_status.json").write_text(
        json.dumps(
            {
                "accepted_candidate_node_id": result["accepted_candidate_node_id"],
                "best_solution_designation": result["best_solution_designation"],
                "clean_campaign_accepted": clean_campaign_accepted,
                "final_designation": final_designation,
                "single_run_candidate_accepted": single_run_candidate_accepted,
                "single_run_breakthrough": single_run_candidate_accepted,
                "seed": candidate_evidence.get("seed"),
                "evaluation_fidelity": candidate_evidence.get("evaluation_fidelity"),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print("TECHJAM_RESULT=" + json.dumps(result, sort_keys=True))
    return 2 if args.campaign_mode == "clean" and not clean_campaign_accepted else 0


if __name__ == "__main__":
    raise SystemExit(main())
