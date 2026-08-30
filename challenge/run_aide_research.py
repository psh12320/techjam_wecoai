"""Budget-gated AIDE tree search with deterministic KuaiRand evaluation."""

from __future__ import annotations

import argparse
import ast
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
from pathlib import Path

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
    parse_candidate_spec,
    validate_candidate_source,
)
from challenge.techjam_recsys.protocol import (  # noqa: E402
    BASELINE_VALID,
    CHAMPION_VALID,
    ROBUST_PRIMARY_TARGET,
    ChallengeMetric,
    ConvergenceTracker,
    ExperimentLedger,
    TrialRecord,
    count_manual_interventions,
)


class KuaiRandAgent(Agent):
    def __init__(self, *args, scheduler: PortfolioScheduler | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler = scheduler or PortfolioScheduler(
            max_debug_depth=self.acfg.search.max_debug_depth
        )
        self.current_assignment = PortfolioAssignment(
            family="rich_fm",
            action="improve",
            reason="Establish the required rich-FM milestone.",
        )

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
                "with keys model_family, features (list), losses (object), "
                "hyperparameters (object), estimated_runtime_seconds (integer), "
                "risks (list), and expected_metric_effects (object with GAUC and "
                "nDCG@5). The model_family value must equal the assigned scientific-change family, not merely the parent's base family. "
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
                "Use a fixed random seed and finish within the execution timeout.",
                "Read the base seed from int(os.environ.get('AIDE_SEED', '0')) and use it for every random generator.",
                "During search train exactly one AIDE_SEED; do not train or ensemble multiple seeds. The runner confirms seeds 0, 1, and 2 only after a breakthrough.",
                "Use vectorized library operations for million-row training; avoid per-example Python loops and repeated numpy.add.at sparse updates.",
                "When importing the organizer evaluator, insert ./input into sys.path before from evaluate import evaluate; do not implement a substitute metric.",
                "For chronological pandas loops, do not access leading-underscore helper columns through named itertuples attributes; use arrays, positional tuples, or non-underscore helper names.",
                "Use at most four CPU threads and keep peak memory below 3 GB; never derive thread count from os.cpu_count().",
                "Print useful training progress, but the external deterministic evaluator is authoritative.",
            ],
            "Portfolio assignment": self.current_assignment.as_prompt(),
        }

    def search_policy(self) -> Node | None:
        parent, assignment = self.scheduler.choose(self.journal)
        self.current_assignment = assignment
        return parent

    def plan_and_code_query(self, prompt, retries=1) -> tuple[str, str]:
        # One request per iteration keeps the paid budget predictable. A malformed
        # response becomes a normal debuggable node instead of triggering hidden
        # retry spend.
        return super().plan_and_code_query(prompt, retries=retries)

    def _attach_candidate_spec(self, node: Node) -> Node:
        node.candidate_spec = parse_candidate_spec(
            node.plan or "", fallback_family=self.current_assignment.family
        )
        node.plan = clean_candidate_plan(node.plan or "")
        return node

    def _draft(self) -> Node:
        return self._attach_candidate_spec(super()._draft())

    def _improve(self, parent_node: Node) -> Node:
        return self._attach_candidate_spec(super()._improve(parent_node))

    def _debug(self, parent_node: Node) -> Node:
        return self._attach_candidate_spec(super()._debug(parent_node))


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


def prompt_fingerprint() -> str:
    """Hash every source that materially determines the generated prompt."""

    digest = hashlib.sha256()
    for path in campaign_source_paths():
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def campaign_source_paths() -> tuple[Path, ...]:
    return (
        ROOT / "challenge" / "task.md",
        ROOT / "challenge" / "prepare_agent_data.py",
        ROOT / "challenge" / "run_aide_research.py",
        ROOT / "challenge" / "requirements-agent.txt",
        ROOT / "challenge" / "techjam_recsys" / "aide_portfolio.py",
        ROOT / "challenge" / "techjam_recsys" / "aide_reviewer.py",
        ROOT / "challenge" / "techjam_recsys" / "campaign_safety.py",
        ROOT / "challenge" / "techjam_recsys" / "metrics.py",
        ROOT / "challenge" / "techjam_recsys" / "protocol.py",
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


def uses_aide_seed_control(code: str) -> bool:
    """Require the environment seed to flow into at least one RNG call."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    seed_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = getattr(node, "value", None)
            if value is not None and "AIDE_SEED" in ast.unparse(value):
                targets = getattr(node, "targets", [getattr(node, "target", None)])
                for target in targets:
                    if isinstance(target, ast.Name):
                        seed_names.add(target.id)
    rng_tokens = ("seed", "manual_seed", "default_rng")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if any(token in name for token in rng_tokens) and any(
                isinstance(arg, ast.Name) and arg.id in seed_names for arg in node.args
            ):
                return True
    return False


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


def confirmation_seed_passes(metrics: dict[str, float] | None) -> bool:
    """A parsed seed result passes only when every champion component improves."""

    return bool(metrics and ChallengeMetric.from_mapping(metrics).beats_champion)


def campaign_final_designation(
    campaign_mode: str,
    confirmation: dict[str, object] | None,
    manual_interventions: int,
    clean_evidence: dict[str, object],
) -> tuple[str, bool, bool]:
    """Return designation, robust-candidate acceptance, and clean acceptance."""

    robust_candidate_accepted = bool(
        confirmation and confirmation.get("accepted")
    )
    clean_campaign_accepted = bool(
        campaign_mode == "clean"
        and robust_candidate_accepted
        and manual_interventions == 0
        and clean_evidence.get("valid")
    )
    if clean_campaign_accepted:
        designation = "competition_ready"
    elif robust_candidate_accepted:
        designation = "robust_development_candidate"
    else:
        designation = "rejected"
    return designation, robust_candidate_accepted, clean_campaign_accepted


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--model", default="gpt-5.4")
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
    parser.add_argument("--confirm-on-breakthrough", action="store_true")
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    confirmation_requested = (
        args.confirm_on_breakthrough or args.campaign_mode == "clean"
    )
    max_search_steps = 46 if confirmation_requested else 49
    if not 1 <= args.steps <= max_search_steps:
        raise ValueError(
            f"steps must be between 1 and {max_search_steps}; the organizer seed and "
            "reserved deterministic confirmation attempts must stay within 50 iterations"
        )
    prompt_hash = prompt_fingerprint()
    dry_run = {
        "paid_execution": bool(args.execute),
        "model": args.model,
        "paid_iterations": args.steps,
        "api_calls_per_iteration": 1,
        "deterministic_review_api_calls": 0,
        "max_output_tokens_per_call": args.max_output_tokens_per_call,
        "worst_case_output_tokens": (args.steps * args.max_output_tokens_per_call),
        "per_run_approval_required": False,
        "cost_notifications_usd": args.cost_notification_step_usd,
        "campaign_mode": args.campaign_mode,
        "confirmation_required": confirmation_requested,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash,
    }
    if not args.execute:
        print("DRY_RUN=" + json.dumps(dry_run, sort_keys=True))
        return 0
    required = {
        "input_usd_per_million": args.input_usd_per_million,
        "output_usd_per_million": args.output_usd_per_million,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise RuntimeError(
            "Paid run blocked; current pricing must be supplied: " + ", ".join(missing)
        )
    estimated_ceiling = estimate_uncached_cost(
        args.max_input_tokens,
        args.max_output_tokens,
        args.input_usd_per_million,
        args.output_usd_per_million,
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
    source_hash = fingerprint_sources(campaign_source_paths())
    dependency_hash = dependency_fingerprint()
    evaluator_hash = sha256_file(args.validation_index)
    manifest = CampaignManifest(
        run_id=run_id,
        campaign_mode=args.campaign_mode,
        prompt_sha256=prompt_hash,
        source_sha256=source_hash,
        input_sha256=input_hash,
        dependency_sha256=dependency_hash,
        evaluator_sha256=evaluator_hash,
    )
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    (cfg.log_dir / "campaign_manifest.json").write_text(
        json.dumps(
            {
                **manifest.__dict__,
                "manifest_sha256": manifest.sha256,
                "candidate_input_files": input_file_hashes,
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
    )
    interpreter = Interpreter(
        cfg.workspace_dir,
        **OmegaConf.to_container(cfg.exec),
        execution_policy=candidate_execution_policy(cfg.workspace_dir, seed=0),
        max_memory_bytes=3 * 1024**3,
    )
    atexit.register(interpreter.cleanup_session)
    ledger = ExperimentLedger(cfg.log_dir / "iterations.jsonl")
    tracker = ConvergenceTracker(max_iterations=min(args.steps + 1, 50))
    backend.reset_usage_totals()
    backend.configure_cost_tracking(
        args.cumulative_cost_file,
        args.input_usd_per_million,
        args.output_usd_per_million,
        args.cost_notification_step_usd,
    )
    run_started = time.time()
    node_trial_ids: dict[str, str] = {}

    def assert_campaign_unchanged() -> None:
        current = {
            "source": fingerprint_sources(campaign_source_paths()),
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
        interpreter.execution_policy = candidate_execution_policy(
            cfg.workspace_dir, seed=int(os.environ.get("AIDE_SEED", "0"))
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
        artifact_ids=(
            [f"predictions/{seed.id}.npy", f"predictions/{seed.id}.csv"]
            if seed_metrics
            else []
        ),
        artifact_sha256=artifact_hashes(cfg.log_dir, seed.id),
        run_id=run_id,
        campaign_manifest_sha256=manifest.sha256,
        declared_model_family="official_fm_seed",
        assignment_compliant=True,
        scheduler_action="seed",
        scheduler_reason="Immutable organizer FM ancestry root.",
        decision="accepted_as_ancestry_root" if seed_metrics else "failed",
        error_type=seed.exc_type,
        error_info=seed.exc_info,
        exit_status="success" if seed_metrics else "failed",
        source_sha256=source_hash,
        input_sha256=input_hash,
        dependency_sha256=dependency_hash,
    )
    ledger.append(seed_record)
    node_trial_ids[seed.id] = seed_record.trial_id
    save_run(cfg, journal)
    if not seed_metrics:
        raise RuntimeError(f"Seed baseline failed: {seed.analysis}")
    tracker.observe(float(seed_metrics["primary"]))
    assert_campaign_unchanged()

    for iteration in range(1, args.steps + 1):
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
            error=format_node_error(node, parse_error),
            recovery=None,
            wall_seconds=wall_elapsed_seconds(trial_started),
            candidate_exec_seconds=bounded_candidate_exec_seconds(
                node.exec_time, args.per_trial_timeout
            ),
            llm_input_tokens=(
                usage_after["input_tokens"] - usage_before["input_tokens"]
            ),
            llm_output_tokens=(
                usage_after["output_tokens"] - usage_before["output_tokens"]
            ),
            manual_interventions=count_manual_interventions(event_path),
            source="aide_generated",
            node_id=node.id,
            code_sha256=candidate_code_sha256(node.code),
            prompt_sha256=prompt_hash,
            prompt_version=PROMPT_VERSION,
            assignment_family=agent.current_assignment.family,
            seed=0,
            artifact_ids=(
                [f"predictions/{node.id}.npy", f"predictions/{node.id}.csv"]
                if metrics
                else []
            ),
            artifact_sha256=artifact_hashes(cfg.log_dir, node.id),
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
    best_generated = max(generated, key=lambda item: item[2].primary, default=None)
    breakthroughs = [item for item in generated if item[2].beats_champion]
    breakthrough = max(breakthroughs, key=lambda item: item[2].primary, default=None)

    confirmation: dict[str, object] | None = None
    if confirmation_requested and breakthrough is not None:
        winning_node, _, _ = breakthrough
        seed_results = []
        original_prediction = cfg.log_dir / "predictions" / f"{winning_node.id}.npy"
        original_prediction_sha256 = (
            sha256_file(original_prediction) if original_prediction.exists() else None
        )
        original_seed = os.environ.get("AIDE_SEED")
        try:
            for confirmation_seed in (0, 1, 2):
                if (
                    wall_elapsed_seconds(run_started) + args.per_trial_timeout
                    >= tracker.max_wall_seconds
                ):
                    seed_results.append(
                        {
                            "seed": confirmation_seed,
                            "metrics": None,
                            "error": "Insufficient six-hour wall-clock reserve.",
                        }
                    )
                    break
                os.environ["AIDE_SEED"] = str(confirmation_seed)
                confirmation_started = time.time()
                confirmation_node = Node(
                    plan=f"Deterministic seed-{confirmation_seed} confirmation.",
                    code=winning_node.code,
                    parent=winning_node,
                    candidate_spec=dict(winning_node.candidate_spec),
                )
                agent.parse_exec_result(
                    confirmation_node, execute(confirmation_node.code, True)
                )
                journal.append(confirmation_node)
                confirmed_metrics, confirmation_error = parse_metrics(
                    confirmation_node.analysis
                )
                seed_passed = confirmation_seed_passes(confirmed_metrics)
                confirmation_record = TrialRecord(
                    iteration=len(journal.nodes) - 1,
                    hypothesis=confirmation_node.plan,
                    model_family=winning_node.candidate_spec.get(
                        "model_family", "research_wildcard"
                    ),
                    status="evaluated" if confirmed_metrics else "failed",
                    config=dict(winning_node.candidate_spec),
                    metrics=confirmed_metrics,
                    parent_trial_id=node_trial_ids.get(winning_node.id),
                    code_diff="",
                    error=format_node_error(confirmation_node, confirmation_error),
                    wall_seconds=wall_elapsed_seconds(confirmation_started),
                    candidate_exec_seconds=bounded_candidate_exec_seconds(
                        confirmation_node.exec_time, args.per_trial_timeout
                    ),
                    manual_interventions=count_manual_interventions(event_path),
                    source="aide_seed_confirmation",
                    node_id=confirmation_node.id,
                    code_sha256=candidate_code_sha256(confirmation_node.code),
                    prompt_sha256=prompt_hash,
                    prompt_version=PROMPT_VERSION,
                    assignment_family=winning_node.candidate_spec.get(
                        "assignment_family"
                    ),
                    seed=confirmation_seed,
                    artifact_ids=(
                        [
                            f"predictions/{confirmation_node.id}.npy",
                            f"predictions/{confirmation_node.id}.csv",
                        ]
                        if confirmed_metrics
                        else []
                    ),
                    artifact_sha256=artifact_hashes(cfg.log_dir, confirmation_node.id),
                    run_id=run_id,
                    campaign_manifest_sha256=manifest.sha256,
                    declared_model_family=winning_node.candidate_spec.get(
                        "declared_model_family"
                    ),
                    assignment_compliant=True,
                    scheduler_action="confirm_seed",
                    scheduler_reason="Behavioral three-seed robustness confirmation.",
                    decision=(
                        "seed_pass"
                        if seed_passed
                        else "seed_fail" if confirmed_metrics else "rejected"
                    ),
                    error_type=confirmation_node.exc_type,
                    error_info=confirmation_node.exc_info,
                    exit_status="success" if confirmed_metrics else "failed",
                    source_sha256=source_hash,
                    input_sha256=input_hash,
                    dependency_sha256=dependency_hash,
                )
                ledger.append(confirmation_record)
                node_trial_ids[confirmation_node.id] = confirmation_record.trial_id
                seed_results.append(
                    {
                        "seed": confirmation_seed,
                        "metrics": confirmed_metrics,
                        "error": confirmation_error,
                        "passed_champion": seed_passed,
                        "prediction_sha256": confirmation_record.artifact_sha256.get(
                            f"predictions/{confirmation_node.id}.npy"
                        ),
                    }
                )
                assert_campaign_unchanged()
        finally:
            if original_seed is None:
                os.environ.pop("AIDE_SEED", None)
            else:
                os.environ["AIDE_SEED"] = original_seed

        complete_metrics = [
            item["metrics"] for item in seed_results if item.get("metrics") is not None
        ]
        mean_metrics = None
        robust = False
        seed_controlled = uses_aide_seed_control(winning_node.code)
        same_seed_reproduced = bool(
            seed_results
            and seed_results[0].get("seed") == 0
            and seed_results[0].get("prediction_sha256") == original_prediction_sha256
        )
        all_seed_breakthroughs = False
        if len(complete_metrics) == 3:
            mean_metrics = {
                key: sum(float(metric[key]) for metric in complete_metrics) / 3.0
                for key in ("GAUC", "nDCG@5", "primary")
            }
            all_seed_breakthroughs = all(
                ChallengeMetric.from_mapping(metric).beats_champion
                for metric in complete_metrics
            )
            robust = (
                mean_metrics["GAUC"] > CHAMPION_VALID["GAUC"]
                and mean_metrics["nDCG@5"] > CHAMPION_VALID["nDCG@5"]
                and mean_metrics["primary"] >= ROBUST_PRIMARY_TARGET
                and all_seed_breakthroughs
                and seed_controlled
                and same_seed_reproduced
            )
        confirmation = {
            "candidate_node_id": winning_node.id,
            "seed_controlled": seed_controlled,
            "same_seed_reproduced": same_seed_reproduced,
            "all_seeds_beat_champion": all_seed_breakthroughs,
            "seeds": seed_results,
            "mean_metrics": mean_metrics,
            "robust_target_primary": ROBUST_PRIMARY_TARGET,
            "accepted": robust,
        }
        save_run(cfg, journal)

    interpreter.cleanup_session()
    usage = backend.get_usage_totals()
    run_estimated_cost_usd = estimate_uncached_cost(
        int(usage.get("input_tokens", 0)),
        int(usage.get("output_tokens", 0)),
        args.input_usd_per_million,
        args.output_usd_per_million,
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
            "single_seed_breakthrough": breakthrough is not None,
            "confirmation_accepted": bool(
                confirmation and confirmation.get("accepted")
            ),
        },
    )
    final_manual_interventions = count_manual_interventions(event_path)
    clean_evidence = events.clean_evidence(manifest.sha256)
    final_designation, robust_candidate_accepted, clean_campaign_accepted = (
        campaign_final_designation(
            args.campaign_mode,
            confirmation,
            final_manual_interventions,
            clean_evidence,
        )
    )
    confirmed_candidate_node_id = (
        confirmation.get("candidate_node_id") if confirmation else None
    )
    result = {
        "run_id": run_id,
        "max_run_usd": args.max_run_usd,
        "estimated_uncached_cost_ceiling": estimated_ceiling,
        "pricing_usd_per_million": {
            "input": args.input_usd_per_million,
            "output": args.output_usd_per_million,
        },
        "campaign_mode": args.campaign_mode,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash,
        "best_primary": float(best.metric.value) if best else None,
        "best_metrics": best_metrics,
        "best_aide_generated_metrics": best_aide_metrics,
        "published_baseline": BASELINE_VALID,
        "reference_champion": CHAMPION_VALID,
        "single_seed_breakthrough": breakthrough is not None,
        "breakthrough": robust_candidate_accepted,
        "confirmation": confirmation,
        "final_designation": final_designation,
        "robust_candidate_accepted": robust_candidate_accepted,
        "robust_candidate_node_id": (
            confirmed_candidate_node_id if robust_candidate_accepted else None
        ),
        "clean_campaign_accepted": clean_campaign_accepted,
        "accepted_candidate_node_id": (
            confirmed_candidate_node_id if clean_campaign_accepted else None
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
        "cumulative_cost": backend.get_cost_tracking_totals(),
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
                "robust_candidate_accepted": robust_candidate_accepted,
                "single_seed_breakthrough": breakthrough is not None,
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
