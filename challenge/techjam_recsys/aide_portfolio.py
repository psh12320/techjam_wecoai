"""Structured portfolio prompts, scheduling, and source policy for AIDE."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from aide.journal import Journal, Node

PROMPT_VERSION = "kuairand-aide-champion-v20"
CHAMPION_VALID = {
    "GAUC": 0.6710518008586268,
    "nDCG@5": 0.5380142516919405,
    "primary": 0.6045330262752837,
}
RICH_FM_MILESTONE_PRIMARY = 0.6035

PORTFOLIO_ORDER = (
    "rich_fm",
    "history_residual",
    "metric_aligned",
    "din_lite",
    "ensemble",
    "duration_auxiliary",
    "fwfm",
    "dcn_v2",
    "lightgcn",
    "catboost",
)

FAMILY_PRIORS = {
    "dcn_v2": 1.10,
    "history_residual": 1.05,
    "duration_auxiliary": 1.04,
    "ensemble": 0.20,
    # Two legal AIDE RankNet trials regressed both components.  Retain this
    # family for later exploration but prefer the reproducibly positive
    # DCN -> narrow-history -> shared multi-exit sequence.
    "metric_aligned": 0.20,
    "catboost": 0.25,
    "lightgcn": 0.65,
    "din_lite": 0.20,
    "fwfm": 0.50,
}

FAMILY_ALIASES = {
    "rich": "rich_fm",
    "rich-fm": "rich_fm",
    "rich_gated_fm": "rich_fm",
    "field_gated_fm": "rich_fm",
    "field-weighted-fm": "fwfm",
    "field_weighted_fm": "fwfm",
    "history": "history_residual",
    "multi_timescale_history": "history_residual",
    "rad": "duration_auxiliary",
    "d2q": "duration_auxiliary",
    "d2co": "duration_auxiliary",
    "cwm": "duration_auxiliary",
    "ranknet": "metric_aligned",
    "lambdaloss": "metric_aligned",
    "din": "din_lite",
    "multi_task_din": "din_lite",
    "dcn": "dcn_v2",
    "light_gcn": "lightgcn",
    "catboost_ranker": "catboost",
}

_SPEC_RE = re.compile(
    r"<candidate_spec>\s*(\{.*?\})\s*</candidate_spec>",
    flags=re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class PortfolioAssignment:
    family: str
    action: str
    reason: str
    utility: float = 0.0
    alternatives: tuple[tuple[str, float], ...] = ()

    def as_prompt(self) -> dict[str, str]:
        return {
            "Assigned family": self.family,
            "Action": self.action,
            "Reason": self.reason,
            "Expected utility": f"{self.utility:.4f}",
            "Alternatives considered": ", ".join(
                f"{family}={utility:.4f}" for family, utility in self.alternatives
            ),
        }


def normalize_family(value: Any, fallback: str = "research_wildcard") -> str:
    family = str(value or "").strip().lower().replace(" ", "_")
    family = FAMILY_ALIASES.get(family, family)
    substring_aliases = (
        ("fwfm", ("fwfm", "field_weighted", "field-weighted")),
        ("history_residual", ("history", "multi_timescale", "recency")),
        ("duration_auxiliary", ("duration_aux", "watch_time", "cwm", "d2q", "rad")),
        ("metric_aligned", ("ranknet", "lambda", "metric_aligned")),
        ("din_lite", ("din", "deep_interest")),
        ("dcn_v2", ("dcn", "deep_cross")),
        ("lightgcn", ("lightgcn", "light_gcn")),
        ("catboost", ("catboost", "yetirank")),
        ("ensemble", ("ensemble", "blend")),
        ("rich_fm", ("rich_fm", "rich_field", "field_gated_fm")),
    )
    for canonical, tokens in substring_aliases:
        if any(token in family for token in tokens):
            return canonical
    return family or fallback


def infer_family(text: str, fallback: str = "research_wildcard") -> str:
    lowered = text.lower()
    keywords = (
        ("rich_fm", ("rich fm", "rich-fm", "field-gated", "field gated")),
        ("fwfm", ("fwfm", "field-weighted", "field weighted")),
        ("din_lite", ("din-lite", "din lite", "deep interest network")),
        ("lightgcn", ("lightgcn", "light gcn")),
        ("dcn_v2", ("dcn-v2", "dcn v2", "deep cross")),
        ("catboost", ("catboost", "yetirank")),
        ("duration_auxiliary", ("cwm", "d2q", "d2co", "rad", "watch-time")),
        ("metric_aligned", ("ranknet", "lambdaloss", "lambda loss")),
        ("history_residual", ("history residual", "multi-timescale", "recency")),
        ("ensemble", ("ensemble", "blend")),
    )
    for family, terms in keywords:
        if any(term in lowered for term in terms):
            return family
    return normalize_family(fallback)


def clean_candidate_plan(plan: str) -> str:
    normalized = (plan or "").replace("<\\/candidate_spec>", "</candidate_spec>")
    return _SPEC_RE.sub("", normalized).strip()


def parse_candidate_spec(
    plan: str,
    *,
    fallback_family: str = "research_wildcard",
) -> dict[str, Any]:
    """Parse the tagged candidate card while retaining a safe fallback."""

    normalized_plan = (plan or "").replace("<\\/candidate_spec>", "</candidate_spec>")
    match = _SPEC_RE.search(normalized_plan)
    parsed: dict[str, Any] = {}
    parse_error: str | None = None
    if match:
        try:
            value = json.loads(match.group(1))
            if isinstance(value, dict):
                parsed = value
            else:
                parse_error = "candidate_spec must be a JSON object"
        except (TypeError, json.JSONDecodeError) as exc:
            parse_error = f"invalid candidate_spec JSON: {exc}"
    else:
        parse_error = "candidate_spec tag missing"

    family = normalize_family(
        parsed.get("model_family"),
        infer_family(normalized_plan, fallback_family),
    )
    features = parsed.get("features", [])
    if not isinstance(features, list):
        features = [str(features)]
    losses = parsed.get("losses", {})
    if isinstance(losses, list):
        losses = {str(name): None for name in losses}
    if not isinstance(losses, dict):
        losses = {"description": str(losses)}
    hyperparameters = parsed.get("hyperparameters", {})
    if not isinstance(hyperparameters, dict):
        hyperparameters = {"description": str(hyperparameters)}
    risks = parsed.get("risks", [])
    if not isinstance(risks, list):
        risks = [str(risks)]
    expected = parsed.get("expected_metric_effects", {})
    if not isinstance(expected, dict):
        expected = {"description": str(expected)}
    try:
        runtime = int(parsed.get("estimated_runtime_seconds", 900))
    except (TypeError, ValueError):
        runtime = 900

    return {
        "model_family": family,
        "declared_model_family": str(parsed.get("model_family") or family),
        "features": [str(value) for value in features],
        "losses": losses,
        "hyperparameters": hyperparameters,
        "estimated_runtime_seconds": max(1, runtime),
        "risks": [str(value) for value in risks],
        "expected_metric_effects": expected,
        "structured": parse_error is None,
        "parse_error": parse_error,
        "assignment_family": normalize_family(fallback_family),
    }


def candidate_code_sha256(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def validate_candidate_source(code: str) -> list[str]:
    """Reject direct champion reuse and obvious workspace/network escapes."""

    violations: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"generated code is not valid Python: {exc}"]

    forbidden_import_roots = {
        "challenge",
        "ctypes",
        "importlib",
        "multiprocessing",
        "runpy",
        "subprocess",
        "socket",
        "requests",
        "urllib",
        "http",
        "ftplib",
    }
    forbidden_artifacts = (
        "champion-v3",
        "champion_test",
        "rich_lite_three_seed",
        "challenge/reports",
        "challenge\\reports",
        "challenge/runs",
        "challenge\\runs",
        "run_enriched_fm",
        "run_rad",
        "train_submission",
    )
    absolute_path = re.compile(r"^[a-zA-Z]:[\\/]")
    forbidden_calls = {
        "__import__",
        "compile",
        "eval",
        "exec",
        "os.popen",
        "os.spawnl",
        "os.spawnle",
        "os.spawnlp",
        "os.spawnlpe",
        "os.spawnv",
        "os.spawnve",
        "os.spawnvp",
        "os.spawnvpe",
        "os.system",
    }

    def dotted_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = dotted_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".", 1)[0] for alias in node.names}
            blocked = sorted(roots & forbidden_import_roots)
            if blocked:
                violations.append("forbidden imports: " + ", ".join(blocked))
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in forbidden_import_roots:
                violations.append(f"forbidden import: {node.module}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.strip()
            lowered = value.lower()
            if any(token in lowered for token in forbidden_artifacts):
                violations.append(
                    "references a frozen champion artifact or implementation"
                )
            if (
                absolute_path.match(value)
                or value.startswith("../")
                or value.startswith("..\\")
                or "/../" in value.replace("\\", "/")
            ):
                violations.append("uses an absolute or parent-relative filesystem path")
        elif isinstance(node, ast.Call):
            name = dotted_name(node.func)
            if name in forbidden_calls:
                violations.append(f"forbidden dynamic or process call: {name}")
            if name in {"Path", "PurePath", "open", "os.path.join"} and any(
                isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                and argument.value.strip() == ".."
                for argument in node.args
            ):
                violations.append("uses an absolute or parent-relative filesystem path")

    if "os.cpu_count(" in code or "multiprocessing.cpu_count(" in code:
        violations.append("derives thread count from host CPU count")
    normalized_code = code.replace("\\", "/")
    if not (
        "validation_predictions.csv" in normalized_code
        and ("working" in normalized_code or "WORK_DIR" in code)
    ):
        violations.append("does not name the required validation prediction path")
    return sorted(set(violations))


class PortfolioScheduler:
    """Deterministic family coverage with bounded autonomous repair."""

    def __init__(self, max_debug_depth: int = 3):
        self.max_debug_depth = max_debug_depth

    @staticmethod
    def family(node: Node) -> str:
        spec = getattr(node, "candidate_spec", None) or {}
        assigned = normalize_family(spec.get("assignment_family"))
        if assigned in PORTFOLIO_ORDER:
            return assigned
        if spec.get("model_family"):
            return normalize_family(spec["model_family"])
        return (
            "official_fm_seed" if node.parent is None else infer_family(node.plan or "")
        )

    def choose(self, journal: Journal) -> tuple[Node | None, PortfolioAssignment]:
        debuggable = [
            node
            for node in journal.buggy_nodes
            if node.is_leaf and node.debug_depth < self.max_debug_depth
        ]
        if debuggable:
            parent = debuggable[-1]
            family = self.family(parent)
            return parent, PortfolioAssignment(
                family=family,
                action="debug",
                reason=(
                    "Repair the latest generated implementation without changing its "
                    "scientific hypothesis or model family."
                ),
                utility=2.0,
                alternatives=((family, 2.0),),
            )

        good = journal.good_nodes
        if not good:
            return None, PortfolioAssignment(
                family="rich_fm",
                action="draft",
                reason="No valid parent exists; construct the rich-FM milestone from scratch.",
                utility=2.0,
                alternatives=(("rich_fm", 2.0),),
            )

        best = journal.get_best_node()
        attempts = {family: 0 for family in PORTFOLIO_ORDER}
        successes: dict[str, list[Node]] = {family: [] for family in PORTFOLIO_ORDER}
        for node in journal.nodes:
            family = self.family(node)
            if family in attempts:
                attempts[family] += 1
                if not node.is_buggy and node.metric is not None:
                    successes[family].append(node)

        rich_reproduced = any(
            node.metric is not None
            and not node.metric.is_worst
            and float(node.metric.value) >= RICH_FM_MILESTONE_PRIMARY
            for node in successes["rich_fm"]
        )
        if not rich_reproduced and attempts["rich_fm"] < 2:
            return best, PortfolioAssignment(
                family="rich_fm",
                action="improve",
                reason=(
                    "First milestone: independently implement the proven 13-field gated "
                    "FM direction and reach the strong-rich coverage threshold before exploring other branches."
                ),
                utility=2.0,
                alternatives=(("rich_fm", 2.0),),
            )

        best_primary = float(best.metric.value) if best and best.metric else 0.0
        utilities: list[tuple[str, float]] = []
        for family in PORTFOLIO_ORDER[1:]:
            prior = FAMILY_PRIORS.get(family, 0.0)
            exploration = 0.50 if attempts[family] == 0 else 0.0
            family_best = max(
                (
                    float(node.metric.value)
                    for node in successes[family]
                    if node.metric is not None and not node.metric.is_worst
                ),
                default=None,
            )
            evidence = 0.0
            if family_best is not None:
                evidence = max(-0.5, min(0.5, (family_best - best_primary) * 250.0))
            failure_penalty = 0.15 * max(0, attempts[family] - len(successes[family]))
            repeat_penalty = 0.10 * attempts[family]
            utilities.append(
                (
                    family,
                    prior + exploration + evidence - failure_penalty - repeat_penalty,
                )
            )
        utilities.sort(key=lambda item: (-item[1], PORTFOLIO_ORDER.index(item[0])))
        family, utility = utilities[0]
        preferred_parent_family = {
            "dcn_v2": "rich_fm",
            "history_residual": "dcn_v2",
            "duration_auxiliary": "history_residual",
        }.get(family)
        parent = best
        if preferred_parent_family and successes[preferred_parent_family]:
            parent = max(
                successes[preferred_parent_family],
                key=lambda node: float(node.metric.value),
            )
        return parent, PortfolioAssignment(
            family=family,
            action="improve" if attempts[family] == 0 else "refine",
            reason=(
                "Portfolio utility selected this family from the recorded alternatives "
                "using expected signal, unexplored-family bonus, observed metric effect, "
                "implementation failures, and repeat cost. Use the required lineage parent "
                "when its dependency exists; otherwise preserve the best valid parent. "
                "Make one attributable change."
            ),
            utility=utility,
            alternatives=tuple(utilities[:5]),
        )
