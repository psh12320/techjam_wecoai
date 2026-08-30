"""Structured portfolio prompts, scheduling, and source policy for AIDE."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from aide.journal import Journal, Node

PROMPT_VERSION = "kuairand-aide-autonomy-v24"
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

FAMILY_EVIDENCE_PRIORS = {
    # Small, frozen evidence terms from the audited ledgers, EDA, and cited
    # literature. They are one scheduler feature, not a forced family order.
    "rich_fm": 0.05,
    "history_residual": 0.25,
    "duration_auxiliary": 0.22,
    "dcn_v2": 0.15,
    "metric_aligned": 0.10,
    "ensemble": 0.08,
    "fwfm": 0.02,
    "din_lite": 0.00,
    "lightgcn": -0.10,
    "catboost": -0.12,
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

REQUIRED_CANDIDATE_FIELDS = (
    "parent_node_id",
    "parent_code_sha256",
    "model_family",
    "eda_observation_ids",
    "literature_citation_ids",
    "scientific_change",
    "change_scope",
    "preserved_parent_components",
    "hypothesis",
    "features",
    "losses",
    "hyperparameters",
    "target_metric",
    "expected_metric_effects",
    "estimated_runtime_seconds",
    "estimated_memory_mb",
    "risks",
    "abort_criteria",
    "falsification_condition",
    "fidelity",
    "internal_validation",
)

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
    parent_node_id: str | None = None
    parent_code_sha256: str | None = None
    locked_candidate_fields: dict[str, Any] = field(default_factory=dict)
    feature_vector: dict[str, float] = field(default_factory=dict)

    def as_prompt(self) -> dict[str, str]:
        return {
            "Assigned family": self.family,
            "Action": self.action,
            "Reason": self.reason,
            "Expected utility": f"{self.utility:.4f}",
            "Alternatives considered": ", ".join(
                f"{family}={utility:.4f}" for family, utility in self.alternatives
            ),
            "Selected parent node": self.parent_node_id or "organizer root/new draft",
            "Selected parent code SHA-256": (
                self.parent_code_sha256 or "none (organizer root/new draft)"
            ),
            "Locked repair fields (copy values verbatim)": json.dumps(
                self.locked_candidate_fields, sort_keys=True
            ),
            "Scheduler evidence": json.dumps(self.feature_vector, sort_keys=True),
        }


@dataclass
class FamilyStats:
    family: str
    attempts: int = 0
    successes: int = 0
    parent_improvements: int = 0
    parent_dominated_attempts: int = 0
    failures: int = 0
    timeouts: int = 0
    total_exec_seconds: float = 0.0
    total_api_cost_usd: float = 0.0
    best_primary: float | None = None
    best_gauc: float | None = None
    best_ndcg5: float | None = None
    max_prediction_correlation: float | None = None
    unique_scientific_changes: int = 0
    missing_internal_validation: int = 0


def node_metrics(node: Node) -> dict[str, float] | None:
    """Read trusted component metrics embedded by the deterministic reviewer."""

    try:
        payload = json.loads(node.analysis or "")
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            return None
        return {
            "GAUC": float(metrics["GAUC"]),
            "nDCG@5": float(metrics["nDCG@5"]),
            "primary": float(metrics["primary"]),
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def pareto_frontier(nodes: list[Node]) -> list[Node]:
    """Return nondominated candidates over GAUC, nDCG@5, and primary."""

    valid = [(node, node_metrics(node)) for node in nodes]
    valid = [(node, metrics) for node, metrics in valid if metrics is not None]
    frontier: list[Node] = []
    for node, metrics in valid:
        dominated = False
        for other, other_metrics in valid:
            if other is node:
                continue
            at_least = all(
                other_metrics[key] >= metrics[key]
                for key in ("GAUC", "nDCG@5", "primary")
            )
            strictly = any(
                other_metrics[key] > metrics[key]
                for key in ("GAUC", "nDCG@5", "primary")
            )
            if at_least and strictly:
                dominated = True
                break
        if not dominated:
            frontier.append(node)
    return sorted(frontier, key=lambda value: value.id)


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

    def string_list(value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        return [str(item) for item in value]

    features = string_list(parsed.get("features"))
    losses = parsed.get("losses", {})
    if isinstance(losses, list):
        losses = {str(name): None for name in losses}
    if not isinstance(losses, dict):
        losses = {"description": str(losses)}
    hyperparameters = parsed.get("hyperparameters", {})
    if not isinstance(hyperparameters, dict):
        hyperparameters = {"description": str(hyperparameters)}
    risks = string_list(parsed.get("risks"))
    expected = parsed.get("expected_metric_effects", {})
    if not isinstance(expected, dict):
        expected = {"description": str(expected)}
    try:
        runtime = int(parsed.get("estimated_runtime_seconds", 900))
    except (TypeError, ValueError):
        runtime = 900
    try:
        memory_mb = int(parsed.get("estimated_memory_mb", 3072))
    except (TypeError, ValueError):
        memory_mb = 3072

    missing_fields = [
        field for field in REQUIRED_CANDIDATE_FIELDS if field not in parsed
    ]
    validation_errors = ([parse_error] if parse_error is not None else []) + [
        f"candidate_spec missing required field: {field}" for field in missing_fields
    ]

    return {
        "parent_node_id": str(parsed.get("parent_node_id") or ""),
        "parent_code_sha256": str(parsed.get("parent_code_sha256") or ""),
        "model_family": family,
        "declared_model_family": str(parsed.get("model_family") or family),
        "eda_observation_ids": string_list(parsed.get("eda_observation_ids")),
        "literature_citation_ids": string_list(parsed.get("literature_citation_ids")),
        "scientific_change": str(parsed.get("scientific_change") or ""),
        "change_scope": str(parsed.get("change_scope") or ""),
        "preserved_parent_components": string_list(
            parsed.get("preserved_parent_components")
        ),
        "hypothesis": str(parsed.get("hypothesis") or ""),
        "features": features,
        "losses": losses,
        "hyperparameters": hyperparameters,
        "target_metric": str(parsed.get("target_metric") or ""),
        "estimated_runtime_seconds": max(1, runtime),
        "estimated_memory_mb": max(1, memory_mb),
        "risks": risks,
        "abort_criteria": string_list(parsed.get("abort_criteria")),
        "falsification_condition": str(parsed.get("falsification_condition") or ""),
        "fidelity": str(parsed.get("fidelity") or ""),
        "internal_validation": (
            parsed.get("internal_validation")
            if isinstance(parsed.get("internal_validation"), dict)
            else {}
        ),
        "expected_metric_effects": expected,
        "structured": parse_error is None,
        "parse_error": parse_error,
        "card_complete": not validation_errors,
        "validation_errors": validation_errors,
        "assignment_family": normalize_family(fallback_family),
    }


def validate_candidate_spec(
    spec: dict[str, Any],
    *,
    expected_parent_node_id: str | None = None,
    expected_parent_code_sha256: str | None = None,
    allowed_eda_observation_ids: set[str] | None = None,
    allowed_literature_citation_ids: set[str] | None = None,
) -> list[str]:
    """Strictly validate a candidate card before spending candidate compute."""

    errors = list(spec.get("validation_errors") or [])
    nonempty_text = (
        "model_family",
        "scientific_change",
        "change_scope",
        "hypothesis",
        "target_metric",
        "falsification_condition",
        "fidelity",
    )
    for field in nonempty_text:
        if not str(spec.get(field) or "").strip():
            errors.append(f"candidate_spec field must be non-empty: {field}")
    for field in (
        "features",
        "preserved_parent_components",
        "losses",
        "expected_metric_effects",
        "risks",
        "abort_criteria",
        "internal_validation",
    ):
        if not spec.get(field):
            errors.append(f"candidate_spec field must be non-empty: {field}")
    effects = spec.get("expected_metric_effects") or {}
    for metric in ("GAUC", "nDCG@5", "primary"):
        if metric not in effects:
            errors.append(f"expected_metric_effects missing {metric}")
    if spec.get("change_scope") not in {
        "features",
        "architecture",
        "loss",
        "training",
        "reranking",
    }:
        errors.append(
            "change_scope must be one of features, architecture, loss, training, or reranking"
        )
    if int(spec.get("estimated_runtime_seconds") or 0) > 900:
        errors.append("estimated_runtime_seconds exceeds the 900-second trial limit")
    if int(spec.get("estimated_memory_mb") or 0) > 3072:
        errors.append("estimated_memory_mb exceeds the 3-GB trial limit")
    if spec.get("fidelity") not in {"screen", "full"}:
        errors.append("fidelity must be 'screen' or 'full'")
    if (
        expected_parent_node_id is not None
        and spec.get("parent_node_id") != expected_parent_node_id
    ):
        errors.append("parent_node_id does not match the selected lineage parent")
    if (
        expected_parent_code_sha256 is not None
        and spec.get("parent_code_sha256") != expected_parent_code_sha256
    ):
        errors.append("parent_code_sha256 does not match the selected lineage parent")
    if allowed_eda_observation_ids is not None:
        unknown = sorted(
            set(spec.get("eda_observation_ids") or []) - allowed_eda_observation_ids
        )
        if unknown:
            errors.append("unknown EDA observation IDs: " + ", ".join(unknown))
    if allowed_literature_citation_ids is not None:
        unknown = sorted(
            set(spec.get("literature_citation_ids") or [])
            - allowed_literature_citation_ids
        )
        if unknown:
            errors.append("unknown literature citation IDs: " + ", ".join(unknown))
    return sorted(set(errors))


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
    """Component-aware Pareto scheduling with bounded autonomous repair."""

    def __init__(self, max_debug_depth: int = 3):
        self.max_debug_depth = max_debug_depth
        self._observations: list[tuple[Any, dict[str, Any]]] = []

    def observe(self, record: Any, diagnostics: dict[str, Any] | None = None) -> None:
        """Persist trusted outcome/cost evidence for later scheduling decisions."""

        self._observations.append((record, dict(diagnostics or {})))

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

    def _family_stats(self, journal: Journal) -> dict[str, FamilyStats]:
        stats = {family: FamilyStats(family=family) for family in PORTFOLIO_ORDER}
        changes: dict[str, set[str]] = {family: set() for family in PORTFOLIO_ORDER}
        seen_nodes: set[str] = set()
        for node in journal.nodes:
            family = self.family(node)
            if family not in stats or family == "official_fm_seed":
                continue
            seen_nodes.add(node.id)
            item = stats[family]
            item.attempts += 1
            spec = node.candidate_spec or {}
            change = str(spec.get("scientific_change") or "").strip().lower()
            if change:
                changes[family].add(change)
            if not spec.get("internal_validation"):
                item.missing_internal_validation += 1
            metrics = node_metrics(node)
            if metrics is None or node.is_buggy:
                item.failures += 1
                if node.exc_type == "TimeoutError":
                    item.timeouts += 1
            else:
                item.successes += 1
                parent_metrics = node_metrics(node.parent) if node.parent else None
                if parent_metrics is not None:
                    dominated = all(
                        parent_metrics[key] >= metrics[key]
                        for key in ("GAUC", "nDCG@5", "primary")
                    ) and any(
                        parent_metrics[key] > metrics[key]
                        for key in ("GAUC", "nDCG@5", "primary")
                    )
                    if dominated:
                        item.parent_dominated_attempts += 1
                    elif all(
                        metrics[key] > parent_metrics[key] for key in ("GAUC", "nDCG@5")
                    ):
                        item.parent_improvements += 1
                item.best_primary = max(
                    item.best_primary or float("-inf"), metrics["primary"]
                )
                item.best_gauc = max(item.best_gauc or float("-inf"), metrics["GAUC"])
                item.best_ndcg5 = max(
                    item.best_ndcg5 or float("-inf"), metrics["nDCG@5"]
                )
            item.total_exec_seconds += float(node.exec_time or 0.0)

        for record, diagnostics in self._observations:
            family = normalize_family(getattr(record, "model_family", None))
            if family not in stats:
                continue
            node_id = getattr(record, "node_id", None)
            item = stats[family]
            if node_id not in seen_nodes:
                item.attempts += 1
                metrics = getattr(record, "metrics", None)
                if metrics:
                    item.successes += 1
                    item.best_primary = max(
                        item.best_primary or float("-inf"), float(metrics["primary"])
                    )
                    item.best_gauc = max(
                        item.best_gauc or float("-inf"), float(metrics["GAUC"])
                    )
                    item.best_ndcg5 = max(
                        item.best_ndcg5 or float("-inf"), float(metrics["nDCG@5"])
                    )
                else:
                    item.failures += 1
                    if getattr(record, "error_type", None) == "TimeoutError":
                        item.timeouts += 1
                item.total_exec_seconds += float(
                    getattr(record, "candidate_exec_seconds", None) or 0.0
                )
            item.total_api_cost_usd += float(
                getattr(record, "actual_api_cost_usd", None)
                or diagnostics.get("api_cost_usd", 0.0)
            )
            correlation = diagnostics.get("max_frontier_prediction_correlation")
            if correlation is not None:
                item.max_prediction_correlation = max(
                    item.max_prediction_correlation or float("-inf"),
                    float(correlation),
                )
        for family, values in changes.items():
            stats[family].unique_scientific_changes = len(values)
        return stats

    @staticmethod
    def _compatible_parent(
        family: str, frontier: list[Node], fallback: Node
    ) -> tuple[Node, float]:
        preferred = {
            "history_residual": {"rich_fm", "dcn_v2"},
            "duration_auxiliary": {"history_residual", "dcn_v2", "rich_fm"},
            "metric_aligned": {"duration_auxiliary", "history_residual", "dcn_v2"},
            "din_lite": {"history_residual", "rich_fm"},
            "ensemble": {"duration_auxiliary", "history_residual", "dcn_v2"},
            "dcn_v2": {"rich_fm"},
            "fwfm": {"rich_fm"},
            "lightgcn": {"rich_fm"},
            "catboost": {"rich_fm"},
        }.get(family, set())
        candidates = [
            node for node in frontier if PortfolioScheduler.family(node) in preferred
        ]
        if not candidates:
            return fallback, 0.0
        parent = max(candidates, key=lambda node: (node_metrics(node) or {})["primary"])
        return parent, 0.20

    def choose(self, journal: Journal) -> tuple[Node | None, PortfolioAssignment]:
        debuggable = [
            node
            for node in journal.buggy_nodes
            if node.is_leaf and node.debug_depth < self.max_debug_depth
        ]
        if debuggable:
            parent = debuggable[-1]
            family = self.family(parent)
            parent_spec = parent.candidate_spec or {}
            locked_fields = {
                key: parent_spec.get(key)
                for key in (
                    "model_family",
                    "scientific_change",
                    "hypothesis",
                    "change_scope",
                )
                if parent_spec.get(key) is not None
            }
            return parent, PortfolioAssignment(
                family=family,
                action="debug",
                reason=(
                    "Repair only the latest implementation bug. Copy every locked repair "
                    "field value verbatim; do not rephrase the scientific change, "
                    "hypothesis, family, or change scope."
                ),
                utility=2.0,
                alternatives=((family, 2.0),),
                locked_candidate_fields=locked_fields,
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
        assert best is not None
        stats = self._family_stats(journal)
        successes: dict[str, list[Node]] = {family: [] for family in PORTFOLIO_ORDER}
        for node in journal.good_nodes:
            family = self.family(node)
            if family in successes and node_metrics(node) is not None:
                successes[family].append(node)

        rich_reproduced = any(
            node.metric is not None
            and not node.metric.is_worst
            and float(node.metric.value) >= RICH_FM_MILESTONE_PRIMARY
            for node in successes["rich_fm"]
        )
        rich_refinement_warranted = stats["rich_fm"].parent_improvements > 0
        portfolio_attempts = sum(item.attempts for item in stats.values())
        force_rich = stats["rich_fm"].attempts == 0 or (
            not rich_reproduced
            and stats["rich_fm"].attempts == 1
            and portfolio_attempts == 1
            and rich_refinement_warranted
        )
        if not rich_reproduced and force_rich:
            rich_parent = (
                max(
                    successes["rich_fm"],
                    key=lambda node: (node_metrics(node) or {})["primary"],
                )
                if successes["rich_fm"]
                else best
            )
            return rich_parent, PortfolioAssignment(
                family="rich_fm",
                action="refine" if successes["rich_fm"] else "improve",
                reason=(
                    "First milestone: test one previously untested, parent-relative "
                    "rich-FM hypothesis grounded in durable evidence. Preserve the "
                    "organizer FM mechanisms outside the declared change scope. A second "
                    "rich-FM refinement is allowed only after a first attempt improves "
                    "both components."
                ),
                utility=2.0,
                alternatives=(("rich_fm", 2.0),),
                parent_node_id=rich_parent.id,
                feature_vector={"rich_milestone_missing": 1.0},
            )

        best_metrics = node_metrics(best) or {
            "GAUC": CHAMPION_VALID["GAUC"],
            "nDCG@5": CHAMPION_VALID["nDCG@5"],
            "primary": float(best.metric.value) if best.metric else 0.0,
        }
        frontier = pareto_frontier(journal.good_nodes) or [best]
        options: list[tuple[str, float, Node, dict[str, float]]] = []
        for family in PORTFOLIO_ORDER:
            item = stats[family]
            exploration = 0.75 if item.attempts == 0 else 0.0
            primary_evidence = (
                0.0
                if item.best_primary is None
                else max(
                    -0.6,
                    min(0.6, (item.best_primary - best_metrics["primary"]) * 250.0),
                )
            )
            weak_component = min(
                best_metrics["GAUC"] - CHAMPION_VALID["GAUC"],
                best_metrics["nDCG@5"] - CHAMPION_VALID["nDCG@5"],
            )
            weak_gain = 0.0
            if item.best_gauc is not None and item.best_ndcg5 is not None:
                weak_gain = max(
                    -0.5,
                    min(
                        0.5,
                        (
                            min(
                                item.best_gauc - CHAMPION_VALID["GAUC"],
                                item.best_ndcg5 - CHAMPION_VALID["nDCG@5"],
                            )
                            - weak_component
                        )
                        * 250.0,
                    ),
                )
            floor_preservation = 0.0
            if item.best_gauc is not None and item.best_ndcg5 is not None:
                floor_preservation = (
                    0.20
                    if (
                        item.best_gauc > CHAMPION_VALID["GAUC"]
                        and item.best_ndcg5 > CHAMPION_VALID["nDCG@5"]
                    )
                    else -0.05
                )
            average_runtime = item.total_exec_seconds / max(1, item.attempts)
            runtime_penalty = min(0.4, average_runtime / 900.0 * 0.4)
            failure_penalty = 0.20 * item.failures + 0.35 * item.timeouts
            parent_regression_penalty = 0.40 * item.parent_dominated_attempts
            repeat_penalty = 0.08 * item.attempts
            duplicate_penalty = 0.15 * max(
                0, item.attempts - item.unique_scientific_changes
            )
            public_tuning_penalty = 0.05 * item.missing_internal_validation
            api_cost_penalty = min(0.25, item.total_api_cost_usd * 0.10)
            diversity_bonus = (
                0.0
                if item.max_prediction_correlation is None
                else max(
                    -0.1, min(0.25, (1.0 - item.max_prediction_correlation) * 0.25)
                )
            )
            parent, compatibility_bonus = self._compatible_parent(
                family, frontier, best
            )
            vector = {
                "frozen_evidence_prior": FAMILY_EVIDENCE_PRIORS.get(family, 0.0),
                "exploration": exploration,
                "primary_evidence": primary_evidence,
                "weak_component_gain": weak_gain,
                "floor_preservation": floor_preservation,
                "diversity": diversity_bonus,
                "parent_compatibility": compatibility_bonus,
                "runtime_penalty": -runtime_penalty,
                "failure_penalty": -failure_penalty,
                "parent_dominated_penalty": -parent_regression_penalty,
                "repeat_penalty": -repeat_penalty,
                "near_duplicate_penalty": -duplicate_penalty,
                "public_tuning_penalty": -public_tuning_penalty,
                "api_cost_penalty": -api_cost_penalty,
            }
            utility = sum(vector.values())
            options.append((family, utility, parent, vector))
        options.sort(key=lambda item: (-item[1], PORTFOLIO_ORDER.index(item[0])))
        family, utility, parent, vector = options[0]
        return parent, PortfolioAssignment(
            family=family,
            action="improve" if stats[family].attempts == 0 else "refine",
            reason=(
                "The Pareto scheduler selected this family-parent pair using both metric "
                "components, floor preservation, exploration, prediction diversity, "
                "runtime/API cost, failure history, and lineage compatibility. Make one "
                "attributable change and preserve the selected parent's working structure."
            ),
            utility=utility,
            alternatives=tuple((item[0], item[1]) for item in options[:5]),
            parent_node_id=parent.id,
            feature_vector=vector,
        )
