from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from aide.journal import Journal, Node
from aide.utils.metric import MetricValue, WorstMetricValue
from challenge.run_aide_research import KuaiRandAgent
from challenge.techjam_recsys.aide_portfolio import (
    PortfolioScheduler,
    normalize_family,
    pareto_frontier,
    parse_candidate_spec,
    validate_candidate_spec,
    validate_candidate_source,
)
from challenge.techjam_recsys.data import _read_log_file


def test_candidate_spec_parses_nested_json_and_normalizes_family() -> None:
    plan = """
<candidate_spec>
{"model_family":"field_gated_fm","features":["user_id","video_id"],
 "losses":{"bce":1.0},"hyperparameters":{"k":16},
 "estimated_runtime_seconds":300,"risks":["overfit"],
 "expected_metric_effects":{"GAUC":"up","nDCG@5":"up"}}
</candidate_spec>
Use a field-gated FM while preserving the evaluator.
"""
    spec = parse_candidate_spec(plan, fallback_family="rich_fm")

    assert spec["model_family"] == "rich_fm"
    assert spec["losses"] == {"bce": 1.0}
    assert spec["hyperparameters"] == {"k": 16}
    assert spec["structured"] is True


def test_missing_candidate_spec_uses_assigned_family() -> None:
    spec = parse_candidate_spec(
        "Try a bounded improvement without a tag.", fallback_family="din_lite"
    )
    assert spec["model_family"] == "din_lite"
    assert spec["structured"] is False
    assert "missing" in spec["parse_error"]
    assert spec["card_complete"] is False


def test_complete_candidate_card_validates_parent_and_evidence_ids() -> None:
    parent_code_hash = "a" * 64
    payload = {
        "parent_node_id": "parent-1",
        "parent_code_sha256": parent_code_hash,
        "model_family": "history_residual",
        "eda_observation_ids": ["EDA-1"],
        "literature_citation_ids": ["LIT-1"],
        "scientific_change": "add one strictly-past count residual",
        "hypothesis": "the count improves cold-user nDCG without lowering GAUC",
        "features": ["strict_prior_count"],
        "losses": {"bce": 1.0},
        "hyperparameters": {"residual_scale": 0.01},
        "target_metric": "nDCG@5 while preserving GAUC",
        "expected_metric_effects": {"GAUC": "flat", "nDCG@5": "up", "primary": "up"},
        "estimated_runtime_seconds": 600,
        "estimated_memory_mb": 2500,
        "risks": ["history construction may be slow"],
        "abort_criteria": ["chronology assertion fails"],
        "falsification_condition": "full-fidelity nDCG does not improve",
        "fidelity": "full",
        "internal_validation": {"split": "last_3_train_days"},
    }
    spec = parse_candidate_spec(
        f"<candidate_spec>{json.dumps(payload)}</candidate_spec>",
        fallback_family="history_residual",
    )
    assert spec["card_complete"] is True
    assert (
        validate_candidate_spec(
            spec,
            expected_parent_node_id="parent-1",
            expected_parent_code_sha256=parent_code_hash,
            allowed_eda_observation_ids={"EDA-1"},
            allowed_literature_citation_ids={"LIT-1"},
        )
        == []
    )


def test_candidate_card_rejects_wrong_parent_unknown_citation_and_screen_overflow() -> (
    None
):
    spec = parse_candidate_spec("no tag", fallback_family="rich_fm")
    spec.update(
        {
            "parent_node_id": "wrong",
            "parent_code_sha256": "wrong",
            "eda_observation_ids": ["UNKNOWN"],
            "literature_citation_ids": ["UNKNOWN"],
            "estimated_runtime_seconds": 901,
            "estimated_memory_mb": 3073,
            "fidelity": "invalid",
        }
    )
    errors = validate_candidate_spec(
        spec,
        expected_parent_node_id="expected",
        expected_parent_code_sha256="a" * 64,
        allowed_eda_observation_ids={"EDA-1"},
        allowed_literature_citation_ids={"LIT-1"},
    )
    assert any("parent_node_id" in error for error in errors)
    assert any("unknown literature" in error for error in errors)
    assert any("900-second" in error for error in errors)
    assert any("3-GB" in error for error in errors)


def test_escaped_candidate_spec_closing_tag_is_parsed() -> None:
    plan = (
        '<candidate_spec>{"model_family":"rich_fm","features":["user_id"],'
        '"losses":{"bce":1},"hyperparameters":{"k":16},'
        '"estimated_runtime_seconds":420,"risks":[],'
        '"expected_metric_effects":{"primary":"up"}}<\\/candidate_spec>'
    )
    spec = parse_candidate_spec(plan, fallback_family="rich_fm")
    assert spec["structured"] is True
    assert spec["estimated_runtime_seconds"] == 420


def test_composite_family_labels_normalize_to_portfolio_families() -> None:
    assert normalize_family("rich_field_gated_fm") == "rich_fm"
    assert normalize_family("rich_field_gated_fm_fwfm") == "fwfm"
    assert normalize_family("candidate_history_residual") == "history_residual"


def test_source_policy_rejects_champion_reuse_and_workspace_escape() -> None:
    rejected = validate_candidate_source(
        "from challenge.run_enriched_fm import main\n"
        "path = 'C:/tmp/champion-v3.json'\n"
        "open('working/validation_predictions.csv', 'w')\n"
    )
    assert any("forbidden import" in value for value in rejected)
    assert any("champion" in value for value in rejected)
    assert any("absolute" in value for value in rejected)


def test_source_policy_accepts_self_contained_workspace_program() -> None:
    code = (
        "from pathlib import Path\n"
        "Path('working/validation_predictions.csv').write_text('row_id,score\\n')\n"
    )
    assert validate_candidate_source(code) == []


def test_source_policy_accepts_composed_prediction_path() -> None:
    code = """
import os
WORK_DIR = './working'
OUTPUT_PATH = os.path.join(WORK_DIR, 'validation_predictions.csv')
open(OUTPUT_PATH, 'w').close()
"""
    assert validate_candidate_source(code) == []


def test_source_policy_does_not_confuse_progress_range_with_parent_path() -> None:
    code = (
        "start, stop = 1, 2\n"
        "print(f'users {start}..{stop}')\n"
        "open('working/validation_predictions.csv', 'w').close()\n"
    )
    assert validate_candidate_source(code) == []


def test_source_policy_rejects_dynamic_import_process_and_composed_parent_path() -> (
    None
):
    rejected = validate_candidate_source(
        "import os\n"
        "module = __import__('re' + 'quests')\n"
        "path = Path('..') / '..' / 'challenge'\n"
        "os.system('echo bad')\n"
        "open('working/validation_predictions.csv', 'w')\n"
    )
    assert any("dynamic" in value for value in rejected)
    assert any("parent-relative" in value for value in rejected)


def test_portfolio_scheduler_forces_rich_milestone_then_coverage() -> None:
    def analysis(gauc: float, ndcg: float) -> str:
        return json.dumps(
            {"metrics": {"GAUC": gauc, "nDCG@5": ndcg, "primary": (gauc + ndcg) / 2}}
        )

    seed = Node(
        code="pass",
        plan="baseline",
        candidate_spec={"model_family": "official_fm_seed"},
        metric=MetricValue(0.601469, maximize=True),
        analysis=analysis(0.667133, 0.535805),
        is_buggy=False,
    )
    journal = Journal(nodes=[seed], metric_maximize=True)
    scheduler = PortfolioScheduler(max_debug_depth=3)

    parent, assignment = scheduler.choose(journal)
    assert parent is seed
    assert assignment.family == "rich_fm"

    rich = Node(
        code="pass",
        plan="rich",
        parent=seed,
        candidate_spec={"model_family": "rich_field_gated_fm"},
        metric=MetricValue(0.6026, maximize=True),
        analysis=analysis(0.6687, 0.5365),
        is_buggy=False,
    )
    journal.append(rich)
    parent, assignment = scheduler.choose(journal)
    assert parent is rich
    assert assignment.family == "rich_fm"

    strong_rich = Node(
        code="pass",
        plan="strong rich",
        parent=rich,
        candidate_spec={"model_family": "rich_field_gated_fm"},
        metric=MetricValue(0.6036, maximize=True),
        analysis=analysis(0.6702, 0.5370),
        is_buggy=False,
    )
    journal.append(strong_rich)
    parent, assignment = scheduler.choose(journal)
    assert parent is strong_rich
    assert assignment.family == "history_residual"

    dcn = Node(
        code="pass",
        plan="dcn v2",
        parent=strong_rich,
        candidate_spec={
            "model_family": "dcn_v2",
            "assignment_family": "dcn_v2",
        },
        metric=MetricValue(0.6040, maximize=True),
        analysis=analysis(0.6705, 0.5375),
        is_buggy=False,
    )
    journal.append(dcn)
    parent, assignment = scheduler.choose(journal)
    assert assignment.family == "history_residual"
    assert parent is dcn
    assert assignment.alternatives

    history = Node(
        code="pass",
        plan="history residual",
        parent=dcn,
        candidate_spec={
            "model_family": "history_residual",
            "assignment_family": "history_residual",
        },
        metric=MetricValue(0.6039, maximize=True),
        analysis=analysis(0.6704, 0.5374),
        is_buggy=False,
    )
    journal.append(history)
    parent, assignment = scheduler.choose(journal)
    assert assignment.family

    failed = Node(
        code="raise RuntimeError",
        plan="RAD-video rank blend",
        parent=history,
        candidate_spec={
            "model_family": "duration_auxiliary",
            "assignment_family": "duration_auxiliary",
        },
        metric=WorstMetricValue(maximize=True),
        is_buggy=True,
    )
    journal.append(failed)
    parent, assignment = scheduler.choose(journal)
    assert parent is failed
    assert assignment.action == "debug"
    assert assignment.family == "duration_auxiliary"


def test_pareto_frontier_keeps_ndcg_specialist_and_removes_dominated_node() -> None:
    def node(node_id: str, gauc: float, ndcg: float) -> Node:
        return Node(
            id=node_id,
            code="pass",
            plan=node_id,
            candidate_spec={"model_family": "history_residual"},
            metric=MetricValue((gauc + ndcg) / 2, maximize=True),
            analysis=json.dumps(
                {
                    "metrics": {
                        "GAUC": gauc,
                        "nDCG@5": ndcg,
                        "primary": (gauc + ndcg) / 2,
                    }
                }
            ),
            is_buggy=False,
        )

    gauc_specialist = node("gauc", 0.6720, 0.5379)
    ndcg_specialist = node("ndcg", 0.6712, 0.5386)
    dominated = node("dominated", 0.6710, 0.5378)
    frontier = pareto_frontier([gauc_specialist, ndcg_specialist, dominated])
    assert {candidate.id for candidate in frontier} == {"gauc", "ndcg"}


def test_scheduler_penalizes_timeout_duplicate_and_moves_to_evidence_backed_family() -> (
    None
):
    def reviewed(gauc: float, ndcg: float) -> str:
        return json.dumps(
            {"metrics": {"GAUC": gauc, "nDCG@5": ndcg, "primary": (gauc + ndcg) / 2}}
        )

    seed = Node(
        code="pass",
        plan="seed",
        candidate_spec={"model_family": "official_fm_seed"},
        metric=MetricValue(0.601469, maximize=True),
        analysis=reviewed(0.667133, 0.535805),
        is_buggy=False,
    )
    rich = Node(
        code="pass",
        plan="rich",
        parent=seed,
        candidate_spec={
            "model_family": "rich_fm",
            "scientific_change": "rich milestone",
            "internal_validation": {"split": "last_3_train_days"},
        },
        metric=MetricValue(0.6038, maximize=True),
        analysis=reviewed(0.6704, 0.5372),
        is_buggy=False,
    )
    timed_out = Node(
        code="pass",
        plan="history",
        parent=rich,
        candidate_spec={
            "model_family": "history_residual",
            "assignment_family": "history_residual",
            "scientific_change": "same slow history",
        },
        metric=WorstMetricValue(maximize=True),
        is_buggy=True,
        exc_type="TimeoutError",
        exec_time=900,
    )
    # Make the failed implementation non-leaf so the scheduler compares
    # scientific branches instead of correctly prioritizing an immediate repair.
    Node(code="pass", plan="abandoned repair", parent=timed_out, is_buggy=True)
    journal = Journal(nodes=[seed, rich, timed_out], metric_maximize=True)
    _, assignment = PortfolioScheduler(max_debug_depth=3).choose(journal)
    assert assignment.family == "duration_auxiliary"
    assert "failure_penalty" in assignment.feature_vector


def test_task_prompt_is_thin_benchmark_contract_not_forced_solution() -> None:
    task = (Path(__file__).parents[1] / "task.md").read_text(encoding="utf-8")
    required = (
        "Experiment memory",
        "Optional research menu",
        "EDA evidence",
        "Literature evidence",
        "last-three-training-days",
        "one bounded scientific improvement",
    )
    for value in required:
        assert value in task
    assert "0.45*rank(history)" not in task
    assert len(task) < 10_000


def test_task_prompt_requires_single_seed_component_gate_and_clean_evidence() -> None:
    task = (Path(__file__).parents[1] / "task.md").read_text(encoding="utf-8")
    required = (
        "seed-0 execution strictly exceeds all three",
        "GAUC `0.6710518008586268`",
        "nDCG@5 `0.5380142516919405`",
        "primary `0.6045330262752837`",
        "zero manual interventions",
        "exactly once at seed 0",
    )
    for value in required:
        assert value in task


def test_improvement_prompt_receives_schema_environment_and_assignment() -> None:
    cfg = SimpleNamespace(
        agent=SimpleNamespace(
            search=SimpleNamespace(max_debug_depth=3),
            data_preview=True,
            expose_prediction=False,
            k_fold_validation=1,
        ),
        exec=SimpleNamespace(timeout=900),
    )
    parent = Node(
        code="pass",
        plan="baseline",
        candidate_spec={"model_family": "official_fm_seed"},
        metric=MetricValue(0.601469, maximize=True),
        is_buggy=False,
    )
    journal = Journal(nodes=[parent], metric_maximize=True)
    agent = KuaiRandAgent(task_desc="SCHEMA CONTRACT", cfg=cfg, journal=journal)
    agent.data_preview = "REAL_DATA_OVERVIEW"
    captured = {}

    def fake_query(prompt, retries=1):
        captured.update(prompt)
        spec = {
            "model_family": "rich_fm",
            "features": ["user_id"],
            "losses": {"bce": 1.0},
            "hyperparameters": {"k": 16},
            "estimated_runtime_seconds": 300,
            "risks": [],
            "expected_metric_effects": {"GAUC": "up", "nDCG@5": "up"},
        }
        return (
            f"<candidate_spec>{json.dumps(spec)}</candidate_spec>\nHypothesis.",
            "from pathlib import Path\n"
            "Path('working/validation_predictions.csv').write_text('row_id,score\\n')",
        )

    agent.plan_and_code_query = fake_query  # type: ignore[method-assign]
    node = agent._improve(parent)

    assert captured["Task description"] == "SCHEMA CONTRACT"
    assert captured["Data Overview"] == "REAL_DATA_OVERVIEW"
    assert "Verified runtime" in captured["Instructions"]
    assert captured["Instructions"]["Portfolio assignment"]["Assigned family"]
    assert node.candidate_spec["model_family"] == "rich_fm"


def test_date_prepass_skips_hidden_outcome_before_dtype_parsing(tmp_path) -> None:
    path = tmp_path / "later.csv"
    path.write_text(
        "date,long_view\n" "20220422,1\n" "20220501,HIDDEN_NOT_AN_INT\n" "20220428,0\n",
        encoding="utf-8",
    )

    public = _read_log_file(path, ["date", "long_view"], max_date=20220428)
    assert public.to_dict("records") == [
        {"date": 20220422, "long_view": 1},
        {"date": 20220428, "long_view": 0},
    ]
