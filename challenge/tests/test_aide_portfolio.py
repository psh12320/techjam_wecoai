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
    parse_candidate_spec,
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
    seed = Node(
        code="pass",
        plan="baseline",
        candidate_spec={"model_family": "official_fm_seed"},
        metric=MetricValue(0.601469, maximize=True),
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
        is_buggy=False,
    )
    journal.append(strong_rich)
    parent, assignment = scheduler.choose(journal)
    assert parent is strong_rich
    assert assignment.family == "dcn_v2"

    dcn = Node(
        code="pass",
        plan="dcn v2",
        parent=strong_rich,
        candidate_spec={
            "model_family": "dcn_v2",
            "assignment_family": "dcn_v2",
        },
        metric=MetricValue(0.6040, maximize=True),
        is_buggy=False,
    )
    journal.append(dcn)
    parent, assignment = scheduler.choose(journal)
    assert parent is dcn
    assert assignment.family == "history_residual"
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
        is_buggy=False,
    )
    journal.append(history)
    parent, assignment = scheduler.choose(journal)
    assert parent is history
    assert assignment.family == "duration_auxiliary"

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


def test_task_prompt_freezes_strong_rich_reproduction_mechanics() -> None:
    task = (Path(__file__).parents[1] / "task.md").read_text(encoding="utf-8")
    required = (
        "3000, 7000, 12000, 20000, 35000, 60000, 120000",
        "`torch.optim.Adam(lr=0.001, weight_decay=1e-6)`",
        "Do not clamp, sigmoid, or otherwise reparameterize them",
        "full public-validation primary each epoch",
        "0.45*rank(history) + 0.45*rank(dcn) + 0.10*rank(RAD-video)",
    )
    for value in required:
        assert value in task


def test_task_prompt_freezes_executed_dcn_history_and_bounded_rad_head() -> None:
    task = (Path(__file__).parents[1] / "task.md").read_text(encoding="utf-8")
    required = (
        "vector-weight cross layers",
        "multiplier starts at `0.015`",
        "cumulative prior-exposure count bin",
        "`8 -> 32 -> 1`",
        "detached already-computed representations",
        "one linear `72 -> 1` projection",
        "no second training phase",
        "no LightGBM",
        "checkpoint on the exact emitted blend",
        "Do not select on history-only logits",
        "Atomic v20 stabilization",
        "ReduceLROnPlateau",
        "min_lr=2.5e-4",
        "do not reset `bad_epochs`",
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
