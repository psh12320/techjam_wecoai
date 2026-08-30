"""Run autonomous paid development campaigns until the portfolio genuinely plateaus."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from challenge.build_experiment_memory import build_memory  # noqa: E402
from challenge.techjam_recsys.aide_portfolio import (  # noqa: E402
    CHAMPION_VALID,
    PORTFOLIO_ORDER,
)
from challenge.techjam_recsys.literature import freeze_manifest  # noqa: E402
from challenge.techjam_recsys.protocol import ExperimentLedger  # noqa: E402


RUNNER = ROOT / "challenge" / "run_aide_research.py"
RUN_ROOT = ROOT / "challenge" / "runs" / "aide"
MEMORY = ROOT / "challenge" / "research_memory" / "experiment_memory.json"
FROZEN_LITERATURE = (
    ROOT / "challenge" / "research_memory" / "literature_manifest.json"
)
DEVELOPMENT_LITERATURE = (
    ROOT
    / "challenge"
    / "research_memory"
    / "literature_manifest_development.json"
)
CHECKPOINT_ROOT = ROOT / "challenge" / "checkpoints"
STATE_PATH = ROOT / "challenge" / "private" / "autonomous_campaign_state.json"
DEV_RUN_RE = re.compile(r"^techjam-aide-v26-dev-(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-campaigns", type=int, default=0)
    parser.add_argument("--plateau-campaigns", type=int, default=3)
    parser.add_argument("--campaign-improvement", type=float, default=0.0002)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--input-usd-per-million", type=float, default=4.0)
    parser.add_argument("--output-usd-per-million", type=float, default=20.0)
    parser.add_argument("--steps", type=int, default=49)
    return parser.parse_args()


def _summary_for_run(run_id: str) -> tuple[Path, dict]:
    matches: list[tuple[Path, dict]] = []
    for path in RUN_ROOT.glob("aide_logs/*/resource_summary.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("run_id") == run_id:
            matches.append((path, value))
    if len(matches) != 1:
        raise RuntimeError(f"expected one resource summary for {run_id}, found {len(matches)}")
    return matches[0]


def _all_summaries() -> list[tuple[Path, dict]]:
    summaries: list[tuple[Path, dict]] = []
    for path in RUN_ROOT.glob("aide_logs/*/resource_summary.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            summaries.append((path, value))
    return summaries


def _next_campaign_index() -> int:
    indices = []
    for _, summary in _all_summaries():
        match = DEV_RUN_RE.match(str(summary.get("run_id") or ""))
        if match:
            indices.append(int(match.group(1)))
    return max(indices, default=0) + 1


def _metric_primary(summary: dict) -> float:
    metrics = summary.get("best_aide_generated_metrics")
    if not isinstance(metrics, dict):
        return float("-inf")
    try:
        return float(metrics.get("primary", float("-inf")))
    except (TypeError, ValueError):
        return float("-inf")


def _initial_progress(improvement: float) -> tuple[float, float, int]:
    """Resume plateau accounting across invocations without reusing a run id."""

    summaries = _all_summaries()
    historical = [
        _metric_primary(summary)
        for _, summary in summaries
        if not DEV_RUN_RE.match(str(summary.get("run_id") or ""))
    ]
    best_primary = max(historical, default=float("-inf"))
    plateau = 0
    development = []
    for _, summary in summaries:
        match = DEV_RUN_RE.match(str(summary.get("run_id") or ""))
        if match:
            development.append((int(match.group(1)), summary))
    for _, summary in sorted(development):
        primary = _metric_primary(summary)
        if primary - best_primary >= improvement:
            best_primary = primary
            plateau = 0
        else:
            best_primary = max(best_primary, primary)
            plateau += 1

    clean_primaries = [
        _metric_primary(summary)
        for _, summary in summaries
        if summary.get("clean_campaign_accepted") is True
    ]
    last_clean = max(clean_primaries, default=float("-inf"))
    return best_primary, last_clean, plateau


def _write_state(**values: object) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(STATE_PATH.suffix + ".tmp")
    temporary.write_text(
        json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(STATE_PATH)


def _all_ledgers() -> list[Path]:
    return sorted(RUN_ROOT.glob("aide_logs/*/iterations.jsonl"))


def _refresh_memory() -> None:
    ledgers = _all_ledgers()
    if not ledgers:
        raise RuntimeError("no AIDE ledgers are available")
    value = build_memory(ledgers, max_entries=32)
    MEMORY.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _attempted_families() -> set[str]:
    attempted: set[str] = set()
    for path in _all_ledgers():
        for record in ExperimentLedger(path).read(validate_chain=True):
            if record.model_family in PORTFOLIO_ORDER:
                attempted.add(record.model_family)
    return attempted


def _run(run_id: str, *, mode: str, literature: Path, online: bool, args: argparse.Namespace) -> dict:
    command = [
        sys.executable,
        str(RUNNER),
        "--execute",
        "--campaign-mode",
        mode,
        "--run-id",
        run_id,
        "--model",
        args.model,
        "--reasoning-effort",
        "xhigh",
        "--repair-reasoning-effort",
        "high",
        "--steps",
        str(args.steps),
        "--input-usd-per-million",
        str(args.input_usd_per_million),
        "--output-usd-per-million",
        str(args.output_usd_per_million),
        "--literature-manifest",
        str(literature),
        "--development-literature-manifest",
        str(DEVELOPMENT_LITERATURE),
    ]
    if online:
        command.append("--online-literature")
    completed = subprocess.run(command, cwd=ROOT, check=False)
    summary_path, summary = _summary_for_run(run_id)
    summary["process_exit_code"] = completed.returncode
    summary["resource_summary_path"] = str(summary_path)
    return summary


def _passes_champion(metrics: object) -> bool:
    return isinstance(metrics, dict) and all(
        float(metrics.get(key, float("-inf"))) > CHAMPION_VALID[key]
        for key in ("GAUC", "nDCG@5", "primary")
    )


def _checkpoint(run_id: str, summary_path: Path, primary: float) -> Path:
    destination = CHECKPOINT_ROOT / f"{run_id}-primary-{primary:.9f}"
    destination.mkdir(parents=True, exist_ok=False)
    run_dir = summary_path.parent
    for name in (
        "resource_summary.json",
        "acceptance_status.json",
        "campaign_manifest.json",
        "iterations.jsonl",
    ):
        source = run_dir / name
        if source.exists():
            shutil.copy2(source, destination / name)
    shutil.copy2(MEMORY, destination / "experiment_memory.json")
    freeze_manifest(DEVELOPMENT_LITERATURE, destination / "literature_manifest.json")
    for source in (
        ROOT / "challenge" / "task.md",
        ROOT / "challenge" / "prompts" / "hard_constraints.md",
        ROOT / "challenge" / "prompts" / "research_menu.md",
        ROOT / "challenge" / "requirements-agent.txt",
        ROOT / "challenge" / "techjam_recsys" / "aide_portfolio.py",
        ROOT / "challenge" / "techjam_recsys" / "campaign_safety.py",
    ):
        shutil.copy2(source, destination / source.name)
    return destination


def main() -> int:
    args = parse_args()
    if args.max_campaigns < 0:
        raise ValueError("max-campaigns must be zero (unlimited) or positive")
    best_primary, last_clean_checkpoint, plateau = _initial_progress(
        args.campaign_improvement
    )
    campaign_index = _next_campaign_index()
    launched = 0
    while args.max_campaigns == 0 or launched < args.max_campaigns:
        launched += 1
        run_id = f"techjam-aide-v26-dev-{campaign_index:03d}"
        summary = _run(
            run_id,
            mode="development",
            literature=FROZEN_LITERATURE,
            online=True,
            args=args,
        )
        _refresh_memory()
        metrics = summary.get("best_aide_generated_metrics")
        primary = (
            float(metrics.get("primary", float("-inf")))
            if isinstance(metrics, dict)
            else float("-inf")
        )
        if primary - best_primary >= args.campaign_improvement:
            best_primary = primary
            plateau = 0
        else:
            best_primary = max(best_primary, primary)
            plateau += 1

        if _passes_champion(metrics) and primary - last_clean_checkpoint >= args.campaign_improvement:
            summary_path, _ = _summary_for_run(run_id)
            checkpoint = _checkpoint(run_id, summary_path, primary)
            clean_id = f"techjam-aide-v26-clean-{campaign_index:03d}"
            clean_summary = _run(
                clean_id,
                mode="clean",
                literature=checkpoint / "literature_manifest.json",
                online=False,
                args=args,
            )
            if clean_summary.get("clean_campaign_accepted"):
                last_clean_checkpoint = primary
                (checkpoint / "clean_result.json").write_text(
                    json.dumps(clean_summary, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

        exhausted = set(PORTFOLIO_ORDER).issubset(_attempted_families())
        _write_state(
            last_completed_run_id=run_id,
            next_campaign_index=campaign_index + 1,
            best_primary=best_primary,
            last_clean_checkpoint=last_clean_checkpoint,
            consecutive_plateau_campaigns=plateau,
            attempted_families=sorted(_attempted_families()),
            high_value_families_exhausted=exhausted,
        )
        if plateau >= args.plateau_campaigns and exhausted:
            print(
                "AUTONOMOUS_PLATEAU="
                + json.dumps(
                    {
                        "campaigns_this_invocation": launched,
                        "last_campaign_index": campaign_index,
                        "best_primary": best_primary,
                        "attempted_families": sorted(_attempted_families()),
                    },
                    sort_keys=True,
                )
            )
            return 0
        campaign_index += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
