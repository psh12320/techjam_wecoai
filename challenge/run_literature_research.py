"""Run bounded literature lookup for the research planner (never candidate code)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from challenge.techjam_recsys.literature import (  # noqa: E402
    LiteratureBounds,
    LiteratureProvider,
    run_literature_research,
)


DEFAULT_MANIFEST = ROOT / "challenge" / "research_memory" / "literature_manifest.json"


def _read_eda(path: Path | None) -> Any:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, (dict, list, str)):
        raise ValueError("EDA JSON must contain an object, list, or string")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--mode", choices=("frozen", "online"), default="frozen")
    parser.add_argument("--eda-json", type=Path)
    parser.add_argument("--weakness", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-queries", type=int, default=6)
    parser.add_argument("--max-results-per-query", type=int, default=5)
    parser.add_argument("--max-total-notes", type=int, default=24)
    parser.add_argument("--max-provider-bytes", type=int, default=256_000)
    parser.add_argument("--max-note-bytes", type=int, default=16_000)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    provider: LiteratureProvider | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "online" and provider is None:
        parser.error(
            "online mode requires a programmatically injected bounded provider adapter"
        )
    bounds = LiteratureBounds(
        max_queries=args.max_queries,
        max_results_per_query=args.max_results_per_query,
        max_total_notes=args.max_total_notes,
        max_provider_bytes=args.max_provider_bytes,
        max_note_bytes=args.max_note_bytes,
    )
    result = run_literature_research(
        manifest_path=args.manifest,
        eda_findings=_read_eda(args.eda_json),
        weaknesses=args.weakness,
        mode=args.mode,
        provider=provider,
        bounds=bounds,
    )
    rendered = json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
