"""Run API-free, leakage-safe EDA before autonomous model research."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from challenge.techjam_recsys.data import (  # noqa: E402
    TRAIN_END,
    VALID_END,
    load_development_splits,
)
from challenge.techjam_recsys.eda import (
    analyze_splits,
    write_eda_artifacts,
)  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate deterministic KuaiRand EDA artifacts without API calls."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "kuairand-starter-kit" / "KuaiRand-Pure" / "data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "challenge" / "artifacts" / "eda",
    )
    parser.add_argument("--basename", default="eda_report")
    parser.add_argument("--max-prompt-chars", type=int, default=4000)
    args = parser.parse_args()

    splits = load_development_splits(args.data_dir)
    if int(splits.train["date"].max()) > TRAIN_END:
        raise RuntimeError("Training data crossed the official development boundary")
    if int(splits.valid["date"].max()) > VALID_END:
        raise RuntimeError("Hidden-test date entered EDA")
    report = analyze_splits(splits)
    paths = write_eda_artifacts(
        report,
        args.output_dir,
        basename=args.basename,
        max_prompt_chars=args.max_prompt_chars,
    )
    print(
        "TECHJAM_EDA="
        + json.dumps(
            {
                "report_sha256": report.stable_hash,
                "artifacts": {name: str(path) for name, path in paths.items()},
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
