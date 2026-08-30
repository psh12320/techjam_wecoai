"""Campaign isolation, drift detection, and zero-intervention evidence."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


CANDIDATE_INPUT_ALLOWLIST = frozenset(
    {
        "baseline.py",
        "data.py",
        "evaluate.py",
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
        "manifest.json",
        "train.csv",
        "user_features_pure.csv",
        "valid.csv",
        "video_features_basic_pure.csv",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_tree(root: Path, *, names: Iterable[str] | None = None) -> str:
    """Hash file names, sizes, and contents in a stable order."""

    root = Path(root).resolve()
    selected = set(names) if names is not None else None
    files = [path for path in root.rglob("*") if path.is_file()]
    if selected is not None:
        files = [
            path for path in files if path.relative_to(root).as_posix() in selected
        ]
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def validate_candidate_input(input_dir: Path) -> dict[str, str]:
    """Fail closed unless the copied candidate input matches the public allowlist."""

    input_dir = Path(input_dir).resolve()
    entries = list(input_dir.iterdir())
    actual = {entry.name for entry in entries}
    unexpected = sorted(actual - CANDIDATE_INPUT_ALLOWLIST)
    missing = sorted(CANDIDATE_INPUT_ALLOWLIST - actual)
    non_files = sorted(entry.name for entry in entries if not entry.is_file())
    symlinks = sorted(entry.name for entry in entries if entry.is_symlink())
    if unexpected or missing or non_files or symlinks:
        raise RuntimeError(
            "Candidate input allowlist violation: "
            f"unexpected={unexpected}, missing={missing}, "
            f"non_files={non_files}, symlinks={symlinks}"
        )
    return {name: sha256_file(input_dir / name) for name in sorted(actual)}


def fingerprint_sources(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(value).resolve() for value in paths), key=str):
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


@dataclass(frozen=True)
class CampaignManifest:
    run_id: str
    campaign_mode: str
    prompt_sha256: str
    source_sha256: str
    input_sha256: str
    dependency_sha256: str
    evaluator_sha256: str
    created_at_unix: float = field(default_factory=time.time)

    @property
    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CampaignEventLedger:
    """Append-only evidence for campaign lifecycle and human intervention events."""

    def __init__(self, path: Path, run_id: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id

    def append(
        self,
        event_type: str,
        *,
        actor: str = "runner",
        details: dict[str, Any] | None = None,
    ) -> None:
        event = {
            "actor": actor,
            "created_at_unix": time.time(),
            "details": details or {},
            "event_type": event_type,
            "run_id": self.run_id,
        }
        descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(
                descriptor,
                (json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode(
                    "utf-8"
                ),
            )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid campaign event on line {line_number}: {exc}"
                    ) from exc
                if event.get("run_id") != self.run_id:
                    raise ValueError(
                        f"Campaign event line {line_number} has the wrong run_id"
                    )
                events.append(event)
        return events

    def clean_evidence(self, manifest_sha256: str) -> dict[str, Any]:
        events = self.read()
        starts = [event for event in events if event.get("event_type") == "run_started"]
        completes = [
            event for event in events if event.get("event_type") == "run_completed"
        ]
        human = [
            event
            for event in events
            if event.get("actor") == "human"
            or event.get("event_type") == "intervention"
        ]
        manifest_matches = bool(starts and completes) and all(
            event.get("details", {}).get("manifest_sha256") == manifest_sha256
            for event in (starts[-1], completes[-1])
        )
        valid = (
            len(starts) == 1 and len(completes) == 1 and not human and manifest_matches
        )
        return {
            "valid": valid,
            "run_started_events": len(starts),
            "run_completed_events": len(completes),
            "human_events": len(human),
            "manifest_matches": manifest_matches,
        }
