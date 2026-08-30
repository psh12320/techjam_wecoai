"""Bounded, cited literature memory for the recommender research planner.

Only the planning process may supply a provider. Candidate programs never receive
the provider, network access, or credentials. A frozen manifest is read-only and
therefore suitable for a clean campaign in an offline final environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit


SCHEMA_VERSION = 1
NOTE_FIELDS = (
    "title",
    "authors",
    "year",
    "url",
    "technique",
    "data",
    "effect",
    "cost",
    "risk",
    "applicability",
)
_CITATION_RE = re.compile(r"^lit-[0-9a-f]{16}$")
_IDENTIFIER_RE = re.compile(r"^(?:arxiv|doi):[^\s]+$", flags=re.IGNORECASE)


class LiteratureValidationError(ValueError):
    """Raised when untrusted literature data violates the frozen schema."""


class LiteratureProvider(Protocol):
    """Injectable online-development adapter; implementations may use a search API."""

    def search(self, query: str, *, limit: int) -> Iterable[Mapping[str, Any]]: ...


@dataclass(frozen=True)
class LiteratureBounds:
    """Hard bounds around planning-layer search and cached payloads."""

    max_queries: int = 6
    max_results_per_query: int = 5
    max_total_notes: int = 24
    max_provider_bytes: int = 256_000
    max_note_bytes: int = 16_000

    def __post_init__(self) -> None:
        limits = {
            "max_queries": (self.max_queries, 1, 20),
            "max_results_per_query": (self.max_results_per_query, 1, 20),
            "max_total_notes": (self.max_total_notes, 1, 100),
            "max_provider_bytes": (self.max_provider_bytes, 1, 2_000_000),
            "max_note_bytes": (self.max_note_bytes, 1, 64_000),
        }
        for name, (value, minimum, maximum) in limits.items():
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}")
        if self.max_note_bytes > self.max_provider_bytes:
            raise ValueError("max_note_bytes cannot exceed max_provider_bytes")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def query_sha256(query: str) -> str:
    return _sha256({"query": " ".join(query.split())})


def _clean_text(value: Any, field: str, *, maximum: int = 4_000) -> str:
    if not isinstance(value, str):
        raise LiteratureValidationError(f"{field} must be a string")
    if any(ord(character) < 32 and character not in "\t\n\r" for character in value):
        raise LiteratureValidationError(f"{field} contains a control character")
    cleaned = " ".join(value.split())
    if not cleaned:
        raise LiteratureValidationError(f"{field} cannot be empty")
    if len(cleaned) > maximum:
        raise LiteratureValidationError(f"{field} exceeds {maximum} characters")
    return cleaned


def _clean_reference(value: Any) -> str:
    reference = _clean_text(value, "url", maximum=2_000)
    if _IDENTIFIER_RE.fullmatch(reference):
        return (
            reference.lower() if reference.lower().startswith("arxiv:") else reference
        )
    parts = urlsplit(reference)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise LiteratureValidationError(
            "url must be an http(s) URL or an arxiv:/doi: identifier"
        )
    if parts.username or parts.password:
        raise LiteratureValidationError("url must not contain credentials")
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, "")
    )


@dataclass(frozen=True)
class LiteratureNote:
    title: str
    authors: tuple[str, ...]
    year: int
    url: str
    technique: str
    data: str
    effect: str
    cost: str
    risk: str
    applicability: str
    citation_id: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LiteratureNote":
        if not isinstance(value, Mapping):
            raise LiteratureValidationError("literature note must be an object")
        permitted = set(NOTE_FIELDS) | {"citation_id"}
        missing = sorted(set(NOTE_FIELDS) - set(value))
        unexpected = sorted(set(value) - permitted)
        if missing or unexpected:
            raise LiteratureValidationError(
                f"literature note schema mismatch: missing={missing}, unexpected={unexpected}"
            )

        raw_authors = value["authors"]
        if (
            not isinstance(raw_authors, Sequence)
            or isinstance(raw_authors, (str, bytes))
            or not raw_authors
            or len(raw_authors) > 32
        ):
            raise LiteratureValidationError(
                "authors must be a non-empty list of at most 32 names"
            )
        authors = tuple(
            _clean_text(author, "authors", maximum=300) for author in raw_authors
        )
        year = value["year"]
        if (
            isinstance(year, bool)
            or not isinstance(year, int)
            or not 1800 <= year <= 2100
        ):
            raise LiteratureValidationError(
                "year must be an integer between 1800 and 2100"
            )

        normalized = {
            "title": _clean_text(value["title"], "title", maximum=1_000),
            "authors": authors,
            "year": year,
            "url": _clean_reference(value["url"]),
            "technique": _clean_text(value["technique"], "technique"),
            "data": _clean_text(value["data"], "data"),
            "effect": _clean_text(value["effect"], "effect"),
            "cost": _clean_text(value["cost"], "cost"),
            "risk": _clean_text(value["risk"], "risk"),
            "applicability": _clean_text(value["applicability"], "applicability"),
        }
        identity = {
            "authors": list(authors),
            "title": normalized["title"],
            "url": normalized["url"],
            "year": year,
        }
        citation_id = f"lit-{_sha256(identity)[:16]}"
        supplied = value.get("citation_id")
        if supplied is not None and supplied != citation_id:
            raise LiteratureValidationError(
                "citation_id does not match the stable source identity"
            )
        return cls(**normalized, citation_id=citation_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "url": self.url,
            "technique": self.technique,
            "data": self.data,
            "effect": self.effect,
            "cost": self.cost,
            "risk": self.risk,
            "applicability": self.applicability,
        }


def _query_entry(query: str, citation_ids: Sequence[str]) -> dict[str, Any]:
    normalized = " ".join(query.split())
    identifiers = list(dict.fromkeys(citation_ids))
    return {
        "query": normalized,
        "query_sha256": query_sha256(normalized),
        "citation_ids": identifiers,
        "response_sha256": _sha256(identifiers),
    }


def new_manifest(*, mode: str) -> dict[str, Any]:
    if mode not in {"development", "frozen"}:
        raise ValueError("manifest mode must be development or frozen")
    return seal_manifest(
        {
            "schema_version": SCHEMA_VERSION,
            "mode": mode,
            "notes": [],
            "query_cache": {},
        }
    )


def seal_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and hash a manifest; hashes never depend on dictionary order."""

    mode = manifest.get("mode")
    if mode not in {"development", "frozen"}:
        raise LiteratureValidationError("manifest mode must be development or frozen")
    if manifest.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise LiteratureValidationError("unsupported literature manifest schema")

    raw_notes = manifest.get("notes", [])
    if not isinstance(raw_notes, list):
        raise LiteratureValidationError("manifest notes must be a list")
    notes = [LiteratureNote.from_mapping(note).as_dict() for note in raw_notes]
    notes.sort(key=lambda note: note["citation_id"])
    if len({note["citation_id"] for note in notes}) != len(notes):
        raise LiteratureValidationError("manifest contains duplicate citation IDs")
    known_ids = {note["citation_id"] for note in notes}

    raw_cache = manifest.get("query_cache", {})
    if not isinstance(raw_cache, Mapping):
        raise LiteratureValidationError("manifest query_cache must be an object")
    cache: dict[str, dict[str, Any]] = {}
    for key, raw_entry in sorted(raw_cache.items()):
        if not isinstance(raw_entry, Mapping):
            raise LiteratureValidationError("query cache entry must be an object")
        required = {"query", "query_sha256", "citation_ids", "response_sha256"}
        if set(raw_entry) != required:
            raise LiteratureValidationError("query cache entry has the wrong schema")
        query = _clean_text(raw_entry["query"], "query", maximum=2_000)
        citation_ids = raw_entry["citation_ids"]
        if not isinstance(citation_ids, list) or any(
            not isinstance(item, str) for item in citation_ids
        ):
            raise LiteratureValidationError(
                "cached citation_ids must be a list of strings"
            )
        entry = _query_entry(query, citation_ids)
        if key != entry["query_sha256"] or raw_entry["query_sha256"] != key:
            raise LiteratureValidationError("query cache key/hash mismatch")
        if raw_entry["response_sha256"] != entry["response_sha256"]:
            raise LiteratureValidationError("query cache response hash mismatch")
        unknown = sorted(set(citation_ids) - known_ids)
        if unknown:
            raise LiteratureValidationError(
                f"query cache references unknown citations: {unknown}"
            )
        cache[key] = entry

    sealed = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "notes": notes,
        "query_cache": cache,
        "notes_sha256": _sha256(notes),
        "cache_sha256": _sha256(cache),
    }
    sealed["manifest_sha256"] = _sha256(sealed)
    return sealed


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "schema_version",
        "mode",
        "notes",
        "query_cache",
        "notes_sha256",
        "cache_sha256",
        "manifest_sha256",
    }
    if set(manifest) != allowed:
        raise LiteratureValidationError(
            "literature manifest has the wrong top-level schema"
        )
    sealed = seal_manifest(manifest)
    for name in ("notes_sha256", "cache_sha256", "manifest_sha256"):
        if manifest.get(name) != sealed[name]:
            raise LiteratureValidationError(f"literature manifest {name} mismatch")
    return sealed


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LiteratureValidationError(
            f"invalid literature manifest JSON: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise LiteratureValidationError("literature manifest must be an object")
    return validate_manifest(value)


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    sealed = seal_manifest(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(sealed, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return sealed


_QUERY_RULES = (
    (
        ("ndcg", "top-five", "top 5", "rerank", "ranking error"),
        "LambdaLoss RankNet top-k recommender reranking original paper",
    ),
    (
        (
            "history",
            "sequential",
            "repeat exposure",
            "candidate-aware",
            "candidate aware",
        ),
        "Deep Interest Network candidate-aware user history recommendation original paper",
    ),
    (
        ("duration", "watch ratio", "play time", "watch time", "censored"),
        "duration debiasing watch-time recommendation D2Q CWM original paper",
    ),
    (
        ("cold-start", "cold start", "unseen item", "unseen author", "unseen tag"),
        "content feature cold-start implicit-feedback recommendation original paper",
    ),
    (
        ("interaction", "factorization", "cross feature", "fm", "dcn"),
        "field-weighted factorization machine DCN-V2 feature interaction original paper",
    ),
    (
        ("diversity", "correlation", "ensemble", "complementary"),
        "out-of-fold diverse recommender ensemble ranking original paper",
    ),
    (
        ("gauc", "group auc", "user auc", "bce"),
        "group-aware implicit-feedback recommendation ranking objective original paper",
    ),
    (
        ("segment", "warm user", "short video", "mixture", "gating"),
        "segment gated mixture of experts recommender ranking original paper",
    ),
)


def build_literature_queries(
    eda_findings: Mapping[str, Any] | Sequence[Any] | str | None,
    weaknesses: Mapping[str, Any] | Sequence[Any] | str | None,
    *,
    max_queries: int = 6,
) -> list[str]:
    """Build a stable, compact query list from EDA evidence and weak metrics."""

    if not 1 <= max_queries <= 20:
        raise ValueError("max_queries must be between 1 and 20")
    evidence = _canonical_json(
        {"eda": eda_findings or {}, "weaknesses": weaknesses or []}
    ).lower()
    queries = [
        query
        for triggers, query in _QUERY_RULES
        if any(trigger in evidence for trigger in triggers)
    ]
    if not queries:
        queries = [
            "implicit-feedback recommender GAUC nDCG established method original paper"
        ]
    return list(dict.fromkeys(queries))[:max_queries]


@dataclass(frozen=True)
class ProviderBatch:
    notes: tuple[LiteratureNote, ...]
    consumed_bytes: int
    rejected_results: int
    byte_limit_reached: bool


class BoundedProviderAdapter:
    """Enforce result and byte limits around an injected, untrusted provider."""

    def __init__(self, provider: LiteratureProvider, bounds: LiteratureBounds):
        self.provider = provider
        self.bounds = bounds

    def search(self, query: str, *, remaining_bytes: int) -> ProviderBatch:
        notes: list[LiteratureNote] = []
        consumed = 0
        rejected = 0
        exhausted = False
        results = self.provider.search(query, limit=self.bounds.max_results_per_query)
        for index, raw in enumerate(results):
            if index >= self.bounds.max_results_per_query:
                break
            try:
                encoded = _canonical_json(raw).encode("utf-8")
            except (TypeError, ValueError):
                rejected += 1
                continue
            size = len(encoded)
            if size > self.bounds.max_note_bytes:
                rejected += 1
                continue
            if consumed + size > remaining_bytes:
                exhausted = True
                break
            consumed += size
            try:
                notes.append(LiteratureNote.from_mapping(raw))
            except LiteratureValidationError:
                rejected += 1
        return ProviderBatch(tuple(notes), consumed, rejected, exhausted)


@dataclass(frozen=True)
class LiteratureResearchResult:
    mode: str
    queries: tuple[str, ...]
    citation_ids: tuple[str, ...]
    cache_hits: int
    cache_misses: int
    provider_bytes: int
    rejected_results: int
    byte_limit_reached: bool
    manifest_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "queries": list(self.queries),
            "citation_ids": list(self.citation_ids),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "provider_bytes": self.provider_bytes,
            "rejected_results": self.rejected_results,
            "byte_limit_reached": self.byte_limit_reached,
            "manifest_sha256": self.manifest_sha256,
        }


def run_literature_research(
    *,
    manifest_path: Path,
    eda_findings: Mapping[str, Any] | Sequence[Any] | str | None,
    weaknesses: Mapping[str, Any] | Sequence[Any] | str | None,
    mode: str,
    provider: LiteratureProvider | None = None,
    bounds: LiteratureBounds | None = None,
) -> LiteratureResearchResult:
    """Resolve bounded notes online or strictly from an immutable frozen cache."""

    if mode not in {"online", "frozen"}:
        raise ValueError("research mode must be online or frozen")
    limits = bounds or LiteratureBounds()
    path = Path(manifest_path)
    if path.exists():
        manifest = load_manifest(path)
    elif mode == "online":
        manifest = new_manifest(mode="development")
    else:
        raise FileNotFoundError(f"frozen literature manifest not found: {path}")

    if mode == "online" and manifest["mode"] == "frozen":
        raise RuntimeError("refusing to mutate a frozen literature manifest")
    if mode == "frozen" and manifest["mode"] != "frozen":
        raise RuntimeError("clean/frozen mode requires a frozen literature manifest")
    if mode == "online" and provider is None:
        raise ValueError("online literature research requires an injected provider")

    queries = build_literature_queries(
        eda_findings, weaknesses, max_queries=limits.max_queries
    )
    note_by_id = {note["citation_id"]: note for note in manifest["notes"]}
    selected: list[str] = []
    hits = 0
    misses = 0
    provider_bytes = 0
    rejected = 0
    byte_limit_reached = False
    adapter = (
        BoundedProviderAdapter(provider, limits)
        if mode == "online" and provider is not None
        else None
    )

    for query in queries:
        key = query_sha256(query)
        cached = manifest["query_cache"].get(key)
        if cached is not None:
            hits += 1
            selected.extend(cached["citation_ids"])
            continue
        misses += 1
        if mode == "frozen":
            continue
        if byte_limit_reached or len(note_by_id) >= limits.max_total_notes:
            continue

        assert adapter is not None
        batch = adapter.search(
            query, remaining_bytes=limits.max_provider_bytes - provider_bytes
        )
        provider_bytes += batch.consumed_bytes
        rejected += batch.rejected_results
        byte_limit_reached = byte_limit_reached or batch.byte_limit_reached
        query_ids: list[str] = []
        for note in batch.notes:
            if note.citation_id not in note_by_id:
                if len(note_by_id) >= limits.max_total_notes:
                    break
                note_by_id[note.citation_id] = note.as_dict()
            elif note_by_id[note.citation_id] != note.as_dict():
                rejected += 1
                continue
            query_ids.append(note.citation_id)
        manifest["query_cache"][key] = _query_entry(query, query_ids)
        selected.extend(query_ids)

    if mode == "online":
        manifest["notes"] = list(note_by_id.values())
        manifest = write_manifest(path, manifest)

    return LiteratureResearchResult(
        mode=mode,
        queries=tuple(queries),
        citation_ids=tuple(dict.fromkeys(selected)),
        cache_hits=hits,
        cache_misses=misses,
        provider_bytes=provider_bytes,
        rejected_results=rejected,
        byte_limit_reached=byte_limit_reached,
        manifest_sha256=manifest["manifest_sha256"],
    )


def freeze_manifest(source: Path, destination: Path) -> dict[str, Any]:
    manifest = load_manifest(source)
    manifest["mode"] = "frozen"
    return write_manifest(destination, manifest)


def resolve_candidate_citations(
    candidate_card: Mapping[str, Any],
    manifest: Mapping[str, Any] | Path,
    *,
    maximum: int = 12,
) -> list[dict[str, Any]]:
    """Resolve stable IDs in a candidate card and reject invented citations."""

    if not isinstance(candidate_card, Mapping):
        raise LiteratureValidationError("candidate card must be an object")
    citations = candidate_card.get(
        "literature_citations", candidate_card.get("citations", [])
    )
    if not isinstance(citations, list) or any(
        not isinstance(citation, str) for citation in citations
    ):
        raise LiteratureValidationError("candidate citations must be a list of IDs")
    if len(citations) > maximum:
        raise LiteratureValidationError(f"candidate cites more than {maximum} sources")
    if any(not _CITATION_RE.fullmatch(citation) for citation in citations):
        raise LiteratureValidationError("candidate contains a malformed citation ID")

    checked = (
        load_manifest(manifest)
        if isinstance(manifest, Path)
        else validate_manifest(manifest)
    )
    note_by_id = {note["citation_id"]: note for note in checked["notes"]}
    unknown = [citation for citation in citations if citation not in note_by_id]
    if unknown:
        raise LiteratureValidationError(f"candidate cites unknown sources: {unknown}")
    return [note_by_id[citation] for citation in dict.fromkeys(citations)]
