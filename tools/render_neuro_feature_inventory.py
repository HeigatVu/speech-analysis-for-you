"""Render the live catalog and neurodegenerative evidence inventory.

The renderer is the only writer of the three generated artifacts. It refuses
to write when any live key is undocumented (:func:`documentation_for` raises),
when the exact live/pack/row/status counts drift, when an active evidence row
covers no key or a key appears in more than one active row, or when the
deferred candidate set changes. ``--check`` renders in memory and compares the
result byte-for-byte with the checked-in artifacts without writing.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from collections import Counter
from pathlib import Path

from speech_features import list_features
from speech_features.catalog import DISORDERS, DOMAINS, EVIDENCE_LEVELS, LANGUAGE_SCOPES, TASK_IDS
from speech_features.evidence import DEFERRED_EVIDENCE, EvidenceRecord, validate_evidence
from speech_features.feature_documentation import documentation_for

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = (
    ROOT
    / "docs/2026-08-10/neurodegenerative-speech-feature-expansion/1/neurodegenerative-speech-feature-inventory.csv"
)
DEFAULT_MARKDOWN = (
    ROOT
    / "docs/2026-08-10/neurodegenerative-speech-feature-expansion/1/neurodegenerative-speech-feature-inventory.md"
)
DEFAULT_CATALOG = (
    ROOT / "docs/2026-08-06/say-vietnamese-speech-library/1/feature-catalog-v1.md"
)

EXPECTED_LIVE_COUNT = 485
EXPECTED_PACK_COUNTS = {
    "acoustic": 167,
    "adult_neuro": 183,
    "motor_neuro": 47,
    "standardized_acoustic": 88,
}
EXPECTED_ROW_COUNT = 492
EXPECTED_STATUS_COUNTS = {"implemented": 397, "optional": 88, "deferred": 7}
EXPECTED_DEFERRED_CANDIDATES = frozenset(
    {
        "ASR-derived confidence and perplexity",
        "ComParE feature set",
        "calibrated absolute loudness without calibration",
        "features lacking a reproducible source formula",
        "learned embeddings",
        "proprietary measures",
        "semantic embedding coherence",
    }
)

INVENTORY_COLUMNS = (
    "candidate",
    "domain",
    "language_scope",
    "tasks",
    "disorders",
    "evidence_level",
    "status",
    "feature_keys",
    "reference",
    "reason",
)

CATALOG_COLUMNS = (
    "key",
    "pack",
    "level",
    "unit",
    "prerequisites",
    "population",
    "formula_version",
    "reference",
    "domain",
    "language_scope",
    "tasks",
    "disorders",
    "evidence_level",
    "formula",
    "missing_data",
)


def _implemented_records(definitions) -> tuple[EvidenceRecord, ...]:
    return tuple(
        EvidenceRecord(
            candidate=definition.key,
            domain=definition.domain,
            language_scope=definition.language_scope,
            tasks=definition.tasks,
            disorders=definition.disorders,
            source=definition.reference,
            evidence_level=definition.evidence_level,
            status="optional" if definition.pack == "standardized_acoustic" else "implemented",
            feature_keys=(definition.key,),
        )
        for definition in definitions
    )


def _enforce_exact_counts(definitions, records) -> None:
    live = len(definitions)
    if live != EXPECTED_LIVE_COUNT:
        raise ValueError(f"live catalog has {live} keys; expected exactly {EXPECTED_LIVE_COUNT}")
    packs = Counter(definition.pack for definition in definitions)
    if dict(packs) != EXPECTED_PACK_COUNTS:
        raise ValueError(f"live pack counts {dict(packs)}; expected {EXPECTED_PACK_COUNTS}")
    if len(records) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"inventory has {len(records)} rows; expected exactly {EXPECTED_ROW_COUNT}"
        )
    statuses = Counter(record.status for record in records)
    if dict(statuses) != EXPECTED_STATUS_COUNTS:
        raise ValueError(f"status counts {dict(statuses)}; expected {EXPECTED_STATUS_COUNTS}")
    deferred = {record.candidate for record in records if record.status == "deferred"}
    if deferred != EXPECTED_DEFERRED_CANDIDATES:
        raise ValueError(
            f"deferred candidates {sorted(deferred)}; expected {sorted(EXPECTED_DEFERRED_CANDIDATES)}"
        )


def _validated_records(definitions) -> tuple[EvidenceRecord, ...]:
    catalog_keys = {definition.key for definition in definitions}
    records = validate_evidence(
        _implemented_records(definitions) + DEFERRED_EVIDENCE,
        catalog_keys,
    )
    for record in records:
        if record.domain not in DOMAINS:
            raise ValueError(f"unknown evidence domain: {record.domain!r}")
        if record.language_scope not in LANGUAGE_SCOPES:
            raise ValueError(f"unknown evidence language scope: {record.language_scope!r}")
        if record.evidence_level not in EVIDENCE_LEVELS:
            raise ValueError(f"unknown evidence level: {record.evidence_level!r}")
        if not set(record.tasks) <= TASK_IDS:
            raise ValueError(f"unknown evidence tasks: {record.candidate}")
        if not set(record.disorders) <= DISORDERS:
            raise ValueError(f"unknown evidence disorders: {record.candidate}")
    covered = {key for record in records for key in record.feature_keys}
    if covered != catalog_keys:
        raise ValueError("active evidence rows must cover every live catalog key exactly")
    active_coverage = Counter(key for record in records for key in record.feature_keys)
    duplicates = {key for key, count in active_coverage.items() if count > 1}
    if duplicates:
        raise ValueError(f"live keys appear in more than one active row: {sorted(duplicates)}")
    _enforce_exact_counts(definitions, records)
    return tuple(
        sorted(records, key=lambda record: (record.domain, record.candidate, record.feature_keys))
    )


def _inventory_row(record: EvidenceRecord) -> dict[str, str]:
    return {
        "candidate": record.candidate,
        "domain": record.domain,
        "language_scope": record.language_scope,
        "tasks": "|".join(sorted(record.tasks)),
        "disorders": "|".join(sorted(record.disorders)),
        "evidence_level": record.evidence_level,
        "status": record.status,
        "feature_keys": "|".join(sorted(record.feature_keys)),
        "reference": record.source,
        "reason": record.reason,
    }


def _markdown_cell(value) -> str:
    text = str(value) if value not in (None, "") else "—"
    return text.replace("|", "\\|").replace("\n", " ")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _render_inventory_csv(records) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=INVENTORY_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(_inventory_row(record) for record in records)
    return stream.getvalue()


def _render_inventory_markdown(records) -> str:
    lines = [
        "# Neurodegenerative speech-feature evidence inventory",
        "",
        "Generated from the live `list_features()` catalog and validated evidence records.",
        "Implemented rows map to stable feature keys; deferred rows state why they are excluded.",
        "This is research evidence bookkeeping, not diagnostic guidance.",
        "",
        "| " + " | ".join(INVENTORY_COLUMNS) + " |",
        "|" + "|".join("---" for _ in INVENTORY_COLUMNS) + "|",
    ]
    for record in records:
        row = _inventory_row(record)
        lines.append(
            "| " + " | ".join(_markdown_cell(row[name]) for name in INVENTORY_COLUMNS) + " |"
        )
    return "\n".join(lines) + "\n"


def _catalog_row(definition) -> dict[str, str]:
    formula, missing_data = documentation_for(definition.key)
    return {
        "key": definition.key,
        "pack": definition.pack,
        "level": definition.level,
        "unit": definition.unit,
        "prerequisites": ", ".join(definition.prerequisites) or "—",
        "population": definition.population,
        "formula_version": str(definition.formula_version),
        "reference": definition.reference,
        "domain": definition.domain,
        "language_scope": definition.language_scope,
        "tasks": ", ".join(definition.tasks) or "—",
        "disorders": ", ".join(definition.disorders) or "—",
        "evidence_level": definition.evidence_level,
        "formula": formula,
        "missing_data": missing_data,
    }


def _render_catalog(definitions) -> str:
    lines = [
        "# SAY feature catalog v1",
        "",
        f"This generated catalog contains all {len(definitions)} live keys returned by `list_features()`",
        "for the `acoustic`, `adult_neuro`, `motor_neuro`, and `standardized_acoustic` packs.",
        "Run `uv run python tools/render_neuro_feature_inventory.py` after catalog metadata changes.",
        "See the [README](../README.md), [feature extraction](feature-extraction.md), and",
        "[neurodegenerative feature guide](neurodegenerative-feature-guide.md).",
        "",
        "> **Research-only.** These descriptive features are not diagnostic, screening, normative,",
        "> prognostic, or treatment measures. Vietnamese clinical validity must be established",
        "> independently for each population, task, annotation workflow, and recording condition.",
        "",
        "## Registered features",
        "",
        "| " + " | ".join(CATALOG_COLUMNS) + " |",
        "|" + "|".join("---" for _ in CATALOG_COLUMNS) + "|",
    ]
    for definition in definitions:
        row = _catalog_row(definition)
        lines.append(
            "| " + " | ".join(_markdown_cell(row[name]) for name in CATALOG_COLUMNS) + " |"
        )
    lines.extend(
        [
            "",
            "## Future child-pack extension",
            "",
            "`PACKS` is a static built-in mapping. No pediatric or child pack is implemented; a future",
            "child pack implements the two-field `FeaturePack` contract and must add reviewed definitions,",
            "an extractor, and validation without changing the",
            "population-neutral document/result contracts. There is no entry-point discovery.",
            "",
            "```python",
            "class ChildPack:",
            '    name = "child"',
            "    version = 1",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def render_texts(definitions, records) -> dict[str, str]:
    return {
        "csv": _render_inventory_csv(records),
        "markdown": _render_inventory_markdown(records),
        "catalog": _render_catalog(definitions),
    }


def render(
    *,
    csv_path: Path = DEFAULT_CSV,
    markdown_path: Path = DEFAULT_MARKDOWN,
    catalog_path: Path = DEFAULT_CATALOG,
) -> tuple[EvidenceRecord, ...]:
    definitions = list_features()
    records = _validated_records(definitions)
    texts = render_texts(definitions, records)
    _write(Path(csv_path), texts["csv"])
    _write(Path(markdown_path), texts["markdown"])
    _write(Path(catalog_path), texts["catalog"])
    return records


def check() -> list[str]:
    """Return the checked-in artifact names that differ from a fresh render."""
    definitions = list_features()
    records = _validated_records(definitions)
    texts = render_texts(definitions, records)
    stale = []
    for name, path in (
        ("csv", DEFAULT_CSV),
        ("markdown", DEFAULT_MARKDOWN),
        ("catalog", DEFAULT_CATALOG),
    ):
        if not path.is_file() or path.read_bytes() != texts[name].encode("utf-8"):
            stale.append(name)
    return stale


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="render the neuro feature inventory artifacts")
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare a fresh render with the checked-in artifacts; exit 1 when stale",
    )
    args = parser.parse_args(argv)
    if args.check:
        stale = check()
        if stale:
            print("stale artifacts: " + ", ".join(stale), file=sys.stderr)
            return 1
        print("all generated artifacts are up to date")
        return 0
    render()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
