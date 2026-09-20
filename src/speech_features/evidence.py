"""Evidence records and inventory validation for the feature catalog (Task 1).

``EvidenceRecord`` is the immutable inventory row consumed by later tasks:
one reviewed candidate feature with its domain, language scope, tasks,
disorders, evidence source, implementation status, and the stable SAY keys it
maps to. ``validate_evidence`` enforces the inventory contract: every row has
a source, active statuses carry catalog keys, and deferred rows carry a
reason.
"""

from __future__ import annotations

from dataclasses import dataclass

ACTIVE_STATUSES = frozenset({"existing", "implemented", "optional"})
ALLOWED_STATUSES = ACTIVE_STATUSES | {"deferred"}


@dataclass(frozen=True)
class EvidenceRecord:
    """One reviewed evidence-inventory row for a candidate feature."""

    candidate: str
    domain: str
    language_scope: str
    tasks: tuple[str, ...]
    disorders: tuple[str, ...]
    source: str
    evidence_level: str
    status: str
    feature_keys: tuple[str, ...] = ()
    reason: str = ""


DEFERRED_EVIDENCE = (
    EvidenceRecord(
        candidate="learned embeddings",
        domain="spectral",
        language_scope="language_sensitive",
        tasks=("connected_speech",),
        disorders=(),
        source="docs/2026-08-10/neurodegenerative-speech-feature-expansion/1/RESEARCH-2026-08-10.md",
        evidence_level="single_study",
        status="deferred",
        reason="requires a trained model and a separately versioned validation contract",
    ),
    EvidenceRecord(
        candidate="ASR-derived confidence and perplexity",
        domain="lexical",
        language_scope="language_specific",
        tasks=("connected_speech",),
        disorders=(),
        source="docs/2026-08-10/neurodegenerative-speech-feature-expansion/1/RESEARCH-2026-08-10.md",
        evidence_level="single_study",
        status="deferred",
        reason="requires an ASR system and Vietnamese held-out transcription validation",
    ),
    EvidenceRecord(
        candidate="semantic embedding coherence",
        domain="semantic",
        language_scope="language_specific",
        tasks=("connected_speech",),
        disorders=(),
        source="docs/2026-08-10/neurodegenerative-speech-feature-expansion/1/RESEARCH-2026-08-10.md",
        evidence_level="single_study",
        status="deferred",
        reason="requires a versioned semantic model and Vietnamese validation",
    ),
    EvidenceRecord(
        candidate="proprietary measures",
        domain="phonation",
        language_scope="language_independent",
        tasks=("connected_speech",),
        disorders=(),
        source="docs/2026-08-10/neurodegenerative-speech-feature-expansion/1/RESEARCH-2026-08-10.md",
        evidence_level="single_study",
        status="deferred",
        reason="no open reproducible implementation is available",
    ),
    EvidenceRecord(
        candidate="ComParE feature set",
        domain="spectral",
        language_scope="language_sensitive",
        tasks=("connected_speech",),
        disorders=(),
        source="docs/2026-08-10/neurodegenerative-speech-feature-expansion/1/RESEARCH-2026-08-10.md",
        evidence_level="standard_feature_set",
        status="deferred",
        reason="the large generic set is outside the reviewed minimal feature scope",
    ),
    EvidenceRecord(
        candidate="calibrated absolute loudness without calibration",
        domain="audio_quality",
        language_scope="language_independent",
        tasks=("connected_speech",),
        disorders=(),
        source="docs/2026-08-10/neurodegenerative-speech-feature-expansion/1/RESEARCH-2026-08-10.md",
        evidence_level="derived_companion",
        status="deferred",
        reason="absolute loudness is not reproducible without recording-chain calibration",
    ),
    EvidenceRecord(
        candidate="features lacking a reproducible source formula",
        domain="audio_quality",
        language_scope="language_independent",
        tasks=(),
        disorders=(),
        source="docs/2026-08-10/neurodegenerative-speech-feature-expansion/1/RESEARCH-2026-08-10.md",
        evidence_level="derived_companion",
        status="deferred",
        reason="a reviewed reproducible formula is required before implementation",
    ),
)


def validate_evidence(records, catalog_keys) -> tuple[EvidenceRecord, ...]:
    """Check the evidence inventory contract and return it as a tuple.

    Raises :class:`ValueError` on unknown statuses, rows without a source,
    active rows without valid catalog keys, and deferred rows without a
    reason.
    """
    checked = []
    for record in records:
        if record.status not in ALLOWED_STATUSES:
            raise ValueError(f"unknown evidence status: {record.status!r}")
        if not record.source:
            raise ValueError(f"missing evidence source: {record.candidate}")
        if record.status in ACTIVE_STATUSES:
            if not record.feature_keys or not set(record.feature_keys) <= set(catalog_keys):
                raise ValueError(f"invalid catalog keys: {record.candidate}")
        if record.status == "deferred" and not record.reason:
            raise ValueError(f"deferred feature lacks reason: {record.candidate}")
        checked.append(record)
    return tuple(checked)


__all__ = [
    "ALLOWED_STATUSES",
    "ACTIVE_STATUSES",
    "DEFERRED_EVIDENCE",
    "EvidenceRecord",
    "validate_evidence",
]
