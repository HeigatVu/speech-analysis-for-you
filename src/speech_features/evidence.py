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


__all__ = ["ALLOWED_STATUSES", "ACTIVE_STATUSES", "EvidenceRecord", "validate_evidence"]
