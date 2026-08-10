"""Feature catalog: definitions, packs, and queries (Task 4).

``FeatureDefinition`` declares validated, immutable metadata for one stable
feature key. ``list_features`` filters the registered catalog with
deterministic ordering and globally unique keys. ``FeaturePack`` is the
smallest metadata protocol shared by the built-in ``acoustic`` and
``adult_neuro`` packs, registered in the static immutable ``PACKS`` mapping;
no entry-point discovery and no placeholder extraction behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from .result import UnknownPackError

CATALOG_VERSION = 1

PACK_LEVELS = frozenset({"recording", "utterance"})
KEY_PREFIXES = frozenset(
    {
        "audio_",
        "time_",
        "voice_",
        "spectral_",
        "lex_",
        "semantic_",
        "morph_",
        "disfluency_",
        "discourse_",
        "resp_",
        "artic_",
        "rhythm_",
        "task_",
    }
)

DOMAINS = frozenset(
    {
        "audio_quality",
        "timing",
        "respiration",
        "phonation",
        "prosody",
        "spectral",
        "articulation",
        "rhythm",
        "lexical",
        "psycholinguistic",
        "morphosyntactic",
        "disfluency",
        "semantic",
        "discourse",
        "task",
    }
)
LANGUAGE_SCOPES = frozenset(
    {"language_independent", "language_sensitive", "language_dependent", "language_specific"}
)
TASK_IDS = frozenset(
    {
        "connected_speech",
        "picture_description",
        "story_recall",
        "semantic_fluency",
        "phonemic_fluency",
        "reading",
        "sustained_vowel",
        "ddk",
    }
)
DISORDERS = frozenset(
    {
        "ad",
        "mci",
        "ppa",
        "ftd",
        "dlb",
        "pd",
        "pdd",
        "als",
        "mnd",
        "hd",
        "ms",
        "ataxia",
        "psp",
        "msa",
        "cbs",
    }
)
EVIDENCE_LEVELS = frozenset(
    {
        "systematic_review",
        "multi_study",
        "single_study",
        "standard_feature_set",
        "derived_companion",
    }
)


@runtime_checkable
class FeaturePack(Protocol):
    """Metadata contract shared by every built-in feature pack."""

    name: str
    version: int


@dataclass(frozen=True)
class _BuiltinPack:
    name: str
    version: int = 1


PACKS: MappingProxyType = MappingProxyType(
    {
        "acoustic": _BuiltinPack(name="acoustic", version=1),
        "adult_neuro": _BuiltinPack(name="adult_neuro", version=1),
        "motor_neuro": _BuiltinPack(name="motor_neuro", version=1),
        "standardized_acoustic": _BuiltinPack(name="standardized_acoustic", version=1),
    }
)


@dataclass(frozen=True)
class FeatureDefinition:
    """Immutable metadata for one stable feature key.

    ``key`` must start with a stable prefix; ``pack`` must be a registered
    pack name; ``level`` is ``recording`` or ``utterance``; ``prerequisites``
    is an immutable tuple of other feature keys.
    """

    key: str
    pack: str
    level: str
    unit: str
    population: str
    reference: str
    prerequisites: tuple[str, ...] = ()
    formula_version: int = 1
    domain: str = "audio_quality"
    language_scope: str = "language_independent"
    tasks: tuple[str, ...] = ()
    disorders: tuple[str, ...] = ()
    evidence_level: str = "derived_companion"

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("feature key must be a non-empty string")
        if not any(self.key.startswith(prefix) for prefix in KEY_PREFIXES):
            raise ValueError(
                f"feature key {self.key!r} must start with one of the stable prefixes {sorted(KEY_PREFIXES)}"
            )
        if self.pack not in PACKS:
            raise ValueError(f"feature pack {self.pack!r} must be one of {sorted(PACKS)}")
        if self.level not in PACK_LEVELS:
            raise ValueError(
                f"feature level must be one of {sorted(PACK_LEVELS)}, got {self.level!r}"
            )
        for name in ("unit", "population", "reference"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"feature {name} must be a non-empty string")
        if not isinstance(self.formula_version, int) or self.formula_version < 1:
            raise ValueError("feature formula_version must be a positive integer")
        object.__setattr__(self, "prerequisites", tuple(self.prerequisites))
        self._validate_enumerated_field("domain", DOMAINS)
        self._validate_enumerated_field("language_scope", LANGUAGE_SCOPES)
        self._validate_enumerated_field("evidence_level", EVIDENCE_LEVELS)
        for name, allowed in (("tasks", TASK_IDS), ("disorders", DISORDERS)):
            values = tuple(getattr(self, name))
            unknown = [value for value in values if value not in allowed]
            if unknown:
                raise ValueError(
                    f"feature {name} has unknown values {sorted(unknown)}; "
                    f"allowed: {sorted(allowed)}"
                )
            object.__setattr__(self, name, values)

    def _validate_enumerated_field(self, name: str, allowed: frozenset) -> None:
        value = getattr(self, name)
        if value not in allowed:
            raise ValueError(f"feature {name} must be one of {sorted(allowed)}, got {value!r}")


_FEATURES: list[FeatureDefinition] = []


def register_feature(definition: FeatureDefinition) -> None:
    """Register one feature definition, rejecting duplicate keys globally."""
    if not isinstance(definition, FeatureDefinition):
        raise ValueError(f"expected FeatureDefinition, got {type(definition).__name__}")
    if any(existing.key == definition.key for existing in _FEATURES):
        raise ValueError(f"duplicate feature key: {definition.key!r}")
    _FEATURES.append(definition)


def list_features(
    *,
    pack: str | None = None,
    level: str | None = None,
    domain: str | None = None,
    language_scope: str | None = None,
    task: str | None = None,
    disorder: str | None = None,
    evidence_level: str | None = None,
) -> tuple[FeatureDefinition, ...]:
    """Return registered definitions, filtered and sorted by key.

    Unknown filters are rejected: unknown packs raise
    :class:`UnknownPackError`; unknown levels and unknown evidence-metadata
    filter values raise :class:`ValueError`.
    """
    if pack is not None and pack not in PACKS:
        raise UnknownPackError(f"unknown pack filter {pack!r}; known packs: {sorted(PACKS)}")
    if level is not None and level not in PACK_LEVELS:
        raise ValueError(f"unknown level filter {level!r}; allowed levels: {sorted(PACK_LEVELS)}")
    filters = (
        ("domain", domain, DOMAINS),
        ("language_scope", language_scope, LANGUAGE_SCOPES),
        ("task", task, TASK_IDS),
        ("disorder", disorder, DISORDERS),
        ("evidence_level", evidence_level, EVIDENCE_LEVELS),
    )
    for name, value, allowed in filters:
        if value is not None and value not in allowed:
            raise ValueError(f"unknown {name} filter {value!r}; allowed values: {sorted(allowed)}")
    selected = (
        definition
        for definition in _FEATURES
        if (pack is None or definition.pack == pack)
        and (level is None or definition.level == level)
        and (domain is None or definition.domain == domain)
        and (language_scope is None or definition.language_scope == language_scope)
        and (task is None or task in definition.tasks)
        and (disorder is None or disorder in definition.disorders)
        and (evidence_level is None or definition.evidence_level == evidence_level)
    )
    return tuple(sorted(selected, key=lambda definition: definition.key))


__all__ = [
    "CATALOG_VERSION",
    "DISORDERS",
    "DOMAINS",
    "EVIDENCE_LEVELS",
    "FeatureDefinition",
    "FeaturePack",
    "KEY_PREFIXES",
    "LANGUAGE_SCOPES",
    "PACK_LEVELS",
    "PACKS",
    "TASK_IDS",
    "list_features",
    "register_feature",
]
