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
    {"audio_", "time_", "voice_", "spectral_", "lex_", "morph_", "disfluency_", "discourse_"}
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


_FEATURES: list[FeatureDefinition] = []


def register_feature(definition: FeatureDefinition) -> None:
    """Register one feature definition, rejecting duplicate keys globally."""
    if not isinstance(definition, FeatureDefinition):
        raise ValueError(f"expected FeatureDefinition, got {type(definition).__name__}")
    if any(existing.key == definition.key for existing in _FEATURES):
        raise ValueError(f"duplicate feature key: {definition.key!r}")
    _FEATURES.append(definition)


def list_features(
    *, pack: str | None = None, level: str | None = None
) -> tuple[FeatureDefinition, ...]:
    """Return registered definitions, filtered and sorted by key.

    Unknown filters are rejected: unknown packs raise
    :class:`UnknownPackError`; unknown levels raise :class:`ValueError`.
    """
    if pack is not None and pack not in PACKS:
        raise UnknownPackError(f"unknown pack filter {pack!r}; known packs: {sorted(PACKS)}")
    if level is not None and level not in PACK_LEVELS:
        raise ValueError(f"unknown level filter {level!r}; allowed levels: {sorted(PACK_LEVELS)}")
    selected = (
        definition
        for definition in _FEATURES
        if (pack is None or definition.pack == pack)
        and (level is None or definition.level == level)
    )
    return tuple(sorted(selected, key=lambda definition: definition.key))


__all__ = [
    "CATALOG_VERSION",
    "FeatureDefinition",
    "FeaturePack",
    "KEY_PREFIXES",
    "PACK_LEVELS",
    "PACKS",
    "list_features",
    "register_feature",
]
