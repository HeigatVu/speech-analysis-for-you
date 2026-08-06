"""Extraction result contracts and stable errors (Task 4).

``FeatureIssue`` is one structured issue; ``FeatureBundle`` is the frozen
result container with pandas ``recordings``/``utterances``/``issues`` tables
plus JSON-serializable provenance; ``ExtractionContext`` is the minimal
immutable, population-neutral input shared by feature packs.

This module is the single obvious place exposing the complete stable error
set: :data:`STABLE_ERROR_CODES` lists all ten codes and every class is
re-exported here, including the pre-existing ``InvalidDocumentError``,
``InvalidChatError``, ``InvalidAudioError``, and ``MissingInputError``
(reused from their original modules, never duplicated). New exception classes
carry a stable ``.code``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import pandas as pd

from .document import InvalidDocumentError
from .formats.chat import InvalidChatError
from .pipeline import InvalidAudioError
from .schema import ExtractionConfig, FeatureExtractionError, MissingInputError

STABLE_ERROR_CODES = frozenset(
    {
        "INVALID_DOCUMENT",
        "INVALID_CHAT",
        "INVALID_AUDIO",
        "UNSUPPORTED_AUDIO",
        "MISSING_INPUT",
        "TARGET_SPEAKER_REQUIRED",
        "MISSING_ANNOTATION",
        "UNKNOWN_PACK",
        "INVALID_CONFIG",
        "EXTRACTION_ERROR",
    }
)

ISSUE_SEVERITIES = frozenset({"warning", "error"})

_RECORDING_IDS = ("recording_id", "speaker_id")
_UTTERANCE_IDS = ("recording_id", "speaker_id", "utterance_id", "start_s", "end_s")
_ISSUE_COLUMNS = (
    "recording_id",
    "speaker_id",
    "utterance_id",
    "feature",
    "code",
    "severity",
    "message",
)


# ---------------------------------------------------------------------------
# Stable extraction errors (new classes; pre-existing ones are reused)
# ---------------------------------------------------------------------------
class UnsupportedAudioError(FeatureExtractionError):
    """Raised when an audio input is not supported standard PCM WAV."""

    code = "UNSUPPORTED_AUDIO"


class TargetSpeakerRequiredError(FeatureExtractionError):
    """Raised when multiple speakers exist but no target is given."""

    code = "TARGET_SPEAKER_REQUIRED"


class MissingAnnotationError(FeatureExtractionError):
    """Raised when a required annotation layer is absent."""

    code = "MISSING_ANNOTATION"


class UnknownPackError(FeatureExtractionError):
    """Raised when a feature pack name is unknown."""

    code = "UNKNOWN_PACK"


class InvalidConfigError(FeatureExtractionError):
    """Raised when extraction configuration is invalid."""

    code = "INVALID_CONFIG"


class ExtractionError(FeatureExtractionError):
    """Raised when a single feature extraction fails unexpectedly."""

    code = "EXTRACTION_ERROR"


# ---------------------------------------------------------------------------
# Structured issue
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureIssue:
    """One structured issue attached to a recording, speaker, or utterance.

    ``code`` is deliberately not restricted to :data:`STABLE_ERROR_CODES` so
    ``UNSUPPORTED_CHAT_TIER`` and later feature-specific warning codes remain
    representable. Severity is ``warning`` or ``error`` only.
    """

    recording_id: str
    speaker_id: str
    code: str
    severity: str
    message: str
    utterance_id: str | None = None
    feature: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("issue code must be a non-empty string")
        if self.severity not in ISSUE_SEVERITIES:
            raise ValueError(
                f"issue severity must be one of {sorted(ISSUE_SEVERITIES)}, got {self.severity!r}"
            )
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("issue message must be a non-empty string")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureBundle:
    """Frozen result of one recording extraction.

    Identifier columns are enforced exactly (recordings and utterances take
    the identifier prefix, issues the full column set), so empty results
    retain them. Feature columns may follow; unavailable values stay ``NaN``
    and must be accompanied by a :class:`FeatureIssue` (never coerced to
    zero). ``provenance`` must be JSON-serializable.
    """

    recordings: pd.DataFrame
    utterances: pd.DataFrame
    issues: pd.DataFrame
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name, table in (
            ("recordings", self.recordings),
            ("utterances", self.utterances),
            ("issues", self.issues),
        ):
            if not isinstance(table, pd.DataFrame):
                raise ValueError(
                    f"bundle {name} must be a pandas DataFrame, got {type(table).__name__}"
                )
        if list(self.recordings.columns[: len(_RECORDING_IDS)]) != list(_RECORDING_IDS):
            raise ValueError("recordings table must start with columns recording_id, speaker_id")
        if list(self.utterances.columns[: len(_UTTERANCE_IDS)]) != list(_UTTERANCE_IDS):
            raise ValueError(
                "utterances table must start with columns recording_id, speaker_id, "
                "utterance_id, start_s, end_s"
            )
        if list(self.issues.columns) != list(_ISSUE_COLUMNS):
            raise ValueError(
                "issues table must have exactly columns recording_id, speaker_id, "
                "utterance_id, feature, code, severity, message"
            )
        if not isinstance(self.provenance, Mapping):
            raise ValueError(
                f"bundle provenance must be a mapping, got {type(self.provenance).__name__}"
            )
        try:
            json.dumps(dict(self.provenance))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"bundle provenance must be JSON-serializable: {exc}") from exc


# ---------------------------------------------------------------------------
# Extraction context
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExtractionContext:
    """Minimal immutable, population-neutral inputs shared by feature packs.

    Identifies the recording, target speaker, and audio/document inputs with
    immutable configuration and provenance. Never carries diagnosis, labels,
    clinical cutoffs, task lexicons, age, or pediatric/adult attributes.
    """

    recording_id: str
    target_speaker: str | None = None
    audio_path: str | None = None
    document_path: str | None = None
    config: ExtractionConfig = field(default_factory=ExtractionConfig)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


__all__ = [
    "ExtractionContext",
    "ExtractionError",
    "FeatureBundle",
    "FeatureIssue",
    "InvalidAudioError",
    "InvalidChatError",
    "InvalidConfigError",
    "InvalidDocumentError",
    "ISSUE_SEVERITIES",
    "MissingAnnotationError",
    "MissingInputError",
    "STABLE_ERROR_CODES",
    "TargetSpeakerRequiredError",
    "UnknownPackError",
    "UnsupportedAudioError",
]
