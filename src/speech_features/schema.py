"""Contract types and validation for the Vietnamese AD speech feature library.

Task 1: immutable extraction config/result types, manifest and transcript
validation, Unicode normalisation, SHA-256 provenance hashes, and structured
extraction errors. Only NumPy/SciPy/stdlib primitives are used here.
"""

from __future__ import annotations

import hashlib
import math
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

# Task names the library can extract features for. The plan's evaluation
# requires "five of six" of these tasks per participant.
KNOWN_TASKS = frozenset(
    {
        "picture_desc_1",
        "picture_desc_2",
        "immediate_recall",
        "delayed_recall",
        "phonemic_fluency",
        "semantic_fluency",
    }
)

TOKEN_KINDS = frozenset({"word", "filler", "fragment", "noise"})
SPEAKERS = frozenset({"participant", "examiner"})
DIAGNOSES = frozenset({"AD", "HC"})

MANIFEST_REQUIRED_FIELDS = frozenset(
    {
        "participant_id",
        "task",
        "recording_id",
        "audio_id",
        "transcript_id",
        "audio_path",
        "transcript_path",
        "task_spec_path",
        "diagnosis",
        "age",
        "sex",
        "education_years",
    }
)


# ---------------------------------------------------------------------------
# Structured errors
# ---------------------------------------------------------------------------
class FeatureExtractionError(Exception):
    """Base class for all errors raised by this library."""


class UnknownTaskError(FeatureExtractionError):
    """Raised when a manifest or spec references an unknown task."""


class InvalidManifestError(FeatureExtractionError):
    """Raised when a manifest mapping is malformed."""


class InvalidTranscriptError(FeatureExtractionError):
    """Raised when a transcript JSON is malformed."""


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def nfc(text: str) -> str:
    """Normalise text to Unicode NFC form."""
    return unicodedata.normalize("NFC", text)


# ---------------------------------------------------------------------------
# Immutable config / result types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExtractionConfig:
    """Fixed extraction defaults. 25 ms frames / 10 ms hop at 16 kHz."""

    sample_rate: int = 16000
    frame_size: int = 400  # 25 ms at 16 kHz
    hop_size: int = 160  # 10 ms at 16 kHz
    pitch_min_hz: float = 70.0
    pitch_max_hz: float = 400.0
    pitch_autocorr_threshold: float = 0.30
    pause_threshold_s: float = 0.20
    long_pause_threshold_s: float = 2.0


@dataclass(frozen=True)
class FeatureResult:
    """One extracted recording. `features` holds numeric features."""

    recording_id: str
    participant_id: str
    task: str
    features: dict = field(default_factory=dict)
    quality_flags: list = field(default_factory=list)
    input_hashes: dict = field(default_factory=dict)
    config: ExtractionConfig = field(default_factory=ExtractionConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))
        object.__setattr__(self, "quality_flags", tuple(self.quality_flags))
        object.__setattr__(self, "input_hashes", MappingProxyType(dict(self.input_hashes)))


# ---------------------------------------------------------------------------
# Provenance hashes
# ---------------------------------------------------------------------------
def sha256_file(path) -> str:
    """Return the hexdigest SHA-256 of a file, streaming to bound memory."""
    p = Path(path)
    if not p.is_file():
        raise FeatureExtractionError(f"input file not found: {p}")
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_hashes(manifest: dict) -> dict:
    """Record SHA-256 provenance for each unique asset path in a manifest.

    Returns a mapping from (asset kind, path) to its hexdigest. Different rows
    may reference different task-spec files, so each unique path is hashed once
    regardless of how many rows share it.
    """
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise InvalidManifestError("manifest must have version == 1")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise InvalidManifestError("manifest must contain a non-empty 'rows' list")
    kinds = {
        "audio_path": "audio",
        "transcript_path": "transcript",
        "task_spec_path": "task_spec",
    }
    result = {}
    for path_key, name in kinds.items():
        for row in rows:
            path = row[path_key]
            result[(name, path)] = sha256_file(path)
    return result


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ManifestRow:
    """A single (participant_id, task) manifest row."""

    participant_id: str
    task: str
    recording_id: str
    audio_id: str
    transcript_id: str
    audio_path: str
    transcript_path: str
    task_spec_path: str
    diagnosis: str
    age: int
    sex: str
    education_years: int

    # pathlib.Path helpers keep the row frozen while exposing usable paths
    @property
    def audio(self) -> Path:
        return Path(self.audio_path)

    @property
    def transcript(self) -> Path:
        return Path(self.transcript_path)

    @property
    def task_spec(self) -> Path:
        return Path(self.task_spec_path)


def validate_manifest(manifest: dict) -> list[ManifestRow]:
    """Validate a manifest mapping and return its rows.

    Rejects unknown version, malformed rows, unknown tasks/diagnoses, and
    duplicate (participant_id, task) pairs.
    """
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise InvalidManifestError("manifest must have version == 1")
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise InvalidManifestError("manifest must contain a 'rows' list")

    seen = set()
    parsed: list[ManifestRow] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise InvalidManifestError(f"row {i} is not an object")
        missing = MANIFEST_REQUIRED_FIELDS - row.keys()
        if missing:
            raise InvalidManifestError(f"row {i} missing fields: {sorted(missing)}")
        if row["task"] not in KNOWN_TASKS:
            raise UnknownTaskError(f"row {i} unknown task: {row['task']!r}")
        if row["diagnosis"] not in DIAGNOSES:
            raise InvalidManifestError(
                f"row {i} diagnosis must be AD or HC, got {row['diagnosis']!r}"
            )
        if not isinstance(row["age"], int) or row["age"] < 0:
            raise InvalidManifestError(f"row {i} age must be a non-negative integer")
        if not isinstance(row["education_years"], int) or row["education_years"] < 0:
            raise InvalidManifestError(f"row {i} education_years must be a non-negative integer")
        participant_id = nfc(row["participant_id"])
        key = (participant_id, row["task"])
        if key in seen:
            raise InvalidManifestError(f"duplicate (participant_id, task): {key}")
        seen.add(key)
        parsed.append(
            ManifestRow(
                participant_id=participant_id,
                task=row["task"],
                recording_id=nfc(row["recording_id"]),
                audio_id=nfc(row["audio_id"]),
                transcript_id=nfc(row["transcript_id"]),
                audio_path=row["audio_path"],
                transcript_path=row["transcript_path"],
                task_spec_path=row["task_spec_path"],
                diagnosis=row["diagnosis"],
                age=row["age"],
                sex=nfc(row["sex"]),
                education_years=row["education_years"],
            )
        )
    return parsed


# ---------------------------------------------------------------------------
# Transcript validation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Token:
    kind: str
    text: str


@dataclass(frozen=True)
class Utterance:
    speaker: str
    start_s: float
    end_s: float
    tokens: list = field(default_factory=list)


def validate_transcript(transcript: dict) -> list[Utterance]:
    """Validate transcript JSON v1 and return ordered utterances.

    Enforces finite, ordered timestamps and known speakers and token kinds.
    """
    if not isinstance(transcript, dict) or transcript.get("version") != 1:
        raise InvalidTranscriptError("transcript must have version == 1")
    utterances = transcript.get("utterances")
    if not isinstance(utterances, list):
        raise InvalidTranscriptError("transcript must contain an 'utterances' list")

    parsed: list[Utterance] = []
    for i, utt in enumerate(utterances):
        if not isinstance(utt, dict):
            raise InvalidTranscriptError(f"utterance {i} is not an object")
        speaker = utt.get("speaker")
        if speaker not in SPEAKERS:
            raise InvalidTranscriptError(f"utterance {i} speaker must be participant or examiner")
        start = utt.get("start_s")
        end = utt.get("end_s")
        try:
            start = float(start)
            end = float(end)
        except (TypeError, ValueError):
            raise InvalidTranscriptError(f"utterance {i} timestamps must be numeric")
        if not (math.isfinite(start) and math.isfinite(end)):
            raise InvalidTranscriptError(f"utterance {i} timestamps must be finite")
        if end < start:
            raise InvalidTranscriptError(f"utterance {i} end before start")

        raw_tokens = utt.get("tokens")
        if not isinstance(raw_tokens, list):
            raise InvalidTranscriptError(f"utterance {i} must contain a 'tokens' list")
        tokens = []
        for j, tok in enumerate(raw_tokens):
            if not isinstance(tok, dict):
                raise InvalidTranscriptError(f"utterance {i} token {j} is not an object")
            kind = tok.get("kind")
            if kind not in TOKEN_KINDS:
                raise InvalidTranscriptError(f"utterance {i} token {j} unknown kind {kind!r}")
            tokens.append(Token(kind=kind, text=nfc(tok.get("text", ""))))
        parsed.append(Utterance(speaker=speaker, start_s=start, end_s=end, tokens=tokens))
    return parsed
