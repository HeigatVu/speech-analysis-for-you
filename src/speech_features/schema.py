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

TASK_SPEC_TASKS = KNOWN_TASKS | {"connected_speech", "ddk", "sustained_vowel"}

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
    """Base class for all errors raised by this library.

    ``code`` is a stable machine-readable string suitable for switching on in
    production code; ``message`` carries the human-readable detail.
    """

    code = "FEATURE_EXTRACTION_ERROR"


class UnknownTaskError(FeatureExtractionError):
    """Raised when a manifest or spec references an unknown task."""

    code = "UNKNOWN_TASK"


class InvalidManifestError(FeatureExtractionError):
    """Raised when a manifest mapping is malformed."""

    code = "INVALID_MANIFEST"


class InvalidTranscriptError(FeatureExtractionError):
    """Raised when a transcript JSON is malformed."""

    code = "INVALID_TRANSCRIPT"


class InvalidTaskSpecError(FeatureExtractionError):
    """Raised when a task-spec JSON is malformed or references an unknown task."""

    code = "INVALID_TASK_SPEC"


class MissingInputError(FeatureExtractionError):
    """Raised when a required input file (audio/transcript/task-spec) is absent."""

    code = "MISSING_INPUT"


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
    """Fixed extraction defaults. 25 ms frames / 10 ms hop at 16 kHz.

    ``lpc_order`` calibrates the resonance features (Task 7) and is reserved
    here so the full calibration surface stays on one config object.
    """

    sample_rate: int = 16000
    frame_size: int = 400  # 25 ms at 16 kHz
    hop_size: int = 160  # 10 ms at 16 kHz
    pitch_min_hz: float = 70.0
    pitch_max_hz: float = 400.0
    pitch_autocorr_threshold: float = 0.30
    pause_threshold_s: float = 0.20
    long_pause_threshold_s: float = 2.0
    lpc_order: int = 12
    nonlinear_min_periods: int = 64
    recurrence_radius_sd: float = 0.1
    entropy_bins: int = 32

    def __post_init__(self) -> None:
        for name in ("nonlinear_min_periods", "entropy_bins"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        radius = self.recurrence_radius_sd
        if (
            isinstance(radius, bool)
            or not isinstance(radius, (int, float))
            or not math.isfinite(radius)
            or radius <= 0
        ):
            raise ValueError("recurrence_radius_sd must be a positive finite number")


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
    """Return the hexdigest SHA-256 of a file, streaming to bound memory.

    Raises :class:`MissingInputError` when the file does not exist.
    """
    p = Path(path)
    if not p.is_file():
        raise MissingInputError(f"input file not found: {p}")
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_hashes(manifest: dict) -> dict:
    """Record SHA-256 provenance for every distinct asset path in a manifest.

    Returns a plain, JSON-serializable mapping ``{kind: {path: sha256}}`` for
    ``audio``, ``transcript`` and ``task_spec``. Paths are keyed by their string
    form so shared paths are hashed (and stored) exactly once and a nested dict
    works with :func:`json.dumps` unchanged.
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
    result = {name: {} for name in kinds.values()}
    for path_key, name in kinds.items():
        for row in rows:
            if not isinstance(row, dict):
                raise InvalidManifestError(f"row must be an object, got {type(row).__name__}")
            path = row[path_key]
            result[name][path] = sha256_file(path)
    return result


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExtractionInputs:
    """The three input files a single extraction needs.

    Deliberately contains only feature-extraction inputs, never diagnosis or
    other clinical fields, so callers can pass inputs around without leaking
    protected attributes.
    """

    audio_path: str
    transcript_path: str
    task_spec_path: str

    # pathlib.Path helpers keep the dataclass frozen while exposing usable paths
    @property
    def audio(self) -> Path:
        return Path(self.audio_path)

    @property
    def transcript(self) -> Path:
        return Path(self.transcript_path)

    @property
    def task_spec(self) -> Path:
        return Path(self.task_spec_path)


@dataclass(frozen=True)
class ManifestRow:
    """A single (participant_id, task) manifest row."""

    participant_id: str
    task: str
    recording_id: str
    audio_id: str
    transcript_id: str
    inputs: ExtractionInputs
    diagnosis: str
    age: int
    sex: str
    education_years: int

    @property
    def audio_path(self) -> str:
        return self.inputs.audio_path

    @property
    def transcript_path(self) -> str:
        return self.inputs.transcript_path

    @property
    def task_spec_path(self) -> str:
        return self.inputs.task_spec_path

    @property
    def audio(self) -> Path:
        return self.inputs.audio

    @property
    def transcript(self) -> Path:
        return self.inputs.transcript

    @property
    def task_spec(self) -> Path:
        return self.inputs.task_spec


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
                inputs=ExtractionInputs(
                    audio_path=row["audio_path"],
                    transcript_path=row["transcript_path"],
                    task_spec_path=row["task_spec_path"],
                ),
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
    """One transcript token. `start_s`/`end_s` are optional, else None."""

    kind: str
    text: str
    start_s: float | None = None
    end_s: float | None = None


@dataclass(frozen=True)
class Utterance:
    speaker: str
    start_s: float
    end_s: float
    tokens: tuple = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tokens", tuple(self.tokens))


@dataclass(frozen=True)
class Transcript:
    """Validated transcript with preserved identifier and language metadata.

    ``transcript_id`` is carried through from the source JSON (null when the
    transcript does not declare it); ``language`` defaults to ``"vi"`` for this
    Vietnamese corpus. ``utterances`` is a deeply immutable tuple.
    """

    transcript_id: str | None = None
    language: str | None = None
    utterances: tuple = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "utterances", tuple(self.utterances))


def _coerce_timestamp(utterance: int, field_name: str, value):
    try:
        ts = float(value)
    except (TypeError, ValueError):
        raise InvalidTranscriptError(f"utterance {utterance} {field_name} must be numeric")
    if not math.isfinite(ts):
        raise InvalidTranscriptError(f"utterance {utterance} {field_name} must be finite")
    return ts


def validate_transcript(transcript: dict) -> Transcript:
    """Validate transcript JSON v1 and return a frozen :class:`Transcript`.

    Enforces finite, ordered utterance timestamps; known speakers and token
    kinds; and finite, ordered optional token timestamps when present. The
    result carries the transcript's ``transcript_id`` and ``language`` and a
    deeply immutable ``utterances`` tuple.
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
        start = _coerce_timestamp(i, "start_s", utt.get("start_s"))
        end = _coerce_timestamp(i, "end_s", utt.get("end_s"))
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
            tok_start = tok.get("start_s")
            tok_end = tok.get("end_s")
            tok_start = (
                None if tok_start is None else _coerce_timestamp(i, f"token {j} start_s", tok_start)
            )
            tok_end = None if tok_end is None else _coerce_timestamp(i, f"token {j} end_s", tok_end)
            if tok_start is not None and tok_end is not None and tok_end < tok_start:
                raise InvalidTranscriptError(f"utterance {i} token {j} end before start")
            tokens.append(
                Token(kind=kind, text=nfc(tok.get("text", "")), start_s=tok_start, end_s=tok_end)
            )
        parsed.append(Utterance(speaker=speaker, start_s=start, end_s=end, tokens=tokens))

    return Transcript(
        transcript_id=transcript.get("transcript_id"),
        language=transcript.get("language", "vi"),
        utterances=parsed,
    )


# ---------------------------------------------------------------------------
# Task-spec validation
# ---------------------------------------------------------------------------
TASK_SPEC_FIELDS = {
    "picture_desc_1": {"concept_aliases", "entity_groups", "action_groups"},
    "picture_desc_2": {"concept_aliases", "entity_groups", "action_groups"},
    "immediate_recall": {"idea_aliases"},
    "delayed_recall": {"idea_aliases"},
    "phonemic_fluency": {"initials", "exclusions"},
    "semantic_fluency": {"item_aliases", "subcategories"},
}


def validate_task_spec(spec: dict) -> int:
    """Validate a versioned task-spec JSON and return its version.

    Enforces a supported ``version`` and the required, task-specific fields the
    plan documents (concept/entity/action groups for picture specs, idea
    aliases for recall, initials/exclusions for phonemic, item aliases +
    subcategories for semantic). When the optional ``task`` field is present it
    must name a known task.
    """
    if not isinstance(spec, dict):
        raise InvalidTaskSpecError("task spec must be a JSON object")
    version = spec.get("version")
    if not isinstance(version, int):
        raise InvalidTaskSpecError("task spec must declare an integer 'version'")
    if version != 1:
        raise InvalidTaskSpecError(f"unsupported task-spec version: {version!r}")
    task = spec.get("task")
    if task is not None and task not in TASK_SPEC_TASKS:
        raise InvalidTaskSpecError(f"task spec declares unknown task: {task!r}")
    required = TASK_SPEC_FIELDS.get(task)
    if required is not None:
        missing = required - spec.keys()
        if missing:
            raise InvalidTaskSpecError(f"task spec for {task!r} missing fields: {sorted(missing)}")
    return version
