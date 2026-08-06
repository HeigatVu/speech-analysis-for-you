"""Offline extraction pipeline (Task 4).

Composes the Task 2 acoustic and Task 3 linguistic/task-spec extractors into a
single immutable :class:`~speech_features.schema.FeatureResult` with SHA-256
provenance. Audio is read with the stdlib ``wave`` module (standard PCM only),
converted from mono/stereo to a finite mono float array, and resampled to
:data:`ExtractionConfig.sample_rate` with ``scipy.signal.resample_poly``.

Batch orchestration (:func:`extract_manifest`) validates the manifest and runs
each row through :func:`extract_recording`, isolating any failing row behind a
stable :class:`FeatureExtractionError.code` instead of aborting the batch.
Diagnosis never enters an extractor call or a :class:`FeatureResult`; the
manifest label is surfaced separately on :class:`BatchResult.labels` for use
only at evaluation time.

No ASR, no librosa/openSMILE/spaCy/embeddings, and no labels here.
"""

from __future__ import annotations

import json
import math
import wave
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from scipy.signal import resample_poly

from .acoustic import extract_acoustic
from . import tasks as task_scorers
from .linguistic import extract_linguistic, participant_tokens
from .schema import (
    ExtractionConfig,
    FeatureExtractionError,
    FeatureResult,
    InvalidManifestError,
    InvalidTaskSpecError,
    InvalidTranscriptError,
    MissingInputError,
    sha256_file,
    validate_manifest,
    validate_task_spec,
    validate_transcript,
)


class InvalidAudioError(FeatureExtractionError):
    """Raised when a WAV file cannot be read as standard PCM mono/stereo."""

    code = "INVALID_AUDIO"


class UnsupportedAudioError(FeatureExtractionError):
    """Raised when a WAV is not standard PCM mono/stereo media."""

    code = "UNSUPPORTED_AUDIO"


_SCALE_BY_WIDTH = {2: 32768.0, 3: 8388608.0, 4: 2147483648.0}

# Dispatch from manifest task name to its external task-spec scorer.
_TASK_SCORER = {
    "picture_desc_1": task_scorers.score_picture,
    "picture_desc_2": task_scorers.score_picture,
    "immediate_recall": task_scorers.score_recall,
    "delayed_recall": task_scorers.score_recall,
    "phonemic_fluency": task_scorers.score_phonemic,
    "semantic_fluency": task_scorers.score_semantic,
}


# ---------------------------------------------------------------------------
# Low-level audio I/O
# ---------------------------------------------------------------------------
def _bytes_to_samples(raw: bytes, width: int, count: int) -> np.ndarray:
    """Decode interleaved PCM integers (``width`` bytes each) as int64."""
    if width == 1:
        return np.frombuffer(raw, dtype=np.uint8).astype(np.int64)[:count]
    if width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.int64)[:count]
    if width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.int64)[:count]
    # 24-bit signed PCM has no native width-3 dtype; assemble each sample from
    # three little-endian bytes with vectorised NumPy ops (no per-sample loop).
    n_bytes = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
    if n_bytes.size % 3 != 0:
        raise ValueError(f"truncated 24-bit PCM buffer ({n_bytes.size} bytes, not a multiple of 3)")
    three = n_bytes[: count * 3].reshape(-1, 3)
    lo, mid, hi = three[:, 0], three[:, 1], three[:, 2]
    u = lo | (mid << 8) | (hi << 16)
    return np.where(u & 0x800000, u - 0x1_000000, u)


def _to_float(samples: np.ndarray, width: int) -> np.ndarray:
    """Map PCM integers to the float domain ``[-1, 1]`` (8-bit is unsigned)."""
    if width == 1:
        return (samples - 128.0) / 128.0
    return samples / _SCALE_BY_WIDTH[width]


def _resample(samples: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    g = math.gcd(src_sr, target_sr)
    return resample_poly(samples, up=target_sr // g, down=src_sr // g)


def read_wav(path, *, sample_rate: int = 16000) -> np.ndarray:
    """Read a standard PCM WAV as a mono float array at ``sample_rate``.

    Accepts mono or stereo integer PCM (8/16/24/32-bit, standard ``wave``
    formats). Stereo is downmixed to mono by the channel mean; the resulting
    samples are resampled to ``sample_rate`` with ``resample_poly``. At least
    finite, non-empty audio is guaranteed; malformed files raise
    :class:`InvalidAudioError` and unsupported media (non-PCM, non-standard
    width or channel count) raises :class:`UnsupportedAudioError`.
    """
    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            src_sr = wf.getframerate()
            comptype = wf.getcomptype()
            n_frames = wf.getnframes()
            raw = wf.readframes(int(n_frames))
    except (wave.Error, OSError, EOFError) as exc:
        # The stdlib wave reader rejects non-PCM format codes at open time
        # ("unknown format: <code>"); those files are unsupported media, not
        # malformed PCM, so they surface the stable UNSUPPORTED_AUDIO code.
        if isinstance(exc, wave.Error) and "unknown format" in str(exc):
            raise UnsupportedAudioError(f"unsupported WAV (not PCM): {exc}") from exc
        raise InvalidAudioError(f"cannot read WAV {path}: {exc}") from exc

    if comptype not in (b"NONE", "NONE", None):
        raise UnsupportedAudioError(f"unsupported WAV (not PCM), compression: {comptype!r}")
    if width not in (1, 2, 3, 4):
        raise UnsupportedAudioError(f"unsupported sample width in WAV: {width} bytes")
    if channels not in (1, 2):
        raise UnsupportedAudioError(f"unsupported channel count in WAV: {channels}")
    if src_sr <= 0 or n_frames <= 0:
        raise InvalidAudioError("WAV is empty or has an invalid sample rate")

    try:
        count = n_frames * channels
        samples = _to_float(_bytes_to_samples(raw, width, count), width)
        if channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)
        if not np.all(np.isfinite(samples)):
            raise InvalidAudioError("WAV contains non-finite samples")
        if src_sr != sample_rate:
            samples = _resample(samples, src_sr, sample_rate)
            # resample_poly is a polynomial all-pole path; still guard the output
            # against any accidental non-finite introduction at the boundary.
            if not np.all(np.isfinite(samples)):
                raise InvalidAudioError("resampled WAV contains non-finite samples")
        return samples
    except InvalidAudioError:
        raise
    except (ValueError, IndexError, TypeError) as exc:
        raise InvalidAudioError(f"cannot decode WAV {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# JSON loading / validation
# ---------------------------------------------------------------------------
def _load_json(path, exc_type, label):
    try:
        with open(str(path), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        raise exc_type(f"cannot read {label} {path}: {exc}") from exc


def _load_transcript(path):
    return validate_transcript(_load_json(path, InvalidTranscriptError, "transcript"))


def _load_task_spec(path):
    spec = _load_json(path, InvalidTaskSpecError, "task spec")
    validate_task_spec(spec)
    task = spec.get("task")
    if task not in _TASK_SCORER:
        raise InvalidTaskSpecError(f"task spec must declare a known 'task', got {task!r}")
    return spec


# ---------------------------------------------------------------------------
# Single-recording extraction
# ---------------------------------------------------------------------------
def extract_recording(
    audio_path,
    transcript_path,
    task_spec_path,
    *,
    config: ExtractionConfig | None = None,
    recording_id: str = "",
    participant_id: str = "",
) -> FeatureResult:
    """Extract one immutable :class:`FeatureResult` from its three input files.

    Hashes all three inputs first so a missing file surfaces as
    :class:`MissingInputError` (``MISSING_INPUT``) before any extraction.
    Acoustic, participant-only linguistic, and the manifest task's external
    scorer features are merged into a single mapping; acoustic/absent-speech
    diagnostics become quality flags. No diagnosis is read or stored.
    """
    cfg = config if config is not None else ExtractionConfig()
    hashes = {
        "audio": sha256_file(audio_path),
        "transcript": sha256_file(transcript_path),
        "task_spec": sha256_file(task_spec_path),
    }

    audio = read_wav(audio_path, sample_rate=cfg.sample_rate)
    transcript = _load_transcript(transcript_path)
    spec = _load_task_spec(task_spec_path)
    task = spec["task"]

    acoustics = extract_acoustic(audio, cfg.sample_rate, config=cfg)
    linguistic = extract_linguistic(transcript)
    task_feats = _TASK_SCORER[task](transcript, spec)

    features = {}
    features.update(acoustics.features)
    features.update(linguistic)
    features.update(task_feats)

    flags = list(acoustics.flags)
    if not participant_tokens(transcript):
        flags.append("no_participant_speech")

    return FeatureResult(
        recording_id=recording_id,
        participant_id=participant_id,
        task=task,
        features=features,
        quality_flags=flags,
        input_hashes=hashes,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BatchFailure:
    """A single manifest row that failed, with its stable error code."""

    key: tuple
    code: str
    error_type: str
    message: str


@dataclass(frozen=True)
class BatchResult:
    """Outcome of a whole-manifest extraction run.

    ``recordings`` maps a ``(participant_id, task, recording_id)`` key to the
    successful :class:`FeatureResult`; ``failures`` carries the per-row
    failures; ``labels`` keeps the manifest diagnosis separate for evaluation;
    ``hashes`` aggregates per-recording provenance by the same key.
    """

    recordings: dict = field(default_factory=dict)
    failures: tuple = field(default_factory=tuple)
    labels: dict = field(default_factory=dict)
    hashes: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "recordings", MappingProxyType(dict(self.recordings)))
        object.__setattr__(self, "failures", tuple(self.failures))
        object.__setattr__(self, "labels", MappingProxyType(dict(self.labels)))
        object.__setattr__(self, "hashes", MappingProxyType(dict(self.hashes)))


def extract_manifest(manifest_path, *, config: ExtractionConfig | None = None) -> BatchResult:
    """Run every manifest row through :func:`extract_recording` with isolation.

    A failing row is recorded as a :class:`BatchFailure` (stable error code,
    row key, message) so the remaining rows still complete. Manifest labels are
    collected into ``labels`` regardless of extraction success, then kept
    separate from the feature results.
    """
    cfg = config if config is not None else ExtractionConfig()
    manifest = _load_json(manifest_path, InvalidManifestError, "manifest")
    rows = validate_manifest(manifest)

    recordings: dict = {}
    labels: dict = {}
    failure_list: list[BatchFailure] = []

    for row in rows:
        key = (row.participant_id, row.task, row.recording_id)
        labels[key] = row.diagnosis
        try:
            result = extract_recording(
                row.audio_path,
                row.transcript_path,
                row.task_spec_path,
                config=cfg,
                recording_id=row.recording_id,
                participant_id=row.participant_id,
            )
            recordings[key] = result
        except FeatureExtractionError as exc:
            failure_list.append(
                BatchFailure(
                    key=key,
                    code=exc.code,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
        except Exception as exc:  # noqa: BLE001 - isolate any per-row failure
            failure_list.append(
                BatchFailure(
                    key=key,
                    code="EXTRACTION_ERROR",
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )

    hashes = {key: result.input_hashes for key, result in recordings.items()}
    return BatchResult(
        recordings=recordings,
        failures=tuple(failure_list),
        labels=labels,
        hashes=hashes,
    )


__all__ = [
    "BatchFailure",
    "BatchResult",
    "InvalidAudioError",
    "MissingInputError",  # re-exported: missing inputs surface through extract_recording
    "UnsupportedAudioError",
    "extract_manifest",
    "extract_recording",
    "read_wav",
]
