"""Legacy AD manifest/task pipeline (moved from ``speech_features.pipeline``).

Supported 0.2.x legacy namespace: ``BatchFailure``, ``BatchResult``,
``extract_recording``, and ``extract_manifest``. Composes the acoustic and
linguistic/task-spec extractors into a single immutable
:class:`~speech_features.schema.FeatureResult` with SHA-256 provenance. Audio
is read through the shared population-neutral
:mod:`speech_features.audio` PCM core.

Batch orchestration (:func:`extract_manifest`) validates the manifest and runs
each row through :func:`extract_recording`, isolating any failing row behind a
stable :class:`FeatureExtractionError.code` instead of aborting the batch.
Diagnosis never enters an extractor call or a :class:`FeatureResult`; the
manifest label is surfaced separately on :class:`BatchResult.labels` for use
only at evaluation time.

Deprecated through 0.2.x; eligible for removal in 0.3.0. New code must use the
label-free :mod:`speech_features.extraction` API instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import MappingProxyType

from ...acoustic import extract_acoustic
from ...audio import (  # noqa: F401 - re-exported for the deprecated pipeline shim
    InvalidAudioError,
    UnsupportedAudioError,
    read_wav,
)
from ...linguistic import extract_linguistic, participant_tokens
from ...schema import (
    ExtractionConfig,
    FeatureExtractionError,
    FeatureResult,
    InvalidManifestError,
    InvalidTaskSpecError,
    InvalidTranscriptError,
    MissingInputError,  # noqa: F401 - re-exported for the deprecated pipeline shim
    sha256_file,
    validate_manifest,
    validate_task_spec,
    validate_transcript,
)
from . import tasks as task_scorers

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
    "extract_manifest",
    "extract_recording",
]
