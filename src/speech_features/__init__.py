"""Vietnamese Alzheimer speech feature library.

Research-only, math-first feature extraction from participant WAV recordings
and reviewed Vietnamese transcripts. See the feature-library plan under
``docs/`` for the full contract.
"""

from .schema import (
    ExtractionConfig,
    ExtractionInputs,
    FeatureExtractionError,
    FeatureResult,
    InvalidManifestError,
    InvalidTaskSpecError,
    InvalidTranscriptError,
    KNOWN_TASKS,
    ManifestRow,
    MissingInputError,
    Token,
    Transcript,
    UnknownTaskError,
    Utterance,
    nfc,
    sha256_file,
    validate_manifest,
    validate_task_spec,
    validate_transcript,
)

__all__ = [
    "ExtractionConfig",
    "ExtractionInputs",
    "FeatureExtractionError",
    "FeatureResult",
    "InvalidManifestError",
    "InvalidTaskSpecError",
    "InvalidTranscriptError",
    "KNOWN_TASKS",
    "ManifestRow",
    "MissingInputError",
    "Token",
    "Transcript",
    "UnknownTaskError",
    "Utterance",
    "nfc",
    "sha256_file",
    "validate_manifest",
    "validate_task_spec",
    "validate_transcript",
]
