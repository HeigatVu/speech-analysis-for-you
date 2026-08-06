"""Vietnamese Alzheimer speech feature library.

Research-only, math-first feature extraction from participant WAV recordings
and reviewed Vietnamese transcripts. See the feature-library plan under
``docs/`` for the full contract.
"""

from .schema import (
    ExtractionConfig,
    FeatureExtractionError,
    FeatureResult,
    InvalidManifestError,
    InvalidTranscriptError,
    ManifestRow,
    KNOWN_TASKS,
    Token,
    UnknownTaskError,
    Utterance,
    nfc,
    sha256_file,
    validate_manifest,
    validate_transcript,
)

__all__ = [
    "ExtractionConfig",
    "FeatureExtractionError",
    "FeatureResult",
    "InvalidManifestError",
    "InvalidTranscriptError",
    "ManifestRow",
    "KNOWN_TASKS",
    "Token",
    "UnknownTaskError",
    "Utterance",
    "nfc",
    "sha256_file",
    "validate_manifest",
    "validate_transcript",
]
