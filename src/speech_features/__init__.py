"""Vietnamese Alzheimer speech feature library.

Research-only, math-first feature extraction from participant WAV recordings
and reviewed Vietnamese transcripts. See the feature-library plan under
``docs/`` for the full contract.
"""

__version__ = "0.2.0"

from .document import (
    AnnotationLayer,
    DocumentSpeaker,
    DocumentToken,
    DocumentUtterance,
    MediaRef,
    SpeechDocument,
    load_document,
    save_document,
)
from .pipeline import (
    BatchFailure,
    BatchResult,
    InvalidAudioError,
    extract_manifest,
    extract_recording,
    read_wav,
)
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
    "AnnotationLayer",
    "BatchFailure",
    "BatchResult",
    "DocumentSpeaker",
    "DocumentToken",
    "DocumentUtterance",
    "ExtractionConfig",
    "ExtractionInputs",
    "FeatureExtractionError",
    "FeatureResult",
    "InvalidAudioError",
    "InvalidManifestError",
    "InvalidTaskSpecError",
    "InvalidTranscriptError",
    "KNOWN_TASKS",
    "ManifestRow",
    "MediaRef",
    "MissingInputError",
    "SpeechDocument",
    "Token",
    "Transcript",
    "UnknownTaskError",
    "Utterance",
    "load_document",
    "nfc",
    "save_document",
    "sha256_file",
    "validate_manifest",
    "validate_task_spec",
    "validate_transcript",
    "extract_manifest",
    "extract_recording",
    "read_wav",
]
