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
from .catalog import (
    CATALOG_VERSION,
    FeatureDefinition,
    FeaturePack,
    PACKS,
    list_features,
)
from .result import (
    ExtractionContext,
    ExtractionError,
    FeatureBundle,
    FeatureIssue,
    InvalidConfigError,
    MissingAnnotationError,
    STABLE_ERROR_CODES,
    TargetSpeakerRequiredError,
    UnknownPackError,
    UnsupportedAudioError,
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

# Importing the built-in packs registers their feature definitions in the
# catalog; plain `import speech_features` must expose the full catalog.
from .features import acoustic as _acoustic_pack  # noqa: F401

__all__ = [
    "AnnotationLayer",
    "BatchFailure",
    "BatchResult",
    "CATALOG_VERSION",
    "DocumentSpeaker",
    "DocumentToken",
    "DocumentUtterance",
    "ExtractionConfig",
    "ExtractionContext",
    "ExtractionError",
    "ExtractionInputs",
    "FeatureBundle",
    "FeatureDefinition",
    "FeatureExtractionError",
    "FeatureIssue",
    "FeaturePack",
    "FeatureResult",
    "InvalidAudioError",
    "InvalidConfigError",
    "InvalidManifestError",
    "InvalidTaskSpecError",
    "InvalidTranscriptError",
    "KNOWN_TASKS",
    "ManifestRow",
    "MediaRef",
    "MissingAnnotationError",
    "MissingInputError",
    "PACKS",
    "STABLE_ERROR_CODES",
    "SpeechDocument",
    "TargetSpeakerRequiredError",
    "Token",
    "Transcript",
    "UnknownPackError",
    "UnknownTaskError",
    "UnsupportedAudioError",
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
    "list_features",
    "read_wav",
]
