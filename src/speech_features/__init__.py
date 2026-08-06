"""Vietnamese Alzheimer speech feature library.

Research-only, math-first feature extraction from participant WAV recordings
and reviewed Vietnamese transcripts. See the feature-library plan under
``docs/`` for the full contract.

The label-free API is ``extract`` / ``extract_batch`` / ``read_wav`` /
``list_features``. The 0.1.x AD pipeline names (``BatchFailure``,
``BatchResult``, ``extract_recording``, ``extract_manifest``) remain
available with a :class:`DeprecationWarning` through the lazy module
``__getattr__``; their implementation lives in
``speech_features.legacy.ad.pipeline``.
"""

__version__ = "0.2.0"

import warnings

from .audio import InvalidAudioError, read_wav
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
from .extraction import (
    extract,
    extract_batch,
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
from .features import linguistic as _linguistic_pack  # noqa: F401

_LEGACY_PIPELINE_NAMES = ("BatchFailure", "BatchResult", "extract_manifest", "extract_recording")


def __getattr__(name: str):
    if name in _LEGACY_PIPELINE_NAMES:
        warnings.warn(
            f"speech_features.{name} is deprecated; use "
            f"speech_features.legacy.ad.pipeline.{name} (removal in 0.3.0)",
            DeprecationWarning,
            stacklevel=2,
        )
        from .legacy.ad import pipeline as _legacy_pipeline

        return getattr(_legacy_pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "extract",
    "extract_batch",
    "extract_manifest",
    "extract_recording",
    "list_features",
    "read_wav",
]
