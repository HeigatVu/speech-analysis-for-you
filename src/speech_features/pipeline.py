"""Deprecated AD pipeline import shim (removed in 0.3.0).

Importing this module emits a :class:`DeprecationWarning`; it only re-exports
the legacy AD pipeline API from :mod:`speech_features.legacy.ad.pipeline`, the
shared PCM core from :mod:`speech_features.audio`, and
:class:`~speech_features.schema.MissingInputError`. No algorithm or business
logic lives here.
"""

from __future__ import annotations

import warnings

from .audio import InvalidAudioError, UnsupportedAudioError, read_wav
from .legacy.ad.pipeline import (
    BatchFailure,
    BatchResult,
    _TASK_SCORER,  # noqa: F401 - re-exported so monkeypatch callers share the registry
    extract_manifest,
    extract_recording,
)
from .schema import MissingInputError

warnings.warn(
    "speech_features.pipeline is deprecated; AD extraction moved to "
    "speech_features.legacy.ad.pipeline and PCM reading to "
    "speech_features.read_wav (removal in 0.3.0)",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "BatchFailure",
    "BatchResult",
    "InvalidAudioError",
    "MissingInputError",
    "UnsupportedAudioError",
    "extract_manifest",
    "extract_recording",
    "read_wav",
]
