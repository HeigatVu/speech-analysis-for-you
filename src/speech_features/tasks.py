"""Deprecated task-scorer import shim (removed in 0.3.0).

Importing this module emits a :class:`DeprecationWarning`; it only re-exports
the legacy scorer API from :mod:`speech_features.legacy.ad.tasks`. No
algorithm or business logic lives here.
"""

from __future__ import annotations

import warnings

from .legacy.ad.tasks import (
    SIMILARITY_THRESHOLD,
    levenshtein,
    match_alias,
    match_key,
    normalised_similarity,
    score_phonemic,
    score_picture,
    score_recall,
    score_semantic,
)

warnings.warn(
    "speech_features.tasks is deprecated; use speech_features.legacy.ad.tasks (removal in 0.3.0)",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "SIMILARITY_THRESHOLD",
    "levenshtein",
    "normalised_similarity",
    "match_alias",
    "match_key",
    "score_picture",
    "score_recall",
    "score_phonemic",
    "score_semantic",
]
