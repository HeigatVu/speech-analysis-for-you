"""Deprecated evaluation import shim (removed in 0.3.0).

Importing this module emits a :class:`DeprecationWarning`; it only re-exports
the legacy AD evaluation API and its private helpers from
:mod:`speech_features.legacy.ad.evaluation`. No algorithm or business logic
lives here.
"""

from __future__ import annotations

import warnings

from .legacy.ad.evaluation import (  # noqa: F401 - private helpers re-exported for compatibility tests
    C_GRID,
    KNOWN_TASKS,
    EvaluationResult,
    _EXCLUDED_TOKENS,
    _check_random_state,
    _collect_participants,
    _default_pipe,
    _get,
    _metrics,
    _pick_c,
    _select_and_collect,
    _select_c,
    bootstrap_ci,
    evaluate_ad_baseline,
)

warnings.warn(
    "speech_features.evaluation is deprecated; use "
    "speech_features.legacy.ad.evaluation (removal in 0.3.0)",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "C_GRID",
    "EvaluationResult",
    "KNOWN_TASKS",
    "bootstrap_ci",
    "evaluate_ad_baseline",
]
