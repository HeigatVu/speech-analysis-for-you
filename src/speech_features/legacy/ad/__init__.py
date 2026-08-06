"""Supported 0.2.x legacy AD namespace.

Lazily exposes the legacy AD pipeline names (``BatchFailure``, ``BatchResult``,
``extract_recording``, ``extract_manifest``) and evaluation names (``C_GRID``,
``EvaluationResult``, ``KNOWN_TASKS``, ``bootstrap_ci``,
``evaluate_ad_baseline``). Plain ``import speech_features.legacy.ad`` is
warning-free and does not import scikit-learn; the evaluation module loads
only when one of its names is accessed.
"""

from __future__ import annotations

_PIPELINE_NAMES = ("BatchFailure", "BatchResult", "extract_manifest", "extract_recording")
_EVALUATION_NAMES = (
    "C_GRID",
    "EvaluationResult",
    "KNOWN_TASKS",
    "bootstrap_ci",
    "evaluate_ad_baseline",
)

__all__ = _PIPELINE_NAMES + _EVALUATION_NAMES


def __getattr__(name: str):
    if name in _PIPELINE_NAMES:
        from . import pipeline

        return getattr(pipeline, name)
    if name in _EVALUATION_NAMES:
        from . import evaluation

        return getattr(evaluation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
