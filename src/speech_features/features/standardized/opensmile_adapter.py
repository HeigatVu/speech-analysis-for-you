"""Lazy adapter for openSMILE's eGeMAPSv02 Functionals."""

from __future__ import annotations

import math
import numbers

import numpy as np

from ...result import ExtractionError, FeatureIssue, InvalidAudioError
from .definitions import EGEMAPS_KEYS, RAW_EGEMAPS_COLUMNS

_FEATURE_SET = "eGeMAPSv02"
_FEATURE_LEVEL = "Functionals"


def _provenance(*, available: bool, version: str | None) -> dict:
    return {
        "available": available,
        "opensmile_version": version,
        "opensmile_feature_set": _FEATURE_SET,
        "opensmile_feature_level": _FEATURE_LEVEL,
    }


def extract_egemaps_features(
    audio,
    sample_rate: int,
    *,
    recording_id: str = "",
    speaker_id: str = "",
) -> tuple[dict[str, float], tuple[FeatureIssue, ...], dict]:
    """Extract the frozen 88-value eGeMAPSv02 Functionals schema."""
    try:
        arr = np.asarray(audio, dtype=float)
        valid_rate = (
            isinstance(sample_rate, numbers.Integral)
            and not isinstance(sample_rate, bool)
            and sample_rate > 0
        )
        valid_audio = arr.ndim == 1 and arr.size > 0 and np.all(np.isfinite(arr))
    except (TypeError, ValueError):
        valid_rate = valid_audio = False
    if not valid_rate or not valid_audio:
        raise InvalidAudioError(
            "standardized acoustic input must be a finite non-empty mono numeric array "
            "with a positive integer sample rate"
        )

    try:
        import opensmile
    except ModuleNotFoundError as exc:
        if exc.name != "opensmile":
            raise
        issue = FeatureIssue(
            recording_id=recording_id,
            speaker_id=speaker_id,
            code="MISSING_OPTIONAL_DEPENDENCY",
            severity="warning",
            message="standardized_acoustic requires the standardized-acoustic extra",
        )
        return (
            dict.fromkeys(EGEMAPS_KEYS, math.nan),
            (issue,),
            _provenance(available=False, version=None),
        )

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    result = smile.process_signal(arr, int(sample_rate))
    try:
        shape = result.shape
        columns = tuple(result.columns)
    except AttributeError as exc:
        raise ExtractionError("openSMILE returned an invalid eGeMAPSv02 table") from exc
    if shape != (1, 88):
        raise ExtractionError(f"openSMILE eGeMAPSv02 result must have shape (1, 88), got {shape!r}")
    if columns != RAW_EGEMAPS_COLUMNS:
        raise ExtractionError("openSMILE eGeMAPSv02 raw column schema or order changed")
    try:
        raw_values = tuple(float(value) for value in result.iloc[0])
    except (TypeError, ValueError) as exc:
        raise ExtractionError("openSMILE eGeMAPSv02 values must be numeric") from exc

    return (
        dict(zip(EGEMAPS_KEYS, raw_values, strict=True)),
        (),
        _provenance(available=True, version=opensmile.__version__),
    )


__all__ = ["extract_egemaps_features"]
