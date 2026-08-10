"""Frozen eGeMAPSv02 Functionals schema and catalog definitions."""

from __future__ import annotations

import re

from ...catalog import FeatureDefinition, register_feature

RAW_EGEMAPS_COLUMNS = (
    "F0semitoneFrom27.5Hz_sma3nz_amean",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "F0semitoneFrom27.5Hz_sma3nz_percentile20.0",
    "F0semitoneFrom27.5Hz_sma3nz_percentile50.0",
    "F0semitoneFrom27.5Hz_sma3nz_percentile80.0",
    "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
    "F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope",
    "F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope",
    "F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope",
    "F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope",
    "loudness_sma3_amean",
    "loudness_sma3_stddevNorm",
    "loudness_sma3_percentile20.0",
    "loudness_sma3_percentile50.0",
    "loudness_sma3_percentile80.0",
    "loudness_sma3_pctlrange0-2",
    "loudness_sma3_meanRisingSlope",
    "loudness_sma3_stddevRisingSlope",
    "loudness_sma3_meanFallingSlope",
    "loudness_sma3_stddevFallingSlope",
    "spectralFlux_sma3_amean",
    "spectralFlux_sma3_stddevNorm",
    "mfcc1_sma3_amean",
    "mfcc1_sma3_stddevNorm",
    "mfcc2_sma3_amean",
    "mfcc2_sma3_stddevNorm",
    "mfcc3_sma3_amean",
    "mfcc3_sma3_stddevNorm",
    "mfcc4_sma3_amean",
    "mfcc4_sma3_stddevNorm",
    "jitterLocal_sma3nz_amean",
    "jitterLocal_sma3nz_stddevNorm",
    "shimmerLocaldB_sma3nz_amean",
    "shimmerLocaldB_sma3nz_stddevNorm",
    "HNRdBACF_sma3nz_amean",
    "HNRdBACF_sma3nz_stddevNorm",
    "logRelF0-H1-H2_sma3nz_amean",
    "logRelF0-H1-H2_sma3nz_stddevNorm",
    "logRelF0-H1-A3_sma3nz_amean",
    "logRelF0-H1-A3_sma3nz_stddevNorm",
    "F1frequency_sma3nz_amean",
    "F1frequency_sma3nz_stddevNorm",
    "F1bandwidth_sma3nz_amean",
    "F1bandwidth_sma3nz_stddevNorm",
    "F1amplitudeLogRelF0_sma3nz_amean",
    "F1amplitudeLogRelF0_sma3nz_stddevNorm",
    "F2frequency_sma3nz_amean",
    "F2frequency_sma3nz_stddevNorm",
    "F2bandwidth_sma3nz_amean",
    "F2bandwidth_sma3nz_stddevNorm",
    "F2amplitudeLogRelF0_sma3nz_amean",
    "F2amplitudeLogRelF0_sma3nz_stddevNorm",
    "F3frequency_sma3nz_amean",
    "F3frequency_sma3nz_stddevNorm",
    "F3bandwidth_sma3nz_amean",
    "F3bandwidth_sma3nz_stddevNorm",
    "F3amplitudeLogRelF0_sma3nz_amean",
    "F3amplitudeLogRelF0_sma3nz_stddevNorm",
    "alphaRatioV_sma3nz_amean",
    "alphaRatioV_sma3nz_stddevNorm",
    "hammarbergIndexV_sma3nz_amean",
    "hammarbergIndexV_sma3nz_stddevNorm",
    "slopeV0-500_sma3nz_amean",
    "slopeV0-500_sma3nz_stddevNorm",
    "slopeV500-1500_sma3nz_amean",
    "slopeV500-1500_sma3nz_stddevNorm",
    "spectralFluxV_sma3nz_amean",
    "spectralFluxV_sma3nz_stddevNorm",
    "mfcc1V_sma3nz_amean",
    "mfcc1V_sma3nz_stddevNorm",
    "mfcc2V_sma3nz_amean",
    "mfcc2V_sma3nz_stddevNorm",
    "mfcc3V_sma3nz_amean",
    "mfcc3V_sma3nz_stddevNorm",
    "mfcc4V_sma3nz_amean",
    "mfcc4V_sma3nz_stddevNorm",
    "alphaRatioUV_sma3nz_amean",
    "hammarbergIndexUV_sma3nz_amean",
    "slopeUV0-500_sma3nz_amean",
    "slopeUV500-1500_sma3nz_amean",
    "spectralFluxUV_sma3nz_amean",
    "loudnessPeaksPerSec",
    "VoicedSegmentsPerSec",
    "MeanVoicedSegmentLengthSec",
    "StddevVoicedSegmentLengthSec",
    "MeanUnvoicedSegmentLength",
    "StddevUnvoicedSegmentLength",
    "equivalentSoundLevel_dBp",
)


def _canonical_key(raw: str) -> str:
    ascii_name = raw.encode("ascii", "ignore").decode("ascii").lower()
    return "egemaps_" + re.sub(r"[^a-z0-9]+", "_", ascii_name).strip("_")


EGEMAPS_KEYS = tuple(_canonical_key(raw) for raw in RAW_EGEMAPS_COLUMNS)
if len(EGEMAPS_KEYS) != 88 or len(EGEMAPS_KEYS) != len(set(EGEMAPS_KEYS)):
    raise ValueError("eGeMAPSv02 Functionals schema must canonicalize to 88 unique keys")


def _domain(raw: str) -> str:
    lower = raw.lower()
    if lower.startswith(("f1", "f2", "f3")):
        return "articulation"
    if lower.startswith(("spectral", "mfcc", "alpharatio", "hammarberg", "slope")):
        return "spectral"
    if lower.startswith(("jitter", "shimmer", "hnr", "logrel")):
        return "phonation"
    if "segment" in lower:
        return "timing"
    return "prosody"


def _unit(raw: str) -> str:
    lower = raw.lower()
    if "stddevnorm" in lower:
        return "ratio"
    if lower.startswith("f0semitone"):
        return "semitones/s" if "slope" in lower else "semitones"
    if lower.startswith("loudness_sma"):
        return "sone/s" if "slope" in lower else "sone"
    if lower.startswith(("f1", "f2", "f3")):
        return "dB" if "amplitude" in lower else "Hz"
    if lower.startswith(("shimmer", "hnr", "logrel", "alpharatio", "hammarberg")):
        return "dB"
    if lower.startswith("jitter"):
        return "ratio"
    if lower.startswith("slope"):
        return "dB/kHz"
    if lower.endswith("persec"):
        return "count/s"
    if "segmentlength" in lower:
        return "s"
    if lower.startswith("equivalentsoundlevel"):
        return "dB"
    return "coefficient"


_DEFINITIONS = tuple(
    FeatureDefinition(
        key=key,
        pack="standardized_acoustic",
        level="recording",
        unit=_unit(raw),
        population="adult",
        reference="https://doi.org/10.1109/TAFFC.2015.2457417",
        domain=_domain(raw),
        language_scope="language_sensitive",
        tasks=("connected_speech", "sustained_vowel"),
        disorders=(),
        evidence_level="standard_feature_set",
    )
    for raw, key in zip(RAW_EGEMAPS_COLUMNS, EGEMAPS_KEYS)
)

_registered = False


def register_egemaps_features() -> None:
    """Register the standardized acoustic pack once."""
    global _registered
    if _registered:
        return
    for definition in _DEFINITIONS:
        register_feature(definition)
    _registered = True


__all__ = ["EGEMAPS_KEYS", "RAW_EGEMAPS_COLUMNS", "register_egemaps_features"]
