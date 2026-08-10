"""Shared energy, spectral-shape, and MFCC summaries."""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.fft import dct
from scipy.stats import kurtosis, skew

from ...result import FeatureIssue
from ...schema import ExtractionConfig
from .definitions import SPECTRUM_KEYS
from .spectrum import _speech_frame_regions

ADVANCED_KEYS = tuple(
    key
    for key in SPECTRUM_KEYS
    if key.startswith("spectral_mfcc_")
    or key
    in {
        "spectral_energy_mean_db",
        "spectral_energy_sd_db",
        "spectral_skewness_mean",
        "spectral_skewness_sd",
        "spectral_kurtosis_mean",
        "spectral_kurtosis_sd",
        "spectral_low_high_energy_ratio_db",
    }
)


def mfcc_frames(frames: np.ndarray, sample_rate: int, n_mfcc: int = 13) -> np.ndarray:
    """Power spectrum -> 26 triangular Mel filters -> log -> orthonormal DCT-II."""
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(frames.shape[1], d=1.0 / sample_rate)
    nyquist_mel = 2595.0 * math.log10(1.0 + (sample_rate / 2.0) / 700.0)
    mel_points = np.linspace(0.0, nyquist_mel, 28)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    filters = np.zeros((26, frequencies.size), dtype=float)
    for index, (left, center, right) in enumerate(zip(hz_points, hz_points[1:], hz_points[2:])):
        filters[index] = np.minimum(
            (frequencies - left) / (center - left),
            (right - frequencies) / (right - center),
        ).clip(0.0, 1.0)
    mel_power = np.maximum(power @ filters.T, np.finfo(float).eps)
    coefficients = dct(np.log(mel_power), type=2, axis=1, norm="ortho")
    return coefficients[:, 1 : n_mfcc + 1]


def distribution_stats(values: np.ndarray) -> tuple[float, float, float, float]:
    """Population mean/SD and scipy unbiased skew/kurtosis; NaN if insufficient."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return math.nan, math.nan, math.nan, math.nan
    mean = float(np.mean(finite))
    sd = float(np.std(finite))
    if np.ptp(finite) == 0.0:
        return mean, sd, math.nan, math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        skewness = float(skew(finite, bias=False)) if finite.size >= 3 else math.nan
        excess_kurtosis = float(kurtosis(finite, bias=False)) if finite.size >= 4 else math.nan
    return mean, sd, skewness, excess_kurtosis


def _issue(recording_id: str, speaker_id: str, feature: str) -> FeatureIssue:
    return FeatureIssue(
        recording_id=recording_id,
        speaker_id=speaker_id,
        code="INSUFFICIENT_SPEECH_FRAMES",
        severity="warning",
        message="insufficient target speech frames; advanced spectral feature unavailable",
        feature=feature,
    )


def advanced_features(
    audio,
    sample_rate: int,
    intervals,
    config: ExtractionConfig,
    recording_id: str,
    speaker_id: str,
    issues: list[FeatureIssue],
) -> dict[str, float]:
    """Compute every expanded spectral/MFCC key over target speech frames."""
    features = {key: math.nan for key in ADVANCED_KEYS}
    regions = _speech_frame_regions(audio, sample_rate, intervals, config)
    if not regions:
        issues.extend(_issue(recording_id, speaker_id, key) for key in ADVANCED_KEYS)
        return features

    frames = np.concatenate(regions)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    floor = np.finfo(float).eps
    energy_db = 10.0 * np.log10(np.maximum(power.sum(axis=1), floor))
    features["spectral_energy_mean_db"] = float(np.mean(energy_db))
    features["spectral_energy_sd_db"] = float(np.std(energy_db))

    frequencies = np.fft.rfftfreq(config.frame_size, d=1.0 / sample_rate)
    total = np.maximum(power.sum(axis=1), floor)
    probability = power / total[:, None]
    centroid = (probability * frequencies).sum(axis=1)
    centered = frequencies[None, :] - centroid[:, None]
    variance = (probability * centered**2).sum(axis=1)
    valid = variance > floor
    spectral_skewness = np.full(frames.shape[0], math.nan)
    spectral_kurtosis = np.full(frames.shape[0], math.nan)
    spectral_skewness[valid] = (probability[valid] * centered[valid] ** 3).sum(axis=1) / variance[
        valid
    ] ** 1.5
    spectral_kurtosis[valid] = (probability[valid] * centered[valid] ** 4).sum(axis=1) / variance[
        valid
    ] ** 2
    for values, mean_key, sd_key in (
        (spectral_skewness, "spectral_skewness_mean", "spectral_skewness_sd"),
        (spectral_kurtosis, "spectral_kurtosis_mean", "spectral_kurtosis_sd"),
    ):
        finite = values[np.isfinite(values)]
        if finite.size:
            features[mean_key] = float(np.mean(finite))
            features[sd_key] = float(np.std(finite))

    split = sample_rate / 4.0
    low = float(power[:, frequencies <= split].sum())
    high = float(power[:, frequencies > split].sum())
    features["spectral_low_high_energy_ratio_db"] = float(
        10.0 * math.log10(max(low, floor) / max(high, floor))
    )

    mfcc = mfcc_frames(frames, sample_rate)
    for coefficient in range(13):
        stats = distribution_stats(mfcc[:, coefficient])
        for stat, value in zip(("mean", "sd", "skewness", "kurtosis"), stats):
            features[f"spectral_mfcc_{coefficient + 1}_{stat}"] = value

    for key, value in features.items():
        if math.isnan(value):
            issues.append(_issue(recording_id, speaker_id, key))
    return features


__all__ = ["advanced_features", "distribution_stats", "mfcc_frames"]
