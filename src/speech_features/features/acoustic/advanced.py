"""Shared energy, spectral-shape, and MFCC summaries."""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.fft import dct
from scipy.spatial.distance import pdist
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

_NONLINEAR_MIN_PERIODS = 64


def _perturbation_quotient(values: np.ndarray, window: int) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < window or not np.all(np.isfinite(values)):
        return math.nan
    denominator = float(np.mean(values))
    if denominator <= 0.0:
        return math.nan
    local_means = np.convolve(values, np.ones(window) / window, mode="valid")
    half = window // 2
    centers = values[half : values.size - half]
    return float(np.mean(np.abs(centers - local_means)) / denominator)


def jitter_rap(periods: np.ndarray) -> float:
    """Relative average perturbation over three pitch periods."""
    return _perturbation_quotient(periods, 3)


def jitter_ppq5(periods: np.ndarray) -> float:
    """Five-period pitch perturbation quotient."""
    return _perturbation_quotient(periods, 5)


def shimmer_apq(amplitudes: np.ndarray, window: int) -> float:
    """Amplitude perturbation quotient for a 3, 5, or 11-frame window."""
    if window not in {3, 5, 11}:
        raise ValueError("shimmer APQ window must be 3, 5, or 11")
    return _perturbation_quotient(amplitudes, window)


def _period_series(periods: np.ndarray, minimum: int) -> np.ndarray | None:
    values = np.asarray(periods, dtype=float)
    if values.size < minimum or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        return None
    return values


def pitch_period_entropy(
    periods: np.ndarray,
    bins: int = 32,
    minimum: int = _NONLINEAR_MIN_PERIODS,
) -> float:
    """Normalized Shannon entropy of detrended log periods over fixed bins."""
    values = _period_series(periods, minimum)
    if values is None or bins <= 1:
        return math.nan
    logged = np.log(values)
    x = np.arange(logged.size, dtype=float)
    detrended = logged - np.polyval(np.polyfit(x, logged, 1), x)
    if np.ptp(detrended) <= np.finfo(float).eps:
        return 0.0
    counts = np.histogram(detrended, bins=bins, range=(detrended.min(), detrended.max()))[0]
    probabilities = counts[counts > 0] / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)) / math.log(bins))


def recurrence_period_density_entropy(
    periods: np.ndarray,
    radius_sd: float = 0.1,
    minimum: int = _NONLINEAR_MIN_PERIODS,
) -> float:
    """Normalized entropy of recurrence counts at lags 1 through min(100, n/2)."""
    values = _period_series(periods, minimum)
    if values is None or radius_sd <= 0.0:
        return math.nan
    max_lag = min(100, values.size // 2)
    radius = radius_sd * float(np.std(values))
    counts = np.asarray(
        [
            np.count_nonzero(np.abs(values[lag:] - values[:-lag]) <= radius)
            for lag in range(1, max_lag + 1)
        ],
        dtype=float,
    )
    counts = counts[counts > 0.0]
    if counts.size == 0:
        return math.nan
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)) / math.log(max_lag))


def detrended_fluctuation_analysis(
    periods: np.ndarray,
    minimum: int = _NONLINEAR_MIN_PERIODS,
) -> float:
    """DFA log-log slope over non-overlapping windows of 4 through 64 periods."""
    values = _period_series(periods, minimum)
    if values is None:
        return math.nan
    profile = np.cumsum(values - np.mean(values))
    scales: list[float] = []
    fluctuations: list[float] = []
    for size in (4, 8, 16, 32, 64):
        segment_count = profile.size // size
        if segment_count == 0:
            continue
        segments = profile[: segment_count * size].reshape(segment_count, size)
        x = np.arange(size, dtype=float)
        centered = x - x.mean()
        slopes = (segments - segments.mean(axis=1, keepdims=True)) @ centered / np.sum(centered**2)
        trends = segments.mean(axis=1, keepdims=True) + slopes[:, None] * centered
        fluctuation = float(np.sqrt(np.mean((segments - trends) ** 2)))
        if fluctuation > 0.0:
            scales.append(float(size))
            fluctuations.append(fluctuation)
    if len(scales) < 2:
        return math.nan
    return float(np.polyfit(np.log(scales), np.log(fluctuations), 1)[0])


def correlation_dimension(
    periods: np.ndarray,
    minimum: int = _NONLINEAR_MIN_PERIODS,
) -> float:
    """Two-dimensional delay-one correlation-sum slope over eight radii."""
    values = _period_series(periods, minimum)
    if values is None:
        return math.nan
    distances = pdist(np.column_stack((values[:-1], values[1:])))
    low, high = np.percentile(distances, (10, 60))
    if low <= 0.0 or high <= low:
        return math.nan
    radii = np.geomspace(low, high, 8)
    sums = np.asarray([np.mean(distances <= radius) for radius in radii])
    if np.any(sums <= 0.0):
        return math.nan
    return float(np.polyfit(np.log(radii), np.log(sums), 1)[0])


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


__all__ = [
    "advanced_features",
    "correlation_dimension",
    "detrended_fluctuation_analysis",
    "distribution_stats",
    "jitter_ppq5",
    "jitter_rap",
    "mfcc_frames",
    "pitch_period_entropy",
    "recurrence_period_density_entropy",
    "shimmer_apq",
]
