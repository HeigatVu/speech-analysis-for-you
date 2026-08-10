"""Spectral summary measures over the target speaker's clips (Task 7).

All measures operate on the VAD-voiced frames of the target intervals (the
same "target speech frames" the intensity summaries use), using the shared
25 ms/10 ms Hamming-windowed framing and mean-energy VAD, so examiner audio
never contributes. Power choices are explicit: the spectrum is the power
spectrum ``P = |rfft(frame)|**2`` over the real-FFT bins with frequencies
``f`` in Hz.

Per-frame measures (each summarized by mean and population SD over frames)
----------------------------------------------------------------------------
- ``spectral_centroid_mean/sd_hz``: power-weighted mean frequency
  ``sum(f * P) / sum(P)``.
- ``spectral_spread_mean/sd_hz``: power-weighted RMS deviation
  ``sqrt(sum(P * (f - centroid)**2) / sum(P))``.
- ``spectral_slope_mean/sd_db_per_hz``: ordinary least-squares slope of the
  dB power spectrum ``10*log10(P)`` (floored at ``1e-12`` so silent bins stay
  finite) over frequency in Hz: ``slope = sum((f - mean_f) * (db - mean_db))
  / sum((f - mean_f)**2)``.
- ``spectral_rolloff_85_mean/sd_hz``: frequency of the first bin whose
  cumulative power reaches 85% of the frame's total power.
- ``spectral_flux_mean/sd``: L2 norm of the difference between consecutively
  normalized (unit-L2-norm) power spectra. Pairs are formed inside each
  target interval only and pooled across intervals; the last frame of one
  interval and the first frame of the next are never paired.
- ``spectral_flatness_mean/sd``: geometric / arithmetic mean power
  ``exp(mean(log(P + 1e-12))) / mean(P + 1e-12)``, bounded in [0, 1].
- ``spectral_entropy_mean/sd``: normalized Shannon entropy of the power
  distribution ``p = P / sum(P)``: ``-sum(p * log(p)) / log(n_bins)``,
  bounded in [0, 1].

Sufficiency
-----------
With fewer than two voiced frames the summaries are unavailable: affected
values are float ``NaN`` with one ``INSUFFICIENT_SPEECH_FRAMES`` issue per
key. With two or more voiced frames but no interval holding two consecutive
voiced frames, the flux keys have no valid intra-interval transition: they
are ``NaN`` with per-key ``INSUFFICIENT_SPEECH_FRAMES`` issues. No value is
fabricated as zero.
"""

from __future__ import annotations

import math

import numpy as np

from ...acoustic import _energy_vad, _frames
from ...result import FeatureIssue
from ...schema import ExtractionConfig
from .definitions import SPECTRUM_KEYS

_LOG_FLOOR = 1e-12
_EXPANDED_KEYS = {
    "spectral_energy_mean_db",
    "spectral_energy_sd_db",
    "spectral_skewness_mean",
    "spectral_skewness_sd",
    "spectral_kurtosis_mean",
    "spectral_kurtosis_sd",
    "spectral_low_high_energy_ratio_db",
}
BASIC_SPECTRUM_KEYS = tuple(
    key
    for key in SPECTRUM_KEYS
    if key not in _EXPANDED_KEYS and not key.startswith("spectral_mfcc_")
)


def _issue(
    recording_id: str, speaker_id: str, code: str, message: str, feature: str
) -> FeatureIssue:
    return FeatureIssue(
        recording_id=recording_id,
        speaker_id=speaker_id,
        code=code,
        severity="warning",
        message=message,
        feature=feature,
    )


def _flag(
    features: dict[str, float],
    keys,
    code: str,
    message: str,
    recording_id: str,
    speaker_id: str,
    issues: list[FeatureIssue],
) -> None:
    for key in keys:
        features[key] = math.nan
        issues.append(_issue(recording_id, speaker_id, code, message, feature=key))


def _speech_frame_regions(audio, sample_rate: int, intervals, config: ExtractionConfig):
    """VAD-voiced frame arrays, kept separate at target-interval boundaries."""
    duration_s = float(audio.size) / sample_rate
    regions = intervals if intervals else [(0.0, duration_s)]
    selected: list[np.ndarray] = []
    for start, end in regions:
        clip = audio[int(round(start * sample_rate)) : int(round(end * sample_rate))]
        if clip.size == 0:
            continue
        frames = _frames(clip, config.frame_size, config.hop_size)
        voiced = _energy_vad((frames**2).sum(axis=1))
        if np.any(voiced):
            selected.append(frames[voiced])
    return selected


def _spectrum_features(audio, sample_rate: int, intervals, config: ExtractionConfig):
    """Per-frame spectral measures and intra-interval flux over target frames.

    Returns ``(values, fluxes)`` where ``values`` is a dict of per-frame
    arrays (centroid, spread, slope, rolloff, flatness, entropy) and
    ``fluxes`` is the list of intra-interval flux arrays.
    """
    freqs = np.fft.rfftfreq(config.frame_size, d=1.0 / sample_rate)
    n_bins = freqs.size
    centroid_all: list[np.ndarray] = []
    spread_all: list[np.ndarray] = []
    slope_all: list[np.ndarray] = []
    rolloff_all: list[np.ndarray] = []
    flatness_all: list[np.ndarray] = []
    entropy_all: list[np.ndarray] = []
    fluxes: list[np.ndarray] = []

    for sub in _speech_frame_regions(audio, sample_rate, intervals, config):
        power = np.abs(np.fft.rfft(sub, axis=1)) ** 2
        total = np.maximum(power.sum(axis=1), 1e-12)
        centroid = (power * freqs[None, :]).sum(axis=1) / total
        spread = np.sqrt((power * (freqs[None, :] - centroid[:, None]) ** 2).sum(axis=1) / total)
        db = 10.0 * np.log10(power + _LOG_FLOOR)
        centered = freqs - freqs.mean()
        slope = (centered * (db - db.mean(axis=1, keepdims=True))).sum(axis=1) / (centered**2).sum()
        cumulative = np.cumsum(power, axis=1) / total[:, None]
        rolloff = freqs[np.argmax(cumulative >= 0.85, axis=1)]
        flatness = np.exp(np.log(power + _LOG_FLOOR).mean(axis=1)) / (power + _LOG_FLOOR).mean(
            axis=1
        )
        prob = power / total[:, None]
        entropy = -(prob * np.log(np.maximum(prob, 1e-300))).sum(axis=1) / math.log(n_bins)

        centroid_all.append(centroid)
        spread_all.append(spread)
        slope_all.append(slope)
        rolloff_all.append(rolloff)
        flatness_all.append(flatness)
        entropy_all.append(entropy)

        if sub.shape[0] > 1:
            normalized = power / np.maximum(np.linalg.norm(power, axis=1, keepdims=True), 1e-12)
            fluxes.append(np.linalg.norm(normalized[1:] - normalized[:-1], axis=1))

    values = {
        "centroid": centroid_all,
        "spread": spread_all,
        "slope": slope_all,
        "rolloff": rolloff_all,
        "flatness": flatness_all,
        "entropy": entropy_all,
    }
    return values, fluxes


def spectrum_features(
    audio,
    sample_rate: int,
    *,
    intervals,
    config: ExtractionConfig,
    recording_id: str,
    speaker_id: str,
    issues: list[FeatureIssue],
) -> dict[str, float]:
    """Compute the 14 ``spectral_*`` keys over the target speaker's clips.

    ``intervals`` is the merged target-speaker interval list from
    :func:`speech_features.features.acoustic.timing.timing_features`; ``None``
    means the whole-recording fallback. Every key is present; unavailable
    values are ``NaN`` and paired with a per-key issue.
    """
    features = {key: math.nan for key in BASIC_SPECTRUM_KEYS}
    values, fluxes = _spectrum_features(audio, sample_rate, intervals, config)
    counts = {name: sum(arr.size for arr in arrays) for name, arrays in values.items()}
    n_frames = counts["centroid"]

    if n_frames < 2:
        _flag(
            features,
            BASIC_SPECTRUM_KEYS,
            "INSUFFICIENT_SPEECH_FRAMES",
            "fewer than two voiced frames in target audio; spectral feature unavailable",
            recording_id,
            speaker_id,
            issues,
        )
        return features

    for name, key, suffix in (
        ("centroid", "spectral_centroid", "_mean_hz"),
        ("spread", "spectral_spread", "_mean_hz"),
        ("slope", "spectral_slope", "_mean_db_per_hz"),
        ("rolloff", "spectral_rolloff_85", "_mean_hz"),
        ("flatness", "spectral_flatness", "_mean"),
        ("entropy", "spectral_entropy", "_mean"),
    ):
        arrays = values[name]
        pooled = np.concatenate(arrays)
        features[key + suffix] = float(np.mean(pooled))
        sd_suffix = suffix.replace("_mean", "_sd")
        features[key + sd_suffix] = float(np.std(pooled))

    flux_all = np.concatenate(fluxes) if fluxes else np.empty(0)
    if flux_all.size == 0:
        _flag(
            features,
            ("spectral_flux_mean", "spectral_flux_sd"),
            "INSUFFICIENT_SPEECH_FRAMES",
            "no consecutive voiced frames within one target interval; flux unavailable",
            recording_id,
            speaker_id,
            issues,
        )
    else:
        features["spectral_flux_mean"] = float(np.mean(flux_all))
        features["spectral_flux_sd"] = float(np.std(flux_all))
    return features
