"""Phonation and prosody measures over the target speaker's clips (Task 6).

All measures reuse the Task 5 shared analysis path: the same 25 ms/10 ms
Hamming-windowed frames (:func:`speech_features.acoustic._frames`), the same
mean-energy voice activity detection (:func:`speech_features.acoustic._energy_vad`),
the same normalized-autocorrelation F0 (:func:`speech_features.acoustic._f0_per_frame`),
and the same HNR conversion (:func:`speech_features.acoustic._hnr_db`).
Only the target speaker's selected clips are analyzed (``intervals`` resolved
by :mod:`speech_features.features.acoustic.timing`; the whole recording only
in the explicit unaligned fallback), so examiner audio never affects these
measures.

Observation model and formulas (math-first; not Praat-equivalent)
----------------------------------------------------------------
- ``voice_f0_mean/median/sd/cv/iqr/range_5_95_hz`` summarize the per-frame F0
  estimates of voiced frames (VAD-voiced with a finite autocorrelation peak
  inside the configured pitch range). CV is SD/mean; the 5--95 range is
  ``p95 - p05``; SD is the population SD.
- ``voice_f0_slope_hz_per_s`` is the ordinary least-squares slope of F0 over
  the actual frame times (seconds) of the voiced frames::

      slope = sum((t - mean_t) * (f0 - mean_f0)) / sum((t - mean_t)**2)

  Frame ``i`` of a region starting at ``start_s`` is timed at
  ``start_s + i * hop_size / sample_rate``.
- ``voice_f0_abs_change_hz`` is the mean absolute difference of consecutive
  voiced F0 estimates in time order (gaps left by unvoiced frames are
  skipped).
- ``voice_voiced_ratio`` = voiced F0 frames / analyzed target frames, where
  the denominator is every frame in the target regions (including silence).
- Intensity is frame RMS converted with ``20*log10(rms)`` and summarized only
  over valid target speech frames (VAD-voiced frames, which always have
  ``rms > 0``); its slope uses their actual frame times, as above.
- Local jitter is the mean absolute consecutive period difference divided by
  the mean period, with the period estimated per voiced frame as ``1/f0``::

      jitter = mean(abs(diff(1/f0))) / mean(1/f0)

  Local shimmer is the same ratio over consecutive voiced-frame RMS
  amplitudes. Both are observation-model dependent (frame-level autocorrelation
  period and RMS amplitude, 10 ms hop) and are deliberately not labeled
  Praat-equivalent.
- ``voice_hnr_*_db`` summarize the shared ``10*log10(r/(1-r))`` conversion of
  the best normalized autocorrelation peak ``r`` per voiced frame, with the
  shared finite ceiling (``r <= 0.999``).
- ``voice_cpp_*_db`` summarize the real-cepstrum peak prominence per voiced
  frame: the peak of ``irfft(log(|rfft(frame)|))`` inside the configured
  pitch-period quefrency range (quefrency ``tau = n / sample_rate`` for
  cepstrum index ``n``, searched over ``[ceil(sr/pitch_max), floor(sr/pitch_min)]``),
  measured relative to the ordinary least-squares line fitted over that same
  local quefrency range: ``cpp = peak - line(tau_peak)``. The log magnitude
  is floored at ``1e-12`` so the cepstrum stays finite.

Sufficiency
-----------
With fewer than two voiced frames the distribution, slope, change, intensity,
HNR, and CPP measures are unavailable: affected values are float ``NaN`` and
each has a :class:`~speech_features.result.FeatureIssue` naming that exact key
with code ``INSUFFICIENT_VOICED_FRAMES``; jitter and shimmer (which need at
least one consecutive period pair) use ``INSUFFICIENT_CYCLES``. With no voiced
frame at all, ``voice_voiced_ratio`` is also ``NaN`` (digital silence never
creates a finite voice feature). No value is fabricated as zero.
"""

from __future__ import annotations

import math

import numpy as np

from ...acoustic import _energy_vad, _f0_per_frame, _frames, _hnr_db
from ...result import FeatureIssue
from ...schema import ExtractionConfig
from .definitions import PHONATION_KEYS

VOICE_F0_KEYS = (
    "voice_f0_mean_hz",
    "voice_f0_median_hz",
    "voice_f0_sd_hz",
    "voice_f0_cv",
    "voice_f0_iqr_hz",
    "voice_f0_range_5_95_hz",
    "voice_f0_slope_hz_per_s",
    "voice_f0_abs_change_hz",
)

VOICE_INTENSITY_KEYS = (
    "voice_intensity_mean_dbfs",
    "voice_intensity_median_dbfs",
    "voice_intensity_sd_db",
    "voice_intensity_iqr_db",
    "voice_intensity_slope_db_per_s",
)

VOICE_HNR_KEYS = (
    "voice_hnr_mean_db",
    "voice_hnr_median_db",
    "voice_hnr_sd_db",
    "voice_hnr_iqr_db",
)

VOICE_CPP_KEYS = (
    "voice_cpp_mean_db",
    "voice_cpp_median_db",
    "voice_cpp_sd_db",
    "voice_cpp_iqr_db",
)

_VOICED_FRAME_KEYS = VOICE_F0_KEYS + VOICE_HNR_KEYS + VOICE_CPP_KEYS + VOICE_INTENSITY_KEYS
_JITTER_KEYS = ("voice_jitter_local", "voice_shimmer_local")

_CEPSTRUM_FLOOR = 1e-12


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


def _quantile(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q))


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Ordinary least-squares slope of ``y`` over ``x``; NaN with < 2 points."""
    if x.size < 2:
        return math.nan
    centered = x - x.mean()
    denom = float(np.sum(centered**2))
    if denom == 0.0:
        return math.nan
    return float(np.sum(centered * (y - y.mean())) / denom)


def _cpp_per_frame(frames: np.ndarray, sample_rate: int, config: ExtractionConfig) -> np.ndarray:
    """Real-cepstrum peak prominence (dB) of each Hamming-windowed frame.

    See the module docstring for the quefrency mapping and baseline. ``frames``
    is the grid of *voiced* frames only; the returned array has one value per
    frame.
    """
    n_min = max(int(math.ceil(sample_rate / config.pitch_max_hz)), 1)
    n_max = min(int(math.floor(sample_rate / config.pitch_min_hz)), frames.shape[1] - 1)
    if n_max <= n_min:
        return np.full(frames.shape[0], math.nan)
    cepstrum = np.fft.irfft(
        np.log(np.maximum(np.abs(np.fft.rfft(frames, axis=1)), _CEPSTRUM_FLOOR)), axis=1
    )
    region = np.arange(n_min, n_max + 1, dtype=float)
    sub = cepstrum[:, n_min : n_max + 1]
    centered = region - region.mean()
    denom = float(np.sum(centered**2))
    slope = (sub - sub.mean(axis=1, keepdims=True)) @ centered / denom
    intercept = sub.mean(axis=1) - slope * region.mean()
    peak = np.argmax(sub, axis=1)
    baseline = slope * region[peak] + intercept
    return sub[np.arange(sub.shape[0]), peak] - baseline


def _summarize(values: np.ndarray, keys: tuple[str, ...]) -> dict[str, float]:
    return {
        keys[0]: float(np.mean(values)),
        keys[1]: float(np.median(values)),
        keys[2]: float(np.std(values)),
        keys[3]: _quantile(values, 75) - _quantile(values, 25),
    }


def phonation_features(
    audio,
    sample_rate: int,
    *,
    intervals,
    config: ExtractionConfig,
    recording_id: str,
    speaker_id: str,
    issues: list[FeatureIssue],
) -> dict[str, float]:
    """Compute the 24 ``voice_*`` keys over the target speaker's clips.

    ``intervals`` is the merged target-speaker interval list from
    :func:`speech_features.features.acoustic.timing.timing_features`; ``None``
    means the whole-recording fallback. Every key is present; unavailable
    values are ``NaN`` and paired with a per-key issue.
    """
    features = {key: math.nan for key in PHONATION_KEYS}
    duration_s = float(audio.size) / sample_rate
    regions = intervals if intervals else [(0.0, duration_s)]
    hop_s = config.hop_size / sample_rate

    f0s: list[np.ndarray] = []
    nccfs: list[np.ndarray] = []
    rms_at_f0: list[np.ndarray] = []
    times_at_f0: list[np.ndarray] = []
    voiced_frames: list[np.ndarray] = []
    speech_rms: list[np.ndarray] = []
    speech_times: list[np.ndarray] = []
    analyzed = 0

    for start, end in regions:
        clip = audio[int(round(start * sample_rate)) : int(round(end * sample_rate))]
        if clip.size == 0:
            continue
        win = _frames(clip, config.frame_size, config.hop_size)
        energy = (win**2).sum(axis=1)
        voiced = _energy_vad(energy)
        frame_times = start + np.arange(win.shape[0], dtype=float) * hop_s
        rms = np.sqrt(energy / config.frame_size)
        f0, nccf = _f0_per_frame(win, config.frame_size, sample_rate, config)
        pitchable = voiced & np.isfinite(f0)
        analyzed += win.shape[0]
        if not np.any(pitchable):
            continue
        f0s.append(f0[pitchable])
        nccfs.append(nccf[pitchable])
        rms_at_f0.append(rms[pitchable])
        times_at_f0.append(frame_times[pitchable])
        voiced_frames.append(win[pitchable])
        voiced_rms = rms[voiced]
        speech_rms.append(voiced_rms[voiced_rms > 0.0])
        speech_times.append(frame_times[voiced][voiced_rms > 0.0])

    n_voiced = sum(arr.size for arr in f0s)
    if analyzed == 0 or n_voiced == 0:
        _flag(
            features,
            _VOICED_FRAME_KEYS + ("voice_voiced_ratio",),
            "INSUFFICIENT_VOICED_FRAMES",
            "no voiced frames in target audio; phonation feature unavailable",
            recording_id,
            speaker_id,
            issues,
        )
        _flag(
            features,
            _JITTER_KEYS,
            "INSUFFICIENT_CYCLES",
            "no voiced frames in target audio; jitter/shimmer unavailable",
            recording_id,
            speaker_id,
            issues,
        )
        return features

    features["voice_voiced_ratio"] = float(n_voiced / analyzed)

    if n_voiced < 2:
        _flag(
            features,
            _VOICED_FRAME_KEYS,
            "INSUFFICIENT_VOICED_FRAMES",
            "fewer than two voiced frames; phonation feature unavailable",
            recording_id,
            speaker_id,
            issues,
        )
        _flag(
            features,
            _JITTER_KEYS,
            "INSUFFICIENT_CYCLES",
            "fewer than two voiced frames; jitter/shimmer unavailable",
            recording_id,
            speaker_id,
            issues,
        )
        return features

    f0 = np.concatenate(f0s)
    times = np.concatenate(times_at_f0)
    voiced_rms = np.concatenate(rms_at_f0)

    f0_mean = float(np.mean(f0))
    features["voice_f0_mean_hz"] = f0_mean
    features["voice_f0_median_hz"] = float(np.median(f0))
    features["voice_f0_sd_hz"] = float(np.std(f0))
    if f0_mean > 0.0:
        features["voice_f0_cv"] = float(np.std(f0)) / f0_mean
    features["voice_f0_iqr_hz"] = _quantile(f0, 75) - _quantile(f0, 25)
    features["voice_f0_range_5_95_hz"] = _quantile(f0, 95) - _quantile(f0, 5)
    features["voice_f0_slope_hz_per_s"] = _ols_slope(times, f0)
    features["voice_f0_abs_change_hz"] = float(np.mean(np.abs(np.diff(f0))))

    periods = 1.0 / f0
    features["voice_jitter_local"] = float(np.mean(np.abs(np.diff(periods))) / np.mean(periods))
    features["voice_shimmer_local"] = float(
        np.mean(np.abs(np.diff(voiced_rms))) / np.mean(voiced_rms)
    )

    hnr = _hnr_db(np.concatenate(nccfs))
    features.update(_summarize(hnr, VOICE_HNR_KEYS))

    cpp = _cpp_per_frame(np.concatenate(voiced_frames), sample_rate, config)
    features.update(_summarize(cpp, VOICE_CPP_KEYS))

    speech_rms_all = np.concatenate(speech_rms)
    speech_times_all = np.concatenate(speech_times)
    if speech_rms_all.size >= 2:
        intensity = 20.0 * np.log10(speech_rms_all)
        features["voice_intensity_mean_dbfs"] = float(np.mean(intensity))
        features["voice_intensity_median_dbfs"] = float(np.median(intensity))
        features["voice_intensity_sd_db"] = float(np.std(intensity))
        features["voice_intensity_iqr_db"] = _quantile(intensity, 75) - _quantile(intensity, 25)
        features["voice_intensity_slope_db_per_s"] = _ols_slope(speech_times_all, intensity)

    return features
