"""Math-first acoustic features for mono PCM float arrays (Task 2).

NumPy/SciPy only. No WAV I/O, no labels, no ASR, no librosa/openSMILE. All
features are deterministic over a mono float array + sample rate. Invalid
denominators or too-few observations return ``NaN`` (never zero) together with
a quality flag, matching the plan's contract.

The public entry point :func:`extract_acoustic` returns an :class:`AcousticResult`
carrying a plain numeric ``features`` mapping and a tuple of ``flags``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from .schema import ExtractionConfig

# Minimum usable duration before we label a recording too short to characterise.
MIN_DURATION_S = 1.0

# Absolute silence floor on frame energy. Frames below this are treated as
# silence regardless of the observed dynamic range, so a very quiet recording
# (ambient or digital noise far below the floor) reads as no-voice rather than
# spuriously voiced.
SILENCE_ENERGY_FLOOR = 1e-6

# Ceiling used when converting a normalized autocorrelation into an HNR (dB)
# so that near-perfect periodicity does not blow up to infinity.
_HNR_NCCF_CEILING = 0.999


@dataclass(frozen=True)
class AcousticResult:
    """Numeric acoustic features plus quality flags for one mono recording.

    ``features`` is an immutable mapping of ``ac_*`` floats; ``flags`` is a
    tuple of strings drawn from ``{"invalid_input", "too_short", "no_voice"}``.
    """

    features: dict = field(default_factory=dict)
    flags: tuple = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))
        object.__setattr__(self, "flags", tuple(self.flags))


def _frames(audio: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    """Reshape 1-D audio into an ``(n_frames, frame_size)`` Hamming-windowed grid.

    Zero-pad the tail so the final partial frame is used rather than dropped;
    every sample influences exactly one windowed frame.
    """
    n_frames = int(np.ceil((len(audio) - frame_size) / hop_size)) + 1
    n_samples = (n_frames - 1) * hop_size + frame_size
    padded = np.zeros(n_samples, dtype=float)
    padded[: len(audio)] = audio
    index = np.arange(frame_size)[None, :] + hop_size * np.arange(n_frames)[:, None]
    frames = padded[index]
    return frames * np.hamming(frame_size)  # Hamming-weighted frames per the plan


def _energy_vad(frame_energy: np.ndarray) -> np.ndarray:
    """Adaptive frame-level voice activity from energy.

    A frame is voiced only when it (a) clears the absolute silence floor and
    (b) clears the dynamic range threshold (the mean of the above-floor
    frames). Combining the two means a genuinely present tone is voiced while
    quiet noise or near-silent audio — even with large dynamic range relative
    to itself — is correctly classified as no-voice.
    """
    floor = SILENCE_ENERGY_FLOOR
    active = frame_energy[frame_energy > floor]
    if active.size == 0:
        return np.zeros(frame_energy.shape, dtype=bool)
    dynamic_threshold = active.mean()
    return (frame_energy > floor) & (frame_energy >= dynamic_threshold)


def _pauses(voiced: np.ndarray, hop_s: float, pause_threshold_s: float) -> np.ndarray:
    """Return durations (seconds) of maximal non-speech runs >= the threshold.

    Runs are counted in integer frame counts so that a silence whose duration
    lands exactly on the threshold is never dropped by the float accumulation
    of ``hop_s``; the count is converted to seconds once, at the end.
    """
    min_frames = math.ceil(pause_threshold_s / hop_s)
    runs: list[float] = []
    gap = 0
    for is_voiced in voiced:
        if not is_voiced:
            gap += 1
        else:
            if gap >= min_frames:
                runs.append(gap * hop_s)
            gap = 0
    if gap >= min_frames:
        runs.append(gap * hop_s)
    return np.asarray(runs, dtype=float)


def _f0_per_frame(
    win: np.ndarray, frame_size: int, sr: int, config: ExtractionConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame F0 (Hz) and best NCCF; NaN/0 where unvoiced.

    For each Hamming-windowed frame we compute the normalized cross-correlation
    coefficient (NCCF) at every lag whose period lies inside
    ``[pitch_min_hz, pitch_max_hz]``. NCCF mean-subtracts each overlapped
    segment and length-normalizes it, so a periodic voiced frame scores near 1
    while power-only (unvoiced) frames stay far below the threshold. The lag
    with the largest NCCF is the frame's period; if that peak is below
    ``pitch_autocorr_threshold`` the frame is explicitly unvoiced (``NaN``).

    Bounds use ``ceil`` for the shortest lag (so lags whose period would exceed
    ``pitch_max_hz`` are never admitted) and ``floor`` for the longest lag (so
    the configured ``pitch_min_hz`` is not undercut), matching the plan's pitch
    range.

    Returns ``(f0_hz, best_nccf)`` where ``best_nccf`` is 0 for unvoiced frames.
    """
    min_lag = max(int(math.ceil(sr / config.pitch_max_hz)), 1)
    max_lag = int(math.floor(sr / config.pitch_min_hz))
    max_lag = min(max_lag, frame_size - 1)
    if max_lag <= min_lag:
        n = win.shape[0]
        return np.full(n, np.nan), np.zeros(n)

    out = np.full(win.shape[0], np.nan)
    best_dot = np.zeros(win.shape[0])
    for lag in range(min_lag, max_lag + 1):
        head = win[:, :-lag]
        tail = win[:, lag:]
        head = head - head.mean(axis=1, keepdims=True)
        tail = tail - tail.mean(axis=1, keepdims=True)
        denom = np.sqrt((head**2).sum(axis=1) * (tail**2).sum(axis=1))
        denom = np.maximum(denom, 1e-12)
        nccf = (head * tail).sum(axis=1) / denom
        better = nccf > best_dot
        best_dot = np.where(better, nccf, best_dot)
        out = np.where(better, sr / lag, out)
    voiced = best_dot >= config.pitch_autocorr_threshold
    return np.where(voiced, out, np.nan), np.where(voiced, best_dot, 0.0)


def _spectral(frames: np.ndarray, sr: int) -> dict[str, float]:
    """Mean spectral centroid, spread (bandwidth), and flatness over frames."""
    spectrum = np.abs(np.fft.rfft(frames, axis=1))
    power = spectrum**2
    freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / sr)
    total = power.sum(axis=1)
    safe_total = np.maximum(total, 1e-12)

    centroid = (power * freqs[None, :]).sum(axis=1) / safe_total
    spread = np.sqrt((power * (freqs[None, :] - centroid[:, None]) ** 2).sum(axis=1) / safe_total)
    # Spectral flatness is the ratio of geometric to arithmetic power, bounded
    # in [0, 1]: near 1 for white noise, near 0 for a tonal/spiky spectrum.
    # Adding a tiny floor to every bin keeps the geometric mean finite without
    # materially distorting the ratio on well-conditioned spectra.
    sm = power.mean(axis=1)
    flatness = np.exp(np.log(power + 1e-12).mean(axis=1)) / np.maximum(sm, 1e-12)

    def _mean(arr):
        vals = arr[np.isfinite(arr)]
        return float(np.mean(vals)) if vals.size else math.nan

    return {
        "ac_spectral_centroid_mean": _mean(centroid),
        "ac_spectral_spread_mean": _mean(spread),
        "ac_spectral_flatness_mean": _mean(flatness),
    }


_MISSING = {
    "ac_frame_energy_mean": math.nan,
    "ac_frame_energy_sd": math.nan,
    "ac_frame_energy_iqr": math.nan,
    "ac_frame_energy_span": math.nan,
    "ac_pitch_voiced_mean": math.nan,
    "ac_pitch_voiced_sd": math.nan,
    "ac_pitch_voiced_cv": math.nan,
    "ac_pitch_voiced_median": math.nan,
    "ac_pitch_voiced_iqr": math.nan,
    "ac_pitch_voiced_span": math.nan,
    "ac_pitch_voiced_delta": math.nan,
    "ac_hnr_median": math.nan,
    "ac_hnr_iqr": math.nan,
    "ac_voice_ratio": math.nan,
    "ac_pause_count": math.nan,
    "ac_pause_rate_per_min": math.nan,
    "ac_pause_mean_s": math.nan,
    "ac_pause_sd_s": math.nan,
    "ac_pause_max_s": math.nan,
    "ac_long_pause_count": math.nan,
    "ac_spectral_centroid_mean": math.nan,
    "ac_spectral_spread_mean": math.nan,
    "ac_spectral_flatness_mean": math.nan,
}


def _quantile(arr: np.ndarray, q: float) -> float:
    """Robust percentile of a 1-D array; NaN when empty."""
    if arr.size == 0:
        return math.nan
    return float(np.percentile(arr, q))


def _hnr_db(best_nccf: np.ndarray) -> np.ndarray:
    """Harmonic-to-noise ratio (dB) from normalized autocorrelation peaks.

    ``HNR = 10*log10(r / (1 - r))`` for the best NCCF ``r`` per voiced frame.
    ``r`` is clamped below the ceiling so near-perfect periodicity stays finite
    instead of dividing by zero.
    """
    r = np.clip(best_nccf, 0.0, _HNR_NCCF_CEILING)
    return 10.0 * np.log10(np.maximum(r, 1e-12) / (1.0 - r))


def extract_acoustic(
    audio, sample_rate: int, *, config: ExtractionConfig | None = None
) -> AcousticResult:
    """Extract numeric acoustic features from a mono PCM float array.

    Parameters
    ----------
    audio : array-like
        Mono samples in the range roughly [-1, 1]. No I/O or resampling is
        performed here.
    sample_rate : int
        Sampling rate in Hz; must be a positive integer.
    config : ExtractionConfig, optional
        Overrides the plan's defaults (frame/hop size, pitch bounds, pause
        thresholds).

    Returns
    -------
    AcousticResult
        Plain numeric features plus quality flags. Invalid input, audio too
        short to characterise, or no voiced frames are flagged rather than
        raised, and unavailable statistics are ``NaN``.
    """
    cfg = config if config is not None else ExtractionConfig()
    result = {k: math.nan for k in _MISSING}

    try:
        arr = np.asarray(audio, dtype=float)
        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(float)
        invalid = sample_rate <= 0 or arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr))
    except (ValueError, TypeError):
        invalid = True

    if invalid:
        return AcousticResult(result, ("invalid_input",))

    duration = arr.size / sample_rate
    if duration < MIN_DURATION_S:
        return AcousticResult(result, ("too_short",))

    win = _frames(arr, cfg.frame_size, cfg.hop_size)
    frame_energy = (win**2).sum(axis=1)
    voiced = _energy_vad(frame_energy)

    features = dict(_MISSING)
    features["ac_frame_energy_mean"] = float(np.mean(frame_energy))
    features["ac_frame_energy_sd"] = float(np.std(frame_energy))
    features["ac_frame_energy_iqr"] = _quantile(frame_energy, 75) - _quantile(frame_energy, 25)
    features["ac_frame_energy_span"] = _quantile(frame_energy, 95) - _quantile(frame_energy, 5)

    spectral = _spectral(win, sample_rate)
    features.update(spectral)

    if not np.any(voiced):
        return AcousticResult(features, ("no_voice",))

    f0, best_nccf = _f0_per_frame(win, cfg.frame_size, sample_rate, cfg)
    pitched = f0[voiced]
    pitched_nccf = best_nccf[voiced]
    voiced_f0 = pitched[np.isfinite(pitched)]
    voiced_nccf = pitched_nccf[np.isfinite(pitched)]

    if voiced_f0.size == 0:
        return AcousticResult(features, ("no_voice",))

    voice_ratio = float(np.count_nonzero(voiced) / voiced.size)
    features["ac_voice_ratio"] = voice_ratio
    features["ac_pitch_voiced_mean"] = float(np.mean(voiced_f0))
    features["ac_pitch_voiced_sd"] = float(np.std(voiced_f0))
    mean = features["ac_pitch_voiced_mean"]
    sd = features["ac_pitch_voiced_sd"]
    # Guard with an explicit finite check rather than boolean truthiness of a
    # mean that could be NaN (NaN is truthy, which would otherwise divide by it).
    if math.isfinite(mean) and mean > 0.0:
        features["ac_pitch_voiced_cv"] = sd / mean
    features["ac_pitch_voiced_median"] = float(np.median(voiced_f0))
    features["ac_pitch_voiced_iqr"] = _quantile(voiced_f0, 75) - _quantile(voiced_f0, 25)
    features["ac_pitch_voiced_span"] = _quantile(voiced_f0, 95) - _quantile(voiced_f0, 5)
    features["ac_pitch_voiced_delta"] = float(np.max(voiced_f0) - np.min(voiced_f0))

    hnr = _hnr_db(voiced_nccf)
    features["ac_hnr_median"] = float(np.median(hnr)) if hnr.size else math.nan
    features["ac_hnr_iqr"] = _quantile(hnr, 75) - _quantile(hnr, 25)

    hop_s = cfg.hop_size / sample_rate
    pauses = _pauses(voiced, hop_s, cfg.pause_threshold_s)
    features["ac_pause_count"] = float(pauses.size)
    duration_min = max(duration / 60.0, 1e-9)
    features["ac_pause_rate_per_min"] = float(pauses.size / duration_min)
    features["ac_pause_mean_s"] = float(np.mean(pauses)) if pauses.size else math.nan
    features["ac_pause_sd_s"] = float(np.std(pauses)) if pauses.size else math.nan
    features["ac_pause_max_s"] = float(np.max(pauses)) if pauses.size else math.nan
    features["ac_long_pause_count"] = float(np.count_nonzero(pauses >= cfg.long_pause_threshold_s))

    return AcousticResult(features, ())
