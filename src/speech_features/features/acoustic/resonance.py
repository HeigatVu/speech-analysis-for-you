"""Resonance measures: LPC formants and bandwidths (Task 7).

Convention (documented, deterministic)
--------------------------------------
Each stable voiced target frame (the same ``voiced & finite F0`` selection
the phonation module uses, i.e. VAD-voiced frames with a detected period)
is analysed with the configured ``lpc_order``:

1. The Hamming-windowed frame (shared :func:`speech_features.acoustic._frames`)
   is pre-emphasised with the standard first-order filter
   ``y[n] = x[n] - 0.97 * x[n-1]``.
2. The biased autocorrelation ``r[k] = sum(y[n] * y[n-k])`` is computed and
   the prediction coefficients ``a`` solve the normal equations
   ``Toeplitz(r[0:order]) * a[1:] = -r[1:order+1]`` (Levinson-Durbin form;
   solved with :func:`scipy.linalg.solve_toeplitz`).
3. The roots of the prediction polynomial ``1 + a[1] z^-1 + ... + a[order] z^-order``
   are extracted; each conjugate pair is taken once via its root with a
   positive angle below Nyquist (``imag > 0``). Only stable roots
   (``abs(root) < 1``) with finite values are kept; unstable or non-finite
   roots are rejected, never fabricated.
4. A root at angle ``theta`` and radius ``rho`` maps to a formant candidate
   with frequency ``f = theta * sr / (2*pi)`` and bandwidth
   ``b = -sr * log(rho) / pi`` (positive for stable roots).
5. Candidates are sorted by frequency; a frame is *accepted* only when it
   yields at least three finite stable candidates, which become
   ``F1 < F2 < F3`` with bandwidths ``B1, B2, B3`` in the same order.

No population-specific formant search ranges are applied: the three lowest
stable roots are the formants, whatever their frequencies.

Sufficiency
-----------
With no stable voiced frame at all, all twelve keys are ``NaN`` with a
per-key ``INSUFFICIENT_SPEECH_FRAMES`` issue. With fewer than two accepted
formant frames, the mean/SD summaries are unavailable: all twelve keys are
``NaN`` with per-key ``INSUFFICIENT_FORMANTS`` issues. A single accepted
frame does not characterise a distribution (matching the Task 6 threshold).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import solve_toeplitz

from ...acoustic import _energy_vad, _f0_per_frame, _frames
from ...result import FeatureIssue
from ...schema import ExtractionConfig
from .definitions import RESONANCE_KEYS

_PRE_EMPHASIS = 0.97


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


def _formant_candidates(frame: np.ndarray, sample_rate: int, order: int):
    """Stable (freq, bandwidth) candidates of one pre-emphasised frame."""
    pre = np.empty_like(frame)
    pre[0] = frame[0]
    pre[1:] = frame[1:] - _PRE_EMPHASIS * frame[:-1]
    r = np.correlate(pre, pre, mode="full")[pre.size - 1 :]
    r = r[: order + 1]
    a = np.r_[1.0, solve_toeplitz((r[:order], r[:order]), -r[1 : order + 1])]
    candidates = []
    for root in np.roots(a):
        if not np.isfinite(root):
            continue
        angle = np.angle(root)
        if angle <= 0.0 or abs(root) >= 1.0:
            continue
        freq = angle * sample_rate / (2 * math.pi)
        bandwidth = -sample_rate * math.log(abs(root)) / math.pi
        if math.isfinite(freq) and math.isfinite(bandwidth) and bandwidth > 0.0:
            candidates.append((freq, bandwidth))
    return sorted(candidates)


def _accepted_frames(
    audio,
    sample_rate: int,
    intervals,
    config: ExtractionConfig,
) -> tuple[np.ndarray, int]:
    """F1-F3/B1-B3 of every accepted frame plus the stable voiced frame count."""
    duration_s = float(audio.size) / sample_rate
    regions = intervals if intervals else [(0.0, duration_s)]
    accepted: list[np.ndarray] = []
    stable_frames = 0
    for start, end in regions:
        clip = audio[int(round(start * sample_rate)) : int(round(end * sample_rate))]
        if clip.size == 0:
            continue
        win = _frames(clip, config.frame_size, config.hop_size)
        energy = (win**2).sum(axis=1)
        voiced = _energy_vad(energy)
        f0, _ = _f0_per_frame(win, config.frame_size, sample_rate, config)
        stable = voiced & np.isfinite(f0)
        stable_frames += int(np.count_nonzero(stable))
        for frame in win[stable]:
            candidates = _formant_candidates(frame, sample_rate, config.lpc_order)
            if len(candidates) >= 3:
                accepted.append(np.asarray(candidates[:3]))
    if not accepted:
        return np.empty((0, 3, 2)), stable_frames
    return np.asarray(accepted), stable_frames


def resonance_features(
    audio,
    sample_rate: int,
    *,
    intervals,
    config: ExtractionConfig,
    recording_id: str,
    speaker_id: str,
    issues: list[FeatureIssue],
) -> dict[str, float]:
    """Compute the 12 ``spectral_f*_hz``/``spectral_b*_hz`` keys.

    ``intervals`` is the merged target-speaker interval list from
    :func:`speech_features.features.acoustic.timing.timing_features`; ``None``
    means the whole-recording fallback. Every key is present; unavailable
    values are ``NaN`` and paired with a per-key issue.
    """
    features = {key: math.nan for key in RESONANCE_KEYS}
    accepted, stable_frames = _accepted_frames(audio, sample_rate, intervals, config)

    if stable_frames == 0:
        for key in RESONANCE_KEYS:
            issues.append(
                _issue(
                    recording_id,
                    speaker_id,
                    "INSUFFICIENT_SPEECH_FRAMES",
                    "no stable voiced frames in target audio; formants unavailable",
                    feature=key,
                )
            )
        return features

    if accepted.shape[0] < 2:
        for key in RESONANCE_KEYS:
            issues.append(
                _issue(
                    recording_id,
                    speaker_id,
                    "INSUFFICIENT_FORMANTS",
                    "fewer than two frames with three stable formants; "
                    "formant summaries unavailable",
                    feature=key,
                )
            )
        return features

    for index, key in enumerate(("spectral_f1", "spectral_f2", "spectral_f3")):
        values = accepted[:, index, 0]
        features[f"{key}_mean_hz"] = float(np.mean(values))
        features[f"{key}_sd_hz"] = float(np.std(values))
    for index, key in enumerate(("spectral_b1", "spectral_b2", "spectral_b3")):
        values = accepted[:, index, 1]
        features[f"{key}_mean_hz"] = float(np.mean(values))
        features[f"{key}_sd_hz"] = float(np.std(values))
    return features
