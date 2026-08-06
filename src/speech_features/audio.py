"""Population-neutral PCM WAV core (Task 11).

Shared standard-PCM WAV decoding shared by the label-free extraction pipeline,
the acoustic pack, and the legacy AD pipeline. Audio is read with the stdlib
``wave`` module (standard PCM only), converted from mono/stereo to a finite
mono float array, and resampled to the target rate with
``scipy.signal.resample_poly``.

No labels, no AD concepts, and no extraction logic here.
"""

from __future__ import annotations

import math
import wave

import numpy as np
from scipy.signal import resample_poly

from .schema import FeatureExtractionError

_SCALE_BY_WIDTH = {2: 32768.0, 3: 8388608.0, 4: 2147483648.0}


class InvalidAudioError(FeatureExtractionError):
    """Raised when a WAV file cannot be read as standard PCM mono/stereo."""

    code = "INVALID_AUDIO"


class UnsupportedAudioError(FeatureExtractionError):
    """Raised when a WAV is not standard PCM mono/stereo media."""

    code = "UNSUPPORTED_AUDIO"


def _bytes_to_samples(raw: bytes, width: int, count: int) -> np.ndarray:
    """Decode interleaved PCM integers (``width`` bytes each) as int64."""
    if width == 1:
        return np.frombuffer(raw, dtype=np.uint8).astype(np.int64)[:count]
    if width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.int64)[:count]
    if width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.int64)[:count]
    # 24-bit signed PCM has no native width-3 dtype; assemble each sample from
    # three little-endian bytes with vectorised NumPy ops (no per-sample loop).
    n_bytes = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
    if n_bytes.size % 3 != 0:
        raise ValueError(f"truncated 24-bit PCM buffer ({n_bytes.size} bytes, not a multiple of 3)")
    three = n_bytes[: count * 3].reshape(-1, 3)
    lo, mid, hi = three[:, 0], three[:, 1], three[:, 2]
    u = lo | (mid << 8) | (hi << 16)
    return np.where(u & 0x800000, u - 0x1_000000, u)


def _to_float(samples: np.ndarray, width: int) -> np.ndarray:
    """Map PCM integers to the float domain ``[-1, 1]`` (8-bit is unsigned)."""
    if width == 1:
        return (samples - 128.0) / 128.0
    return samples / _SCALE_BY_WIDTH[width]


def _resample(samples: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    g = math.gcd(src_sr, target_sr)
    return resample_poly(samples, up=target_sr // g, down=src_sr // g)


def _read_wav_with_width(path, sample_rate: int = 16000) -> tuple[np.ndarray, int]:
    """Read a standard PCM WAV as a mono float array, returning its width too.

    Accepts mono or stereo integer PCM (8/16/24/32-bit, standard ``wave``
    formats). Stereo is downmixed to mono by the channel mean; the resulting
    samples are resampled to ``sample_rate`` with ``resample_poly``. At least
    finite, non-empty audio is guaranteed; malformed files raise
    :class:`InvalidAudioError` and unsupported media (non-PCM, non-standard
    width or channel count) raises :class:`UnsupportedAudioError`.
    """
    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            src_sr = wf.getframerate()
            comptype = wf.getcomptype()
            n_frames = wf.getnframes()
            raw = wf.readframes(int(n_frames))
    except (wave.Error, OSError, EOFError) as exc:
        # The stdlib wave reader rejects non-PCM format codes at open time
        # ("unknown format: <code>"); those files are unsupported media, not
        # malformed PCM, so they surface the stable UNSUPPORTED_AUDIO code.
        if isinstance(exc, wave.Error) and "unknown format" in str(exc):
            raise UnsupportedAudioError(f"unsupported WAV (not PCM): {exc}") from exc
        raise InvalidAudioError(f"cannot read WAV {path}: {exc}") from exc

    if comptype not in (b"NONE", "NONE", None):
        raise UnsupportedAudioError(f"unsupported WAV (not PCM), compression: {comptype!r}")
    if width not in (1, 2, 3, 4):
        raise UnsupportedAudioError(f"unsupported sample width in WAV: {width} bytes")
    if channels not in (1, 2):
        raise UnsupportedAudioError(f"unsupported channel count in WAV: {channels}")
    if src_sr <= 0 or n_frames <= 0:
        raise InvalidAudioError("WAV is empty or has an invalid sample rate")

    try:
        count = n_frames * channels
        samples = _to_float(_bytes_to_samples(raw, width, count), width)
        if channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)
        if not np.all(np.isfinite(samples)):
            raise InvalidAudioError("WAV contains non-finite samples")
        if src_sr != sample_rate:
            samples = _resample(samples, src_sr, sample_rate)
            # resample_poly is a polynomial all-pole path; still guard the output
            # against any accidental non-finite introduction at the boundary.
            if not np.all(np.isfinite(samples)):
                raise InvalidAudioError("resampled WAV contains non-finite samples")
        return samples, width
    except InvalidAudioError:
        raise
    except (ValueError, IndexError, TypeError) as exc:
        raise InvalidAudioError(f"cannot decode WAV {path}: {exc}") from exc


def read_wav(path, *, sample_rate: int = 16000) -> np.ndarray:
    """Read a standard PCM WAV as a mono float array at ``sample_rate``.

    Thin wrapper over :func:`_read_wav_with_width` returning only the samples;
    see its docstring for the supported formats and raised errors.
    """
    samples, _ = _read_wav_with_width(path, sample_rate=sample_rate)
    return samples


__all__ = [
    "InvalidAudioError",
    "UnsupportedAudioError",
    "read_wav",
]
