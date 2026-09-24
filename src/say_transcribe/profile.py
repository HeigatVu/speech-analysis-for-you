"""P0 preprocessing profile: narrowband GSM-FR round trip + two-pass EBU R128 loudnorm.

Signal chain (SPEC "Preprocessing profile"): band-pass 200 Hz-3.4 kHz (zero-phase)
-> 8 kHz -> FFmpeg ``libgsm`` round trip (decode padding trimmed to the exact
input sample count) -> 16 kHz -> FFmpeg ``loudnorm`` two-pass, ``linear=true``,
I=-23, TP=-1, LRA=50. All intermediates stay in memory or in pipes; nothing is
written to disk. The master audio is never touched here.

For any input the output sample count is the duration-ideal count
``(len(input) * PROFILE_RATE / sample_rate)`` rounded to the nearest sample —
exactly the input count for 16 kHz input — and the GSM frame padding never
leaks past the trim.
"""

from dataclasses import dataclass
import json
import math
import shutil
import subprocess

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

BANDPASS_HZ = (200.0, 3400.0)
GSM_RATE = 8000
PROFILE_RATE = 16000
LOUDNORM_FILTER = "loudnorm=I=-23:TP=-1:LRA=50"


class ProfileError(Exception):
    """A preprocessing profile stage failed under a stable, redacted error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class LoudnessReport:
    input_i: float
    input_tp: float
    input_lra: float
    output_i: float
    output_tp: float
    output_lra: float
    normalization_type: str  # "linear" | "dynamic"; dynamic means the fallback fired


@dataclass(frozen=True)
class ProfileResult:
    samples: np.ndarray  # float32 mono at PROFILE_RATE
    sample_rate: int
    loudness: LoudnessReport


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise ProfileError("CODEC_UNAVAILABLE", "ffmpeg binary is not available")


def _run_ffmpeg(args: list[str], stdin: bytes, error_code: str) -> tuple[bytes, str]:
    # Do not pass -loglevel below the default info level: loudnorm prints its
    # JSON stats at info level and they are the pass-1/2 measurement source.
    try:
        process = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", *args],
            input=stdin,
            capture_output=True,
            check=False,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        raise ProfileError(error_code, "ffmpeg stage timed out") from None
    except OSError:
        raise ProfileError(error_code, "ffmpeg stage failed") from None
    if process.returncode != 0:
        raise ProfileError(error_code, "ffmpeg stage failed")
    return process.stdout, process.stderr.decode("utf-8", errors="replace")


def _to_float(mono_samples: np.ndarray, sample_width: int) -> np.ndarray:
    if sample_width == 2:
        scale = 32768.0
    elif sample_width == 4:
        scale = 2147483648.0
    else:
        raise ProfileError("INVALID_AUDIO", "unsupported sample width")
    samples = mono_samples.astype(np.float64) / scale
    if not np.isfinite(samples).all():
        raise ProfileError("INVALID_AUDIO", "input contains non-finite samples")
    return samples


def _to_s16le(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()


def bandpass_narrowband(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Zero-phase 200 Hz-3.4 kHz band-pass emulating narrowband transmission."""
    if sample_rate <= 2 * BANDPASS_HZ[1]:
        raise ProfileError("INVALID_AUDIO", "sample rate below narrowband envelope")
    sos = butter(4, BANDPASS_HZ, btype="band", fs=sample_rate, output="sos")
    try:
        return sosfiltfilt(sos, samples)
    except ValueError:
        raise ProfileError("INVALID_AUDIO", "signal too short for band-pass filtering") from None


def gsm_roundtrip(samples_8k: np.ndarray) -> np.ndarray:
    """Encode/decode through FFmpeg libgsm at 8 kHz; trim frame padding to input length."""
    _require_ffmpeg()
    pcm_in = _to_s16le(samples_8k)
    encoded, _ = _run_ffmpeg(
        ["-f", "s16le", "-ar", str(GSM_RATE), "-ac", "1", "-i", "pipe:0",
         "-c:a", "libgsm", "-f", "gsm", "pipe:1"],
        pcm_in,
        "CODEC_UNAVAILABLE",
    )
    decoded, _ = _run_ffmpeg(
        ["-f", "gsm", "-ar", str(GSM_RATE), "-ac", "1", "-i", "pipe:0",
         "-f", "s16le", "-ar", str(GSM_RATE), "-ac", "1", "pipe:1"],
        encoded,
        "CODEC_UNAVAILABLE",
    )
    decoded_samples = np.frombuffer(decoded, dtype="<i2")
    if len(decoded_samples) < len(pcm_in) // 2:
        raise ProfileError("CODEC_UNAVAILABLE", "GSM round trip returned fewer samples than sent")
    trimmed = decoded_samples[: len(samples_8k)]
    return trimmed.astype(np.float64) / 32768.0


def _parse_loudnorm_json(stderr_text: str) -> dict | None:
    start = stderr_text.rfind("{")
    end = stderr_text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(stderr_text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _finite(stats: dict, *keys: str) -> bool:
    for key in keys:
        try:
            value = float(stats[key])
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isfinite(value):
            return False
    return True


def loudnorm_two_pass(samples_16k: np.ndarray) -> tuple[np.ndarray, LoudnessReport]:
    """Two-pass EBU R128 loudnorm; records measurements and flags dynamic fallback."""
    _require_ffmpeg()
    pcm_in = _to_s16le(samples_16k)
    _, pass1_stderr = _run_ffmpeg(
        ["-f", "s16le", "-ar", str(PROFILE_RATE), "-ac", "1", "-i", "pipe:0",
         "-af", f"{LOUDNORM_FILTER}:print_format=json", "-f", "null", "-"],
        pcm_in,
        "CODEC_UNAVAILABLE",
    )
    measured = _parse_loudnorm_json(pass1_stderr)
    if measured is None:
        raise ProfileError("LOUDNORM_FAILED", "loudnorm pass-1 stats could not be parsed")
    if not _finite(measured, "input_i", "input_tp", "input_lra", "input_thresh", "target_offset"):
        raise ProfileError("LOUDNORM_FAILED", "loudnorm measured stats are unusable")

    applied_filter = (
        f"{LOUDNORM_FILTER}:linear=true"
        f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
        f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
        f":offset={measured['target_offset']}:print_format=json"
    )
    pcm_out, pass2_stderr = _run_ffmpeg(
        ["-f", "s16le", "-ar", str(PROFILE_RATE), "-ac", "1", "-i", "pipe:0",
         "-af", applied_filter, "-f", "s16le", "-ar", str(PROFILE_RATE), "-ac", "1", "pipe:1"],
        pcm_in,
        "CODEC_UNAVAILABLE",
    )
    applied = _parse_loudnorm_json(pass2_stderr)
    if applied is None:
        raise ProfileError("LOUDNORM_FAILED", "loudnorm pass-2 stats could not be parsed")
    if not _finite(applied, "output_i", "output_tp", "output_lra"):
        raise ProfileError("LOUDNORM_FAILED", "loudnorm output stats are unusable")
    if applied.get("normalization_type") not in ("linear", "dynamic"):
        raise ProfileError("LOUDNORM_FAILED", "loudnorm normalization type is unusable")

    out_samples = np.frombuffer(pcm_out, dtype="<i2")
    if len(out_samples) != len(samples_16k):
        raise ProfileError("LOUDNORM_FAILED", "loudnorm changed the sample count")

    report = LoudnessReport(
        input_i=float(measured["input_i"]),
        input_tp=float(measured["input_tp"]),
        input_lra=float(measured["input_lra"]),
        output_i=float(applied["output_i"]),
        output_tp=float(applied["output_tp"]),
        output_lra=float(applied["output_lra"]),
        normalization_type=str(applied["normalization_type"]),
    )
    return out_samples.astype(np.float64) / 32768.0, report


def apply_p0_profile(mono_samples: np.ndarray, sample_rate: int, sample_width: int) -> ProfileResult:
    """Run the full P0 profile on a channel-selected mono signal.

    Output length is ``(len(input) * PROFILE_RATE / sample_rate)`` rounded to the
    nearest sample, enforced by trimming/padding before loudnorm — so cross-arm
    sample alignment is preserved at any native rate. ffmpeg availability is
    checked only after caller-data validation, so ``INVALID_AUDIO`` stays
    reachable on machines without ffmpeg.
    """
    samples = _to_float(mono_samples, sample_width)
    band = bandpass_narrowband(samples, sample_rate)

    to_gsm = math.gcd(GSM_RATE, sample_rate)
    at_8k = resample_poly(band, GSM_RATE // to_gsm, sample_rate // to_gsm)
    round_tripped = gsm_roundtrip(at_8k)

    at_16k = resample_poly(round_tripped, PROFILE_RATE // GSM_RATE, 1)
    # ponytail: resample_poly's ceil leaves at most a couple of stray samples at
    # mixed native rates; trim/pad to the duration-ideal count so arms align.
    expected = (len(samples) * PROFILE_RATE + sample_rate // 2) // sample_rate
    if len(at_16k) < expected:
        at_16k = np.pad(at_16k, (0, expected - len(at_16k)))
    else:
        at_16k = at_16k[:expected]

    normalized, loudness = loudnorm_two_pass(at_16k)
    return ProfileResult(
        samples=normalized.astype(np.float32),
        sample_rate=PROFILE_RATE,
        loudness=loudness,
    )
