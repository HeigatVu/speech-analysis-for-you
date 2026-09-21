from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from say_transcribe.audio_profile import AudioProfile, TELL_INSPIRED_V1


class PreprocessError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


def check_sox_support(sox_bin: str = "sox") -> str:
    """Verify that SoX binary is installed and supports GSM format.

    Returns the resolved SoX binary path.
    """
    bin_path = shutil.which(sox_bin)
    if not bin_path:
        raise PreprocessError("SOX_UNAVAILABLE", "SoX binary not found")

    try:
        res = subprocess.run(
            [bin_path, "--help-format", "gsm"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30.0,
        )
    except subprocess.TimeoutExpired:
        raise PreprocessError("SOX_TIMEOUT", "SoX capability check timed out") from None
    except OSError:
        raise PreprocessError("SOX_UNAVAILABLE", "Failed to execute SoX") from None

    output = (res.stdout + res.stderr).lower()
    if res.returncode != 0 or "gsm" not in output:
        raise PreprocessError(
            "SOX_GSM_UNAVAILABLE", "SoX installation lacks GSM format support"
        )
    return bin_path


def _run_sox(cmd: list[str], timeout: float, failure_message: str) -> None:
    """Run a SoX conversion step, raising a stable, redacted error on failure."""
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise PreprocessError("SOX_TIMEOUT", failure_message) from None

    if res.returncode != 0:
        raise PreprocessError("SOX_EXECUTION_FAILED", failure_message)


def _check_clipping(samples: np.ndarray, threshold_count: int = 5) -> None:
    """Detect if audio samples are saturated at int16 limits."""
    clipped = (samples == 32767) | (samples == -32768)
    if int(np.sum(clipped)) >= threshold_count:
        raise PreprocessError("CLIPPING_ERROR", "Output audio exhibits clipping saturation")


def _validate_artifact(path: Path, profile: AudioProfile, expected_samples: int) -> np.ndarray:
    """Validate a converted WAV's format, alignment, and clipping. Returns its samples."""
    try:
        with wave.open(str(path), "rb") as wf:
            out_channels = wf.getnchannels()
            out_rate = wf.getframerate()
            out_width = wf.getsampwidth()
            out_frames = wf.getnframes()
            raw_bytes = wf.readframes(out_frames)
    except Exception:
        raise PreprocessError(
            "INVALID_PREPROCESS_ARTIFACT", "Failed to read preprocessed WAV"
        ) from None

    if out_channels != 1 or out_rate != profile.target_sample_rate or out_width != 2:
        raise PreprocessError(
            "INVALID_PREPROCESS_ARTIFACT", "Output format does not match specification"
        )

    if len(raw_bytes) != out_frames * out_channels * out_width:
        raise PreprocessError(
            "INVALID_PREPROCESS_ARTIFACT", "Preprocessed WAV data is truncated"
        )

    drift = out_frames - expected_samples
    if drift < 0 or drift > profile.max_trailing_padding_samples:
        raise PreprocessError(
            "PREPROCESS_ALIGNMENT_ERROR",
            f"Sample drift {drift} outside allowable range 0..{profile.max_trailing_padding_samples}",
        )

    samples = np.frombuffer(raw_bytes, dtype=np.int16)
    _check_clipping(samples)
    return samples


def _atomic_write(src: Path, dst: Path) -> None:
    """Copy src to dst with permissions 0o600, replacing dst atomically."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = dst.with_suffix(f".tmp.{os.urandom(4).hex()}")
    fd: int | None = None
    try:
        fd = os.open(tmp_target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as target_stream:
            fd = None
            with src.open("rb") as source_stream:
                shutil.copyfileobj(source_stream, target_stream)
        os.replace(tmp_target, dst)
    finally:
        if fd is not None:
            os.close(fd)
        if tmp_target.exists():
            tmp_target.unlink(missing_ok=True)


def run_sox_gsm_stage(
    input_path: Path,
    output_path: Path,
    profile: AudioProfile = TELL_INSPIRED_V1,
    *,
    sox_bin: str = "sox",
    timeout: float = 30.0,
) -> Path:
    """Execute deterministic SoX GSM Full Rate round-trip and PCM16 bandpass.

    Stages:
    1. Verify mono input and check SoX + GSM capability.
    2. Encode mono input to GSM Full Rate (8 kHz, 13 kbps) in secure temp dir.
    3. Decode GSM to 16 kHz PCM16 with highpass at 200 Hz and lowpass at 3400 Hz.
       No compand is applied.
    4. Check trailing padding drift (0 <= drift <= max_trailing_padding_samples).
    5. Check absence of clipping saturation.
    6. Atomically replace output_path with permissions 0o600.
    """
    if not input_path.is_file():
        raise PreprocessError("INPUT_FILE_NOT_FOUND", "Input clip not found")
    if input_path.resolve() == output_path.resolve():
        raise PreprocessError(
            "OUTPUT_PATH_CONFLICT", "Output path must differ from input clip"
        )

    try:
        with wave.open(str(input_path), "rb") as wf:
            channels = wf.getnchannels()
            in_rate = wf.getframerate()
            in_frames = wf.getnframes()
    except Exception:
        raise PreprocessError("INVALID_AUDIO", "Failed to parse WAV header") from None

    if channels != 1:
        raise PreprocessError(
            "INVALID_AUDIO_CHANNEL", f"Expected mono audio, got {channels} channels"
        )

    bin_path = check_sox_support(sox_bin)

    expected_samples = int(round(in_frames * profile.target_sample_rate / in_rate))

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        os.chmod(tmp_dir, 0o700)

        stage1_gsm = tmp_dir / "stage1.gsm"
        stage2_wav = tmp_dir / "stage2.wav"

        # Step 1: encode to GSM 8000 Hz mono
        cmd1 = [
            bin_path,
            str(input_path),
            "-r",
            str(profile.gsm_sample_rate),
            "-c",
            "1",
            str(stage1_gsm),
        ]
        _run_sox(cmd1, timeout, "GSM encoding stage failed")

        # Step 2: decode to 16 kHz PCM16 with highpass and lowpass (no compand)
        cmd2 = [
            bin_path,
            str(stage1_gsm),
            "-t",
            "wav",
            "-b",
            "16",
            "-e",
            "signed-integer",
            "-c",
            "1",
            str(stage2_wav),
            "rate",
            "-v",
            str(profile.target_sample_rate),
            "highpass",
            str(int(profile.highpass_hz)),
            "lowpass",
            str(int(profile.lowpass_hz)),
        ]
        _run_sox(cmd2, timeout, "PCM decode stage failed")

        _validate_artifact(stage2_wav, profile, expected_samples)
        _atomic_write(stage2_wav, output_path)

    return output_path
