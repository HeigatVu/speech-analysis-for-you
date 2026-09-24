import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


class AudioPreparationError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Audio:
    channels: int
    sample_rate: int
    sample_width: int
    total_samples: int
    samples: np.ndarray  # Shape: (total_samples, channels) or (total_samples, )


def read_wav(path: Path) -> Audio:
    """Read standard wav. Raise AudioPreparationError("INVALID_AUDIO") on invalid files"""
    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except Exception:
        raise AudioPreparationError("INVALID_AUDIO", "Failed to read WAV file") from None

    if sample_width == 2:
        dtype = np.int16
    elif sample_width == 4:
        dtype = np.int32
    else:
        raise AudioPreparationError(
            "INVALID_AUDIO", f"Unsupported sample width: {sample_width}"
        )

    if len(raw) != n_frames * channels * sample_width:
        raise AudioPreparationError("INVALID_AUDIO", "Incomplete WAV data")

    data = np.frombuffer(raw, dtype=dtype)
    if channels > 1:
        data = data.reshape(-1, channels)

    return Audio(
        channels=channels,
        sample_rate=sample_rate,
        sample_width=sample_width,
        total_samples=n_frames,
        samples=data,
    )


def extract_channel(audio: Audio, channel_index: int) -> np.ndarray:
    """Extract single channel without mixing channels. Validate channel bounds."""

    if channel_index < 0 or channel_index >= audio.channels:
        raise AudioPreparationError(
            "INVALID_AUDIO_CHANNEL", f"channel {channel_index} out of range"
        )
    if audio.channels == 1:
        return audio.samples

    return audio.samples[:, channel_index]


def resample_to_16kHz(
    mono_samples: np.ndarray, original_rate: int, sample_width: int
) -> np.ndarray:
    """Map PCM to float and resample to 16k Hz"""
    scale = 32768.0 if sample_width == 2 else 2147483648.0

    float_audio = mono_samples.astype(np.float32) / scale
    if original_rate == 16000:
        return float_audio

    gcd = math.gcd(16000, original_rate)
    up = 16000 // gcd
    down = original_rate // gcd
    resampled = resample_poly(float_audio, up, down).astype(np.float32)
    return np.clip(resampled, -1.0, 1.0)
