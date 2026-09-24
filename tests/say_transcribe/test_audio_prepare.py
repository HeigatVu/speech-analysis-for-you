import io
import wave
from pathlib import Path

import numpy as np
import pytest

from say_transcribe.audio import (
    Audio,
    AudioPreparationError,
    extract_channel,
    read_wav,
    resample_to_16kHz,
)


def _make_wav_bytes(
    channels: int,
    sample_rate: int,
    sample_width: int,
    data: np.ndarray,
) -> bytes:
    """Helper to synthesize PCM WAV bytes in-memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        if sample_width == 2:
            raw = (data.astype(np.int16)).tobytes()
        elif sample_width == 4:
            raw = (data.astype(np.int32)).tobytes()
        else:
            raise ValueError(f"unsupported width {sample_width}")
        wf.writeframes(raw)
    return buf.getvalue()


@pytest.fixture
def stereo_pilot_audio(tmp_path: Path):
    """Create a 48 kHz stereo WAV where Left is active (sine) and Right is silent."""
    sample_rate = 48000
    duration_s = 2.0
    n_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    left = (np.sin(2 * np.pi * 440 * t) * 20000).astype(np.int16)
    right = np.zeros(n_samples, dtype=np.int16)
    interleaved = np.empty((n_samples, 2), dtype=np.int16)
    interleaved[:, 0] = left
    interleaved[:, 1] = right

    wav_bytes = _make_wav_bytes(2, sample_rate, 2, interleaved)
    audio_file = tmp_path / "pilot_session_01.wav"
    audio_file.write_bytes(wav_bytes)
    return audio_file, sample_rate, 2, interleaved


def test_read_wav_success(stereo_pilot_audio):
    audio_file, sample_rate, sample_width, interleaved = stereo_pilot_audio
    audio = read_wav(audio_file)
    assert isinstance(audio, Audio)
    assert audio.channels == 2
    assert audio.sample_rate == sample_rate
    assert audio.sample_width == sample_width
    assert audio.total_samples == interleaved.shape[0]
    np.testing.assert_array_equal(audio.samples, interleaved)


def test_extract_channel_active_left_silent_right_prevents_mean_downmix(stereo_pilot_audio):
    audio_file, sample_rate, sample_width, interleaved = stereo_pilot_audio
    audio = read_wav(audio_file)

    left = extract_channel(audio, 0)
    assert left.shape == (interleaved.shape[0],)
    np.testing.assert_array_equal(left, interleaved[:, 0])
    assert np.max(np.abs(left)) > 15000

    right = extract_channel(audio, 1)
    assert right.shape == (interleaved.shape[0],)
    np.testing.assert_array_equal(right, interleaved[:, 1])
    assert np.all(right == 0)


def test_invalid_channel_index_fails_with_stable_code(stereo_pilot_audio):
    audio_file, _, _, _ = stereo_pilot_audio
    audio = read_wav(audio_file)

    with pytest.raises(AudioPreparationError) as exc_info:
        extract_channel(audio, -1)
    assert exc_info.value.code == "INVALID_AUDIO_CHANNEL"

    with pytest.raises(AudioPreparationError) as exc_info:
        extract_channel(audio, 2)
    assert exc_info.value.code == "INVALID_AUDIO_CHANNEL"


def test_resample_to_16khz_in_memory(stereo_pilot_audio):
    audio_file, sample_rate, sample_width, _ = stereo_pilot_audio
    audio = read_wav(audio_file)
    left = extract_channel(audio, 0)

    resampled = resample_to_16kHz(left, sample_rate, sample_width)
    assert isinstance(resampled, np.ndarray)
    assert resampled.dtype == np.float32
    assert resampled.ndim == 1
    # 2.0 seconds at 16 kHz = 32000 samples
    assert abs(len(resampled) - 32000) <= 2
    assert np.max(np.abs(resampled)) <= 1.0


def test_resample_to_16khz_identity():
    data = np.array([0, 16384, -16384], dtype=np.int16)
    resampled = resample_to_16kHz(data, 16000, 2)
    assert resampled.dtype == np.float32
    np.testing.assert_allclose(resampled, [0.0, 0.5, -0.5], atol=1e-4)


def test_mono_int32_input_passes_through_and_scales_correctly(tmp_path):
    sample_rate = 16000
    n_samples = 1600
    max_int32 = 2147483647
    data = np.full(n_samples, max_int32 // 2, dtype=np.int32)
    wav_bytes = _make_wav_bytes(1, sample_rate, 4, data)

    audio_file = tmp_path / "mono32.wav"
    audio_file.write_bytes(wav_bytes)

    audio = read_wav(audio_file)
    assert audio.channels == 1
    assert audio.sample_width == 4

    mono = extract_channel(audio, 0)
    np.testing.assert_array_equal(mono, data)

    resampled = resample_to_16kHz(mono, sample_rate, 4)
    assert 0.49 < np.max(resampled) < 0.51


def test_corrupt_wav_error_does_not_chain_raw_exception(tmp_path):
    garbage = b"not a real wav file"
    audio_file = tmp_path / "corrupt.wav"
    audio_file.write_bytes(garbage)

    with pytest.raises(AudioPreparationError) as exc_info:
        read_wav(audio_file)
    assert exc_info.value.code == "INVALID_AUDIO"
    assert exc_info.value.__cause__ is None


def test_truncated_stereo_wav_fails_with_stable_code(tmp_path):
    wav_bytes = _make_wav_bytes(2, 16000, 2, np.zeros((100, 2), dtype=np.int16))
    audio_file = tmp_path / "truncated.wav"
    audio_file.write_bytes(wav_bytes[:-2])

    with pytest.raises(AudioPreparationError) as exc_info:
        read_wav(audio_file)

    assert exc_info.value.code == "INVALID_AUDIO"
    assert exc_info.value.__cause__ is None


def test_unsupported_sample_width_fails_with_stable_code(tmp_path):
    sample_rate = 16000
    n_samples = 800
    data = np.linspace(0, 255, n_samples, dtype=np.uint8)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())
    wav_bytes = buf.getvalue()

    audio_file = tmp_path / "eight_bit.wav"
    audio_file.write_bytes(wav_bytes)

    with pytest.raises(AudioPreparationError) as exc_info:
        read_wav(audio_file)
    assert exc_info.value.code == "INVALID_AUDIO"
