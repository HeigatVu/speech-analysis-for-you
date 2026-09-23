import io
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

from say_transcribe.asr import (
    AsrError,
    AsrResult,
    AsrSegment,
    PhoWhisperBackend,
    WordTiming,
    compute_sha256,
    transcribe,
)
from say_transcribe.audio import AudioPreparationError


def _make_wav_bytes(
    channels: int,
    sample_rate: int,
    sample_width: int,
    data: np.ndarray,
) -> bytes:
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
def sample_wav(tmp_path: Path):
    sample_rate = 16000
    duration_s = 1.0
    n_samples = int(sample_rate * duration_s)
    data = (np.sin(2 * np.pi * 440 * np.linspace(0, duration_s, n_samples)) * 10000).astype(np.int16)
    wav_bytes = _make_wav_bytes(1, sample_rate, 2, data)
    path = tmp_path / "sample.wav"
    path.write_bytes(wav_bytes)
    return path


class FakePhoWhisperBackend(PhoWhisperBackend):
    def __init__(self, segments: Sequence[dict[str, Any]]) -> None:
        super().__init__(model_id="fake", device="cpu")
        self._segments = segments

    def load(self) -> None:
        pass

    def transcribe_audio(self, audio_16k_mono: np.ndarray) -> Sequence[dict[str, Any]]:
        return self._segments


def test_lazy_import_does_not_load_torch_or_transformers():
    code = (
        "import sys, say_transcribe\n"
        "assert 'torch' not in sys.modules, 'torch was eagerly imported'\n"
        "assert 'transformers' not in sys.modules, 'transformers was eagerly imported'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"Import test failed:\n{result.stderr}"


def test_compute_sha256(sample_wav):
    sha = compute_sha256(sample_wav)
    assert len(sha) == 64
    assert all(c in "0123456789abcdef" for c in sha)


def test_transcribe_happy_path_with_word_timestamps(sample_wav):
    fake_segments = [
        {
            "start_ms": 100,
            "end_ms": 900,
            "text": "xin chào",
            "words": [
                {"word": "xin", "start_ms": 100, "end_ms": 450},
                {"word": "chào", "start_ms": 460, "end_ms": 900},
            ],
        }
    ]
    backend = FakePhoWhisperBackend(fake_segments)
    res = transcribe(sample_wav, channel_index=0, backend=backend)

    assert isinstance(res, AsrResult)
    assert len(res.segments) == 1
    assert res.warnings == ()

    seg = res.segments[0]
    assert isinstance(seg, AsrSegment)
    assert seg.start_ms == 100
    assert seg.end_ms == 900
    assert seg.text == "xin chào"
    assert len(seg.words) == 2
    assert seg.words[0] == WordTiming(word="xin", start_ms=100, end_ms=450)
    assert seg.words[1] == WordTiming(word="chào", start_ms=460, end_ms=900)


def test_transcribe_missing_word_timing_records_warning(sample_wav):
    fake_segments = [
        {
            "start_ms": 50,
            "end_ms": 400,
            "text": "cảm ơn",
            "words": [],  # Word timings unavailable
        },
        {
            "start_ms": 500,
            "end_ms": 800,
            "text": "tạm biệt",
            "words": [
                {"word": "tạm", "start_ms": 500, "end_ms": 650},
                {"word": "biệt", "start_ms": 660, "end_ms": 800},
            ],
        },
    ]
    backend = FakePhoWhisperBackend(fake_segments)
    res = transcribe(sample_wav, channel_index=0, backend=backend)

    assert len(res.segments) == 2
    assert res.warnings == ("CHAT_WORD_TIMING_UNAVAILABLE:1",)


def test_gpu_unavailable_raises_distinct_code():
    backend = PhoWhisperBackend(device="cuda")

    # In an environment without cuda available or if cuda fails
    with pytest.raises(AsrError) as exc_info:
        # Monkeypatch torch to simulate no cuda
        import sys
        import types

        fake_torch = types.ModuleType("torch")
        fake_cuda = types.ModuleType("torch.cuda")
        fake_cuda.is_available = lambda: False
        fake_torch.cuda = fake_cuda

        orig_torch = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        try:
            backend.load()
        finally:
            if orig_torch is not None:
                sys.modules["torch"] = orig_torch
            else:
                sys.modules.pop("torch", None)

    assert exc_info.value.code == "GPU_UNAVAILABLE"
    assert "cuda" in exc_info.value.message.lower()


def test_model_unavailable_raises_distinct_code():
    backend = PhoWhisperBackend(model_id="nonexistent-model-12345", device="cpu")
    with pytest.raises(AsrError) as exc_info:
        backend.load()
    assert exc_info.value.code == "MODEL_UNAVAILABLE"


def test_invalid_channel_index_fails_with_stable_code(sample_wav):
    with pytest.raises(AudioPreparationError) as exc_info:
        transcribe(sample_wav, channel_index=5)
    assert exc_info.value.code == "INVALID_AUDIO_CHANNEL"


def test_errors_do_not_leak_paths_or_raw_exceptions(tmp_path):
    backend = PhoWhisperBackend(model_id=str(tmp_path / "secret_model"), device="cpu")
    with pytest.raises(AsrError) as exc_info:
        backend.load()
    assert str(tmp_path) not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
