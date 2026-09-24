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


def test_transcribe_audio_windows_at_20s_and_offsets_timestamps():
    backend = PhoWhisperBackend(model_id="fake", device="cpu")
    calls: list[int] = []

    def fake_pipe(inp, **kwargs):
        calls.append(len(inp["raw"]))
        return {
            "chunks": [
                {"text": "hở", "timestamp": (0.0, 0.5)},
                {"text": ".", "timestamp": (0.5, 0.6)},
            ]
        }

    backend._pipe = fake_pipe
    audio = np.zeros(16000 * 21, dtype=np.float32)
    segs = backend.transcribe_audio(audio)

    assert len(calls) == 2
    assert calls[0] == 16000 * 20
    assert len(segs) == 2
    assert segs[0]["start_ms"] == 0
    assert segs[1]["start_ms"] == 20000
    assert segs[1]["words"][0]["start_ms"] == 20000
    assert segs[1]["end_ms"] == 20600


def test_short_audio_caps_generation_to_avoid_repetition_loops():
    backend = PhoWhisperBackend(model_id="fake", device="cpu")
    token_limits = []

    def fake_pipe(_input, **kwargs):
        token_limits.append(kwargs["max_new_tokens"])
        return {"chunks": []}

    backend._pipe = fake_pipe
    backend.transcribe_audio(np.zeros(16000 // 2, dtype=np.float32))
    backend.transcribe_audio(np.zeros(20 * 16000, dtype=np.float32))

    assert token_limits == [32, 400]


def test_repetition_loop_becomes_unintelligible_marker_instead_of_hallucinated_text():
    backend = PhoWhisperBackend(model_id="fake", device="cpu")
    looped = [{"text": "a", "timestamp": (i * 0.1, i * 0.1 + 0.05)} for i in range(30)]
    looped.append({"text": ".", "timestamp": None})

    def fake_pipe(_input, **kwargs):
        return {"chunks": looped}

    backend._pipe = fake_pipe
    segs = backend.transcribe_audio(np.zeros(16000, dtype=np.float32))

    assert len(segs) == 1
    assert segs[0]["text"] == "xxx"
    assert segs[0]["words"] == [{"word": "xxx", "start_ms": None, "end_ms": None}]
    assert segs[0]["start_ms"] is None
    assert segs[0]["end_ms"] is None


def test_unk_token_becomes_unintelligible_marker():
    backend = PhoWhisperBackend(model_id="fake", device="cpu")

    def fake_pipe(_input, **kwargs):
        return {"chunks": [{"text": "unk", "timestamp": (0.0, 0.5)}, {"text": ".", "timestamp": None}]}

    backend._pipe = fake_pipe
    segs = backend.transcribe_audio(np.zeros(16000, dtype=np.float32))

    assert segs[0]["text"] == "xxx"
    assert "unk" not in segs[0]["text"]


def test_unk_token_with_attached_punctuation_becomes_unintelligible_marker():
    # Real PhoWhisper word-level chunks attach trailing punctuation to the word
    # itself (single chunk "unk.") rather than emitting it as a separate chunk.
    backend = PhoWhisperBackend(model_id="fake", device="cpu")

    def fake_pipe(_input, **kwargs):
        return {"chunks": [{"text": "unk.", "timestamp": (0.0, 0.5)}]}

    backend._pipe = fake_pipe
    segs = backend.transcribe_audio(np.zeros(16000, dtype=np.float32))

    assert segs[0]["text"] == "xxx"
    assert "unk" not in segs[0]["text"]


def test_transcribe_flags_repetition_suspected_segment_with_a_warning(sample_wav):
    words = [{"word": "a", "start_ms": None, "end_ms": None} for _ in range(30)]
    fake_segments = [
        {
            "start_ms": None,
            "end_ms": None,
            "text": " ".join(w["word"] for w in words),
            "words": words,
            "repetition_suspected": True,
        }
    ]
    backend = FakePhoWhisperBackend(fake_segments)
    result = transcribe(sample_wav, channel_index=0, backend=backend)

    assert result.segments[0].text == " ".join(["a"] * 30)
    assert "ASR_REPETITION_SUSPECTED:1" in result.warnings


@pytest.mark.parametrize(
    "start_ms,end_ms",
    [
        (None, None),
        (100, None),
        (500, 500),
        (800, 400),
        (float("nan"), 500),
        (-1, 10),
        (1000, 1100),
    ],
)
def test_invalid_word_timing_is_unavailable_without_losing_text(
    sample_wav, start_ms, end_ms
):
    backend = FakePhoWhisperBackend(
        [
            {
                "start_ms": 100,
                "end_ms": 900,
                "text": "xin lỗi",
                "words": [
                    {"word": "xin", "start_ms": 100, "end_ms": 300},
                    {"word": "lỗi", "start_ms": start_ms, "end_ms": end_ms},
                ],
            }
        ]
    )

    result = transcribe(sample_wav, backend=backend)

    assert result.segments[0].text == "xin lỗi"
    assert result.segments[0].words == (
        WordTiming("xin", 100, 300),
        WordTiming("lỗi", None, None),
    )
    assert result.warnings == ("CHAT_WORD_TIMING_UNAVAILABLE:1",)


def test_all_untimed_asr_text_does_not_get_a_fabricated_zero_span(sample_wav):
    backend = FakePhoWhisperBackend(
        [{"start_ms": None, "end_ms": None, "text": "xin chào", "words": []}]
    )

    result = transcribe(sample_wav, backend=backend)

    assert result.segments[0] == AsrSegment(None, None, "xin chào", ())
    assert result.warnings == (
        "CHAT_WORD_TIMING_UNAVAILABLE:1",
        "CHAT_UTTERANCE_TIMING_UNAVAILABLE:1",
    )


def test_word_chunks_without_timestamps_keep_text_and_missing_segment_span():
    backend = PhoWhisperBackend(model_id="fake", device="cpu")
    backend._pipe = lambda *_args, **_kwargs: {
        "chunks": [{"text": "không", "timestamp": (None, None)}]
    }

    segments = backend.transcribe_audio(np.zeros(16000, dtype=np.float32))

    assert segments[0]["text"] == "không"
    assert segments[0]["start_ms"] is None
    assert segments[0]["end_ms"] is None


def test_word_timing_outside_its_inference_window_is_discarded():
    backend = PhoWhisperBackend(model_id="fake", device="cpu")
    backend._pipe = lambda *_args, **_kwargs: {
        "chunks": [{"text": "ngoài", "timestamp": (20.1, 20.2)}]
    }

    segments = backend.transcribe_audio(np.zeros(20 * 16000, dtype=np.float32))

    assert segments[0]["text"] == "ngoài"
    assert segments[0]["start_ms"] is None
    assert segments[0]["end_ms"] is None
    assert segments[0]["words"][0]["start_ms"] is None
    assert segments[0]["words"][0]["end_ms"] is None


def test_word_timing_outside_declared_segment_is_discarded(sample_wav):
    backend = FakePhoWhisperBackend(
        [
            {
                "start_ms": 100,
                "end_ms": 200,
                "text": "xin",
                "words": [{"word": "xin", "start_ms": 50, "end_ms": 150}],
            }
        ]
    )

    result = transcribe(sample_wav, backend=backend)

    assert result.segments[0].start_ms == 100
    assert result.segments[0].end_ms == 200
    assert result.segments[0].words == (WordTiming("xin", None, None),)
    assert result.warnings == ("CHAT_WORD_TIMING_UNAVAILABLE:1",)


def test_backwards_word_timing_is_discarded(sample_wav):
    backend = FakePhoWhisperBackend(
        [
            {
                "start_ms": 50,
                "end_ms": 200,
                "text": "xin rồi",
                "words": [
                    {"word": "xin", "start_ms": 100, "end_ms": 150},
                    {"word": "rồi", "start_ms": 50, "end_ms": 90},
                ],
            }
        ]
    )

    result = transcribe(sample_wav, backend=backend)

    assert result.segments[0].text == "xin rồi"
    assert result.segments[0].words == (
        WordTiming("xin", 100, 150),
        WordTiming("rồi", None, None),
    )
    assert result.warnings == ("CHAT_WORD_TIMING_UNAVAILABLE:1",)


def test_partial_word_times_do_not_create_an_utterance_span(sample_wav):
    backend = FakePhoWhisperBackend(
        [
            {
                "start_ms": None,
                "end_ms": None,
                "text": "xin chào",
                "words": [
                    {"word": "xin", "start_ms": 100, "end_ms": 200},
                    {"word": "chào", "start_ms": None, "end_ms": None},
                ],
            }
        ]
    )

    result = transcribe(sample_wav, backend=backend)

    assert result.segments[0].text == "xin chào"
    assert result.segments[0].start_ms is None
    assert result.segments[0].end_ms is None
    assert result.warnings == (
        "CHAT_WORD_TIMING_UNAVAILABLE:1",
        "CHAT_UTTERANCE_TIMING_UNAVAILABLE:1",
    )


def test_chunk_words_with_missing_lexical_boundary_do_not_create_segment_span():
    backend = PhoWhisperBackend(model_id="fake", device="cpu")
    backend._pipe = lambda *_args, **_kwargs: {
        "chunks": [
            {"text": "xin", "timestamp": (None, None)},
            {"text": "chào", "timestamp": (0.1, 0.2)},
        ]
    }

    segments = backend.transcribe_audio(np.zeros(16000, dtype=np.float32))

    assert segments[0]["text"] == "xin chào"
    assert segments[0]["start_ms"] is None
    assert segments[0]["end_ms"] is None
