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
    res = transcribe(sample_wav, expected_sha256=compute_sha256(sample_wav), channel_index=0, backend=backend)

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
    res = transcribe(sample_wav, expected_sha256=compute_sha256(sample_wav), channel_index=0, backend=backend)

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


def _capture_cuda_load(monkeypatch, **backend_kwargs):
    """Load a PhoWhisperBackend on a faked CUDA device; report what it asked for."""
    import sys
    import types

    import torch

    captured: dict[str, object] = {}

    class _HalfSpy:
        halved = False

        def half(self):
            _HalfSpy.halved = True
            return self

    class _FakePipe:
        model = _HalfSpy()

    def fake_pipeline(task, **kwargs):
        captured.update(kwargs)
        return _FakePipe()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.pipeline = fake_pipeline

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    # transformers is a lazy module whose `from transformers import pipeline` binding
    # survives monkeypatch.setattr, so swap the module itself, as the cuda test above does.
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    PhoWhisperBackend(model_id="fake", device="cuda", **backend_kwargs).load()
    return captured, _HalfSpy


def test_cuda_loads_the_checkpoint_in_fp16_instead_of_halving_afterwards(monkeypatch):
    """Half the weights at load time, not after the fp32 copy already owns the card.

    Loading fp32 and calling model.half() afterwards puts the full-precision
    checkpoint on the GPU first and leaves the caching allocator holding ~6.6GB
    before a single token is generated; a word-timestamp window for
    phowhisper-large then needs ~9.1GB and OOMs a 12GB card.
    """
    import torch

    captured, half_spy = _capture_cuda_load(monkeypatch)

    assert captured["dtype"] is torch.float16
    assert half_spy.halved is False
    assert captured["return_timestamps"] == "word"


def test_segment_timestamps_are_selectable_without_changing_the_text(monkeypatch):
    """Text-only callers must not pay 4.4GB for timings they never read.

    Word timestamps cost ~4.4GB more per window than segment ones (9.06GB vs 4.63GB
    peak, measured 2026-10-03) because they capture cross-attention for the
    alignment; the decoded text is byte-identical, so a text-only caller asks for
    segment timestamps and fits a 12GB card.
    """
    captured, _ = _capture_cuda_load(monkeypatch, timestamps="segment")

    assert captured["return_timestamps"] is True


def test_qwen3_cuda_loads_the_checkpoint_in_fp16_instead_of_halving_afterwards(monkeypatch):
    """Same transient-fp32 cost as phowhisper: measured 4.70GB peak at load vs 4.08GB
    when fp16 is requested at load time (2026-10-03); window peaks are identical."""
    import sys
    import types

    import torch

    from say_transcribe.asr import Qwen3AsrBackend

    captured: dict[str, object] = {}

    class _Model:
        halved = False

        def to(self, device):
            return self

        def half(self):
            _Model.halved = True
            return self

    def fake_from_pretrained(model_id, **kwargs):
        captured.update(kwargs)
        return _Model()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: object()
    )
    fake_transformers.AutoModelForMultimodalLM = types.SimpleNamespace(
        from_pretrained=fake_from_pretrained
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    Qwen3AsrBackend._build("fake", None, "cuda")

    assert captured["dtype"] is torch.float16
    assert _Model.halved is False


def test_unknown_timestamp_mode_is_rejected():
    with pytest.raises(ValueError):
        PhoWhisperBackend(model_id="fake", device="cpu", timestamps="phoneme")


def test_model_unavailable_raises_distinct_code():
    backend = PhoWhisperBackend(model_id="nonexistent-model-12345", device="cpu")
    with pytest.raises(AsrError) as exc_info:
        backend.load()
    assert exc_info.value.code == "MODEL_UNAVAILABLE"


def test_invalid_channel_index_fails_with_stable_code(sample_wav):
    with pytest.raises(AudioPreparationError) as exc_info:
        transcribe(sample_wav, expected_sha256=compute_sha256(sample_wav), channel_index=5)
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
    result = transcribe(sample_wav, expected_sha256=compute_sha256(sample_wav), channel_index=0, backend=backend)

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

    result = transcribe(sample_wav, expected_sha256=compute_sha256(sample_wav), backend=backend)

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

    result = transcribe(sample_wav, expected_sha256=compute_sha256(sample_wav), backend=backend)

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

    result = transcribe(sample_wav, expected_sha256=compute_sha256(sample_wav), backend=backend)

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

    result = transcribe(sample_wav, expected_sha256=compute_sha256(sample_wav), backend=backend)

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

    result = transcribe(sample_wav, expected_sha256=compute_sha256(sample_wav), backend=backend)

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


class FakeQwen3Processor:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[dict[str, Any]] = []

    def apply_transcription_request(self, audio, language=None, prompt=None, **kwargs):
        self.calls.append({"audio": audio, "language": language, "prompt": prompt})
        return {"input_ids": np.zeros((1, 7), dtype=np.int64)}

    def decode(self, ids, return_format="raw"):
        assert return_format == "transcription_only"
        return self._text


class FakeQwen3Model:
    def __init__(self, failure: bool = False) -> None:
        self._failure = failure

    def generate(self, **inputs):
        if self._failure:
            raise RuntimeError("boom")
        return np.zeros((1, 12), dtype=np.int64)


def _fake_qwen3(text: str = "chào thế giới", failure: bool = False) -> Any:
    from say_transcribe.asr import Qwen3AsrBackend

    backend = Qwen3AsrBackend()
    backend._processor = FakeQwen3Processor(text)
    backend._model = FakeQwen3Model(failure=failure)
    return backend


def test_qwen3_segment_shape_and_forced_language():
    backend = _fake_qwen3()
    segments = backend.transcribe_audio(np.zeros(16000, dtype=np.float32))
    assert segments == [
        {"start_ms": 0, "end_ms": 1000, "text": "chào thế giới", "words": []}
    ]
    assert backend._processor.calls[0]["language"] == "Vietnamese"


def test_qwen3_segments_carry_their_window_span_on_the_passed_audio_timeline():
    """Qwen3 has no word timer, but each decode knows the window it decoded, so every
    segment is bounded: bullets and diarization overlap work, and a later aligner can
    time the words inside that span."""
    backend = _fake_qwen3(text="mot hai")
    segments = backend.transcribe_audio(np.zeros(45 * 16000, dtype=np.float32))
    spans = [(s["start_ms"], s["end_ms"]) for s in segments]
    assert len(spans) == 3
    assert spans[0][0] == 0 and spans[-1][1] == 45000
    assert all(start < end for start, end in spans)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    assert all(s["words"] == [] for s in segments)


def test_qwen3_empty_decode_yields_no_segments():
    backend = _fake_qwen3(text="   ")
    assert backend.transcribe_audio(np.zeros(16000, dtype=np.float32)) == []


def test_qwen3_chunks_long_audio_like_phowhisper():
    backend = _fake_qwen3(text="mot")
    backend.transcribe_audio(np.zeros(45 * 16000, dtype=np.float32))
    assert len(backend._processor.calls) == 3


def test_qwen3_load_failure_maps_to_model_unavailable(monkeypatch):
    from say_transcribe.asr import Qwen3AsrBackend

    def boom(model_id, revision, device):
        raise RuntimeError("no network")

    monkeypatch.setattr(Qwen3AsrBackend, "_build", boom)
    with pytest.raises(AsrError) as excinfo:
        Qwen3AsrBackend().load()
    assert excinfo.value.code == "MODEL_UNAVAILABLE"


def test_qwen3_inference_failure_maps_to_model_unavailable():
    backend = _fake_qwen3(failure=True)
    with pytest.raises(AsrError) as excinfo:
        backend.transcribe_audio(np.zeros(16000, dtype=np.float32))
    assert excinfo.value.code == "MODEL_UNAVAILABLE"


def test_make_asr_backend_maps_names():
    from say_transcribe.asr import Qwen3AsrBackend, make_asr_backend

    assert isinstance(make_asr_backend("qwen3-asr"), Qwen3AsrBackend)
    pho = make_asr_backend("phowhisper-large")
    assert pho.model_id == "vinai/phowhisper-large"
    assert make_asr_backend("phowhisper-medium").model_id == "vinai/phowhisper-medium"
    with pytest.raises(AsrError) as excinfo:
        make_asr_backend("unknown-model")
    assert excinfo.value.code == "MODEL_UNKNOWN"


def _fake_chunks(tokens, step=0.3):
    return {
        "chunks": [
            {"text": t, "timestamp": (i * step, i * step + 0.2)} for i, t in enumerate(tokens)
        ]
    }


def test_loop_split_into_one_word_segments_by_punctuation_becomes_one_marker():
    # Whisper attaches "." to each looped word, so the loop is flushed as 55
    # one-word segments that the per-segment guard never sees as a loop.
    backend = PhoWhisperBackend(model_id="fake", device="cpu")
    backend._pipe = lambda _input, **kwargs: _fake_chunks(["hai."] * 55)

    segs = backend.transcribe_audio(np.zeros(20 * 16000, dtype=np.float32))

    assert [s["text"] for s in segs] == ["xxx"]
    assert segs[0]["repetition_suspected"] is True


def test_eight_identical_words_in_one_segment_are_a_loop():
    backend = PhoWhisperBackend(model_id="fake", device="cpu")
    backend._pipe = lambda _input, **kwargs: _fake_chunks(["hai"] * 8 + ["."])

    segs = backend.transcribe_audio(np.zeros(16000 * 5, dtype=np.float32))

    assert [s["text"] for s in segs] == ["xxx"]


def test_short_valid_vietnamese_repeats_are_kept():
    backend = PhoWhisperBackend(model_id="fake", device="cpu")
    backend._pipe = lambda _input, **kwargs: _fake_chunks(
        ["một", "một", "bảy.", "ừ", "ừ."]
    )

    segs = backend.transcribe_audio(np.zeros(16000 * 5, dtype=np.float32))

    # The repeats survive; Whisper's own sentence periods do not reach this layer.
    assert [s["text"] for s in segs] == ["một một bảy", "ừ ừ"]


def _noise(seconds, seed=0):
    return (np.random.default_rng(seed).uniform(-0.5, 0.5, int(seconds * 16000))).astype(
        np.float32
    )


def test_window_cut_snaps_into_a_silent_gap_before_the_hard_limit():
    from say_transcribe.asr import _window_bounds

    audio = _noise(45)
    audio[int(18.0 * 16000) : int(18.6 * 16000)] = 0.0

    bounds = _window_bounds(audio)

    assert 18.0 * 16000 <= bounds[0][1] <= 18.6 * 16000


def test_windows_cover_the_input_without_gaps_and_stay_within_the_limit():
    from say_transcribe.asr import _window_bounds

    audio = _noise(65)
    audio[int(37.0 * 16000) : int(37.5 * 16000)] = 0.0

    bounds = _window_bounds(audio)

    assert bounds[0][0] == 0 and bounds[-1][1] == len(audio)
    assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:]))
    assert all(0 < end - start <= 20 * 16000 for start, end in bounds)


def test_window_cut_stays_at_the_hard_limit_when_nothing_is_quieter():
    from say_transcribe.asr import _window_bounds

    bounds = _window_bounds(np.zeros(45 * 16000, dtype=np.float32))

    assert bounds == [(0, 320000), (320000, 640000), (640000, 720000)]


def test_window_bounds_for_short_and_empty_audio():
    from say_transcribe.asr import _window_bounds

    assert _window_bounds(np.zeros(0, dtype=np.float32)) == []
    assert _window_bounds(_noise(5)) == [(0, 5 * 16000)]


def test_backend_offsets_follow_the_snapped_windows():
    from say_transcribe.asr import _window_bounds

    audio = _noise(30)
    audio[int(19.0 * 16000) : int(19.4 * 16000)] = 0.0
    bounds = _window_bounds(audio)
    backend = PhoWhisperBackend(model_id="fake", device="cpu")
    backend._pipe = lambda _input, **kwargs: {
        "chunks": [{"text": "xin.", "timestamp": (0.0, 0.2)}]
    }

    segs = backend.transcribe_audio(audio)

    assert [s["start_ms"] for s in segs] == [round(start / 16) for start, _ in bounds]
    assert bounds[0][1] != 20 * 16000


# Real p002 output: Whisper glued the sentence period onto the last word, which made
# chatter reject the file (E316 on the main tier, then E316/E342 on %wor) and counted
# as a phantom substitution against the reference. The ASR boundary owns the split.
_GLUED_SEGMENT = {
    "start_ms": 0,
    "end_ms": 900,
    "text": "như thế.",
    "words": [
        {"word": "như", "start_ms": 0, "end_ms": 400},
        {"word": "thế.", "start_ms": 400, "end_ms": 900},
    ],
}


def test_transcribe_leaves_no_sentence_punctuation_on_words(sample_wav):
    backend = FakePhoWhisperBackend([_GLUED_SEGMENT])

    res = transcribe(
        sample_wav, expected_sha256=compute_sha256(sample_wav), channel_index=0, backend=backend
    )

    assert [w.word for w in res.segments[0].words] == ["như", "thế"]
    assert res.segments[0].text == "như thế"


def test_window_pipeline_cleans_the_same_punctuation():
    from say_transcribe.asr import result_from_windows

    res = result_from_windows(
        np.zeros(16000, dtype=np.float32), "a" * 64, ((0, 16000, (_GLUED_SEGMENT,)),)
    )

    assert [w.word for w in res.segments[0].words] == ["như", "thế"]
    assert res.segments[0].text == "như thế"


def test_punctuation_only_word_is_dropped_rather_than_emptied():
    from say_transcribe.asr import result_from_windows

    segment = {
        "start_ms": 0,
        "end_ms": 900,
        "text": "thế .",
        "words": [
            {"word": "thế", "start_ms": 0, "end_ms": 400},
            {"word": ".", "start_ms": 400, "end_ms": 500},
        ],
    }

    res = result_from_windows(
        np.zeros(16000, dtype=np.float32), "a" * 64, ((0, 16000, (segment,)),)
    )

    assert [w.word for w in res.segments[0].words] == ["thế"]
    assert res.segments[0].text == "thế"


def test_cleaning_leaves_chat_markup_and_diacritics_verbatim():
    from say_transcribe.asr import result_from_windows

    kept = ["&-ờ", "[/]", "cứu_hoả", "(.)", "xxx", "được", "cô_bé"]
    segment = {
        "start_ms": 0,
        "end_ms": 900,
        "text": " ".join(kept),
        "words": [
            {"word": token, "start_ms": i * 100, "end_ms": i * 100 + 90}
            for i, token in enumerate(kept)
        ],
    }

    res = result_from_windows(
        np.zeros(16000, dtype=np.float32), "a" * 64, ((0, 16000, (segment,)),)
    )

    assert [w.word for w in res.segments[0].words] == kept
    assert res.segments[0].text == " ".join(kept)


@pytest.mark.parametrize(
    "text, spoken",
    [
        ("70", "bảy mươi"),
        ("15", "mười lăm"),
        ("21", "hai mươi mốt"),
        ("105", "một trăm lẻ năm"),
        ("1005", "một nghìn không trăm lẻ năm"),
        ("2024", "hai nghìn không trăm hai mươi tư"),
        ("30%", "ba mươi phần trăm"),
        ("%", "phần trăm"),
        ("tôi 70 tuổi.", "tôi bảy mươi tuổi."),
        ("0123", "0123"),  # leading zero: an identifier, not a quantity
        ("3.5", "3.5"),  # decimals are left for chatter to flag, never guessed
    ],
)
def test_spell_numbers_reads_vietnamese_quantities(text, spoken):
    from say_transcribe.asr import _spell_numbers

    assert _spell_numbers(text) == spoken


def test_qwen3_numerals_and_percent_are_spoken_so_the_main_tier_parses():
    backend = _fake_qwen3(text="giảm 70 %")
    [segment] = backend.transcribe_audio(np.zeros(16000, dtype=np.float32))
    assert segment["text"] == "giảm bảy mươi phần trăm"


def test_whisper_large_v3_pins_vietnamese_so_it_never_auto_detects():
    from say_transcribe.asr import make_asr_backend

    backend = make_asr_backend("whisper-large-v3")
    assert isinstance(backend, PhoWhisperBackend)
    assert backend.model_id == "openai/whisper-large-v3"
    assert backend.language == "vi"

    seen: dict[str, object] = {}

    def fake_pipe(inp, **kwargs):
        seen.update(kwargs["generate_kwargs"])
        return {"chunks": [{"text": "xin", "timestamp": (0.0, 0.5)}]}

    backend._pipe = fake_pipe
    backend.transcribe_audio(np.zeros(16000, dtype=np.float32))
    assert seen["language"] == "vi" and seen["task"] == "transcribe"


def test_phowhisper_does_not_force_a_language():
    backend = PhoWhisperBackend(model_id="fake", device="cpu")
    seen: dict[str, object] = {}

    def fake_pipe(inp, **kwargs):
        seen.update(kwargs["generate_kwargs"])
        return {"chunks": []}

    backend._pipe = fake_pipe
    backend.transcribe_audio(np.zeros(16000, dtype=np.float32))
    assert "language" not in seen and "task" not in seen


def test_wav2vec2_vi_is_a_pinned_ctc_arm_with_native_word_timing():
    from say_transcribe.asr import ASR_MODEL_CHOICES, Wav2Vec2AsrBackend, make_asr_backend

    backend = make_asr_backend("wav2vec2-vi")
    assert isinstance(backend, Wav2Vec2AsrBackend)
    assert backend.model_id == "nguyenvulebinh/wav2vec2-base-vi-vlsp2020"
    assert backend.revision == "50a30dadb3ec98a0d4cdb1eb1ea315aff538f7c2"
    assert {"whisper-large-v3", "wav2vec2-vi"} <= set(ASR_MODEL_CHOICES)

    calls: list[dict[str, object]] = []

    def fake_pipe(inp, **kwargs):
        calls.append(kwargs)  # a CTC pipeline takes no generation arguments
        return {"chunks": [{"text": "chào", "timestamp": (0.1, 0.4)}]}

    backend._pipe = fake_pipe
    segs = backend.transcribe_audio(np.zeros(21 * 16000, dtype=np.float32))

    assert calls == [{}, {}]
    assert segs[0]["words"][0]["start_ms"] == 100
    assert segs[1]["words"][0]["start_ms"] == 20100  # second window, master timeline
