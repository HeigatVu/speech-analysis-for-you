import subprocess
import sys

import pytest

from say_transcribe.asr import AsrError, AsrSegment
from say_transcribe.diarize import (
    DRAFT_SPEAKER_NOTE,
    DiarizationTurn,
    PyannoteBackend,
    SpeakerDiarizationResult,
    assign_speakers,
)


def test_lazy_import_does_not_load_pyannote():
    code = (
        "import sys, say_transcribe\n"
        "assert 'pyannote' not in sys.modules, 'pyannote was eagerly imported'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"Import test failed:\n{result.stderr}"


def test_talk_time_rule_assigns_par_and_inv():
    # Speaker "SPEAKER_00": 3000 ms total talk time
    # Speaker "SPEAKER_01": 1000 ms total talk time
    # Thus SPEAKER_00 must become PAR, and SPEAKER_01 must become INV
    turns = [
        DiarizationTurn(start_ms=0, end_ms=2000, cluster_id="SPEAKER_00"),
        DiarizationTurn(start_ms=2100, end_ms=3100, cluster_id="SPEAKER_01"),
        DiarizationTurn(start_ms=3200, end_ms=4200, cluster_id="SPEAKER_00"),
    ]

    segments = [
        AsrSegment(start_ms=100, end_ms=1800, text="lời của người tham gia", words=()),
        AsrSegment(start_ms=2150, end_ms=2900, text="câu hỏi của điều tra viên", words=()),
        AsrSegment(start_ms=3300, end_ms=4000, text="tiếp tục câu trả lời", words=()),
    ]

    res = assign_speakers(segments, turns)
    assert isinstance(res, SpeakerDiarizationResult)
    assert res.is_draft is True
    assert res.draft_comment == DRAFT_SPEAKER_NOTE
    assert res.cluster_to_role["SPEAKER_00"] == "PAR"
    assert res.cluster_to_role["SPEAKER_01"] == "INV"

    assert res.utterance_speakers == ("PAR", "INV", "PAR")


def test_single_speaker_defaults_to_par():
    turns = [DiarizationTurn(start_ms=0, end_ms=1000, cluster_id="SPK_ONLY")]
    segments = [AsrSegment(start_ms=100, end_ms=900, text="xin chào", words=())]

    res = assign_speakers(segments, turns)
    assert res.is_draft is True
    assert res.utterance_speakers == ("PAR",)
    assert res.cluster_to_role["SPK_ONLY"] == "PAR"


def test_no_turns_defaults_to_par():
    segments = [AsrSegment(start_ms=100, end_ms=900, text="xin chào", words=())]
    res = assign_speakers(segments, ())
    assert res.is_draft is True
    assert res.utterance_speakers == ("PAR",)


def test_pyannote_backend_error_codes(monkeypatch):
    import types

    class Pipeline:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise RuntimeError("model unavailable")

    fake_pyannote_audio = types.ModuleType("pyannote.audio")
    fake_pyannote_audio.Pipeline = Pipeline
    monkeypatch.setitem(sys.modules, "pyannote", types.ModuleType("pyannote"))
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote_audio)
    backend = PyannoteBackend(model_id="nonexistent-pyannote-model")
    with pytest.raises(AsrError) as exc_info:
        backend.load()
    assert exc_info.value.code == "MODEL_UNAVAILABLE"


def test_pyannote_backend_gpu_unavailable():
    backend = PyannoteBackend(device="cuda")
    import types

    fake_torch = types.ModuleType("torch")
    fake_cuda = types.ModuleType("torch.cuda")
    fake_cuda.is_available = lambda: False
    fake_torch.cuda = fake_cuda

    orig_torch = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch
    try:
        with pytest.raises(AsrError) as exc_info:
            backend.load()
        assert exc_info.value.code == "GPU_UNAVAILABLE"
    finally:
        if orig_torch is not None:
            sys.modules["torch"] = orig_torch
        else:
            sys.modules.pop("torch", None)


def test_wavlm_cluster_backend_separates_speakers_by_segment():
    import numpy as np

    from say_transcribe.asr import AsrSegment
    from say_transcribe.diarize import WavlmClusterBackend

    backend = WavlmClusterBackend(device="cpu")
    backend._model = object()  # sentinel: skip real load
    rng = np.random.default_rng(0)

    segments = tuple(
        AsrSegment(start_ms=i * 2000, end_ms=i * 2000 + 1500, text="x", words=()) for i in range(8)
    )
    calls = []

    def fake_embed(wav):
        calls.append(len(wav))
        v = np.array([1.0, 0.0]) if len(calls) <= 4 else np.array([0.0, 1.0])
        return v + rng.normal(0, 0.01, 2)

    backend._embed = fake_embed
    audio = np.zeros(16000 * 20, dtype=np.float32)
    turns = backend.diarize(audio, segments=segments)

    assert len(turns) == 8
    labels = [t.cluster_id for t in turns]
    assert labels.count(labels[0]) == 4
    assert labels[0] != labels[-1]
    assert turns[0].start_ms == 0 and turns[0].end_ms == 1500


def test_pyannote_v4_uses_token_argument_and_speaker_diarization_result(monkeypatch):
    import types

    import numpy as np

    calls = {}
    annotation = types.SimpleNamespace(
        itertracks=lambda yield_label: [(types.SimpleNamespace(start=0.1, end=0.8), None, "SPK")]
    )

    class Pipeline:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.update(model_id=model_id, **kwargs)
            return lambda audio: types.SimpleNamespace(speaker_diarization=annotation)

    fake_pyannote_audio = types.ModuleType("pyannote.audio")
    fake_pyannote_audio.Pipeline = Pipeline
    fake_torch = types.ModuleType("torch")
    fake_torch.from_numpy = lambda value: types.SimpleNamespace(unsqueeze=lambda axis: value)
    monkeypatch.setitem(sys.modules, "pyannote", types.ModuleType("pyannote"))
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote_audio)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("HF_TOKEN", "synthetic-token")

    backend = PyannoteBackend()
    turns = backend.diarize(np.zeros(16000, dtype=np.float32))

    assert calls["token"] == "synthetic-token"
    assert "use_auth_token" not in calls
    assert turns == [DiarizationTurn(100, 800, "SPK")]


def test_wavlm_windows_stop_at_utterance_end_and_keep_segment_association():
    import numpy as np

    from say_transcribe.diarize import WavlmClusterBackend

    backend = WavlmClusterBackend(device="cpu")
    backend._model = object()
    segments = (
        AsrSegment(start_ms=2500, end_ms=2900, text="outside audio", words=()),
        AsrSegment(start_ms=1000, end_ms=1400, text="kept utterance", words=()),
    )
    embedded = []
    backend._embed = lambda wav: (embedded.append(wav.copy()) or np.array([1.0, 0.0]))
    audio = np.arange(16000 * 2.5, dtype=np.float32)

    turns = backend.diarize(audio, segments=segments)

    assert [len(wav) for wav in embedded] == [6400]
    assert turns == (DiarizationTurn(1000, 1400, "0"),)


def test_assign_speakers_defaults_untimed_segment_to_par():
    segment = AsrSegment(start_ms=None, end_ms=None, text="xin chào", words=())

    result = assign_speakers((segment,), (DiarizationTurn(0, 1000, "INV"),))

    assert result.utterance_speakers == ("PAR",)
