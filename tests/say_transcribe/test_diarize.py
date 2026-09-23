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


def test_pyannote_backend_error_codes():
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
