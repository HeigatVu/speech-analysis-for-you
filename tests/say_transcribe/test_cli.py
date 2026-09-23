import io
from pathlib import Path
import wave

import numpy as np

from say_transcribe.cli import main


def _make_wav_file(path: Path, sample_rate: int = 16000, duration_s: float = 1.0) -> Path:
    n_samples = int(sample_rate * duration_s)
    data = (np.sin(2 * np.pi * 440 * np.linspace(0, duration_s, n_samples)) * 10000).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())
    path.write_bytes(buf.getvalue())
    return path


def test_cli_diagnose_exit_code_zero(capsys):
    ret = main(["diagnose"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "say-transcribe diagnostic report:" in captured.out


def test_cli_exit_code_two_on_validation_failure(tmp_path: Path, capsys):
    # Non-existent audio file
    non_existent = tmp_path / "not_there.wav"
    ret = main(["run", str(non_existent), "--channel", "0", "--out", str(tmp_path)])
    assert ret == 2

    # Negative channel index
    audio_file = _make_wav_file(tmp_path / "audio.wav")
    ret = main(["run", str(audio_file), "--channel", "-1", "--out", str(tmp_path)])
    assert ret == 2


def test_cli_exit_code_three_on_model_or_gpu_unavailable(tmp_path: Path, monkeypatch, capsys):
    audio_file = _make_wav_file(tmp_path / "audio.wav")

    # Force transcribe to raise AsrError("GPU_UNAVAILABLE", ...)
    from say_transcribe.asr import AsrError

    def fail_transcribe(*args, **kwargs):
        raise AsrError("GPU_UNAVAILABLE", "CUDA device requested but not available")

    monkeypatch.setattr("say_transcribe.cli.transcribe", fail_transcribe)

    ret = main(["run", str(audio_file), "--channel", "0", "--out", str(tmp_path / "out"), "--device", "cuda"])
    assert ret == 3
    captured = capsys.readouterr()
    assert "[GPU_UNAVAILABLE]" in captured.err
    # Verify no absolute path or transcript leaked
    assert str(tmp_path) not in captured.err


def test_cli_exit_code_four_on_pipeline_failure(tmp_path: Path, capsys):
    audio_file = _make_wav_file(tmp_path / "audio.wav")
    # Channel 5 is out of bounds for mono audio -> INVALID_AUDIO_CHANNEL -> exit 4
    ret = main(["run", str(audio_file), "--channel", "5", "--out", str(tmp_path / "out")])
    assert ret == 4
    captured = capsys.readouterr()
    assert "[INVALID_AUDIO_CHANNEL]" in captured.err


def test_cli_run_happy_path_exit_code_zero(tmp_path: Path, monkeypatch, capsys):
    audio_file = _make_wav_file(tmp_path / "session_01_master.wav")
    out_dir = tmp_path / "output"

    from say_transcribe.asr import AsrResult, AsrSegment, WordTiming

    def fake_transcribe(*args, **kwargs):
        return AsrResult(
            source_sha256="abc123" * 10 + "abcd",
            segments=(
                AsrSegment(
                    start_ms=0,
                    end_ms=1000,
                    text="tôi là sinh viên .",
                    words=(
                        WordTiming(word="tôi", start_ms=0, end_ms=300),
                        WordTiming(word="là", start_ms=300, end_ms=500),
                        WordTiming(word="sinh", start_ms=500, end_ms=700),
                        WordTiming(word="viên", start_ms=700, end_ms=950),
                        WordTiming(word=".", start_ms=None, end_ms=None),
                    ),
                ),
            ),
            warnings=(),
        )

    class FakePyannote:
        def diarize(self, *args, **kwargs):
            return ()

    class FakeStanza:
        pass

    monkeypatch.setattr("say_transcribe.cli.transcribe", fake_transcribe)
    monkeypatch.setattr(
        "say_transcribe.cli.group_utterance_words",
        lambda seg: tuple([
            __import__("say_transcribe.word_grouping", fromlist=["GroupedWord"]).GroupedWord("tôi", 0, 300, ()),
            __import__("say_transcribe.word_grouping", fromlist=["GroupedWord"]).GroupedWord("là", 300, 500, ()),
            __import__("say_transcribe.word_grouping", fromlist=["GroupedWord"]).GroupedWord("sinh_viên", 500, 950, ()),
            __import__("say_transcribe.word_grouping", fromlist=["GroupedWord"]).GroupedWord(".", None, None, ()),
        ]),
    )
    monkeypatch.setattr("say_transcribe.cli.project_morphosyntax", lambda *args, **kwargs: None)

    ret = main(["run", str(audio_file), "--channel", "0", "--out", str(out_dir)])
    assert ret == 0
    assert (out_dir / "session_01.cha").exists()


def test_cli_evaluate_happy_path(tmp_path: Path):
    gold_dir = tmp_path / "gold"
    pred_dir = tmp_path / "pred"
    gold_dir.mkdir()
    pred_dir.mkdir()
    dummy_cha = """@UTF8
@Begin
@Languages:\tvie
@Participants:\tPAR Participant, INV Investigator
@ID:\tvie|corpus|PAR|||||Participant|||
@ID:\tvie|corpus|INV|||||Investigator|||
@Media:\ts01, audio
*PAR:\txin chào .\t\x150_1000\x15
@End
"""
    (gold_dir / "s01.cha").write_text(dummy_cha, encoding="utf-8")
    (pred_dir / "s01.cha").write_text(dummy_cha, encoding="utf-8")
    out_json = tmp_path / "report.json"

    ret = main(["evaluate", str(gold_dir), str(pred_dir), "--out", str(out_json)])
    assert ret == 0
    assert out_json.exists()


def test_no_dropped_v4_error_codes_in_cli_source():
    cli_code = Path(__file__).parent.parent.parent / "src" / "say_transcribe" / "cli.py"
    content = cli_code.read_text(encoding="utf-8")
    dropped_codes = [
        "INVALID_ANNOTATIONS",
        "SOURCE_HASH_MISMATCH",
        "UNAPPROVED_ANNOTATIONS",
        "CLAP_NOT_FOUND",
        "CLAP_COUNT_MISMATCH",
        "SOX_UNAVAILABLE",
        "SOX_GSM_UNAVAILABLE",
        "FFMPEG_UNAVAILABLE",
        "LOUDNESS_UNMEASURABLE",
        "DENOISER_UNAVAILABLE",
        "VAD_UNAVAILABLE",
        "MODEL_INTEGRITY_ERROR",
        "PREPROCESS_ALIGNMENT_ERROR",
        "INVALID_PREPROCESS_ARTIFACT",
    ]
    for code in dropped_codes:
        assert code not in content, f"Dropped code {code} appeared in cli.py"
