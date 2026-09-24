import io
from pathlib import Path
import wave

import numpy as np
import pytest

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


def test_cli_run_refuses_to_overwrite_existing_transcript(tmp_path: Path, monkeypatch, capsys):
    audio_file = _make_wav_file(tmp_path / "session_01_master.wav")
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    existing = out_dir / "session_01.cha"
    existing.write_text("manual transcript, never overwrite", encoding="utf-8")

    from say_transcribe.asr import AsrResult

    monkeypatch.setattr(
        "say_transcribe.cli.transcribe",
        lambda *args, **kwargs: AsrResult(source_sha256="a" * 64, segments=(), warnings=()),
    )
    monkeypatch.setattr(
        "say_transcribe.cli.PyannoteBackend",
        lambda **kwargs: type("D", (), {"diarize": lambda self, _: ()})(),
    )

    ret = main(["run", str(audio_file), "--channel", "0", "--out", str(out_dir)])
    assert ret == 2
    captured = capsys.readouterr()
    assert "[OUTPUT_EXISTS]" in captured.err
    assert existing.read_text(encoding="utf-8") == "manual transcript, never overwrite"


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


def test_cli_run_survives_word_grouping_mismatch(tmp_path: Path, monkeypatch, capsys):
    from say_transcribe.asr import AsrResult, AsrSegment, WordTiming
    from say_transcribe.word_grouping import WordGroupingError

    audio_file = _make_wav_file(tmp_path / "audio.wav")

    def fake_transcribe(*args, **kwargs):
        seg = AsrSegment(
            start_ms=0,
            end_ms=900,
            text="hở",
            words=(WordTiming(word="hở", start_ms=0, end_ms=900),),
        )
        return AsrResult(source_sha256="0" * 64, segments=(seg,), warnings=())

    def raise_grouping(*args, **kwargs):
        raise WordGroupingError("WORD_GROUPING_UNALIGNED", "count mismatch")

    monkeypatch.setattr("say_transcribe.cli.transcribe", fake_transcribe)
    monkeypatch.setattr("say_transcribe.cli.group_utterance_words", raise_grouping)
    monkeypatch.setattr(
        "say_transcribe.cli.assign_speakers",
        lambda segments, turns: type("R", (), {"utterance_speakers": ["PAR"]})(),
    )
    monkeypatch.setattr("say_transcribe.cli.project_morphosyntax", lambda *args, **kwargs: None)

    ret = main(["run", str(audio_file), "--channel", "0", "--out", str(tmp_path / "out")])
    assert ret == 0
    assert (tmp_path / "out" / "audio.cha").is_file()
    captured = capsys.readouterr()
    assert "[WORD_GROUPING_UNALIGNED:1]" in captured.err


def test_cli_reports_redacted_warning_when_diarization_and_fallback_fail(
    tmp_path: Path, monkeypatch, capsys
):
    from say_transcribe.asr import AsrResult, AsrSegment

    audio_file = _make_wav_file(tmp_path / "audio.wav")

    def fake_transcribe(*args, **kwargs):
        return AsrResult(
            source_sha256="0" * 64,
            segments=(AsrSegment(start_ms=0, end_ms=900, text="xin chào", words=()),),
            warnings=(),
        )

    class FailedBackend:
        def diarize(self, *args, **kwargs):
            raise RuntimeError(f"private text at {audio_file}: private transcript")

    monkeypatch.setattr("say_transcribe.cli.transcribe", fake_transcribe)
    monkeypatch.setattr("say_transcribe.cli.PyannoteBackend", lambda **kwargs: FailedBackend())
    monkeypatch.setattr("say_transcribe.cli.WavlmClusterBackend", lambda **kwargs: FailedBackend())
    monkeypatch.setattr("say_transcribe.cli.group_utterance_words", lambda seg: ())
    monkeypatch.setattr("say_transcribe.cli.project_morphosyntax", lambda *args, **kwargs: None)

    ret = main(["run", str(audio_file), "--channel", "0", "--out", str(tmp_path / "out")])

    assert ret == 0
    captured = capsys.readouterr()
    assert "[DIARIZATION_UNAVAILABLE:DEFAULTING_TO_PAR]" in captured.err
    assert str(tmp_path) not in captured.err
    assert "private text" not in captured.err
    assert "private transcript" not in captured.err


@pytest.mark.parametrize("primary_fails", [False, True])
def test_cli_warns_when_diarization_returns_no_turns(
    tmp_path: Path, monkeypatch, capsys, primary_fails: bool
):
    from say_transcribe.asr import AsrResult, AsrSegment

    audio_file = _make_wav_file(tmp_path / "audio.wav")

    def fake_transcribe(*args, **kwargs):
        return AsrResult(
            source_sha256="0" * 64,
            segments=(AsrSegment(start_ms=0, end_ms=900, text="xin chào", words=()),),
            warnings=(),
        )

    class PrimaryBackend:
        def diarize(self, *args, **kwargs):
            if primary_fails:
                raise RuntimeError("private primary error")
            return ()

    class EmptyFallbackBackend:
        def diarize(self, *args, **kwargs):
            return ()

    monkeypatch.setattr("say_transcribe.cli.transcribe", fake_transcribe)
    monkeypatch.setattr("say_transcribe.cli.PyannoteBackend", lambda **kwargs: PrimaryBackend())
    monkeypatch.setattr(
        "say_transcribe.cli.WavlmClusterBackend", lambda **kwargs: EmptyFallbackBackend()
    )
    monkeypatch.setattr("say_transcribe.cli.group_utterance_words", lambda seg: ())
    monkeypatch.setattr("say_transcribe.cli.project_morphosyntax", lambda *args, **kwargs: None)

    ret = main(["run", str(audio_file), "--channel", "0", "--out", str(tmp_path / "out")])

    assert ret == 0
    captured = capsys.readouterr()
    assert "[DIARIZATION_UNAVAILABLE:DEFAULTING_TO_PAR]" in captured.err
    assert "[PYANNOTE_UNAVAILABLE:USING_WAVLM_FALLBACK]" not in captured.err
    assert "private primary error" not in captured.err
    transcript = (tmp_path / "out" / "audio.cha").read_text()
    assert "speaker labels unavailable; defaulted to PAR; review before use" in transcript
    assert "auto-diarized" not in transcript
