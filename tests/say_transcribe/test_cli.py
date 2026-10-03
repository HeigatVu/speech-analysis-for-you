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

    monkeypatch.setattr("say_transcribe.cli.transcribe", fake_transcribe)
    monkeypatch.setattr("say_transcribe.cli.PyannoteBackend", lambda **kwargs: FakePyannote())
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
        "say_transcribe.cli.PyannoteBackend",
        lambda **kwargs: type("D", (), {"diarize": lambda self, _: ()})(),
    )
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


def test_cli_fails_redacted_when_diarization_backend_fails(
    tmp_path: Path, monkeypatch, capsys
):
    from say_transcribe.asr import AsrError, AsrResult, AsrSegment

    audio_file = _make_wav_file(tmp_path / "audio.wav")

    def fake_transcribe(*args, **kwargs):
        return AsrResult(
            source_sha256="0" * 64,
            segments=(AsrSegment(start_ms=0, end_ms=900, text="xin chào", words=()),),
            warnings=(),
        )

    class FailedBackend:
        def diarize(self, *args, **kwargs):
            raise AsrError(
                "MODEL_UNAVAILABLE", f"private text at {audio_file}: private transcript"
            )

    monkeypatch.setattr("say_transcribe.cli.transcribe", fake_transcribe)
    monkeypatch.setattr("say_transcribe.cli.PyannoteBackend", lambda **kwargs: FailedBackend())
    monkeypatch.setattr("say_transcribe.cli.group_utterance_words", lambda seg: ())
    monkeypatch.setattr("say_transcribe.cli.project_morphosyntax", lambda *args, **kwargs: None)

    ret = main(["run", str(audio_file), "--channel", "0", "--out", str(tmp_path / "out")])

    assert ret == 3
    captured = capsys.readouterr()
    assert "[MODEL_UNAVAILABLE]" in captured.err
    assert "USING_WAVLM_FALLBACK" not in captured.err
    assert str(tmp_path) not in captured.err
    assert "private text" not in captured.err
    assert "private transcript" not in captured.err
    assert not (tmp_path / "out" / "audio.cha").exists()


def test_cli_warns_when_diarization_returns_no_turns(
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

    class EmptyBackend:
        def diarize(self, *args, **kwargs):
            return ()

    monkeypatch.setattr("say_transcribe.cli.transcribe", fake_transcribe)
    monkeypatch.setattr("say_transcribe.cli.PyannoteBackend", lambda **kwargs: EmptyBackend())
    monkeypatch.setattr("say_transcribe.cli.group_utterance_words", lambda seg: ())
    monkeypatch.setattr("say_transcribe.cli.project_morphosyntax", lambda *args, **kwargs: None)

    ret = main(["run", str(audio_file), "--channel", "0", "--out", str(tmp_path / "out")])

    assert ret == 0
    captured = capsys.readouterr()
    assert "[DIARIZATION_UNAVAILABLE:DEFAULTING_TO_PAR]" in captured.err
    assert "USING_WAVLM_FALLBACK" not in captured.err
    transcript = (tmp_path / "out" / "audio.cha").read_text()
    assert "speaker labels unavailable; defaulted to PAR; review before use" in transcript
    assert "auto-diarized" not in transcript


def test_cli_diarizer_wavlm_selects_wavlm_backend(tmp_path: Path, monkeypatch):
    from say_transcribe.asr import AsrResult, AsrSegment

    audio_file = _make_wav_file(tmp_path / "audio.wav")
    called: dict[str, bool] = {"wavlm": False, "pyannote": False}

    def fake_transcribe(*args, **kwargs):
        return AsrResult(
            source_sha256="0" * 64,
            segments=(AsrSegment(start_ms=0, end_ms=900, text="xin chào", words=()),),
            warnings=(),
        )

    class RecordingWavlm:
        def __init__(self, **kwargs):
            called["wavlm"] = True

        def diarize(self, audio, sample_rate=16000, segments=None):
            assert segments is not None
            return ()

    def forbid_pyannote(**kwargs):
        called["pyannote"] = True
        raise AssertionError("pyannote must not be constructed when --diarizer wavlm")

    monkeypatch.setattr("say_transcribe.cli.transcribe", fake_transcribe)
    monkeypatch.setattr("say_transcribe.cli.WavlmClusterBackend", RecordingWavlm)
    monkeypatch.setattr("say_transcribe.cli.PyannoteBackend", forbid_pyannote)
    monkeypatch.setattr("say_transcribe.cli.group_utterance_words", lambda seg: ())
    monkeypatch.setattr("say_transcribe.cli.project_morphosyntax", lambda *args, **kwargs: None)

    ret = main(
        ["run", str(audio_file), "--channel", "0", "--out", str(tmp_path / "out"),
         "--diarizer", "wavlm"]
    )

    assert ret == 0
    assert called == {"wavlm": True, "pyannote": False}


def test_cli_asr_model_flag_on_run_compare_and_study():
    from say_transcribe.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(
        ["run", "a.wav", "--channel", "0", "--out", "o"]
    ).asr_model == "phowhisper-medium"
    assert parser.parse_args(
        ["run", "a.wav", "--channel", "0", "--out", "o", "--asr-model", "qwen3-asr"]
    ).asr_model == "qwen3-asr"
    assert parser.parse_args(
        ["compare", "a.wav", "--channel", "0", "--out", "o", "--expected-sha256", "0" * 64,
         "--asr-model", "phowhisper-large"]
    ).asr_model == "phowhisper-large"
    assert parser.parse_args(
        ["preprocess-study", "m.json", "--out", "o", "--asr-model", "qwen3-asr"]
    ).asr_model == "qwen3-asr"


class _FakeStanzaWord:
    def __init__(self, index: int, text: str) -> None:
        self.id = index
        self.text = text
        self.lemma = text
        self.upos = "noun"
        self.feats = None
        self.head = 0 if index == 1 else 1
        self.deprel = "root" if index == 1 else "nsubj"


class _FakeStanzaDoc:
    def __init__(self, tokens) -> None:
        words = [_FakeStanzaWord(i, t) for i, t in enumerate(tokens, start=1)]
        self.sentences = [type("S", (), {"words": words})()]


class _FakeStanzaBackend:
    def __init__(self, **kwargs) -> None:
        pass

    def parse_pretokenized(self, tokens):
        return _FakeStanzaDoc(list(tokens))


def _phase1_utterance():
    from say_transcribe.chat_writer import UtteranceRecord
    from say_transcribe.word_grouping import GroupedWord

    words = (
        GroupedWord("tôi", 0, 300, ()),
        GroupedWord("là_sinh_viên", 300, 900, ()),
        GroupedWord(".", None, None, ()),
    )
    return UtteranceRecord("PAR", 0, 900, "tôi là sinh viên .", words, None)


def _write_phase1(path: Path, session_id: str) -> Path:
    from say_transcribe.chat_writer import write_chat_file

    return write_chat_file(path, session_id, "c" * 64, (_phase1_utterance(),))


def test_cli_run_no_morphosyntax_defers_mor_gra(tmp_path: Path, monkeypatch, capsys):
    audio_file = _make_wav_file(tmp_path / "session_02_master.wav")
    out_dir = tmp_path / "out"

    from say_transcribe.asr import AsrResult, AsrSegment, WordTiming
    from say_transcribe.word_grouping import GroupedWord

    def fake_transcribe(*args, **kwargs):
        return AsrResult(
            source_sha256="b" * 64,
            segments=(
                AsrSegment(
                    start_ms=0,
                    end_ms=900,
                    text="tôi là sinh viên .",
                    words=(
                        WordTiming(word="tôi", start_ms=0, end_ms=300),
                        WordTiming(word="là", start_ms=300, end_ms=500),
                        WordTiming(word="sinh", start_ms=500, end_ms=700),
                        WordTiming(word="viên", start_ms=700, end_ms=900),
                        WordTiming(word=".", start_ms=None, end_ms=None),
                    ),
                ),
            ),
            warnings=(),
        )

    def forbid_stanza(*args, **kwargs):
        raise AssertionError("phase 1 must not load Stanza")

    monkeypatch.setattr("say_transcribe.cli.transcribe", fake_transcribe)
    monkeypatch.setattr(
        "say_transcribe.cli.PyannoteBackend",
        lambda **kwargs: type("D", (), {"diarize": lambda self, _: ()})(),
    )
    monkeypatch.setattr(
        "say_transcribe.cli.group_utterance_words",
        lambda seg: (
            GroupedWord("tôi", 0, 300, ()),
            GroupedWord("là_sinh_viên", 300, 900, ()),
            GroupedWord(".", None, None, ()),
        ),
    )
    monkeypatch.setattr("say_transcribe.cli.StanzaBackend", forbid_stanza)
    monkeypatch.setattr(
        "say_transcribe.cli.project_morphosyntax",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("deferred tier projected")),
    )

    ret = main(
        [
            "run",
            str(audio_file),
            "--channel",
            "0",
            "--out",
            str(out_dir),
            "--no-morphosyntax",
        ]
    )

    assert ret == 0
    text = (out_dir / "session_02.cha").read_text(encoding="utf-8")
    assert "*PAR:" in text
    assert "%wor:" in text
    assert "%mor" not in text
    assert "%gra" not in text


def test_cli_tag_adds_morphosyntax_to_reviewed_transcript(
    tmp_path: Path, monkeypatch, capsys
):
    from speech_features.formats.chat import decode_chat

    phase1 = _write_phase1(tmp_path / "review" / "p002.cha", "p002")
    # The reviewer corrects the speaker label; the timings and text stay as produced.
    reviewed = phase1.read_text(encoding="utf-8").replace("*PAR:", "*INV:")
    phase1.write_text(reviewed, encoding="utf-8")

    monkeypatch.setattr("say_transcribe.cli.StanzaBackend", _FakeStanzaBackend)
    out_file = tmp_path / "final" / "p002.cha"
    ret = main(["tag", str(phase1), "--out", str(out_file)])

    assert ret == 0
    tagged = out_file.read_text(encoding="utf-8")
    assert "*INV:" in tagged
    assert "*PAR:" not in tagged
    assert "%wor:" in tagged and "%mor:" in tagged and "%gra:" in tagged
    assert "\x150_300\x15" in tagged
    assert f"@Comment:\tsource_sha256 {'c' * 64}" in tagged

    document = decode_chat(tagged)
    assert document.utterances[0].speaker_id == "INV"
    assert [token.text for token in document.utterances[0].tokens] == [
        "tôi",
        "là_sinh_viên",
        ".",
    ]
    # The reviewed file itself is never modified (SPEC §8).
    assert phase1.read_text(encoding="utf-8") == reviewed

    ret = main(["tag", str(phase1), "--out", str(out_file)])
    assert ret == 2
    assert "[OUTPUT_EXISTS]" in capsys.readouterr().err


def test_cli_tag_rejects_speaker_code_outside_the_writer_contract(
    tmp_path: Path, monkeypatch, capsys
):
    phase1 = _write_phase1(tmp_path / "p003.cha", "p003")
    text = phase1.read_text(encoding="utf-8")
    text = text.replace(
        "@Participants:\tPAR Participant, INV Investigator",
        "@Participants:\tPAR Participant, INV Investigator, MED Media",
    ).replace("*PAR:", "*MED:")
    phase1.write_text(text, encoding="utf-8")

    monkeypatch.setattr("say_transcribe.cli.StanzaBackend", _FakeStanzaBackend)
    out_file = tmp_path / "final" / "p003.cha"
    ret = main(["tag", str(phase1), "--out", str(out_file)])

    assert ret == 4
    assert "[CHAT_VALIDATION_FAILED]" in capsys.readouterr().err
    assert not out_file.exists()


def test_cli_tag_reports_model_unavailable_instead_of_writing_untagged(
    tmp_path: Path, monkeypatch, capsys
):
    phase1 = _write_phase1(tmp_path / "p005.cha", "p005")

    class _FailingStanza:
        def __init__(self, **kwargs) -> None:
            pass

        def parse_pretokenized(self, tokens):
            raise RuntimeError("no Vietnamese Stanza resources")

    monkeypatch.setattr("say_transcribe.cli.StanzaBackend", _FailingStanza)
    out_file = tmp_path / "final" / "p005.cha"
    ret = main(["tag", str(phase1), "--out", str(out_file)])

    assert ret == 3
    assert "[MODEL_UNAVAILABLE]" in capsys.readouterr().err
    assert not out_file.exists()


def test_cli_tag_sends_only_real_words_to_stanza(tmp_path: Path, monkeypatch):
    seen: list[list[str]] = []

    class RecordingStanza(_FakeStanzaBackend):
        def parse_pretokenized(self, tokens):
            seen.append(list(tokens))
            return super().parse_pretokenized(tokens)

    phase1 = tmp_path / "review" / "p003.cha"
    _write_phase1(phase1, "p003")
    reviewed = phase1.read_text(encoding="utf-8").replace(
        "*PAR:\ttôi là_sinh_viên .", "*PAR:\t&-ờ tôi [/] tôi xxx (.) là_sinh_viên ."
    )
    lines = [line for line in reviewed.splitlines() if not line.startswith("%wor:")]
    phase1.write_text("\n".join(lines) + "\n", encoding="utf-8")

    monkeypatch.setattr("say_transcribe.cli.StanzaBackend", RecordingStanza)
    out_file = tmp_path / "final" / "p003.cha"
    assert main(["tag", str(phase1), "--out", str(out_file)]) == 0

    assert seen == [["tôi", "là_sinh_viên", "."]]
    tagged = out_file.read_text(encoding="utf-8")
    assert "*PAR:\t&-ờ tôi [/] tôi xxx (.) là_sinh_viên ." in tagged
    assert tagged.count("%mor:") == 1
