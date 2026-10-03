from types import SimpleNamespace

import numpy as np

from say_transcribe import cli
from say_transcribe.asr import AsrResult, compute_sha256


def test_compare_hash_mismatch_stops_before_audio_decode(tmp_path, monkeypatch, capsys):
    audio_path = tmp_path / "p001_master.wav"
    audio_path.write_bytes(b"synthetic master")
    monkeypatch.setattr(cli, "compute_sha256", lambda _: "a" * 64)

    def unexpected_decode(*args, **kwargs):
        raise AssertionError("audio decoded before the approved hash matched")

    monkeypatch.setattr(cli, "read_wav", unexpected_decode)
    monkeypatch.setattr(cli, "transcribe", unexpected_decode)

    result = cli.main(
        [
            "compare",
            str(audio_path),
            "--channel",
            "0",
            "--out",
            str(tmp_path / "out"),
            "--expected-sha256",
            "b" * 64,
        ]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert "[SOURCE_HASH_MISMATCH]" in captured.err
    assert str(audio_path) not in captured.err
    assert not (tmp_path / "out").exists()


def test_compare_writes_three_variants_and_reuses_vad_asr_cache(tmp_path, monkeypatch, capsys):
    audio_path = tmp_path / "p001_master.wav"
    audio_path.write_bytes(b"synthetic master")
    expected_hash = compute_sha256(audio_path)
    output_dir = tmp_path / "comparison"
    audio_16k = np.zeros(16_000, dtype=np.float32)
    calls = {
        "baseline_backend": None,
        "window_asr": [],
        "window_results": [],
        "vad_thresholds": [],
    }
    backend = object()
    baseline = AsrResult(expected_hash, (), ())
    vad_result = AsrResult(expected_hash, (), ())

    monkeypatch.setattr(cli, "compute_sha256", lambda _: expected_hash)
    monkeypatch.setattr(cli, "make_asr_backend", lambda *args, **kwargs: backend)

    def fake_transcribe(**kwargs):
        calls["baseline_backend"] = kwargs["backend"]
        return baseline

    monkeypatch.setattr(cli, "transcribe", fake_transcribe)
    monkeypatch.setattr(
        cli,
        "read_wav",
        lambda _: SimpleNamespace(sample_rate=16_000, sample_width=2),
    )
    monkeypatch.setattr(cli, "extract_channel", lambda audio, channel: np.zeros(10))
    monkeypatch.setattr(cli, "resample_to_16kHz", lambda *args: audio_16k)

    def fake_speech_windows(audio, *, threshold):
        calls["vad_thresholds"].append(threshold)
        return [(0, len(audio))]

    monkeypatch.setattr(cli, "get_speech_windows", fake_speech_windows)

    cached_asr = object()

    def fake_transcribe_windows(audio, windows, asr_backend):
        calls["window_asr"].append((audio, windows, asr_backend))
        return cached_asr

    def fake_result_from_windows(audio, source_sha256, cached, *, aligner=None):
        calls["window_results"].append((audio, source_sha256, cached, aligner))
        if aligner is not None:
            raise cli.AlignmentError()
        return vad_result

    monkeypatch.setattr(cli, "transcribe_windows", fake_transcribe_windows)
    monkeypatch.setattr(cli, "result_from_windows", fake_result_from_windows)
    monkeypatch.setattr(
        cli, "PyannoteBackend", lambda **kwargs: type("D", (), {"diarize": lambda self, _: ()})()
    )
    monkeypatch.setattr(cli, "StanzaBackend", lambda **kwargs: object())
    result = cli.main(
        [
            "compare",
            str(audio_path),
            "--channel",
            "0",
            "--out",
            str(output_dir),
            "--expected-sha256",
            expected_hash,
            "--vad-threshold",
            "0.35",
        ]
    )

    assert result == 0
    assert len(calls["window_asr"]) == 1
    assert calls["baseline_backend"] is backend
    assert calls["vad_thresholds"] == [0.35]
    assert calls["window_asr"][0][2] is backend
    assert [call[2] for call in calls["window_results"]] == [cached_asr, cached_asr]
    assert [call[3] for call in calls["window_results"]] == [None, cli.align_words]
    for variant in ("baseline", "vad_asr", "vad_asr_aligned"):
        assert (output_dir / variant / "p001.cha").is_file()
    captured = capsys.readouterr()
    assert "[DIARIZATION_UNAVAILABLE:DEFAULTING_TO_PAR]" in captured.err
    assert "[ALIGNMENT_UNAVAILABLE]" in captured.err


def test_compare_rejects_out_of_range_threshold_before_any_asr(tmp_path, monkeypatch, capsys):
    audio_path = tmp_path / "p001_master.wav"
    audio_path.write_bytes(b"synthetic master")
    expected_hash = compute_sha256(audio_path)
    monkeypatch.setattr(cli, "compute_sha256", lambda _: expected_hash)

    def unexpected_transcribe(*args, **kwargs):
        raise AssertionError("ASR ran before the VAD threshold was validated")

    monkeypatch.setattr(cli, "transcribe", unexpected_transcribe)

    result = cli.main(
        [
            "compare",
            str(audio_path),
            "--channel",
            "0",
            "--out",
            str(tmp_path / "out"),
            "--expected-sha256",
            expected_hash,
            "--vad-threshold",
            "1.5",
        ]
    )

    assert result == 2
    captured = capsys.readouterr()
    assert "[INVALID_ARGUMENT]" in captured.err
    assert not (tmp_path / "out").exists()


def test_compare_refuses_to_overwrite_an_earlier_variant(tmp_path, monkeypatch, capsys):
    audio_path = tmp_path / "p001_master.wav"
    audio_path.write_bytes(b"synthetic master")
    expected_hash = compute_sha256(audio_path)
    output_dir = tmp_path / "comparison"
    (output_dir / "baseline").mkdir(parents=True)
    (output_dir / "baseline" / "p001.cha").write_text("earlier automatic version", encoding="utf-8")
    audio_16k = np.zeros(16_000, dtype=np.float32)
    backend = object()
    baseline = AsrResult(expected_hash, (), ())

    monkeypatch.setattr(cli, "compute_sha256", lambda _: expected_hash)
    monkeypatch.setattr(cli, "make_asr_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(cli, "transcribe", lambda **kwargs: baseline)
    monkeypatch.setattr(
        cli, "read_wav", lambda _: SimpleNamespace(sample_rate=16_000, sample_width=2)
    )
    monkeypatch.setattr(cli, "extract_channel", lambda audio, channel: np.zeros(10))
    monkeypatch.setattr(cli, "resample_to_16kHz", lambda *args: audio_16k)
    monkeypatch.setattr(cli, "get_speech_windows", lambda audio, *, threshold: [(0, len(audio))])
    monkeypatch.setattr(cli, "transcribe_windows", lambda audio, windows, backend: object())
    monkeypatch.setattr(cli, "result_from_windows", lambda *args, **kwargs: baseline)
    monkeypatch.setattr(
        cli, "PyannoteBackend", lambda **kwargs: type("D", (), {"diarize": lambda self, _: ()})()
    )
    monkeypatch.setattr(cli, "StanzaBackend", lambda **kwargs: object())

    result = cli.main(
        [
            "compare",
            str(audio_path),
            "--channel",
            "0",
            "--out",
            str(output_dir),
            "--expected-sha256",
            expected_hash,
        ]
    )

    assert result == 2
    captured = capsys.readouterr()
    assert "[OUTPUT_EXISTS]" in captured.err
    assert (output_dir / "baseline" / "p001.cha").read_text(encoding="utf-8") == (
        "earlier automatic version"
    )


def test_compare_refuses_to_overwrite_when_later_variant_exists(tmp_path, monkeypatch, capsys):
    audio_path = tmp_path / "p001_master.wav"
    audio_path.write_bytes(b"synthetic master")
    expected_hash = compute_sha256(audio_path)
    output_dir = tmp_path / "comparison"
    (output_dir / "vad_asr_aligned").mkdir(parents=True)
    (output_dir / "vad_asr_aligned" / "p001.cha").write_text("existing aligned", encoding="utf-8")
    audio_16k = np.zeros(16_000, dtype=np.float32)
    backend = object()
    baseline = AsrResult(expected_hash, (), ())

    monkeypatch.setattr(cli, "compute_sha256", lambda _: expected_hash)
    monkeypatch.setattr(cli, "make_asr_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(cli, "transcribe", lambda **kwargs: baseline)
    monkeypatch.setattr(
        cli, "read_wav", lambda _: SimpleNamespace(sample_rate=16_000, sample_width=2)
    )
    monkeypatch.setattr(cli, "extract_channel", lambda audio, channel: np.zeros(10))
    monkeypatch.setattr(cli, "resample_to_16kHz", lambda *args: audio_16k)
    monkeypatch.setattr(cli, "get_speech_windows", lambda audio, *, threshold: [(0, len(audio))])
    monkeypatch.setattr(cli, "transcribe_windows", lambda audio, windows, backend: object())
    monkeypatch.setattr(cli, "result_from_windows", lambda *args, **kwargs: baseline)
    monkeypatch.setattr(
        cli, "PyannoteBackend", lambda **kwargs: type("D", (), {"diarize": lambda self, _: ()})()
    )
    monkeypatch.setattr(cli, "StanzaBackend", lambda **kwargs: object())

    result = cli.main(
        [
            "compare",
            str(audio_path),
            "--channel",
            "0",
            "--out",
            str(output_dir),
            "--expected-sha256",
            expected_hash,
        ]
    )

    assert result == 2
    captured = capsys.readouterr()
    assert "[OUTPUT_EXISTS]" in captured.err
    # Baseline must NOT have been written
    assert not (output_dir / "baseline" / "p001.cha").exists()


def test_compare_no_morphosyntax_writes_phase1_variants(tmp_path, monkeypatch):
    """--no-morphosyntax defers %mor/%gra for every compare variant, without Stanza."""
    from say_transcribe.asr import AsrSegment, WordTiming
    from say_transcribe.word_grouping import GroupedWord

    audio_path = tmp_path / "p004_master.wav"
    audio_path.write_bytes(b"synthetic master")
    expected_hash = compute_sha256(audio_path)
    output_dir = tmp_path / "comparison"
    segment = AsrSegment(
        start_ms=0,
        end_ms=300,
        text="tôi .",
        words=(WordTiming(word="tôi", start_ms=0, end_ms=300), WordTiming(word=".", start_ms=None, end_ms=None)),
    )
    result = AsrResult(expected_hash, (segment,), ())

    monkeypatch.setattr(cli, "compute_sha256", lambda _: expected_hash)
    monkeypatch.setattr(cli, "make_asr_backend", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "transcribe", lambda **kwargs: result)
    monkeypatch.setattr(
        cli, "read_wav", lambda _: SimpleNamespace(sample_rate=16_000, sample_width=2)
    )
    monkeypatch.setattr(cli, "extract_channel", lambda audio, channel: np.zeros(10))
    monkeypatch.setattr(
        cli, "resample_to_16kHz", lambda *args: np.zeros(16_000, dtype=np.float32)
    )
    monkeypatch.setattr(cli, "get_speech_windows", lambda audio, *, threshold: [(0, 16_000)])
    monkeypatch.setattr(cli, "transcribe_windows", lambda *args: object())
    monkeypatch.setattr(cli, "result_from_windows", lambda *args, **kwargs: result)
    monkeypatch.setattr(
        cli, "PyannoteBackend", lambda **kwargs: type("D", (), {"diarize": lambda self, _: ()})()
    )
    monkeypatch.setattr(
        cli,
        "group_utterance_words",
        lambda seg: (GroupedWord("tôi", 0, 300, ()), GroupedWord(".", None, None, ())),
    )

    def forbid_stanza(*args, **kwargs):
        raise AssertionError("phase 1 must not load Stanza")

    monkeypatch.setattr(cli, "StanzaBackend", forbid_stanza)
    monkeypatch.setattr(
        cli,
        "project_morphosyntax",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("deferred tier projected")),
    )

    code = cli.main(
        [
            "compare",
            str(audio_path),
            "--channel",
            "0",
            "--out",
            str(output_dir),
            "--expected-sha256",
            expected_hash,
            "--no-morphosyntax",
        ]
    )

    assert code == 0
    for variant in ("baseline", "vad_asr", "vad_asr_aligned"):
        text = (output_dir / variant / "p004.cha").read_text(encoding="utf-8")
        assert "%wor:" in text
        assert "%mor" not in text
        assert "%gra" not in text


def test_comparison_variant_cleans_words_and_tags_only_real_words(tmp_path, monkeypatch):
    from say_transcribe.asr import AsrResult, AsrSegment
    from say_transcribe.cli import _write_comparison_result
    from say_transcribe.word_grouping import GroupedWord

    seen = []

    class RecordingStanza:
        def parse_pretokenized(self, tokens):
            seen.append(list(tokens))
            words = [
                type("W", (), dict(id=i, text=t, lemma=t, upos="noun", feats=None,
                                   head=0 if i == 1 else 1, deprel="root" if i == 1 else "dep"))()
                for i, t in enumerate(tokens, start=1)
            ]
            return type("D", (), {"sentences": [type("S", (), {"words": words})()]})()

    spoken = [",", "ờ", "tôi", "tôi", "xxx", "đi", "."]
    monkeypatch.setattr(
        "say_transcribe.cli.group_utterance_words",
        lambda segment: tuple(
            GroupedWord(w, None if w in {",", "xxx", "."} else i * 100, None if w in {",", "xxx", "."} else i * 100 + 90, ())
            for i, w in enumerate(spoken)
        ),
    )
    result = AsrResult("a" * 64, (AsrSegment(0, 700, " ".join(spoken), ()),), ())
    out = tmp_path / "s1.cha"

    _write_comparison_result(result, out, "s1", (), "cpu", RecordingStanza(), [])

    text = out.read_text(encoding="utf-8")
    assert "*PAR:\t&-ờ tôi [/] tôi xxx đi . " in text
    assert seen == [["tôi", "đi", "."]]
