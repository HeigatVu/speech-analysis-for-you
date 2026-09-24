from types import SimpleNamespace

import numpy as np

from say_transcribe import cli
from say_transcribe.asr import AsrResult


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
    expected_hash = "a" * 64
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
    monkeypatch.setattr(cli, "PhoWhisperBackend", lambda **kwargs: backend)

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
    expected_hash = "a" * 64
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
    expected_hash = "a" * 64
    output_dir = tmp_path / "comparison"
    (output_dir / "baseline").mkdir(parents=True)
    (output_dir / "baseline" / "p001.cha").write_text("earlier automatic version", encoding="utf-8")
    audio_16k = np.zeros(16_000, dtype=np.float32)
    backend = object()
    baseline = AsrResult(expected_hash, (), ())

    monkeypatch.setattr(cli, "compute_sha256", lambda _: expected_hash)
    monkeypatch.setattr(cli, "PhoWhisperBackend", lambda **kwargs: backend)
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
    expected_hash = "a" * 64
    output_dir = tmp_path / "comparison"
    (output_dir / "vad_asr_aligned").mkdir(parents=True)
    (output_dir / "vad_asr_aligned" / "p001.cha").write_text("existing aligned", encoding="utf-8")
    audio_16k = np.zeros(16_000, dtype=np.float32)
    backend = object()
    baseline = AsrResult(expected_hash, (), ())

    monkeypatch.setattr(cli, "compute_sha256", lambda _: expected_hash)
    monkeypatch.setattr(cli, "PhoWhisperBackend", lambda **kwargs: backend)
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
