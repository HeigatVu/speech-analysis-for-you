import pytest
from say_transcribe import asr
from say_transcribe.cli import main


@pytest.mark.parametrize(
    "approved,code", [("0" * 64, "SOURCE_HASH_MISMATCH"), ("invalid", "INVALID_ARGUMENT")]
)
def test_transcribe_hash_gate_before_decoder_or_backend(tmp_path, monkeypatch, approved, code):
    path = tmp_path / "private.wav"
    path.write_bytes(b"master")

    def forbidden(*args, **kwargs):
        pytest.fail("decoder or backend reached before hash gate")

    monkeypatch.setattr(asr, "read_wav", forbidden)
    monkeypatch.setattr(asr, "PhoWhisperBackend", forbidden)
    with pytest.raises(asr.AsrError) as caught:
        asr.transcribe(path, expected_sha256=approved)
    assert caught.value.code == code
    assert str(path) not in str(caught.value)


def test_run_hash_gate_before_first_decoder(tmp_path, monkeypatch, capsys):
    from say_transcribe import cli

    path = tmp_path / "private.wav"
    path.write_bytes(b"master")

    def forbidden(*args, **kwargs):
        pytest.fail("decoder or model reached before hash gate")

    monkeypatch.setattr(cli, "read_wav", forbidden)
    monkeypatch.setattr(cli, "make_asr_backend", forbidden)
    assert (
        main(
            [
                "run",
                str(path),
                "--channel",
                "0",
                "--out",
                str(tmp_path / "out"),
                "--expected-sha256",
                "0" * 64,
            ]
        )
        == 2
    )
    error = capsys.readouterr().err
    assert "[SOURCE_HASH_MISMATCH]" in error
    assert str(path) not in error


def test_transcribe_rechecks_master_after_backend(tmp_path, monkeypatch):
    import numpy as np
    from types import SimpleNamespace

    path = tmp_path / "master.wav"
    path.write_bytes(b"approved")
    approved = asr.compute_sha256(path)
    monkeypatch.setattr(
        asr, "read_wav", lambda path: SimpleNamespace(sample_rate=16000, sample_width=2)
    )
    monkeypatch.setattr(asr, "extract_channel", lambda audio, channel: np.zeros(160))
    monkeypatch.setattr(asr, "resample_to_16kHz", lambda *args: np.zeros(160))

    class MutatingBackend:
        def transcribe_audio(self, samples):
            path.write_bytes(b"changed")
            return ()

    with pytest.raises(asr.AsrError) as caught:
        asr.transcribe(path, backend=MutatingBackend(), expected_sha256=approved.upper())
    assert caught.value.code == "SOURCE_HASH_MISMATCH"


def test_run_requires_approved_hash(tmp_path, capsys):
    assert main(["run", "master.wav", "--channel", "0", "--out", str(tmp_path)]) == 2
    assert "--expected-sha256" in capsys.readouterr().err


def test_tag_redacts_raw_validation_exception(tmp_path, monkeypatch, capsys):
    from say_transcribe import cli

    path = tmp_path / "private.cha"
    path.write_text("private transcript")

    def broken(*args, **kwargs):
        raise OSError(str(path) + " private transcript")

    monkeypatch.setattr(cli, "decode_chat", broken)
    assert main(["tag", str(path), "--out", str(tmp_path / "out")]) == 4
    error = capsys.readouterr().err
    assert "[CHAT_VALIDATION_FAILED]" in error
    assert str(path) not in error
    assert "private transcript" not in error


def test_run_redacts_backend_error_message(tmp_path, monkeypatch, capsys):
    from say_transcribe import cli
    import wave

    path = tmp_path / "master.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * 160)

    def broken(*args, **kwargs):
        raise asr.AsrError("MODEL_UNAVAILABLE", str(path) + " private transcript")

    monkeypatch.setattr(cli, "make_asr_backend", broken)
    assert (
        main(
            [
                "run",
                str(path),
                "--channel",
                "0",
                "--out",
                str(tmp_path / "out"),
                "--expected-sha256",
                asr.compute_sha256(path),
            ]
        )
        == 3
    )
    error = capsys.readouterr().err
    assert "[MODEL_UNAVAILABLE]" in error
    assert str(path) not in error
    assert "private transcript" not in error
