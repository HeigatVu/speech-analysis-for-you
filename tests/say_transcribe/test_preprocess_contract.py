from __future__ import annotations

import subprocess
import wave
from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from say_transcribe.audio_profile import (
    AudioProfileError,
    TELL_INSPIRED_V1,
    get_audio_profile,
)
from say_transcribe.preprocess import (
    PreprocessError,
    check_sox_support,
    run_sox_gsm_stage,
)


def _make_mono_wav(
    path: Path, sample_rate: int = 16000, n_samples: int = 16000, sample_width: int = 2
) -> None:
    """Create a synthetic mono PCM16 wav."""
    t = np.linspace(0, n_samples / sample_rate, n_samples, endpoint=False)
    data = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())


def _make_stereo_wav(path: Path, sample_rate: int = 16000, n_samples: int = 16000) -> None:
    """Create a synthetic stereo PCM16 wav."""
    data = np.zeros((n_samples, 2), dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())


def _build_fake_sox_runner(
    target_samples: int = 16000,
    clipped: bool = False,
    command_log: list[list[str]] | None = None,
    fail_stage: int | None = None,
    timeout_stage: int | None = None,
    stderr_text: str = "sox FATAL formats.c:262 sox_open_read failed",
) -> Callable[..., MagicMock]:
    """Returns a fake subprocess.run implementation simulating SoX behavior.

    fail_stage / timeout_stage select which conversion call (1 = GSM encode,
    2 = PCM decode) fails or times out; the probe call is never affected.
    """
    stage = {"n": 0}

    def fake_run(cmd: list[str], **kwargs: Any) -> MagicMock:
        if command_log is not None:
            command_log.append(list(cmd))

        if "--help-format" in cmd:
            return MagicMock(returncode=0, stdout="GSM 06.10 format supported", stderr="")

        stage["n"] += 1
        current = stage["n"]

        if timeout_stage == current:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 30.0))
        if fail_stage == current:
            return MagicMock(returncode=1, stdout="", stderr=stderr_text)

        # Find output wav if this is the decode step
        for i, arg in enumerate(cmd):
            if arg.endswith(".wav") and i > 2:
                if clipped:
                    data = np.full(target_samples, 32767, dtype=np.int16)
                    with wave.open(arg, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes(data.tobytes())
                else:
                    _make_mono_wav(Path(arg), sample_rate=16000, n_samples=target_samples)
                break

        return MagicMock(returncode=0, stdout="", stderr="")

    return fake_run


class TestAudioProfile:
    def test_tell_inspired_v1_defaults(self) -> None:
        profile = TELL_INSPIRED_V1
        assert profile.profile_id == "tell-inspired-v1"
        assert profile.target_sample_rate == 16000
        assert profile.highpass_hz == 200.0
        assert profile.lowpass_hz == 3400.0
        assert profile.max_trailing_padding_samples == 319
        assert profile.gsm_sample_rate == 8000

    def test_audio_profile_immutable(self) -> None:
        profile = TELL_INSPIRED_V1
        with pytest.raises(Exception):
            profile.target_sample_rate = 8000  # type: ignore[misc]

    def test_get_audio_profile(self) -> None:
        p = get_audio_profile("tell-inspired-v1")
        assert p == TELL_INSPIRED_V1
        with pytest.raises(AudioProfileError) as exc_info:
            get_audio_profile("unknown-profile")
        assert exc_info.value.code == "UNKNOWN_PROFILE"


class TestSoxSupportChecks:
    def test_sox_unavailable_raises_stable_code(self) -> None:
        with patch("shutil.which", return_value=None):
            with pytest.raises(PreprocessError) as exc_info:
                check_sox_support("sox")
            assert exc_info.value.code == "SOX_UNAVAILABLE"

    def test_sox_gsm_unavailable_raises_stable_code(self) -> None:
        with patch("shutil.which", return_value="/usr/bin/sox"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Unsupported")
                with pytest.raises(PreprocessError) as exc_info:
                    check_sox_support("sox")
                assert exc_info.value.code == "SOX_GSM_UNAVAILABLE"

    def test_sox_gsm_available_passes(self) -> None:
        with patch("shutil.which", return_value="/usr/bin/sox"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="GSM 06.10", stderr="")
                assert check_sox_support("sox") == "/usr/bin/sox"

    def test_sox_probe_timeout_raises_stable_code(self) -> None:
        with patch("shutil.which", return_value="/usr/bin/sox"):
            with patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired("sox", 30.0),
            ):
                with pytest.raises(PreprocessError) as exc_info:
                    check_sox_support("sox")
            assert exc_info.value.code == "SOX_TIMEOUT"


class TestSoxGsmStageSubprocessWrapper:
    def test_rejects_multi_channel_input(self, tmp_path: Path) -> None:
        stereo_wav = tmp_path / "stereo.wav"
        out_wav = tmp_path / "out.wav"
        _make_stereo_wav(stereo_wav)

        with pytest.raises(PreprocessError) as exc_info:
            run_sox_gsm_stage(stereo_wav, out_wav)
        assert exc_info.value.code == "INVALID_AUDIO_CHANNEL"

    def test_rejects_nonexistent_input(self, tmp_path: Path) -> None:
        with pytest.raises(PreprocessError) as exc_info:
            run_sox_gsm_stage(tmp_path / "nonexistent.wav", tmp_path / "out.wav")
        assert exc_info.value.code == "INPUT_FILE_NOT_FOUND"

    def test_rejects_output_path_equal_to_input(self, tmp_path: Path) -> None:
        in_wav = tmp_path / "in.wav"
        _make_mono_wav(in_wav)

        with pytest.raises(PreprocessError) as exc_info:
            run_sox_gsm_stage(in_wav, in_wav)
        assert exc_info.value.code == "OUTPUT_PATH_CONFLICT"

    def test_subprocess_commands_and_no_compand(self, tmp_path: Path) -> None:
        in_wav = tmp_path / "in.wav"
        out_wav = tmp_path / "out.wav"
        _make_mono_wav(in_wav, sample_rate=16000, n_samples=16000)

        commands_run: list[list[str]] = []
        runner = _build_fake_sox_runner(target_samples=16000, command_log=commands_run)

        with patch("shutil.which", return_value="/usr/bin/sox"), \
             patch("subprocess.run", side_effect=runner):
            result = run_sox_gsm_stage(in_wav, out_wav)
            assert result == out_wav
            assert out_wav.is_file()

        # Check that probe + 2 conversion calls were made
        assert len(commands_run) == 3
        probe_cmd, cmd1, cmd2 = commands_run

        assert "--help-format" in probe_cmd

        # Step 1: encode to GSM (8 kHz mono)
        assert cmd1[0] == "/usr/bin/sox"
        assert "-r" in cmd1 and cmd1[cmd1.index("-r") + 1] == "8000"
        assert "-c" in cmd1 and cmd1[cmd1.index("-c") + 1] == "1"

        # Step 2: decode to 16 kHz PCM16 with highpass and lowpass
        assert cmd2[0] == "/usr/bin/sox"
        assert "-b" in cmd2 and cmd2[cmd2.index("-b") + 1] == "16"
        assert "-e" in cmd2 and cmd2[cmd2.index("-e") + 1] == "signed-integer"
        assert "rate" in cmd2 and "-v" in cmd2 and cmd2[cmd2.index("-v") + 1] == "16000"
        assert "highpass" in cmd2 and cmd2[cmd2.index("highpass") + 1] == "200"
        assert "lowpass" in cmd2 and cmd2[cmd2.index("lowpass") + 1] == "3400"

        # Explicitly verify NO compand in any stage
        for cmd in commands_run:
            assert "compand" not in cmd
            assert "mcompand" not in cmd

    def test_preserves_native_input_file(self, tmp_path: Path) -> None:
        in_wav = tmp_path / "in.wav"
        out_wav = tmp_path / "out.wav"
        _make_mono_wav(in_wav, sample_rate=16000, n_samples=16000)
        original_bytes = in_wav.read_bytes()

        runner = _build_fake_sox_runner(target_samples=16000)
        with patch("shutil.which", return_value="/usr/bin/sox"), \
             patch("subprocess.run", side_effect=runner):
            run_sox_gsm_stage(in_wav, out_wav)

        assert in_wav.read_bytes() == original_bytes

    def test_acceptable_trailing_gsm_padding_drift(self, tmp_path: Path) -> None:
        in_wav = tmp_path / "in.wav"
        out_wav = tmp_path / "out.wav"
        _make_mono_wav(in_wav, sample_rate=16000, n_samples=16000)

        # Output with +300 samples (within 0..319 allowable drift)
        runner = _build_fake_sox_runner(target_samples=16300)
        with patch("shutil.which", return_value="/usr/bin/sox"), \
             patch("subprocess.run", side_effect=runner):
            run_sox_gsm_stage(in_wav, out_wav)
            assert out_wav.is_file()

    def test_excessive_drift_raises_alignment_error(self, tmp_path: Path) -> None:
        in_wav = tmp_path / "in.wav"
        out_wav = tmp_path / "out.wav"
        _make_mono_wav(in_wav, sample_rate=16000, n_samples=16000)

        # Output with +400 samples drift (> 319 allowed)
        runner = _build_fake_sox_runner(target_samples=16400)
        with patch("shutil.which", return_value="/usr/bin/sox"), \
             patch("subprocess.run", side_effect=runner):
            with pytest.raises(PreprocessError) as exc_info:
                run_sox_gsm_stage(in_wav, out_wav)
            assert exc_info.value.code == "PREPROCESS_ALIGNMENT_ERROR"

    def test_negative_drift_raises_alignment_error(self, tmp_path: Path) -> None:
        in_wav = tmp_path / "in.wav"
        out_wav = tmp_path / "out.wav"
        _make_mono_wav(in_wav, sample_rate=16000, n_samples=16000)

        # Output shortened by 100 samples (negative drift)
        runner = _build_fake_sox_runner(target_samples=15900)
        with patch("shutil.which", return_value="/usr/bin/sox"), \
             patch("subprocess.run", side_effect=runner):
            with pytest.raises(PreprocessError) as exc_info:
                run_sox_gsm_stage(in_wav, out_wav)
            assert exc_info.value.code == "PREPROCESS_ALIGNMENT_ERROR"

    def test_clipping_detection_raises_error(self, tmp_path: Path) -> None:
        in_wav = tmp_path / "in.wav"
        out_wav = tmp_path / "out.wav"
        _make_mono_wav(in_wav, sample_rate=16000, n_samples=16000)

        runner = _build_fake_sox_runner(target_samples=16000, clipped=True)
        with patch("shutil.which", return_value="/usr/bin/sox"), \
             patch("subprocess.run", side_effect=runner):
            with pytest.raises(PreprocessError) as exc_info:
                run_sox_gsm_stage(in_wav, out_wav)
            assert exc_info.value.code == "CLIPPING_ERROR"


class TestErrorMessageRedaction:
    """AGENTS.md 'Strict redaction': errors must never carry absolute paths
    or raw subprocess/exception text, which may leak clip filenames."""

    def test_missing_input_file_does_not_leak_path(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.wav"
        with pytest.raises(PreprocessError) as exc_info:
            run_sox_gsm_stage(missing, tmp_path / "out.wav")
        assert str(tmp_path) not in str(exc_info.value)
        assert str(tmp_path) not in exc_info.value.message

    def test_encode_failure_does_not_leak_sox_stderr(self, tmp_path: Path) -> None:
        in_wav = tmp_path / "in.wav"
        out_wav = tmp_path / "out.wav"
        _make_mono_wav(in_wav, sample_rate=16000, n_samples=16000)

        # sensitive-data-guard: allow — synthetic stderr fixture, not a real path/credential
        secret = "sox FATAL /home/ducvu/private/patient_session_01.wav: no such file"
        runner = _build_fake_sox_runner(fail_stage=1, stderr_text=secret)
        with patch("shutil.which", return_value="/usr/bin/sox"), \
             patch("subprocess.run", side_effect=runner):
            with pytest.raises(PreprocessError) as exc_info:
                run_sox_gsm_stage(in_wav, out_wav)
            assert exc_info.value.code == "SOX_EXECUTION_FAILED"
            assert secret not in str(exc_info.value)
            assert secret not in exc_info.value.message


class TestSoxGsmStageFailureModes:
    """Stable error codes with no prior coverage: timeouts, execution failure
    on the decode stage, and invalid conversion output."""

    def test_encode_timeout_raises_stable_code(self, tmp_path: Path) -> None:
        in_wav = tmp_path / "in.wav"
        out_wav = tmp_path / "out.wav"
        _make_mono_wav(in_wav, sample_rate=16000, n_samples=16000)

        runner = _build_fake_sox_runner(timeout_stage=1)
        with patch("shutil.which", return_value="/usr/bin/sox"), \
             patch("subprocess.run", side_effect=runner):
            with pytest.raises(PreprocessError) as exc_info:
                run_sox_gsm_stage(in_wav, out_wav)
            assert exc_info.value.code == "SOX_TIMEOUT"

    def test_decode_timeout_raises_stable_code(self, tmp_path: Path) -> None:
        in_wav = tmp_path / "in.wav"
        out_wav = tmp_path / "out.wav"
        _make_mono_wav(in_wav, sample_rate=16000, n_samples=16000)

        runner = _build_fake_sox_runner(timeout_stage=2)
        with patch("shutil.which", return_value="/usr/bin/sox"), \
             patch("subprocess.run", side_effect=runner):
            with pytest.raises(PreprocessError) as exc_info:
                run_sox_gsm_stage(in_wav, out_wav)
            assert exc_info.value.code == "SOX_TIMEOUT"

    def test_decode_execution_failure_raises_stable_code(self, tmp_path: Path) -> None:
        in_wav = tmp_path / "in.wav"
        out_wav = tmp_path / "out.wav"
        _make_mono_wav(in_wav, sample_rate=16000, n_samples=16000)

        runner = _build_fake_sox_runner(fail_stage=2)
        with patch("shutil.which", return_value="/usr/bin/sox"), \
             patch("subprocess.run", side_effect=runner):
            with pytest.raises(PreprocessError) as exc_info:
                run_sox_gsm_stage(in_wav, out_wav)
            assert exc_info.value.code == "SOX_EXECUTION_FAILED"

    def test_wrong_output_sample_rate_raises_invalid_artifact(self, tmp_path: Path) -> None:
        # core-logic-test-guard: allow — asserts exc_info.value.code below;
        # the nested fake_run def confuses the static check.
        in_wav = tmp_path / "in.wav"
        out_wav = tmp_path / "out.wav"
        _make_mono_wav(in_wav, sample_rate=16000, n_samples=16000)

        def fake_run(cmd: list[str], **kwargs: Any) -> MagicMock:
            if "--help-format" in cmd:
                return MagicMock(returncode=0, stdout="GSM 06.10", stderr="")
            for i, arg in enumerate(cmd):
                if arg.endswith(".wav") and i > 2:
                    # Wrong sample rate: SoX claimed 16k but wrote 8k.
                    _make_mono_wav(Path(arg), sample_rate=8000, n_samples=16000)
                    break
            return MagicMock(returncode=0, stdout="", stderr="")

        # core-logic-test-guard: allow — asserts on exc_info.value.code below;
        # the nested fake_run def appears to confuse the static check.
        with patch("shutil.which", return_value="/usr/bin/sox"), \
             patch("subprocess.run", side_effect=fake_run):
            with pytest.raises(PreprocessError) as exc_info:
                run_sox_gsm_stage(in_wav, out_wav)
            assert exc_info.value.code == "INVALID_PREPROCESS_ARTIFACT"

    def test_unreadable_output_wav_raises_invalid_artifact(self, tmp_path: Path) -> None:
        # core-logic-test-guard: allow — asserts exc_info.value.code below;
        # the nested fake_run def confuses the static check.
        in_wav = tmp_path / "in.wav"
        out_wav = tmp_path / "out.wav"
        _make_mono_wav(in_wav, sample_rate=16000, n_samples=16000)

        def fake_run(cmd: list[str], **kwargs: Any) -> MagicMock:
            if "--help-format" in cmd:
                return MagicMock(returncode=0, stdout="GSM 06.10", stderr="")
            for i, arg in enumerate(cmd):
                if arg.endswith(".wav") and i > 2:
                    Path(arg).write_bytes(b"not a wav file")
                    break
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("shutil.which", return_value="/usr/bin/sox"), \
             patch("subprocess.run", side_effect=fake_run):
            with pytest.raises(PreprocessError) as exc_info:
                run_sox_gsm_stage(in_wav, out_wav)
            assert exc_info.value.code == "INVALID_PREPROCESS_ARTIFACT"

    def test_truncated_output_wav_raises_invalid_artifact(self, tmp_path: Path) -> None:
        in_wav = tmp_path / "in.wav"
        out_wav = tmp_path / "out.wav"
        _make_mono_wav(in_wav, sample_rate=16000, n_samples=16000)

        def fake_run(cmd: list[str], **kwargs: Any) -> MagicMock:
            if "--help-format" in cmd:
                return MagicMock(returncode=0, stdout="GSM 06.10", stderr="")
            for i, arg in enumerate(cmd):
                if arg.endswith(".wav") and i > 2:
                    with wave.open(arg, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes(b"\0" * (16000 * 2))
                    output = Path(arg)
                    output.write_bytes(output.read_bytes()[:-2])
                    break
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("shutil.which", return_value="/usr/bin/sox"), patch(
            "subprocess.run", side_effect=fake_run
        ):
            with pytest.raises(PreprocessError) as exc_info:
                run_sox_gsm_stage(in_wav, out_wav)
        assert exc_info.value.code == "INVALID_PREPROCESS_ARTIFACT"

    def test_non_wav_input_raises_invalid_audio(self, tmp_path: Path) -> None:
        bogus = tmp_path / "in.wav"
        bogus.write_bytes(b"not a wav file")

        with pytest.raises(PreprocessError) as exc_info:
            run_sox_gsm_stage(bogus, tmp_path / "out.wav")
        assert exc_info.value.code == "INVALID_AUDIO"


class TestSoxGsmStageArtifactInvariants:
    """SPEC-2026-09-21 line 51: deterministic, permission-locked output with
    no partial-write residue left behind on either success or failure."""

    def test_output_file_permissions_are_0600(self, tmp_path: Path) -> None:
        in_wav = tmp_path / "in.wav"
        out_wav = tmp_path / "out.wav"
        _make_mono_wav(in_wav, sample_rate=16000, n_samples=16000)

        runner = _build_fake_sox_runner(target_samples=16000)
        with patch("shutil.which", return_value="/usr/bin/sox"), \
             patch("subprocess.run", side_effect=runner):
            run_sox_gsm_stage(in_wav, out_wav)

        assert (out_wav.stat().st_mode & 0o777) == 0o600

    def test_no_tmp_residue_after_success(self, tmp_path: Path) -> None:
        in_wav = tmp_path / "in.wav"
        out_wav = tmp_path / "out.wav"
        _make_mono_wav(in_wav, sample_rate=16000, n_samples=16000)

        runner = _build_fake_sox_runner(target_samples=16000)
        with patch("shutil.which", return_value="/usr/bin/sox"), \
             patch("subprocess.run", side_effect=runner):
            run_sox_gsm_stage(in_wav, out_wav)

        assert list(tmp_path.glob("*.tmp.*")) == []

    def test_no_tmp_residue_after_alignment_failure(self, tmp_path: Path) -> None:
        in_wav = tmp_path / "in.wav"
        out_wav = tmp_path / "out.wav"
        _make_mono_wav(in_wav, sample_rate=16000, n_samples=16000)

        runner = _build_fake_sox_runner(target_samples=16400)  # excessive drift
        with patch("shutil.which", return_value="/usr/bin/sox"), \
             patch("subprocess.run", side_effect=runner):
            with pytest.raises(PreprocessError):
                run_sox_gsm_stage(in_wav, out_wav)

        assert list(tmp_path.glob("*.tmp.*")) == []
        assert not out_wav.exists()


class TestClippingThreshold:
    def test_below_threshold_does_not_raise(self) -> None:
        samples = np.zeros(100, dtype=np.int16)
        samples[:4] = 32767
        from say_transcribe.preprocess import _check_clipping

        # 4 clipped samples: below the default threshold of 5
        assert _check_clipping(samples) is None

    def test_at_threshold_raises(self) -> None:
        samples = np.zeros(100, dtype=np.int16)
        samples[:5] = 32767
        from say_transcribe.preprocess import _check_clipping

        with pytest.raises(PreprocessError) as exc_info:
            _check_clipping(samples)
        assert exc_info.value.code == "CLIPPING_ERROR"


class TestNonStandardInputRate:
    def test_8khz_input_rescales_expected_samples(self, tmp_path: Path) -> None:
        in_wav = tmp_path / "in.wav"
        out_wav = tmp_path / "out.wav"
        # 1 second at 8 kHz in, resampled to 16 kHz target -> 16000 expected out samples.
        _make_mono_wav(in_wav, sample_rate=8000, n_samples=8000)

        runner = _build_fake_sox_runner(target_samples=16000)
        with patch("shutil.which", return_value="/usr/bin/sox"), \
             patch("subprocess.run", side_effect=runner):
            run_sox_gsm_stage(in_wav, out_wav)
            assert out_wav.is_file()
