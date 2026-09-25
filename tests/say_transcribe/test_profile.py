import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from say_transcribe.profile import (
    ProfileError,
    apply_p0_profile,
    bandpass_narrowband,
    gsm_roundtrip,
    loudnorm_two_pass,
)

NEEDS_FFMPEG = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)
_SR = 16000


def _tone(seconds: float, freq: float = 1000.0, amplitude: float = 0.5) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * freq * np.arange(int(_SR * seconds)) / _SR)


def _to_i16(samples: np.ndarray) -> np.ndarray:
    return (np.clip(samples, -1.0, 1.0) * 32767.0).round().astype(np.int16)


def test_bandpass_attenuates_out_of_band_tone():
    in_band = bandpass_narrowband(_tone(1.0, freq=1000.0), _SR)
    out_of_band = bandpass_narrowband(_tone(1.0, freq=5000.0), _SR)

    assert np.sqrt(np.mean(out_of_band**2)) < 0.1 * np.sqrt(np.mean(in_band**2))


def test_bandpass_rejects_unusable_inputs():
    with pytest.raises(ProfileError) as rate_error:
        bandpass_narrowband(_tone(1.0), 4000)
    assert rate_error.value.code == "INVALID_AUDIO"

    with pytest.raises(ProfileError) as short_error:
        bandpass_narrowband(np.zeros(8), _SR)
    assert short_error.value.code == "INVALID_AUDIO"


@NEEDS_FFMPEG
def test_gsm_roundtrip_trims_frame_padding_to_input_count():
    samples = 0.5 * np.sin(2 * np.pi * 1000 * np.arange(4037) / 8000)

    round_tripped = gsm_roundtrip(samples)

    assert len(round_tripped) == 4037
    assert np.isfinite(round_tripped).all()
    assert np.corrcoef(samples, round_tripped)[0, 1] > 0.9


@NEEDS_FFMPEG
def test_apply_p0_profile_preserves_sample_count_at_16k():
    result = apply_p0_profile(_to_i16(_tone(2.0)), _SR, 2)

    assert result.sample_rate == 16000
    assert result.samples.dtype == np.float32
    assert len(result.samples) == 2 * _SR
    assert np.isfinite(result.samples).all()
    assert result.loudness.normalization_type in ("linear", "dynamic")
    assert np.isfinite(result.loudness.input_i)
    assert np.isfinite(result.loudness.output_i)
    assert -24.0 <= result.loudness.output_i <= -22.0
    assert result.loudness.output_tp <= -1.0


@NEEDS_FFMPEG
@pytest.mark.parametrize(
    "sample_rate, seconds, expected",
    [
        (16000, 1.0000625, 16001),  # odd 16 kHz count stays exact
        (48000, 1.00014583, 16002),  # mixed native rate: duration-ideal count
    ],
)
def test_apply_p0_profile_output_length_is_duration_ideal(sample_rate, seconds, expected):
    raw = 0.5 * np.sin(2 * np.pi * 1000 * np.arange(int(sample_rate * seconds)) / sample_rate)

    result = apply_p0_profile(_to_i16(raw), sample_rate, 2)

    assert len(result.samples) == expected


@NEEDS_FFMPEG
def test_loudnorm_flags_dynamic_fallback():
    # Quiet sustained tone + isolated near-full-scale click: integrated loudness
    # needs a gain that would push the true peak past TP=-1, so loudnorm falls
    # back to dynamic mode despite linear=true and must be flagged.
    # (SPEC's "source LRA above 50" trigger is unconstructible: EBU relative
    # gating caps measured LRA far below 50, so the TP-overflow path is the
    # reachable fallback.)
    samples = 0.02 * np.sin(2 * np.pi * 440 * np.arange(_SR * 8) / _SR)
    samples[_SR * 4 : _SR * 4 + 40] = 0.95 * np.hanning(40)

    result = apply_p0_profile(_to_i16(samples), _SR, 2)

    assert result.loudness.normalization_type == "dynamic"
    assert np.isfinite(result.loudness.input_lra)


@NEEDS_FFMPEG
def test_loudnorm_handles_clipping_and_short_inputs():
    clipped = np.sign(_tone(1.0))
    short = _tone(0.5)

    for raw in (clipped, short):
        result = apply_p0_profile(_to_i16(raw), _SR, 2)
        assert len(result.samples) == len(raw)
        assert np.isfinite(result.samples).all()
        assert np.abs(result.samples).max() <= 1.0


@NEEDS_FFMPEG
def test_silent_input_reports_unusable_stats():
    with pytest.raises(ProfileError) as excinfo:
        loudnorm_two_pass(np.zeros(2 * _SR))

    assert excinfo.value.code == "LOUDNORM_FAILED"
    assert "unusable" in excinfo.value.message


def test_missing_ffmpeg_raises_codec_unavailable(monkeypatch):
    monkeypatch.setattr("say_transcribe.profile.shutil.which", lambda name: None)

    with pytest.raises(ProfileError) as excinfo:
        apply_p0_profile(_to_i16(_tone(0.5)), _SR, 2)

    assert excinfo.value.code == "CODEC_UNAVAILABLE"


def test_ffmpeg_stage_failure_raises_codec_unavailable(monkeypatch):
    monkeypatch.setattr("say_transcribe.profile.shutil.which", lambda name: "/fake/ffmpeg")

    def fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=1, stdout=b"", stderr=b"Unknown encoder")

    monkeypatch.setattr("say_transcribe.profile.subprocess.run", fake_run)

    with pytest.raises(ProfileError) as excinfo:
        gsm_roundtrip(np.zeros(800))

    assert excinfo.value.code == "CODEC_UNAVAILABLE"
    assert "/" not in excinfo.value.message


def test_ffmpeg_stage_timeout_raises(monkeypatch):
    monkeypatch.setattr("say_transcribe.profile.shutil.which", lambda name: "/fake/ffmpeg")

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=300)

    monkeypatch.setattr("say_transcribe.profile.subprocess.run", fake_run)

    with pytest.raises(ProfileError) as excinfo:
        gsm_roundtrip(np.zeros(800))

    assert excinfo.value.code == "CODEC_UNAVAILABLE"
    assert "timed out" in excinfo.value.message


def test_gsm_roundtrip_short_decode_raises_codec_unavailable(monkeypatch):
    monkeypatch.setattr("say_transcribe.profile.shutil.which", lambda name: "/fake/ffmpeg")
    # 800 samples in, but the decode stage only returns 500: under the input
    # count, so the guard must reject it rather than silently truncate.
    responses = iter(
        [
            SimpleNamespace(returncode=0, stdout=b"encoded", stderr=b""),
            SimpleNamespace(returncode=0, stdout=b"\x00\x00" * 500, stderr=b""),
        ]
    )
    monkeypatch.setattr(
        "say_transcribe.profile.subprocess.run", lambda cmd, **kwargs: next(responses)
    )

    with pytest.raises(ProfileError) as excinfo:
        gsm_roundtrip(np.zeros(800))

    assert excinfo.value.code == "CODEC_UNAVAILABLE"


_PASS1_JSON = """
{
\t"input_i" : "-23.00",
\t"input_tp" : "-1.00",
\t"input_lra" : "5.00",
\t"input_thresh" : "-33.00",
\t"target_offset" : "0.00"
}
"""


def test_loudnorm_unparseable_stats_raise_loudnorm_failed(monkeypatch, tmp_path):
    monkeypatch.setattr("say_transcribe.profile.shutil.which", lambda name: "/fake/ffmpeg")
    responses = iter(
        [
            SimpleNamespace(returncode=0, stdout=b"", stderr=_PASS1_JSON.encode()),
            SimpleNamespace(returncode=0, stdout=b"\x00\x00" * 800, stderr=b"not json"),
        ]
    )
    monkeypatch.setattr(
        "say_transcribe.profile.subprocess.run", lambda cmd, **kwargs: next(responses)
    )

    with pytest.raises(ProfileError) as excinfo:
        loudnorm_two_pass(np.zeros(800))

    assert excinfo.value.code == "LOUDNORM_FAILED"
    assert "pass-2" in excinfo.value.message
    assert str(tmp_path) not in excinfo.value.message


def test_loudnorm_pass1_unparseable_stats_raise(monkeypatch):
    monkeypatch.setattr("say_transcribe.profile.shutil.which", lambda name: "/fake/ffmpeg")
    monkeypatch.setattr(
        "say_transcribe.profile.subprocess.run",
        lambda cmd, **kwargs: SimpleNamespace(returncode=0, stdout=b"", stderr=b"garbage"),
    )

    with pytest.raises(ProfileError) as excinfo:
        loudnorm_two_pass(np.zeros(800))

    assert excinfo.value.code == "LOUDNORM_FAILED"
    assert "pass-1" in excinfo.value.message


def test_profile_rejects_unsupported_sample_width():
    with pytest.raises(ProfileError) as excinfo:
        apply_p0_profile(np.zeros(16, dtype=np.int16), _SR, 3)

    assert excinfo.value.code == "INVALID_AUDIO"


def test_profile_rejects_non_finite_samples():
    samples = _tone(0.5)
    samples[0] = np.nan

    with pytest.raises(ProfileError) as excinfo:
        apply_p0_profile(samples, _SR, 2)

    assert excinfo.value.code == "INVALID_AUDIO"


def test_input_validation_precedes_ffmpeg_check(monkeypatch):
    # Caller-data errors must stay reachable on machines without ffmpeg.
    monkeypatch.setattr("say_transcribe.profile.shutil.which", lambda name: None)

    with pytest.raises(ProfileError) as excinfo:
        apply_p0_profile(np.zeros(16, dtype=np.int16), _SR, 3)

    assert excinfo.value.code == "INVALID_AUDIO"
