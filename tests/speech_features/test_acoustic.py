"""Behavioral tests for math-first acoustic feature extraction (Task 2).

Accepts mono PCM float arrays only; no WAV I/O, no labels, no ASR. Covers
sample-rate/config checks, frame/hop analysis, energy VAD (dynamic range +
absolute silence floor), VAD-derived silence/pause summaries, normalized
autocorrelation F0 with explicit unvoiced handling and ceil/floor lag bounds,
voiced F0 median/IQR/5-95% span/delta + HNR median/IQR, frame-energy
mean/SD/IQR/span, bounded spectral flatness, and quality flags.

Regression tests in this module encode the Agy Task-2 review fixes so they
fail ("RED") before the corrected maths and pass ("GREEN") after.
"""

import math
import os
import struct
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

import speech_features.acoustic as acoustic
from speech_features import list_features
from speech_features.document import (
    DocumentSpeaker,
    DocumentToken,
    DocumentUtterance,
    SpeechDocument,
)
from speech_features.features import acoustic as acoustic_pack
from speech_features.pipeline import InvalidAudioError, UnsupportedAudioError
from speech_features.result import (
    FeatureBundle,
    InvalidConfigError,
    MissingAnnotationError,
    TargetSpeakerRequiredError,
)
from speech_features.schema import ExtractionConfig


def _tone(freq, duration_s, sample_rate=16000, amplitude=0.5):
    t = np.arange(int(round(sample_rate * duration_s))) / sample_rate
    return amplitude * np.sin(2 * math.pi * freq * t)


class TestExtractAcoustic:
    def test_returns_plain_numeric_features_and_flags(self):
        result = acoustic.extract_acoustic(_tone(145, 1.0), 16000)
        assert result.features
        assert all(isinstance(v, float) for v in result.features.values())
        assert isinstance(result.flags, tuple)
        assert result.flags == ()

    def test_tone_pitch_bounds_and_voicing(self):
        result = acoustic.extract_acoustic(_tone(145, 1.0), 16000)
        assert 130 < result.features["ac_pitch_voiced_mean"] < 160
        assert result.features["ac_voice_ratio"] > 0.5
        assert math.isfinite(result.features["ac_pitch_voiced_sd"])

    def test_silence_yields_no_voice_flag_and_nan_pitch(self):
        result = acoustic.extract_acoustic(np.zeros(16000, dtype=float), 16000)
        assert "no_voice" in result.flags
        assert math.isnan(result.features["ac_pitch_voiced_mean"])
        assert math.isnan(result.features["ac_pitch_voiced_cv"])

    def test_voiced_silence_voiced_counts_one_pause(self):
        signal = np.concatenate([_tone(145, 0.3), np.zeros(int(16000 * 0.5)), _tone(145, 0.3)])
        result = acoustic.extract_acoustic(signal, 16000)
        assert "no_voice" not in result.flags
        assert result.features["ac_pause_count"] == 1
        assert result.features["ac_pause_max_s"] >= 0.3

    def test_short_signal_yields_too_short_flag(self):
        result = acoustic.extract_acoustic(np.full(400, 0.5), 16000)
        assert "too_short" in result.flags

    def test_invalid_sample_rate_yields_invalid_input_flag(self):
        result = acoustic.extract_acoustic(_tone(145, 1.0), 0)
        assert "invalid_input" in result.flags
        assert math.isnan(result.features["ac_pitch_voiced_mean"])

    def test_energy_and_spectral_features_finite_for_tone(self):
        result = acoustic.extract_acoustic(_tone(145, 1.0), 16000)
        assert result.features["ac_frame_energy_mean"] > 0
        assert result.features["ac_frame_energy_sd"] >= 0
        assert 0 < result.features["ac_spectral_centroid_mean"] < 8000
        assert math.isfinite(result.features["ac_spectral_spread_mean"])
        assert math.isfinite(result.features["ac_spectral_flatness_mean"])


class TestConfigRespected:
    def test_pitch_bounds_restrict_f0(self):
        config = ExtractionConfig(pitch_min_hz=200.0, pitch_max_hz=250.0)
        low = 145  # below the configured lower bound
        result = acoustic.extract_acoustic(_tone(low, 1.0), 16000, config=config)
        # Explicit unvoiced handling: a tone outside configured F0 bounds is
        # not a valid candidate, so no voiced pitch is produced.
        assert "no_voice" in result.flags
        assert math.isnan(result.features["ac_pitch_voiced_mean"])

    def test_pause_threshold_removes_short_gaps(self):
        gap = 0.1  # below the default 0.20 s pause threshold
        signal = np.concatenate([_tone(145, 0.5), np.zeros(int(16000 * gap)), _tone(145, 0.5)])
        result = acoustic.extract_acoustic(signal, 16000)
        assert result.features["ac_pause_count"] == 0


class TestSpectralFlatness:
    def test_tone_flatness_near_zero_and_bounded(self):
        result = acoustic.extract_acoustic(_tone(145, 1.0), 16000)
        flatness = result.features["ac_spectral_flatness_mean"]
        # A pure tone has a spiky spectrum: geometric mean << arithmetic mean.
        assert math.isfinite(flatness)
        assert 0.0 <= flatness <= 1.0
        assert flatness < 0.5

    def test_white_noise_flatness_bounded_near_one(self):
        rng = np.random.default_rng(0)
        noise = rng.uniform(-1.0, 1.0, 16000)
        result = acoustic.extract_acoustic(noise, 16000)
        flatness = result.features["ac_spectral_flatness_mean"]
        assert 0.0 <= flatness <= 1.0
        assert flatness > 0.5


class TestEnergyVADSilenceFloor:
    def test_quiet_noise_yields_no_voice(self):
        # Amplitude far below the absolute silence floor: even with dynamic
        # range from the mean, this must read as silence.
        quiet = np.full(16000, 1e-12, dtype=float)
        result = acoustic.extract_acoustic(quiet, 16000)
        assert "no_voice" in result.flags
        assert math.isnan(result.features["ac_pitch_voiced_mean"])
        # No voice → pause statistics are unavailable (NaN), never zero.
        assert math.isnan(result.features["ac_pause_count"])

    def test_very_low_noise_yields_no_voice(self):
        rng = np.random.default_rng(1)
        quiet_noise = rng.normal(0.0, 1e-11, 16000)
        result = acoustic.extract_acoustic(quiet_noise, 16000)
        assert "no_voice" in result.flags


class TestPauseFrameCounting:
    def test_exact_threshold_pause_is_counted(self):
        # A silence run of exactly the 0.20 s threshold must register as a
        # pause even if per-frame hop accumulation drifts in float.
        gap = 0.20
        signal = np.concatenate(
            [_tone(145, 0.4), np.zeros(int(round(16000 * gap))), _tone(145, 0.4)]
        )
        result = acoustic.extract_acoustic(signal, 16000)
        assert result.features["ac_pause_count"] == 1


class TestPitchLagBounds:
    def test_configured_upper_bound_tone_voiced(self):
        config = ExtractionConfig(pitch_min_hz=70.0, pitch_max_hz=400.0)
        result = acoustic.extract_acoustic(_tone(400, 1.0), 16000, config=config)
        assert "no_voice" not in result.flags
        assert math.isfinite(result.features["ac_pitch_voiced_mean"])

    def test_pitch_lag_min_is_ceil_of_sr_over_max(self):
        # sr / pitch_max = 16000 / 350 = 45.71: ceil(45.71) = 46. Integer
        # truncation (int() -> 45) would admit lag 45 whose period (~355.6 Hz)
        # exceeds the configured 350 Hz ceiling. ceil must be used.
        config = ExtractionConfig(pitch_min_hz=70.0, pitch_max_hz=350.0)
        min_lag = math.ceil(16000 / config.pitch_max_hz)
        max_lag = math.floor(16000 / config.pitch_min_hz)
        win = np.zeros((1, config.frame_size))
        f0, nccf = acoustic._f0_per_frame(win, config.frame_size, 16000, config)
        assert min_lag == 46
        assert max_lag == math.floor(16000 / 70.0)
        assert f0.shape == (1,)


class TestPlanRequiredSummaries:
    @pytest.fixture(autouse=True)
    def tone_result(self):
        self.result = acoustic.extract_acoustic(_tone(145, 1.0), 16000)

    def test_pitch_median_iqr_span_delta_present(self):
        for key in (
            "ac_pitch_voiced_median",
            "ac_pitch_voiced_iqr",
            "ac_pitch_voiced_span",
            "ac_pitch_voiced_delta",
        ):
            assert key in self.result.features
            assert math.isfinite(self.result.features[key])

    def test_pitch_median_within_tone_bounds(self):
        assert 130 < self.result.features["ac_pitch_voiced_median"] < 160
        assert self.result.features["ac_pitch_voiced_span"] >= 0

    def test_energy_iqr_and_span_present(self):
        assert math.isfinite(self.result.features["ac_frame_energy_iqr"])
        assert math.isfinite(self.result.features["ac_frame_energy_span"])
        assert self.result.features["ac_frame_energy_span"] >= 0

    def test_hnr_median_and_iqr_present_and_positive_for_tone(self):
        assert math.isfinite(self.result.features["ac_hnr_median"])
        assert math.isfinite(self.result.features["ac_hnr_iqr"])
        assert self.result.features["ac_hnr_median"] >= 0
        assert self.result.features["ac_hnr_iqr"] >= 0

    def test_cv_uses_finite_mean(self):
        # A NaN or zero mean must never be used as a division base blindly; the
        # CV must come out finite (or NaN-flagged) without crashing.
        result = acoustic.extract_acoustic(_tone(145, 1.0), 16000)
        assert result.features["ac_pitch_voiced_cv"] is not None


class TestNanTruthinessAvoided:
    def test_cv_not_evaluated_by_boolean_truth_of_nan(self):
        # Regression: the previous guard ``if features["ac_pitch_voiced_mean"]``
        # treats NaN (truthy) as usable. A finite-check guard must be used, and
        # an all-silence signal must not produce a garbage CV.
        sil = acoustic.extract_acoustic(np.zeros(16000, dtype=float), 16000)
        assert math.isnan(sil.features["ac_pitch_voiced_mean"])
        assert math.isnan(sil.features["ac_pitch_voiced_cv"])


# ---------------------------------------------------------------------------
# Task 5: acoustic core, audio quality, and timing (brief keys and behaviors)
# ---------------------------------------------------------------------------
EXPECTED_ACOUSTIC_KEYS = (
    "audio_clipping_ratio",
    "audio_dc_offset",
    "audio_duration_s",
    "audio_rms_dbfs",
    "time_articulation_rate_syllables_per_s",
    "time_long_pause_count",
    "time_overlap_s",
    "time_pause_count",
    "time_pause_max_s",
    "time_pause_mean_s",
    "time_pause_rate_per_min",
    "time_pause_sd_s",
    "time_response_latency_s",
    "time_speech_ratio",
    "time_speech_s",
    "time_syllables_per_min",
    "time_voiced_segment_mean_s",
    "time_voiced_segment_sd_s",
    "time_words_per_min",
)


def _to_pcm(mono, width):
    mono = np.clip(np.asarray(mono, dtype=float), -1.0, 1.0)
    if width == 1:  # unsigned 8-bit
        return np.round(mono * 128.0 + 128.0).astype(np.uint8).tobytes()
    if width == 2:
        return np.round(mono * 32767.0).astype(np.int16).astype("<i2").tobytes()
    if width == 3:  # signed 24-bit, three little-endian bytes
        pcm = np.round(mono * 8388607.0).astype(np.int32)
        lo = (pcm & 0xFF).astype(np.uint8)
        mid = ((pcm >> 8) & 0xFF).astype(np.uint8)
        hi = ((pcm >> 16) & 0xFF).astype(np.uint8)
        return np.stack([lo, mid, hi], axis=1).reshape(-1).tobytes()
    return np.round(mono * 2147483647.0).astype(np.int32).astype("<i4").tobytes()


def _write_pcm_wav(path, mono, sample_rate=16000, n_channels=1, width=2):
    mono = np.asarray(mono, dtype=float)
    if n_channels == 2:
        data = np.stack([mono, mono], axis=1).reshape(-1)
    else:
        data = mono.copy()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(width)
        wf.setframerate(sample_rate)
        wf.writeframes(_to_pcm(data, width))
    return str(path)


def _write_pcm_codes(path, codes, width):
    """Write a mono WAV whose payload is the exact integer PCM codes given."""
    codes = np.asarray(codes)
    if width == 1:
        payload = codes.astype(np.uint8).tobytes()
    elif width == 2:
        payload = codes.astype("<i2").tobytes()
    elif width == 3:
        lo = (codes & 0xFF).astype(np.uint8)
        mid = ((codes >> 8) & 0xFF).astype(np.uint8)
        hi = ((codes >> 16) & 0xFF).astype(np.uint8)
        payload = np.stack([lo, mid, hi], axis=1).reshape(-1).tobytes()
    else:
        payload = codes.astype("<i4").tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(width)
        wf.setframerate(16000)
        wf.writeframes(payload)
    return str(path)


def _write_raw_wav(path, *, format_tag=1, n_channels=1, sample_rate=16000, bits=8, payload=None):
    """Write a WAV with a hand-built header so unsupported formats are reachable."""
    payload = b"\x00" * 160 if payload is None else payload
    block_align = n_channels * bits // 8
    header = b"RIFF" + struct.pack("<I", 36 + len(payload)) + b"WAVE"
    header += b"fmt " + struct.pack(
        "<IHHIIHH",
        16,
        format_tag,
        n_channels,
        sample_rate,
        sample_rate * block_align,
        block_align,
        bits,
    )
    header += b"data" + struct.pack("<I", len(payload))
    path.write_bytes(header + payload)
    return str(path)


def _doc(utterances, speakers):
    return SpeechDocument(document_id="d1", speakers=speakers, utterances=utterances)


def _utt(uid, speaker, start, end, tokens=()):
    return DocumentUtterance(id=uid, speaker_id=speaker, start_s=start, end_s=end, tokens=tokens)


def _tok(tid, text, kind="word", word_id=None):
    return DocumentToken(id=tid, text=text, kind=kind, word_id=word_id)


class TestAcousticPackCatalog:
    def test_exact_nineteen_quality_and_timing_keys_registered(self):
        features = list_features(pack="acoustic")
        assert [f.key for f in features] == list(EXPECTED_ACOUSTIC_KEYS)

    def test_definitions_carry_full_metadata(self):
        for definition in list_features(pack="acoustic"):
            assert definition.pack == "acoustic"
            assert definition.level == "recording"
            assert definition.unit
            assert definition.population
            assert definition.reference
            assert definition.formula_version >= 1
            assert isinstance(definition.prerequisites, tuple)


class TestRecordingQuality:
    def test_duration_dc_offset_and_rms_dbfs_on_tone(self):
        # Quality is whole-recording by design; without a document the explicit
        # fallback flag is required and emits the UNALIGNED_SPEAKER warning.
        features, issues = acoustic_pack.extract_acoustic_features(
            _tone(145, 1.0), 16000, allow_unaligned=True
        )
        assert features["audio_duration_s"] == pytest.approx(1.0, abs=1e-6)
        assert features["audio_dc_offset"] == pytest.approx(0.0, abs=1e-6)
        assert features["audio_clipping_ratio"] == 0.0
        # A 0.5-amplitude sine has RMS 0.5/sqrt(2): 20*log10(rms) = -9.03 dBFS.
        assert features["audio_rms_dbfs"] == pytest.approx(-9.03, abs=0.1)
        assert any(issue.code == "UNALIGNED_SPEAKER" for issue in issues)

    def test_clipping_ratio_counts_full_scale_samples(self):
        signal = _tone(145, 1.0)
        signal[: int(0.1 * 16000)] = 1.0
        features, _ = acoustic_pack.extract_acoustic_features(signal, 16000, allow_unaligned=True)
        assert features["audio_clipping_ratio"] == pytest.approx(0.1, abs=0.005)

    def test_digital_silence_rms_is_nan_with_issue(self):
        features, issues = acoustic_pack.extract_acoustic_features(
            np.zeros(16000), 16000, allow_unaligned=True
        )
        assert math.isnan(features["audio_rms_dbfs"])
        assert any(issue.code == "NO_AUDIO" for issue in issues)

    def test_invalid_array_raises_invalid_audio(self):
        with pytest.raises(InvalidAudioError) as exc:
            acoustic_pack.extract_acoustic_features(np.zeros(16000), 0)
        assert exc.value.code == "INVALID_AUDIO"


class TestPcmWavSupport:
    @pytest.mark.parametrize("width", [1, 2, 3, 4])
    def test_standard_pcm_widths_decode(self, tmp_path, width):
        path = _write_pcm_wav(tmp_path / f"w{width}.wav", _tone(145, 1.0), width=width)
        bundle = acoustic_pack.extract_acoustic_bundle(path, allow_unaligned=True)
        assert bundle.recordings.loc[0, "audio_duration_s"] == pytest.approx(1.0, abs=1e-6)

    def test_stereo_downmix_and_resample(self, tmp_path):
        path = _write_pcm_wav(tmp_path / "s.wav", _tone(145, 1.0, 22050), 22050, n_channels=2)
        bundle = acoustic_pack.extract_acoustic_bundle(path, allow_unaligned=True)
        assert bundle.recordings.loc[0, "audio_duration_s"] == pytest.approx(1.0, abs=0.02)
        assert math.isfinite(bundle.recordings.loc[0, "audio_rms_dbfs"])

    def test_non_pcm_wav_raises_unsupported_audio(self, tmp_path):
        path = _write_raw_wav(tmp_path / "ulaw.wav", format_tag=7, bits=8)
        with pytest.raises(UnsupportedAudioError) as exc:
            acoustic_pack.extract_acoustic_bundle(path)
        assert exc.value.code == "UNSUPPORTED_AUDIO"

    def test_unsupported_sample_width_raises_unsupported_audio(self, tmp_path):
        path = _write_raw_wav(tmp_path / "f48.wav", format_tag=1, bits=48)
        with pytest.raises(UnsupportedAudioError) as exc:
            acoustic_pack.extract_acoustic_bundle(path)
        assert exc.value.code == "UNSUPPORTED_AUDIO"

    def test_unsupported_channel_count_raises_unsupported_audio(self, tmp_path):
        path = _write_raw_wav(tmp_path / "5ch.wav", format_tag=1, n_channels=3, bits=16)
        with pytest.raises(UnsupportedAudioError) as exc:
            acoustic_pack.extract_acoustic_bundle(path)
        assert exc.value.code == "UNSUPPORTED_AUDIO"

    def test_malformed_wav_raises_invalid_audio(self, tmp_path):
        path = tmp_path / "bad.wav"
        path.write_bytes(b"NOTARIFFFILE")
        with pytest.raises(InvalidAudioError) as exc:
            acoustic_pack.extract_acoustic_bundle(path)
        assert exc.value.code == "INVALID_AUDIO"


class TestTargetSpeakerIsolation:
    def test_multiple_speakers_without_target_raises(self):
        doc = _doc(
            (_utt("u1", "p1", 0.0, 1.0), _utt("u2", "e1", 1.5, 2.5)),
            (
                DocumentSpeaker(id="p1", role="participant"),
                DocumentSpeaker(id="e1", role="examiner"),
            ),
        )
        with pytest.raises(TargetSpeakerRequiredError) as exc:
            acoustic_pack.extract_acoustic_features(_tone(145, 3.0), 16000, document=doc)
        assert exc.value.code == "TARGET_SPEAKER_REQUIRED"

    def test_unknown_target_speaker_is_invalid(self):
        doc = _doc((_utt("u1", "p1", 0.0, 1.0),), (DocumentSpeaker(id="p1"),))
        with pytest.raises(InvalidConfigError) as exc:
            acoustic_pack.extract_acoustic_features(
                _tone(145, 2.0), 16000, document=doc, target_speaker="ghost"
            )
        assert exc.value.code == "INVALID_CONFIG"

    def test_single_speaker_is_implicit_target(self):
        doc = _doc((_utt("u1", "p1", 0.5, 1.5),), (DocumentSpeaker(id="p1"),))
        features, issues = acoustic_pack.extract_acoustic_features(
            _tone(145, 2.0), 16000, document=doc
        )
        assert features["time_speech_s"] == pytest.approx(1.0, abs=1e-9)
        # Only the (expected) missing word-token annotation is raised: no
        # unaligned, no-audio, or no-speech warnings.
        assert not any(
            issue.code in {"UNALIGNED_SPEAKER", "NO_AUDIO", "NO_SPEECH"} for issue in issues
        )

    def test_examiner_audio_does_not_contaminate_participant_measures(self):
        # 160 Hz = exactly 4 periods per 25 ms frame, so frame energies are
        # uniform and the shared VAD marks every frame voiced (a 145 Hz tone
        # ripples at the 4th decimal and the mean-threshold VAD splits it).
        examiner = _tone(1000, 1.0, amplitude=0.9)
        participant = _tone(160, 1.0, amplitude=0.2)
        audio = np.concatenate([examiner, np.zeros(int(16000 * 0.5)), participant])
        doc = _doc(
            (_utt("u1", "e1", 0.0, 1.0), _utt("u2", "p1", 1.5, 2.5)),
            (
                DocumentSpeaker(id="p1", role="participant"),
                DocumentSpeaker(id="e1", role="examiner"),
            ),
        )
        features, _ = acoustic_pack.extract_acoustic_features(
            audio, 16000, document=doc, target_speaker="p1"
        )
        assert features["time_speech_s"] == pytest.approx(1.0, abs=1e-9)
        assert features["time_voiced_segment_mean_s"] == pytest.approx(0.98, abs=0.05)
        assert features["time_pause_count"] == 0.0

    def test_no_document_requires_explicit_fallback(self):
        with pytest.raises(MissingAnnotationError) as exc:
            acoustic_pack.extract_acoustic_features(_tone(145, 2.0), 16000)
        assert exc.value.code == "MISSING_ANNOTATION"

    def test_unaligned_fallback_emits_warning_issue(self):
        features, issues = acoustic_pack.extract_acoustic_features(
            _tone(145, 2.0), 16000, allow_unaligned=True
        )
        assert any(issue.code == "UNALIGNED_SPEAKER" for issue in issues)
        assert math.isnan(features["time_speech_s"])

    def test_target_without_utterances_needs_fallback(self):
        doc = _doc((), (DocumentSpeaker(id="p1"), DocumentSpeaker(id="e1")))
        with pytest.raises(MissingAnnotationError):
            acoustic_pack.extract_acoustic_features(
                _tone(145, 2.0), 16000, document=doc, target_speaker="p1"
            )
        features, issues = acoustic_pack.extract_acoustic_features(
            _tone(145, 2.0), 16000, document=doc, target_speaker="p1", allow_unaligned=True
        )
        assert any(issue.code == "UNALIGNED_SPEAKER" for issue in issues)


class TestTimingMeasures:
    def test_pause_and_voiced_segment_summaries(self):
        # Recording [0.3, 2.3); aligned interval [0.3, 1.3) holds tone 0.3 s,
        # silence 0.5 s, tone 0.2 s -> one 0.5 s pause, voiced segments 0.3/0.2.
        # 160 Hz tones keep frame energies uniform for the shared mean VAD.
        audio = np.concatenate(
            [
                np.zeros(int(16000 * 0.3)),
                _tone(160, 0.3),
                np.zeros(int(16000 * 0.5)),
                _tone(160, 0.2),
                np.zeros(int(16000 * 0.7)),
            ]
        )
        doc = _doc((_utt("u1", "p1", 0.3, 1.3),), (DocumentSpeaker(id="p1"),))
        features, _ = acoustic_pack.extract_acoustic_features(
            audio, 16000, document=doc, target_speaker="p1"
        )
        assert features["time_speech_s"] == pytest.approx(1.0, abs=1e-9)
        assert features["time_speech_ratio"] == pytest.approx(1.0 / 2.0, abs=1e-9)
        assert features["time_pause_count"] == 1.0
        assert features["time_pause_mean_s"] == pytest.approx(0.5, abs=0.03)
        assert features["time_pause_max_s"] == pytest.approx(0.5, abs=0.03)
        assert features["time_pause_sd_s"] == pytest.approx(0.0, abs=0.03)
        assert features["time_pause_rate_per_min"] == pytest.approx(30.0, abs=0.5)
        assert features["time_long_pause_count"] == 0.0
        assert features["time_voiced_segment_mean_s"] == pytest.approx(0.24, abs=0.04)
        assert features["time_voiced_segment_sd_s"] == pytest.approx(0.06, abs=0.02)

    def test_long_pause_count(self):
        audio = np.concatenate(
            [
                np.zeros(int(16000 * 0.2)),
                _tone(145, 0.3),
                np.zeros(int(16000 * 2.2)),
                _tone(145, 0.2),
                np.zeros(int(16000 * 0.1)),
            ]
        )
        doc = _doc((_utt("u1", "p1", 0.2, 2.9),), (DocumentSpeaker(id="p1"),))
        features, _ = acoustic_pack.extract_acoustic_features(
            audio, 16000, document=doc, target_speaker="p1"
        )
        assert features["time_pause_count"] == 1.0
        assert features["time_long_pause_count"] == 1.0

    def test_pause_threshold_config_respected(self):
        audio = np.concatenate(
            [
                np.zeros(int(16000 * 0.2)),
                _tone(145, 0.3),
                np.zeros(int(16000 * 0.3)),
                _tone(145, 0.2),
            ]
        )
        doc = _doc((_utt("u1", "p1", 0.2, 1.0),), (DocumentSpeaker(id="p1"),))
        features, _ = acoustic_pack.extract_acoustic_features(
            audio,
            16000,
            document=doc,
            target_speaker="p1",
            config=ExtractionConfig(pause_threshold_s=0.5),
        )
        assert features["time_pause_count"] == 0.0

    def test_response_latency_is_mean_of_examiner_to_participant_gaps(self):
        audio = _tone(145, 5.0)
        doc = _doc(
            (
                _utt("u1", "e1", 0.0, 1.0),
                _utt("u2", "p1", 1.5, 2.5),
                _utt("u3", "e1", 3.0, 4.0),
                _utt("u4", "p1", 4.2, 5.0),
            ),
            (
                DocumentSpeaker(id="p1", role="participant"),
                DocumentSpeaker(id="e1", role="examiner"),
            ),
        )
        features, _ = acoustic_pack.extract_acoustic_features(
            audio, 16000, document=doc, target_speaker="p1"
        )
        assert features["time_response_latency_s"] == pytest.approx(0.35, abs=1e-9)

    def test_overlap_seconds(self):
        audio = _tone(145, 2.0)
        doc = _doc(
            (_utt("u1", "e1", 0.0, 1.0), _utt("u2", "p1", 0.6, 1.6)),
            (
                DocumentSpeaker(id="p1", role="participant"),
                DocumentSpeaker(id="e1", role="examiner"),
            ),
        )
        features, _ = acoustic_pack.extract_acoustic_features(
            audio, 16000, document=doc, target_speaker="p1"
        )
        assert features["time_overlap_s"] == pytest.approx(0.4, abs=1e-9)

    def test_word_and_syllable_rates_with_word_id_grouping(self):
        doc = _doc(
            (
                _utt(
                    "u1",
                    "p1",
                    0.0,
                    1.0,
                    tokens=(
                        _tok("t1", "con", word_id="w1"),
                        _tok("t2", "m\u00e8o", word_id="w1"),
                        _tok("t3", "ch\u1ea1y", word_id="w2"),
                        _tok("t4", "nhanh"),
                        _tok("t5", "\u00e0", kind="filler"),
                    ),
                ),
            ),
            (DocumentSpeaker(id="p1"),),
        )
        features, _ = acoustic_pack.extract_acoustic_features(_tone(145, 1.0), 16000, document=doc)
        assert features["time_speech_s"] == pytest.approx(1.0, abs=1e-9)
        # 3 words (w1, w2, ungrouped token) in 1 s; 4 word-kind tokens (syllables).
        assert features["time_words_per_min"] == pytest.approx(180.0, abs=1e-6)
        assert features["time_syllables_per_min"] == pytest.approx(240.0, abs=1e-6)
        assert features["time_articulation_rate_syllables_per_s"] == pytest.approx(4.0, abs=1e-6)

    def test_no_word_tokens_yield_nan_with_issue(self):
        doc = _doc(
            (_utt("u1", "p1", 0.0, 1.0, tokens=(_tok("t1", "\u00e0", kind="filler"),)),),
            (DocumentSpeaker(id="p1"),),
        )
        features, issues = acoustic_pack.extract_acoustic_features(
            _tone(145, 1.0), 16000, document=doc
        )
        assert math.isnan(features["time_words_per_min"])
        assert math.isnan(features["time_syllables_per_min"])
        assert any(issue.code == "MISSING_ANNOTATION" for issue in issues)

    def test_latency_and_overlap_nan_without_examiner(self):
        doc = _doc((_utt("u1", "p1", 0.0, 1.0),), (DocumentSpeaker(id="p1"),))
        features, issues = acoustic_pack.extract_acoustic_features(
            _tone(145, 1.0), 16000, document=doc
        )
        assert math.isnan(features["time_response_latency_s"])
        assert math.isnan(features["time_overlap_s"])
        assert any(issue.code == "MISSING_ANNOTATION" for issue in issues)

    def test_no_voice_yields_nan_pause_stats_with_issue(self):
        doc = _doc((_utt("u1", "p1", 0.2, 1.2),), (DocumentSpeaker(id="p1"),))
        features, issues = acoustic_pack.extract_acoustic_features(
            np.zeros(16000), 16000, document=doc, target_speaker="p1"
        )
        assert math.isnan(features["time_voiced_segment_mean_s"])
        assert math.isnan(features["time_pause_count"])
        assert any(issue.code == "NO_SPEECH" for issue in issues)


class TestBundleContract:
    def test_recordings_columns_deterministic_and_values_float(self, tmp_path):
        path = _write_pcm_wav(tmp_path / "a.wav", _tone(145, 1.0))
        doc = _doc((_utt("u1", "p1", 0.5, 1.5),), (DocumentSpeaker(id="p1"),))
        bundle = acoustic_pack.extract_acoustic_bundle(
            path, doc, target_speaker="p1", recording_id="rec-1"
        )
        assert isinstance(bundle, FeatureBundle)
        assert list(bundle.recordings.columns) == [
            "recording_id",
            "speaker_id",
            *EXPECTED_ACOUSTIC_KEYS,
        ]
        assert bundle.recordings.loc[0, "recording_id"] == "rec-1"
        assert bundle.recordings.loc[0, "speaker_id"] == "p1"
        assert all(isinstance(value, float) for value in bundle.recordings.iloc[0, 2:])
        assert list(bundle.utterances.columns) == [
            "recording_id",
            "speaker_id",
            "utterance_id",
            "start_s",
            "end_s",
        ]
        assert bundle.utterances.empty

    def test_bundle_issues_carry_stable_codes_and_schema(self, tmp_path):
        path = _write_pcm_wav(tmp_path / "s.wav", np.zeros(16000))
        bundle = acoustic_pack.extract_acoustic_bundle(path, allow_unaligned=True, recording_id="r")
        assert list(bundle.issues.columns) == [
            "recording_id",
            "speaker_id",
            "utterance_id",
            "feature",
            "code",
            "severity",
            "message",
        ]
        assert set(bundle.issues["code"]) <= {
            "UNALIGNED_SPEAKER",
            "NO_AUDIO",
            "NO_SPEECH",
            "MISSING_ANNOTATION",
        }
        assert math.isnan(bundle.recordings.loc[0, "audio_rms_dbfs"])


class TestCatalogRegistrationIsolation:
    """Agy fix 1: plain package import must register the acoustic pack."""

    def test_fresh_process_sees_all_acoustic_keys(self):
        src = Path(__file__).resolve().parents[2] / "src"
        code = (
            "import speech_features as sf\n"
            f"expected = {EXPECTED_ACOUSTIC_KEYS!r}\n"
            "keys = [f.key for f in sf.list_features(pack='acoustic')]\n"
            "assert keys == list(expected), (keys, expected)\n"
            "print(len(keys))\n"
        )
        env = {**os.environ, "PYTHONPATH": str(src)}
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            cwd=src.parent,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == "19"


class TestClippingWidthBoundaries:
    """Agy fix 2: positive full-scale PCM must count as clipped per width."""

    @pytest.mark.parametrize(
        ("width", "codes", "expected_ratio"),
        [
            (1, [255, 0, 128], 2 / 3),
            (2, [32767, -32768, 0], 2 / 3),
            (3, [8388607, -8388608, 0], 2 / 3),
            (4, [2147483647, -2147483648, 0], 2 / 3),
        ],
    )
    def test_full_scale_positive_and_negative_count_as_clipped(
        self, tmp_path, width, codes, expected_ratio
    ):
        path = _write_pcm_codes(tmp_path / f"clip{width}.wav", codes, width)
        bundle = acoustic_pack.extract_acoustic_bundle(path, allow_unaligned=True)
        assert bundle.recordings.loc[0, "audio_clipping_ratio"] == pytest.approx(expected_ratio)

    def test_array_path_keeps_unity_boundary(self):
        signal = _tone(145, 1.0)
        signal[: int(0.1 * 16000)] = 0.99997
        features, _ = acoustic_pack.extract_acoustic_features(signal, 16000, allow_unaligned=True)
        assert features["audio_clipping_ratio"] == 0.0


class TestResponseLatencyContinuation:
    """Agy fix 3: target continuation after an examiner turn is not a latency."""

    def test_continuation_after_examiner_turn_is_not_counted(self):
        audio = _tone(145, 5.0)
        doc = _doc(
            (
                _utt("u1", "e1", 0.0, 1.0),
                _utt("u2", "p1", 1.5, 2.5),
                _utt("u3", "p1", 3.0, 4.0),
            ),
            (
                DocumentSpeaker(id="p1", role="participant"),
                DocumentSpeaker(id="e1", role="examiner"),
            ),
        )
        features, _ = acoustic_pack.extract_acoustic_features(
            audio, 16000, document=doc, target_speaker="p1"
        )
        # Only u2 (0.5 s after the examiner) is a response; u3 continues the
        # participant's own turn and must not add a 2.0 s pseudo-latency.
        assert features["time_response_latency_s"] == pytest.approx(0.5, abs=1e-9)


class TestNanIssuePairing:
    """Agy fix 4: every NaN from a missing prerequisite names its exact key."""

    _TRANSCRIPT_KEYS = (
        "time_speech_s",
        "time_speech_ratio",
        "time_response_latency_s",
        "time_overlap_s",
        "time_words_per_min",
        "time_syllables_per_min",
        "time_articulation_rate_syllables_per_s",
    )
    _VAD_KEYS = (
        "time_voiced_segment_mean_s",
        "time_voiced_segment_sd_s",
        "time_pause_count",
        "time_pause_rate_per_min",
        "time_pause_mean_s",
        "time_pause_sd_s",
        "time_pause_max_s",
        "time_long_pause_count",
    )

    def test_unaligned_nan_transcript_features_named_by_issues(self):
        features, issues = acoustic_pack.extract_acoustic_features(
            _tone(145, 2.0), 16000, allow_unaligned=True
        )
        for key in self._TRANSCRIPT_KEYS:
            assert math.isnan(features[key]), key
            assert any(issue.feature == key for issue in issues), key

    def test_no_voice_nan_vad_features_named_by_issues(self):
        doc = _doc((_utt("u1", "p1", 0.2, 1.2),), (DocumentSpeaker(id="p1"),))
        features, issues = acoustic_pack.extract_acoustic_features(
            np.zeros(16000), 16000, document=doc, target_speaker="p1"
        )
        for key in self._VAD_KEYS:
            assert math.isnan(features[key]), key
            assert any(issue.feature == key for issue in issues), key

    def test_no_word_tokens_nan_rates_named_by_issues(self):
        doc = _doc(
            (_utt("u1", "p1", 0.0, 1.0, tokens=(_tok("t1", "\u00e0", kind="filler"),)),),
            (DocumentSpeaker(id="p1"),),
        )
        features, issues = acoustic_pack.extract_acoustic_features(
            _tone(145, 1.0), 16000, document=doc
        )
        for key in (
            "time_words_per_min",
            "time_syllables_per_min",
            "time_articulation_rate_syllables_per_s",
        ):
            assert math.isnan(features[key]), key
            assert any(issue.feature == key for issue in issues), key

    def test_no_examiner_nan_latency_and_overlap_named_by_issues(self):
        doc = _doc((_utt("u1", "p1", 0.0, 1.0),), (DocumentSpeaker(id="p1"),))
        features, issues = acoustic_pack.extract_acoustic_features(
            _tone(145, 1.0), 16000, document=doc
        )
        for key in ("time_response_latency_s", "time_overlap_s"):
            assert math.isnan(features[key]), key
            assert any(issue.feature == key for issue in issues), key
