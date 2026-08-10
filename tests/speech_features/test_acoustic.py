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
from scipy.signal import butter, sosfilt

import speech_features.acoustic as acoustic
from speech_features import list_features
from speech_features.document import (
    DocumentSpeaker,
    DocumentToken,
    DocumentUtterance,
    SpeechDocument,
)
from speech_features.features import acoustic as acoustic_pack
from speech_features.features.acoustic.resonance import _formant_candidates
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


def _chirp(f0, f1, duration_s, sample_rate=16000, amplitude=0.5):
    n = int(round(sample_rate * duration_s))
    t = np.arange(n) / sample_rate
    phase = 2 * math.pi * (f0 * t + 0.5 * (f1 - f0) / duration_s * t**2)
    return amplitude * np.sin(phase)


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
    "spectral_b1_mean_hz",
    "spectral_b1_sd_hz",
    "spectral_b2_mean_hz",
    "spectral_b2_sd_hz",
    "spectral_b3_mean_hz",
    "spectral_b3_sd_hz",
    "spectral_centroid_mean_hz",
    "spectral_centroid_sd_hz",
    "spectral_entropy_mean",
    "spectral_entropy_sd",
    "spectral_f1_mean_hz",
    "spectral_f1_sd_hz",
    "spectral_f2_mean_hz",
    "spectral_f2_sd_hz",
    "spectral_f3_mean_hz",
    "spectral_f3_sd_hz",
    "spectral_flatness_mean",
    "spectral_flatness_sd",
    "spectral_flux_mean",
    "spectral_flux_sd",
    "spectral_rolloff_85_mean_hz",
    "spectral_rolloff_85_sd_hz",
    "spectral_slope_mean_db_per_hz",
    "spectral_slope_sd_db_per_hz",
    "spectral_spread_mean_hz",
    "spectral_spread_sd_hz",
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
    "time_syllable_duration_cv",
    "time_syllable_duration_mean_s",
    "time_syllable_duration_npvi",
    "time_syllable_duration_sd_s",
    "time_syllables_per_min",
    "time_voiced_segment_mean_s",
    "time_voiced_segment_sd_s",
    "time_words_per_min",
    "voice_cpp_iqr_db",
    "voice_cpp_mean_db",
    "voice_cpp_median_db",
    "voice_cpp_sd_db",
    "voice_f0_abs_change_hz",
    "voice_f0_cv",
    "voice_f0_iqr_hz",
    "voice_f0_mean_hz",
    "voice_f0_median_hz",
    "voice_f0_range_5_95_hz",
    "voice_f0_sd_hz",
    "voice_f0_slope_hz_per_s",
    "voice_hnr_iqr_db",
    "voice_hnr_mean_db",
    "voice_hnr_median_db",
    "voice_hnr_sd_db",
    "voice_intensity_iqr_db",
    "voice_intensity_mean_dbfs",
    "voice_intensity_median_dbfs",
    "voice_intensity_sd_db",
    "voice_intensity_slope_db_per_s",
    "voice_jitter_local",
    "voice_shimmer_local",
    "voice_voiced_ratio",
)

EXPECTED_NEURO_ACOUSTIC_KEYS = (
    "spectral_energy_mean_db",
    "spectral_energy_sd_db",
    "spectral_kurtosis_mean",
    "spectral_kurtosis_sd",
    "spectral_low_high_energy_ratio_db",
    "spectral_skewness_mean",
    "spectral_skewness_sd",
    "time_between_utterance_pause_proportion",
    "time_max_local_speech_rate_wpm",
    "time_pause_cv",
    "time_pause_iqr_s",
    "time_pause_median_s",
    "time_pause_proportion",
    "time_pause_total_s",
    "time_speech_segment_count",
    "time_speech_segment_cv",
    "time_speech_segment_iqr_s",
    "time_speech_segment_max_s",
    "time_speech_segment_median_s",
    "time_speech_segment_rate_per_min",
    "time_timing_acceleration_per_min2",
    "time_timing_event_entropy",
    "time_timing_event_rate_per_min",
    "voice_break_count",
    "voice_break_proportion",
    "voice_break_rate_per_min",
    "voice_f0_mad_semitones",
    "voice_f0_range_semitones",
    "voice_intensity_cv",
    "voice_intensity_range_db",
    "voice_nhr_mean_db",
    *(
        f"spectral_mfcc_{coefficient}_{stat}"
        for coefficient in range(1, 14)
        for stat in ("mean", "sd", "skewness", "kurtosis")
    ),
)
EXPECTED_ACOUSTIC_KEYS = tuple(sorted((*EXPECTED_ACOUSTIC_KEYS, *EXPECTED_NEURO_ACOUSTIC_KEYS)))

# The 30 Task 7 keys in their formula groups (sorted order is the catalog's).
RESONANCE_KEYS = (
    "spectral_f1_mean_hz",
    "spectral_f1_sd_hz",
    "spectral_f2_mean_hz",
    "spectral_f2_sd_hz",
    "spectral_f3_mean_hz",
    "spectral_f3_sd_hz",
    "spectral_b1_mean_hz",
    "spectral_b1_sd_hz",
    "spectral_b2_mean_hz",
    "spectral_b2_sd_hz",
    "spectral_b3_mean_hz",
    "spectral_b3_sd_hz",
)
SPECTRUM_KEYS = (
    "spectral_centroid_mean_hz",
    "spectral_centroid_sd_hz",
    "spectral_spread_mean_hz",
    "spectral_spread_sd_hz",
    "spectral_slope_mean_db_per_hz",
    "spectral_slope_sd_db_per_hz",
    "spectral_rolloff_85_mean_hz",
    "spectral_rolloff_85_sd_hz",
    "spectral_flux_mean",
    "spectral_flux_sd",
    "spectral_flatness_mean",
    "spectral_flatness_sd",
    "spectral_entropy_mean",
    "spectral_entropy_sd",
)
RHYTHM_KEYS = (
    "time_syllable_duration_mean_s",
    "time_syllable_duration_sd_s",
    "time_syllable_duration_cv",
    "time_syllable_duration_npvi",
)

# The 24 Task 6 keys in their formula groups (sorted order is the catalog's).
VOICE_F0_KEYS = (
    "voice_f0_mean_hz",
    "voice_f0_median_hz",
    "voice_f0_sd_hz",
    "voice_f0_cv",
    "voice_f0_iqr_hz",
    "voice_f0_range_5_95_hz",
    "voice_f0_slope_hz_per_s",
    "voice_f0_abs_change_hz",
)
VOICE_INTENSITY_KEYS = (
    "voice_intensity_mean_dbfs",
    "voice_intensity_median_dbfs",
    "voice_intensity_sd_db",
    "voice_intensity_iqr_db",
    "voice_intensity_slope_db_per_s",
)
VOICE_HNR_KEYS = (
    "voice_hnr_mean_db",
    "voice_hnr_median_db",
    "voice_hnr_sd_db",
    "voice_hnr_iqr_db",
)
VOICE_CPP_KEYS = (
    "voice_cpp_mean_db",
    "voice_cpp_median_db",
    "voice_cpp_sd_db",
    "voice_cpp_iqr_db",
)
VOICE_KEYS = (
    *VOICE_F0_KEYS,
    "voice_voiced_ratio",
    *VOICE_INTENSITY_KEYS,
    "voice_jitter_local",
    "voice_shimmer_local",
    *VOICE_HNR_KEYS,
    *VOICE_CPP_KEYS,
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
    def test_exact_acoustic_keys_registered(self):
        features = list_features(pack="acoustic")
        assert [f.key for f in features] == list(EXPECTED_ACOUSTIC_KEYS)
        assert EXPECTED_ACOUSTIC_KEYS == tuple(sorted(EXPECTED_ACOUSTIC_KEYS))
        assert set(VOICE_KEYS) <= set(EXPECTED_ACOUSTIC_KEYS)
        assert set((*RESONANCE_KEYS, *SPECTRUM_KEYS)) <= set(EXPECTED_ACOUSTIC_KEYS)
        assert set(RHYTHM_KEYS) <= set(EXPECTED_ACOUSTIC_KEYS)

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
            "INSUFFICIENT_VOICED_FRAMES",
            "INSUFFICIENT_CYCLES",
            "INSUFFICIENT_SPEECH_FRAMES",
            "INSUFFICIENT_FORMANTS",
        }
        assert math.isnan(bundle.recordings.loc[0, "audio_rms_dbfs"])
        assert math.isnan(bundle.recordings.loc[0, "voice_f0_mean_hz"])


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
        assert completed.stdout.strip() == str(len(EXPECTED_ACOUSTIC_KEYS))


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


# ---------------------------------------------------------------------------
# Task 6: acoustic phonation and prosody (brief keys and behaviors)
# ---------------------------------------------------------------------------
class TestPhonationTone:
    """A deterministic 200 Hz tone; F0 lands on an autocorrelation lag."""

    def _voice(self, signal):
        features, _ = acoustic_pack.extract_acoustic_features(signal, 16000, allow_unaligned=True)
        return features

    def test_f0_summaries_match_tone(self):
        f = self._voice(_tone(200, 1.0))
        for key in ("voice_f0_mean_hz", "voice_f0_median_hz"):
            assert 195.0 < f[key] < 210.0, key
        assert f["voice_f0_sd_hz"] < 1.0
        assert f["voice_f0_cv"] < 0.01
        assert f["voice_f0_iqr_hz"] < 1.0
        assert f["voice_f0_range_5_95_hz"] < 1.0
        assert abs(f["voice_f0_slope_hz_per_s"]) < 0.5
        assert f["voice_f0_abs_change_hz"] < 0.1
        assert f["voice_voiced_ratio"] > 0.9

    def test_intensity_summaries_match_tone_level(self):
        f = self._voice(_tone(200, 1.0, amplitude=0.5))
        # Hamming-windowed frame RMS: 0.5/sqrt(2) * sqrt(mean(hamming^2)).
        assert f["voice_intensity_mean_dbfs"] == pytest.approx(-13.0, abs=0.5)
        assert f["voice_intensity_median_dbfs"] == pytest.approx(-13.0, abs=0.5)
        assert f["voice_intensity_sd_db"] < 0.1
        assert f["voice_intensity_iqr_db"] < 0.1
        assert abs(f["voice_intensity_slope_db_per_s"]) < 0.5

    def test_jitter_and_shimmer_vanish_on_a_steady_tone(self):
        f = self._voice(_tone(200, 1.0))
        assert f["voice_jitter_local"] < 0.001
        assert f["voice_shimmer_local"] < 0.001

    def test_hnr_positive_and_cpp_prominent_on_tone(self):
        f = self._voice(_tone(200, 1.0))
        for key in VOICE_HNR_KEYS:
            assert math.isfinite(f[key]), key
        assert f["voice_hnr_mean_db"] > 3.0
        assert f["voice_hnr_median_db"] > 3.0
        for key in VOICE_CPP_KEYS:
            assert math.isfinite(f[key]), key
        assert f["voice_cpp_mean_db"] > 0.02
        assert f["voice_cpp_median_db"] > 0.02

    def test_all_voice_keys_finite_on_tone(self):
        f = self._voice(_tone(200, 1.0))
        for key in VOICE_KEYS:
            assert math.isfinite(f[key]), key


class TestF0SlopeAndChange:
    def test_chirp_slope_is_ols_over_voiced_frame_times(self):
        # Linear 150 -> 250 Hz sweep over 2 s: slope 50 Hz/s. The shared
        # mean-energy VAD keeps most frames voiced; the OLS slope over the
        # actual voiced frame times must track the nominal rate.
        f, _ = acoustic_pack.extract_acoustic_features(
            _chirp(150, 250, 2.0), 16000, allow_unaligned=True
        )
        assert f["voice_f0_slope_hz_per_s"] == pytest.approx(50.0, abs=5.0)
        assert f["voice_f0_sd_hz"] > 10.0
        # Mean absolute change of consecutive voiced F0 estimates: with 10 ms
        # hops, the nominal per-hop sweep step is 0.5 Hz; lag quantization
        # widens it, so the assertion is a broad band around the formula.
        assert 0.3 < f["voice_f0_abs_change_hz"] < 1.2
        assert math.isfinite(f["voice_f0_mean_hz"])
        assert math.isfinite(f["voice_f0_median_hz"])

    def test_flat_tone_has_no_slope_and_no_change(self):
        f, _ = acoustic_pack.extract_acoustic_features(_tone(200, 2.0), 16000, allow_unaligned=True)
        assert abs(f["voice_f0_slope_hz_per_s"]) < 0.5
        assert f["voice_f0_abs_change_hz"] < 0.1


class TestAmplitudeModulation:
    def _am(self, depth, duration_s=1.0):
        n = int(round(16000 * duration_s))
        t = np.arange(n) / 16000
        a = 1.0 - depth * 0.5 * (1.0 + np.cos(2 * math.pi * 5.0 * t))
        return a * np.sin(2 * math.pi * 200.0 * t)

    def test_shimmer_rises_with_amplitude_modulation(self):
        modulated, _ = acoustic_pack.extract_acoustic_features(
            self._am(0.8), 16000, allow_unaligned=True
        )
        flat, _ = acoustic_pack.extract_acoustic_features(
            self._am(0.0), 16000, allow_unaligned=True
        )
        assert modulated["voice_shimmer_local"] > 0.02
        assert modulated["voice_shimmer_local"] > flat["voice_shimmer_local"] + 0.01
        assert flat["voice_shimmer_local"] < 0.001

    def test_intensity_variability_tracks_modulation(self):
        modulated, _ = acoustic_pack.extract_acoustic_features(
            self._am(0.8), 16000, allow_unaligned=True
        )
        flat, _ = acoustic_pack.extract_acoustic_features(
            self._am(0.0), 16000, allow_unaligned=True
        )
        assert modulated["voice_intensity_sd_db"] > 0.5
        assert modulated["voice_intensity_sd_db"] > flat["voice_intensity_sd_db"] + 0.5


class TestPeriodModulation:
    def _fm(self, deviation, duration_s=1.0):
        n = int(round(16000 * duration_s))
        t = np.arange(n) / 16000
        phase = (
            2
            * math.pi
            * (200.0 * t - deviation / (2 * math.pi * 5.0) * np.cos(2 * math.pi * 5.0 * t))
        )
        return 0.5 * np.sin(phase)

    def test_jitter_rises_with_period_modulation(self):
        modulated, _ = acoustic_pack.extract_acoustic_features(
            self._fm(20.0), 16000, allow_unaligned=True
        )
        flat, _ = acoustic_pack.extract_acoustic_features(
            self._fm(0.0), 16000, allow_unaligned=True
        )
        assert modulated["voice_jitter_local"] > 0.01
        assert modulated["voice_jitter_local"] > flat["voice_jitter_local"] + 0.005
        assert flat["voice_jitter_local"] < 0.001

    def test_f0_change_and_variability_track_modulation(self):
        modulated, _ = acoustic_pack.extract_acoustic_features(
            self._fm(20.0), 16000, allow_unaligned=True
        )
        assert modulated["voice_f0_abs_change_hz"] > 1.0
        assert modulated["voice_f0_sd_hz"] > 5.0


class TestHnrMonotonicity:
    def test_hnr_decreases_as_additive_noise_increases(self):
        rng = np.random.default_rng(7)
        base = _tone(200, 1.0)
        means = []
        for noise_amplitude in (0.0, 0.1, 0.5):
            signal = base + noise_amplitude * rng.normal(0.0, 1.0, 16000)
            f, _ = acoustic_pack.extract_acoustic_features(signal, 16000, allow_unaligned=True)
            assert math.isfinite(f["voice_hnr_mean_db"])
            means.append(f["voice_hnr_mean_db"])
        assert means[0] > means[1] + 0.5
        assert means[1] > means[2] + 0.5


class TestCppOrdering:
    def test_tonal_signal_has_higher_cpp_than_noisy_signal(self):
        # A harmonic-rich sawtooth has a prominent cepstral peak at the pitch
        # period; adding broadband noise flattens the spectrum and must lower
        # the cepstral peak prominence while F0 stays detected.
        rng = np.random.default_rng(11)
        n = 16000
        t = np.arange(n) / 16000
        sawtooth = 0.5 * (2.0 * ((200.0 * t) % 1.0) - 1.0)
        noisy = sawtooth + 0.2 * rng.normal(0.0, 1.0, n)
        clean, _ = acoustic_pack.extract_acoustic_features(sawtooth, 16000, allow_unaligned=True)
        degraded, _ = acoustic_pack.extract_acoustic_features(noisy, 16000, allow_unaligned=True)
        assert math.isfinite(clean["voice_cpp_mean_db"])
        assert math.isfinite(degraded["voice_cpp_mean_db"])
        assert clean["voice_cpp_mean_db"] > degraded["voice_cpp_mean_db"] + 0.3


class TestPhonationInsufficiency:
    def test_digital_silence_yields_nan_with_exact_key_issues(self):
        features, issues = acoustic_pack.extract_acoustic_features(
            np.zeros(16000), 16000, allow_unaligned=True
        )
        for key in VOICE_KEYS:
            assert math.isnan(features[key]), key
            assert any(issue.feature == key for issue in issues), key
        for key in ("voice_jitter_local", "voice_shimmer_local"):
            assert any(
                issue.code == "INSUFFICIENT_CYCLES" for issue in issues if issue.feature == key
            )
        for key in VOICE_KEYS:
            if key in ("voice_jitter_local", "voice_shimmer_local"):
                continue
            assert any(
                issue.code == "INSUFFICIENT_VOICED_FRAMES"
                for issue in issues
                if issue.feature == key
            ), key

    def test_single_voiced_frame_is_insufficient(self):
        # Exactly one 25 ms frame (400 samples): one voiced F0 estimate cannot
        # characterise phonation; distribution, slope, jitter, shimmer, HNR
        # and CPP must be NaN with issues, while the ratio stays exact.
        features, issues = acoustic_pack.extract_acoustic_features(
            _tone(160, 0.025), 16000, allow_unaligned=True
        )
        assert features["voice_voiced_ratio"] == pytest.approx(1.0)
        for key in (*VOICE_F0_KEYS, *VOICE_INTENSITY_KEYS, *VOICE_HNR_KEYS, *VOICE_CPP_KEYS):
            assert math.isnan(features[key]), key
            assert any(
                issue.code == "INSUFFICIENT_VOICED_FRAMES" and issue.feature == key
                for issue in issues
            ), key
        for key in ("voice_jitter_local", "voice_shimmer_local"):
            assert math.isnan(features[key]), key
            assert any(
                issue.code == "INSUFFICIENT_CYCLES" and issue.feature == key for issue in issues
            ), key


class TestPhonationSpeakerIsolation:
    def test_examiner_audio_does_not_affect_voice_measures(self):
        # Loud 1 kHz examiner tone before a quiet 160 Hz participant tone:
        # every voice measure must describe the participant's clips only.
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
        assert 150.0 < features["voice_f0_mean_hz"] < 175.0
        assert features["voice_f0_sd_hz"] < 1.0
        assert features["voice_intensity_mean_dbfs"] == pytest.approx(-21.0, abs=1.5)
        assert math.isfinite(features["voice_hnr_mean_db"])
        assert math.isfinite(features["voice_cpp_mean_db"])


# ---------------------------------------------------------------------------
# Task 6 fix round 1 (Agy review f2e44a1..535b55d): CPP dB scale, interval
# boundary transitions, intensity observation invariant
# ---------------------------------------------------------------------------
def _reference_cpp(signal, sample_rate=16000):
    """Reference CPP from the documented ``20*log10(|X|)`` cepstrum.

    Mirrors the phonation module formula: real cepstrum of the dB-scaled log
    magnitude spectrum, peak inside the pitch-period quefrency range, minus
    the OLS baseline over that same range. A natural-log cepstrum would yield
    values 20/ln(10) ~ 8.69 times smaller and fail the equality below.
    """
    cfg = ExtractionConfig()
    win = acoustic._frames(signal, cfg.frame_size, cfg.hop_size)
    energy = (win**2).sum(axis=1)
    voiced = acoustic._energy_vad(energy)
    f0, _ = acoustic._f0_per_frame(win, cfg.frame_size, sample_rate, cfg)
    frames = win[voiced & np.isfinite(f0)]
    n_min = max(int(math.ceil(sample_rate / cfg.pitch_max_hz)), 1)
    n_max = min(int(math.floor(sample_rate / cfg.pitch_min_hz)), cfg.frame_size - 1)
    region = np.arange(n_min, n_max + 1, dtype=float)
    cepstrum = np.fft.irfft(
        20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(frames, axis=1)), 1e-12)), axis=1
    )
    sub = cepstrum[:, n_min : n_max + 1]
    centered = region - region.mean()
    slope = (sub - sub.mean(axis=1, keepdims=True)) @ centered / (centered**2).sum()
    intercept = sub.mean(axis=1) - slope * region.mean()
    peak = np.argmax(sub, axis=1)
    baseline = slope * region[peak] + intercept
    return float(np.mean(sub[np.arange(sub.shape[0]), peak] - baseline))


class TestCppDbScale:
    """Agy fix 1 (Critical): CPP must be dB-scaled, not natural-log."""

    def test_cpp_matches_db_scaled_reference_formula(self):
        signal = _tone(200, 0.2)
        features, _ = acoustic_pack.extract_acoustic_features(signal, 16000, allow_unaligned=True)
        reference = _reference_cpp(signal)
        assert features["voice_cpp_mean_db"] == pytest.approx(reference, rel=1e-9)
        # dB scaling of a 200 Hz tone lands near 0.53 dB; a natural-log
        # cepstrum stays near 0.06 and must not be mistaken for a dB value.
        assert features["voice_cpp_mean_db"] > 0.3


class TestIntervalBoundaryTransitions:
    """Agy fix 2 (Important): no cross-boundary F0/period/RMS transitions."""

    def test_disjoint_intervals_create_no_cross_boundary_transitions(self):
        # Two disjoint steady intervals with different F0 (200 Hz, 400 Hz) and
        # amplitude (0.5, 0.3): both tones are exactly periodic in the 25 ms
        # frame, so every intra-interval transition is zero. F0 absolute
        # change, jitter, and shimmer must stay at zero instead of absorbing
        # a fake transition across the boundary.
        audio = np.concatenate(
            [
                np.zeros(int(16000 * 0.2)),
                _tone(200, 0.5, amplitude=0.5),
                np.zeros(int(16000 * 0.5)),
                _tone(400, 0.5, amplitude=0.3),
            ]
        )
        doc = _doc(
            (_utt("u1", "p1", 0.2, 0.7), _utt("u2", "p1", 1.2, 1.7)),
            (DocumentSpeaker(id="p1"),),
        )
        features, _ = acoustic_pack.extract_acoustic_features(audio, 16000, document=doc)
        assert features["voice_f0_abs_change_hz"] < 0.1
        assert features["voice_jitter_local"] < 0.001
        assert features["voice_shimmer_local"] < 0.001
        # Both intervals still contribute to the distribution summaries.
        assert 250.0 < features["voice_f0_mean_hz"] < 350.0
        assert math.isfinite(features["voice_f0_sd_hz"])
        assert math.isfinite(features["voice_intensity_mean_dbfs"])


class TestIntensityInvariant:
    """Agy fix 3 (Important): prove the intensity observation invariant.

    With at least two voiced F0 frames, every such frame cleared the absolute
    silence energy floor (1e-6), so its RMS is positive and the intensity
    summaries always have at least the voiced F0 frames to summarize. The
    smallest clearing case is exactly two voiced frames (560 samples of a
    tiled 200 Hz period, two bitwise-identical frames).
    """

    def test_two_voiced_frames_yield_finite_intensity_summaries(self):
        # Two bitwise-identical frames: tile one exact 200 Hz period so the
        # frame energies are exactly equal and the mean-energy VAD marks both
        # frames voiced (a continuous-phase sine differs in the last ULP and
        # drops one frame, so it cannot build this minimal case).
        period = np.sin(2 * math.pi * 200 * np.arange(80) / 16000)
        features, _ = acoustic_pack.extract_acoustic_features(
            np.tile(period, 7), 16000, allow_unaligned=True
        )
        assert features["voice_voiced_ratio"] == pytest.approx(1.0)
        assert math.isfinite(features["voice_f0_mean_hz"])
        for key in VOICE_INTENSITY_KEYS:
            assert math.isfinite(features[key]), key


class TestCycleInsufficiencyAcrossIntervals:
    """Agy fix 2 edge: voiced frames split across intervals have no pairs."""

    def test_no_intra_interval_transition_yields_nan_with_issues(self):
        # Two intervals holding one voiced frame each: two voiced frames in
        # total, but no consecutive pair inside any interval. The diff-based
        # keys must be NaN with per-key issues, never a cross-boundary value.
        audio = np.concatenate(
            [
                np.zeros(int(16000 * 0.2)),
                _tone(160, 0.025),
                np.zeros(int(16000 * 0.5)),
                _tone(200, 0.025),
            ]
        )
        doc = _doc(
            (_utt("u1", "p1", 0.2, 0.225), (_utt("u2", "p1", 0.725, 0.75))),
            (DocumentSpeaker(id="p1"),),
        )
        features, issues = acoustic_pack.extract_acoustic_features(audio, 16000, document=doc)
        for key in ("voice_f0_abs_change_hz", "voice_jitter_local", "voice_shimmer_local"):
            assert math.isnan(features[key]), key
            assert any(
                issue.code == "INSUFFICIENT_CYCLES" and issue.feature == key for issue in issues
            ), key
        assert math.isfinite(features["voice_f0_mean_hz"])


# ---------------------------------------------------------------------------
# Task 7: acoustic resonance, spectrum, and rhythm (brief keys and behaviors)
# ---------------------------------------------------------------------------
def _resonator(x, f, b):
    """One second-order digital resonator with known pole frequency/bandwidth."""
    r = math.exp(-math.pi * b / 16000)
    th = 2 * math.pi * f / 16000
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = x[i] + 2 * r * math.cos(th) * out[i - 1] - r * r * out[i - 2]
    return out


def _vowel(fs=(500.0, 1500.0, 2500.0), bs=(80.0, 120.0, 160.0), f0=100.0, duration_s=1.0):
    """Deterministic source-filter vowel: impulse train + three resonators.

    The 100 Hz impulse train makes the source exactly periodic over the 400
    sample frame (4 periods), so every frame is bitwise identical and the
    shared mean-energy VAD marks all frames voiced; the resonators place
    spectral peaks at exactly the configured formants.
    """
    n = int(round(16000 * duration_s))
    src = np.zeros(n)
    src[:: int(round(16000 / f0))] = 1.0
    y = src.astype(float)
    for f, b in zip(fs, bs):
        y = _resonator(y, f, b)
    y /= np.max(np.abs(y))
    return 0.5 * y


def _lowpass_noise(cutoff, order, seed, duration_s=1.0):
    rng = np.random.default_rng(seed)
    sos = butter(order, cutoff, fs=16000, output="sos")
    return sosfilt(sos, rng.normal(0.0, 1.0, int(round(16000 * duration_s))))


class TestResonanceSourceFilterVowel:
    """LPC formants/bandwidths recovered from a known source-filter vowel.

    Tolerances are intentionally wide where LPC/frame resolution is coarse:
    the pre-emphasis + short-frame autocorrelation bias shifts F1 by ~9% and
    bandwidths by up to ~45% on the synthetic vowel, so the assertions bound
    the estimates around the nominal values instead of demanding exactness.
    """

    def test_formants_and_bandwidths_match_the_vowel(self):
        f, _ = acoustic_pack.extract_acoustic_features(_vowel(), 16000, allow_unaligned=True)
        # F1 within 10%, F2/F3 within 5% (measured 546.6 / 1503.2 / 2481.5).
        assert f["spectral_f1_mean_hz"] == pytest.approx(500.0, rel=0.10)
        assert f["spectral_f2_mean_hz"] == pytest.approx(1500.0, rel=0.05)
        assert f["spectral_f3_mean_hz"] == pytest.approx(2500.0, rel=0.05)
        # Bandwidths within 50% (measured 115.9 / 73.9 / 190.3 vs 80/120/160).
        assert f["spectral_b1_mean_hz"] == pytest.approx(80.0, rel=0.50)
        assert f["spectral_b2_mean_hz"] == pytest.approx(120.0, rel=0.50)
        assert f["spectral_b3_mean_hz"] == pytest.approx(160.0, rel=0.50)

    def test_formants_ordered_and_stable_across_frames(self):
        f, _ = acoustic_pack.extract_acoustic_features(_vowel(), 16000, allow_unaligned=True)
        assert f["spectral_f1_mean_hz"] < f["spectral_f2_mean_hz"] < f["spectral_f3_mean_hz"]
        # The vowel is exactly periodic in the frame, so per-frame estimates
        # are nearly constant; a real recording would show larger SDs.
        for key in RESONANCE_KEYS:
            assert math.isfinite(f[key]), key
        for key in ("spectral_f1_sd_hz", "spectral_f2_sd_hz", "spectral_f3_sd_hz"):
            assert f[key] < 10.0, key

    def test_lpc_order_from_config_is_respected(self):
        f, _ = acoustic_pack.extract_acoustic_features(
            _vowel(),
            16000,
            allow_unaligned=True,
            config=ExtractionConfig(lpc_order=10),
        )
        assert math.isfinite(f["spectral_f1_mean_hz"])


class TestResonanceInsufficiency:
    def test_digital_silence_yields_nan_with_exact_key_issues(self):
        features, issues = acoustic_pack.extract_acoustic_features(
            np.zeros(16000), 16000, allow_unaligned=True
        )
        for key in RESONANCE_KEYS:
            assert math.isnan(features[key]), key
            assert any(
                issue.code == "INSUFFICIENT_SPEECH_FRAMES" and issue.feature == key
                for issue in issues
            ), key

    def test_noise_yields_nan_formants_with_speech_frame_issues(self):
        # White noise is VAD-voiced but has no stable periodic excitation, so
        # no frame qualifies as a stable voiced frame; no formant is invented.
        rng = np.random.default_rng(7)
        features, issues = acoustic_pack.extract_acoustic_features(
            0.3 * rng.normal(0.0, 1.0, 16000), 16000, allow_unaligned=True
        )
        for key in RESONANCE_KEYS:
            assert math.isnan(features[key]), key
            assert any(
                issue.code == "INSUFFICIENT_SPEECH_FRAMES" and issue.feature == key
                for issue in issues
            ), key

    def test_single_voiced_frame_yields_insufficient_formants(self):
        # One 25 ms frame of the vowel: one stable voiced frame is not enough
        # to characterise the distribution; no mean/SD is fabricated.
        features, issues = acoustic_pack.extract_acoustic_features(
            _vowel(duration_s=0.025), 16000, allow_unaligned=True
        )
        for key in RESONANCE_KEYS:
            assert math.isnan(features[key]), key
            assert any(
                issue.code == "INSUFFICIENT_FORMANTS" and issue.feature == key for issue in issues
            ), key


class TestSpectrumToneAndNoise:
    """Centroid/spread/slope/rolloff/flatness/entropy on tone vs noise."""

    def test_tone_spectrum_matches_its_frequency(self):
        f, _ = acoustic_pack.extract_acoustic_features(
            _tone(1000, 1.0), 16000, allow_unaligned=True
        )
        assert f["spectral_centroid_mean_hz"] == pytest.approx(1000.0, abs=40.0)
        assert f["spectral_rolloff_85_mean_hz"] == pytest.approx(1000.0, abs=80.0)
        assert f["spectral_spread_mean_hz"] < 300.0
        assert f["spectral_flatness_mean"] < 0.05
        assert f["spectral_entropy_mean"] < 0.3
        assert abs(f["spectral_slope_mean_db_per_hz"]) < 0.02
        # A steady tone has identical frames: near-zero per-frame variability.
        assert f["spectral_centroid_sd_hz"] < 1.0
        assert f["spectral_flux_mean"] < 1e-3

    def test_white_noise_spectrum_is_flat_and_broad(self):
        rng = np.random.default_rng(7)
        f, _ = acoustic_pack.extract_acoustic_features(
            0.3 * rng.normal(0.0, 1.0, 16000), 16000, allow_unaligned=True
        )
        # Uniform power over 0..8 kHz: centroid at SR/4 = 4000, 85% rolloff at
        # 0.85 * 8000 = 6800, near-flat dB slope, high flatness/entropy.
        assert f["spectral_centroid_mean_hz"] == pytest.approx(4000.0, rel=0.15)
        assert f["spectral_rolloff_85_mean_hz"] == pytest.approx(6800.0, abs=500.0)
        assert f["spectral_spread_mean_hz"] > 1500.0
        assert abs(f["spectral_slope_mean_db_per_hz"]) < 0.005
        assert 0.3 < f["spectral_flatness_mean"] < 0.8
        assert f["spectral_entropy_mean"] > 0.85
        assert f["spectral_centroid_sd_hz"] > 50.0

    def test_lowpass_noise_has_negative_slope_and_low_rolloff(self):
        f, _ = acoustic_pack.extract_acoustic_features(
            _lowpass_noise(2000, 4, seed=8), 16000, allow_unaligned=True
        )
        # Butterworth LP at 2 kHz: most power below the cutoff, dB level
        # falling with frequency -> OLS slope clearly negative.
        assert f["spectral_slope_mean_db_per_hz"] < -0.005
        assert f["spectral_centroid_mean_hz"] < 2000.0
        assert f["spectral_rolloff_85_mean_hz"] < 3000.0
        assert f["spectral_flatness_mean"] < 0.1

    def test_all_spectrum_keys_finite_on_noise(self):
        rng = np.random.default_rng(9)
        f, _ = acoustic_pack.extract_acoustic_features(
            0.3 * rng.normal(0.0, 1.0, 16000), 16000, allow_unaligned=True
        )
        for key in SPECTRUM_KEYS:
            assert math.isfinite(f[key]), key


class TestSpectrumFlux:
    def test_alternating_tones_raise_flux_above_steady_tone(self):
        n = 16000
        t = np.arange(n) / 16000
        freq = 200.0 + 100.0 * (np.arange(n) // 400 % 2)  # 200/300 Hz per frame
        alternating, _ = acoustic_pack.extract_acoustic_features(
            0.5 * np.sin(2 * math.pi * freq * t), 16000, allow_unaligned=True
        )
        steady, _ = acoustic_pack.extract_acoustic_features(
            _tone(250, 1.0), 16000, allow_unaligned=True
        )
        assert alternating["spectral_flux_mean"] > 0.3
        assert alternating["spectral_flux_mean"] > steady["spectral_flux_mean"] + 0.2

    def test_no_flux_across_disjoint_target_intervals(self):
        # Two steady disjoint intervals at 200 Hz and 800 Hz: every frame
        # inside each interval is identical, so intra-interval flux is zero.
        # If the implementation paired the last frame of the first interval
        # with the first frame of the second, flux would jump to ~1.4.
        audio = np.concatenate(
            [
                np.zeros(int(16000 * 0.2)),
                _tone(200, 0.5),
                np.zeros(int(16000 * 0.5)),
                _tone(800, 0.5),
            ]
        )
        doc = _doc(
            (_utt("u1", "p1", 0.2, 0.7), _utt("u2", "p1", 1.2, 1.7)),
            (DocumentSpeaker(id="p1"),),
        )
        features, _ = acoustic_pack.extract_acoustic_features(audio, 16000, document=doc)
        assert features["spectral_flux_mean"] < 1e-3
        # Both intervals still contribute to the pooled distribution.
        assert features["spectral_centroid_mean_hz"] == pytest.approx(500.0, abs=5.0)
        assert math.isfinite(features["spectral_centroid_sd_hz"])

    def test_flux_needs_a_consecutive_pair_inside_one_interval(self):
        # Two intervals holding one voiced frame each: no intra-interval
        # consecutive pair exists, so flux must be NaN with per-key issues.
        audio = np.concatenate(
            [
                np.zeros(int(16000 * 0.2)),
                _tone(160, 0.025),
                np.zeros(int(16000 * 0.5)),
                _tone(200, 0.025),
            ]
        )
        doc = _doc(
            (_utt("u1", "p1", 0.2, 0.225), (_utt("u2", "p1", 0.725, 0.75))),
            (DocumentSpeaker(id="p1"),),
        )
        features, issues = acoustic_pack.extract_acoustic_features(audio, 16000, document=doc)
        for key in ("spectral_flux_mean", "spectral_flux_sd"):
            assert math.isnan(features[key]), key
            assert any(
                issue.code == "INSUFFICIENT_SPEECH_FRAMES" and issue.feature == key
                for issue in issues
            ), key


class TestSpectrumInsufficiency:
    def test_digital_silence_yields_nan_with_exact_key_issues(self):
        features, issues = acoustic_pack.extract_acoustic_features(
            np.zeros(16000), 16000, allow_unaligned=True
        )
        for key in SPECTRUM_KEYS:
            assert math.isnan(features[key]), key
            assert any(
                issue.code == "INSUFFICIENT_SPEECH_FRAMES" and issue.feature == key
                for issue in issues
            ), key

    def test_single_voiced_frame_is_insufficient(self):
        features, issues = acoustic_pack.extract_acoustic_features(
            _tone(160, 0.025), 16000, allow_unaligned=True
        )
        for key in SPECTRUM_KEYS:
            assert math.isnan(features[key]), key
            assert any(
                issue.code == "INSUFFICIENT_SPEECH_FRAMES" and issue.feature == key
                for issue in issues
            ), key


class TestRhythmHandCalculated:
    """Mean/SD/CV and nPVI from explicit token times, hand-computed."""

    def _doc_with_durations(self, utterance_durations, shared_word_id=False):
        utterances = []
        for i, durations in enumerate(utterance_durations):
            tokens = []
            start = 0.0
            for j, d in enumerate(durations):
                word_id = "w1" if shared_word_id else None
                tokens.append(
                    DocumentToken(
                        id=f"u{i}_t{j}",
                        text="xa",
                        kind="word",
                        start_s=start,
                        end_s=start + d,
                        word_id=word_id,
                    )
                )
                start += d
            utterances.append(_utt(f"u{i}", "p1", 0.0, start, tokens=tokens))
        return _doc(tuple(utterances), (DocumentSpeaker(id="p1"),))

    def test_hand_calculated_mean_sd_cv_npvi(self):
        # Durations [0.2, 0.4, 0.6] in u0 and [0.3, 0.5] in u1.
        # Pooled: mean 0.4; population SD = sqrt(0.1/5) = 0.141421; CV = 0.353553.
        # nPVI: within u0, pairs (0.2,0.4)->0.6667 and (0.4,0.6)->0.4; within
        # u1, pair (0.3,0.5)->0.5; pooled mean 0.522222 -> nPVI 52.2222. A
        # cross-utterance pair (0.6,0.3) would change it to 55.83.
        doc = self._doc_with_durations([[0.2, 0.4, 0.6], [0.3, 0.5]])
        features, _ = acoustic_pack.extract_acoustic_features(
            _tone(145, 2.0), 16000, document=doc, target_speaker="p1"
        )
        assert features["time_syllable_duration_mean_s"] == pytest.approx(0.4, abs=1e-9)
        assert features["time_syllable_duration_sd_s"] == pytest.approx(
            0.14142135623730953, rel=1e-9
        )
        assert features["time_syllable_duration_cv"] == pytest.approx(0.3535533905932738, rel=1e-9)
        assert features["time_syllable_duration_npvi"] == pytest.approx(52.22222222222222, rel=1e-9)

    def test_shared_word_id_does_not_collapse_syllable_tokens(self):
        # Two tokens sharing one word_id still contribute two durations:
        # no Vietnamese word-boundary inference, no syllable merging.
        doc = self._doc_with_durations([[0.2, 0.3]], shared_word_id=True)
        features, _ = acoustic_pack.extract_acoustic_features(
            _tone(145, 1.0), 16000, document=doc, target_speaker="p1"
        )
        assert features["time_syllable_duration_mean_s"] == pytest.approx(0.25, abs=1e-9)
        assert features["time_syllable_duration_sd_s"] == pytest.approx(0.05, abs=1e-9)

    def test_single_utterance_boundaries_never_paired(self):
        # One duration per utterance: mean/SD/CV are defined over the two
        # pooled observations, but nPVI needs a consecutive pair inside one
        # utterance and must never pair across the boundary.
        doc = self._doc_with_durations([[0.2], [0.4]])
        features, issues = acoustic_pack.extract_acoustic_features(
            _tone(145, 2.0), 16000, document=doc, target_speaker="p1"
        )
        assert features["time_syllable_duration_mean_s"] == pytest.approx(0.3, abs=1e-9)
        assert features["time_syllable_duration_sd_s"] == pytest.approx(0.1, abs=1e-9)
        assert features["time_syllable_duration_cv"] == pytest.approx(0.3333333333333333, rel=1e-9)
        assert math.isnan(features["time_syllable_duration_npvi"])
        assert any(
            issue.code == "MISSING_ANNOTATION" and issue.feature == "time_syllable_duration_npvi"
            for issue in issues
        )


class TestRhythmInsufficiency:
    def test_no_document_yields_nan_with_missing_annotation(self):
        features, issues = acoustic_pack.extract_acoustic_features(
            _tone(145, 1.0), 16000, allow_unaligned=True
        )
        for key in RHYTHM_KEYS:
            assert math.isnan(features[key]), key
            assert any(
                issue.code == "MISSING_ANNOTATION" and issue.feature == key for issue in issues
            ), key

    def test_no_word_tokens_with_times_yield_nan_with_issues(self):
        doc = _doc(
            (_utt("u1", "p1", 0.0, 1.0, tokens=(_tok("t1", "\u00e0", kind="filler"),)),),
            (DocumentSpeaker(id="p1"),),
        )
        features, issues = acoustic_pack.extract_acoustic_features(
            _tone(145, 1.0), 16000, document=doc
        )
        for key in RHYTHM_KEYS:
            assert math.isnan(features[key]), key
            assert any(
                issue.code == "MISSING_ANNOTATION" and issue.feature == key for issue in issues
            ), key

    def test_tokens_without_times_are_excluded(self):
        doc = _doc(
            (
                _utt(
                    "u1",
                    "p1",
                    0.0,
                    1.0,
                    tokens=(DocumentToken(id="t1", text="xa", kind="word"),),
                ),
            ),
            (DocumentSpeaker(id="p1"),),
        )
        features, issues = acoustic_pack.extract_acoustic_features(
            _tone(145, 1.0), 16000, document=doc
        )
        for key in RHYTHM_KEYS:
            assert math.isnan(features[key]), key
            assert any(
                issue.code == "MISSING_ANNOTATION" and issue.feature == key for issue in issues
            ), key


class TestResonanceSpectrumIsolation:
    def test_examiner_audio_does_not_affect_resonance_or_spectrum(self):
        # Loud 1 kHz examiner tone before a quiet participant vowel: formants
        # and spectral summaries must describe the vowel clips only.
        examiner = _tone(1000, 1.0, amplitude=0.9)
        participant = _vowel(duration_s=1.0)
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
        assert features["spectral_f1_mean_hz"] == pytest.approx(500.0, rel=0.15)
        assert features["spectral_f2_mean_hz"] == pytest.approx(1500.0, rel=0.10)
        # The vowel's centroid (~650 Hz) is far below the examiner's 1 kHz
        # tone; a leaked examiner tone would drag it toward 1000.
        assert features["spectral_centroid_mean_hz"] < 900.0
        assert math.isfinite(features["spectral_flux_mean"])


# ---------------------------------------------------------------------------
# Task 7 fix round 1 (Agy review 29d0fbb..548a1fc): LPC candidate rejection
# ---------------------------------------------------------------------------
class TestLpcCandidateRejection:
    """Degenerate frames must be rejected, never raise out of extraction."""

    def test_singular_autocorrelation_frame_rejected_without_raising(self):
        # A rank-deficient frame (all-zero pre-emphasized frame) makes the
        # Toeplitz solve singular: scipy raises LinAlgError("Singular
        # principal minor"). The candidate extractor must reject the frame
        # (return no candidates) so extraction falls through to the existing
        # per-key insufficiency issues instead of crashing.
        frame = np.zeros(400)
        candidates = _formant_candidates(frame, 16000, 12)
        assert candidates == []

    def test_nonfinite_autocorrelation_frame_rejected_without_raising(self):
        # A frame containing NaN makes solve_toeplitz raise ValueError
        # ("array must not contain infs or NaNs"); the frame must be rejected
        # as having no candidates rather than propagating the exception.
        frame = np.zeros(400)
        frame[10] = np.nan
        candidates = _formant_candidates(frame, 16000, 12)
        assert candidates == []

    def test_nyquist_root_is_rejected(self):
        # An alternating +-1 frame has a real negative LPC pole: its angle is
        # exactly math.pi, i.e. exactly the Nyquist frequency (sr/2). The
        # candidate filter must require 0 < angle < pi strictly, so no
        # candidate may sit at (or within float rounding of) sr/2.
        frame = np.ones(400)
        frame[1::2] = -1.0
        candidates = _formant_candidates(frame, 16000, 12)
        assert candidates
        for freq, bandwidth in candidates:
            assert 0.0 < freq < 8000.0 - 1e-6, (freq, bandwidth)
            assert bandwidth > 0.0

    def test_rejected_frames_flow_into_existing_insufficiency_behavior(self):
        # Composition guarantee: a degenerate frame yields no candidates, so
        # extraction's existing per-key insufficiency branch (fewer than two
        # accepted formant frames -> INSUFFICIENT_FORMANTS) is what surfaces.
        # The reachable end-to-end case is a single stable voiced frame, whose
        # per-key issues are asserted here against the fixed extractor.
        frame = np.zeros(400)
        assert _formant_candidates(frame, 16000, 12) == []
        features, issues = acoustic_pack.extract_acoustic_features(
            _vowel(duration_s=0.025), 16000, allow_unaligned=True
        )
        for key in RESONANCE_KEYS:
            assert math.isnan(features[key]), key
            assert any(
                issue.code == "INSUFFICIENT_FORMANTS" and issue.feature == key for issue in issues
            ), key
