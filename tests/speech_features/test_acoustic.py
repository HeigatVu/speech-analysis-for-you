"""Behavioral tests for math-first acoustic feature extraction (Task 2).

Accepts mono PCM float arrays only; no WAV I/O, no labels, no ASR. Covers
sample-rate/config checks, frame/hop analysis, energy VAD, VAD-derived
silence/pause summaries, normalized-autocorrelation F0 with explicit
unvoiced handling, voiced F0 and frame-energy statistics, spectral
centroid/spread/flatness, and quality flags (too-short, no-voice, invalid).
"""

import math

import numpy as np

import speech_features.acoustic as acoustic
from speech_features.schema import ExtractionConfig


def _tone(freq, duration_s, sample_rate=16000, amplitude=0.5):
    t = np.arange(int(round(sample_rate * duration_s))) / sample_rate
    return amplitude * np.sin(2 * math.pi * freq * t)


class TestExtractAcoustic:
    def test_returns_plain_numeric_features_and_flags(self):
        result = acoustic.extract_acoustic(_tone(145, 1.0), 16000)
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
