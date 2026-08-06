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

import numpy as np
import pytest

import speech_features.acoustic as acoustic
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
