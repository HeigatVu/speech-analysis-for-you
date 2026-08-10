"""Expanded shared neuro acoustic feature behavior (Task 2)."""

import importlib
import math

import numpy as np
import pytest

from speech_features import list_features
from speech_features.document import (
    DocumentSpeaker,
    DocumentToken,
    DocumentUtterance,
    SpeechDocument,
)
from speech_features.features import acoustic as acoustic_pack
from speech_features.features.acoustic import timing as acoustic_timing
from speech_features.schema import ExtractionConfig


TIMING_KEYS = {
    "time_pause_total_s",
    "time_pause_median_s",
    "time_pause_iqr_s",
    "time_pause_cv",
    "time_pause_proportion",
    "time_speech_segment_count",
    "time_speech_segment_rate_per_min",
    "time_speech_segment_median_s",
    "time_speech_segment_iqr_s",
    "time_speech_segment_cv",
    "time_speech_segment_max_s",
    "time_between_utterance_pause_proportion",
    "time_max_local_speech_rate_wpm",
    "time_timing_event_rate_per_min",
    "time_timing_event_entropy",
    "time_timing_acceleration_per_min2",
}

VOICE_KEYS = {
    "voice_break_count",
    "voice_break_rate_per_min",
    "voice_break_proportion",
    "voice_f0_range_semitones",
    "voice_f0_mad_semitones",
    "voice_intensity_range_db",
    "voice_intensity_cv",
    "voice_nhr_mean_db",
}

MFCC_KEYS = {
    f"spectral_mfcc_{coefficient}_{stat}"
    for coefficient in range(1, 14)
    for stat in ("mean", "sd", "skewness", "kurtosis")
}

SPECTRAL_KEYS = {
    "spectral_energy_mean_db",
    "spectral_energy_sd_db",
    "spectral_skewness_mean",
    "spectral_skewness_sd",
    "spectral_kurtosis_mean",
    "spectral_kurtosis_sd",
    "spectral_low_high_energy_ratio_db",
    *MFCC_KEYS,
}

NONLINEAR_KEYS = {
    "voice_jitter_rap",
    "voice_jitter_ppq5",
    "voice_jitter_ddp",
    "voice_shimmer_apq3",
    "voice_shimmer_apq5",
    "voice_shimmer_apq11",
    "voice_shimmer_dda",
    "voice_pitch_period_entropy",
    "voice_rpde",
    "voice_dfa",
    "voice_correlation_dimension",
}

NEW_KEYS = TIMING_KEYS | VOICE_KEYS | SPECTRAL_KEYS | NONLINEAR_KEYS

TIMING_DISORDERS = {
    "ad",
    "als",
    "ataxia",
    "cbs",
    "dlb",
    "ftd",
    "hd",
    "mci",
    "mnd",
    "ms",
    "msa",
    "pd",
    "pdd",
    "ppa",
    "psp",
}
PROSODY_DISORDERS = TIMING_DISORDERS - {"ataxia"}
PHONATION_DISORDERS = TIMING_DISORDERS
SPECTRAL_DISORDERS = {"ad", "als", "ataxia", "hd", "mci", "ms", "msa", "pd", "psp"}


def _tone(freq, duration_s, sample_rate=16000, amplitude=0.5):
    time = np.arange(int(round(sample_rate * duration_s))) / sample_rate
    return amplitude * np.sin(2 * math.pi * freq * time)


def _tokens(prefix, count):
    return tuple(DocumentToken(id=f"{prefix}-{i}", text="xa", kind="word") for i in range(count))


@pytest.fixture
def aligned_document():
    """Three speech segments with 0.3 s and 0.7 s intervening pauses."""
    return SpeechDocument(
        document_id="d1",
        speakers=(DocumentSpeaker(id="PAR", role="participant"),),
        utterances=(
            DocumentUtterance(
                id="u1", speaker_id="PAR", start_s=0.0, end_s=1.0, tokens=_tokens("a", 2)
            ),
            DocumentUtterance(
                id="u2", speaker_id="PAR", start_s=1.3, end_s=2.3, tokens=_tokens("b", 3)
            ),
            DocumentUtterance(
                id="u3", speaker_id="PAR", start_s=3.0, end_s=5.0, tokens=_tokens("c", 5)
            ),
        ),
    )


def test_pause_companion_summaries_are_hand_calculated(aligned_document):
    values, _ = acoustic_pack.extract_acoustic_features(
        np.ones(80000) * 0.1,
        16000,
        document=aligned_document,
        target_speaker="PAR",
    )
    assert values["time_pause_total_s"] == pytest.approx(1.0)
    assert values["time_pause_median_s"] == pytest.approx(0.5)
    assert values["time_pause_iqr_s"] == pytest.approx(0.2)
    assert values["time_pause_cv"] == pytest.approx(0.4)
    assert values["time_pause_proportion"] == pytest.approx(0.2)
    assert values["time_between_utterance_pause_proportion"] == pytest.approx(1.0)


def test_speech_segment_summaries_are_hand_calculated(aligned_document):
    values, _ = acoustic_pack.extract_acoustic_features(
        np.ones(80000) * 0.1,
        16000,
        document=aligned_document,
        target_speaker="PAR",
    )
    assert values["time_speech_segment_count"] == 3.0
    assert values["time_speech_segment_rate_per_min"] == pytest.approx(36.0)
    assert values["time_speech_segment_median_s"] == pytest.approx(1.0)
    assert values["time_speech_segment_iqr_s"] == pytest.approx(0.5)
    assert values["time_speech_segment_cv"] == pytest.approx(math.sqrt(0.5) / 2.0)
    assert values["time_speech_segment_max_s"] == pytest.approx(2.0)
    assert values["time_max_local_speech_rate_wpm"] == pytest.approx(150.0)
    assert math.isfinite(values["time_timing_event_rate_per_min"])
    assert 0.0 <= values["time_timing_event_entropy"] <= 1.0
    assert math.isfinite(values["time_timing_acceleration_per_min2"])


def test_timing_event_summaries_coalesce_touching_boundary_spans():
    spans = [
        ("voiced", 0.0, 0.4),
        ("unvoiced", 0.4, 0.6),
        ("unvoiced", 0.6, 0.8),
        ("voiced", 0.8, 1.8),
        ("unvoiced", 1.8, 2.0),
    ]
    rate, entropy, acceleration = acoustic_timing._event_summaries(
        spans, duration_s=2.0, pause_threshold_s=0.25
    )
    assert rate == pytest.approx(120.0)
    assert entropy == pytest.approx(
        -(0.5 * math.log(0.5) + 2 * 0.25 * math.log(0.25)) / math.log(3.0)
    )
    assert acceleration == pytest.approx(-3600.0)


def test_timing_events_classify_touching_unvoiced_fragments_by_merged_duration():
    spans = [
        ("voiced", 0.0, 0.1),
        ("unvoiced", 0.1, 0.25),
        ("unvoiced", 0.25, 0.4),
        ("voiced", 0.4, 0.7),
        ("unvoiced", 0.7, 0.8),
        ("voiced", 0.8, 1.0),
    ]
    rate, entropy, acceleration = acoustic_timing._event_summaries(
        spans, duration_s=1.0, pause_threshold_s=0.25
    )
    assert rate == pytest.approx(300.0)
    assert entropy == pytest.approx(
        -(0.6 * math.log(0.6) + 2 * 0.2 * math.log(0.2)) / math.log(3.0)
    )
    assert acceleration == pytest.approx(-7200.0)


def test_timing_events_merge_unvoiced_interval_tail_with_touching_gap():
    interval_tail = acoustic_timing._timing_events(
        np.array([True, False]),
        hop_s=0.1,
        start_s=0.0,
        end_s=0.2,
    )
    spans = [
        *interval_tail,
        ("unvoiced", 0.2, 0.4),
        ("voiced", 0.4, 0.7),
        ("unvoiced", 0.7, 0.8),
        ("voiced", 0.8, 1.0),
    ]
    rate, entropy, acceleration = acoustic_timing._event_summaries(
        spans, duration_s=1.0, pause_threshold_s=0.25
    )
    assert rate == pytest.approx(300.0)
    assert entropy == pytest.approx(
        -(0.6 * math.log(0.6) + 2 * 0.2 * math.log(0.2)) / math.log(3.0)
    )
    assert acceleration == pytest.approx(-7200.0)


def test_mfcc_statistics_have_fixed_52_key_schema_and_evidence_metadata():
    definitions = {definition.key: definition for definition in list_features(pack="acoustic")}
    assert NEW_KEYS <= definitions.keys()
    assert len(MFCC_KEYS) == 52
    for key in TIMING_KEYS | {
        "voice_break_count",
        "voice_break_rate_per_min",
        "voice_break_proportion",
        "voice_nhr_mean_db",
    }:
        assert definitions[key].language_scope == "language_independent"
    for key in SPECTRAL_KEYS | {
        "voice_f0_range_semitones",
        "voice_f0_mad_semitones",
        "voice_intensity_range_db",
        "voice_intensity_cv",
    }:
        assert definitions[key].language_scope == "language_sensitive"
    for key in NEW_KEYS:
        assert definitions[key].domain != "audio_quality"
        assert definitions[key].disorders
    for key in TIMING_KEYS:
        assert set(definitions[key].disorders) == TIMING_DISORDERS
    for key in {
        "voice_f0_range_semitones",
        "voice_f0_mad_semitones",
        "voice_intensity_range_db",
        "voice_intensity_cv",
    }:
        assert set(definitions[key].disorders) == PROSODY_DISORDERS
    for key in {
        "voice_break_count",
        "voice_break_rate_per_min",
        "voice_break_proportion",
        "voice_nhr_mean_db",
    }:
        assert set(definitions[key].disorders) == PHONATION_DISORDERS
    for key in SPECTRAL_KEYS:
        assert set(definitions[key].disorders) == SPECTRAL_DISORDERS


def test_voice_break_and_companion_prosody_features():
    audio = np.concatenate([_tone(200, 0.35), np.zeros(4800), _tone(200, 0.35)])
    values, _ = acoustic_pack.extract_acoustic_features(audio, 16000, allow_unaligned=True)
    assert values["voice_break_count"] == 1.0
    assert values["voice_break_rate_per_min"] == pytest.approx(60.0)
    assert values["voice_break_proportion"] == pytest.approx(0.3, abs=0.04)
    assert values["voice_f0_range_semitones"] < 0.1
    assert values["voice_f0_mad_semitones"] < 0.1
    assert values["voice_intensity_range_db"] < 0.2
    assert 0.0 <= values["voice_intensity_cv"] < 0.02
    assert values["voice_nhr_mean_db"] == pytest.approx(-values["voice_hnr_mean_db"])


def test_mfcc_and_distribution_helpers_follow_the_documented_math():
    advanced = importlib.import_module("speech_features.features.acoustic.advanced")
    frames = np.tile(np.hamming(400), (5, 1))
    mfcc = advanced.mfcc_frames(frames, 16000)
    assert mfcc.shape == (5, 13)
    assert np.all(np.isfinite(mfcc))

    mean, sd, skewness, kurtosis = advanced.distribution_stats(np.array([1.0, 2.0, 3.0, 4.0]))
    assert mean == pytest.approx(2.5)
    assert sd == pytest.approx(math.sqrt(1.25))
    assert skewness == pytest.approx(0.0, abs=1e-12)
    assert kurtosis == pytest.approx(-1.2)

    _, _, scaled_skewness, scaled_kurtosis = advanced.distribution_stats(
        np.array([1.0e-16, 2.0e-16, 3.0e-16, 4.0e-16])
    )
    assert scaled_skewness == pytest.approx(0.0, abs=1e-12)
    assert scaled_kurtosis == pytest.approx(-1.2)


def test_mfcc_matches_independent_impulse_reference():
    advanced = importlib.import_module("speech_features.features.acoustic.advanced")
    frame = np.zeros((1, 400))
    frame[0, 0] = 1.0
    expected = np.array(
        [
            -3.553535330072,
            -0.009933319359746,
            -0.4017414460036,
            -0.009832238488196,
            -0.1487526118346,
            -0.008951638292058,
            -0.07943676061544,
            -0.01038120950582,
            -0.05000846643649,
            -0.006230073484105,
            -0.03029032784004,
            -0.001810014016426,
            -0.0220753411054,
        ]
    )
    assert advanced.mfcc_frames(frame, 16000)[0] == pytest.approx(expected, abs=1e-10)


def test_period_perturbation_formulas_match_hand_calculation():
    advanced = importlib.import_module("speech_features.features.acoustic.advanced")
    periods = np.array([0.0100, 0.0101, 0.0099, 0.0102, 0.0098])
    rap_residuals = [abs(periods[i] - periods[i - 1 : i + 2].mean()) for i in range(1, 4)]
    ppq5_residual = abs(periods[2] - periods.mean())
    assert advanced.jitter_rap(periods) == pytest.approx(np.mean(rap_residuals) / periods.mean())
    assert advanced.jitter_ppq5(periods) == pytest.approx(ppq5_residual / periods.mean())


def test_amplitude_perturbation_formulas_match_hand_calculation():
    advanced = importlib.import_module("speech_features.features.acoustic.advanced")
    amplitudes = np.array([1.00, 1.02, 0.99, 1.03, 0.98, 1.04, 0.97, 1.05, 0.96, 1.06, 0.95])
    for window in (3, 5, 11):
        half = window // 2
        residuals = [
            abs(amplitudes[i] - amplitudes[i - half : i + half + 1].mean())
            for i in range(half, amplitudes.size - half)
        ]
        assert advanced.shimmer_apq(amplitudes, window) == pytest.approx(
            np.mean(residuals) / amplitudes.mean()
        )


def test_perturbation_windows_do_not_cross_disjoint_regions(monkeypatch):
    phonation = importlib.import_module("speech_features.features.acoustic.phonation")
    sample_rate = 16000
    region_samples = 2640  # exactly 15 frames at the default frame/hop sizes
    gap_samples = 1600
    audio = np.concatenate(
        (
            np.full(region_samples, 0.2),
            np.zeros(gap_samples),
            np.full(region_samples, 0.8),
        )
    )
    intervals = [
        (0.0, region_samples / sample_rate),
        ((region_samples + gap_samples) / sample_rate, audio.size / sample_rate),
    ]

    def fixed_region_pitch(frames, *_args):
        frequency = 100.0 if np.mean(np.abs(frames)) < 0.25 else 200.0
        return np.full(frames.shape[0], frequency), np.full(frames.shape[0], 0.8)

    monkeypatch.setattr(phonation, "_f0_per_frame", fixed_region_pitch)
    values = phonation.phonation_features(
        audio,
        sample_rate,
        intervals=intervals,
        config=ExtractionConfig(),
        recording_id="r",
        speaker_id="PAR",
        issues=[],
    )
    for key in (
        "voice_jitter_rap",
        "voice_jitter_ppq5",
        "voice_jitter_ddp",
        "voice_shimmer_apq3",
        "voice_shimmer_apq5",
        "voice_shimmer_apq11",
        "voice_shimmer_dda",
    ):
        assert values[key] == pytest.approx(0.0, abs=1e-15), key


def test_constant_period_signal_has_zero_pitch_period_entropy():
    advanced = importlib.import_module("speech_features.features.acoustic.advanced")
    assert advanced.pitch_period_entropy(np.ones(100)) == pytest.approx(0.0)


def test_rpde_matches_hand_calculated_recurrence_lag_entropy():
    advanced = importlib.import_module("speech_features.features.acoustic.advanced")
    periods = np.tile([1.0, 2.0], 32)
    recurrence_counts = np.arange(62, 31, -2, dtype=float)
    probabilities = recurrence_counts / recurrence_counts.sum()
    expected = -np.sum(probabilities * np.log(probabilities)) / np.log(32.0)
    assert advanced.recurrence_period_density_entropy(periods) == pytest.approx(expected)


def test_nonlinear_helpers_enforce_minimum_and_are_finite_at_boundary():
    advanced = importlib.import_module("speech_features.features.acoustic.advanced")
    too_short = 0.01 + np.arange(63) * 1e-7
    assert math.isnan(advanced.pitch_period_entropy(too_short))
    assert math.isnan(advanced.recurrence_period_density_entropy(too_short))
    assert math.isnan(advanced.detrended_fluctuation_analysis(too_short))
    assert math.isnan(advanced.correlation_dimension(too_short))

    periods = 0.01 + np.random.default_rng(4).normal(0.0, 1e-4, 64)
    for value in (
        advanced.pitch_period_entropy(periods),
        advanced.recurrence_period_density_entropy(periods),
        advanced.detrended_fluctuation_analysis(periods),
        advanced.correlation_dimension(periods),
    ):
        assert math.isfinite(value)


def test_nonlinear_minimum_one_with_two_periods_is_unavailable_not_exception(monkeypatch):
    advanced = importlib.import_module("speech_features.features.acoustic.advanced")
    phonation = importlib.import_module("speech_features.features.acoustic.phonation")
    periods = np.array([0.01, 0.011])
    assert math.isnan(advanced.correlation_dimension(periods, minimum=1))

    def two_periods(frames, *_args):
        assert frames.shape[0] == 2
        return 1.0 / periods, np.full(2, 0.8)

    monkeypatch.setattr(phonation, "_f0_per_frame", two_periods)
    issues = []
    with np.errstate(all="raise"):
        values = phonation.phonation_features(
            np.full(560, 0.5),
            16000,
            intervals=None,
            config=ExtractionConfig(nonlinear_min_periods=1),
            recording_id="r",
            speaker_id="PAR",
            issues=issues,
        )
    assert math.isnan(values["voice_correlation_dimension"])
    matching = [issue for issue in issues if issue.feature == "voice_correlation_dimension"]
    assert len(matching) == 1
    assert matching[0].code == "INSUFFICIENT_VOICING"


def test_nonlinear_correlation_dimension_large_input_skips_pairwise_allocation(monkeypatch):
    advanced = importlib.import_module("speech_features.features.acoustic.advanced")

    def unexpected_pdist(_embedding):
        pytest.fail("pdist must not run above the exact-computation ceiling")

    monkeypatch.setattr(advanced, "pdist", unexpected_pdist)
    periods = np.linspace(0.009, 0.011, 60_000)
    assert math.isnan(advanced.correlation_dimension(periods, minimum=1))


def test_nonlinear_calibration_defaults_and_rejects_non_positive_values():
    config = ExtractionConfig()
    assert config.nonlinear_min_periods == 64
    assert config.recurrence_radius_sd == pytest.approx(0.1)
    assert config.entropy_bins == 32
    for field in ("nonlinear_min_periods", "recurrence_radius_sd", "entropy_bins"):
        for value in (0, -1):
            with pytest.raises(ValueError, match=field):
                ExtractionConfig(**{field: value})


def test_nonlinear_keys_have_one_stable_issue_when_voicing_is_insufficient():
    values, issues = acoustic_pack.extract_acoustic_features(
        np.zeros(16000), 16000, allow_unaligned=True
    )
    assert NONLINEAR_KEYS <= values.keys()
    for key in NONLINEAR_KEYS:
        assert math.isnan(values[key])
        matching = [issue for issue in issues if issue.feature == key]
        assert len(matching) == 1, key
        assert matching[0].code == "INSUFFICIENT_VOICING"


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1.0], (1.0, 0.0, math.nan, math.nan)),
        ([1.0, 3.0], (2.0, 1.0, math.nan, math.nan)),
        ([1.0, 2.0, 3.0], (2.0, math.sqrt(2.0 / 3.0), 0.0, math.nan)),
        ([1.0, 2.0, 3.0, 4.0], (2.5, math.sqrt(1.25), 0.0, -1.2)),
    ],
)
def test_distribution_stats_observation_boundaries(values, expected):
    advanced = importlib.import_module("speech_features.features.acoustic.advanced")
    actual = advanced.distribution_stats(np.asarray(values))
    for observed, wanted in zip(actual, expected):
        if math.isnan(wanted):
            assert math.isnan(observed)
        else:
            assert observed == pytest.approx(wanted)


def test_advanced_spectrum_is_finite_on_nonstationary_noise():
    rng = np.random.default_rng(9)
    audio = rng.normal(0.0, np.linspace(0.1, 0.4, 32000))
    values, _ = acoustic_pack.extract_acoustic_features(audio, 16000, allow_unaligned=True)
    assert SPECTRAL_KEYS <= values.keys()
    for key in SPECTRAL_KEYS:
        assert math.isfinite(values[key]), key


def test_every_registered_key_is_returned_and_every_nan_has_exactly_one_stable_issue():
    values, issues = acoustic_pack.extract_acoustic_features(
        np.zeros(16000), 16000, allow_unaligned=True
    )
    assert set(values) == {definition.key for definition in list_features(pack="acoustic")}
    for key, value in values.items():
        matching = [issue for issue in issues if issue.feature == key]
        if math.isnan(value):
            assert len(matching) == 1, key
            assert matching[0].code
        else:
            assert not matching, key
