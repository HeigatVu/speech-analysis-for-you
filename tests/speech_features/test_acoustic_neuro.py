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

NEW_KEYS = TIMING_KEYS | VOICE_KEYS | SPECTRAL_KEYS


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
