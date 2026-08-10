import math

import pytest

from speech_features import (
    AnnotationLayer,
    DocumentSpeaker,
    DocumentToken,
    DocumentUtterance,
    InvalidConfigError,
    SpeechDocument,
    TargetSpeakerRequiredError,
    list_features,
)
from speech_features.features.motor import ALL_KEYS, extract_motor_features


def _document(tokens, layers=None, *, speakers=("PAR",)):
    layers = layers or {}
    document_speakers = tuple(DocumentSpeaker(id=speaker) for speaker in speakers)
    utterances = tuple(
        DocumentUtterance(
            id=f"u{index}",
            speaker_id=speaker,
            start_s=0.0,
            end_s=max((token.end_s or 0.0 for token in speaker_tokens), default=0.0),
            tokens=tuple(speaker_tokens),
        )
        for index, (speaker, speaker_tokens) in enumerate(tokens.items(), start=1)
    )
    annotations = tuple(
        AnnotationLayer(layer=name, source="reviewed", values=values)
        for name, values in layers.items()
    )
    return SpeechDocument(
        document_id="vi-fixture",
        speakers=document_speakers,
        utterances=utterances,
        annotations=annotations,
    )


def _tokens(*rows):
    return [
        DocumentToken(id=token_id, text=text, start_s=start, end_s=end)
        for token_id, text, start, end in rows
    ]


@pytest.fixture
def vowel_document():
    tokens = _tokens(
        ("i1", "đi", 0.0, 0.2),
        ("a1", "ba", 1.0, 1.2),
        ("u1", "thu", 2.0, 2.2),
    )
    return _document(
        {"PAR": tokens},
        {
            "segment_type": {token.id: "vowel" for token in tokens},
            "segment_label": {"i1": "I", "a1": "a", "u1": "u"},
            "f1_hz": {"i1": 300.0, "a1": 800.0, "u1": 350.0},
            "f2_hz": {"i1": 2400.0, "a1": 1200.0, "u1": 700.0},
        },
    )


def test_vowel_space_and_centralization_from_reviewed_annotations(vowel_document):
    values, issues = extract_motor_features(vowel_document, target_speaker="PAR")

    assert values["artic_vowel_space_area_hz2"] == pytest.approx(395_000.0)
    assert values["artic_vowel_articulation_index"] == pytest.approx(3200 / 2550)
    assert values["artic_formant_centralization_ratio"] == pytest.approx(2550 / 3200)
    assert not {
        issue.feature
        for issue in issues
        if issue.feature
        in {
            "artic_vowel_space_area_hz2",
            "artic_vowel_articulation_index",
            "artic_formant_centralization_ratio",
        }
    }


def test_i_a_u_space_ignores_unrelated_vowel_without_formants(vowel_document):
    tokens = (
        *vowel_document.utterances[0].tokens,
        DocumentToken("e1", "ê", start_s=3.0, end_s=3.2),
    )
    layers = {layer.layer: dict(layer.values) for layer in vowel_document.annotations}
    layers["segment_type"]["e1"] = "vowel"
    layers["segment_label"]["e1"] = "e"
    document = _document({"PAR": tokens}, layers)

    values, issues = extract_motor_features(document, target_speaker="PAR")

    assert values["artic_vowel_space_area_hz2"] == pytest.approx(395_000.0)
    assert values["artic_vowel_articulation_index"] == pytest.approx(3200 / 2550)
    assert values["artic_formant_centralization_ratio"] == pytest.approx(2550 / 3200)
    assert not {
        issue.feature
        for issue in issues
        if issue.feature
        in {
            "artic_vowel_space_area_hz2",
            "artic_vowel_articulation_index",
            "artic_formant_centralization_ratio",
        }
    }


def test_rhythm_formulas_use_reviewed_vowel_and_consonant_durations():
    tokens = _tokens(
        ("v1", "a", 0.0, 0.1),
        ("c1", "m", 0.1, 0.3),
        ("v2", "i", 0.3, 0.5),
        ("c2", "n", 0.5, 0.6),
    )
    document = _document(
        {"PAR": tokens},
        {
            "segment_type": {"v1": "vowel", "c1": "consonant", "v2": "vowel", "c2": "consonant"},
        },
    )

    values, _ = extract_motor_features(document, target_speaker="PAR")

    assert values["rhythm_percent_vocalic"] == pytest.approx(50.0)
    assert values["rhythm_varco_v"] == pytest.approx(100.0 / 3.0)
    assert values["rhythm_varco_c"] == pytest.approx(100.0 / 3.0)
    assert values["rhythm_rpvi_v"] == pytest.approx(0.1)
    assert values["rhythm_rpvi_c"] == pytest.approx(0.1)
    assert values["rhythm_npvi_v"] == pytest.approx(200.0 / 3.0)
    assert values["rhythm_npvi_c"] == pytest.approx(200.0 / 3.0)


def test_rhythm_collapses_adjacent_same_class_tokens_into_intervals():
    tokens = _tokens(
        ("v1", "a", 0.0, 0.1),
        ("v2", "i", 0.1, 0.3),
        ("c1", "m", 0.3, 0.4),
        ("c2", "n", 0.4, 0.7),
        ("v3", "u", 0.7, 1.2),
    )
    document = _document(
        {"PAR": tokens},
        {
            "segment_type": {
                "v1": "vowel",
                "v2": "vowel",
                "c1": "consonant",
                "c2": "consonant",
                "v3": "vowel",
            },
        },
    )

    values, issues = extract_motor_features(document, target_speaker="PAR")

    assert values["rhythm_percent_vocalic"] == pytest.approx(200.0 / 3.0)
    assert values["rhythm_varco_v"] == pytest.approx(25.0)
    assert values["rhythm_varco_c"] == pytest.approx(0.0)
    assert values["rhythm_rpvi_v"] == pytest.approx(0.2)
    assert values["rhythm_npvi_v"] == pytest.approx(50.0)
    assert math.isnan(values["rhythm_rpvi_c"])
    assert math.isnan(values["rhythm_npvi_c"])
    assert [(issue.feature, issue.code) for issue in issues].count(
        ("rhythm_rpvi_c", "MISSING_ANNOTATION")
    ) == 1
    assert [(issue.feature, issue.code) for issue in issues].count(
        ("rhythm_npvi_c", "MISSING_ANNOTATION")
    ) == 1


def test_phone_annotation_means_and_durations_are_type_scoped():
    tokens = _tokens(
        ("s1", "t", 0.0, 0.1),
        ("s2", "k", 0.2, 0.4),
        ("f1", "x", 0.5, 0.8),
    )
    document = _document(
        {"PAR": tokens},
        {
            "segment_type": {"s1": "stop", "s2": "stop", "f1": "fricative"},
            "vot_s": {"s1": 0.02, "s2": 0.04},
            "stop_gap_s": {"s1": 0.01, "s2": 0.03},
            "spectral_moment_1": {"f1": 1000.0},
            "spectral_moment_2": {"f1": 200.0},
            "spectral_moment_3": {"f1": 0.5},
            "spectral_moment_4": {"f1": 3.0},
            "resonance_attenuation_db": {"s1": -2.0, "s2": -4.0, "f1": -6.0},
        },
    )

    values, _ = extract_motor_features(document, target_speaker="PAR")

    assert values["artic_vot_mean_s"] == pytest.approx(0.03)
    assert values["artic_vot_sd_s"] == pytest.approx(0.01)
    assert values["artic_stop_gap_mean_s"] == pytest.approx(0.02)
    assert values["artic_consonant_duration_mean_s"] == pytest.approx(0.2)
    assert values["artic_fricative_m1_mean"] == pytest.approx(1000.0)
    assert values["artic_fricative_m4_mean"] == pytest.approx(3.0)
    assert values["artic_resonance_attenuation_mean_db"] == pytest.approx(-4.0)


def test_ddk_uses_ordered_onsets_only_for_a_ddk_task():
    tokens = _tokens(
        ("d1", "pa", 0.0, 0.1),
        ("d2", "ta", 0.2, 0.32),
        ("d3", "ka", 0.5, 0.61),
        ("d4", "pa", 0.9, 1.0),
    )
    document = _document(
        {"PAR": tokens},
        {
            "segment_label": {"d1": "pa", "d2": "ta", "d3": "ka", "d4": "pa"},
        },
    )

    values, _ = extract_motor_features(document, target_speaker="PAR", task_spec={"task": "ddk"})

    assert values["task_ddk_rate_syllables_s"] == pytest.approx(4.0)
    assert values["task_ddk_inter_onset_mean_s"] == pytest.approx(0.3)
    assert values["task_ddk_inter_onset_median_s"] == pytest.approx(0.3)
    assert values["task_ddk_inter_onset_sd_s"] == pytest.approx(0.0816496580927726)
    assert values["task_ddk_inter_onset_cv"] == pytest.approx(0.2721655269759087)
    assert values["task_ddk_instability_s"] == pytest.approx(0.1)
    assert values["task_ddk_acceleration_syllables_s2"] == pytest.approx(-1325 / 327)
    assert values["task_ddk_decay_ratio"] == pytest.approx(0.5)
    assert values["task_ddk_voiced_interval_mean_s"] == pytest.approx(0.1075)
    assert values["task_ddk_sequential_alternating_ratio"] == pytest.approx(1.0)

    gated, issues = extract_motor_features(
        document, target_speaker="PAR", task_spec={"task": "connected_speech"}
    )
    assert math.isnan(gated["task_ddk_rate_syllables_s"])
    assert [(issue.feature, issue.code) for issue in issues].count(
        ("task_ddk_rate_syllables_s", "MISSING_ANNOTATION")
    ) == 1


def test_respiratory_features_group_tokens_and_require_loudness_calibration():
    tokens = _tokens(
        ("b1a", "xin", 0.0, 0.2),
        ("b1b", "chào", 0.3, 0.8),
        ("b2a", "bạn", 1.0, 1.4),
    )
    document = _document(
        {"PAR": tokens},
        {
            "segment_type": {"b1a": "speech", "b1b": "pause", "b2a": "speech"},
            "breath_group": {"b1a": "1", "b1b": "1", "b2a": "2"},
            "respiration_db": {"b1a": 30.0, "b1b": 30.0, "b2a": 32.0},
            "speech_db": {"b1a": 50.0, "b1b": 52.0, "b2a": 54.0},
        },
    )

    values, _ = extract_motor_features(
        document,
        target_speaker="PAR",
        task_spec={"task": "connected_speech", "calibrated_amplitude": True},
    )

    assert values["resp_breath_group_count"] == 2.0
    assert values["resp_breath_group_mean_s"] == pytest.approx(0.6)
    assert values["resp_breath_group_sd_s"] == pytest.approx(0.2)
    assert values["resp_rate_per_min"] == pytest.approx(2 * 60 / 1.4)
    assert values["resp_pauses_per_breath"] == pytest.approx(0.5)
    assert values["resp_relative_loudness_db"] == pytest.approx(64.0 / 3.0)

    uncalibrated, issues = extract_motor_features(document, target_speaker="PAR")
    assert math.isnan(uncalibrated["resp_relative_loudness_db"])
    assert [(issue.feature, issue.code) for issue in issues].count(
        ("resp_relative_loudness_db", "UNCALIBRATED_AUDIO")
    ) == 1


def test_breath_groups_count_contiguous_label_runs():
    tokens = _tokens(
        ("b1", "xin", 0.0, 0.2),
        ("b2", "à", 0.2, 0.5),
        ("b3", "bạn", 0.5, 1.0),
    )
    document = _document(
        {"PAR": tokens},
        {
            "segment_type": {"b1": "speech", "b2": "pause", "b3": "speech"},
            "breath_group": {"b1": "1", "b2": "2", "b3": "1"},
        },
    )

    values, _ = extract_motor_features(document, target_speaker="PAR")

    assert values["resp_breath_group_count"] == 3.0
    assert values["resp_breath_group_mean_s"] == pytest.approx(1.0 / 3.0)
    assert values["resp_breath_group_sd_s"] == pytest.approx(0.1247219128924647)
    assert values["resp_rate_per_min"] == pytest.approx(180.0)
    assert values["resp_pauses_per_breath"] == pytest.approx(1.0 / 3.0)


def test_sustained_vowel_features_are_task_gated():
    tokens = _tokens(
        ("v1", "a", 0.0, 1.0),
        ("g1", "a", 1.0, 1.2),
        ("s1", "a", 1.2, 2.0),
    )
    document = _document(
        {"PAR": tokens},
        {
            "segment_type": {"v1": "vowel", "g1": "gaping", "s1": "subharmonic"},
            "f0_hz": {"v1": 100.0, "s1": 200.0},
            "power_db": {"v1": 60.0, "s1": 54.0},
        },
    )

    values, _ = extract_motor_features(
        document, target_speaker="PAR", task_spec={"task": "sustained_vowel"}
    )

    assert values["task_max_phonation_time_s"] == pytest.approx(2.0)
    assert values["voice_gaping_interval_rate_per_min"] == pytest.approx(30.0)
    assert values["voice_subharmonic_interval_proportion"] == pytest.approx(0.4)
    assert values["voice_sustained_f0_sd_semitones"] == pytest.approx(6.0)
    assert values["voice_sustained_power_sd_db"] == pytest.approx(3.0)

    gated, issues = extract_motor_features(
        document, target_speaker="PAR", task_spec={"task": "connected_speech"}
    )
    assert math.isnan(gated["task_max_phonation_time_s"])
    assert [(issue.feature, issue.code) for issue in issues].count(
        ("task_max_phonation_time_s", "MISSING_ANNOTATION")
    ) == 1


def test_missing_phone_annotations_are_nan_with_one_issue():
    token = DocumentToken(id="t1", text="ta", start_s=0.0, end_s=0.1)
    document = _document({"PAR": [token]})

    values, issues = extract_motor_features(document, target_speaker="PAR")

    assert math.isnan(values["artic_vot_mean_s"])
    assert [(issue.feature, issue.code) for issue in issues].count(
        ("artic_vot_mean_s", "MISSING_ANNOTATION")
    ) == 1
    assert all(
        sum(issue.feature == key for issue in issues) == 1
        for key, value in values.items()
        if math.isnan(value)
    )


def test_non_finite_numeric_annotations_are_invalid_not_missing():
    token = DocumentToken(id="t1", text="ta", start_s=0.0, end_s=0.1)
    document = _document(
        {"PAR": [token]},
        {
            "segment_type": {"t1": "stop"},
            "vot_s": {"t1": math.inf},
        },
    )

    values, issues = extract_motor_features(document, target_speaker="PAR")

    assert math.isnan(values["artic_vot_mean_s"])
    assert {(issue.feature, issue.code) for issue in issues} >= {
        ("artic_vot_mean_s", "INVALID_TASK_ANNOTATION"),
        ("artic_vot_sd_s", "INVALID_TASK_ANNOTATION"),
    }


@pytest.mark.parametrize("invalid_segment_type", [1, math.inf])
def test_invalid_segment_types_keep_invalid_issue_code_and_cardinality(invalid_segment_type):
    token = DocumentToken(id="t1", text="a", start_s=0.0, end_s=1.0)
    document = _document(
        {"PAR": [token]},
        {
            "segment_type": {"t1": invalid_segment_type},
            "breath_group": {"t1": "1"},
            "f0_hz": {"t1": 100.0},
            "power_db": {"t1": 60.0},
        },
    )

    _, issues = extract_motor_features(
        document, target_speaker="PAR", task_spec={"task": "sustained_vowel"}
    )

    dependent_keys = {
        *(key for key in ALL_KEYS if key.startswith(("artic_", "rhythm_"))),
        "resp_pauses_per_breath",
        "voice_gaping_interval_rate_per_min",
        "voice_subharmonic_interval_proportion",
        "voice_sustained_f0_sd_semitones",
        "voice_sustained_power_sd_db",
    }
    for key in dependent_keys:
        assert [(issue.feature, issue.code) for issue in issues].count(
            (key, "INVALID_TASK_ANNOTATION")
        ) == 1
        assert sum(issue.feature == key for issue in issues) == 1


def test_uncalibrated_loudness_keeps_its_specific_issue_without_a_document():
    values, issues = extract_motor_features(None)

    assert math.isnan(values["resp_relative_loudness_db"])
    assert [(issue.feature, issue.code) for issue in issues].count(
        ("resp_relative_loudness_db", "UNCALIBRATED_AUDIO")
    ) == 1


def test_incomplete_annotation_layer_is_missing_for_each_dependent_key(vowel_document):
    layers = {layer.layer: dict(layer.values) for layer in vowel_document.annotations}
    del layers["f1_hz"]["u1"]
    document = _document(
        {"PAR": list(vowel_document.utterances[0].tokens)},
        layers,
    )

    values, issues = extract_motor_features(document, target_speaker="PAR")

    assert math.isnan(values["artic_vowel_space_area_hz2"])
    assert [(issue.feature, issue.code) for issue in issues].count(
        ("artic_vowel_space_area_hz2", "MISSING_ANNOTATION")
    ) == 1


def test_motor_pack_registers_every_exact_key():
    expected = {
        "artic_vowel_space_area_hz2",
        "artic_vowel_articulation_index",
        "artic_formant_centralization_ratio",
        "artic_vowel_dispersion_mean_hz",
        "artic_vowel_dispersion_sd_hz",
        "artic_f1_within_vowel_sd_hz",
        "artic_f2_within_vowel_sd_hz",
        "artic_formant_transition_slope_mean_hz_s",
        "artic_vot_mean_s",
        "artic_vot_sd_s",
        "artic_stop_gap_mean_s",
        "artic_stop_gap_sd_s",
        "artic_consonant_duration_mean_s",
        "artic_consonant_duration_sd_s",
        "artic_fricative_m1_mean",
        "artic_fricative_m2_mean",
        "artic_fricative_m3_mean",
        "artic_fricative_m4_mean",
        "artic_resonance_attenuation_mean_db",
        "rhythm_percent_vocalic",
        "rhythm_varco_v",
        "rhythm_varco_c",
        "rhythm_rpvi_v",
        "rhythm_npvi_v",
        "rhythm_rpvi_c",
        "rhythm_npvi_c",
        "task_ddk_rate_syllables_s",
        "task_ddk_inter_onset_mean_s",
        "task_ddk_inter_onset_median_s",
        "task_ddk_inter_onset_sd_s",
        "task_ddk_inter_onset_cv",
        "task_ddk_instability_s",
        "task_ddk_acceleration_syllables_s2",
        "task_ddk_decay_ratio",
        "task_ddk_voiced_interval_mean_s",
        "task_ddk_sequential_alternating_ratio",
        "task_max_phonation_time_s",
        "resp_breath_group_count",
        "resp_breath_group_mean_s",
        "resp_breath_group_sd_s",
        "resp_rate_per_min",
        "resp_pauses_per_breath",
        "resp_relative_loudness_db",
        "voice_gaping_interval_rate_per_min",
        "voice_subharmonic_interval_proportion",
        "voice_sustained_f0_sd_semitones",
        "voice_sustained_power_sd_db",
    }

    assert len(expected) == 47
    assert set(ALL_KEYS) == expected
    assert {definition.key for definition in list_features(pack="motor_neuro")} == expected


def test_target_speaker_resolution_matches_existing_pack_contracts():
    token = DocumentToken(id="p1", text="a", start_s=0.0, end_s=0.2)
    examiner = DocumentToken(id="e1", text="ừ", start_s=0.0, end_s=0.1)
    multi = _document({"PAR": [token], "INV": [examiner]}, speakers=("PAR", "INV"))

    with pytest.raises(TargetSpeakerRequiredError):
        extract_motor_features(multi)
    with pytest.raises(InvalidConfigError):
        extract_motor_features(multi, target_speaker="NOPE")

    values, issues = extract_motor_features(multi, target_speaker="PAR")
    assert set(values) == set(ALL_KEYS)
    assert {issue.speaker_id for issue in issues} == {"PAR"}
