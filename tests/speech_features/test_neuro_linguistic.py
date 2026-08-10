"""Behavioral tests for Task 5 clinical linguistic and task measures."""

import math

import pytest

from speech_features.catalog import FeatureDefinition, list_features
from speech_features.document import (
    AnnotationLayer,
    DocumentSpeaker,
    DocumentToken,
    DocumentUtterance,
    SpeechDocument,
)
from speech_features.features.linguistic import (
    extract_adult_neuro_features,
    extract_clinical_linguistic_features,
    extract_lexical_features,
    extract_morphosyntax_features,
    extract_structured_task_features,
)
from speech_features.features.linguistic.definitions import (
    ADULT_NEURO_RECORDING_KEYS,
    CLINICAL_LINGUISTIC_KEYS,
    ERROR_KEYS,
    ERROR_TYPES,
    STRUCTURAL_PSYCHOLINGUISTIC_KEYS,
    TASK_KEYS,
)
from speech_features.schema import InvalidTaskSpecError


def _token(token_id, text, *, start_s=None, end_s=None):
    return DocumentToken(
        id=token_id,
        text=text,
        kind="word",
        language="vie",
        start_s=start_s,
        end_s=end_s,
    )


def _utterance(utterance_id, speaker_id, start_s, end_s, *tokens):
    return DocumentUtterance(
        id=utterance_id,
        speaker_id=speaker_id,
        start_s=start_s,
        end_s=end_s,
        tokens=tokens,
    )


def _document(*utterances, annotations=()):
    return SpeechDocument(
        document_id="d1",
        speakers=(
            DocumentSpeaker(id="PAR", role="participant"),
            DocumentSpeaker(id="INV", role="examiner"),
        ),
        utterances=utterances,
        annotations=annotations,
    )


def _layer(name, values):
    return AnnotationLayer(layer=name, source="reviewed", values=values)


def _issues_for(issues, feature):
    return [issue for issue in issues if issue.feature == feature]


def _assert_nan_issue_pairing(values, issues):
    for key, value in values.items():
        paired = _issues_for(issues, key)
        if math.isnan(value):
            assert len(paired) == 1, key
        else:
            assert paired == [], key


def _annotated_document():
    target = (
        _utterance("u1", "PAR", 0.0, 10.0, _token("t1", "MÈO"), _token("t2", "ngủ")),
        _utterance("u2", "PAR", 20.0, 30.0, _token("t3", "me\u0300o"), _token("t4", "chạy")),
        _utterance("u3", "PAR", 40.0, 60.0, _token("t5", "chó"), _token("t6", "chạy")),
    )
    examiner = _utterance("x1", "INV", 10.0, 12.0, _token("x", "không"))
    target_ids = [f"t{i}" for i in range(1, 7)]
    annotations = (
        _layer(
            "sentence_id", dict(zip(target_ids, ("s1", "s1", "s2", "s2", "s3", "s3"))) | {"x": "sx"}
        ),
        _layer(
            "t_unit_id",
            dict(zip(target_ids, ("tu1", "tu1", "tu2", "tu2", "tu3", "tu3"))) | {"x": "tux"},
        ),
        _layer(
            "clause_id", dict(zip(target_ids, ("c1", "c1", "c2", "c3", "c4", "c4"))) | {"x": "cx"}
        ),
        _layer(
            "clause_type",
            dict(zip(target_ids, ("main", "main", "dependent", "embedded", "main", "main")))
            | {"x": "dependent"},
        ),
        _layer(
            "phrase_type",
            dict(
                zip(
                    target_ids,
                    ("coordinate", "none", "complex_nominal", "verb_phrase", "none", "none"),
                )
            )
            | {"x": "coordinate"},
        ),
        _layer(
            "sentence_status",
            dict(
                zip(
                    target_ids,
                    (
                        "well_formed",
                        "well_formed",
                        "incomplete",
                        "incomplete",
                        "reduced",
                        "reduced",
                    ),
                )
            )
            | {"x": "incomplete"},
        ),
        _layer("yngve_depth", dict(zip(target_ids, (0, 1, 2, 1, 3, 2))) | {"x": 99}),
        _layer("frequency", dict(zip(target_ids, (1, 2, 3, 4, 5, 6))) | {"x": 99}),
        _layer("log_frequency", dict(zip(target_ids, (0, 1, 1, 2, 2, 3))) | {"x": 99}),
        _layer("familiarity", dict(zip(target_ids, (2, 2, 3, 3, 4, 4))) | {"x": 99}),
        _layer("age_of_acquisition", dict(zip(target_ids, (2, 3, 4, 5, 6, 7))) | {"x": 99}),
        _layer("imageability", dict(zip(target_ids, (1, 1, 2, 2, 3, 3))) | {"x": 99}),
        _layer("concreteness", dict(zip(target_ids, (1, 3, 4, 5, 5, 6))) | {"x": 99}),
        _layer(
            "cohesion_type",
            {"t1": "referential", "t2": "temporal", "t3": "causal", "t4": "referential"},
        ),
        _layer("pronoun_reference_correct", {"t1": 1, "t2": 0, "x": 1}),
        _layer("topic_relevant", dict(zip(target_ids, (1, 1, 0, 1, 0, 0))) | {"x": 1}),
        _layer(
            "discourse_role",
            {
                "t1": "marker",
                "t2": "relevant_detail",
                "t3": "relevant_detail",
                "t4": "irrelevant_detail",
                "t5": "microproposition",
                "t6": "macroproposition",
            },
        ),
        _layer("information_unit", {"t1": "cat", "t2": "cat", "t3": "dog", "t4": "hallucination"}),
    )
    return _document(*target, examiner, annotations=annotations)


def test_catalog_accepts_the_registered_semantic_prefix():
    definition = FeatureDefinition(
        key="semantic_test_density",
        pack="adult_neuro",
        level="recording",
        unit="ratio",
        population="adult",
        reference="test",
        domain="semantic",
    )
    assert definition.key == "semantic_test_density"


def test_every_task_5_key_has_an_explicit_clinical_unit():
    count_keys = {
        "morph_sentence_count",
        "morph_t_unit_count",
        "morph_coordinate_phrase_count",
        "morph_complex_nominal_count",
        "morph_verb_phrase_count",
        "morph_embedding_count",
        "discourse_microproposition_count",
        "discourse_macroproposition_count",
        "discourse_information_unit_count",
        "task_fluency_response_count",
        "task_fluency_valid_count",
        "task_fluency_valid_unique",
        "task_fluency_repeats",
        "task_fluency_intrusions",
        "task_fluency_first_half_valid",
        "task_fluency_second_half_valid",
        "task_fluency_production_change",
        "task_fluency_clusters",
        "task_fluency_cluster_size_mean",
        "task_fluency_switches",
    } | {f"disfluency_{error_type}_error_count" for error_type in ERROR_TYPES}
    ratio_keys = {
        "morph_dependent_clause_ratio",
        "morph_well_formed_sentence_ratio",
        "morph_incomplete_sentence_ratio",
        "morph_reduced_sentence_ratio",
        "semantic_idea_density",
        "semantic_proposition_density",
        "discourse_referential_cohesion_ratio",
        "discourse_temporal_cohesion_ratio",
        "discourse_causal_cohesion_ratio",
        "discourse_correct_pronoun_ratio",
        "discourse_local_lexical_coherence",
        "discourse_global_coherence_ratio",
        "discourse_topic_maintenance_ratio",
        "discourse_marker_ratio",
        "discourse_relevant_detail_ratio",
        "discourse_irrelevant_detail_ratio",
        "discourse_content_accuracy_ratio",
        "task_picture_concept_coverage",
        "task_picture_concept_density",
        "task_picture_repeat_ratio",
        "task_picture_entity_coverage",
        "task_picture_action_coverage",
        "task_recall_idea_coverage",
        "task_recall_idea_density",
        "task_recall_repeat_ratio",
    } | {f"disfluency_{error_type}_error_ratio" for error_type in ERROR_TYPES}
    expected = {key: "count" for key in count_keys}
    expected.update({key: "ratio" for key in ratio_keys})
    expected.update(
        {
            "morph_words_per_sentence": "words",
            "morph_words_per_t_unit": "words",
            "morph_words_per_clause": "words",
            "morph_clauses_per_sentence": "clauses/sentence",
            "morph_yngve_depth_mean": "index",
            "morph_yngve_depth_max": "index",
            "lex_frequency_mean": "score",
            "lex_log_frequency_mean": "score",
            "lex_familiarity_mean": "score",
            "lex_age_of_acquisition_mean": "score",
            "lex_imageability_mean": "score",
            "lex_concreteness_mean": "score",
            "discourse_information_efficiency_per_min": "count/min",
            "task_recall_order_score": "score",
            "task_fluency_rate": "count/min",
        }
    )
    task_5_keys = set(CLINICAL_LINGUISTIC_KEYS + TASK_KEYS)
    assert set(expected) == task_5_keys
    registered = {definition.key: definition for definition in list_features(pack="adult_neuro")}
    assert {key: registered[key].unit for key in task_5_keys} == expected


def test_structural_and_psycholinguistic_formulas_use_target_words_only():
    values, issues = extract_clinical_linguistic_features(
        _annotated_document(), target_speaker="PAR"
    )
    expected = {
        "morph_sentence_count": 3,
        "morph_t_unit_count": 3,
        "morph_words_per_sentence": 2,
        "morph_words_per_t_unit": 2,
        "morph_words_per_clause": 1.5,
        "morph_clauses_per_sentence": 4 / 3,
        "morph_coordinate_phrase_count": 1,
        "morph_complex_nominal_count": 1,
        "morph_verb_phrase_count": 1,
        "morph_embedding_count": 1,
        "morph_dependent_clause_ratio": 0.5,
        "morph_well_formed_sentence_ratio": 1 / 3,
        "morph_incomplete_sentence_ratio": 1 / 3,
        "morph_reduced_sentence_ratio": 1 / 3,
        "morph_yngve_depth_mean": 1.5,
        "morph_yngve_depth_max": 3,
        "lex_frequency_mean": 3.5,
        "lex_log_frequency_mean": 1.5,
        "lex_familiarity_mean": 3,
        "lex_age_of_acquisition_mean": 4.5,
        "lex_imageability_mean": 2,
        "lex_concreteness_mean": 4,
    }
    for key, expected_value in expected.items():
        assert values[key] == pytest.approx(expected_value), key
        assert _issues_for(issues, key) == []


def test_clause_ids_are_scoped_by_sentence_for_all_clause_formulas():
    document = _document(
        _utterance("u1", "PAR", 0.0, 1.0, _token("t1", "a"), _token("t2", "b")),
        _utterance("u2", "PAR", 1.0, 2.0, _token("t3", "c"), _token("t4", "d")),
        annotations=(
            _layer("sentence_id", {"t1": "s1", "t2": "s1", "t3": "s2", "t4": "s2"}),
            _layer("clause_id", {"t1": "c1", "t2": "c2", "t3": "c1", "t4": "c2"}),
            _layer(
                "clause_type",
                {"t1": "main", "t2": "dependent", "t3": "embedded", "t4": "main"},
            ),
        ),
    )
    values, issues = extract_clinical_linguistic_features(document, target_speaker="PAR")
    assert values["morph_words_per_clause"] == pytest.approx(1.0)
    assert values["morph_clauses_per_sentence"] == pytest.approx(2.0)
    assert values["morph_dependent_clause_ratio"] == pytest.approx(0.5)
    assert values["morph_embedding_count"] == pytest.approx(1.0)
    for key in (
        "morph_words_per_clause",
        "morph_clauses_per_sentence",
        "morph_dependent_clause_ratio",
        "morph_embedding_count",
    ):
        assert _issues_for(issues, key) == []


def test_inconsistent_type_only_invalidates_dependent_clause_features():
    document = _document(
        _utterance("u1", "PAR", 0.0, 1.0, _token("t1", "a"), _token("t2", "b")),
        annotations=(
            _layer("sentence_id", {"t1": "s1", "t2": "s1"}),
            _layer("clause_id", {"t1": "c1", "t2": "c1"}),
            _layer("clause_type", {"t1": "main", "t2": "dependent"}),
        ),
    )
    values, issues = extract_clinical_linguistic_features(document, target_speaker="PAR")
    assert values["morph_words_per_clause"] == pytest.approx(2.0)
    assert values["morph_clauses_per_sentence"] == pytest.approx(1.0)
    for key in ("morph_dependent_clause_ratio", "morph_embedding_count"):
        assert math.isnan(values[key])
        assert len(_issues_for(issues, key)) == 1
        assert _issues_for(issues, key)[0].code == "MISSING_ANNOTATION"


def test_phrase_counts_use_contiguous_runs_within_each_target_utterance():
    document = _document(
        _utterance(
            "u1",
            "PAR",
            0.0,
            1.0,
            *(_token(f"t{i}", str(i)) for i in range(1, 7)),
        ),
        _utterance(
            "u2",
            "PAR",
            1.0,
            2.0,
            _token("t7", "7"),
            _token("t8", "8"),
            _token("t9", "9"),
        ),
        annotations=(
            _layer(
                "phrase_type",
                {
                    "t1": "coordinate",
                    "t2": "coordinate",
                    "t3": "none",
                    "t4": "verb_phrase",
                    "t5": "verb_phrase",
                    "t6": "coordinate",
                    "t7": "coordinate",
                    "t8": "complex_nominal",
                    "t9": "complex_nominal",
                },
            ),
        ),
    )
    values, issues = extract_clinical_linguistic_features(document, target_speaker="PAR")
    assert values["morph_coordinate_phrase_count"] == pytest.approx(3.0)
    assert values["morph_complex_nominal_count"] == pytest.approx(1.0)
    assert values["morph_verb_phrase_count"] == pytest.approx(1.0)
    for key in (
        "morph_coordinate_phrase_count",
        "morph_complex_nominal_count",
        "morph_verb_phrase_count",
    ):
        assert _issues_for(issues, key) == []


def test_structural_identifier_layers_reject_normalized_placeholders():
    document = _document(
        _utterance("u1", "PAR", 0.0, 1.0, _token("t1", "a")),
        annotations=(
            _layer("sentence_id", {"t1": "NONE"}),
            _layer("t_unit_id", {"t1": "null"}),
            _layer("clause_id", {"t1": "N/A"}),
            _layer("clause_type", {"t1": "main"}),
            _layer("sentence_status", {"t1": "well_formed"}),
        ),
    )
    values, issues = extract_clinical_linguistic_features(document, target_speaker="PAR")
    for key in (
        "morph_sentence_count",
        "morph_t_unit_count",
        "morph_words_per_clause",
        "morph_clauses_per_sentence",
        "morph_dependent_clause_ratio",
        "morph_embedding_count",
    ):
        assert math.isnan(values[key])
        assert len(_issues_for(issues, key)) == 1
        assert _issues_for(issues, key)[0].code == "MISSING_ANNOTATION"


def test_unknown_clause_type_only_invalidates_dependent_clause_features():
    document = _document(
        _utterance("u1", "PAR", 0.0, 1.0, _token("t1", "a")),
        annotations=(
            _layer("sentence_id", {"t1": "s1"}),
            _layer("clause_id", {"t1": "c1"}),
            _layer("clause_type", {"t1": "mystery"}),
        ),
    )
    values, issues = extract_clinical_linguistic_features(document, target_speaker="PAR")
    assert values["morph_words_per_clause"] == pytest.approx(1.0)
    assert values["morph_clauses_per_sentence"] == pytest.approx(1.0)
    for key in ("morph_dependent_clause_ratio", "morph_embedding_count"):
        assert math.isnan(values[key])
        assert len(_issues_for(issues, key)) == 1
        assert _issues_for(issues, key)[0].code == "MISSING_ANNOTATION"


def test_unknown_sentence_status_only_invalidates_status_ratios():
    document = _document(
        _utterance("u1", "PAR", 0.0, 1.0, _token("t1", "a")),
        annotations=(
            _layer("sentence_id", {"t1": "s1"}),
            _layer("sentence_status", {"t1": "mystery"}),
        ),
    )
    values, issues = extract_clinical_linguistic_features(document, target_speaker="PAR")
    assert values["morph_sentence_count"] == pytest.approx(1.0)
    for key in (
        "morph_well_formed_sentence_ratio",
        "morph_incomplete_sentence_ratio",
        "morph_reduced_sentence_ratio",
    ):
        assert math.isnan(values[key])
        assert len(_issues_for(issues, key)) == 1
        assert _issues_for(issues, key)[0].code == "MISSING_ANNOTATION"


def test_psycholinguistic_layer_requires_complete_target_word_coverage():
    document = _annotated_document()
    annotations = tuple(
        _layer(layer.layer, {key: value for key, value in layer.values.items() if key != "t6"})
        if layer.layer == "frequency"
        else layer
        for layer in document.annotations
    )
    values, issues = extract_clinical_linguistic_features(
        SpeechDocument(
            document_id=document.document_id,
            speakers=document.speakers,
            utterances=document.utterances,
            annotations=annotations,
        ),
        target_speaker="PAR",
    )
    assert math.isnan(values["lex_frequency_mean"])
    assert len(_issues_for(issues, "lex_frequency_mean")) == 1
    assert _issues_for(issues, "lex_frequency_mean")[0].code == "MISSING_ANNOTATION"


def test_each_reviewed_error_type_has_a_count_and_target_word_ratio():
    tokens = tuple(_token(f"e{i}", error) for i, error in enumerate(ERROR_TYPES))
    document = _document(
        _utterance("u1", "PAR", 0.0, 13.0, *tokens),
        _utterance("x1", "INV", 13.0, 14.0, _token("x", "semantic")),
        annotations=(
            _layer(
                "error_type",
                {token.id: error for token, error in zip(tokens, ERROR_TYPES)} | {"x": "semantic"},
            ),
        ),
    )
    values, issues = extract_clinical_linguistic_features(document, target_speaker="PAR")
    assert len(ERROR_KEYS) == 26
    for error_type in ERROR_TYPES:
        count_key = f"disfluency_{error_type}_error_count"
        ratio_key = f"disfluency_{error_type}_error_ratio"
        assert values[count_key] == 1
        assert values[ratio_key] == pytest.approx(1 / len(ERROR_TYPES))
        assert _issues_for(issues, count_key) == []
        assert _issues_for(issues, ratio_key) == []


def test_reviewed_discourse_formulas_and_vietnamese_local_coherence():
    spec = {
        "version": 1,
        "task": "picture_desc_1",
        "concept_aliases": {"cat": ["mèo"], "dog": ["chó"]},
        "entity_groups": {},
        "action_groups": {},
    }
    values, issues = extract_clinical_linguistic_features(
        _annotated_document(), target_speaker="PAR", task_spec=spec
    )
    expected = {
        "discourse_referential_cohesion_ratio": 0.5,
        "discourse_temporal_cohesion_ratio": 0.25,
        "discourse_causal_cohesion_ratio": 0.25,
        "discourse_correct_pronoun_ratio": 0.5,
        "discourse_local_lexical_coherence": 1 / 3,
        "discourse_global_coherence_ratio": 0.5,
        "discourse_topic_maintenance_ratio": 2 / 3,
        "discourse_marker_ratio": 1 / 6,
        "discourse_relevant_detail_ratio": 2 / 3,
        "discourse_irrelevant_detail_ratio": 1 / 3,
        "discourse_microproposition_count": 1,
        "discourse_macroproposition_count": 1,
        "discourse_information_unit_count": 3,
        "discourse_content_accuracy_ratio": 2 / 3,
        "discourse_information_efficiency_per_min": 3,
        "semantic_idea_density": 0.5,
        "semantic_proposition_density": 1 / 3,
    }
    for key, expected_value in expected.items():
        assert values[key] == pytest.approx(expected_value), key
        assert _issues_for(issues, key) == []


PICTURE_SPEC = {
    "version": 1,
    "task": "picture_desc_1",
    "concept_aliases": {"cat": ["mèo"], "dog": ["chó"], "run": ["chạy"]},
    "entity_groups": {"cat": ["mèo"], "dog": ["chó"]},
    "action_groups": {"motion": ["chạy"]},
}


def test_picture_task_matches_existing_validated_formula_and_excludes_examiner():
    document = _document(
        _utterance(
            "u1", "PAR", 0.0, 30.0, _token("t1", "MÈO"), _token("t2", "mèo"), _token("t3", "chó")
        ),
        _utterance("x1", "INV", 30.0, 60.0, _token("x", "chạy")),
    )
    values, issues = extract_structured_task_features(document, PICTURE_SPEC, target_speaker="PAR")
    assert values["task_picture_concept_coverage"] == pytest.approx(2 / 3)
    assert values["task_picture_concept_density"] == pytest.approx(2 / 3)
    assert values["task_picture_repeat_ratio"] == pytest.approx(1 / 3)
    assert values["task_picture_entity_coverage"] == pytest.approx(1.0)
    assert values["task_picture_action_coverage"] == pytest.approx(0.0)
    _assert_nan_issue_pairing(values, issues)


def test_recall_task_matches_existing_coverage_repeat_and_lcs_formulas():
    spec = {
        "version": 1,
        "task": "immediate_recall",
        "idea_aliases": {"camera": ["máy ảnh"], "umbrella": ["ô"], "frog": ["ếch"]},
    }
    document = _document(
        _utterance(
            "u1",
            "PAR",
            0.0,
            60.0,
            _token("t1", "ếch"),
            _token("t2", "ô"),
            _token("t3", "máy ảnh"),
            _token("t4", "ếch"),
        )
    )
    values, issues = extract_structured_task_features(document, spec, target_speaker="PAR")
    assert values["task_recall_idea_coverage"] == pytest.approx(1.0)
    assert values["task_recall_idea_density"] == pytest.approx(3 / 4)
    assert values["task_recall_repeat_ratio"] == pytest.approx(1 / 4)
    assert values["task_recall_order_score"] == pytest.approx(1 / 3)
    _assert_nan_issue_pairing(values, issues)


def test_phonemic_fluency_uses_target_timestamps_for_halves_and_rate():
    spec = {
        "version": 1,
        "task": "phonemic_fluency",
        "initials": ["c"],
        "exclusions": [],
    }
    document = _document(
        _utterance("u1", "PAR", 0.0, 1.0, _token("t1", "cá")),
        _utterance("u2", "PAR", 10.0, 11.0, _token("t2", "cá")),
        _utterance("u3", "PAR", 40.0, 41.0, _token("t3", "bàn")),
        _utterance("u4", "PAR", 59.0, 60.0, _token("t4", "cây")),
        _utterance("x1", "INV", 20.0, 21.0, _token("x", "cò")),
    )
    values, issues = extract_structured_task_features(document, spec, target_speaker="PAR")
    expected = {
        "task_fluency_response_count": 4,
        "task_fluency_valid_count": 3,
        "task_fluency_valid_unique": 2,
        "task_fluency_repeats": 1,
        "task_fluency_intrusions": 1,
        "task_fluency_first_half_valid": 2,
        "task_fluency_second_half_valid": 1,
        "task_fluency_production_change": -1,
        "task_fluency_rate": 2,
    }
    for key, expected_value in expected.items():
        assert values[key] == pytest.approx(expected_value), key
    _assert_nan_issue_pairing(values, issues)


def test_semantic_fluency_clusters_switches_and_mean_cluster_size():
    spec = {
        "version": 1,
        "task": "semantic_fluency",
        "item_aliases": {"cat": ["mèo"], "dog": ["chó"], "apple": ["táo"]},
        "subcategories": {"cat": "animal", "dog": "animal", "apple": "fruit"},
    }
    document = _document(
        _utterance("u1", "PAR", 0.0, 1.0, _token("t1", "mèo")),
        _utterance("u2", "PAR", 10.0, 11.0, _token("t2", "chó")),
        _utterance("u3", "PAR", 40.0, 41.0, _token("t3", "táo")),
        _utterance("u4", "PAR", 59.0, 60.0, _token("t4", "mèo")),
    )
    values, issues = extract_structured_task_features(document, spec, target_speaker="PAR")
    expected = {
        "task_fluency_response_count": 4,
        "task_fluency_valid_count": 4,
        "task_fluency_valid_unique": 3,
        "task_fluency_repeats": 1,
        "task_fluency_intrusions": 0,
        "task_fluency_first_half_valid": 2,
        "task_fluency_second_half_valid": 2,
        "task_fluency_production_change": 0,
        "task_fluency_rate": 3,
        "task_fluency_clusters": 3,
        "task_fluency_cluster_size_mean": 4 / 3,
        "task_fluency_switches": 2,
    }
    for key, expected_value in expected.items():
        assert values[key] == pytest.approx(expected_value), key
    _assert_nan_issue_pairing(values, issues)


def test_structured_task_extractor_preserves_version_1_spec_validation():
    with pytest.raises(InvalidTaskSpecError):
        extract_structured_task_features(
            _annotated_document(),
            {"version": 1, "task": "picture_desc_1", "concept_aliases": {}},
            target_speaker="PAR",
        )


def test_clinical_extractor_validates_task_spec_without_information_layer():
    document = _document(
        _utterance("u1", "PAR", 0.0, 1.0, _token("t1", "mèo")),
    )
    with pytest.raises(InvalidTaskSpecError):
        extract_clinical_linguistic_features(
            document,
            target_speaker="PAR",
            task_spec={
                "version": 2,
                "task": "picture_desc_1",
                "concept_aliases": {},
                "entity_groups": {},
                "action_groups": {},
            },
        )


def test_adult_neuro_adds_every_registered_key_without_changing_existing_outputs():
    document = _annotated_document()
    lexical, lexical_issues = extract_lexical_features(document, target_speaker="PAR")
    morph, rows, morph_issues = extract_morphosyntax_features(document, target_speaker="PAR")
    values, got_rows, issues = extract_adult_neuro_features(document, target_speaker="PAR")

    assert set(values) == set(ADULT_NEURO_RECORDING_KEYS)
    for key, expected in {**lexical, **morph}.items():
        if math.isnan(expected):
            assert math.isnan(values[key])
        else:
            assert values[key] == expected
    assert got_rows == rows
    assert list(issues[: len(lexical_issues) + len(morph_issues)]) == list(
        lexical_issues + morph_issues
    )
    assert set(STRUCTURAL_PSYCHOLINGUISTIC_KEYS) <= set(values)
    assert set(CLINICAL_LINGUISTIC_KEYS) <= set(values)
    assert set(TASK_KEYS) <= set(values)
    for key in TASK_KEYS:
        assert math.isnan(values[key])
        paired = _issues_for(issues, key)
        assert len(paired) == 1
        assert paired[0].code == "MISSING_ANNOTATION"
    _assert_nan_issue_pairing(values, issues)
