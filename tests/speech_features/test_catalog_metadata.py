"""Tests for the evidence-aware feature catalog metadata (Task 1).

Covers the additive immutable metadata fields (``domain``, ``language_scope``,
``tasks``, ``disorders``, ``evidence_level``), their normalization and
validation against the frozen allowed sets, the new ``motor_neuro`` and
``standardized_acoustic`` packs, single-value ``list_features`` filters with
unknown-value rejection, and the evidence record contract in
:mod:`speech_features.evidence`.
"""

import dataclasses

import pytest

import speech_features
from speech_features import FeatureDefinition, list_features
from speech_features import catalog
from speech_features.catalog import (
    DISORDERS,
    DOMAINS,
    EVIDENCE_LEVELS,
    KEY_PREFIXES,
    LANGUAGE_SCOPES,
    TASK_IDS,
    register_feature,
)
from speech_features.evidence import EvidenceRecord, validate_evidence

RESPIRATION_REFERENCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC9950294/"


@pytest.fixture(autouse=True)
def _restore_catalog():
    snapshot = list(catalog._FEATURES)
    catalog._FEATURES[:] = [
        definition for definition in snapshot if definition.pack != "motor_neuro"
    ]
    yield
    catalog._FEATURES[:] = snapshot


def valid_definition_kwargs() -> dict:
    return dict(
        key="resp_breath_group_count",
        pack="motor_neuro",
        level="recording",
        unit="count",
        population="adult",
        reference=RESPIRATION_REFERENCE,
        domain="respiration",
        language_scope="language_independent",
        tasks=["connected_speech"],
        disorders=["als", "pd"],
        evidence_level="systematic_review",
    )


def test_feature_metadata_is_normalized_immutable_and_filterable():
    definition = FeatureDefinition(
        key="resp_breath_group_count",
        pack="motor_neuro",
        level="recording",
        unit="count",
        population="adult",
        reference=RESPIRATION_REFERENCE,
        domain="respiration",
        language_scope="language_independent",
        tasks=["connected_speech"],
        disorders=["als", "pd"],
        evidence_level="systematic_review",
    )
    assert definition.tasks == ("connected_speech",)
    assert definition.disorders == ("als", "pd")
    register_feature(definition)
    assert definition in list_features(domain="respiration", disorder="als")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("domain", "unknown"),
        ("language_scope", "neutral"),
        ("tasks", ("interview",)),
        ("disorders", ("dementia",)),
        ("evidence_level", "strong"),
    ],
)
def test_feature_metadata_rejects_unknown_values(field, value):
    kwargs = valid_definition_kwargs()
    kwargs[field] = value
    with pytest.raises(ValueError):
        FeatureDefinition(**kwargs)


def test_metadata_sets_are_frozen_and_exact():
    assert isinstance(DOMAINS, frozenset)
    assert DOMAINS == {
        "audio_quality",
        "timing",
        "respiration",
        "phonation",
        "prosody",
        "spectral",
        "articulation",
        "rhythm",
        "lexical",
        "psycholinguistic",
        "morphosyntactic",
        "disfluency",
        "semantic",
        "discourse",
        "task",
    }
    assert LANGUAGE_SCOPES == {
        "language_independent",
        "language_sensitive",
        "language_dependent",
        "language_specific",
    }
    assert TASK_IDS == {
        "connected_speech",
        "picture_description",
        "story_recall",
        "semantic_fluency",
        "phonemic_fluency",
        "reading",
        "sustained_vowel",
        "ddk",
    }
    assert DISORDERS == {
        "ad",
        "mci",
        "ppa",
        "ftd",
        "dlb",
        "pd",
        "pdd",
        "als",
        "mnd",
        "hd",
        "ms",
        "ataxia",
        "psp",
        "msa",
        "cbs",
    }
    assert EVIDENCE_LEVELS == {
        "systematic_review",
        "multi_study",
        "single_study",
        "standard_feature_set",
        "derived_companion",
    }


def test_motor_feature_prefixes_are_stable_catalog_prefixes():
    assert {"artic_", "rhythm_", "task_"} <= KEY_PREFIXES


def test_definition_defaults_keep_existing_constructors_valid():
    definition = FeatureDefinition(
        key="time_response_latency",
        pack="adult_neuro",
        level="recording",
        unit="s",
        population="adult",
        reference="SAY catalog v1",
    )
    assert definition.domain == "audio_quality"
    assert definition.language_scope == "language_independent"
    assert definition.tasks == ()
    assert definition.disorders == ()
    assert definition.evidence_level == "derived_companion"


def test_definition_metadata_is_immutable():
    definition = FeatureDefinition(**valid_definition_kwargs())
    with pytest.raises(dataclasses.FrozenInstanceError):
        definition.domain = "timing"


def test_list_features_new_packs_and_filters():
    register_feature(FeatureDefinition(**valid_definition_kwargs()))
    assert set(speech_features.PACKS) == {
        "acoustic",
        "adult_neuro",
        "motor_neuro",
        "standardized_acoustic",
    }
    assert [f.key for f in list_features(pack="motor_neuro")] == ["resp_breath_group_count"]
    assert "resp_breath_group_count" in {
        definition.key for definition in list_features(task="connected_speech")
    }
    assert list_features(disorder="als") == list_features(disorder="pd")
    assert [f.key for f in list_features(evidence_level="systematic_review")] == [
        "resp_breath_group_count"
    ]


@pytest.mark.parametrize(
    ("filter_name", "bad_value"),
    [
        ("domain", "unknown"),
        ("language_scope", "neutral"),
        ("task", "interview"),
        ("disorder", "dementia"),
        ("evidence_level", "strong"),
    ],
)
def test_list_features_rejects_unknown_filter_values(filter_name, bad_value):
    with pytest.raises(ValueError, match=filter_name):
        list_features(**{filter_name: bad_value})


def _record(**overrides) -> EvidenceRecord:
    base = dict(
        candidate="resp_breath_group_count",
        domain="respiration",
        language_scope="language_independent",
        tasks=("connected_speech",),
        disorders=("als", "pd"),
        source=RESPIRATION_REFERENCE,
        evidence_level="systematic_review",
        status="implemented",
        feature_keys=("resp_breath_group_count",),
    )
    base.update(overrides)
    return EvidenceRecord(**base)


def test_validate_evidence_accepts_all_statuses():
    records = [
        _record(status="existing"),
        _record(status="implemented"),
        _record(status="optional"),
        _record(status="deferred", feature_keys=(), reason="no annotation layer yet"),
    ]
    checked = validate_evidence(records, catalog_keys={"resp_breath_group_count"})
    assert checked == tuple(records)
    assert isinstance(checked, tuple)


def test_validate_evidence_rejects_unknown_status():
    with pytest.raises(ValueError, match="status"):
        validate_evidence([_record(status="planned")], catalog_keys=set())


def test_validate_evidence_requires_source():
    with pytest.raises(ValueError, match="source"):
        validate_evidence([_record(source="")], catalog_keys={"resp_breath_group_count"})


@pytest.mark.parametrize("status", ["existing", "implemented", "optional"])
def test_validate_evidence_requires_catalog_keys_for_active_statuses(status):
    with pytest.raises(ValueError, match="catalog keys"):
        validate_evidence([_record(status=status, feature_keys=())], catalog_keys=set())
    with pytest.raises(ValueError, match="catalog keys"):
        validate_evidence(
            [_record(status=status, feature_keys=("missing_key",))],
            catalog_keys={"resp_breath_group_count"},
        )


def test_validate_evidence_requires_reason_for_deferred():
    with pytest.raises(ValueError, match="reason"):
        validate_evidence(
            [_record(status="deferred", feature_keys=(), reason="")],
            catalog_keys=set(),
        )
