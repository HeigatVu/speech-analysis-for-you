"""Tests for the feature catalog and result contracts (Task 4).

Covers ``FeatureDefinition`` validation and immutability, globally unique
keys, deterministic ``list_features`` ordering and filtering, ``FeatureIssue``
severities and free codes, ``FeatureBundle`` identifier columns on empty and
populated tables, NaN preservation with paired issues, JSON-serializable
provenance, the ``FeaturePack`` protocol with static pack registration, the
complete stable error set, and root re-exports.
"""

import dataclasses
import json
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

import speech_features
from speech_features import (
    CATALOG_VERSION,
    ExtractionContext,
    FeatureBundle,
    FeatureDefinition,
    FeatureIssue,
    FeaturePack,
    PACKS,
    STABLE_ERROR_CODES,
    UnknownPackError,
    list_features,
)
from speech_features import catalog, result
from speech_features.schema import FeatureExtractionError, MissingInputError

RECORDING_COLUMNS = ["recording_id", "speaker_id"]
UTTERANCE_COLUMNS = ["recording_id", "speaker_id", "utterance_id", "start_s", "end_s"]
ISSUE_COLUMNS = [
    "recording_id",
    "speaker_id",
    "utterance_id",
    "feature",
    "code",
    "severity",
    "message",
]

STABLE_CODES = frozenset(
    {
        "INVALID_DOCUMENT",
        "INVALID_CHAT",
        "INVALID_AUDIO",
        "UNSUPPORTED_AUDIO",
        "MISSING_INPUT",
        "TARGET_SPEAKER_REQUIRED",
        "MISSING_ANNOTATION",
        "UNKNOWN_PACK",
        "INVALID_CONFIG",
        "EXTRACTION_ERROR",
    }
)

ROOT_EXPORTS = (
    "CATALOG_VERSION",
    "ExtractionContext",
    "ExtractionError",
    "FeatureBundle",
    "FeatureDefinition",
    "FeatureIssue",
    "FeaturePack",
    "InvalidAudioError",
    "InvalidConfigError",
    "MissingAnnotationError",
    "MissingInputError",
    "PACKS",
    "STABLE_ERROR_CODES",
    "TargetSpeakerRequiredError",
    "UnknownPackError",
    "UnsupportedAudioError",
    "list_features",
)

STABLE_ERROR_CLASSES = {
    "InvalidDocumentError": "INVALID_DOCUMENT",
    "InvalidChatError": "INVALID_CHAT",
    "InvalidAudioError": "INVALID_AUDIO",
    "UnsupportedAudioError": "UNSUPPORTED_AUDIO",
    "MissingInputError": "MISSING_INPUT",
    "TargetSpeakerRequiredError": "TARGET_SPEAKER_REQUIRED",
    "MissingAnnotationError": "MISSING_ANNOTATION",
    "UnknownPackError": "UNKNOWN_PACK",
    "InvalidConfigError": "INVALID_CONFIG",
    "ExtractionError": "EXTRACTION_ERROR",
}


@pytest.fixture(autouse=True)
def _restore_catalog():
    snapshot = list(catalog._FEATURES)
    yield
    catalog._FEATURES[:] = snapshot


def _definition(**overrides):
    base = dict(
        key="time_response_latency",
        pack="adult_neuro",
        level="recording",
        unit="s",
        prerequisites=(),
        population="adult",
        formula_version=1,
        reference="SAY catalog v1",
    )
    base.update(overrides)
    return FeatureDefinition(**base)


def _empty_bundle():
    return FeatureBundle(
        recordings=pd.DataFrame(columns=RECORDING_COLUMNS),
        utterances=pd.DataFrame(columns=UTTERANCE_COLUMNS),
        issues=pd.DataFrame(columns=ISSUE_COLUMNS),
        provenance={"catalog_version": 1},
    )


# ---------------------------------------------------------------------------
# FeatureDefinition validation and immutability
# ---------------------------------------------------------------------------
def test_definition_accepts_valid_metadata():
    definition = _definition()
    assert definition.key == "time_response_latency"
    assert definition.pack == "adult_neuro"
    assert definition.level == "recording"
    assert definition.unit == "s"
    assert definition.prerequisites == ()
    assert definition.population == "adult"
    assert definition.formula_version == 1
    assert definition.reference == "SAY catalog v1"


def test_definition_rejects_invalid_key_prefix():
    with pytest.raises(ValueError, match="prefix"):
        _definition(key="foo_latency")


def test_definition_rejects_unknown_pack():
    with pytest.raises(ValueError, match="pack"):
        _definition(pack="pediatric")


def test_definition_rejects_invalid_level():
    with pytest.raises(ValueError, match="level"):
        _definition(level="word")


@pytest.mark.parametrize("field", ["unit", "population", "reference"])
def test_definition_rejects_empty_metadata(field):
    with pytest.raises(ValueError, match=field):
        _definition(**{field: ""})


def test_definition_rejects_invalid_formula_version():
    with pytest.raises(ValueError, match="formula_version"):
        _definition(formula_version=0)


def test_definition_is_frozen_and_prerequisites_immutable():
    definition = _definition(prerequisites=["audio_duration"])
    assert isinstance(definition.prerequisites, tuple)
    assert definition.prerequisites == ("audio_duration",)
    with pytest.raises(dataclasses.FrozenInstanceError):
        definition.key = "time_other"


# ---------------------------------------------------------------------------
# list_features: ordering, filters, uniqueness
# ---------------------------------------------------------------------------
def test_list_features_mirrors_registered_definitions():
    assert list_features() == tuple(sorted(catalog._FEATURES, key=lambda d: d.key))
    assert list_features(pack="acoustic") == tuple(
        f for f in list_features() if f.pack == "acoustic"
    )


def test_list_features_deterministic_order_and_filters():
    catalog.register_feature(_definition(key="lex_test_ttr", pack="adult_neuro", level="utterance"))
    catalog.register_feature(
        _definition(key="time_test_latency", pack="adult_neuro", level="recording")
    )
    catalog.register_feature(
        _definition(key="audio_test_duration", pack="acoustic", level="recording")
    )
    features = list_features()
    assert isinstance(features, tuple)
    assert [f.key for f in features] == sorted(f.key for f in features)
    acoustic = list_features(pack="acoustic")
    assert "audio_test_duration" in [f.key for f in acoustic]
    assert all(f.pack == "acoustic" for f in acoustic)
    assert all(f.level == "recording" for f in list_features(level="recording"))
    # The real adult_neuro utterance-level keys (Task 9) plus the fixture key
    # are returned in deterministic sorted order.
    assert [f.key for f in list_features(pack="adult_neuro", level="utterance")] == sorted(
        [
            "lex_test_ttr",
            "discourse_response_latency_s",
            "discourse_turn_overlap_s",
            "discourse_turn_syllable_count",
            "discourse_turn_word_count",
        ]
    )


def test_list_features_rejects_unknown_pack():
    with pytest.raises(UnknownPackError) as exc:
        list_features(pack="pediatric")
    assert exc.value.code == "UNKNOWN_PACK"


def test_list_features_rejects_unknown_level():
    with pytest.raises(ValueError, match="level"):
        list_features(level="word")


def test_duplicate_key_rejected():
    catalog.register_feature(_definition())
    with pytest.raises(ValueError, match="duplicate"):
        catalog.register_feature(_definition(key="time_response_latency"))


# ---------------------------------------------------------------------------
# FeatureIssue
# ---------------------------------------------------------------------------
def test_issue_accepts_valid_fields():
    issue = FeatureIssue(
        recording_id="r1",
        speaker_id="s1",
        code="MISSING_ANNOTATION",
        severity="warning",
        message="no reviewed annotations",
    )
    assert issue.utterance_id is None
    assert issue.feature is None


def test_issue_accepts_unsupported_chat_tier_code():
    issue = FeatureIssue(
        recording_id="r1",
        speaker_id="s1",
        code="UNSUPPORTED_CHAT_TIER",
        severity="warning",
        message="tier %xyz preserved verbatim",
    )
    assert issue.code == "UNSUPPORTED_CHAT_TIER"


def test_issue_accepts_both_severities():
    for severity in ("warning", "error"):
        FeatureIssue(
            recording_id="r1",
            speaker_id="s1",
            code="EXTRACTION_ERROR",
            severity=severity,
            message="m",
        )


def test_issue_rejects_unknown_severity():
    with pytest.raises(ValueError, match="severity"):
        FeatureIssue(
            recording_id="r1",
            speaker_id="s1",
            code="EXTRACTION_ERROR",
            severity="fatal",
            message="m",
        )


def test_issue_rejects_empty_code_or_message():
    with pytest.raises(ValueError, match="code"):
        FeatureIssue(recording_id="r1", speaker_id="s1", code="", severity="error", message="m")
    with pytest.raises(ValueError, match="message"):
        FeatureIssue(recording_id="r1", speaker_id="s1", code="X", severity="error", message="")


def test_issue_is_frozen():
    issue = FeatureIssue(
        recording_id="r1",
        speaker_id="s1",
        code="X",
        severity="warning",
        message="m",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        issue.code = "Y"


# ---------------------------------------------------------------------------
# FeatureBundle
# ---------------------------------------------------------------------------
def test_bundle_empty_tables_retain_columns():
    bundle = _empty_bundle()
    assert list(bundle.recordings.columns) == RECORDING_COLUMNS
    assert list(bundle.utterances.columns) == UTTERANCE_COLUMNS
    assert list(bundle.issues.columns) == ISSUE_COLUMNS
    assert all(bundle.recordings.empty for bundle in (bundle,))
    assert bundle.recordings.empty
    assert bundle.utterances.empty
    assert bundle.issues.empty


def test_bundle_accepts_populated_tables():
    bundle = FeatureBundle(
        recordings=pd.DataFrame(
            {"recording_id": ["r1"], "speaker_id": ["s1"], "audio_test_duration": [12.5]}
        ),
        utterances=pd.DataFrame(
            {
                "recording_id": ["r1"],
                "speaker_id": ["s1"],
                "utterance_id": ["u1"],
                "start_s": [0.0],
                "end_s": [3.0],
            }
        ),
        issues=pd.DataFrame(
            {
                "recording_id": ["r1"],
                "speaker_id": ["s1"],
                "utterance_id": [None],
                "feature": [None],
                "code": ["MISSING_ANNOTATION"],
                "severity": ["warning"],
                "message": ["no annotations"],
            }
        ),
        provenance={"catalog_version": 1},
    )
    assert list(bundle.recordings.columns) == [
        "recording_id",
        "speaker_id",
        "audio_test_duration",
    ]


def test_bundle_rejects_missing_identifier_columns():
    with pytest.raises(ValueError, match="recordings"):
        FeatureBundle(
            recordings=pd.DataFrame(columns=["recording_id"]),
            utterances=pd.DataFrame(columns=UTTERANCE_COLUMNS),
            issues=pd.DataFrame(columns=ISSUE_COLUMNS),
            provenance={},
        )
    with pytest.raises(ValueError, match="utterances"):
        FeatureBundle(
            recordings=pd.DataFrame(columns=RECORDING_COLUMNS),
            utterances=pd.DataFrame(columns=["recording_id", "speaker_id"]),
            issues=pd.DataFrame(columns=ISSUE_COLUMNS),
            provenance={},
        )
    with pytest.raises(ValueError, match="issues"):
        FeatureBundle(
            recordings=pd.DataFrame(columns=RECORDING_COLUMNS),
            utterances=pd.DataFrame(columns=UTTERANCE_COLUMNS),
            issues=pd.DataFrame(columns=["recording_id", "code"]),
            provenance={},
        )


@pytest.mark.parametrize("field", ["recordings", "utterances", "issues"])
def test_bundle_rejects_non_dataframe(field):
    tables = {
        "recordings": pd.DataFrame(columns=RECORDING_COLUMNS),
        "utterances": pd.DataFrame(columns=UTTERANCE_COLUMNS),
        "issues": pd.DataFrame(columns=ISSUE_COLUMNS),
    }
    tables[field] = None
    with pytest.raises(ValueError, match="DataFrame"):
        FeatureBundle(**tables, provenance={})


def test_bundle_preserves_nan_with_issue():
    utterances = pd.DataFrame(
        {
            "recording_id": ["r1"],
            "speaker_id": ["s1"],
            "utterance_id": ["u1"],
            "start_s": [0.0],
            "end_s": [3.0],
            "morph_test_ttr": [np.nan],
        }
    )
    issues = pd.DataFrame(
        {
            "recording_id": ["r1"],
            "speaker_id": ["s1"],
            "utterance_id": ["u1"],
            "feature": ["morph_test_ttr"],
            "code": ["MISSING_ANNOTATION"],
            "severity": ["warning"],
            "message": ["no reviewed annotations"],
        }
    )
    bundle = FeatureBundle(
        recordings=pd.DataFrame(columns=RECORDING_COLUMNS),
        utterances=utterances,
        issues=issues,
        provenance={},
    )
    assert np.isnan(bundle.utterances.loc[0, "morph_test_ttr"])
    assert not (bundle.utterances["morph_test_ttr"] == 0.0).any()
    assert list(bundle.issues["code"]) == ["MISSING_ANNOTATION"]


def test_bundle_requires_json_serializable_provenance():
    with pytest.raises(ValueError, match="provenance"):
        FeatureBundle(
            recordings=pd.DataFrame(columns=RECORDING_COLUMNS),
            utterances=pd.DataFrame(columns=UTTERANCE_COLUMNS),
            issues=pd.DataFrame(columns=ISSUE_COLUMNS),
            provenance={"bad": object()},
        )
    provenance = {"catalog_version": 1, "packs": ["acoustic"], "hashes": {"a": "x"}}
    bundle = _empty_bundle()
    bundle = FeatureBundle(
        recordings=pd.DataFrame(columns=RECORDING_COLUMNS),
        utterances=pd.DataFrame(columns=UTTERANCE_COLUMNS),
        issues=pd.DataFrame(columns=ISSUE_COLUMNS),
        provenance=provenance,
    )
    assert json.loads(json.dumps(bundle.provenance)) == provenance


def test_bundle_is_frozen():
    bundle = _empty_bundle()
    with pytest.raises(dataclasses.FrozenInstanceError):
        bundle.recordings = pd.DataFrame()


# ---------------------------------------------------------------------------
# ExtractionContext
# ---------------------------------------------------------------------------
def test_context_is_neutral_and_frozen():
    context = ExtractionContext(
        recording_id="r1",
        target_speaker="s1",
        audio_path="audio.wav",
        document_path="doc.json",
    )
    assert context.recording_id == "r1"
    assert context.target_speaker == "s1"
    assert context.audio_path == "audio.wav"
    assert context.document_path == "doc.json"
    fields = {f.name for f in dataclasses.fields(ExtractionContext)}
    assert fields <= {
        "recording_id",
        "target_speaker",
        "audio_path",
        "document_path",
        "config",
        "provenance",
    }
    assert isinstance(context.provenance, MappingProxyType)
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.recording_id = "r2"


# ---------------------------------------------------------------------------
# FeaturePack protocol and static registration
# ---------------------------------------------------------------------------
def test_pack_registry_is_static_and_immutable():
    assert isinstance(PACKS, MappingProxyType)
    assert set(PACKS) == {"acoustic", "adult_neuro"}


def test_every_pack_satisfies_protocol():
    for pack in PACKS.values():
        assert isinstance(pack, FeaturePack)
        assert pack.name in PACKS
        assert pack.version == 1


# ---------------------------------------------------------------------------
# Stable errors
# ---------------------------------------------------------------------------
def test_stable_error_codes_complete():
    assert STABLE_ERROR_CODES == STABLE_CODES


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("UnsupportedAudioError", "UNSUPPORTED_AUDIO"),
        ("TargetSpeakerRequiredError", "TARGET_SPEAKER_REQUIRED"),
        ("MissingAnnotationError", "MISSING_ANNOTATION"),
        ("UnknownPackError", "UNKNOWN_PACK"),
        ("InvalidConfigError", "INVALID_CONFIG"),
        ("ExtractionError", "EXTRACTION_ERROR"),
    ],
)
def test_new_error_classes_have_stable_codes(name, code):
    cls = getattr(result, name)
    assert cls.code == code
    assert issubclass(cls, FeatureExtractionError)


def test_existing_error_classes_reused():
    assert result.InvalidDocumentError is speech_features.document.InvalidDocumentError
    assert result.InvalidChatError is speech_features.formats.chat.InvalidChatError
    assert result.InvalidAudioError is speech_features.pipeline.InvalidAudioError
    assert result.MissingInputError is MissingInputError
    assert MissingInputError.code == "MISSING_INPUT"


def test_complete_stable_error_set_exposed_in_one_place():
    for name, code in STABLE_ERROR_CLASSES.items():
        cls = getattr(result, name)
        assert cls.code == code
        assert issubclass(cls, Exception)


# ---------------------------------------------------------------------------
# Root exports
# ---------------------------------------------------------------------------
def test_catalog_version_is_one():
    assert CATALOG_VERSION == 1


def test_root_exports_are_public():
    for name in ROOT_EXPORTS:
        assert hasattr(speech_features, name)
        assert name in speech_features.__all__
