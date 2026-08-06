"""Tests for the population-neutral SpeechDocument model (Task 2).

Covers JSON v2 validation (immutability, duplicate IDs, finite ordered times,
token-within-utterance times, dependency heads, annotation confidence, ISO
language ``vie``), Vietnamese text preservation (NFC, ``d``/``đ``,
multi-syllable tokens, shared ``word_id``, per-token language), deterministic
JSON v1 migration, format detection, overwrite refusal, and root re-exports.
"""

import dataclasses
import json
import unicodedata
from types import MappingProxyType

import pytest

import speech_features
from speech_features.document import (
    AnnotationLayer,
    DocumentToken,
    DocumentUtterance,
    InvalidDocumentError,
    SpeechDocument,
    load_document,
    save_document,
)

NEW_ROOT_API = (
    "AnnotationLayer",
    "DocumentToken",
    "DocumentUtterance",
    "DocumentSpeaker",
    "MediaRef",
    "SpeechDocument",
    "load_document",
    "save_document",
)


def _v2(**overrides):
    doc = {
        "version": 2,
        "document_id": "tr-1",
        "language": "vie",
        "media": [{"id": "m1", "kind": "audio", "path": "audio.wav", "sha256": "abc123"}],
        "speakers": [{"id": "p1", "name": "Người tham gia", "role": "participant"}],
        "utterances": [
            {
                "id": "u0001",
                "speaker_id": "p1",
                "start_s": 0.0,
                "end_s": 2.0,
                "tokens": [
                    {"id": "u0001_t0001", "text": "Xin chào", "start_s": 0.1, "end_s": 0.4},
                    {"id": "u0001_t0002", "text": "đi", "start_s": 0.5, "end_s": 0.8},
                ],
            }
        ],
        "annotations": [
            {
                "layer": "pos",
                "source": "reviewed",
                "confidence": 0.95,
                "values": {"u0001_t0001": "intj"},
            }
        ],
        "raw_tiers": {"@Begin": "123", "%mor": "\tXin chào\tv"},
    }
    doc.update(overrides)
    return doc


def _write(tmp_path, name, payload):
    p = tmp_path / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _v1():
    return {
        "version": 1,
        "transcript_id": "tr-1",
        "language": "vi",
        "utterances": [
            {
                "speaker": "examiner",
                "start_s": 0.0,
                "end_s": 1.0,
                "tokens": [
                    {"kind": "word", "text": "Xin chào"},
                    {"kind": "filler", "text": "à"},
                ],
            },
            {
                "speaker": "participant",
                "start_s": 1.2,
                "end_s": 2.5,
                "tokens": [
                    {"kind": "word", "text": "con", "word_id": "grp-1"},
                    {"kind": "word", "text": "đi", "word_id": "grp-1"},
                    {"kind": "word", "text": "chơi", "start_s": 1.5, "end_s": 2.0},
                ],
            },
        ],
    }


class TestDocumentValidation:
    def test_valid_v2_document_loads(self, tmp_path):
        doc = load_document(_write(tmp_path, "doc.json", _v2()))
        assert doc.document_id == "tr-1"
        assert doc.language == "vie"
        assert doc.media[0].id == "m1"
        assert doc.speakers[0].id == "p1"
        assert doc.utterances[0].tokens[0].text == "Xin chào"
        assert doc.annotations[0].layer == "pos"
        assert doc.raw_tiers["@Begin"] == "123"

    def test_document_is_frozen_and_deeply_immutable(self, tmp_path):
        doc = load_document(_write(tmp_path, "doc.json", _v2()))
        assert isinstance(doc.utterances, tuple)
        assert isinstance(doc.utterances[0].tokens, tuple)
        assert isinstance(doc.annotations[0].values, MappingProxyType)
        assert isinstance(doc.raw_tiers, MappingProxyType)
        with pytest.raises(dataclasses.FrozenInstanceError):
            doc.document_id = "other"
        with pytest.raises(dataclasses.FrozenInstanceError):
            doc.utterances[0].tokens[0].text = "other"

    def test_direct_construction_maps_to_immutable_containers(self):
        doc = SpeechDocument(
            document_id="d",
            utterances=[
                DocumentUtterance(
                    id="u0001",
                    speaker_id="p1",
                    start_s=0.0,
                    end_s=1.0,
                    tokens=[DocumentToken(id="u0001_t0001", text="con")],
                )
            ],
            annotations=[AnnotationLayer(layer="pos", values={"u0001_t0001": "noun"})],
            raw_tiers={"@Begin": "b"},
        )
        assert isinstance(doc.utterances, tuple)
        assert isinstance(doc.utterances[0].tokens, tuple)
        assert isinstance(doc.annotations[0].values, MappingProxyType)
        assert isinstance(doc.raw_tiers, MappingProxyType)

    @pytest.mark.parametrize(
        "override",
        [
            {"speakers": [{"id": "p1", "name": "a"}, {"id": "p1", "name": "b"}]},
            {"media": [{"id": "m1"}, {"id": "m1"}]},
            {
                "utterances": [
                    {"id": "u0001", "speaker_id": "p1", "start_s": 0.0, "end_s": 1.0},
                    {"id": "u0001", "speaker_id": "p1", "start_s": 1.0, "end_s": 2.0},
                ]
            },
            {
                "utterances": [
                    {
                        "id": "u0001",
                        "speaker_id": "p1",
                        "start_s": 0.0,
                        "end_s": 1.0,
                        "tokens": [
                            {"id": "t1", "text": "a"},
                            {"id": "t1", "text": "b"},
                        ],
                    }
                ]
            },
        ],
    )
    def test_duplicate_ids_rejected(self, tmp_path, override):
        with pytest.raises(InvalidDocumentError, match="duplicate"):
            load_document(_write(tmp_path, "doc.json", _v2(**override)))

    def test_non_finite_times_rejected(self, tmp_path):
        for value in (float("inf"), float("-inf"), float("nan")):
            with pytest.raises(InvalidDocumentError, match="finite"):
                load_document(
                    _write(
                        tmp_path,
                        "doc.json",
                        _v2(
                            utterances=[
                                {"id": "u0001", "speaker_id": "p1", "start_s": value, "end_s": 1.0}
                            ]
                        ),
                    )
                )

    def test_end_before_start_rejected(self, tmp_path):
        with pytest.raises(InvalidDocumentError, match="end"):
            load_document(
                _write(
                    tmp_path,
                    "doc.json",
                    _v2(
                        utterances=[
                            {"id": "u0001", "speaker_id": "p1", "start_s": 2.0, "end_s": 1.0}
                        ]
                    ),
                )
            )

    def test_token_within_utterance_times(self, tmp_path):
        with pytest.raises(InvalidDocumentError, match="utterance"):
            load_document(
                _write(
                    tmp_path,
                    "doc.json",
                    _v2(
                        utterances=[
                            {
                                "id": "u0001",
                                "speaker_id": "p1",
                                "start_s": 0.0,
                                "end_s": 2.0,
                                "tokens": [
                                    {"id": "t1", "text": "a", "start_s": -0.1, "end_s": 0.4}
                                ],
                            }
                        ]
                    ),
                )
            )
        with pytest.raises(InvalidDocumentError, match="utterance"):
            load_document(
                _write(
                    tmp_path,
                    "doc.json",
                    _v2(
                        utterances=[
                            {
                                "id": "u0001",
                                "speaker_id": "p1",
                                "start_s": 0.0,
                                "end_s": 2.0,
                                "tokens": [{"id": "t1", "text": "a", "start_s": 0.1, "end_s": 2.5}],
                            }
                        ]
                    ),
                )
            )

    def test_dependency_heads_must_exist(self, tmp_path):
        with pytest.raises(InvalidDocumentError, match="dep_head"):
            load_document(
                _write(
                    tmp_path,
                    "doc.json",
                    _v2(
                        utterances=[
                            {
                                "id": "u0001",
                                "speaker_id": "p1",
                                "start_s": 0.0,
                                "end_s": 2.0,
                                "tokens": [{"id": "t1", "text": "a", "dep_head": "missing_tok"}],
                            }
                        ]
                    ),
                )
            )
        doc = load_document(
            _write(
                tmp_path,
                "doc.json",
                _v2(
                    utterances=[
                        {
                            "id": "u0001",
                            "speaker_id": "p1",
                            "start_s": 0.0,
                            "end_s": 2.0,
                            "tokens": [
                                {"id": "t1", "text": "a", "dep_head": "t2", "dep_rel": "root"},
                                {"id": "t2", "text": "b"},
                            ],
                        }
                    ]
                ),
            )
        )
        assert doc.utterances[0].tokens[0].dep_head == "t2"

    def test_annotation_confidence_in_unit_interval(self, tmp_path):
        for bad in (1.5, -0.1):
            with pytest.raises(InvalidDocumentError, match="confidence"):
                load_document(
                    _write(
                        tmp_path,
                        "doc.json",
                        _v2(annotations=[{"layer": "pos", "confidence": bad, "values": {}}]),
                    )
                )
        doc = load_document(
            _write(
                tmp_path,
                "doc.json",
                _v2(
                    annotations=[
                        {"layer": "pos", "confidence": 0.0, "values": {}},
                        {"layer": "gra", "confidence": 1.0, "values": {}},
                    ]
                ),
            )
        )
        assert doc.annotations[0].confidence == 0.0
        assert doc.annotations[1].confidence == 1.0

    def test_language_is_exactly_vie_by_default(self, tmp_path):
        doc = load_document(
            _write(tmp_path, "doc.json", _v2(utterances=[])),
        )
        assert doc.language == "vie"
        for lang in ("vi", "en", "Vie"):
            with pytest.raises(InvalidDocumentError, match="vie"):
                load_document(_write(tmp_path, "doc.json", _v2(language=lang, utterances=[])))

    def test_unknown_version_rejected(self, tmp_path):
        with pytest.raises(InvalidDocumentError, match="version"):
            load_document(_write(tmp_path, "doc.json", {"version": 3}))


class TestVietnameseText:
    def test_nfc_preserved_on_load(self, tmp_path):
        decomposed = "Xin cha\u0300o"
        assert decomposed != unicodedata.normalize("NFC", decomposed)
        doc = load_document(
            _write(
                tmp_path,
                "doc.json",
                _v2(
                    utterances=[
                        {
                            "id": "u0001",
                            "speaker_id": "p1",
                            "start_s": 0.0,
                            "end_s": 2.0,
                            "tokens": [{"id": "t1", "text": decomposed}],
                        }
                    ]
                ),
            )
        )
        assert doc.utterances[0].tokens[0].text == "Xin chào"
        assert doc.utterances[0].tokens[0].text == unicodedata.normalize("NFC", decomposed)

    def test_d_vs_d_dinh_preserved(self, tmp_path):
        doc = load_document(
            _write(
                tmp_path,
                "doc.json",
                _v2(
                    utterances=[
                        {
                            "id": "u0001",
                            "speaker_id": "p1",
                            "start_s": 0.0,
                            "end_s": 2.0,
                            "tokens": [
                                {"id": "t1", "text": "đi"},
                                {"id": "t2", "text": "di"},
                            ],
                        }
                    ]
                ),
            )
        )
        assert doc.utterances[0].tokens[0].text == "đi"
        assert doc.utterances[0].tokens[1].text == "di"
        assert doc.utterances[0].tokens[0].text != doc.utterances[0].tokens[1].text

    def test_multi_syllable_token_never_split(self, tmp_path):
        doc = load_document(
            _write(
                tmp_path,
                "doc.json",
                _v2(
                    utterances=[
                        {
                            "id": "u0001",
                            "speaker_id": "p1",
                            "start_s": 0.0,
                            "end_s": 2.0,
                            "tokens": [{"id": "t1", "text": "Xin chào"}],
                        }
                    ]
                ),
            )
        )
        tokens = doc.utterances[0].tokens
        assert len(tokens) == 1
        assert tokens[0].text == "Xin chào"

    def test_shared_word_id_round_trips(self, tmp_path):
        p = _write(
            tmp_path,
            "doc.json",
            _v2(
                utterances=[
                    {
                        "id": "u0001",
                        "speaker_id": "p1",
                        "start_s": 0.0,
                        "end_s": 2.0,
                        "tokens": [
                            {"id": "t1", "text": "thành", "word_id": "w1"},
                            {"id": "t2", "text": "phố", "word_id": "w1"},
                        ],
                    }
                ]
            ),
        )
        doc = load_document(p)
        assert [t.word_id for t in doc.utterances[0].tokens] == ["w1", "w1"]
        save_document(doc, tmp_path / "rt.json")
        assert load_document(tmp_path / "rt.json") == doc

    def test_token_language_for_code_switching(self, tmp_path):
        p = _write(
            tmp_path,
            "doc.json",
            _v2(
                utterances=[
                    {
                        "id": "u0001",
                        "speaker_id": "p1",
                        "start_s": 0.0,
                        "end_s": 2.0,
                        "tokens": [
                            {"id": "t1", "text": "OK"},
                            {"id": "t2", "text": "đi", "language": "en"},
                        ],
                    }
                ]
            ),
        )
        doc = load_document(p)
        assert doc.utterances[0].tokens[1].language == "en"
        assert doc.utterances[0].tokens[0].language is None


class TestV1Migration:
    def test_v1_migrates_deterministically(self, tmp_path):
        p = _write(tmp_path, "old.json", _v1())
        first = load_document(p)
        assert first.document_id == "tr-1"
        assert first.language == "vie"
        assert [u.id for u in first.utterances] == ["u0001", "u0002"]
        assert [t.id for t in first.utterances[0].tokens] == ["u0001_t0001", "u0001_t0002"]
        assert [t.id for t in first.utterances[1].tokens] == [
            "u0002_t0001",
            "u0002_t0002",
            "u0002_t0003",
        ]
        assert [s.id for s in first.speakers] == ["examiner", "participant"]
        second = load_document(_write(tmp_path, "old2.json", _v1()))
        assert second == first

    def test_v1_word_tokens_stay_whole_and_grouping_preserved(self, tmp_path):
        doc = load_document(_write(tmp_path, "old.json", _v1()))
        first_utt_tokens = doc.utterances[0].tokens
        assert [t.text for t in first_utt_tokens] == ["Xin chào", "à"]
        grouped = doc.utterances[1].tokens[:2]
        assert [t.word_id for t in grouped] == ["grp-1", "grp-1"]
        assert doc.utterances[1].tokens[2].word_id is None
        assert doc.utterances[1].tokens[2].start_s == 1.5
        assert doc.utterances[1].tokens[2].end_s == 2.0

    def test_v1_migration_round_trip_via_v2(self, tmp_path):
        p = _write(tmp_path, "old.json", _v1())
        doc = load_document(p)
        save_document(doc, tmp_path / "new.json")
        assert load_document(tmp_path / "new.json") == doc


class TestIO:
    def test_format_detected_by_extension(self, tmp_path):
        p = _write(tmp_path, "doc.json", _v2())
        assert load_document(p).document_id == "tr-1"

    def test_format_detected_by_content(self, tmp_path):
        p = _write(tmp_path, "document.data", _v2())
        assert load_document(p).document_id == "tr-1"

    def test_explicit_format_ignores_extension(self, tmp_path):
        p = _write(tmp_path, "doc.data", _v2())
        assert load_document(p, format="json").document_id == "tr-1"

    def test_save_round_trip_preserves_document(self, tmp_path):
        doc = load_document(_write(tmp_path, "doc.json", _v2()))
        save_document(doc, tmp_path / "copy.json")
        assert load_document(tmp_path / "copy.json") == doc

    def test_save_refuses_overwrite_unless_force(self, tmp_path):
        doc = load_document(_write(tmp_path, "doc.json", _v2()))
        p = tmp_path / "target.json"
        p.write_text("original", encoding="utf-8")
        with pytest.raises(FileExistsError, match="overwrite"):
            save_document(doc, p)
        save_document(doc, p, force=True)
        assert load_document(p) == doc


class TestRootExports:
    def test_new_document_api_reexported_from_root(self):
        for name in NEW_ROOT_API:
            assert hasattr(speech_features, name), name

    def test_no_extra_document_api_leaks_to_root(self):
        assert not hasattr(speech_features, "validate_document")
        assert not hasattr(speech_features, "InvalidDocumentError")

    def test_legacy_root_exports_remain_intact(self):
        assert speech_features.Token is not None
        assert speech_features.Utterance is not None
        assert speech_features.Transcript is not None
