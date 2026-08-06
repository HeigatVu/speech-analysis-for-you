import dataclasses
import hashlib
import json

import pytest

import speech_features.schema as schema


class TestExtractionConfig:
    def test_defaults_match_plan(self):
        config = schema.ExtractionConfig()
        assert config.sample_rate == 16000
        assert config.frame_size == 400  # 25 ms at 16 kHz
        assert config.hop_size == 160  # 10 ms
        assert config.pitch_min_hz == 70.0
        assert config.pitch_max_hz == 400.0
        assert config.pitch_autocorr_threshold == 0.30
        assert config.pause_threshold_s == 0.20
        assert config.long_pause_threshold_s == 2.0
        assert config.lpc_order == 12

    def test_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            schema.ExtractionConfig().sample_rate = 8000

    def test_accepts_overrides(self):
        config = schema.ExtractionConfig(sample_rate=8000, pause_threshold_s=0.5)
        assert config.sample_rate == 8000
        assert config.pause_threshold_s == 0.5


class TestFeatureResult:
    def test_holds_contract_fields(self):
        result = schema.FeatureResult(
            recording_id="rec-1",
            participant_id="p-1",
            task="picture_desc_1",
            features={"time_words_per_min": 90.0},
            quality_flags=["ok"],
            input_hashes={
                "audio": "a",
                "transcript": "b",
                "task_spec": "c",
            },
            config=schema.ExtractionConfig(),
        )
        assert result.recording_id == "rec-1"
        assert result.participant_id == "p-1"
        assert result.task == "picture_desc_1"
        assert result.features["time_words_per_min"] == 90.0

    def test_is_immutable(self):
        result = schema.FeatureResult(
            recording_id="r",
            participant_id="p",
            task="picture_desc_1",
            features={},
            quality_flags=[],
            input_hashes={},
            config=schema.ExtractionConfig(),
        )
        with pytest.raises(TypeError):
            result.features["x"] = 1.0


class TestNormalisation:
    def test_nfc_normalises_unaccented_fragments(self):
        # "e\u0302" (e + combining circumflex) normalises to "\u00ea"
        raw = "e\u0302"
        assert schema.nfc(raw) == "\u00ea"

    def test_leaves_already_normalised_untouched(self):
        assert schema.nfc("\u00ea") == "\u00ea"


class TestProvenanceHashes:
    def test_sha256_of_stream_matches_digest(self, tmp_path):
        path = tmp_path / "audio.wav"
        content = b"RIFF" + b"\x00" * 100
        path.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        assert schema.sha256_file(path) == digest
        assert len(schema.sha256_file(path)) == 64

    def test_sha256_matches_known_vector(self, tmp_path):
        path = tmp_path / "t.txt"
        path.write_text("hello\n", encoding="utf-8")
        import hashlib

        expected = hashlib.sha256(b"hello\n").hexdigest()
        assert schema.sha256_file(path) == expected

    def test_missing_file_raises_structured_error(self, tmp_path):
        with pytest.raises(schema.MissingInputError):
            schema.sha256_file(tmp_path / "missing.wav")

    def test_manifest_hashes_all_paths(self, tmp_path):
        audio = tmp_path / "audio.wav"
        transcript = tmp_path / "t.json"
        task_spec = tmp_path / "spec.json"
        audio.write_bytes(b"audio")
        transcript.write_bytes(b"transcript")
        task_spec.write_bytes(b"spec")
        manifest = {
            "version": 1,
            "rows": [
                {
                    "participant_id": "p-1",
                    "task": "picture_desc_1",
                    "recording_id": "rec-1",
                    "audio_id": "a-1",
                    "transcript_id": "tr-1",
                    "audio_path": str(audio),
                    "transcript_path": str(transcript),
                    "task_spec_path": str(task_spec),
                    "diagnosis": "AD",
                    "age": 72,
                    "sex": "F",
                    "education_years": 12,
                }
            ],
        }
        result = schema.manifest_hashes(manifest)
        assert result["audio"][str(audio)] == hashlib.sha256(b"audio").hexdigest()
        assert result["transcript"][str(transcript)] == hashlib.sha256(b"transcript").hexdigest()
        assert result["task_spec"][str(task_spec)] == hashlib.sha256(b"spec").hexdigest()

    def test_manifest_hashes_all_rows(self, tmp_path):
        spec_a = tmp_path / "a.json"
        spec_b = tmp_path / "b.json"
        audio = tmp_path / "audio.wav"
        transcript = tmp_path / "t.json"
        audio.write_bytes(b"audio")
        transcript.write_bytes(b"transcript")
        spec_a.write_bytes(b"spec-a")
        spec_b.write_bytes(b"spec-b")
        base = {
            "participant_id": "p-1",
            "recording_id": "rec-1",
            "audio_id": "a-1",
            "transcript_id": "tr-1",
            "audio_path": str(audio),
            "transcript_path": str(transcript),
            "diagnosis": "HC",
            "age": 68,
            "sex": "M",
            "education_years": 14,
        }
        manifest = {
            "version": 1,
            "rows": [
                dict(base, task="phonemic_fluency", task_spec_path=str(spec_a)),
                dict(base, task="semantic_fluency", task_spec_path=str(spec_b)),
            ],
        }
        result = schema.manifest_hashes(manifest)
        assert result["task_spec"][str(spec_a)] == hashlib.sha256(b"spec-a").hexdigest()
        assert result["task_spec"][str(spec_b)] == hashlib.sha256(b"spec-b").hexdigest()
        assert isinstance(result["audio"], dict)
        assert isinstance(result["transcript"], dict)
        assert len(result["task_spec"]) == 2  # two distinct task specs
        assert len(result["audio"]) == 1  # shared audio hashed once (dedup)

    def test_manifest_hashes_is_json_serializable(self, tmp_path):
        audio = tmp_path / "audio.wav"
        transcript = tmp_path / "t.json"
        task_spec = tmp_path / "spec.json"
        audio.write_bytes(b"audio")
        transcript.write_bytes(b"transcript")
        task_spec.write_bytes(b"spec")
        manifest = {
            "version": 1,
            "rows": [
                {
                    "participant_id": "p-1",
                    "task": "picture_desc_1",
                    "recording_id": "rec-1",
                    "audio_id": "a-1",
                    "transcript_id": "tr-1",
                    "audio_path": str(audio),
                    "transcript_path": str(transcript),
                    "task_spec_path": str(task_spec),
                    "diagnosis": "AD",
                    "age": 72,
                    "sex": "F",
                    "education_years": 12,
                }
            ],
        }
        result = schema.manifest_hashes(manifest)
        json.dumps(result)  # must not raise
        assert json.loads(json.dumps(result)) == result

    def test_manifest_hashes_raises_missing_input_code(self, tmp_path):
        manifest = {
            "version": 1,
            "rows": [
                {
                    "participant_id": "p-1",
                    "task": "picture_desc_1",
                    "recording_id": "rec-1",
                    "audio_id": "a-1",
                    "transcript_id": "tr-1",
                    "audio_path": str(tmp_path / "missing.wav"),
                    "transcript_path": str(tmp_path / "t.json"),
                    "task_spec_path": str(tmp_path / "spec.json"),
                    "diagnosis": "AD",
                    "age": 72,
                    "sex": "F",
                    "education_years": 12,
                }
            ],
        }
        with pytest.raises(schema.MissingInputError) as e:
            schema.manifest_hashes(manifest)
        assert e.value.code == "MISSING_INPUT"

    def test_manifest_hashes_rejects_empty_rows(self):
        with pytest.raises(schema.InvalidManifestError):
            schema.manifest_hashes({"version": 1, "rows": []})

    def test_manifest_hashes_rejects_non_dict_row(self):
        with pytest.raises(schema.InvalidManifestError):
            schema.manifest_hashes({"version": 1, "rows": ["not-a-dict"]})


class TestManifestValidation:
    def test_valid_manifest_parses(self, tmp_path):
        manifest = self._valid_manifest(tmp_path)
        rows = schema.validate_manifest(manifest)
        assert len(rows) == 1
        row = rows[0]
        assert row.participant_id == "p-1"
        assert row.diagnosis == "HC"
        assert row.age == 68

    def test_rejects_missing_participant_id(self, tmp_path):
        manifest = self._valid_manifest(tmp_path)
        del manifest["rows"][0]["participant_id"]
        with pytest.raises(schema.InvalidManifestError):
            schema.validate_manifest(manifest)

    def test_rejects_unknown_diagnosis(self, tmp_path):
        manifest = self._valid_manifest(tmp_path)
        manifest["rows"][0]["diagnosis"] = "maybe"
        with pytest.raises(schema.InvalidManifestError):
            schema.validate_manifest(manifest)

    def test_rejects_unknown_task(self, tmp_path):
        manifest = self._valid_manifest(tmp_path)
        row = manifest["rows"][0]
        row["task"] = "not_a_known_task"
        with pytest.raises(schema.UnknownTaskError):
            schema.validate_manifest(manifest)

    def test_rejects_negative_age(self, tmp_path):
        manifest = self._valid_manifest(tmp_path)
        manifest["rows"][0]["age"] = -1
        with pytest.raises(schema.InvalidManifestError):
            schema.validate_manifest(manifest)

    def test_rejects_negative_education_years(self, tmp_path):
        manifest = self._valid_manifest(tmp_path)
        manifest["rows"][0]["education_years"] = -1
        with pytest.raises(schema.InvalidManifestError):
            schema.validate_manifest(manifest)

    def test_rejects_duplicate_participant_task(self, tmp_path):
        manifest = self._valid_manifest(tmp_path)
        manifest["rows"].append(dict(manifest["rows"][0]))
        with pytest.raises(schema.InvalidManifestError):
            schema.validate_manifest(manifest)

    def test_duplicate_detection_is_nfc_normalised(self, tmp_path):
        manifest = self._valid_manifest(tmp_path)
        manifest["rows"][0]["participant_id"] = "e\u0302"  # e + combining circumflex
        manifest["rows"].append(dict(manifest["rows"][0]))
        manifest["rows"][1]["participant_id"] = "\u00ea"  # NFC-equivalent, precomposed
        with pytest.raises(schema.InvalidManifestError):
            schema.validate_manifest(manifest)

    @staticmethod
    def _valid_manifest(tmp_path):
        return {
            "version": 1,
            "rows": [
                {
                    "participant_id": "p-1",
                    "task": "picture_desc_1",
                    "recording_id": "rec-1",
                    "audio_id": "a-1",
                    "transcript_id": "tr-1",
                    "audio_path": str(tmp_path / "audio.wav"),
                    "transcript_path": str(tmp_path / "t.json"),
                    "task_spec_path": str(tmp_path / "spec.json"),
                    "diagnosis": "HC",
                    "age": 68,
                    "sex": "M",
                    "education_years": 14,
                }
            ],
        }


class TestTranscriptValidation:
    def test_valid_transcript_v1_parses(self):
        transcript = {
            "version": 1,
            "transcript_id": "tr-1",
            "language": "vi",
            "utterances": [
                {
                    "speaker": "examiner",
                    "start_s": 0.0,
                    "end_s": 1.0,
                    "tokens": [{"kind": "word", "text": "Xin ch\u00e0o"}],
                },
                {
                    "speaker": "participant",
                    "start_s": 1.2,
                    "end_s": 2.0,
                    "tokens": [
                        {"kind": "filler", "text": "\u00e0"},
                        {"kind": "word", "text": "con"},
                    ],
                },
            ],
        }
        parsed = schema.validate_transcript(transcript)
        assert len(parsed.utterances) == 2
        assert parsed.utterances[1].tokens[0].kind == "filler"
        assert parsed.transcript_id == "tr-1"
        assert parsed.language == "vi"

    def test_preserves_transcript_id_and_language(self):
        parsed = schema.validate_transcript(
            {
                "version": 1,
                "transcript_id": "tr-42",
                "language": "vi",
                "utterances": [
                    {
                        "speaker": "participant",
                        "start_s": 0.0,
                        "end_s": 0.5,
                        "tokens": [{"kind": "word", "text": "con"}],
                    }
                ],
            }
        )
        assert parsed.transcript_id == "tr-42"
        assert parsed.language == "vi"
        assert len(parsed.utterances) == 1

    def test_transcript_defaults_language_to_vi(self):
        parsed = schema.validate_transcript(
            {
                "version": 1,
                "transcript_id": "tr-1",
                "utterances": [],
            }
        )
        assert parsed.language == "vi"

    def test_transcript_is_immutable_and_deeply_immutable(self):
        parsed = schema.validate_transcript(
            {
                "version": 1,
                "transcript_id": "tr-1",
                "language": "vi",
                "utterances": [
                    {
                        "speaker": "participant",
                        "start_s": 0.0,
                        "end_s": 0.5,
                        "tokens": [{"kind": "word", "text": "con"}],
                    }
                ],
            }
        )
        assert isinstance(parsed.utterances, tuple)
        assert isinstance(parsed.utterances[0].tokens, tuple)
        with pytest.raises(dataclasses.FrozenInstanceError):
            parsed.transcript_id = "other"

    def test_rejects_non_finite_timestamps(self):
        transcript = {
            "version": 1,
            "utterances": [
                {
                    "speaker": "participant",
                    "start_s": 0.0,
                    "end_s": float("inf"),
                    "tokens": [{"kind": "word", "text": "con"}],
                }
            ],
        }
        with pytest.raises(schema.InvalidTranscriptError):
            schema.validate_transcript(transcript)

    def test_rejects_end_before_start(self):
        transcript = {
            "version": 1,
            "utterances": [
                {
                    "speaker": "participant",
                    "start_s": 2.0,
                    "end_s": 1.0,
                    "tokens": [{"kind": "word", "text": "con"}],
                }
            ],
        }
        with pytest.raises(schema.InvalidTranscriptError):
            schema.validate_transcript(transcript)

    def test_rejects_unknown_token_kind(self):
        transcript = {
            "version": 1,
            "utterances": [
                {
                    "speaker": "participant",
                    "start_s": 0.0,
                    "end_s": 1.0,
                    "tokens": [{"kind": "interjection", "text": "con"}],
                }
            ],
        }
        with pytest.raises(schema.InvalidTranscriptError):
            schema.validate_transcript(transcript)

    def test_rejects_unknown_speaker(self):
        transcript = {
            "version": 1,
            "utterances": [
                {
                    "speaker": "nobody",
                    "start_s": 0.0,
                    "end_s": 1.0,
                    "tokens": [{"kind": "word", "text": "con"}],
                }
            ],
        }
        with pytest.raises(schema.InvalidTranscriptError):
            schema.validate_transcript(transcript)

    def test_optional_token_timestamps_preserved(self):
        parsed = schema.validate_transcript(
            {
                "version": 1,
                "transcript_id": "tr-1",
                "utterances": [
                    {
                        "speaker": "participant",
                        "start_s": 0.0,
                        "end_s": 2.0,
                        "tokens": [
                            {"kind": "word", "text": "con", "start_s": 0.1, "end_s": 0.4},
                            {"kind": "filler", "text": "\u00e0"},
                        ],
                    }
                ],
            }
        )
        tok = parsed.utterances[0].tokens
        assert tok[0].start_s == 0.1
        assert tok[0].end_s == 0.4
        assert tok[1].start_s is None
        assert tok[1].end_s is None

    def test_rejects_bad_token_timestamps(self):
        def make(token):
            return {
                "version": 1,
                "utterances": [
                    {
                        "speaker": "participant",
                        "start_s": 0.0,
                        "end_s": 2.0,
                        "tokens": [{"kind": "word", "text": "con", **token}],
                    }
                ],
            }

        with pytest.raises(schema.InvalidTranscriptError):
            schema.validate_transcript(make({"start_s": 0.5, "end_s": 0.1}))
        with pytest.raises(schema.InvalidTranscriptError):
            schema.validate_transcript(make({"start_s": float("inf")}))


class TestTaskSpecValidation:
    @staticmethod
    def _picture_spec():
        return {
            "version": 1,
            "task": "picture_desc_1",
            "concept_aliases": {"con": ["con"]},
            "entity_groups": {"animals": ["con"]},
            "action_groups": {"motion": ["ch\u1ea1y"]},
        }

    def test_valid_picture_spec_passes(self):
        assert schema.validate_task_spec(self._picture_spec()) == 1

    def test_rejects_missing_version(self):
        spec = self._picture_spec()
        del spec["version"]
        with pytest.raises(schema.InvalidTaskSpecError):
            schema.validate_task_spec(spec)

    def test_rejects_unknown_version(self):
        spec = self._picture_spec()
        spec["version"] = 99
        with pytest.raises(schema.InvalidTaskSpecError):
            schema.validate_task_spec(spec)

    def test_rejects_missing_task_fields(self):
        spec = self._picture_spec()
        del spec["entity_groups"]
        with pytest.raises(schema.InvalidTaskSpecError):
            schema.validate_task_spec(spec)

    def test_phonemic_spec_requires_initials_and_exclusions(self):
        spec = {
            "version": 1,
            "task": "phonemic_fluency",
            "initials": ["b", "c"],
        }
        with pytest.raises(schema.InvalidTaskSpecError):
            schema.validate_task_spec(spec)

    def test_rejects_unknown_task_value_when_present(self):
        spec = {"version": 1, "task": "not_a_known_task"}
        with pytest.raises(schema.InvalidTaskSpecError):
            schema.validate_task_spec(spec)

    def test_allows_spec_without_optional_task_field(self):
        spec = {"version": 1}
        assert schema.validate_task_spec(spec) == 1


class TestStructuredErrorCodes:
    def test_each_error_has_machine_readable_code(self, tmp_path):
        assert schema.FeatureExtractionError("x").code == "FEATURE_EXTRACTION_ERROR"
        assert schema.InvalidManifestError("x").code == "INVALID_MANIFEST"
        assert schema.InvalidTranscriptError("x").code == "INVALID_TRANSCRIPT"
        assert schema.InvalidTaskSpecError("x").code == "INVALID_TASK_SPEC"
        assert schema.UnknownTaskError("x").code == "UNKNOWN_TASK"

    def test_missing_file_has_code(self, tmp_path):
        with pytest.raises(schema.MissingInputError) as e:
            schema.sha256_file(tmp_path / "nope.wav")
        assert e.value.code == "MISSING_INPUT"


class TestExtractionInputsSeparation:
    def _manifest(self, tmp_path):
        return {
            "version": 1,
            "rows": [
                {
                    "participant_id": "p-1",
                    "task": "picture_desc_1",
                    "recording_id": "rec-1",
                    "audio_id": "a-1",
                    "transcript_id": "tr-1",
                    "audio_path": str(tmp_path / "audio.wav"),
                    "transcript_path": str(tmp_path / "t.json"),
                    "task_spec_path": str(tmp_path / "spec.json"),
                    "diagnosis": "AD",
                    "age": 72,
                    "sex": "F",
                    "education_years": 12,
                }
            ],
        }

    def test_row_exposes_extraction_inputs(self, tmp_path):
        row = schema.validate_manifest(self._manifest(tmp_path))[0]
        inputs = row.inputs
        assert isinstance(inputs, schema.ExtractionInputs)
        assert inputs.audio_path == str(tmp_path / "audio.wav")
        assert inputs.transcript_path == str(tmp_path / "t.json")
        assert inputs.task_spec_path == str(tmp_path / "spec.json")
        assert schema.ExtractionInputs is not None  # only inputs, no diagnosis

    def test_extraction_inputs_is_immutable(self, tmp_path):
        row = schema.validate_manifest(self._manifest(tmp_path))[0]
        with pytest.raises(dataclasses.FrozenInstanceError):
            row.inputs.audio_path = "other.wav"


class TestStructuredErrors:
    def test_error_hierarchy(self):
        assert issubclass(schema.InvalidManifestError, schema.FeatureExtractionError)
        assert issubclass(schema.InvalidTranscriptError, schema.FeatureExtractionError)
        assert issubclass(schema.UnknownTaskError, schema.FeatureExtractionError)

    def test_error_has_message(self):
        err = schema.FeatureExtractionError("boom")
        assert str(err) == "boom"
