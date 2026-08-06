import dataclasses
import hashlib

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
        with pytest.raises(schema.FeatureExtractionError):
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
        assert result[("audio", str(audio))] == hashlib.sha256(b"audio").hexdigest()
        assert result[("transcript", str(transcript))] == hashlib.sha256(b"transcript").hexdigest()
        assert result[("task_spec", str(task_spec))] == hashlib.sha256(b"spec").hexdigest()

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
        assert result[("task_spec", str(spec_a))] == hashlib.sha256(b"spec-a").hexdigest()
        assert result[("task_spec", str(spec_b))] == hashlib.sha256(b"spec-b").hexdigest()
        assert len(result) == 4  # audio + transcript shared + two distinct task specs

    def test_manifest_hashes_rejects_empty_rows(self):
        with pytest.raises(schema.InvalidManifestError):
            schema.manifest_hashes({"version": 1, "rows": []})


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
        utterances = schema.validate_transcript(transcript)
        assert len(utterances) == 2
        assert utterances[1].tokens[0].kind == "filler"

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


class TestStructuredErrors:
    def test_error_hierarchy(self):
        assert issubclass(schema.InvalidManifestError, schema.FeatureExtractionError)
        assert issubclass(schema.InvalidTranscriptError, schema.FeatureExtractionError)
        assert issubclass(schema.UnknownTaskError, schema.FeatureExtractionError)

    def test_error_has_message(self):
        err = schema.FeatureExtractionError("boom")
        assert str(err) == "boom"
