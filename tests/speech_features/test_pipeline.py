"""Behavioral tests for the offline extraction pipeline (Task 4).

Composes acoustic, linguistic-lexical, and external task-spec features into a
single immutable :class:`FeatureResult` with SHA-256 provenance. Uses only
stdlib ``wave`` (standard PCM) + ``scipy.signal.resample_poly`` for
resampling. Guarantees diagnosis never enters extractor calls or result
fields, and that a batch isolates each failing row behind a stable
``FeatureExtractionError.code`` instead of aborting.
"""

import dataclasses
import json
import math
import wave

import numpy as np
import pytest

import speech_features.pipeline as pipeline
from speech_features.schema import KNOWN_TASKS, FeatureResult, sha256_file


def _tone(freq, dur, sr=16000, amp=0.5):
    t = np.arange(int(round(sr * dur))) / sr
    return amp * np.sin(2 * math.pi * freq * t)


def _write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _write_wav(path, mono, sample_rate=16000, n_channels=1):
    mono = np.asarray(mono, dtype=float)
    if n_channels == 2:
        data = np.stack([mono, mono], axis=1).reshape(-1)
    else:
        data = mono.copy()
    pcm = np.clip(data, -1.0, 1.0) * 32767
    pcm = np.round(pcm).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.astype("<i2").tobytes())
    return str(path)


def _picture_spec():
    return {
        "version": 1,
        "task": "picture_desc_1",
        "concept_aliases": {"cat": ["con m\u00e8o"], "dog": ["ch\u00f3"]},
        "entity_groups": {"animals": ["con m\u00e8o", "ch\u00f3"]},
        "action_groups": {"motion": ["ch\u1ea1y"]},
    }


def _phonemic_spec():
    return {
        "version": 1,
        "task": "phonemic_fluency",
        "initials": ["c"],
        "exclusions": ["con g\u00ec"],
    }


def _transcript(words):
    return {
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
                "end_s": 2.5,
                "tokens": [{"kind": "word", "text": w} for w in words],
            },
        ],
    }


def _manifest(rows):
    return {"version": 1, "rows": rows}


class TestEndToEnd:
    def test_extract_recording_composes_all_feature_families(self, tmp_path):
        audio = _write_wav(tmp_path / "a.wav", _tone(145, 2.0), 22050)
        transcript = _write_json(tmp_path / "t.json", _transcript(["con m\u00e8o"]))
        spec = _write_json(tmp_path / "s.json", _picture_spec())

        res = pipeline.extract_recording(
            audio,
            transcript,
            spec,
            recording_id="rec-1",
            participant_id="p-1",
        )

        assert isinstance(res, FeatureResult)
        assert res.recording_id == "rec-1"
        assert res.participant_id == "p-1"
        assert res.task == "picture_desc_1"
        assert "ac_pitch_voiced_mean" in res.features
        assert "lex_word_count" in res.features
        assert "picture_concept_coverage" in res.features
        assert res.quality_flags == ()
        assert set(res.input_hashes) == {"audio", "transcript", "task_spec"}
        assert len(res.input_hashes["audio"]) == 64
        assert res.config.sample_rate == 16000

    def test_result_is_immutable(self, tmp_path):
        audio = _write_wav(tmp_path / "a.wav", _tone(145, 1.0))
        transcript = _write_json(tmp_path / "t.json", _transcript(["con m\u00e8o"]))
        spec = _write_json(tmp_path / "s.json", _picture_spec())
        res = pipeline.extract_recording(audio, transcript, spec)
        with pytest.raises(TypeError):
            res.features["x"] = 1.0

    def test_provenance_hashes_match_file_digests(self, tmp_path):
        audio = _write_wav(tmp_path / "a.wav", _tone(145, 1.0))
        transcript = _write_json(tmp_path / "t.json", _transcript(["con m\u00e8o"]))
        spec = _write_json(tmp_path / "s.json", _picture_spec())
        res = pipeline.extract_recording(audio, transcript, spec)
        assert res.input_hashes["audio"] == sha256_file(audio)
        assert res.input_hashes["transcript"] == sha256_file(transcript)
        assert res.input_hashes["task_spec"] == sha256_file(spec)


class TestWavReading:
    def test_missing_audio_raises_missing_input_code(self, tmp_path):
        transcript = _write_json(tmp_path / "t.json", _transcript(["con m\u00e8o"]))
        spec = _write_json(tmp_path / "s.json", _picture_spec())
        with pytest.raises(pipeline.MissingInputError) as e:
            pipeline.extract_recording(str(tmp_path / "missing.wav"), transcript, spec)
        assert e.value.code == "MISSING_INPUT"

    def test_malformed_wav_raises_invalid_audio_code(self, tmp_path):
        path = tmp_path / "bad.wav"
        path.write_bytes(b"NOTARIFFFILE")
        with pytest.raises(pipeline.InvalidAudioError) as e:
            pipeline.read_wav(path)
        assert e.value.code == "INVALID_AUDIO"

    def test_read_wav_resamples_to_target_rate(self, tmp_path):
        path = _write_wav(tmp_path / "a.wav", _tone(145, 1.0, 22050), 22050)
        samples = pipeline.read_wav(path, sample_rate=16000)
        assert samples.ndim == 1
        assert abs(len(samples) - 16000) < 200

    def test_stereo_wav_converts_to_mono(self, tmp_path):
        mono = _tone(145, 1.0)
        path = _write_wav(tmp_path / "s.wav", mono, 16000, n_channels=2)
        samples = pipeline.read_wav(path)
        assert samples.ndim == 1
        assert abs(len(samples) - 16000) < 200

    def test_read_is_finite_for_standard_pcm(self, tmp_path):
        # Standard integer PCM is always finite; the read must preserve that.
        path = _write_wav(tmp_path / "ok.wav", _tone(145, 1.0), 16000)
        samples = pipeline.read_wav(path, sample_rate=16000)
        assert bool(np.all(np.isfinite(samples)))

    def test_empty_audio_raises_invalid_audio(self, tmp_path):
        path = _write_wav(tmp_path / "empty.wav", np.zeros(0, dtype=float))
        with pytest.raises(pipeline.InvalidAudioError) as e:
            pipeline.read_wav(path, sample_rate=16000)
        assert e.value.code == "INVALID_AUDIO"


class TestLabelExclusion:
    def test_diagnosis_is_not_a_field_of_feature_result(self):
        field_names = [f.name for f in dataclasses.fields(FeatureResult)]
        assert "diagnosis" not in field_names

    def test_diagnosis_absent_from_features(self, tmp_path):
        audio = _write_wav(tmp_path / "a.wav", _tone(145, 1.0))
        transcript = _write_json(tmp_path / "t.json", _transcript(["con m\u00e8o"]))
        spec = _write_json(tmp_path / "s.json", _picture_spec())
        res = pipeline.extract_recording(audio, transcript, spec)
        assert "diagnosis" not in res.features

    def test_labels_kept_separate_in_batch(self, tmp_path):
        manifest = self._manifest(tmp_path)
        result = pipeline.extract_manifest(str(manifest))
        assert result.labels[("p-1", "picture_desc_1", "rec-1")] == "HC"
        recording = result.recordings[("p-1", "picture_desc_1", "rec-1")]
        assert "diagnosis" not in recording.features

    @staticmethod
    def _manifest(tmp_path):
        audio = _write_wav(tmp_path / "a.wav", _tone(145, 1.0))
        transcript = _write_json(tmp_path / "t.json", _transcript(["con m\u00e8o"]))
        spec = _write_json(tmp_path / "s.json", _picture_spec())
        manifest = _manifest(
            [
                {
                    "participant_id": "p-1",
                    "task": "picture_desc_1",
                    "recording_id": "rec-1",
                    "audio_id": "a-1",
                    "transcript_id": "tr-1",
                    "audio_path": audio,
                    "transcript_path": transcript,
                    "task_spec_path": spec,
                    "diagnosis": "HC",
                    "age": 68,
                    "sex": "M",
                    "education_years": 14,
                }
            ]
        )
        manifest_path = tmp_path / "manifest.json"
        return _write_json(manifest_path, manifest)


class TestFailureIsolation:
    def test_bad_row_is_isolated_not_crashing(self, tmp_path):
        good_audio = _write_wav(tmp_path / "good.wav", _tone(145, 1.0))
        good_tr = _write_json(tmp_path / "good.json", _transcript(["con m\u00e8o"]))
        good_spec = _write_json(tmp_path / "spec1.json", _picture_spec())
        bad_tr = _write_json(tmp_path / "bad.json", {"version": 1, "utterances": "nope"})
        phono_spec = _write_json(tmp_path / "spec2.json", _phonemic_spec())

        manifest = _manifest(
            [
                {
                    "participant_id": "p-1",
                    "task": "picture_desc_1",
                    "recording_id": "rec-1",
                    "audio_id": "a-1",
                    "transcript_id": "tr-1",
                    "audio_path": good_audio,
                    "transcript_path": good_tr,
                    "task_spec_path": good_spec,
                    "diagnosis": "AD",
                    "age": 72,
                    "sex": "F",
                    "education_years": 12,
                },
                {
                    "participant_id": "p-2",
                    "task": "phonemic_fluency",
                    "recording_id": "rec-2",
                    "audio_id": "a-2",
                    "transcript_id": "tr-2",
                    "audio_path": good_audio,
                    "transcript_path": bad_tr,
                    "task_spec_path": phono_spec,
                    "diagnosis": "HC",
                    "age": 66,
                    "sex": "M",
                    "education_years": 10,
                },
            ]
        )
        manifest_path = _write_json(tmp_path / "manifest.json", manifest)
        result = pipeline.extract_manifest(manifest_path)

        assert len(result.recordings) == 1
        assert len(result.failures) == 1
        failure = result.failures[0]
        assert failure.key == ("p-2", "phonemic_fluency", "rec-2")
        assert failure.code == "INVALID_TRANSCRIPT"
        # labels recorded for both rows, independent of extraction outcome
        assert result.labels[("p-1", "picture_desc_1", "rec-1")] == "AD"
        assert result.labels[("p-2", "phonemic_fluency", "rec-2")] == "HC"

    def test_missing_input_row_failure_code(self, tmp_path):
        good_audio = _write_wav(tmp_path / "good.wav", _tone(145, 1.0))
        good_tr = _write_json(tmp_path / "good.json", _transcript(["con m\u00e8o"]))
        good_spec = _write_json(tmp_path / "spec1.json", _picture_spec())
        manifest = _manifest(
            [
                {
                    "participant_id": "p-1",
                    "task": "picture_desc_1",
                    "recording_id": "rec-1",
                    "audio_id": "a-1",
                    "transcript_id": "tr-1",
                    "audio_path": good_audio,
                    "transcript_path": good_tr,
                    "task_spec_path": good_spec,
                    "diagnosis": "AD",
                    "age": 72,
                    "sex": "F",
                    "education_years": 12,
                },
                {
                    "participant_id": "p-2",
                    "task": "phonemic_fluency",
                    "recording_id": "rec-2",
                    "audio_id": "a-2",
                    "transcript_id": "tr-2",
                    "audio_path": str(tmp_path / "missing.wav"),
                    "transcript_path": str(tmp_path / "missing.json"),
                    "task_spec_path": str(tmp_path / "missing-spec.json"),
                    "diagnosis": "HC",
                    "age": 66,
                    "sex": "M",
                    "education_years": 10,
                },
            ]
        )
        manifest_path = _write_json(tmp_path / "manifest.json", manifest)
        result = pipeline.extract_manifest(manifest_path)
        assert result.failures[0].code == "MISSING_INPUT"


class TestReproducibility:
    def test_hashes_are_reproducible(self, tmp_path):
        audio = _write_wav(tmp_path / "a.wav", _tone(145, 1.0))
        transcript = _write_json(tmp_path / "t.json", _transcript(["con m\u00e8o"]))
        spec = _write_json(tmp_path / "s.json", _picture_spec())
        r1 = pipeline.extract_recording(audio, transcript, spec, recording_id="r")
        r2 = pipeline.extract_recording(audio, transcript, spec, recording_id="r")
        assert r1.input_hashes == r2.input_hashes
        assert r1.task == r2.task
        for k in ("ac_pitch_voiced_mean", "lex_word_count", "picture_concept_coverage"):
            assert r1.features[k] == r2.features[k]

    def test_every_known_task_has_a_scorer(self):
        assert set(pipeline._TASK_SCORER) == set(KNOWN_TASKS)
