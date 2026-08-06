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
import pandas as pd
import pytest

import speech_features
import speech_features.pipeline as pipeline
from speech_features.document import load_document
from speech_features.result import (
    ExtractionError,
    FeatureBundle,
    InvalidConfigError,
    MissingAnnotationError,
    TargetSpeakerRequiredError,
    UnknownPackError,
)
from speech_features.schema import (
    KNOWN_TASKS,
    ExtractionConfig,
    FeatureResult,
    InvalidManifestError,
    sha256_file,
)


def _tone(freq, dur, sr=16000, amp=0.5):
    t = np.arange(int(round(sr * dur))) / sr
    return amp * np.sin(2 * math.pi * freq * t)


def _write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _to_pcm(mono, width):
    """Map a mono float array in [-1, 1] to a PCM byte payload of `width` bytes."""
    mono = np.clip(np.asarray(mono, dtype=float), -1.0, 1.0)
    if width == 1:  # unsigned 8-bit
        pcm = np.round(mono * 128.0 + 128.0).astype(np.uint8)
        return pcm.tobytes()
    if width == 2:
        pcm = np.round(mono * 32767.0).astype(np.int16)
        return pcm.astype("<i2").tobytes()
    if width == 3:  # signed 24-bit (little-endian, three bytes)
        pcm = np.round(mono * 8388607.0).astype(np.int32)
        lo = (pcm & 0xFF).astype(np.uint8)
        mid = ((pcm >> 8) & 0xFF).astype(np.uint8)
        hi = ((pcm >> 16) & 0xFF).astype(np.uint8)
        return np.stack([lo, mid, hi], axis=1).reshape(-1).tobytes()
    pcm = np.round(mono * 2147483647.0).astype(np.int32)
    return pcm.astype("<i4").tobytes()


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


def _write_wav_width(path, mono, width, sample_rate=16000, n_channels=1):
    """Write a mono or stereo WAV at an arbitrary integer PCM width."""
    mono = np.asarray(mono, dtype=float)
    if n_channels == 2:
        data = np.stack([mono, mono], axis=1).reshape(-1)
    else:
        data = mono.copy()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(width)
        wf.setframerate(sample_rate)
        wf.writeframes(_to_pcm(data, width))
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


class TestPcmWidthDecoding:
    """Agy finding: 24-bit PCM must decode via NumPy vector ops; 8/24-bit decode must be verified."""

    def test_8bit_unsigned_pcm_decodes_correctly(self, tmp_path):
        values = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=float)
        path = _write_wav_width(tmp_path / "u8.wav", values, width=1)
        samples = pipeline.read_wav(path, sample_rate=16000)
        expect = (
            np.round(values * 128.0 + 128.0).astype(np.uint8).astype(np.float64) - 128.0
        ) / 128.0
        assert samples.shape == values.shape
        np.testing.assert_allclose(samples, expect, atol=1e-4)

    def test_24bit_signed_pcm_decodes_correctly(self, tmp_path):
        values = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=float)
        path = _write_wav_width(tmp_path / "s24.wav", values, width=3)
        samples = pipeline.read_wav(path, sample_rate=16000)
        expect = np.round(values * 8388607.0) / 8388608.0
        assert samples.shape == values.shape
        np.testing.assert_allclose(samples, expect, atol=1e-4)

    def test_24bit_signed_pcm_round_trips_a_tone(self, tmp_path):
        mono = _tone(145, 1.0)
        path = _write_wav_width(tmp_path / "tone24.wav", mono, width=3)
        samples = pipeline.read_wav(path, sample_rate=16000)
        assert samples.ndim == 1
        assert abs(len(samples) - 16000) < 200
        assert bool(np.all(np.isfinite(samples)))
        assert float(np.max(np.abs(samples))) > 0.1

    def test_24bit_stereo_downmixes_to_mono(self, tmp_path):
        mono = _tone(145, 1.0)
        path = _write_wav_width(tmp_path / "s24.wav", mono, width=3, n_channels=2)
        samples = pipeline.read_wav(path, sample_rate=16000)
        assert samples.ndim == 1
        assert bool(np.all(np.isfinite(samples)))


class TestInvalidAudioIsolation:
    """Agy findings: decode/downmix/resample must raise InvalidAudioError; a bad row is isolated."""

    def test_truncated_wav_raises_invalid_audio(self, tmp_path):
        path = tmp_path / "t.wav"
        _write_wav_width(path, _tone(145, 1.0), width=3)
        path.write_bytes(path.read_bytes()[:-1])  # strip one byte: payload not a multiple of 3
        with pytest.raises(pipeline.InvalidAudioError) as e:
            pipeline.read_wav(path, sample_rate=16000)
        assert e.value.code == "INVALID_AUDIO"

    def test_truncated_wav_is_isolated_as_batch_failure(self, tmp_path):
        good_audio = _write_wav(tmp_path / "good.wav", _tone(145, 1.0))
        good_tr = _write_json(tmp_path / "good.json", _transcript(["con m\u00e8o"]))
        good_spec = _write_json(tmp_path / "spec1.json", _picture_spec())
        trunc = tmp_path / "trunc.wav"
        _write_wav_width(trunc, _tone(145, 1.0), width=3)
        trunc.write_bytes(trunc.read_bytes()[:-1])  # corrupt payload: not a multiple of 3
        bad_tr = _write_json(tmp_path / "tr.json", _transcript(["con m\u00e8o"]))
        bad_spec = _write_json(tmp_path / "spec2.json", _picture_spec())

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
                    "task": "picture_desc_1",
                    "recording_id": "rec-2",
                    "audio_id": "a-2",
                    "transcript_id": "tr-2",
                    "audio_path": str(trunc),
                    "transcript_path": bad_tr,
                    "task_spec_path": bad_spec,
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
        assert result.failures[0].key == ("p-2", "picture_desc_1", "rec-2")
        assert result.failures[0].code == "INVALID_AUDIO"

    def test_read_wav_output_is_finite_after_resample(self, tmp_path):
        path = _write_wav(tmp_path / "a.wav", _tone(145, 1.0, 22050), 22050)
        samples = pipeline.read_wav(path, sample_rate=16000)
        assert bool(np.all(np.isfinite(samples)))
        assert abs(len(samples) - 16000) < 200


class TestGenericRowFallback:
    """Agy finding: any per-row exception becomes a BatchFailure, never aborts the batch."""

    def test_arbitrary_row_exception_becomes_batch_failure(self, tmp_path, monkeypatch):
        good_audio = _write_wav(tmp_path / "good.wav", _tone(145, 1.0))
        good_tr = _write_json(tmp_path / "good.json", _transcript(["con m\u00e8o"]))
        good_spec = _write_json(tmp_path / "spec1.json", _picture_spec())
        bad_tr = _write_json(tmp_path / "bad.json", _transcript(["con m\u00e8o"]))
        phono_spec = _write_json(tmp_path / "spec2.json", _phonemic_spec())

        def _boom(transcript, spec):
            raise RuntimeError("arbitrary scorer failure")

        monkeypatch.setitem(pipeline._TASK_SCORER, "phonemic_fluency", _boom)

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
        assert failure.error_type == "RuntimeError"


def _document_json(
    document_id="doc-1",
    speakers=("PAR",),
    utterance_speaker="PAR",
    words=("con", "m\u00e8o"),
):
    """Hand-built synthetic JSON v2 speech document with one speaker."""
    return {
        "version": 2,
        "document_id": document_id,
        "language": "vie",
        "media": [],
        "speakers": [{"id": speaker, "name": "", "role": "participant"} for speaker in speakers],
        "utterances": (
            [
                {
                    "id": "u1",
                    "speaker_id": utterance_speaker,
                    "start_s": 0.5,
                    "end_s": 2.0,
                    "tokens": [
                        {"id": f"u1_t{i + 1:04d}", "text": word, "kind": "word"}
                        for i, word in enumerate(words)
                    ],
                }
            ]
            if words is not None
            else []
        ),
        "annotations": [],
        "raw_tiers": {},
    }


def _document(tmp_path, name="doc.json", **kwargs):
    return load_document(_write_json(tmp_path / name, _document_json(**kwargs)))


class _Warning:
    """Minimal stand-in for a structured document warning entry."""

    def __init__(self, code, tier, line):
        self.code = code
        self.tier = tier
        self.line = line


def _write_manifest(tmp_path, rows, *, name="manifest.json", version=2):
    path = tmp_path / name
    _write_json(path, {"version": version, "rows": rows})
    return str(path)


def _manifest_row(
    tmp_path,
    recording_id,
    *,
    audio="a.wav",
    transcript="t.json",
    targets=None,
    speaker="PAR",
    words=("con", "m\u00e8o"),
):
    """Write one good manifest row's files (relative to tmp_path) and return the row."""
    if not (tmp_path / audio).exists():
        _write_wav(tmp_path / audio, _tone(145, 1.0))
    if not (tmp_path / transcript).exists():
        _write_json(
            tmp_path / transcript,
            _document_json(
                document_id=recording_id,
                speakers=(speaker,),
                utterance_speaker=speaker,
                words=words,
            ),
        )
    row = {"recording_id": recording_id, "audio_path": audio, "transcript_path": transcript}
    if targets is not None:
        row["target_speakers"] = targets
    return row


class TestLabelFreeExtraction:
    """Task 10: label-free `extract` and `extract_batch` over SpeechDocument."""

    def test_package_root_exports_extract_and_extract_batch(self):
        assert callable(speech_features.extract)
        assert callable(speech_features.extract_batch)

    def test_extract_default_schema_has_exact_columns(self, tmp_path):
        audio = _write_wav(tmp_path / "a.wav", _tone(145, 1.0))
        doc = _document(tmp_path)
        bundle = speech_features.extract(audio, doc)
        assert isinstance(bundle, FeatureBundle)
        assert list(bundle.recordings.columns[:2]) == ["recording_id", "speaker_id"]
        assert list(bundle.utterances.columns[:5]) == [
            "recording_id",
            "speaker_id",
            "utterance_id",
            "start_s",
            "end_s",
        ]
        assert list(bundle.issues.columns) == [
            "recording_id",
            "speaker_id",
            "utterance_id",
            "feature",
            "code",
            "severity",
            "message",
        ]
        assert len(bundle.recordings.columns) == 2 + 166
        assert len(bundle.utterances.columns) == 5 + 4
        assert bundle.recordings.iloc[0]["recording_id"] == "doc-1"
        assert bundle.recordings.iloc[0]["speaker_id"] == "PAR"
        assert len(bundle.recordings) == 1
        assert len(bundle.utterances) == 1

    def test_extract_default_columns_match_registered_catalog(self, tmp_path):
        from speech_features.features.acoustic.definitions import ALL_KEYS as acoustic_keys
        from speech_features.features.linguistic.definitions import (
            ALL_KEYS as lexical_keys,
            DISCOURSE_UTTERANCE_KEYS,
            TASK9_RECORDING_KEYS,
        )

        bundle = speech_features.extract(
            _write_wav(tmp_path / "a.wav", _tone(145, 1.0)), _document(tmp_path)
        )
        assert len(acoustic_keys) == 73
        assert len(lexical_keys) + len(TASK9_RECORDING_KEYS) == 93
        assert len(DISCOURSE_UTTERANCE_KEYS) == 4
        assert list(bundle.recordings.columns[2:]) == sorted(
            acoustic_keys + lexical_keys + TASK9_RECORDING_KEYS
        )
        assert list(bundle.utterances.columns[5:]) == sorted(DISCOURSE_UTTERANCE_KEYS)

    def test_pack_filtering_selects_catalog_columns(self, tmp_path):
        from speech_features.features.acoustic.definitions import ALL_KEYS as acoustic_keys
        from speech_features.features.linguistic.definitions import (
            ALL_KEYS as lexical_keys,
            DISCOURSE_UTTERANCE_KEYS,
            TASK9_RECORDING_KEYS,
        )

        audio = _write_wav(tmp_path / "a.wav", _tone(145, 1.0))
        doc = _document(tmp_path)
        neuro = speech_features.extract(audio, doc, packs=("adult_neuro",))
        assert list(neuro.recordings.columns[2:]) == sorted(lexical_keys + TASK9_RECORDING_KEYS)
        assert list(neuro.utterances.columns[5:]) == sorted(DISCOURSE_UTTERANCE_KEYS)
        acoustic = speech_features.extract(audio, doc, packs=("acoustic",))
        assert list(acoustic.recordings.columns[2:]) == sorted(acoustic_keys)
        assert list(acoustic.utterances.columns) == [
            "recording_id",
            "speaker_id",
            "utterance_id",
            "start_s",
            "end_s",
        ]
        assert len(acoustic.utterances) == 0

    def test_level_filtering_empties_excluded_table(self, tmp_path):
        audio = _write_wav(tmp_path / "a.wav", _tone(145, 1.0))
        doc = _document(tmp_path)
        recording_only = speech_features.extract(audio, doc, levels=("recording",))
        assert list(recording_only.utterances.columns) == [
            "recording_id",
            "speaker_id",
            "utterance_id",
            "start_s",
            "end_s",
        ]
        assert len(recording_only.utterances) == 0
        utterance_keys = (
            "discourse_turn_word_count",
            "discourse_turn_syllable_count",
            "discourse_response_latency_s",
            "discourse_turn_overlap_s",
        )
        assert not any(
            r.feature in utterance_keys for r in recording_only.issues.itertuples(index=False)
        )
        utterance_only = speech_features.extract(audio, doc, levels=("utterance",))
        assert list(utterance_only.recordings.columns) == ["recording_id", "speaker_id"]
        assert len(utterance_only.recordings) == 0
        assert len(utterance_only.utterances) == 1

    def test_utterance_rows_sorted_chronologically_not_by_id(self, tmp_path):
        audio = _write_wav(tmp_path / "a.wav", _tone(145, 1.0))
        data = _document_json()
        data["utterances"] = [
            {
                "id": "u2",
                "speaker_id": "PAR",
                "start_s": 0.0,
                "end_s": 1.0,
                "tokens": [{"id": "u2_t0001", "text": "con", "kind": "word"}],
            },
            {
                "id": "u10",
                "speaker_id": "PAR",
                "start_s": 1.5,
                "end_s": 2.5,
                "tokens": [{"id": "u10_t0001", "text": "m\u00e8o", "kind": "word"}],
            },
        ]
        doc = load_document(_write_json(tmp_path / "doc.json", data))
        bundle = speech_features.extract(audio, doc, packs=("adult_neuro",), levels=("utterance",))
        assert list(bundle.utterances["utterance_id"]) == ["u2", "u10"]
        assert list(bundle.utterances["start_s"]) == [0.0, 1.5]

    def test_pack_order_cannot_change_columns(self, tmp_path):
        audio = _write_wav(tmp_path / "a.wav", _tone(145, 1.0))
        doc = _document(tmp_path)
        forward = speech_features.extract(audio, doc, packs=("acoustic", "adult_neuro"))
        reverse = speech_features.extract(audio, doc, packs=("adult_neuro", "acoustic"))
        assert list(forward.recordings.columns) == list(reverse.recordings.columns)
        assert list(forward.provenance["packs"]) == ["acoustic", "adult_neuro"]
        assert list(reverse.provenance["packs"]) == ["acoustic", "adult_neuro"]

    def test_acoustic_audio_read_exactly_once(self, tmp_path, monkeypatch):
        import speech_features.extraction as extraction

        calls = []
        original = extraction._read_wav_with_width

        def counting(path, sample_rate=16000):
            calls.append(str(path))
            return original(path, sample_rate=sample_rate)

        monkeypatch.setattr(extraction, "_read_wav_with_width", counting)
        speech_features.extract(
            _write_wav(tmp_path / "a.wav", _tone(145, 1.0)), _document(tmp_path)
        )
        assert len(calls) == 1

    def test_extract_without_acoustic_never_reads_audio(self, tmp_path, monkeypatch):
        import speech_features.extraction as extraction

        def boom(path, sample_rate=16000):
            raise AssertionError("audio must not be read without the acoustic pack")

        monkeypatch.setattr(extraction, "_read_wav_with_width", boom)
        bundle = speech_features.extract(
            _write_wav(tmp_path / "a.wav", _tone(145, 1.0)),
            _document(tmp_path),
            packs=("adult_neuro",),
        )
        assert len(bundle.recordings) == 1

    def test_missing_audio_raises_missing_input_before_extraction(self, tmp_path):
        with pytest.raises(pipeline.MissingInputError) as exc:
            speech_features.extract(str(tmp_path / "missing.wav"), _document(tmp_path))
        assert exc.value.code == "MISSING_INPUT"

    def test_target_resolution_rules(self, tmp_path):
        audio = _write_wav(tmp_path / "a.wav", _tone(145, 1.0))
        single = _document(tmp_path)
        bundle = speech_features.extract(audio, single, target_speaker="PAR")
        assert bundle.recordings.iloc[0]["speaker_id"] == "PAR"
        multi = _document(tmp_path, speakers=("PAR", "EXA"), utterance_speaker="PAR")
        with pytest.raises(TargetSpeakerRequiredError):
            speech_features.extract(audio, multi)
        with pytest.raises(InvalidConfigError) as exc:
            speech_features.extract(audio, multi, target_speaker="NOPE")
        assert exc.value.code == "INVALID_CONFIG"
        bundle = speech_features.extract(audio, multi, target_speaker="EXA", packs=("adult_neuro",))
        assert bundle.recordings.iloc[0]["speaker_id"] == "EXA"

    def test_missing_alignment_raises_missing_annotation(self, tmp_path):
        doc = _document(tmp_path, words=None)
        with pytest.raises(MissingAnnotationError) as exc:
            speech_features.extract(_write_wav(tmp_path / "a.wav", _tone(145, 1.0)), doc)
        assert exc.value.code == "MISSING_ANNOTATION"

    def test_empty_document_resolves_empty_speaker_id(self, tmp_path):
        bundle = speech_features.extract(
            _write_wav(tmp_path / "a.wav", _tone(145, 1.0)),
            _document(tmp_path, words=None, speakers=()),
            packs=("adult_neuro",),
        )
        assert bundle.recordings.iloc[0]["speaker_id"] == ""
        assert bundle.provenance["target_speakers"] == [""]

    def test_non_document_input_rejected(self, tmp_path):
        with pytest.raises(InvalidConfigError) as exc:
            speech_features.extract(_write_wav(tmp_path / "a.wav", _tone(145, 1.0)), {"x": 1})
        assert exc.value.code == "INVALID_CONFIG"

    def test_invalid_pack_selections(self, tmp_path):
        audio = _write_wav(tmp_path / "a.wav", _tone(145, 1.0))
        doc = _document(tmp_path)
        with pytest.raises(UnknownPackError) as exc:
            speech_features.extract(audio, doc, packs=("bogus",))
        assert exc.value.code == "UNKNOWN_PACK"
        for bad in (("acoustic", "acoustic"), (), "acoustic", ("acoustic", 5)):
            with pytest.raises((UnknownPackError, InvalidConfigError)):
                speech_features.extract(audio, doc, packs=bad)

    def test_invalid_level_selections(self, tmp_path):
        audio = _write_wav(tmp_path / "a.wav", _tone(145, 1.0))
        doc = _document(tmp_path)
        for bad in (
            ("recording", "utterance", "utterance"),
            (),
            "recording",
            ("recording", "bogus"),
        ):
            with pytest.raises(InvalidConfigError):
                speech_features.extract(audio, doc, levels=bad)

    def test_selection_validation_trust_boundary(self, tmp_path):
        audio = _write_wav(tmp_path / "a.wav", _tone(145, 1.0))
        doc = _document(tmp_path)
        for bad in (
            {"acoustic"},
            frozenset({"acoustic"}),
            (["acoustic"],),
            [["acoustic"]],
            ({"acoustic", "adult_neuro"}),
            ("acoustic", ["acoustic"]),
        ):
            with pytest.raises(InvalidConfigError):
                speech_features.extract(audio, doc, packs=bad)
        for bad in ({"recording"}, frozenset({"utterance"}), (["recording"],), [["recording"]]):
            with pytest.raises(InvalidConfigError):
                speech_features.extract(audio, doc, levels=bad)
        with pytest.raises(InvalidConfigError):
            speech_features.extract(audio, doc, target_speaker=["PAR"])
        with pytest.raises(InvalidConfigError):
            speech_features.extract(audio, doc, target_speaker=object())

    def test_invalid_config_object_rejected(self, tmp_path):
        with pytest.raises(InvalidConfigError) as exc:
            speech_features.extract(
                _write_wav(tmp_path / "a.wav", _tone(145, 1.0)),
                _document(tmp_path),
                config=object(),
            )
        assert exc.value.code == "INVALID_CONFIG"

    def test_document_warnings_become_warning_issues(self, tmp_path):
        import dataclasses

        doc = dataclasses.replace(
            _document(tmp_path),
            warnings=(_Warning("UNSUPPORTED_CHAT_TIER", "%xspa", "*PAR: some tier"),),
        )
        bundle = speech_features.extract(_write_wav(tmp_path / "a.wav", _tone(145, 1.0)), doc)
        assert len(bundle.recordings) == 1
        matches = [
            r for r in bundle.issues.itertuples(index=False) if r.code == "UNSUPPORTED_CHAT_TIER"
        ]
        assert len(matches) == 1
        issue = matches[0]
        assert issue.severity == "warning"
        assert issue.speaker_id == "PAR"
        assert "%xspa" in issue.message
        assert "*PAR: some tier" in issue.message
        assert pd.isna(issue.feature)
        assert pd.isna(issue.utterance_id)

    def test_nan_preserved_and_issues_name_nan_features(self, tmp_path):
        bundle = speech_features.extract(
            _write_wav(tmp_path / "a.wav", _tone(145, 1.0)),
            _document(tmp_path, words=None),
            packs=("adult_neuro",),
        )
        row = bundle.recordings.iloc[0]
        assert pd.isna(row["lex_token_count"])
        values = {column: row[column] for column in bundle.recordings.columns[2:]}
        for issue in bundle.issues.itertuples(index=False):
            if issue.severity == "warning" and issue.feature in values:
                assert pd.isna(values[issue.feature]), issue.feature

    def test_provenance_contract(self, tmp_path):
        audio = _write_wav(tmp_path / "a.wav", _tone(145, 1.0))
        doc = _document(tmp_path)
        bundle = speech_features.extract(audio, doc)
        prov = dict(bundle.provenance)
        assert prov["package_version"] == "0.2.0"
        assert prov["catalog_version"] == 1
        assert prov["packs"] == ["acoustic", "adult_neuro"]
        assert prov["levels"] == ["recording", "utterance"]
        assert prov["config"] == dataclasses.asdict(ExtractionConfig())
        assert prov["target_speakers"] == ["PAR"]
        assert prov["audio_sha256"] == sha256_file(audio)
        assert prov["transcript_hash_kind"] == "canonical_document"
        assert prov["annotation_sources"] == []

    def test_provenance_canonical_transcript_hash(self, tmp_path):
        import hashlib

        from speech_features.formats.json import encode_json

        doc = _document(tmp_path)
        bundle = speech_features.extract(_write_wav(tmp_path / "a.wav", _tone(145, 1.0)), doc)
        expected = hashlib.sha256(encode_json(doc).encode("utf-8")).hexdigest()
        assert bundle.provenance["transcript_sha256"] == expected

    def test_provenance_uses_source_sha256_when_available(self, tmp_path):
        import dataclasses

        doc = dataclasses.replace(_document(tmp_path), source_sha256="a" * 64)
        bundle = speech_features.extract(_write_wav(tmp_path / "a.wav", _tone(145, 1.0)), doc)
        assert bundle.provenance["transcript_sha256"] == "a" * 64
        assert bundle.provenance["transcript_hash_kind"] == "source"

    def test_provenance_annotation_sources_sorted(self, tmp_path):
        data = _document_json()
        data["annotations"] = [
            {
                "layer": "upos",
                "source": "udpipe",
                "confidence": 0.8,
                "values": {"u1_t0001": "NOUN", "u1_t0002": "NOUN"},
            },
            {
                "layer": "lemma",
                "source": "spacy",
                "confidence": 0.9,
                "values": {"u1_t0001": "con", "u1_t0002": "m\u00e8o"},
            },
        ]
        doc = load_document(_write_json(tmp_path / "doc.json", data))
        bundle = speech_features.extract(_write_wav(tmp_path / "a.wav", _tone(145, 1.0)), doc)
        assert bundle.provenance["annotation_sources"] == [
            {"layer": "lemma", "source": "spacy", "confidence": 0.9},
            {"layer": "upos", "source": "udpipe", "confidence": 0.8},
        ]

    def test_no_clinical_fields_anywhere(self, tmp_path):
        bundle = speech_features.extract(
            _write_wav(tmp_path / "a.wav", _tone(145, 1.0)), _document(tmp_path)
        )
        columns = (
            set(bundle.recordings.columns)
            | set(bundle.utterances.columns)
            | set(bundle.issues.columns)
        )
        for banned in (
            "diagnosis",
            "label",
            "task",
            "age",
            "sex",
            "education_years",
            "participant_id",
        ):
            assert banned not in columns
        for banned in ("diagnosis", "label", "task", "age", "sex"):
            assert banned not in {key.lower() for key in bundle.provenance}
            assert banned not in {key.lower() for key in bundle.provenance["config"]}

    def test_single_unexpected_failure_becomes_extraction_error(self, tmp_path, monkeypatch):
        import speech_features.extraction as extraction

        def boom(*args, **kwargs):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(extraction, "_extract_acoustic_pack", boom)
        with pytest.raises(ExtractionError) as exc:
            speech_features.extract(
                _write_wav(tmp_path / "a.wav", _tone(145, 1.0)), _document(tmp_path)
            )
        assert exc.value.code == "EXTRACTION_ERROR"


class TestLabelFreeBatch:
    """Task 10: manifest v2 and `extract_batch` isolation/provenance."""

    def test_batch_returns_deterministic_schema_and_provenance(self, tmp_path):
        rows = [
            _manifest_row(tmp_path, "r2", audio="a2.wav", transcript="t2.json"),
            _manifest_row(tmp_path, "r1", audio="a1.wav", transcript="t1.json"),
        ]
        manifest = _write_manifest(tmp_path, rows)
        bundle = speech_features.extract_batch(manifest)
        assert list(bundle.recordings["recording_id"]) == ["r1", "r2"]
        assert len(bundle.recordings.columns) == 2 + 166
        assert len(bundle.utterances.columns) == 5 + 4
        prov = dict(bundle.provenance)
        assert prov["manifest_sha256"] == sha256_file(manifest)
        assert prov["packs"] == ["acoustic", "adult_neuro"]
        assert prov["levels"] == ["recording", "utterance"]
        assert prov["counts"] == {"total": 2, "success": 2, "failure": 0}
        assert prov["recordings"]["r1"]["audio_sha256"] == sha256_file(tmp_path / "a1.wav")
        assert prov["recordings"]["r1"]["transcript_sha256"] == sha256_file(tmp_path / "t1.json")
        assert prov["recordings"]["r1"]["target_speakers"] == ["PAR"]

    def test_batch_relative_paths_resolve_against_manifest(self, tmp_path, monkeypatch):
        manifest_dir = tmp_path / "manifests"
        manifest_dir.mkdir()
        audio = manifest_dir / "audio" / "r1.wav"
        transcript = manifest_dir / "transcripts" / "r1.json"
        audio.parent.mkdir()
        transcript.parent.mkdir()
        _write_wav(audio, _tone(145, 1.0))
        _write_json(transcript, _document_json(document_id="r1"))
        monkeypatch.chdir(tmp_path)
        manifest = _write_manifest(
            manifest_dir,
            [
                {
                    "recording_id": "r1",
                    "audio_path": "audio/r1.wav",
                    "transcript_path": "transcripts/r1.json",
                }
            ],
        )
        bundle = speech_features.extract_batch(manifest)
        assert len(bundle.recordings) == 1
        assert bundle.recordings.iloc[0]["recording_id"] == "r1"

    def test_batch_rejects_invalid_manifests(self, tmp_path):
        good = _manifest_row(tmp_path, "r1")
        manifest = _write_manifest(tmp_path, [good], version=1)
        with pytest.raises(InvalidManifestError):
            speech_features.extract_batch(manifest)
        bad_rows = [
            None,
            [],
            [{"version": 2}],
            [[1, 2]],
            [{**good, "diagnosis": "AD"}],
            [{**good, "recording_id": ""}],
            [{**good, "audio_path": ""}],
            [{**good, "target_speakers": []}],
            [{**good, "target_speakers": ["PAR", "PAR"]}],
            [{**good, "target_speakers": [""]}],
            [{**good, "target_speakers": [["PAR"]]}],
        ]
        for rows in bad_rows:
            with pytest.raises(InvalidManifestError):
                speech_features.extract_batch(_write_manifest(tmp_path, rows))
        dup = _write_manifest(
            tmp_path, [_manifest_row(tmp_path, "r1"), _manifest_row(tmp_path, "r1")]
        )
        with pytest.raises(InvalidManifestError):
            speech_features.extract_batch(dup)

    def test_batch_row_failure_isolation(self, tmp_path):
        good = _manifest_row(tmp_path, "r1", audio="a1.wav", transcript="t1.json")
        bad = _manifest_row(tmp_path, "r2", audio="a2.wav", transcript="missing.json")
        (tmp_path / "missing.json").unlink()
        bundle = speech_features.extract_batch(_write_manifest(tmp_path, [good, bad]))
        assert list(bundle.recordings["recording_id"]) == ["r1"]
        errors = {
            (r.recording_id, r.speaker_id, r.code)
            for r in bundle.issues[bundle.issues["severity"] == "error"].itertuples(index=False)
        }
        assert ("r2", "", "MISSING_INPUT") in errors
        assert bundle.provenance["counts"] == {"total": 2, "success": 1, "failure": 1}
        entry = bundle.provenance["recordings"]["r2"]
        assert entry["error_code"] == "MISSING_INPUT"
        assert entry["audio_sha256"] == sha256_file(tmp_path / "a2.wav")
        assert "transcript_sha256" not in entry

    def test_batch_missing_audio_keeps_available_transcript_hash(self, tmp_path):
        row = _manifest_row(tmp_path, "r1", audio="missing.wav", transcript="t1.json")
        (tmp_path / "missing.wav").unlink()
        bundle = speech_features.extract_batch(_write_manifest(tmp_path, [row]))
        assert len(bundle.recordings) == 0
        errors = {
            (r.recording_id, r.speaker_id, r.code)
            for r in bundle.issues[bundle.issues["severity"] == "error"].itertuples(index=False)
        }
        assert ("r1", "", "MISSING_INPUT") in errors
        entry = bundle.provenance["recordings"]["r1"]
        assert entry["error_code"] == "MISSING_INPUT"
        assert entry["transcript_sha256"] == sha256_file(tmp_path / "t1.json")
        assert "audio_sha256" not in entry

    def test_batch_invalid_transcript_keeps_both_file_hashes(self, tmp_path):
        row = _manifest_row(tmp_path, "r1", audio="a1.wav", transcript="bad.json")
        _write_json(tmp_path / "bad.json", {"version": 2, "document_id": 5})
        bundle = speech_features.extract_batch(_write_manifest(tmp_path, [row]))
        errors = {
            (r.recording_id, r.speaker_id, r.code)
            for r in bundle.issues[bundle.issues["severity"] == "error"].itertuples(index=False)
        }
        assert ("r1", "", "INVALID_DOCUMENT") in errors
        entry = bundle.provenance["recordings"]["r1"]
        assert entry["error_code"] == "INVALID_DOCUMENT"
        assert entry["audio_sha256"] == sha256_file(tmp_path / "a1.wav")
        assert entry["transcript_sha256"] == sha256_file(tmp_path / "bad.json")

    def test_batch_malformed_document_is_isolated(self, tmp_path):
        good = _manifest_row(tmp_path, "r1", audio="a1.wav", transcript="t1.json")
        bad = _manifest_row(tmp_path, "r2", audio="a2.wav", transcript="bad.json")
        _write_json(tmp_path / "bad.json", {"version": 2, "document_id": 5})
        bundle = speech_features.extract_batch(_write_manifest(tmp_path, [good, bad]))
        assert list(bundle.recordings["recording_id"]) == ["r1"]
        errors = {
            (r.recording_id, r.speaker_id, r.code)
            for r in bundle.issues[bundle.issues["severity"] == "error"].itertuples(index=False)
        }
        assert ("r2", "", "INVALID_DOCUMENT") in errors
        assert bundle.provenance["counts"] == {"total": 2, "success": 1, "failure": 1}

    def test_batch_unexpected_loader_exception_is_isolated(self, tmp_path, monkeypatch):
        import speech_features.extraction as extraction

        good = _manifest_row(tmp_path, "r1", audio="a1.wav", transcript="t1.json")
        bad = _manifest_row(tmp_path, "r2", audio="a2.wav", transcript="t2.json")
        original = extraction.load_document

        def boom(path):
            if "t2" in str(path):
                raise RuntimeError("loader exploded")
            return original(path)

        monkeypatch.setattr(extraction, "load_document", boom)
        bundle = speech_features.extract_batch(_write_manifest(tmp_path, [good, bad]))
        assert list(bundle.recordings["recording_id"]) == ["r1"]
        errors = {
            (r.recording_id, r.speaker_id, r.code)
            for r in bundle.issues[bundle.issues["severity"] == "error"].itertuples(index=False)
        }
        assert ("r2", "", "EXTRACTION_ERROR") in errors
        assert bundle.provenance["counts"] == {"total": 2, "success": 1, "failure": 1}
        assert bundle.provenance["recordings"]["r2"]["error_code"] == "EXTRACTION_ERROR"

    def test_batch_missing_alignment_is_isolated(self, tmp_path):
        row = _manifest_row(tmp_path, "r1", transcript="t1.json")
        _write_json(tmp_path / "t1.json", _document_json(words=None))
        bundle = speech_features.extract_batch(_write_manifest(tmp_path, [row]))
        errors = {
            (r.recording_id, r.speaker_id, r.code)
            for r in bundle.issues[bundle.issues["severity"] == "error"].itertuples(index=False)
        }
        assert ("r1", "PAR", "MISSING_ANNOTATION") in errors
        assert bundle.provenance["counts"] == {"total": 1, "success": 0, "failure": 1}
        assert len(bundle.recordings) == 0

    def test_batch_issues_preserve_nulls(self, tmp_path):
        row = _manifest_row(tmp_path, "r1", transcript="missing.json")
        (tmp_path / "missing.json").unlink()
        bundle = speech_features.extract_batch(_write_manifest(tmp_path, [row]))
        issue = bundle.issues[bundle.issues["code"] == "MISSING_INPUT"].iloc[0]
        assert pd.isna(issue["utterance_id"])
        assert pd.isna(issue["feature"])
        assert issue["severity"] == "error"
        rows = [
            tuple("" if pd.isna(value) else value for value in r)
            for r in bundle.issues.itertuples(index=False)
        ]
        assert rows == sorted(rows)

    def test_batch_target_failure_isolation(self, tmp_path):
        row = _manifest_row(tmp_path, "r1", targets=["PAR", "NOPE"])
        bundle = speech_features.extract_batch(_write_manifest(tmp_path, [row]))
        assert list(bundle.recordings["speaker_id"]) == ["PAR"]
        errors = {
            (r.recording_id, r.speaker_id, r.code)
            for r in bundle.issues[bundle.issues["severity"] == "error"].itertuples(index=False)
        }
        assert ("r1", "NOPE", "INVALID_CONFIG") in errors
        assert bundle.provenance["counts"] == {"total": 2, "success": 1, "failure": 1}
        assert bundle.provenance["recordings"]["r1"]["target_speakers"] == ["PAR", "NOPE"]

    def test_batch_multi_target_shares_audio_load(self, tmp_path, monkeypatch):
        import speech_features.extraction as extraction

        calls = []
        original = extraction._read_wav_with_width

        def counting(path, sample_rate=16000):
            calls.append(str(path))
            return original(path, sample_rate=sample_rate)

        monkeypatch.setattr(extraction, "_read_wav_with_width", counting)
        bundle = speech_features.extract_batch(
            _write_manifest(tmp_path, [_manifest_row(tmp_path, "r1", targets=["PAR", "PAR2"])])
        )
        assert len(calls) == 1
        assert list(bundle.recordings["speaker_id"]) == ["PAR"]

    def test_batch_multi_speaker_inference_is_row_failure(self, tmp_path):
        row = _manifest_row(tmp_path, "r1", speaker="PAR")
        row["target_speakers"] = None
        data = _document_json(speakers=("PAR", "EXA"), utterance_speaker="PAR")
        _write_json(tmp_path / row["transcript_path"], data)
        bundle = speech_features.extract_batch(_write_manifest(tmp_path, [row]))
        assert len(bundle.recordings) == 0
        errors = {
            (r.recording_id, r.speaker_id, r.code)
            for r in bundle.issues[bundle.issues["severity"] == "error"].itertuples(index=False)
        }
        assert ("r1", "", "TARGET_SPEAKER_REQUIRED") in errors

    def test_batch_all_rows_fail_keeps_empty_schema(self, tmp_path):
        row = _manifest_row(tmp_path, "r1", transcript="missing.json")
        (tmp_path / "missing.json").unlink()
        bundle = speech_features.extract_batch(_write_manifest(tmp_path, [row]))
        assert len(bundle.recordings) == 0
        assert len(bundle.utterances) == 0
        assert len(bundle.recordings.columns) == 2 + 166
        assert len(bundle.utterances.columns) == 5 + 4
        assert "error" in set(bundle.issues["severity"])
        assert list(bundle.recordings.columns[:2]) == ["recording_id", "speaker_id"]
        json.dumps(dict(bundle.provenance))

    def test_batch_always_returns_both_levels(self, tmp_path):
        bundle = speech_features.extract_batch(
            _write_manifest(tmp_path, [_manifest_row(tmp_path, "r1")]), packs=("acoustic",)
        )
        assert list(bundle.utterances.columns) == [
            "recording_id",
            "speaker_id",
            "utterance_id",
            "start_s",
            "end_s",
        ]
        assert len(bundle.recordings.columns) == 2 + 73

    def test_batch_issues_sorted_deterministically(self, tmp_path):
        good1 = _manifest_row(tmp_path, "r1", audio="a1.wav", transcript="t1.json")
        good2 = _manifest_row(tmp_path, "r2", audio="a2.wav", transcript="t2.json")
        bundle = speech_features.extract_batch(_write_manifest(tmp_path, [good2, good1]))
        rows = [
            tuple("" if pd.isna(value) else value for value in r)
            for r in bundle.issues.itertuples(index=False)
        ]
        assert rows == sorted(rows)
        assert list(bundle.recordings["recording_id"]) == ["r1", "r2"]

    def test_batch_unexpected_failure_is_isolated(self, tmp_path, monkeypatch):
        import speech_features.extraction as extraction

        def boom(*args, **kwargs):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(extraction, "_extract_acoustic_pack", boom)
        row = _manifest_row(tmp_path, "r1")
        bundle = speech_features.extract_batch(_write_manifest(tmp_path, [row]))
        errors = {
            (r.recording_id, r.speaker_id, r.code)
            for r in bundle.issues[bundle.issues["severity"] == "error"].itertuples(index=False)
        }
        assert ("r1", "PAR", "EXTRACTION_ERROR") in errors
        assert bundle.provenance["counts"] == {"total": 1, "success": 0, "failure": 1}
        assert "audio_sha256" in bundle.provenance["recordings"]["r1"]
