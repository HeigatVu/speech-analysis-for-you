import dataclasses

import pytest
from say_transcribe.annotations import (
    AnnotationValidationError,
    load_approved_annotations,
)


def valid_p001_synthetic_payload() -> dict:
    return {
        "schema_version": "1.0.0",
        "source_audio": "pilot/audio/p001/p001_master.wav",
        "channel_index": 0,
        "sample_rate": 44100,
        "source_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "review_status": "approved",
        "tasks": [
            {
                "task_id": "sustained_phonation",
                "start_ms": 27372,
                "end_ms": 53827,
                "turns": [
                    {
                        "turn_id": "turn_sp_01",
                        "speaker_role": "participant",
                        "start_ms": 27372,
                        "end_ms": 53827,
                    }
                ],
            },
            {
                "task_id": "reading",
                "start_ms": 57452,
                "end_ms": 90000,
                "turns": [
                    {
                        "turn_id": "turn_rd_01",
                        "speaker_role": "investigator",
                        "start_ms": 57452,
                        "end_ms": 69000,
                    },
                    {
                        "turn_id": "turn_rd_02",
                        "speaker_role": "participant",
                        "start_ms": 70000,
                        "end_ms": 88500,
                    },
                ],
            },
        ],
    }


def test_valid_annotations_parsing():
    data = valid_p001_synthetic_payload()
    session = load_approved_annotations(data)
    assert session.source_audio == "pilot/audio/p001/p001_master.wav"
    assert session.sample_rate == 44100
    assert session.channel_index == 0
    assert len(session.tasks) == 2
    assert session.tasks[1].turns[1].speaker_role == "participant"


def test_rejects_absolute_paths():
    data = valid_p001_synthetic_payload()
    data["source_audio"] = (
        "/shared-data/hcmiu-bhl-corpus/pilot/audio/p001/p001_master.wav"
    )
    with pytest.raises(AnnotationValidationError) as exc:
        load_approved_annotations(data)
    assert exc.value.code == "INVALID_ANNOTATIONS"
    assert "source_audio" in exc.value.message


def test_rejects_unapproved_status():
    data = valid_p001_synthetic_payload()
    data["review_status"] = "draft"
    with pytest.raises(AnnotationValidationError) as exc:
        load_approved_annotations(data)
    assert exc.value.code == "UNAPPROVED_ANNOTATIONS"


def test_rejects_malformed_sha256():
    data = valid_p001_synthetic_payload()
    data["source_sha256"] = "96B51B3B"
    with pytest.raises(AnnotationValidationError) as exc:
        load_approved_annotations(data)
    assert exc.value.code == "INVALID_ANNOTATIONS"


def test_rejects_invalid_speaker_role():
    data = valid_p001_synthetic_payload()
    data["tasks"][0]["turns"][0]["speaker_role"] = "PAR"
    with pytest.raises(AnnotationValidationError) as exc:
        load_approved_annotations(data)
    assert exc.value.code == "INVALID_ANNOTATIONS"


def test_rejects_overlapping_turns():
    data = valid_p001_synthetic_payload()
    data["tasks"][1]["turns"][1]["start_ms"] = 65000
    with pytest.raises(AnnotationValidationError) as exc:
        load_approved_annotations(data)
    assert exc.value.code == "INVALID_ANNOTATIONS"


def test_rejects_unknow_top_level_fields():
    data = valid_p001_synthetic_payload()
    data["participant_metadata"] = {"age": 60}
    with pytest.raises(AnnotationValidationError) as exc:
        load_approved_annotations(data)
    assert exc.value.code == "INVALID_ANNOTATIONS"


def test_rejects_duplicate_task_ids():
    data = valid_p001_synthetic_payload()
    data["tasks"][1]["task_id"] = "sustained_phonation"
    with pytest.raises(AnnotationValidationError) as exc:
        load_approved_annotations(data)
    assert exc.value.code == "INVALID_ANNOTATIONS"


def test_rejects_duplicate_turn_ids():
    data = valid_p001_synthetic_payload()
    data["tasks"][1]["turns"][1]["turn_id"] = "turn_rd_01"
    with pytest.raises(AnnotationValidationError) as exc:
        load_approved_annotations(data)
    assert exc.value.code == "INVALID_ANNOTATIONS"


def test_rejects_turn_outside_task_bounds():
    data = valid_p001_synthetic_payload()
    data["tasks"][0]["turns"][0]["end_ms"] = 99999
    with pytest.raises(AnnotationValidationError) as exc:
        load_approved_annotations(data)
    assert exc.value.code == "INVALID_ANNOTATIONS"


def test_rejects_bool_bounds():
    data = valid_p001_synthetic_payload()
    data["tasks"][0]["start_ms"] = True
    with pytest.raises(AnnotationValidationError) as exc:
        load_approved_annotations(data)
    assert exc.value.code == "INVALID_ANNOTATIONS"


def test_rejects_invalid_channel_index():
    data = valid_p001_synthetic_payload()
    data["channel_index"] = -1
    with pytest.raises(AnnotationValidationError) as exc:
        load_approved_annotations(data)
    assert exc.value.code == "INVALID_ANNOTATIONS"


def test_session_and_children_are_deeply_immutable():
    data = valid_p001_synthetic_payload()
    session = load_approved_annotations(data)
    assert isinstance(session.tasks, tuple)
    assert isinstance(session.tasks[0].turns, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        session.source_audio = "elsewhere.wav"
    with pytest.raises(dataclasses.FrozenInstanceError):
        session.tasks[0].turns[0].start_ms = 0


def test_accepts_matching_audio_hash(tmp_path):
    data = valid_p001_synthetic_payload()
    audio_path = tmp_path / data["source_audio"]
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"")  # sha256("") matches the fixture's source_sha256
    session = load_approved_annotations(data, audio_root=tmp_path)
    assert session.source_sha256 == data["source_sha256"]


def test_rejects_stale_audio_hash(tmp_path):
    data = valid_p001_synthetic_payload()
    audio_path = tmp_path / data["source_audio"]
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"not the approved audio")
    with pytest.raises(AnnotationValidationError) as exc:
        load_approved_annotations(data, audio_root=tmp_path)
    assert exc.value.code == "SOURCE_HASH_MISMATCH"
