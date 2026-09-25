"""T6: acoustic pack and PAR-only eGeMAPS extraction for one arm's signal."""

import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from say_transcribe.study import (
    StudyError,
    eligible_feature_intervals,
    extract_arm_features,
    reference_intervals,
    require_egemaps_available,
)
from speech_features.catalog import list_features
from speech_features.features.acoustic import extract_acoustic_features
from speech_features.formats.chat import decode_chat
from speech_features.schema import FeatureExtractionError

_BULLET = "\x15"
_HEADER = (
    "@UTF8\n@Begin\n@Languages:\tvie\n"
    "@Participants:\tPAR Participant, INV Investigator\n"
    "@ID:\tvie|corpus|PAR|||||Participant|||\n"
    "@ID:\tvie|corpus|INV|||||Investigator|||\n"
    "@Media:\tsynthetic, audio\n"
)
_RATE = 16000


def _document(utterances: list[tuple[str, int, int]]):
    body = "".join(
        f"*{speaker}:\tmột hai .\t{_BULLET}{start}_{end}{_BULLET}\n"
        for speaker, start, end in utterances
    )
    return decode_chat(_HEADER + body + "@End\n")


def _key_with_domain(pack: str, domain: str) -> str:
    return next(d.key for d in list_features(pack=pack) if d.domain == domain)


@pytest.fixture(autouse=True)
def _opensmile_present(monkeypatch):
    """Tests exercise the extraction logic; the dependency gate has its own test."""
    monkeypatch.setattr("say_transcribe.study._opensmile_available", lambda: True)


def _patch_extractors(monkeypatch, acoustic, egemaps):
    calls = {"acoustic": [], "egemaps": []}

    def fake_acoustic(audio, sample_rate, **kwargs):
        calls["acoustic"].append((audio, sample_rate, kwargs))
        return acoustic, ()

    def fake_egemaps(audio, sample_rate, **kwargs):
        calls["egemaps"].append((audio, sample_rate, kwargs))
        return egemaps, (), {}

    monkeypatch.setattr("say_transcribe.study.extract_acoustic_features", fake_acoustic)
    monkeypatch.setattr("say_transcribe.study.extract_egemaps_features", fake_egemaps)
    return calls


def test_real_acoustic_extraction_never_invokes_the_stereo_reader(monkeypatch):
    """The arm's mono array reaches the real extractor; no file reader, no downmix.

    The real acoustic extractor stays bound (it needs no optional dependency), so
    this test fails if the code under test hands a path, a stereo view, or a
    re-read of the master to the extraction stage.
    """
    reader_calls: list[tuple] = []

    def forbidden_reader(*args, **kwargs):
        reader_calls.append(args)
        raise AssertionError("the stereo-downmixing WAV reader was invoked")

    for target in (
        "speech_features.audio._read_wav_with_width",
        "speech_features.audio.read_wav",
        "speech_features.features.acoustic._read_wav_with_width",
        "speech_features.extraction._read_wav_with_width",
    ):
        monkeypatch.setattr(target, forbidden_reader, raising=False)
    monkeypatch.setattr(
        "say_transcribe.study.extract_egemaps_features",
        lambda audio, rate, **kwargs: ({"egemaps_f0": 1.0}, (), {}),
    )
    tone = (0.2 * np.sin(2 * np.pi * 145 * np.arange(3 * _RATE) / _RATE)).astype(np.float32)
    document = _document([("PAR", 0, 2000)])
    reference = reference_intervals(document)
    expected, _ = extract_acoustic_features(
        tone, _RATE, document=document, target_speaker="PAR"
    )

    record = extract_arm_features(tone, _RATE, document, reference)

    assert reader_calls == []
    assert record["acoustic"]["values"] == len(expected) > 100
    assert record["acoustic"]["finite"] == sum(
        1 for value in expected.values() if isinstance(value, float) and math.isfinite(value)
    )
    assert record["acoustic"]["finite"] > 0
    assert record["samples"] == tone.shape[0]


def test_acoustic_pack_is_scoped_to_the_participant_intervals(monkeypatch):
    calls = _patch_extractors(monkeypatch, {"audio_rms_dbfs": -20.0}, {"egemaps_f0": 1.0})
    monotone = np.zeros(_RATE * 3, dtype=np.float32)
    document = _document([("PAR", 0, 2000)])
    reference = reference_intervals(document)

    extract_arm_features(monotone, _RATE, document, reference)

    _, sample_rate, kwargs = calls["acoustic"][0]
    assert sample_rate == _RATE
    assert kwargs == {"document": document, "target_speaker": "PAR"}


def test_egemaps_skips_interviewer_speech_and_short_utterances(monkeypatch):
    calls = _patch_extractors(monkeypatch, {"audio_rms_dbfs": -20.0}, {"egemaps_f0": 1.0})
    monotone = np.zeros(_RATE * 6, dtype=np.float32)
    document = _document([("PAR", 0, 2000), ("PAR", 2000, 2500), ("INV", 3000, 5000)])
    reference = reference_intervals(document)

    record = extract_arm_features(monotone, _RATE, document, reference)

    assert len(calls["egemaps"]) == 1
    audio, sample_rate, kwargs = calls["egemaps"][0]
    assert audio.shape == (2 * _RATE,)
    assert sample_rate == _RATE
    assert kwargs == {"recording_id": "", "speaker_id": "PAR"}
    assert record["egemaps"]["utterances"] == 1
    assert record["egemaps"]["valid_utterances"] == 1
    assert record["egemaps"]["truncated_utterances"] == 0


def test_eligible_intervals_use_the_one_second_rule():
    document = _document([("PAR", 0, 1000), ("PAR", 1000, 1999), ("PAR", 2000, 3001)])
    reference = reference_intervals(document)

    assert eligible_feature_intervals(reference) == ((0, 1000), (2000, 3001))


def test_short_or_missing_opensmile_fails_before_any_extraction(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("extraction ran without the eGeMAPS dependency")

    monkeypatch.setattr("say_transcribe.study._opensmile_available", lambda: False)
    monkeypatch.setattr("say_transcribe.study.extract_acoustic_features", forbidden)
    monkeypatch.setattr("say_transcribe.study.extract_egemaps_features", forbidden)
    document = _document([("PAR", 0, 2000)])

    with pytest.raises(StudyError) as excinfo:
        extract_arm_features(
            np.zeros(_RATE * 3, dtype=np.float32), _RATE, document, reference_intervals(document)
        )

    assert excinfo.value.code == "FEATURE_EXTRACTION_FAILED"
    assert excinfo.value.__cause__ is None


def test_opensmile_availability_matches_the_interpreter_in_a_fresh_process():
    """The gate's probe is the real function, not the module fixture's stub."""
    code = (
        "import importlib.util, sys, say_transcribe.study as study\n"
        "expected = importlib.util.find_spec('opensmile') is not None\n"
        "assert study._opensmile_available() == expected, 'availability probe disagrees'\n"
        "if not expected:\n"
        "    try:\n"
        "        study.require_egemaps_available()\n"
        "    except study.StudyError as error:\n"
        "        assert error.code == 'FEATURE_EXTRACTION_FAILED', error.code\n"
        "    else:\n"
        "        raise AssertionError('the gate did not fire without opensmile')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_study_import_does_not_pull_heavy_model_libraries():
    """Importing the study module must not load torch, openSMILE, or a model stack."""
    code = (
        "import sys, say_transcribe.study\n"
        "for name in ('torch', 'opensmile', 'transformers', 'onnxruntime', 'librosa', 'soundfile'):\n"
        "    assert name not in sys.modules, f'{name} was eagerly imported'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_require_egemaps_reports_the_real_environment_state(monkeypatch):
    """The gate reflects the actual interpreter, not a stub."""
    import importlib.util

    monkeypatch.setattr(
        "say_transcribe.study._opensmile_available",
        lambda: importlib.util.find_spec("opensmile") is not None,
    )
    if importlib.util.find_spec("opensmile") is None:
        with pytest.raises(StudyError) as excinfo:
            require_egemaps_available()
        assert excinfo.value.code == "FEATURE_EXTRACTION_FAILED"
    else:  # pragma: no cover - only on machines with the extra installed
        require_egemaps_available()


def test_finite_value_coverage_is_reported_per_family(monkeypatch):
    timing_key = _key_with_domain("acoustic", "timing")
    spectral_key = _key_with_domain("acoustic", "spectral")
    egemaps_prosody = _key_with_domain("standardized_acoustic", "prosody")
    egemaps_spectral = _key_with_domain("standardized_acoustic", "spectral")
    _patch_extractors(
        monkeypatch,
        {timing_key: 1.0, spectral_key: float("nan")},
        {egemaps_prosody: 2.0, egemaps_spectral: float("nan")},
    )
    monotone = np.zeros(_RATE * 3, dtype=np.float32)
    document = _document([("PAR", 0, 2000)])
    reference = reference_intervals(document)

    record = extract_arm_features(monotone, _RATE, document, reference)

    assert record["acoustic"]["values"] == 2
    assert record["acoustic"]["finite"] == 1
    assert record["acoustic"]["valid"] is False
    assert record["acoustic"]["families"]["timing"] == {"values": 1, "finite": 1, "coverage": 1.0}
    assert record["acoustic"]["families"]["spectral"] == {"values": 1, "finite": 0, "coverage": 0.0}
    assert record["egemaps"]["families"]["prosody"]["coverage"] == 1.0
    assert record["egemaps"]["families"]["spectral"]["coverage"] == 0.0
    assert record["egemaps"]["coverage"] == pytest.approx(0.5)


def test_utterance_beyond_the_signal_is_reported_as_truncated(monkeypatch):
    calls = _patch_extractors(monkeypatch, {"audio_rms_dbfs": -20.0}, {"egemaps_f0": 1.0})
    document = _document([("PAR", 0, 2000), ("PAR", 9000, 11000)])
    reference = reference_intervals(document)

    record = extract_arm_features(np.zeros(_RATE * 3, dtype=np.float32), _RATE, document, reference)

    assert len(calls["egemaps"]) == 1
    assert record["egemaps"]["utterances"] == 2
    assert record["egemaps"]["extracted"] == 1
    assert record["egemaps"]["truncated_utterances"] == 1
    assert record["egemaps"]["valid_utterances"] == 1
    assert record["egemaps"]["utterances_detail"][1]["reason"] == "TRUNCATED"


def test_partially_covered_utterance_is_truncated_not_valid(monkeypatch):
    """A slice the arm signal cannot fully cover never counts as a >= 1 s extraction."""
    calls = _patch_extractors(monkeypatch, {"audio_rms_dbfs": -20.0}, {"egemaps_f0": 1.0})
    document = _document([("PAR", 0, 2000), ("PAR", 9000, 11000)])
    reference = reference_intervals(document)

    record = extract_arm_features(np.zeros(_RATE * 3, dtype=np.float32), _RATE, document, reference)

    assert len(calls["egemaps"]) == 1
    assert record["egemaps"]["utterances"] == 2
    assert record["egemaps"]["extracted"] == 1
    assert record["egemaps"]["truncated_utterances"] == 1
    assert record["egemaps"]["valid_utterances"] == 1
    assert record["egemaps"]["values"] == 1
    assert record["egemaps"]["utterances_detail"][1]["reason"] == "TRUNCATED"
    assert record["egemaps"]["utterances_detail"][1]["valid"] is False


def test_extraction_errors_map_to_a_stable_code(monkeypatch):
    """A library extraction error must not escape as a raw library exception."""
    document = _document([("PAR", 0, 2000)])

    def broken_acoustic(audio, rate, **kwargs):
        raise FeatureExtractionError("no usable aligned target-speaker intervals")

    monkeypatch.setattr("say_transcribe.study.extract_acoustic_features", broken_acoustic)
    with pytest.raises(StudyError) as acoustic_error:
        extract_arm_features(
            np.zeros(_RATE * 3, dtype=np.float32), _RATE, document, reference_intervals(document)
        )
    assert acoustic_error.value.code == "FEATURE_EXTRACTION_FAILED"
    assert acoustic_error.value.__cause__ is None

    _patch_extractors(monkeypatch, {"audio_rms_dbfs": -20.0}, {"egemaps_f0": 1.0})

    def broken_egemaps(audio, rate, **kwargs):
        raise FeatureExtractionError("openSMILE returned an invalid table")

    monkeypatch.setattr("say_transcribe.study.extract_egemaps_features", broken_egemaps)
    with pytest.raises(StudyError) as egemaps_error:
        extract_arm_features(
            np.zeros(_RATE * 3, dtype=np.float32), _RATE, document, reference_intervals(document)
        )
    assert egemaps_error.value.code == "FEATURE_EXTRACTION_FAILED"
    assert egemaps_error.value.__cause__ is None


def test_issue_codes_are_counted_without_messages(monkeypatch):
    issue = SimpleNamespace(code="MISSING_OPTIONAL_DEPENDENCY", message="private detail")
    _patch_extractors(
        monkeypatch,
        {"audio_rms_dbfs": -20.0},
        {"egemaps_f0": 1.0},
    )
    monkeypatch.setattr(
        "say_transcribe.study.extract_acoustic_features",
        lambda audio, rate, **kwargs: ({"audio_rms_dbfs": -20.0}, (issue,)),
    )  # returns (features, issues) like the shared extractor
    document = _document([("PAR", 0, 2000)])

    record = extract_arm_features(
        np.zeros(_RATE * 3, dtype=np.float32), _RATE, document, reference_intervals(document)
    )

    assert record["acoustic"]["issue_codes"] == {"MISSING_OPTIONAL_DEPENDENCY": 1}
    assert "private detail" not in repr(record)


def test_record_carries_no_paths_or_identity(monkeypatch, tmp_path: Path):
    """The per-arm feature record is counts only: nothing to redact downstream."""
    _patch_extractors(monkeypatch, {"audio_rms_dbfs": -20.0}, {"egemaps_f0": 1.0})
    document = _document([("PAR", 0, 2000)])

    record = extract_arm_features(
        np.zeros(_RATE * 3, dtype=np.float32), _RATE, document, reference_intervals(document)
    )
    text = repr(record)

    assert str(tmp_path) not in text
    assert "một" not in text and "hai" not in text
    assert ".cha" not in text
