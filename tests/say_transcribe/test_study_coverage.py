"""T5: speech coverage and retained-utterance metrics against frozen PAR intervals."""

import inspect
from pathlib import Path

import numpy as np
import pytest

from say_transcribe.study import (
    StudyError,
    arm_vad_windows,
    coverage_metrics,
    load_reference_document,
    reference_intervals,
    windows_to_ms,
)

_BULLET = "\x15"
_HEADER = (
    "@UTF8\n@Begin\n@Languages:\tvie\n"
    "@Participants:\tPAR Participant, INV Investigator\n"
    "@ID:\tvie|corpus|PAR|||||Participant|||\n"
    "@ID:\tvie|corpus|INV|||||Investigator|||\n"
    "@Media:\tsynthetic, audio\n"
)


_WORDS = "một hai"


def _reference(tmp_path: Path, utterances: list[tuple[str, int, int]], name: str = "ref.cha") -> Path:
    lines = [_HEADER]
    for speaker, start, end in utterances:
        lines.append(f"*{speaker}:\t{_WORDS} .\t{_BULLET}{start}_{end}{_BULLET}\n")
    lines.append("@End\n")
    path = tmp_path / name
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _intervals(tmp_path: Path, utterances: list[tuple[str, int, int]]):
    path = _reference(tmp_path, utterances)
    return reference_intervals(load_reference_document(path.read_text(encoding="utf-8")))


def test_coverage_counts_only_participant_speech_overlapping_vad(tmp_path: Path):
    reference = _intervals(tmp_path, [("PAR", 0, 1000), ("INV", 1000, 2000), ("PAR", 2000, 3000)])

    metrics = coverage_metrics([(0, 600), (2400, 3000)], reference)

    assert metrics["reference_speech_ms"] == 2000
    assert metrics["retained_speech_ms"] == 1200
    assert metrics["coverage"] == pytest.approx(0.6)
    assert metrics["utterance_count"] == 2
    assert metrics["retained_utterance_count"] == 2
    assert metrics["retained_utterance_fraction"] == pytest.approx(1.0)
    assert metrics["zero_duration_utterances"] == 0


def test_interviewer_speech_is_never_counted(tmp_path: Path):
    reference = _intervals(tmp_path, [("PAR", 0, 1000), ("INV", 1000, 5000)])

    metrics = coverage_metrics([(0, 5000)], reference)

    assert metrics["reference_speech_ms"] == 1000
    assert metrics["retained_speech_ms"] == 1000
    assert metrics["coverage"] == pytest.approx(1.0)
    assert metrics["utterance_count"] == 1


@pytest.mark.parametrize(
    "window,retained",
    [((0, 500), True), ((0, 499), False), ((501, 1000), False), ((250, 750), True)],
)
def test_retained_utterance_boundary_is_exactly_half(tmp_path: Path, window, retained):
    reference = _intervals(tmp_path, [("PAR", 0, 1000)])

    metrics = coverage_metrics([window], reference)

    assert metrics["retained_utterance_count"] == int(retained)
    assert metrics["coverage"] == pytest.approx((window[1] - window[0]) / 1000)


def test_metrics_ignore_window_order_duplicates_and_adjacency(tmp_path: Path):
    reference = _intervals(tmp_path, [("PAR", 0, 1000), ("PAR", 2000, 3000)])
    canonical = coverage_metrics([(0, 600), (2400, 3000)], reference)

    shuffled = coverage_metrics([(2400, 3000), (0, 600)], reference)
    overlapping = coverage_metrics([(0, 300), (200, 600), (2400, 3000), (2400, 3000)], reference)

    assert shuffled == canonical
    assert overlapping == canonical
    # Windows also covering interviewer-only audio cannot push coverage past the
    # reference participant speech the metric is defined against.
    bridged = coverage_metrics([(0, 600), (600, 2400), (2400, 3000)], reference)

    assert bridged["retained_speech_ms"] == 2000
    assert bridged["coverage"] == pytest.approx(1.0)


def test_metrics_are_arm_agnostic_because_only_windows_and_reference_enter(tmp_path: Path):
    reference = _intervals(tmp_path, [("PAR", 0, 1000)])
    parameters = list(inspect.signature(coverage_metrics).parameters)

    assert parameters == ["vad_windows_ms", "reference"]
    assert coverage_metrics([(0, 1000)], reference) == coverage_metrics(tuple([(0, 1000)]), reference)


def test_zero_duration_utterance_is_reported_not_hidden(tmp_path: Path):
    reference = _intervals(tmp_path, [("PAR", 0, 1000), ("PAR", 1500, 1500)])

    metrics = coverage_metrics([(0, 1000)], reference)

    assert reference.zero_duration == 1
    assert metrics["zero_duration_utterances"] == 1
    assert metrics["utterance_count"] == 1
    assert metrics["reference_speech_ms"] == 1000
    assert metrics["coverage"] == pytest.approx(1.0)


def test_no_participant_speech_fails_loudly(tmp_path: Path):
    with pytest.raises(StudyError) as excinfo:
        _intervals(tmp_path, [("INV", 0, 1000)])

    assert excinfo.value.code == "INVALID_ARGUMENT"


def test_participant_utterance_without_a_bullet_fails_loudly(tmp_path: Path):
    """An untimed tier is rejected at decode, so no reference slips through untimed."""
    path = tmp_path / "untimed.cha"
    path.write_text(_HEADER + f"*PAR:\t{_WORDS} .\n@End\n", encoding="utf-8")

    with pytest.raises(StudyError) as excinfo:
        load_reference_document(path.read_text(encoding="utf-8"))

    assert excinfo.value.code == "INVALID_ARGUMENT"
    assert excinfo.value.__cause__ is None


def test_reversed_utterance_timing_fails_loudly(tmp_path: Path):
    """A transposed inline bullet must not be reclassified as a zero-length utterance."""
    path = tmp_path / "reversed.cha"
    path.write_text(
        _HEADER + f"*PAR:\t{_WORDS} .\t{_BULLET}12000_2000{_BULLET}\n@End\n", encoding="utf-8"
    )

    with pytest.raises(StudyError) as excinfo:
        reference_intervals(load_reference_document(path.read_text(encoding="utf-8")))

    assert excinfo.value.code == "INVALID_ARGUMENT"
    assert excinfo.value.__cause__ is None


def test_reference_without_a_usable_interval_fails_loudly(tmp_path: Path):
    """All-zero-length participant speech has no defined coverage ratio."""
    with pytest.raises(StudyError) as excinfo:
        _intervals(tmp_path, [("PAR", 1500, 1500)])

    assert excinfo.value.code == "INVALID_ARGUMENT"
    assert "usable" in excinfo.value.message


def test_malformed_reference_is_not_silently_scored(tmp_path: Path):
    with pytest.raises(StudyError) as excinfo:
        load_reference_document("@Begin\n*PAR:\t" + _WORDS + " .\n@End\n")

    assert excinfo.value.code == "INVALID_ARGUMENT"
    assert excinfo.value.__cause__ is None


def test_windows_to_ms_converts_sample_indices():
    assert windows_to_ms([(0, 16000), (24000, 32000)]) == ((0, 1000), (1500, 2000))
    assert windows_to_ms([(1, 3)], sample_rate=2) == ((500, 1500),)


def test_coverage_uses_the_shared_vad_threshold(monkeypatch):
    """The arm's windows come from the frozen study threshold, not a default."""
    seen: list[float] = []

    def fake_windows(audio, *, threshold):
        seen.append(threshold)
        return [(0, 16000)]

    monkeypatch.setattr("say_transcribe.study.get_speech_windows", fake_windows)

    assert arm_vad_windows(np.zeros(16000)) == ((0, 16000),)
    assert seen == [0.2]
