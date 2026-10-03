"""T13: the six-arm ablation of the ported ASR/VAD robustness behaviors."""

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

from say_transcribe import ablation
from say_transcribe.ablation import (
    ABLATION_ARM_ORDER,
    ARM_BY_NAME,
    ArmRun,
    arm_windows,
    content_counts,
    promotion_verdict,
    result_counts,
    run_arm,
    session_extras,
)
from say_transcribe.asr import AsrResult, AsrSegment, WordTiming, compute_sha256
from say_transcribe.study import ReferenceIntervals
from say_transcribe.vad import get_speech_windows

_DIGEST = "a" * 64


def _long_speech_with_a_quiet_patch() -> tuple[np.ndarray, int]:
    """45 s of loud speech with 500 ms of silence at 18.5 s, inside the 2 s search."""
    rate = 16000
    audio = np.full(45 * rate, 0.5, dtype=np.float32)
    quiet_start = 18 * rate + rate // 2
    audio[quiet_start : quiet_start + rate // 2] = 0.0
    return audio, quiet_start


def test_the_arm_table_isolates_one_ported_behavior_at_a_time() -> None:
    assert ABLATION_ARM_ORDER == (
        "main",
        "compatibility",
        "quiet-vad",
        "empty-retry",
        "silence-cut",
        "combined",
    )
    compatibility = ARM_BY_NAME["compatibility"]
    assert (compatibility.boost, compatibility.retry, compatibility.snap) == (False, False, False)
    combined = ARM_BY_NAME["combined"]
    assert (combined.boost, combined.retry, combined.snap) == (True, True, True)
    # Each individual arm turns on exactly one behavior relative to compatibility.
    for name in ("quiet-vad", "empty-retry", "silence-cut"):
        arm = ARM_BY_NAME[name]
        assert sum((arm.boost, arm.retry, arm.snap)) == 1, name


def test_the_main_arm_is_the_shipped_default_and_cannot_drift_from_it() -> None:
    defaults = inspect.signature(get_speech_windows).parameters
    main = ARM_BY_NAME["main"]

    assert (main.boost, main.retry, main.snap) == tuple(
        defaults[name].default for name in ("boost", "retry", "snap")
    )


def test_arm_windows_hand_their_boost_flag_to_detection() -> None:
    rate = 16000
    quiet = np.zeros(rate, dtype=np.float32)
    quiet[::7] = 0.02
    quiet[3::7] = -0.02
    peaks: list[float] = []

    def detector(clip: np.ndarray) -> list[tuple[int, int]]:
        peaks.append(float(np.abs(clip).max()))
        return [(0, 8000)]

    arm_windows(quiet, ARM_BY_NAME["quiet-vad"], detector=detector)
    arm_windows(quiet, ARM_BY_NAME["compatibility"], detector=detector)

    assert peaks[0] == pytest.approx(0.071, rel=1e-3)
    assert peaks[1] == pytest.approx(0.02)


def test_only_the_silence_cut_arm_moves_the_hard_cut_into_the_quiet_patch() -> None:
    rate = 16000
    audio, quiet_start = _long_speech_with_a_quiet_patch()
    stuck = lambda _: [(0, 40 * rate)]  # noqa: E731 - one-line stub detector

    snapped = arm_windows(audio, ARM_BY_NAME["silence-cut"], detector=stuck)
    hard = arm_windows(audio, ARM_BY_NAME["main"], detector=stuck)

    assert hard[0] == (0, 20 * rate)
    assert quiet_start <= snapped[0][1] <= quiet_start + rate // 2


class _StubBackend:
    """One 500 ms two-word segment, whatever audio it is given."""

    def __init__(self) -> None:
        self.slices: list[int] = []

    def transcribe_audio(self, audio: np.ndarray) -> list[dict[str, object]]:
        self.slices.append(len(audio))
        return [
            {
                "start_ms": 0,
                "end_ms": 500,
                "text": "tôi đi",
                "words": [
                    {"word": "tôi", "start_ms": 0, "end_ms": 200},
                    {"word": "đi", "start_ms": 200, "end_ms": 500},
                ],
            }
        ]


def test_run_arm_offsets_its_segments_onto_the_master_timeline() -> None:
    rate = 16000
    audio = np.zeros(3 * rate, dtype=np.float32)
    backend = _StubBackend()

    run = run_arm(
        audio, _DIGEST, backend, ARM_BY_NAME["main"], detector=lambda _: [(rate, 2 * rate)]
    )

    # VAD pads each span by 30 ms, so the detected (1 s, 2 s) span becomes
    # (15520, 32480) samples and the 500 ms segment lands at 970-1470 ms.
    assert run.windows == ((15520, 32480),)
    assert backend.slices == [32480 - 15520]
    assert run.result.source_sha256 == _DIGEST
    assert run.result.segments[0].start_ms == 970
    assert run.result.segments[0].end_ms == 1470
    assert run.seconds >= 0.0


def test_result_counts_report_invalid_timing_and_repetition_flags() -> None:
    result = AsrResult(
        _DIGEST,
        (
            AsrSegment(0, 500, "tôi đi", (WordTiming("tôi", 0, 200), WordTiming("đi", None, None))),
            AsrSegment(None, None, "ờ", ()),
        ),
        ("ASR_REPETITION_SUSPECTED:1", "CHAT_WORD_TIMING_UNAVAILABLE:1"),
    )

    assert result_counts(result) == {
        "words": 2,
        "words_without_timing": 1,
        "utterances_without_timing": 1,
        "repetition_flags": 1,
    }


def test_content_counts_measure_what_an_arm_actually_produced() -> None:
    assert content_counts((AsrSegment(0, 500, "tôi đi", ()),)) == {
        "words": 2,
        "syllables": 2,
        "chars": 5,
    }


def test_session_extras_pair_coverage_with_the_arm_output() -> None:
    reference = ReferenceIntervals(utterances=((0, 1000), (2000, 3000)), zero_duration=0)
    run = ArmRun(
        windows=((0, 16000),),  # samples: 0-1000 ms of 16 kHz audio
        result=AsrResult(_DIGEST, (AsrSegment(0, 500, "tôi đi", ()),), ()),
        seconds=1.5,
    )

    extras = session_extras(run, reference)

    assert extras["coverage"]["coverage"] == pytest.approx(0.5)
    assert extras["coverage"]["retained_utterance_count"] == 1
    assert extras["runtime_s"] == pytest.approx(1.5)
    assert extras["counts"]["words_without_timing"] == 0
    assert extras["content"]["syllables"] == 2


def _arm_record(
    syer: float, coverage: float, missing_timing: int, syllables: int
) -> dict[str, object]:
    return {
        "scores": {"syer": {"edits": 1, "ref_count": 10, "rate": syer}},
        "coverage": {"coverage": coverage},
        "counts": {"words": 10, "words_without_timing": missing_timing},
        "content": {"words": 10, "syllables": syllables, "chars": 30},
    }


def test_promotion_needs_a_p001_gain_without_content_coverage_or_timing_regression() -> None:
    main = _arm_record(0.40, 0.90, 4, 100)
    comparisons = {
        "combined": _arm_record(0.35, 0.90, 4, 100),  # strictly better accuracy
        "quiet-vad": _arm_record(0.35, 0.80, 4, 100),  # coverage regression
        "empty-retry": _arm_record(0.35, 0.90, 6, 100),  # timing regression
        "silence-cut": _arm_record(0.35, 0.90, 4, 90),  # content regression
        "compatibility": _arm_record(0.45, 0.95, 2, 120),  # no accuracy gain
    }

    verdict = promotion_verdict({"p001": {"arms": {"main": main, **comparisons}}})

    assert verdict["combined"] == {
        "promote": True,
        "accuracy_gain": True,
        "content_ok": True,
        "coverage_ok": True,
        "timing_ok": True,
    }
    for name in ("quiet-vad", "empty-retry", "silence-cut", "compatibility"):
        assert verdict[name]["promote"] is False, name
    assert "main" not in verdict


def test_promotion_verdict_is_empty_without_the_primary_session() -> None:
    assert promotion_verdict({"p002": {"arms": {"main": _arm_record(0.4, 0.9, 0, 10)}}}) == {}


_REFERENCE = (
    "@UTF8\n@Begin\n@Languages:\tvie\n"
    "@Participants:\tPAR Participant\n"
    "@ID:\tvie|corpus|PAR|||||Participant|||\n"
    "@Media:\tp001, audio\n"
    "*PAR:\ttôi đi . \x150_1000\x15\n"
    "@End\n"
)


def _manifest(tmp_path, digest: str):
    audio_path = tmp_path / "p001_master.wav"
    audio_path.write_bytes(b"synthetic master")
    reference_path = tmp_path / "p001.cha"
    reference_path.write_text(_REFERENCE, encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "session_id": "p001",
                        "participant_id": "p001",
                        "audio_path": str(audio_path),
                        "reference_path": str(reference_path),
                        "channel_index": 0,
                        "sha256": digest,
                        "split": "dev",
                        "asr_revision": "vinai/phowhisper-large",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_ablation_runs_every_arm_and_writes_a_redacted_report(tmp_path, monkeypatch) -> None:
    audio_path = tmp_path / "p001_master.wav"
    audio_path.write_bytes(b"synthetic master")
    digest = compute_sha256(audio_path)
    manifest_path = _manifest(tmp_path, digest)
    audio_16k = np.zeros(4 * 16000, dtype=np.float32)
    monkeypatch.setattr(
        ablation,
        "prepare_views",
        lambda row: SimpleNamespace(audio_16k=audio_16k),
    )
    backend = _StubBackend()
    out_dir = tmp_path / "ablation"

    code = ablation.run_ablation(
        manifest_path,
        out_dir,
        model="phowhisper-large",
        revision="b" * 40,
        device="cpu",
        backend=backend,
        detector=lambda _: [(0, 2 * 16000)],
    )

    assert code == 0
    report_text = (out_dir / "ablation.json").read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert list(report["sessions"]["p001"]["arms"]) == list(ABLATION_ARM_ORDER)
    assert report["config"]["asr_model"] == "phowhisper-large"
    assert report["config"]["asr_revision"] == "b" * 40
    # The revision the arms actually ran, not the manifest row's declared one.
    assert report["sessions"]["p001"]["asr_revision"] == "b" * 40
    assert report["config"]["channel_index"] == 0
    # Every arm transcribes identically here, so no arm may be promoted.
    assert set(report["promotion"]) == set(ABLATION_ARM_ORDER[1:])
    assert not any(ruling["promote"] for ruling in report["promotion"].values())
    assert (out_dir / "records" / "p001.json").exists()
    assert str(tmp_path) not in report_text
    assert "/shared-data" not in report_text
    # Identical windows mean identical decodes, so the runner decodes once and says so
    # rather than spending the same minutes five more times for the same answer.
    assert len(backend.slices) == 1
    arms = report["sessions"]["p001"]["arms"]
    assert arms["main"]["reused_from"] is None
    assert {name: arm["reused_from"] for name, arm in arms.items() if name != "main"} == {
        name: "main" for name in ABLATION_ARM_ORDER[1:]
    }
    markdown = (out_dir / "ABLATION.md").read_text(encoding="utf-8")
    for name in ABLATION_ARM_ORDER:
        assert name in markdown
    # Every quantity the ablation is required to record is visible in the rendered report,
    # not only in the JSON.
    header = next(line for line in markdown.splitlines() if line.startswith("| arm |"))
    assert "words w/o timing" in header
    assert "utterances w/o timing" in header
    assert "repetition flags" in header
    assert "Device `cpu`, GPU memory not applicable." in markdown
    assert "`compatibility` from `main`" in markdown


def test_the_cli_exposes_the_ablation_with_a_pinned_revision(tmp_path, monkeypatch) -> None:
    from say_transcribe import cli

    audio_path = tmp_path / "p001_master.wav"
    audio_path.write_bytes(b"synthetic master")
    manifest_path = _manifest(tmp_path, compute_sha256(audio_path))
    audio_16k = np.zeros(4 * 16000, dtype=np.float32)
    monkeypatch.setattr(ablation, "prepare_views", lambda row: SimpleNamespace(audio_16k=audio_16k))
    monkeypatch.setattr(ablation, "make_asr_backend", lambda *a, **k: _StubBackend())
    monkeypatch.setattr(ablation, "get_speech_windows", lambda *a, **k: [(0, 2 * 16000)])
    out_dir = tmp_path / "cli-ablation"

    code = cli.main(
        [
            "ablate",
            str(manifest_path),
            "--out",
            str(out_dir),
            "--asr-model",
            "phowhisper-large",
            "--revision",
            "b" * 40,
        ]
    )

    assert code == 0
    assert (out_dir / "ABLATION.md").exists()
    # An unpinned revision is refused, so no arm result is ever unattributable.
    assert cli.main(["ablate", str(manifest_path), "--out", str(out_dir)]) == 2
