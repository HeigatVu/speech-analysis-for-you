"""T7: gates, exact tie-breaking, participant bootstrap, and the redacted summary."""

from fractions import Fraction
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

from say_transcribe.asr import PhoWhisperBackend
from say_transcribe.manifest import load_manifest
from say_transcribe.study import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    StudyError,
    _backend_for_revision,
    bootstrap_mean_interval,
    build_summary,
    compute_session,
    evaluate_gates,
    macro_mean,
    paired_difference,
    participant_syer,
    run_study,
    runtime_per_audio_hour,
    select_denoiser,
    write_summary,
)


def _arm(
    *,
    edits: int = 0,
    ref_count: int = 100,
    runtime_s: float | None = 10.0,
    reference_ms: int = 1000,
    retained_ms: int = 1000,
    utterances: int = 20,
    retained_utterances: int = 20,
    acoustic_values: int = 167,
    acoustic_finite: int = 167,
    acoustic_valid: bool = True,
    egemaps_utterances: int = 2,
    egemaps_extracted: int = 2,
    egemaps_valid: int = 2,
    egemaps_values: int = 176,
    egemaps_finite: int = 176,
) -> dict:
    arm = {
        "segments": [],
        "warnings": [],
        "scores": {
            "syer": {
                "edits": edits,
                "ref_count": ref_count,
                "rate": edits / ref_count if ref_count else 0.0,
            }
        },
        "vad": {
            "reference_speech_ms": reference_ms,
            "retained_speech_ms": retained_ms,
            "utterance_count": utterances,
            "retained_utterance_count": retained_utterances,
        },
        "features": {
            "acoustic": {
                "values": acoustic_values,
                "finite": acoustic_finite,
                "valid": acoustic_valid,
            },
            "egemaps": {
                "utterances": egemaps_utterances,
                "extracted": egemaps_extracted,
                "valid_utterances": egemaps_valid,
                "values": egemaps_values,
                "finite": egemaps_finite,
            },
        },
    }
    if runtime_s is not None:
        arm["runtime_s"] = runtime_s
    return arm


def _session(
    participant: str,
    session: str,
    arms: dict[str, dict],
    *,
    split: str = "dev",
    audio_seconds: float = 3600.0,
    normalization_type: str = "linear",
) -> dict:
    return {
        "session_id": session,
        "participant_id": participant,
        "split": split,
        "channel_index": 0,
        "sha256": "ab" * 32,
        "asr_revision": "rev1",
        "asr_model": "vinai/phowhisper-medium",
        "audio_seconds": audio_seconds,
        "profile": {"normalization_type": normalization_type},
        "arms": arms,
    }


def _participants(names: tuple[str, ...], arms: dict[str, dict], **session_kwargs) -> list[dict]:
    """One dev session per participant, with exactly the arms named here."""
    return [
        _session(
            name,
            f"{name}-s1",
            {arm: _arm(**kwargs) for arm, kwargs in arms.items()},
            **session_kwargs,
        )
        for name in names
    ]


_ALL = ("p1", "p2", "p3")


# --- gates ---


def test_gates_pass_exactly_at_the_95_percent_threshold():
    """19 valid vectors of 20 (95%) pass, on the same edge as the retention ratios."""
    records = [
        _session(
            "p1",
            "s1",
            {
                "P0": _arm(
                    reference_ms=1000,
                    retained_ms=950,
                    utterances=20,
                    retained_utterances=19,
                    egemaps_extracted=19,
                    egemaps_valid=18,
                )
            },
        )
    ]

    gates = evaluate_gates(records)["P0"]

    assert gates["checks"] == {
        "duration_retention": True,
        "utterance_retention": True,
        "feature_validity": True,
        "no_failed_session": True,
    }
    assert gates["features"]["validity"] == Fraction(19, 20)
    assert gates["passed"] is True


@pytest.mark.parametrize(
    "arm,expected",
    [
        ({"retained_ms": 949}, "duration_retention"),
        ({"retained_utterances": 18}, "utterance_retention"),
        ({"egemaps_extracted": 20, "egemaps_valid": 18}, "feature_validity"),
    ],
)
def test_each_gate_fails_just_below_its_threshold(arm, expected):
    records = [_session("p1", "s1", {"P0": _arm(**arm)})]

    gates = evaluate_gates(records)["P0"]

    assert gates["checks"][expected] is False
    assert gates["passed"] is False


def test_retention_checks_do_not_apply_to_the_vad_free_baseline():
    records = [_session("p1", "s1", {"N0": _arm()})]

    gates = evaluate_gates(records)["N0"]

    assert gates["checks"]["duration_retention"] is None
    assert gates["checks"]["utterance_retention"] is None
    assert gates["applicable"] == ["feature_validity", "no_failed_session"]
    assert gates["passed"] is True


def test_missing_evidence_fails_the_gate_instead_of_being_skipped():
    """An arm with no coverage or feature record must never pass a validity gate."""
    bare = {"segments": [], "warnings": [], "scores": {"syer": {"edits": 0, "ref_count": 100}}}
    records = [_session("p1", "s1", {"P0": _arm(), "PF": dict(bare)})]

    gates = evaluate_gates(records)["PF"]

    assert gates["checks"]["duration_retention"] is False
    assert gates["checks"]["feature_validity"] is False
    assert gates["checks"]["no_failed_session"] is False
    assert gates["passed"] is False

    decision = select_denoiser(records)

    assert decision["selected"] is None
    assert decision["eligible"] == []


def test_a_systematic_session_failure_blocks_the_gate_while_pooled_validity_is_high():
    broken = _session(
        "p1",
        "s1",
        {
            "P0": _arm(
                acoustic_values=10,
                acoustic_finite=0,
                acoustic_valid=False,
                egemaps_values=0,
                egemaps_finite=0,
                egemaps_utterances=0,
                egemaps_extracted=0,
                egemaps_valid=0,
            )
        },
    )
    healthy = _session("p2", "s2", {"P0": _arm(acoustic_values=1000, acoustic_finite=1000)})

    gates = evaluate_gates([broken, healthy])["P0"]

    assert float(gates["features"]["value_coverage"]) == pytest.approx(1176 / 1186)
    assert gates["features"]["failed_sessions"] == 1
    assert gates["checks"]["no_failed_session"] is False
    assert gates["passed"] is False


# --- aggregation, no-worse-than-P0, tie-break ---


def test_participant_syer_pools_sessions_rather_than_averaging_rates():
    records = [
        _session("p1", "s1", {"P0": _arm(edits=10, ref_count=100)}),
        _session("p1", "s2", {"P0": _arm(edits=30, ref_count=300)}),
    ]

    pooled = participant_syer(records)

    assert float(pooled["p1"]["P0"]) == pytest.approx(40 / 400)
    assert float(macro_mean(pooled["p1"])) == pytest.approx(0.1)


def test_macro_mean_is_none_without_participants():
    assert macro_mean({}) is None


def test_an_arm_with_no_scorable_output_is_never_selected():
    records = _participants(_ALL, {"P0": {"edits": 30, "ref_count": 100}, "PF": {"ref_count": 0}})

    decision = select_denoiser(records)

    assert decision["macro_syer"]["PF"] is None
    assert decision["unscorable"] == ["PF"]
    assert decision["selected"] is None
    assert "validity gates" in decision["reason"]


def test_candidate_exactly_equal_to_p0_is_no_worse_and_candidate_worse_is_rejected():
    equal = _participants(
        _ALL, {"P0": {"edits": 30, "ref_count": 100}, "PF": {"edits": 30, "ref_count": 100}}
    )

    decision = select_denoiser(equal)

    assert decision["selected"] == "PF"
    assert decision["no_worse_than_p0"] == ["PF"]

    worse = _participants(
        _ALL, {"P0": {"edits": 30, "ref_count": 100}, "PF": {"edits": 31, "ref_count": 100}}
    )

    rejected = select_denoiser(worse)

    assert rejected["selected"] is None
    assert "at least as good as P0" in rejected["reason"]


def test_no_configured_candidate_says_so_instead_of_blaming_the_gates():
    decision = select_denoiser(_participants(_ALL, {"P0": {"edits": 30, "ref_count": 100}}))

    assert decision["candidates"] == []
    assert decision["reason"] == "no denoiser arm was configured for this run"


def test_one_point_difference_is_a_tie_and_breaks_on_missed_utterances():
    records = _participants(
        _ALL,
        {
            "P0": {"edits": 400, "ref_count": 1000},
            "PF": {"edits": 300, "ref_count": 1000, "retained_utterances": 19},
            "PD": {"edits": 310, "ref_count": 1000, "retained_utterances": 20},
        },
    )

    decision = select_denoiser(records)

    assert decision["best"] == "PF"
    assert decision["contenders"] == ["PF", "PD"]
    assert decision["missed_utterances"] == {"PF": 3, "PD": 0}
    assert decision["selected"] == "PD"


def test_more_than_one_point_is_not_a_tie_and_the_better_arm_wins():
    records = _participants(
        _ALL,
        {
            "P0": {"edits": 400, "ref_count": 1000},
            "PF": {"edits": 300, "ref_count": 1000},
            "PD": {"edits": 311, "ref_count": 1000},
        },
    )

    decision = select_denoiser(records)

    assert decision["contenders"] == ["PF"]
    assert decision["tie_break"] == "clear SyER difference"
    assert decision["selected"] == "PF"


def test_tie_break_falls_back_to_runtime_per_audio_hour():
    records = _participants(
        _ALL,
        {
            "P0": {"edits": 400, "ref_count": 1000},
            "PF": {"edits": 300, "ref_count": 1000, "runtime_s": 100.0},
            "PD": {"edits": 300, "ref_count": 1000, "runtime_s": 50.0},
        },
    )

    decision = select_denoiser(records)

    assert decision["missed_utterances"] == {"PF": 0, "PD": 0}
    assert decision["runtime_s_per_audio_hour"] == {"PF": 100.0, "PD": 50.0}
    assert decision["selected"] == "PD"


def test_runtime_is_seconds_per_audio_hour():
    records = [_session("p1", "s1", {"P0": _arm(runtime_s=10.0)}, audio_seconds=1800.0)]

    assert runtime_per_audio_hour(records, "P0") == pytest.approx(20.0)


def test_a_missing_runtime_never_wins_the_tie_break():
    """An unmeasured arm must not look instantaneous."""
    records = _participants(
        _ALL,
        {
            "P0": {"edits": 400, "ref_count": 1000},
            "PF": {"edits": 300, "ref_count": 1000, "runtime_s": 100.0},
            "PD": {"edits": 300, "ref_count": 1000, "runtime_s": None},
        },
    )

    decision = select_denoiser(records)

    assert decision["runtime_s_per_audio_hour"] == {"PF": 100.0, "PD": None}
    assert decision["selected"] == "PF"


# --- bootstrap ---


def test_bootstrap_resamples_participants_not_sessions():
    """A session-level bootstrap would be dominated by the participant with 3 sessions."""
    participant_values = {"p1": Fraction(0), "p2": Fraction(1, 2), "p3": Fraction(1)}
    session_values = {
        "p1-s1": Fraction(0),
        "p1-s2": Fraction(0),
        "p1-s3": Fraction(0),
        "p2-s1": Fraction(1, 2),
        "p3-s1": Fraction(1),
    }
    records = [
        _session("p1", "p1-s1", {"P0": _arm()}),
        _session("p1", "p1-s2", {"P0": _arm()}),
        _session("p1", "p1-s3", {"P0": _arm()}),
        _session("p2", "p2-s1", {"P0": _arm()}),
        _session("p3", "p3-s1", {"P0": _arm()}),
    ]

    interval = bootstrap_mean_interval(participant_values)
    counterfactual = bootstrap_mean_interval(session_values)

    assert interval["participants"] == 3
    assert [round(value, 6) for value in interval["ci95"]] == [0.0, 1.0]
    assert counterfactual["ci95"][1] < 1.0
    assert interval["resamples"] == BOOTSTRAP_RESAMPLES
    assert interval["seed"] == BOOTSTRAP_SEED
    assert len(records) == 5  # sessions never enter the draw
    assert bootstrap_mean_interval(participant_values) == interval


def test_bootstrap_is_deterministic_for_the_frozen_seed():
    values = {f"p{i}": Fraction(i, 10) for i in range(6)}

    first = bootstrap_mean_interval(values, resamples=50, seed=42)

    assert bootstrap_mean_interval(values, resamples=50, seed=7)["ci95"] != first["ci95"]
    assert bootstrap_mean_interval(values, resamples=50, seed=42) == first
    assert bootstrap_mean_interval(values, resamples=0)["ci95"] is None


def test_sign_agreement_counts_participants_the_candidate_improved():
    # Lower SyER is better. p1, p2 improve (candidate < baseline); p3 does not.
    candidate = {"p1": Fraction(1, 10), "p2": Fraction(2, 10), "p3": Fraction(5, 10)}
    baseline = {"p1": Fraction(3, 10), "p2": Fraction(3, 10), "p3": Fraction(1, 10)}

    summary = paired_difference(candidate, baseline)

    assert summary["sign_agreement"] == pytest.approx(2 / 3)


def test_two_participant_split_reports_sign_agreement_instead_of_an_interval():
    candidate = {"p1": Fraction(1, 10), "p2": Fraction(4, 10)}
    baseline = {"p1": Fraction(3, 10), "p2": Fraction(2, 10)}

    summary = paired_difference(candidate, baseline)

    assert summary["participants"] == 2
    assert summary["bootstrap"] is None
    assert summary["sign_agreement"] == pytest.approx(0.5)
    assert summary["mean_difference"] == pytest.approx(0.0)
    assert "bootstrap" in summary["note"]


# --- split-scoped summary, selection, redaction ---


def test_no_denoiser_passing_the_gates_selects_none_and_says_so():
    records = _participants(
        _ALL,
        {
            "P0": {"edits": 400, "ref_count": 1000},
            "PF": {"edits": 100, "ref_count": 1000, "retained_ms": 500},
            "PD": {"edits": 100, "ref_count": 1000, "acoustic_valid": False},
        },
    )

    summary = build_summary(records)

    assert summary["selection"]["selected"] is None
    assert summary["selection"]["reason"] == "no denoiser passed the validity gates"
    assert summary["selection"]["eligible"] == []
    assert summary["selection"]["split"] == "dev"
    assert summary["splits"]["dev"]["arms"]["PF"]["gates_passed"] is False
    assert "no denoiser" in json.dumps(summary).lower()


def test_selection_uses_development_records_only():
    dev = _participants(
        _ALL, {"P0": {"edits": 400, "ref_count": 1000}, "PD": {"edits": 100, "ref_count": 1000}}
    )
    held_out = [
        _session(
            "p9",
            "s9",
            {"P0": _arm(edits=100, ref_count=1000), "PD": _arm(edits=900, ref_count=1000)},
            split="held_out",
        )
    ]

    summary = build_summary(dev + held_out)

    assert summary["selection"]["selected"] == "PD"
    assert summary["selection"]["macro_syer"]["PD"] == pytest.approx(0.1)
    assert summary["sessions"] == {"total": 4, "dev": 3, "held_out": 1}


def test_each_split_reports_its_own_aggregates():
    """The dev block and the held-out block must not be pooled into one number."""
    dev = _participants(
        _ALL, {"P0": {"edits": 400, "ref_count": 1000}, "PD": {"edits": 100, "ref_count": 1000}}
    )
    held_out = [
        _session(
            name,
            f"{name}-s1",
            {"P0": _arm(edits=800, ref_count=1000), "PD": _arm(edits=900, ref_count=1000)},
            split="held_out",
        )
        for name in ("p9", "p9b")
    ]

    summary = build_summary(dev + held_out)

    dev_block = summary["splits"]["dev"]
    held_block = summary["splits"]["held_out"]

    assert dev_block["arms"]["PD"]["syer_macro_mean"] == pytest.approx(0.1)
    assert held_block["arms"]["PD"]["syer_macro_mean"] == pytest.approx(0.9)
    assert held_block["arms"]["PD"]["versus_p0"]["mean_difference"] == pytest.approx(0.1)
    assert summary["selection"]["macro_syer"]["PD"] == pytest.approx(0.1)
    # Two held-out participants cannot support a bootstrap interval (SPEC).
    assert held_block["arms"]["PD"]["syer_bootstrap"] is None
    assert "uncertainty" in held_block["arms"]["PD"]
    assert held_block["identifiability_warning"].startswith("fewer than 3 participants")


def test_dynamic_loudness_fallback_is_flagged_in_the_summary():
    records = _participants(
        _ALL, {"P0": {"edits": 100, "ref_count": 1000}}, normalization_type="dynamic"
    )

    summary = build_summary(records)

    assert summary["splits"]["dev"]["normalization_type_counts"] == {"dynamic": 3}
    assert any("dynamic loudness" in note for note in summary["notes"])


def test_summary_and_written_file_are_redacted(tmp_path: Path):
    secret_participant = "participant-secret-1"
    secret_session = "session-secret-1"
    record = _session(
        secret_participant,
        secret_session,
        {"P0": _arm(edits=10, ref_count=100), "PF": _arm(edits=5, ref_count=100)},
    )
    record["reference_path"] = "/SECRETPRIVATE/p001.cha"
    record["audio_path"] = "/SECRETPRIVATE/p001_master.wav"
    record["arms"]["P0"]["segments"] = [{"start_ms": 0, "end_ms": 1000, "text": "một hai ba"}]

    summary = build_summary([record])
    target = write_summary(summary, tmp_path / "private-out")
    text = target.read_text(encoding="utf-8")
    lowered = text.lower()

    for forbidden in (
        secret_participant,
        secret_session,
        "một hai ba",
        "secretprivate",
        "/shared-data",
        ".wav",
        ".cha",
        "audio_path",
        "reference_path",
        "segments",
    ):
        assert forbidden not in lowered, f"summary leaked {forbidden!r}"

    assert json.loads(text)["participants"] == 1
    assert summary["splits"]["dev"]["arms"]["PF"]["versus_p0"]["mean_difference"] == pytest.approx(-0.05)


# --- ASR revision pinning and reference bounds ---


def test_backend_forwards_the_pinned_revision(monkeypatch):
    captured: dict = {}

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    transformers = ModuleType("transformers")
    transformers.pipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    PhoWhisperBackend(device="cpu", revision="abc123").load()

    assert captured["revision"] == "abc123"
    assert captured["model"] == "vinai/phowhisper-medium"


def test_backend_for_revision_reuses_one_backend_per_revision(monkeypatch):
    built: list[str] = []

    def fake_backend(*, device, revision):
        built.append(revision)
        return SimpleNamespace(device=device, revision=revision)

    monkeypatch.setattr("say_transcribe.study.PhoWhisperBackend", fake_backend)
    cache: dict = {}

    first = _backend_for_revision("rev-a", "cpu", cache)
    second = _backend_for_revision("rev-a", "cpu", cache)
    third = _backend_for_revision("rev-b", "cpu", cache)

    assert first is second
    assert third is not first
    assert built == ["rev-a", "rev-b"]


def _one_row_manifest(tmp_path: Path, reference_text: str) -> Path:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"stub")
    reference = tmp_path / "r.cha"
    reference.write_text(reference_text, encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "session_id": "s1",
                        "participant_id": "p1",
                        "audio_path": str(audio),
                        "channel_index": 0,
                        "sha256": "ab" * 32,
                        "split": "dev",
                        "reference_path": str(reference),
                        "asr_revision": "rev-a",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_study_uses_the_revision_pinned_by_each_manifest_row(tmp_path, monkeypatch):
    collected: list[str] = []

    def fake_backend(*, device, revision):
        collected.append(revision)
        return SimpleNamespace(model_id="vinai/phowhisper-medium")

    def fake_session(row, backend, denoisers, device):
        return {
            "session_id": row.session_id,
            "participant_id": row.participant_id,
            "split": row.split,
            "arms": {},
            "asr_revision": row.asr_revision,
        }

    monkeypatch.setattr("say_transcribe.study.PhoWhisperBackend", fake_backend)
    monkeypatch.setattr("say_transcribe.study.compute_session", fake_session)
    monkeypatch.setattr("say_transcribe.study.write_session_record", lambda record, out_dir: None)
    monkeypatch.setattr("say_transcribe.study.write_summary", lambda summary, out_dir: None)
    manifest = _one_row_manifest(
        tmp_path,
        "@UTF8\n@Begin\n@Languages:\tvie\n@Participants:\tPAR Participant\n"
        "@ID:\tvie|corpus|PAR|||||Participant|||\n@Media:\ts, audio\n"
        "*PAR:\tmột hai .\t\x150_1000\x15\n@End\n",
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["rows"].append(
        {**payload["rows"][0], "session_id": "s2", "participant_id": "p2", "asr_revision": "rev-b"}
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    assert run_study(manifest_path=manifest, out_dir=tmp_path / "out") == 0
    assert collected == ["rev-a", "rev-b"]


def test_reference_timing_past_the_audio_fails_loudly(tmp_path, monkeypatch):
    """A unit error in the reference must not read as near-zero coverage."""
    manifest = _one_row_manifest(
        tmp_path,
        "@UTF8\n@Begin\n@Languages:\tvie\n@Participants:\tPAR Participant\n"
        "@ID:\tvie|corpus|PAR|||||Participant|||\n@Media:\ts, audio\n"
        "*PAR:\tmột hai .\t\x150_60000\x15\n@End\n",
    )
    row = load_manifest(manifest)[0]
    monkeypatch.setattr("say_transcribe.study.verify_source", lambda row: "ab" * 32)
    monkeypatch.setattr("say_transcribe.study.require_egemaps_available", lambda: None)
    monkeypatch.setattr(
        "say_transcribe.study.compute_arm_results",
        lambda *args, **kwargs: SimpleNamespace(
            results={}, signals={"N1": np.zeros(16000)}, windows={}, seconds={}, loudness=None
        ),
    )

    with pytest.raises(StudyError) as excinfo:
        compute_session(row, object(), {})

    assert excinfo.value.code == "INVALID_ARGUMENT"
    assert "past the end" in excinfo.value.message
    assert excinfo.value.__cause__ is None
