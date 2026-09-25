"""T7: gates, exact tie-breaking, participant bootstrap, and the redacted summary."""

import json
from pathlib import Path

import pytest

from say_transcribe.study import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    bootstrap_mean_interval,
    build_summary,
    evaluate_gates,
    macro_mean,
    paired_difference,
    participant_syer,
    runtime_per_audio_hour,
    select_denoiser,
    write_summary,
)


def _arm(
    *,
    edits: int = 0,
    ref_count: int = 100,
    runtime_s: float = 10.0,
    reference_ms: int = 1000,
    retained_ms: int = 1000,
    utterances: int = 20,
    retained_utterances: int = 20,
    acoustic_values: int = 100,
    acoustic_finite: int = 100,
    egemaps_values: int = 100,
    egemaps_finite: int = 100,
    egemaps_utterances: int = 2,
    egemaps_valid: int = 2,
) -> dict:
    return {
        "segments": [],
        "warnings": [],
        "scores": {"syer": {"edits": edits, "ref_count": ref_count, "rate": edits / ref_count}},
        "runtime_s": runtime_s,
        "vad": {
            "reference_speech_ms": reference_ms,
            "retained_speech_ms": retained_ms,
            "utterance_count": utterances,
            "retained_utterance_count": retained_utterances,
        },
        "features": {
            "acoustic": {"values": acoustic_values, "finite": acoustic_finite},
            "egemaps": {
                "utterances": egemaps_utterances,
                "valid_utterances": egemaps_valid,
                "values": egemaps_values,
                "finite": egemaps_finite,
            },
        },
    }


def _session(
    participant: str,
    session: str,
    arms: dict[str, dict],
    *,
    split: str = "dev",
    audio_seconds: float = 3600.0,
) -> dict:
    return {
        "session_id": session,
        "participant_id": participant,
        "split": split,
        "channel_index": 0,
        "sha256": "ab" * 32,
        "asr_revision": "rev1",
        "audio_seconds": audio_seconds,
        "arms": arms,
    }


def _participants(names: tuple[str, ...], arms: dict[str, dict]) -> list[dict]:
    """One dev session per participant, with exactly the arms named here."""
    return [
        _session(name, f"{name}-s1", {arm: _arm(**kwargs) for arm, kwargs in arms.items()})
        for name in names
    ]


_ALL = ("p1", "p2", "p3")


# --- gates ---


def test_gates_pass_exactly_at_the_95_percent_threshold():
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
                    acoustic_values=100,
                    acoustic_finite=95,
                    egemaps_values=100,
                    egemaps_finite=95,
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
    assert gates["passed"] is True


@pytest.mark.parametrize(
    "arm,expected",
    [
        ({"retained_ms": 949}, "duration_retention"),
        ({"retained_utterances": 18}, "utterance_retention"),
        ({"acoustic_finite": 94, "egemaps_finite": 94}, "feature_validity"),
    ],
)
def test_each_gate_fails_just_below_its_threshold(arm, expected):
    records = [_session("p1", "s1", {"P0": _arm(**arm)})]

    gates = evaluate_gates(records)["P0"]

    assert gates["checks"][expected] is False
    assert gates["passed"] is False


def test_a_systematic_session_failure_blocks_the_gate_while_pooled_validity_is_high():
    broken = _session("p1", "s1", {"P0": _arm(acoustic_values=10, acoustic_finite=0, egemaps_values=0, egemaps_finite=0, egemaps_utterances=0, egemaps_valid=0)})
    healthy = _session("p2", "s2", {"P0": _arm(acoustic_values=1000, acoustic_finite=1000)})

    gates = evaluate_gates([broken, healthy])["P0"]

    assert float(gates["features"]["validity"]) == pytest.approx(1100 / 1110)
    assert gates["checks"]["feature_validity"] is True
    assert gates["checks"]["no_failed_session"] is False
    assert gates["passed"] is False
    assert gates["features"]["failed_sessions"] == 1


# --- aggregation and no-worse-than-P0 ---


def test_participant_syer_pools_sessions_rather_than_averaging_rates():
    records = [
        _session("p1", "s1", {"P0": _arm(edits=10, ref_count=100)}),
        _session("p1", "s2", {"P0": _arm(edits=30, ref_count=300)}),
    ]

    pooled = participant_syer(records)

    assert float(pooled["p1"]["P0"]) == pytest.approx(40 / 400)
    assert float(macro_mean(pooled["p1"])) == pytest.approx(0.1)


def test_candidate_exactly_equal_to_p0_is_no_worse_and_candidate_worse_is_rejected():
    equal = _participants(_ALL, {"P0": {"edits": 30, "ref_count": 100}, "PF": {"edits": 30, "ref_count": 100}})

    decision = select_denoiser(equal)

    assert decision["selected"] == "PF"
    assert decision["no_worse_than_p0"] == ["PF"]

    worse = _participants(_ALL, {"P0": {"edits": 30, "ref_count": 100}, "PF": {"edits": 31, "ref_count": 100}})

    rejected = select_denoiser(worse)

    assert rejected["selected"] is None
    assert "at least as good as P0" in rejected["reason"]


# --- practical tie at exactly one percentage point ---


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
    assert decision["selected"] == "PD"
    assert decision["missed_utterances"] == {"PF": 3, "PD": 0}


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
    assert decision["selected"] == "PF"
    assert decision["tie_break"] == "clear SyER difference"


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


# --- bootstrap ---


def test_bootstrap_resamples_participants_not_sessions_or_utterances():
    values = {"p1": 0, "p2": 0.5, "p3": 1}
    from fractions import Fraction

    values = {name: Fraction(value) for name, value in values.items()}
    sessions = [_session(name, f"{name}-s1", {"P0": _arm()}) for name in values]

    interval = bootstrap_mean_interval(values)

    # Only participant-level resampling can draw an all-p1 or all-p3 sample, so
    # the interval must reach the participant extremes; session- or
    # utterance-level resampling would be far narrower.
    assert interval["participants"] == len(values)
    assert interval["resamples"] == BOOTSTRAP_RESAMPLES
    assert interval["seed"] == BOOTSTRAP_SEED
    assert interval["ci95"][0] == pytest.approx(0.0, abs=1e-9)
    assert interval["ci95"][1] == pytest.approx(1.0, abs=1e-9)
    assert interval["mean"] == pytest.approx(0.5)
    assert len(sessions) == 3
    assert bootstrap_mean_interval(values) == interval


def test_bootstrap_is_deterministic_for_the_frozen_seed():
    from fractions import Fraction

    values = {f"p{i}": Fraction(i, 10) for i in range(6)}

    first = bootstrap_mean_interval(values, resamples=50, seed=42)
    second = bootstrap_mean_interval(values, resamples=50, seed=7)

    assert first["ci95"] != second["ci95"]
    assert bootstrap_mean_interval(values, resamples=50, seed=42) == first


def test_two_participant_split_reports_sign_agreement_instead_of_an_interval():
    from fractions import Fraction

    candidate = {"p1": Fraction(1, 10), "p2": Fraction(4, 10)}
    baseline = {"p1": Fraction(3, 10), "p2": Fraction(2, 10)}

    summary = paired_difference(candidate, baseline)

    assert summary["participants"] == 2
    assert summary["bootstrap"] is None
    assert summary["sign_agreement"] == pytest.approx(0.5)
    assert summary["mean_difference"] == pytest.approx(0.0)
    assert "bootstrap" in summary["note"]


# --- selection failure path and summary ---


def test_no_denoiser_passing_the_gates_selects_none_and_says_so():
    records = _participants(
        _ALL,
        {
            "P0": {"edits": 400, "ref_count": 1000},
            "PF": {"edits": 100, "ref_count": 1000, "retained_ms": 500},
            "PD": {"edits": 100, "ref_count": 1000, "acoustic_finite": 0},
        },
    )

    summary = build_summary(records)

    assert summary["selection"]["selected"] is None
    assert summary["selection"]["reason"] == "no denoiser passed the validity gates"
    assert summary["selection"]["eligible"] == []
    assert summary["arms"]["PF"]["gates_passed"] is False
    assert "no denoiser" in json.dumps(summary).lower()


def test_selection_uses_development_records_only():
    dev = _participants(_ALL, {"P0": {"edits": 400, "ref_count": 1000}, "PD": {"edits": 100, "ref_count": 1000}})
    held_out = [
        _session("p9", "s9", {"P0": _arm(edits=100, ref_count=1000), "PD": _arm(edits=900, ref_count=1000)}, split="held_out")
    ]

    summary = build_summary(dev + held_out)

    assert summary["selection"]["selected"] == "PD"
    assert summary["selection"]["macro_syer"]["PD"] == pytest.approx(0.1)
    assert summary["sessions"] == {"total": 4, "dev": 3, "held_out": 1}


def test_summary_and_written_file_are_redacted(tmp_path: Path):
    secret_participant = "participant-secret-1"
    secret_session = "session-secret-1"
    record = _session(
        secret_participant,
        secret_session,
        {"P0": _arm(edits=10, ref_count=100), "PF": _arm(edits=5, ref_count=100)},
    )
    record["arms"]["P0"]["segments"] = [{"start_ms": 0, "end_ms": 1000, "text": "một hai ba"}]

    summary = build_summary([record])
    target = write_summary(summary, tmp_path / "private-out")
    text = target.read_text(encoding="utf-8")
    lowered = text.lower()

    for forbidden in (
        secret_participant,
        secret_session,
        "một hai ba",
        "/shared-data",
        ".wav",
        ".cha",
        "audio_path",
        "reference_path",
        "segments",
    ):
        assert forbidden not in lowered, f"summary leaked {forbidden!r}"

    assert json.loads(text)["participants"] == 1
    assert summary["arms"]["PF"]["versus_p0"]["mean_difference"] == pytest.approx(-0.05)
