import json
import unicodedata
from pathlib import Path

from say_transcribe.evaluate import (
    compute_der,
    extract_session_items,
    levenshtein,
    run_evaluation,
)


def test_levenshtein_distance():
    assert levenshtein([], []) == 0
    assert levenshtein(["a", "b"], ["a", "b"]) == 0
    assert levenshtein(["a", "b"], ["a", "c"]) == 1
    assert levenshtein(["a"], ["a", "b"]) == 1
    assert levenshtein(["a", "b"], ["a"]) == 1


def test_der_computation():
    ref = [(0, 1000, "PAR"), (1100, 2000, "INV")]
    hyp = [(0, 1000, "PAR"), (1100, 2000, "INV")]
    # Perfect match -> DER 0.0
    assert compute_der(ref, hyp, collar_ms=100) == 0.0

    # Opposite speaker -> DER 1.0
    hyp_opp = [(0, 1000, "INV"), (1100, 2000, "PAR")]
    assert compute_der(ref, hyp_opp, collar_ms=100) == 1.0


def test_extract_session_items_scores_an_untimed_draft_without_raising():
    draft_cha = """@Begin
*PAR:\ttôi là sinh_viên .
@End
"""
    items = extract_session_items(draft_cha)

    assert items["words"] == ["tôi", "là", "sinh_viên"]
    assert items["syllables"] == ["tôi", "là", "sinh", "viên"]
    assert items["intervals"] == []


def test_extract_session_items_scans_main_tier_despite_malformed_headers():
    malformed_cha = """@Window:\t0_1000
@Begin
@Lnaugage:\tvie
*PAR:\tem chào \x150_500\x15
@End
"""
    items = extract_session_items(malformed_cha)

    assert items["words"] == ["em", "chào"]


def test_run_evaluation_synthetic_pairs(tmp_path: Path):
    gold_dir = tmp_path / "gold"
    pred_dir = tmp_path / "pred"
    gold_dir.mkdir()
    pred_dir.mkdir()

    gold_cha = """@UTF8
@Begin
@Languages:\tvie
@Participants:\tPAR Participant, INV Investigator
@ID:\tvie|corpus|PAR|||||Participant|||
@ID:\tvie|corpus|INV|||||Investigator|||
@Media:\ts01, audio
*PAR:\ttôi là sinh_viên .\t\x150_2000\x15
%wor:\ttôi \x150_400\x15 là \x15400_800\x15 sinh_viên \x15800_1800\x15 .
%mor:\tpron|tôi aux|là noun|sinh_viên .
%gra:\t1|3|NSUBJ 2|3|COP 3|0|ROOT 4|3|PUNCT
@End
"""

    pred_cha = """@UTF8
@Begin
@Languages:\tvie
@Participants:\tPAR Participant, INV Investigator
@ID:\tvie|corpus|PAR|||||Participant|||
@ID:\tvie|corpus|INV|||||Investigator|||
@Media:\ts01, audio
*PAR:\ttôi là sinh_viên .\t\x150_2000\x15
%wor:\ttôi \x150_400\x15 là \x15400_800\x15 sinh_viên \x15800_1800\x15 .
%mor:\tpron|tôi aux|là noun|sinh_viên .
%gra:\t1|3|NSUBJ 2|3|COP 3|0|ROOT 4|3|PUNCT
@End
"""

    (gold_dir / "s01.cha").write_text(gold_cha, encoding="utf-8")
    (pred_dir / "s01.cha").write_text(pred_cha, encoding="utf-8")

    out_json = tmp_path / "report.json"
    ret = run_evaluation(gold_dir, pred_dir, out_json, bootstrap_samples=100, seed=42)
    assert ret == 0
    assert out_json.exists()

    report_str = out_json.read_text(encoding="utf-8")
    report = json.loads(report_str)

    assert report["sessions_evaluated"] == 1
    assert report["format_pass_rate"] == 1.0
    assert report["metrics"]["syer"]["mean"] == 0.0
    assert report["metrics"]["wer"]["mean"] == 0.0
    assert report["metrics"]["pos_accuracy"] == 1.0
    assert report["metrics"]["uas"] == 1.0
    assert report["metrics"]["las"] == 1.0

    # Ensure no transcript words or raw timestamps leak in the report
    assert "tôi" not in report_str
    assert "sinh_viên" not in report_str
    assert "0_2000" not in report_str


def test_deterministic_bootstrap_seeded(tmp_path: Path):
    gold_dir = tmp_path / "gold"
    pred_dir = tmp_path / "pred"
    gold_dir.mkdir()
    pred_dir.mkdir()

    for i in range(2):
        gold = f"""@UTF8
@Begin
@Languages:\tvie
@Participants:\tPAR Participant, INV Investigator
@ID:\tvie|corpus|PAR|||||Participant|||
@ID:\tvie|corpus|INV|||||Investigator|||
@Media:\ts0{i}, audio
*PAR:\thọc_sinh .\t\x150_1000\x15
%mor:\tnoun|học_sinh .
%gra:\t1|0|ROOT 2|1|PUNCT
@End
"""
        pred = f"""@UTF8
@Begin
@Languages:\tvie
@Participants:\tPAR Participant, INV Investigator
@ID:\tvie|corpus|PAR|||||Participant|||
@ID:\tvie|corpus|INV|||||Investigator|||
@Media:\ts0{i}, audio
*PAR:\tgiáo_viên .\t\x150_1000\x15
%mor:\tnoun|giáo_viên .
%gra:\t1|0|ROOT 2|1|PUNCT
@End
"""
        (gold_dir / f"s0{i}.cha").write_text(gold, encoding="utf-8")
        (pred_dir / f"s0{i}.cha").write_text(pred, encoding="utf-8")

    out1 = tmp_path / "rep1.json"
    out2 = tmp_path / "rep2.json"
    run_evaluation(gold_dir, pred_dir, out1, bootstrap_samples=500, seed=123)
    run_evaluation(gold_dir, pred_dir, out2, bootstrap_samples=500, seed=123)

    assert out1.read_text() == out2.read_text()


_SPOKEN_HEADER = """@UTF8
@Begin
@Languages:\tvie
@Participants:\tPAR Participant
@ID:\tvie|corpus|PAR|||||Participant|||
@Media:\tp001, audio
"""


def _spoken_document(main_tier: str, bullet: str = "\x150_1000\x15") -> str:
    return f"{_SPOKEN_HEADER}*PAR:\t{main_tier}\t{bullet}\n@End\n"


def test_spoken_domain_keeps_fillers_and_retraces_without_annotation_syntax():
    items = extract_session_items(
        _spoken_document("tôi đi <tôi đi> [/] tôi đi &-ờ xxx [: x] (.) .")
    )

    assert items["words"] == ["tôi", "đi", "tôi", "đi", "tôi", "đi", "ờ"]
    assert items["syllables"] == ["tôi", "đi", "tôi", "đi", "tôi", "đi", "ờ"]
    assert items["chars"] == list("tôiđitôiđitôiđiờ")


def test_spoken_domain_keeps_vietnamese_surface_intact():
    items = extract_session_items(_spoken_document("đi &-ờ &-đ &-vâng ."))

    assert items["words"] == ["đi", "ờ", "đ", "vâng"]
    assert items["words"][1] == "\u1edd"
    assert items["words"][2] == "\u0111"
    assert all(word == unicodedata.normalize("NFC", word) for word in items["words"])


def test_uncertainty_queue_reports_codes_and_positions_only():
    items = extract_session_items(_spoken_document("tôi xxx &+ư &~ờ [: nhà] ."))

    assert items["uncertainty"] == [
        {"utterance_index": 0, "code": "NONWORD_FRAGMENT"},
        {"utterance_index": 0, "code": "REPLACEMENT_ANNOTATION"},
        {"utterance_index": 0, "code": "UNTRANSCRIBED_SPAN"},
    ]
    for entry in items["uncertainty"]:
        assert set(entry) == {"utterance_index", "code"}


def test_uncertainty_queue_does_not_repeat_an_already_reported_code():
    items = extract_session_items(_spoken_document("xxx xxx xxx ."))

    assert items["uncertainty"] == [{"utterance_index": 0, "code": "UNTRANSCRIBED_SPAN"}]


def test_zero_duration_utterance_enters_the_uncertainty_queue():
    items = extract_session_items(_spoken_document("tôi .", bullet="\x150_0\x15"))

    assert items["uncertainty"] == [{"utterance_index": 0, "code": "ZERO_DURATION_UTTERANCE"}]


def test_timed_document_without_markup_has_an_empty_uncertainty_queue():
    items = extract_session_items(_spoken_document("tôi đi học ."))

    assert items["uncertainty"] == []


_TWO_SPEAKER_HEADER = _SPOKEN_HEADER.replace(
    "PAR Participant", "PAR Participant, INV Investigator"
).replace("@Media", "@ID:\tvie|corpus|INV|||||Investigator|||\n@Media")


def test_a_mostly_timed_prediction_stays_strictly_parsed_and_markup_is_not_scored():
    """One untimed utterance must not push the whole file onto the text-only path,
    where `[/]`, `<` and `&-` would be scored as words and DER would be lost."""
    text = (
        f"{_TWO_SPEAKER_HEADER}"
        "*PAR:\ttôi đi <tôi đi> [/] tôi đi .\t\x150_4000\x15\n"
        "*INV:\tvâng .\n"
        "@End\n"
    )
    items = extract_session_items(text)

    assert "[/]" not in items["words"] and "<tôi" not in items["words"]
    assert items["words"].count("tôi") == 3 and "vâng" in items["words"]
    assert items["intervals"] == [(0, 4000, "PAR")]  # the untimed utterance has no interval


def _write_pair(tmp_path: Path, hyp_text: str):
    gold_dir, pred_dir = tmp_path / "gold", tmp_path / "pred"
    gold_dir.mkdir()
    pred_dir.mkdir()
    (gold_dir / "s01.cha").write_text(
        f"{_TWO_SPEAKER_HEADER}*PAR:\ttôi đi học .\t\x150_4000\x15\n"
        "*INV:\tvâng .\t\x154000_6000\x15\n@End\n",
        encoding="utf-8",
    )
    (pred_dir / "s01.cha").write_text(hyp_text, encoding="utf-8")
    return gold_dir, pred_dir


def test_der_is_scored_when_part_of_the_prediction_is_untimed(tmp_path: Path):
    gold_dir, pred_dir = _write_pair(
        tmp_path,
        f"{_TWO_SPEAKER_HEADER}*PAR:\ttôi đi học .\t\x150_4000\x15\n*INV:\tvâng .\n@End\n",
    )
    report = tmp_path / "r.json"
    run_evaluation(gold_dir, pred_dir, report)
    metrics = json.loads(report.read_text())

    assert metrics["metrics"]["syer"]["mean"] == 0.0  # markup-free, strictly parsed
    assert metrics["der_sessions_scored"] == 1
    assert 0.0 < metrics["metrics"]["der"]["mean"] < 1.0  # INV's 2 s is missed, PAR is right


def test_der_is_reported_absent_not_one_when_the_prediction_has_no_timing(tmp_path: Path):
    gold_dir, pred_dir = _write_pair(
        tmp_path,
        f"{_TWO_SPEAKER_HEADER}*PAR:\ttôi đi học .\n*INV:\tvâng .\n@End\n",
    )
    report = tmp_path / "r.json"
    run_evaluation(gold_dir, pred_dir, report)
    metrics = json.loads(report.read_text())

    assert metrics["der_sessions_scored"] == 0
    assert metrics["metrics"]["der"]["mean"] is None
