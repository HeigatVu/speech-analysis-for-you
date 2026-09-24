import json
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
