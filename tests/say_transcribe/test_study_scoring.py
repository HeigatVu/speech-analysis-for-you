import json
import unicodedata

from say_transcribe.evaluate import (
    extract_session_items,
    items_from_texts,
    run_evaluation,
    score_uncapped,
)


def test_score_uncapped_returns_rates_above_one_for_insertions():
    gold = items_from_texts(["tôi là"])
    hyp = items_from_texts(["tôi là sinh viên năm hai ba"])

    scores = score_uncapped(gold, hyp)

    assert scores["syer"] == {"edits": 5, "ref_count": 2, "rate": 2.5}
    assert scores["syer"]["rate"] > 1.0


def test_score_uncapped_counts_deletions_and_substitutions():
    gold = items_from_texts(["một hai ba"])
    hyp = items_from_texts(["một bốn"])

    scores = score_uncapped(gold, hyp)

    # one substitution (hai->bốn), one deletion (ba)
    assert scores["syer"] == {"edits": 2, "ref_count": 3, "rate": 2 / 3}


def test_score_uncapped_zero_for_identical_texts():
    gold = items_from_texts(["tôi là sinh_viên"])
    hyp = items_from_texts(["tôi là sinh_viên"])

    scores = score_uncapped(gold, hyp)

    for metric in scores.values():
        assert metric["edits"] == 0
        assert metric["rate"] == 0.0
    assert scores["syer"]["ref_count"] == 4


def test_score_uncapped_empty_reference_keeps_insertion_count():
    gold = items_from_texts([])
    hyp = items_from_texts(["tôi là"])

    scores = score_uncapped(gold, hyp)

    assert scores["syer"] == {"edits": 2, "ref_count": 0, "rate": 0.0}


def test_items_from_texts_matches_extract_session_items_conventions():
    # Every _PUNCTUATION_ONLY token, so a mismatch between the two tokenizers'
    # punctuation sets (gold vs hypothesis) can't slip past this test. Needs a
    # full CHAT header (@Participants/@ID/@Media) so decode_chat takes the
    # strict path instead of falling back to the lenient main-tier scan.
    text = "tôi là sinh_viên . ? ! , ... …"
    cha_text = (
        "@UTF8\n@Begin\n@Languages:\tvie\n"
        "@Participants:\tPAR Participant, INV Investigator\n"
        "@ID:\tvie|corpus|PAR|||||Participant|||\n"
        "@ID:\tvie|corpus|INV|||||Investigator|||\n"
        "@Media:\ts01, audio\n"
        f"*PAR:\t{text}\t\x150_2000\x15\n@End\n"
    )

    from_texts = items_from_texts([text])
    from_cha = extract_session_items(cha_text)

    for key in ("syllables", "words", "chars"):
        assert from_texts[key] == from_cha[key]
    assert from_cha["words"] == ["tôi", "là", "sinh_viên"]


def test_items_from_texts_preserves_nfc_vietnamese_text():
    decomposed = unicodedata.normalize("NFD", "tiếng Việt đúng")
    items = items_from_texts([decomposed])

    assert items["words"] == ["tiếng", "Việt", "đúng"]
    assert items["syllables"] == ["tiếng", "Việt", "đúng"]
    for word in items["words"]:
        assert word == unicodedata.normalize("NFC", word)
    assert "đúng" in items["words"]


def test_run_evaluation_caps_at_one_are_unchanged(tmp_path):
    gold_dir = tmp_path / "gold"
    pred_dir = tmp_path / "pred"
    gold_dir.mkdir()
    pred_dir.mkdir()
    (gold_dir / "s1.cha").write_text("@Begin\n*PAR:\ttôi là .\n@End\n", encoding="utf-8")
    (pred_dir / "s1.cha").write_text(
        "@Begin\n*PAR:\ttôi là sinh viên năm hai ba .\n@End\n", encoding="utf-8"
    )

    report_path = tmp_path / "report.json"
    assert run_evaluation(gold_dir, pred_dir, report_path) == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    # Same pair scores 2.5 uncapped; the legacy path must keep clamping at 1.0.
    for metric in ("syer", "cer", "wer"):
        assert report["metrics"][metric]["mean"] == 1.0
    assert score_uncapped(
        extract_session_items((gold_dir / "s1.cha").read_text(encoding="utf-8")),
        extract_session_items((pred_dir / "s1.cha").read_text(encoding="utf-8")),
    )["syer"]["rate"] == 2.5
