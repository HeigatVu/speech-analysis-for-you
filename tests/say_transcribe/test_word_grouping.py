import subprocess
import sys
import unicodedata

import pytest

from say_transcribe.asr import AsrSegment, WordTiming
from say_transcribe.word_grouping import (
    WordGroupingError,
    group_utterance_words,
)


def test_lazy_import_does_not_load_underthesea():
    code = (
        "import sys, say_transcribe\n"
        "assert 'underthesea' not in sys.modules, 'underthesea was eagerly imported'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"Import test failed:\n{result.stderr}"


def test_grouping_preserves_nfc_tones_and_d_đ():
    text = "Đồng hồ và đàn dương cầm rất đẹp."
    tokens = ["Đồng_hồ", "và", "đàn_dương_cầm", "rất", "đẹp", "."]

    words = [
        WordTiming(word="Đồng", start_ms=0, end_ms=200),
        WordTiming(word="hồ", start_ms=210, end_ms=400),
        WordTiming(word="và", start_ms=450, end_ms=600),
        WordTiming(word="đàn", start_ms=650, end_ms=800),
        WordTiming(word="dương", start_ms=810, end_ms=1000),
        WordTiming(word="cầm", start_ms=1010, end_ms=1200),
        WordTiming(word="rất", start_ms=1250, end_ms=1400),
        WordTiming(word="đẹp", start_ms=1410, end_ms=1600),
        WordTiming(word=".", start_ms=1600, end_ms=1650),
    ]
    seg = AsrSegment(start_ms=0, end_ms=1650, text=text, words=tuple(words))

    res = group_utterance_words(seg, tokenizer=lambda t: tokens)
    assert len(res) == 6

    # Verify NFC normalization and character preservation
    for gw in res:
        assert unicodedata.is_normalized("NFC", gw.word)

    assert res[0].word == "Đồng_hồ"
    assert res[0].start_ms == 0
    assert res[0].end_ms == 400

    assert res[1].word == "và"
    assert res[1].start_ms == 450
    assert res[1].end_ms == 600

    assert res[2].word == "đàn_dương_cầm"
    assert res[2].start_ms == 650
    assert res[2].end_ms == 1200

    # Utterance-final punctuation carries no span
    assert res[5].word == "."
    assert res[5].start_ms is None
    assert res[5].end_ms is None


def test_grouping_never_merges_across_utterance_boundaries():
    # Two utterances: words in seg1 must never be combined with words in seg2
    seg1 = AsrSegment(
        start_ms=0,
        end_ms=500,
        text="học",
        words=(WordTiming(word="học", start_ms=0, end_ms=500),),
    )
    seg2 = AsrSegment(
        start_ms=600,
        end_ms=1100,
        text="sinh",
        words=(WordTiming(word="sinh", start_ms=600, end_ms=1100),),
    )

    res1 = group_utterance_words(seg1, tokenizer=lambda t: ["học"])
    res2 = group_utterance_words(seg2, tokenizer=lambda t: ["sinh"])

    assert len(res1) == 1
    assert res1[0].word == "học"
    assert res1[0].start_ms == 0
    assert res1[0].end_ms == 500

    assert len(res2) == 1
    assert res2[0].word == "sinh"
    assert res2[0].start_ms == 600
    assert res2[0].end_ms == 1100


def test_word_grouping_unaligned_raises_on_syllable_mismatch():
    # Input has 2 syllables ("học", "sinh") but tokenizer produced 3 syllables
    seg = AsrSegment(
        start_ms=0,
        end_ms=1000,
        text="học sinh",
        words=(
            WordTiming(word="học", start_ms=0, end_ms=400),
            WordTiming(word="sinh", start_ms=450, end_ms=900),
        ),
    )

    with pytest.raises(WordGroupingError) as exc_info:
        group_utterance_words(seg, tokenizer=lambda t: ["học_sinh_giỏi"])
    assert exc_info.value.code == "WORD_GROUPING_UNALIGNED"

    # Syllable text mismatch
    with pytest.raises(WordGroupingError) as exc_info:
        group_utterance_words(seg, tokenizer=lambda t: ["giáo_viên"])
    assert exc_info.value.code == "WORD_GROUPING_UNALIGNED"


def test_grouping_without_word_timings_succeeds():
    seg = AsrSegment(
        start_ms=0,
        end_ms=1000,
        text="sinh viên",
        words=(),
    )
    res = group_utterance_words(seg, tokenizer=lambda t: ["sinh_viên"])
    assert len(res) == 1
    assert res[0].word == "sinh_viên"
    assert res[0].start_ms is None
    assert res[0].end_ms is None


def test_glued_punctuation_splits_and_aligns():
    from say_transcribe.asr import AsrSegment, WordTiming

    seg = AsrSegment(
        start_ms=0,
        end_ms=900,
        text="hở ,",
        words=(WordTiming(word="hở,", start_ms=100, end_ms=500),),
    )
    grouped = group_utterance_words(seg, tokenizer=lambda t: ["hở", ","])
    assert [g.word for g in grouped] == ["hở", ","]
    assert (grouped[0].start_ms, grouped[0].end_ms) == (100, 500)
    assert (grouped[1].start_ms, grouped[1].end_ms) == (None, None)
