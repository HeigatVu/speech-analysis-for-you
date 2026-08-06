"""Behavioral tests for Vietnamese lexical/disfluency features (Task 3).

Participant-only transcript measures built from the Task 1 immutable
``Transcript`` contract. Covers Unicode NFC/casefold normalization that
preserves diacritics and ``đ``; explicit token classes (word/filler/fragment/
noise); token/character counts and mean token length; lexical diversity (TTR,
hapax ratio, Brunet W, Honore R) with safe NaNs on degenerate input; repetition
and filler rates; and a simple external function-word-set ratio.
"""

import math

import pytest

import speech_features.linguistic as linguistic
from speech_features.schema import Token, Transcript, Utterance, validate_transcript


def _utterance(speaker, *token_kinds_and_texts):
    tokens = []
    for kind, text in token_kinds_and_texts:
        tokens.append(Token(kind=kind, text=text))
    return Utterance(speaker=speaker, start_s=0.0, end_s=1.0, tokens=tokens)


def _transcript(*utterances):
    return Transcript(utterances=utterances)


class TestNormaliseToken:
    def test_casefolds_and_nfc_normalises(self):
        # "GIÔ" (latin capital I + O + combining circumflex) normalises to NFC
        # ("giô") and is casefolded to lowercase.
        assert linguistic.normalise_token("GI\u00d4") == "gi\u00f4"

    def test_preserves_diacritics_and_d_across_casefold(self):
        # Vietnamese uppercase "Đồng" casefolds to "đồng" (Đ -> đ), keeping the
        # distinct d/đ graphemes intact (never conflated by NFC or casefold).
        assert linguistic.normalise_token("Đồng") == "\u0111\u1ed3ng"
        d_dong = linguistic.normalise_token("Đồng")
        no_d_dong = linguistic.normalise_token("Dồng")
        assert d_dong != no_d_dong  # đ vs d are distinct tokens

    def test_is_idempotent(self):
        t = linguistic.normalise_token("GiÔng")
        assert linguistic.normalise_token(t) == t


class TestTokenExtraction:
    def test_participant_only_excludes_examiner(self):
        tr = _transcript(
            _utterance("examiner", ("word", "instruction")),
            _utterance(
                "participant",
                ("word", "Mèo"),
                ("word", "con"),
                ("filler", "ừm"),
                ("word", "chạy"),
            ),
        )
        words = linguistic.participant_word_tokens(tr)
        assert [t.text for t in words] == ["Mèo", "con", "chạy"]

    def test_participant_tokens_keep_all_kinds(self):
        tr = _transcript(
            _utterance(
                "participant",
                ("word", "con"),
                ("filler", "ơ"),
                ("fragment", "chu"),
                ("noise", "[cough]"),
            )
        )
        kinds = [t.kind for t in linguistic.participant_tokens(tr)]
        assert kinds == ["word", "filler", "fragment", "noise"]

    def test_empty_participant_speech_yields_empty(self):
        tr = _transcript(_utterance("examiner", ("word", "go")))
        assert linguistic.participant_tokens(tr) == ()


class TestRawCountsAndLength:
    def test_token_and_word_counts(self):
        tr = _transcript(
            _utterance(
                "participant",
                ("word", "Mèo"),
                ("filler", "ừm"),
                ("word", "con"),
                ("noise", "[:]"),
            )
        )
        stats = linguistic.token_stats(tr)
        assert stats["lex_token_count"] == 4
        assert stats["lex_word_count"] == 2
        assert stats["lex_char_count"] == 6  # Mèo(3) + con(3); filler/n"ừm" excluded
        assert stats["lex_mean_token_length"] == pytest.approx(3.0)

    def test_char_count_counts_only_word_tokens(self):
        tr = _transcript(
            _utterance(
                "participant",
                ("word", "nhà"),
                ("filler", "à"),
            ),
        )
        stats = linguistic.token_stats(tr)
        assert stats["lex_char_count"] == 3
        assert stats["lex_fragment_count"] == 0
        assert stats["lex_noise_count"] == 0

    def test_mean_token_length_zero_for_no_words(self):
        tr = _transcript(_utterance("participant", ("filler", "ừm")))
        stats = linguistic.token_stats(tr)
        assert math.isnan(stats["lex_mean_token_length"])


class TestLexicalDiversity:
    def test_ttr_hapax_basic(self):
        # 4 words, 3 unique, 2 hapaxes
        stats = linguistic.extract_linguistic(
            _transcript(
                _utterance(
                    "participant",
                    ("word", "con"),
                    ("word", "con"),
                    ("word", "mèo"),
                    ("word", "chó"),
                )
            )
        )
        assert stats["lex_word_count"] == 4
        assert stats["lex_unique_count"] == 3
        assert stats["lex_ttr"] == pytest.approx(3 / 4)
        # con appears twice, meo and cho once each -> 2 types with freq 1
        assert stats["lex_hapax_ratio"] == pytest.approx(2 / 4)

    def test_brunet_w(self):
        # W = V^0.172
        stats = linguistic.extract_linguistic(
            _transcript(
                _utterance(
                    "participant",
                    ("word", "a"),
                    ("word", "b"),
                    ("word", "c"),
                    ("word", "d"),
                )
            )
        )
        assert stats["lex_brunet_w"] == pytest.approx(4**0.172)

    def test_honore_r(self):
        # R = 100 * log(N) / (1 - V1/V); types freq-1 / types
        stats = linguistic.extract_linguistic(
            _transcript(
                _utterance(
                    "participant",
                    ("word", "a"),
                    ("word", "a"),
                    ("word", "b"),
                    ("word", "c"),
                )
            )
        )
        N, V, V1 = 4, 3, 2
        expected = 100 * math.log(N) / (1 - V1 / V)
        assert stats["lex_honore_r"] == pytest.approx(expected)

    def test_degenerate_no_words_returns_nan(self):
        tr = _transcript(_utterance("participant", ("filler", "ừm")))
        stats = linguistic.extract_linguistic(tr)
        assert math.isnan(stats["lex_ttr"])
        assert math.isnan(stats["lex_hapax_ratio"])
        assert math.isnan(stats["lex_brunet_w"])
        assert math.isnan(stats["lex_honore_r"])

    def test_single_word_brunet_and_honore(self):
        # V1 == V (1/1) makes the Honore denominator zero -> NaN, not inf/zero
        stats = linguistic.extract_linguistic(
            _transcript(_utterance("participant", ("word", "mèo")))
        )
        assert stats["lex_brunet_w"] == pytest.approx(1**0.172)
        assert math.isnan(stats["lex_honore_r"])


class TestDisfluencyAndFunctionWords:
    def test_repetition_ratio_counts_adjacent_duplicate_words(self):
        stats = linguistic.extract_linguistic(
            _transcript(
                _utterance(
                    "participant",
                    ("word", "con"),
                    ("word", "con"),
                    ("word", "mèo"),
                    ("word", "mèo"),
                    ("word", "chó"),
                )
            )
        )
        # 2 adjacent repetitions out of 5 words
        assert stats["lex_repetition_ratio"] == pytest.approx(2 / 5)

    def test_repetition_ratio_ignores_spaced_repeat(self):
        stats = linguistic.extract_linguistic(
            _transcript(
                _utterance(
                    "participant",
                    ("word", "con"),
                    ("word", "mèo"),
                    ("word", "con"),
                )
            )
        )
        assert stats["lex_repetition_ratio"] == pytest.approx(0.0)

    def test_repetition_ratio_nan_for_no_words(self):
        tr = _transcript(_utterance("participant", ("filler", "ừm")))
        assert math.isnan(linguistic.extract_linguistic(tr)["lex_repetition_ratio"])

    def test_filler_ratio(self):
        tr = _transcript(
            _utterance(
                "participant",
                ("word", "con"),
                ("filler", "ừm"),
                ("word", "mèo"),
                ("filler", "ơ"),
            )
        )
        stats = linguistic.extract_linguistic(tr)
        # fillers / all participant tokens
        assert stats["lex_filler_ratio"] == pytest.approx(2 / 4)
        assert stats["lex_fragment_ratio"] == pytest.approx(0.0)
        assert stats["lex_noise_ratio"] == pytest.approx(0.0)

    def test_function_word_ratio(self):
        tr = _transcript(
            _utterance(
                "participant",
                ("word", "và"),
                ("word", "con"),
                ("word", "mèo"),
                ("word", "và"),
            )
        )
        stats = linguistic.extract_linguistic(tr, function_words={"và", "của"})
        assert stats["lex_function_word_ratio"] == pytest.approx(2 / 4)

    def test_function_word_ratio_matches_normalised_forms(self):
        # Function-word set is NFC/casefold keys; participant "VÀ" folds to "và".
        tr = _transcript(_utterance("participant", ("word", "VÀ")))
        stats = linguistic.extract_linguistic(tr, function_words={"và"})
        assert stats["lex_function_word_ratio"] == pytest.approx(1.0)

    def test_function_word_ratio_nan_for_no_words(self):
        tr = _transcript(_utterance("participant", ("filler", "ừm")))
        stats = linguistic.extract_linguistic(tr, function_words={"và"})
        assert math.isnan(stats["lex_function_word_ratio"])


class TestTranscriptInputFlexible:
    def test_accepts_validated_transcript_dict(self):
        raw = {
            "version": 1,
            "utterances": [
                {
                    "speaker": "participant",
                    "start_s": 0.0,
                    "end_s": 1.0,
                    "tokens": [{"kind": "word", "text": "Mèo"}],
                }
            ],
        }
        stats = linguistic.extract_linguistic(validate_transcript(raw))
        assert stats["lex_word_count"] == 1
