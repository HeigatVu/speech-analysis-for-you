"""Behavioral tests for Vietnamese lexical/disfluency features (Task 3).

Participant-only transcript measures built from the Task 1 immutable
``Transcript`` contract. Covers Unicode NFC/casefold normalization that
preserves diacritics and ``đ``; explicit token classes (word/filler/fragment/
noise); token/character counts and mean token length; lexical diversity (TTR,
hapax ratio, Brunet W, Honore R) with safe NaNs on degenerate input; repetition
and filler rates; and a simple external function-word-set ratio.
"""

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

import speech_features.linguistic as linguistic
from speech_features.catalog import list_features
from speech_features.document import (
    AnnotationLayer,
    DocumentSpeaker,
    DocumentToken,
    DocumentUtterance,
    SpeechDocument,
)
from speech_features.features.linguistic import extract_lexical_features
from speech_features.result import InvalidConfigError, TargetSpeakerRequiredError
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


# ---------------------------------------------------------------------------
# Task 8: adult-neuro lexical/disfluency features over SpeechDocument
# ---------------------------------------------------------------------------

ADULT_NEURO_KEYS = (
    "lex_utterance_count",
    "lex_token_count",
    "lex_word_count",
    "lex_syllable_count",
    "lex_character_count",
    "lex_unique_token_count",
    "lex_mlu_words",
    "lex_mlu_syllables",
    "lex_token_length_mean_characters",
    "lex_token_length_sd_characters",
    "lex_word_length_mean_syllables",
    "lex_word_length_sd_syllables",
    "lex_token_ttr",
    "lex_token_mattr_20",
    "lex_token_mtld",
    "lex_token_hdd_42",
    "lex_token_hapax_ratio",
    "lex_token_brunet_w",
    "lex_token_honore_r",
    "lex_token_entropy",
    "lex_lemma_ttr",
    "lex_lemma_mattr_20",
    "lex_lemma_mtld",
    "lex_lemma_hdd_42",
    "lex_lemma_hapax_ratio",
    "lex_lemma_brunet_w",
    "lex_lemma_honore_r",
    "lex_lemma_entropy",
    "disfluency_filler_count",
    "disfluency_filler_ratio",
    "disfluency_fragment_count",
    "disfluency_fragment_ratio",
    "disfluency_immediate_repetition_count",
    "disfluency_immediate_repetition_ratio",
    "disfluency_retracing_count",
    "disfluency_retracing_ratio",
    "disfluency_revision_count",
    "disfluency_revision_ratio",
    "disfluency_maze_count",
    "disfluency_maze_ratio",
    "disfluency_annotated_error_count",
    "disfluency_annotated_error_ratio",
)


def _token(tid, text, kind="word", word_id=None):
    return DocumentToken(id=tid, text=text, kind=kind, word_id=word_id)


def _rich_vi_document():
    """Hand-calculated Vietnamese fixture.

    14 word-kind tokens -> 12 explicit words (xe/đạp share word_id inside one
    utterance and again in the next; shared word_id never groups across
    utterances). Word forms: mèo x2, xe x2, đạp x2, and 8 hapax types:
    con, chạy, nhanh, đồng, dồng, và, khác, đi -> N=14, V=11, V1=8.
    Lemma forms: mèo x2, đồng x2, xe x2, đạp x2 + con, chạy, nhanh, và,
    khác, đi -> N=14, V=10, V1=6 (dồng and đồng share the lemma "đồng").
    """
    utt1 = DocumentUtterance(
        id="u1",
        speaker_id="p1",
        start_s=0.0,
        end_s=10.0,
        tokens=(
            _token("t1", "Mèo"),
            _token("t2", "mèo"),
            _token("t3", "con"),
            _token("t4", "chạy"),
            _token("t5", "nhanh"),
            _token("t6", "Đồng"),
            _token("t7", "và"),
            _token("t8", "Dồng"),
            _token("t9", "khác"),
            _token("t10", "xe", word_id="w_xe"),
            _token("t11", "đạp", word_id="w_xe"),
            _token("t12", "ừm", kind="filler"),
            _token("t13", "chu", kind="fragment"),
            _token("t14", "[/]", kind="retracing"),
            _token("t15", "[//]", kind="revision"),
            _token("t16", "[*]", kind="error"),
        ),
    )
    utt2 = DocumentUtterance(
        id="u2",
        speaker_id="p1",
        start_s=11.0,
        end_s=12.0,
        tokens=(
            _token("t17", "xe", word_id="w_xe"),
            _token("t18", "đạp", word_id="w_xe"),
            _token("t19", "ơ", kind="filler"),
            _token("t20", "đi"),
        ),
    )
    utt3 = DocumentUtterance(
        id="u3",
        speaker_id="ex1",
        start_s=0.0,
        end_s=5.0,
        tokens=(
            _token("t21", "nhìn"),
            _token("t22", "chỉ"),
            _token("t23", "vào"),
            _token("t24", "ừm", kind="filler"),
        ),
    )
    lemmas = {
        "t1": "mèo",
        "t2": "mèo",
        "t3": "con",
        "t4": "chạy",
        "t5": "nhanh",
        "t6": "đồng",
        "t7": "và",
        "t8": "đồng",
        "t9": "khác",
        "t10": "xe",
        "t11": "đạp",
        "t17": "xe",
        "t18": "đạp",
        "t20": "đi",
        "t21": "nhìn",
        "t22": "chỉ",
        "t23": "vào",
    }
    return SpeechDocument(
        document_id="rec1",
        speakers=(DocumentSpeaker(id="p1"), DocumentSpeaker(id="ex1")),
        utterances=(utt1, utt2, utt3),
        annotations=(AnnotationLayer(layer="lemma", source="reviewed", values=lemmas),),
    )


def _mattr_mtld_hdd_document():
    """84 word-kind tokens: mèo, và x80, Đồng, xe, đạp (xe/đạp grouped).

    N=84, V=5 (mèo, và, đồng, xe, đạp), V1=4. Hand-calculated windows and
    MTLD factors in the tests below.
    """
    tokens = [_token("t1", "Mèo")]
    tokens.extend(_token(f"t{i}", "và") for i in range(2, 82))
    tokens.extend(
        [
            _token("t82", "Đồng"),
            _token("t83", "xe", word_id="w_xe"),
            _token("t84", "đạp", word_id="w_xe"),
        ]
    )
    return SpeechDocument(
        document_id="rec2",
        speakers=(DocumentSpeaker(id="p1"),),
        utterances=(
            DocumentUtterance(
                id="u1", speaker_id="p1", start_s=0.0, end_s=20.0, tokens=tuple(tokens)
            ),
        ),
    )


def _single_token_document():
    return SpeechDocument(
        document_id="rec3",
        speakers=(DocumentSpeaker(id="p1"),),
        utterances=(
            DocumentUtterance(
                id="u1",
                speaker_id="p1",
                start_s=0.0,
                end_s=1.0,
                tokens=(_token("t1", "mèo"),),
            ),
        ),
    )


def _no_sample_document():
    """Only an examiner utterance: the target has no language sample."""
    return SpeechDocument(
        document_id="rec4",
        speakers=(DocumentSpeaker(id="p1"), DocumentSpeaker(id="ex1")),
        utterances=(
            DocumentUtterance(
                id="u1",
                speaker_id="ex1",
                start_s=0.0,
                end_s=5.0,
                tokens=(_token("t1", "nhìn"),),
            ),
        ),
    )


def _issues_by_feature(issues):
    by_feature = {}
    for issue in issues:
        assert issue.severity == "warning"
        assert issue.feature is not None
        by_feature.setdefault(issue.feature, []).append(issue)
    return by_feature


def _assert_nan_issue_pairing(features, issues):
    """Every NaN has exactly one matching per-key issue and vice versa."""
    by_feature = _issues_by_feature(issues)
    for key, value in features.items():
        if math.isnan(value):
            assert key in by_feature, f"NaN {key} without an issue"
            assert len(by_feature[key]) == 1, f"NaN {key} with {len(by_feature[key])} issues"
            del by_feature[key]
        else:
            assert key not in by_feature, f"computed {key} has a stray issue"
    assert by_feature == {}, f"issues without matching NaN keys: {sorted(by_feature)}"


class TestAdultNeuroCountsMluAndLengths:
    def test_hand_calculated_counts_machine_and_grouping(self):
        features, _ = extract_lexical_features(_rich_vi_document(), target_speaker="p1")
        assert features["lex_utterance_count"] == 2.0
        assert features["lex_token_count"] == 20.0
        assert features["lex_word_count"] == 12.0
        assert features["lex_syllable_count"] == 14.0
        assert features["lex_character_count"] == 44.0
        assert features["lex_unique_token_count"] == 11.0
        assert features["lex_mlu_words"] == pytest.approx(6.0)
        assert features["lex_mlu_syllables"] == pytest.approx(7.0)

    def test_word_id_never_groups_across_utterances(self):
        # "xe đạp" appears in u1 and again in u2 with the same word_id:
        # two separate words, four syllable tokens.
        features, _ = extract_lexical_features(_rich_vi_document(), target_speaker="p1")
        assert features["lex_word_count"] == 12.0
        assert features["lex_syllable_count"] == 14.0

    def test_unicode_equality_and_d_diacritic_preserved(self):
        # "Mèo"/"mèo" are the same type; "Đồng"/"Dồng" stay distinct and keep
        # their diacritics and grapheme identity in the character counts.
        features, _ = extract_lexical_features(_rich_vi_document(), target_speaker="p1")
        assert features["lex_unique_token_count"] == 11.0
        assert features["lex_character_count"] == 44.0
        assert features["lex_token_ttr"] == pytest.approx(11 / 14)

    def test_token_and_word_length_summaries_population_sd(self):
        features, _ = extract_lexical_features(_rich_vi_document(), target_speaker="p1")
        lengths = [3, 3, 3, 4, 5, 4, 2, 4, 4, 2, 3, 2, 3, 2]
        mean = 44 / 14
        sd = (sum((length - mean) ** 2 for length in lengths) / len(lengths)) ** 0.5
        assert features["lex_token_length_mean_characters"] == pytest.approx(mean)
        assert features["lex_token_length_sd_characters"] == pytest.approx(sd)
        word_mean = 14 / 12  # 10 single-syllable words + 2 two-syllable groups
        word_sd = ((10 * (1 - 14 / 12) ** 2 + 2 * (2 - 14 / 12) ** 2) / 12) ** 0.5
        assert features["lex_word_length_mean_syllables"] == pytest.approx(word_mean)
        assert features["lex_word_length_sd_syllables"] == pytest.approx(word_sd)


class TestAdultNeuroSurfaceDiversity:
    def test_hand_calculated_base_diversity_formulas(self):
        features, _ = extract_lexical_features(_rich_vi_document(), target_speaker="p1")
        # N=14, V=11 (mèo x2, xe x2, đạp x2, 8 hapax), V1=8.
        assert features["lex_token_ttr"] == pytest.approx(11 / 14)
        assert features["lex_token_hapax_ratio"] == pytest.approx(8 / 14)
        assert features["lex_token_brunet_w"] == pytest.approx(14 ** (11**-0.165))
        assert features["lex_token_honore_r"] == pytest.approx(100 * math.log(14) / (1 - 8 / 11))
        expected_entropy = -(
            3 * (2 / 14) * math.log(2 / 14) + 8 * (1 / 14) * math.log(1 / 14)
        ) / math.log(11)
        assert features["lex_token_entropy"] == pytest.approx(expected_entropy)

    def test_mattr_20_sliding_window_hand_calculation(self):
        # 84 forms: mèo, và x80, đồng, xe, đạp. Window i covers [i, i+20):
        # i=0 -> 2 types; i=1..61 -> 1 type; i=62 -> 2; i=63 -> 3; i=64 -> 4.
        features, _ = extract_lexical_features(_mattr_mtld_hdd_document())
        expected = (0.1 + 61 * 0.05 + 0.1 + 0.15 + 0.2) / 65
        assert features["lex_token_mattr_20"] == pytest.approx(expected)

    def test_mtld_072_forward_reverse_hand_calculation(self):
        # Forward: first close at the second "và" (TTR 2/3), then 39 more
        # closes on và-pairs, trailing "Đồng xe đạp" all-distinct (TTR 1,
        # partial 0) -> 84/40. Reverse: 39 closes + trailing partial 0 ->
        # 84/39. Reported value is the mean.
        features, _ = extract_lexical_features(_mattr_mtld_hdd_document())
        expected = (84 / 40 + 84 / 39) / 2
        assert features["lex_token_mtld"] == pytest.approx(expected)

    def test_hdd_42_hand_calculation(self):
        # N=84: và has N-fi=4 < 42 -> ratio 0; each hapax type has
        # C(83,42)/C(84,42) = 42/84 = 0.5. HD-D = (4*0.5 + 1)/42 = 1/14.
        features, _ = extract_lexical_features(_mattr_mtld_hdd_document())
        assert features["lex_token_hdd_42"] == pytest.approx(1 / 14)

    def test_large_fixture_counts_and_diversity(self):
        features, _ = extract_lexical_features(_mattr_mtld_hdd_document())
        assert features["lex_utterance_count"] == 1.0
        assert features["lex_token_count"] == 84.0
        assert features["lex_word_count"] == 83.0
        assert features["lex_syllable_count"] == 84.0
        assert features["lex_character_count"] == 172.0
        assert features["lex_unique_token_count"] == 5.0
        assert features["lex_mlu_words"] == pytest.approx(83.0)
        assert features["lex_mlu_syllables"] == pytest.approx(84.0)
        assert features["lex_token_ttr"] == pytest.approx(5 / 84)
        assert features["lex_token_hapax_ratio"] == pytest.approx(4 / 84)
        assert features["lex_token_brunet_w"] == pytest.approx(84 ** (5**-0.165))
        assert features["lex_token_honore_r"] == pytest.approx(100 * math.log(84) / (1 - 4 / 5))
        expected_entropy = -(
            4 * (1 / 84) * math.log(1 / 84) + (80 / 84) * math.log(80 / 84)
        ) / math.log(5)
        assert features["lex_token_entropy"] == pytest.approx(expected_entropy)
        lengths = [3] + [2] * 80 + [4, 2, 3]
        mean = 172 / 84
        sd = (sum((length - mean) ** 2 for length in lengths) / len(lengths)) ** 0.5
        assert features["lex_token_length_mean_characters"] == pytest.approx(mean)
        assert features["lex_token_length_sd_characters"] == pytest.approx(sd)
        word_mean = 84 / 83
        word_sd = ((82 * (1 - 84 / 83) ** 2 + (2 - 84 / 83) ** 2) / 83) ** 0.5
        assert features["lex_word_length_mean_syllables"] == pytest.approx(word_mean)
        assert features["lex_word_length_sd_syllables"] == pytest.approx(word_sd)

    def test_single_token_entropy_defined_as_zero_and_degenerate_na_ns(self):
        features, _ = extract_lexical_features(_single_token_document())
        assert features["lex_token_entropy"] == 0.0
        assert features["lex_token_ttr"] == pytest.approx(1.0)
        assert features["lex_token_hapax_ratio"] == pytest.approx(1.0)
        assert features["lex_token_brunet_w"] == pytest.approx(1.0)
        # Single-type samples: MTLD has a zero factor denominator and Honore's
        # divisor 1 - V1/V vanishes; both are NaN, never zero or inf.
        assert math.isnan(features["lex_token_mtld"])
        assert math.isnan(features["lex_token_honore_r"])
        assert math.isnan(features["lex_token_mattr_20"])
        assert math.isnan(features["lex_token_hdd_42"])


class TestAdultNeuroLemmaDiversity:
    def test_complete_lemma_layer_hand_calculated(self):
        features, issues = extract_lexical_features(_rich_vi_document(), target_speaker="p1")
        # Lemma forms: mèo x2, đồng x2, xe x2, đạp x2 + 6 hapax -> N=14,
        # V=10, V1=6. "Dồng" and "Đồng" share the lemma "đồng".
        assert features["lex_lemma_ttr"] == pytest.approx(10 / 14)
        assert features["lex_lemma_hapax_ratio"] == pytest.approx(6 / 14)
        assert features["lex_lemma_brunet_w"] == pytest.approx(14 ** (10**-0.165))
        assert features["lex_lemma_honore_r"] == pytest.approx(100 * math.log(14) / (1 - 6 / 10))
        expected_entropy = -(
            4 * (2 / 14) * math.log(2 / 14) + 6 * (1 / 14) * math.log(1 / 14)
        ) / math.log(10)
        assert features["lex_lemma_entropy"] == pytest.approx(expected_entropy)
        assert math.isnan(features["lex_lemma_mattr_20"])
        assert math.isnan(features["lex_lemma_hdd_42"])
        # Forward: one close on the mèo pair, trailing 12 tokens/9 types
        # (TTR 9/12, partial (1-9/12)/0.28). Reverse: one close at
        # (đạp, xe, đạp, xe) -> trailing 9 tokens/7 types (TTR 7/9).
        expected_lemma_mtld = (14 / (1 + (1 - 9 / 12) / 0.28) + 14 / (1 + (1 - 7 / 9) / 0.28)) / 2
        assert features["lex_lemma_mtld"] == pytest.approx(expected_lemma_mtld)
        codes = [issue.code for issue in issues if issue.feature == "lex_lemma_mattr_20"]
        assert codes == ["INSUFFICIENT_TOKENS"]

    def test_absent_lemma_layer_is_missing_annotation(self):
        features, issues = extract_lexical_features(_mattr_mtld_hdd_document())
        for key in (
            "lex_lemma_ttr",
            "lex_lemma_mattr_20",
            "lex_lemma_mtld",
            "lex_lemma_hdd_42",
            "lex_lemma_hapax_ratio",
            "lex_lemma_brunet_w",
            "lex_lemma_honore_r",
            "lex_lemma_entropy",
        ):
            assert math.isnan(features[key])
        by_feature = _issues_by_feature(issues)
        for key in (
            "lex_lemma_ttr",
            "lex_lemma_mattr_20",
            "lex_lemma_mtld",
            "lex_lemma_hdd_42",
            "lex_lemma_hapax_ratio",
            "lex_lemma_brunet_w",
            "lex_lemma_honore_r",
            "lex_lemma_entropy",
        ):
            assert by_feature[key][0].code == "MISSING_ANNOTATION"

    def test_incomplete_lemma_layer_is_missing_annotation(self):
        document = _rich_vi_document()
        layer = next(a for a in document.annotations if a.layer == "lemma")
        values = dict(layer.values)
        del values["t20"]
        document = SpeechDocument(
            document_id=document.document_id,
            speakers=document.speakers,
            utterances=document.utterances,
            annotations=(AnnotationLayer(layer="lemma", source="reviewed", values=values),),
        )
        features, issues = extract_lexical_features(document, target_speaker="p1")
        assert math.isnan(features["lex_lemma_ttr"])
        assert math.isnan(features["lex_lemma_entropy"])
        by_feature = _issues_by_feature(issues)
        for key in ("lex_lemma_ttr", "lex_lemma_entropy"):
            assert by_feature[key][0].code == "MISSING_ANNOTATION"

    def test_non_string_lemma_value_is_missing_annotation(self):
        document = _rich_vi_document()
        layer = next(a for a in document.annotations if a.layer == "lemma")
        values = dict(layer.values)
        values["t20"] = 42
        document = SpeechDocument(
            document_id=document.document_id,
            speakers=document.speakers,
            utterances=document.utterances,
            annotations=(AnnotationLayer(layer="lemma", source="reviewed", values=values),),
        )
        features, issues = extract_lexical_features(document, target_speaker="p1")
        assert math.isnan(features["lex_lemma_ttr"])
        by_feature = _issues_by_feature(issues)
        assert by_feature["lex_lemma_ttr"][0].code == "MISSING_ANNOTATION"


class TestAdultNeuroDisfluency:
    def test_hand_calculated_kind_counts_and_ratios(self):
        features, _ = extract_lexical_features(_rich_vi_document(), target_speaker="p1")
        assert features["disfluency_filler_count"] == 2.0
        assert features["disfluency_fragment_count"] == 1.0
        assert features["disfluency_retracing_count"] == 1.0
        assert features["disfluency_revision_count"] == 1.0
        assert features["disfluency_annotated_error_count"] == 1.0
        # 20 explicit tokens: fillers/retracing/revision/errors are 1/20.
        assert features["disfluency_filler_ratio"] == pytest.approx(2 / 20)
        assert features["disfluency_fragment_ratio"] == pytest.approx(1 / 20)
        assert features["disfluency_retracing_ratio"] == pytest.approx(1 / 20)
        assert features["disfluency_revision_ratio"] == pytest.approx(1 / 20)
        assert features["disfluency_annotated_error_ratio"] == pytest.approx(1 / 20)

    def test_immediate_repetition_is_utterance_local_with_syllable_denominator(self):
        # Only "Mèo mèo" (u1) repeats; "xe đạp" pairs never repeat and nothing
        # crosses utterance boundaries. The ratio divides by 14 syllables, not
        # by the 20-token count.
        features, _ = extract_lexical_features(_rich_vi_document(), target_speaker="p1")
        assert features["disfluency_immediate_repetition_count"] == 1.0
        assert features["disfluency_immediate_repetition_ratio"] == pytest.approx(1 / 14)

    def test_maze_formula_and_ratio_denominator(self):
        # Maze = fillers + fragments + immediate repetitions + retracings +
        # revisions = 2+1+1+1+1; errors and noise are excluded.
        features, _ = extract_lexical_features(_rich_vi_document(), target_speaker="p1")
        assert features["disfluency_maze_count"] == 6.0
        assert features["disfluency_maze_ratio"] == pytest.approx(6 / 20)

    def test_absent_event_kinds_are_known_zero_with_sample(self):
        features, issues = extract_lexical_features(_single_token_document())
        assert features["disfluency_filler_count"] == 0.0
        assert features["disfluency_retracing_count"] == 0.0
        assert features["disfluency_annotated_error_count"] == 0.0
        assert features["disfluency_immediate_repetition_count"] == 0.0
        assert features["disfluency_maze_count"] == 0.0
        assert features["disfluency_maze_ratio"] == 0.0
        assert not any(issue.feature.startswith("disfluency_") for issue in issues)

    def test_no_word_tokens_repetition_ratio_insufficient(self):
        document = SpeechDocument(
            document_id="rec5",
            speakers=(DocumentSpeaker(id="p1"),),
            utterances=(
                DocumentUtterance(
                    id="u1",
                    speaker_id="p1",
                    start_s=0.0,
                    end_s=1.0,
                    tokens=(_token("t1", "ừm", kind="filler"),),
                ),
            ),
        )
        features, issues = extract_lexical_features(document)
        assert features["disfluency_filler_count"] == 1.0
        assert features["disfluency_filler_ratio"] == pytest.approx(1.0)
        assert math.isnan(features["disfluency_immediate_repetition_ratio"])
        by_feature = _issues_by_feature(issues)
        assert by_feature["disfluency_immediate_repetition_ratio"][0].code == "INSUFFICIENT_TOKENS"


class TestAdultNeuroTargetIsolation:
    def test_examiner_content_never_affects_values(self):
        # u3's nhìn/chỉ/vào/ừm must not leak into any feature.
        features, _ = extract_lexical_features(_rich_vi_document(), target_speaker="p1")
        assert features["lex_token_count"] == 20.0
        assert features["lex_word_count"] == 12.0
        assert features["lex_utterance_count"] == 2.0
        assert features["lex_unique_token_count"] == 11.0
        assert features["disfluency_filler_count"] == 2.0
        assert features["lex_lemma_ttr"] == pytest.approx(10 / 14)

    def test_single_speaker_is_inferred(self):
        features, _ = extract_lexical_features(_mattr_mtld_hdd_document())
        assert features["lex_utterance_count"] == 1.0

    def test_explicit_target_wins_over_inference(self):
        features, _ = extract_lexical_features(_rich_vi_document(), target_speaker="p1")
        assert features["lex_utterance_count"] == 2.0

    def test_multiple_speakers_without_target_raise(self):
        with pytest.raises(TargetSpeakerRequiredError) as exc:
            extract_lexical_features(_rich_vi_document())
        assert exc.value.code == "TARGET_SPEAKER_REQUIRED"

    def test_unknown_target_raises_invalid_config(self):
        with pytest.raises(InvalidConfigError) as exc:
            extract_lexical_features(_rich_vi_document(), target_speaker="nobody")
        assert exc.value.code == "INVALID_CONFIG"


class TestAdultNeuroMissingSamplesAndIssues:
    def test_no_target_sample_all_keys_nan_with_one_issue_each(self):
        features, issues = extract_lexical_features(_no_sample_document(), target_speaker="p1")
        assert set(features) == set(ADULT_NEURO_KEYS)
        assert all(math.isnan(value) for value in features.values())
        assert len(issues) == 42
        by_feature = _issues_by_feature(issues)
        assert set(by_feature) == set(ADULT_NEURO_KEYS)
        assert all(issue.code == "MISSING_ANNOTATION" for issue in issues)
        _assert_nan_issue_pairing(features, issues)

    def test_nan_issue_pairing_on_rich_fixture(self):
        features, issues = extract_lexical_features(_rich_vi_document(), target_speaker="p1")
        assert set(features) == set(ADULT_NEURO_KEYS)
        _assert_nan_issue_pairing(features, issues)

    def test_nan_issue_pairing_on_single_token_fixture(self):
        features, issues = extract_lexical_features(_single_token_document())
        _assert_nan_issue_pairing(features, issues)

    def test_insufficient_tokens_codes_for_small_surface_sample(self):
        features, issues = extract_lexical_features(_single_token_document())
        by_feature = _issues_by_feature(issues)
        for key in (
            "lex_token_mattr_20",
            "lex_token_mtld",
            "lex_token_hdd_42",
            "lex_token_honore_r",
            "lex_token_length_sd_characters",
            "lex_word_length_sd_syllables",
        ):
            assert by_feature[key][0].code == "INSUFFICIENT_TOKENS"


class TestAdultNeuroCatalog:
    def test_exactly_42_adult_neuro_keys_with_locked_metadata(self):
        definitions = list_features(pack="adult_neuro")
        assert [d.key for d in definitions] == sorted(ADULT_NEURO_KEYS)
        for definition in definitions:
            assert definition.pack == "adult_neuro"
            assert definition.level == "recording"
            assert definition.population == "adult"
            assert definition.formula_version == 1
            assert definition.reference == "SAY catalog v1"

    def test_fresh_import_registers_exactly_42_keys(self):
        src = Path(__file__).resolve().parents[2] / "src"
        code = (
            "import speech_features as sf\n"
            f"expected = {sorted(ADULT_NEURO_KEYS)!r}\n"
            "keys = [f.key for f in sf.list_features(pack='adult_neuro')]\n"
            "assert len(keys) == len(set(keys)), 'duplicate aliases registered'\n"
            "assert keys == expected, (keys, expected)\n"
            "print(len(keys))\n"
        )
        env = {**os.environ, "PYTHONPATH": str(src)}
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            cwd=src.parent,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == "42"
