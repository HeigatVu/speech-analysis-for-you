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
from speech_features.features.linguistic import (
    extract_adult_neuro_features,
    extract_lexical_features,
    extract_morphosyntax_features,
)
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
    def test_exactly_97_adult_neuro_keys_with_locked_metadata(self):
        definitions = list_features(pack="adult_neuro")
        assert [d.key for d in definitions] == sorted(FULL_ADULT_NEURO_KEYS)
        by_key = {d.key: d for d in definitions}
        for definition in definitions:
            assert definition.pack == "adult_neuro"
            assert definition.population == "adult"
            assert definition.formula_version == 1
            assert definition.reference == "SAY catalog v1"
        for key in TASK9_RECORDING_KEYS:
            assert by_key[key].level == "recording"
        for key in DISCOURSE_UTTERANCE_KEYS:
            assert by_key[key].level == "utterance"
        for key, unit in TASK9_UNITS.items():
            assert by_key[key].unit == unit
        # Every count/ratio disfluency pair declares its count dependency,
        # including the syllable-denominated immediate-repetition ratio.
        assert by_key["disfluency_immediate_repetition_ratio"].prerequisites == (
            "disfluency_immediate_repetition_count",
        )
        # Exact Task 9 prerequisites fixed by the pre-review.
        assert by_key["morph_clause_rate_per_utterance"].prerequisites == (
            "morph_clause_count",
            "lex_utterance_count",
        )
        assert by_key["morph_subordination_ratio"].prerequisites == (
            "morph_subordinate_clause_count",
            "morph_clause_count",
        )
        assert by_key["discourse_overlap_ratio"].prerequisites == ("discourse_overlap_s",)
        for key in TASK9_KEYS:
            if key not in (
                "morph_clause_rate_per_utterance",
                "morph_subordination_ratio",
                "discourse_overlap_ratio",
            ):
                assert by_key[key].prerequisites == ()
        # Brunet W is a dimensionless lexical-richness index, not a count.
        assert by_key["lex_token_brunet_w"].unit == "index"
        assert by_key["lex_lemma_brunet_w"].unit == "index"

    def test_fresh_import_registers_exactly_97_keys(self):
        src = Path(__file__).resolve().parents[2] / "src"
        code = (
            "import speech_features as sf\n"
            f"expected = {sorted(FULL_ADULT_NEURO_KEYS)!r}\n"
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
        assert completed.stdout.strip() == "97"


# ---------------------------------------------------------------------------
# Task 9: adult-neuro morphosyntax and conversation features
# ---------------------------------------------------------------------------

TASK9_UPOS_KEYS = (
    "morph_upos_adj_ratio",
    "morph_upos_adp_ratio",
    "morph_upos_adv_ratio",
    "morph_upos_aux_ratio",
    "morph_upos_cconj_ratio",
    "morph_upos_det_ratio",
    "morph_upos_intj_ratio",
    "morph_upos_noun_ratio",
    "morph_upos_num_ratio",
    "morph_upos_part_ratio",
    "morph_upos_pron_ratio",
    "morph_upos_propn_ratio",
    "morph_upos_punct_ratio",
    "morph_upos_sconj_ratio",
    "morph_upos_sym_ratio",
    "morph_upos_verb_ratio",
    "morph_upos_x_ratio",
)

TASK9_DEP_KEYS = (
    "morph_dep_root_ratio",
    "morph_dep_subject_ratio",
    "morph_dep_object_ratio",
    "morph_dep_nominal_modifier_ratio",
    "morph_dep_adverbial_modifier_ratio",
    "morph_dep_clausal_complement_ratio",
    "morph_dep_coordination_ratio",
    "morph_dep_function_ratio",
    "morph_dep_other_ratio",
)

TASK9_COMPOSITION_KEYS = (
    "morph_content_word_ratio",
    "morph_function_word_ratio",
    "morph_noun_verb_ratio",
    "morph_pronoun_noun_ratio",
    "morph_classifier_ratio",
    "morph_particle_ratio",
    "morph_code_switch_ratio",
)

TASK9_STRUCTURE_KEYS = (
    "morph_dependency_length_mean_tokens",
    "morph_dependency_length_sd_tokens",
    "morph_tree_depth_mean",
    "morph_tree_depth_max",
    "morph_clause_count",
    "morph_clause_rate_per_utterance",
    "morph_subordinate_clause_count",
    "morph_subordination_ratio",
)

DISCOURSE_RECORDING_KEYS = (
    "discourse_turn_count",
    "discourse_turn_length_mean_words",
    "discourse_turn_length_sd_words",
    "discourse_turn_length_mean_syllables",
    "discourse_turn_length_sd_syllables",
    "discourse_examiner_prompt_ratio",
    "discourse_response_latency_mean_s",
    "discourse_response_latency_sd_s",
    "discourse_overlap_s",
    "discourse_overlap_ratio",
)

DISCOURSE_UTTERANCE_KEYS = (
    "discourse_turn_word_count",
    "discourse_turn_syllable_count",
    "discourse_response_latency_s",
    "discourse_turn_overlap_s",
)

TASK9_RECORDING_KEYS = (
    TASK9_UPOS_KEYS
    + TASK9_DEP_KEYS
    + TASK9_COMPOSITION_KEYS
    + TASK9_STRUCTURE_KEYS
    + DISCOURSE_RECORDING_KEYS
)
TASK9_KEYS = TASK9_RECORDING_KEYS + DISCOURSE_UTTERANCE_KEYS
FULL_ADULT_NEURO_KEYS = ADULT_NEURO_KEYS + TASK9_KEYS

TASK9_UNITS = {key: "ratio" for key in TASK9_UPOS_KEYS + TASK9_DEP_KEYS}
TASK9_UNITS.update(
    {
        key: "ratio"
        for key in (
            "morph_content_word_ratio",
            "morph_function_word_ratio",
            "morph_noun_verb_ratio",
            "morph_pronoun_noun_ratio",
            "morph_classifier_ratio",
            "morph_particle_ratio",
            "morph_code_switch_ratio",
        )
    }
)
TASK9_UNITS.update(
    {
        "morph_dependency_length_mean_tokens": "tokens",
        "morph_dependency_length_sd_tokens": "tokens",
        "morph_tree_depth_mean": "edges",
        "morph_tree_depth_max": "edges",
        "morph_clause_count": "count",
        "morph_clause_rate_per_utterance": "clauses/utterance",
        "morph_subordinate_clause_count": "count",
        "morph_subordination_ratio": "ratio",
        "discourse_turn_count": "count",
        "discourse_turn_length_mean_words": "words",
        "discourse_turn_length_sd_words": "words",
        "discourse_turn_length_mean_syllables": "syllables",
        "discourse_turn_length_sd_syllables": "syllables",
        "discourse_examiner_prompt_ratio": "ratio",
        "discourse_response_latency_mean_s": "s",
        "discourse_response_latency_sd_s": "s",
        "discourse_overlap_s": "s",
        "discourse_overlap_ratio": "ratio",
        "discourse_turn_word_count": "words",
        "discourse_turn_syllable_count": "syllables",
        "discourse_response_latency_s": "s",
        "discourse_turn_overlap_s": "s",
    }
)


def _morph_token(tid, text, upos, dep_rel, dep_head, language="vie", word_id=None):
    """One word-kind token with reviewed UPOS/dependency/code-switch fields."""
    return DocumentToken(
        id=tid,
        text=text,
        kind="word",
        word_id=word_id,
        language=language,
        dep_rel=dep_rel,
        dep_head=dep_head,
    )


def _morph_rich_document():
    """Hand-calculated Task 9 fixture: target p1, examiner ex1.

    u1 (t1-t12) + u2 (t13-t25) are the 25 target word tokens; u3 is
    examiner-only and must never leak into any value. Reviewed annotations:
    upos covers all 12 composition tags plus INTJ/NUM/DET/CCONJ/PROPN/SCONJ;
    classifier marks t1/t23; particle marks t12/t24; only t25 is language
    "en" (code switching). Dependency edges/depths/clauses are hand-traced in
    the tests below. u3 (12.5-13.5) overlaps u2 (13.0-16.0) for a 0.5 s
    recording/utterance overlap; u2 is the only prompted target turn.
    """
    u1 = DocumentUtterance(
        id="u1",
        speaker_id="p1",
        start_s=0.0,
        end_s=12.0,
        tokens=(
            _morph_token("t1", "Con", "noun", "nsubj", "t2"),
            _morph_token("t2", "mèo", "NOUN", "root", None),
            _morph_token("t3", "chạy", "VERB", "acl:relcl", "t2"),
            _morph_token("t4", "nhanh", "ADV", "advmod", "t3"),
            _morph_token("t5", "đã", "AUX", "aux", "t6"),
            _morph_token("t6", "ăn", "VERB", "conj", "t2"),
            _morph_token("t7", "cơm", "NOUN", "obj", "t6"),
            _morph_token("t8", "với", "ADP", "case", "t9"),
            _morph_token("t9", "nước", "NOUN", "nmod", "t7"),
            _morph_token("t10", "của", "ADP", "case", "t11"),
            _morph_token("t11", "bà", "NOUN", "nmod:poss", "t2"),
            _morph_token("t12", "ơi", "INTJ", "discourse", "t2"),
        ),
    )
    u2 = DocumentUtterance(
        id="u2",
        speaker_id="p1",
        start_s=13.0,
        end_s=16.0,
        tokens=(
            _morph_token("t13", "Nếu", "SCONJ", "mark", "t16"),
            _morph_token("t14", "mưa", "VERB", "advcl", "t16"),
            _morph_token("t15", "tôi", "PRON", "nsubj", "t16"),
            _morph_token("t16", "đi", "VERB", "root", None),
            _morph_token("t17", "làm", "VERB", "xcomp", "t16"),
            _morph_token("t18", "việc", "NOUN", "obj", "t17"),
            _morph_token("t19", "nhanh", "ADJ", "advmod", "t17"),
            _morph_token("t20", "và", "CCONJ", "cc", "t21"),
            _morph_token("t21", "Hà", "PROPN", "conj", "t17"),
            _morph_token("t22", "một", "NUM", "nummod", "t21"),
            _morph_token("t23", "con", "DET", "det", "t24"),
            _morph_token("t24", "nhé", "PART", "discourse", "t17"),
            _morph_token("t25", "ok", "NOUN", "discourse", "t17", language="en"),
        ),
    )
    u3 = DocumentUtterance(
        id="u3",
        speaker_id="ex1",
        start_s=12.5,
        end_s=13.5,
        tokens=(_token("t26", "nhìn"), _token("t27", "chỉ"), _token("t28", "vào")),
    )
    upos = {
        "t1": "noun",  # lowercase must be case-normalised to NOUN
        "t2": "NOUN",
        "t3": "VERB",
        "t4": "ADV",
        "t5": "AUX",
        "t6": "VERB",
        "t7": "NOUN",
        "t8": "ADP",
        "t9": "NOUN",
        "t10": "ADP",
        "t11": "NOUN",
        "t12": "INTJ",
        "t13": "SCONJ",
        "t14": "VERB",
        "t15": "PRON",
        "t16": "VERB",
        "t17": "VERB",
        "t18": "NOUN",
        "t19": "ADJ",
        "t20": "CCONJ",
        "t21": "PROPN",
        "t22": "NUM",
        "t23": "DET",
        "t24": "PART",
        "t25": "NOUN",
    }
    classifier = {tid: 0 for tid in upos}
    classifier.update({"t1": 1, "t23": 1.0})
    particle = {tid: 0 for tid in upos}
    particle.update({"t12": 1, "t24": 1})
    return SpeechDocument(
        document_id="rec-morph",
        speakers=(DocumentSpeaker(id="p1"), DocumentSpeaker(id="ex1")),
        utterances=(u1, u2, u3),
        annotations=(
            AnnotationLayer(layer="upos", source="reviewed", values=upos),
            AnnotationLayer(layer="classifier", source="reviewed", values=classifier),
            AnnotationLayer(layer="particle", source="reviewed", values=particle),
        ),
    )


def _conversation_document():
    """Hand-calculated multi-speaker timing fixture.

    Ordered turns: e1(0.0), p1(0.8), p2(2.5), e2(3.5), p3(3.8), e3(6.0),
    p4(8.0). Prompted target turns: p1, p3, p4 (p2 follows p1). Latencies
    [0.0, 0.0, 1.0]. Overlap intersections: p1x e1 = [0.8, 1.0],
    p2 x e2 = [3.5, 3.9], p3 x e2 = [3.8, 3.9]; their union is 0.6 s and the
    target interval union is 5.7 s. p1 groups "xe đạp" under one word_id
    (4 words / 5 syllables); other turns are ungrouped words.
    """
    return SpeechDocument(
        document_id="rec-conv",
        speakers=(DocumentSpeaker(id="p1"), DocumentSpeaker(id="ex1")),
        utterances=(
            DocumentUtterance(
                id="u1",
                speaker_id="ex1",
                start_s=0.0,
                end_s=1.0,
                tokens=(_token("t1", "nhìn"), _token("t2", "chỉ")),
            ),
            DocumentUtterance(
                id="u2",
                speaker_id="p1",
                start_s=0.8,
                end_s=3.0,
                tokens=(
                    _token("t3", "Tôi"),
                    _token("t4", "ăn"),
                    _token("t5", "cơm"),
                    _token("t6", "xe", word_id="w_xe"),
                    _token("t7", "đạp", word_id="w_xe"),
                ),
            ),
            DocumentUtterance(
                id="u3",
                speaker_id="p1",
                start_s=2.5,
                end_s=4.0,
                tokens=(_token("t8", "rồi"), _token("t9", "đi")),
            ),
            DocumentUtterance(
                id="u4",
                speaker_id="ex1",
                start_s=3.5,
                end_s=3.9,
                tokens=(_token("t10", "sao"),),
            ),
            DocumentUtterance(
                id="u5",
                speaker_id="p1",
                start_s=3.8,
                end_s=5.5,
                tokens=(_token("t11", "không"), _token("t12", "biết")),
            ),
            DocumentUtterance(
                id="u6",
                speaker_id="ex1",
                start_s=6.0,
                end_s=7.0,
                tokens=(_token("t13", "thế"),),
            ),
            DocumentUtterance(
                id="u7",
                speaker_id="p1",
                start_s=8.0,
                end_s=9.0,
                tokens=(_token("t14", "sao"), _token("t15", "cũng"), _token("t16", "được")),
            ),
        ),
    )


def _single_speaker_conversation_document():
    """Two target turns and no non-target turns: conversation unavailable."""
    return SpeechDocument(
        document_id="rec-single",
        speakers=(DocumentSpeaker(id="p1"),),
        utterances=(
            DocumentUtterance(
                id="u1",
                speaker_id="p1",
                start_s=0.0,
                end_s=1.0,
                tokens=(_token("t1", "mèo"),),
            ),
            DocumentUtterance(
                id="u2",
                speaker_id="p1",
                start_s=2.0,
                end_s=3.0,
                tokens=(_token("t2", "chạy"), _token("t3", "nhanh"), _token("t4", "đi")),
            ),
        ),
    )


def _single_turn_conversation_document():
    return SpeechDocument(
        document_id="rec-single-turn",
        speakers=(DocumentSpeaker(id="p1"),),
        utterances=(
            DocumentUtterance(
                id="u1",
                speaker_id="p1",
                start_s=0.0,
                end_s=1.0,
                tokens=(_token("t1", "mèo"), _token("t2", "đi")),
            ),
        ),
    )


def _no_prompt_conversation_document():
    """The only target turn precedes every examiner turn: no prompts."""
    return SpeechDocument(
        document_id="rec-no-prompt",
        speakers=(DocumentSpeaker(id="p1"), DocumentSpeaker(id="ex1")),
        utterances=(
            DocumentUtterance(
                id="u1",
                speaker_id="p1",
                start_s=0.0,
                end_s=1.0,
                tokens=(_token("t1", "mèo"),),
            ),
            DocumentUtterance(
                id="u2",
                speaker_id="ex1",
                start_s=2.0,
                end_s=3.0,
                tokens=(_token("t2", "nhìn"),),
            ),
        ),
    )


def _no_target_conversation_document():
    """The target is documented but says nothing: no language sample."""
    return SpeechDocument(
        document_id="rec-no-target",
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


def _dep_rejection_document(*utterances):
    word_tokens = [token for u in utterances for token in u.tokens if token.kind == "word"]
    upos = {token.id: "NOUN" for token in word_tokens}
    binary = {token.id: 0 for token in word_tokens}
    return SpeechDocument(
        document_id="rec-dep",
        speakers=(DocumentSpeaker(id="p1"),),
        utterances=utterances,
        annotations=(
            AnnotationLayer(layer="upos", source="reviewed", values=upos),
            AnnotationLayer(layer="classifier", source="reviewed", values=binary),
            AnnotationLayer(layer="particle", source="reviewed", values=binary),
        ),
    )


def _assert_task9_nan_issue_pairing(features, rows, issues):
    """Every recording/utterance NaN has exactly one matching warning issue."""
    recording_by_feature = {}
    utterance_by_feature = {}
    for issue in issues:
        assert issue.severity == "warning"
        assert issue.feature is not None
        if issue.utterance_id is None:
            recording_by_feature.setdefault(issue.feature, []).append(issue)
        else:
            utterance_by_feature.setdefault((issue.utterance_id, issue.feature), []).append(issue)
    for key, value in features.items():
        matching = recording_by_feature.get(key, [])
        if math.isnan(value):
            assert len(matching) == 1, f"NaN {key} needs exactly one issue, got {len(matching)}"
            recording_by_feature.pop(key)
        else:
            assert not matching, f"computed {key} has a stray issue"
    assert recording_by_feature == {}, (
        f"recording issues without NaN keys: {sorted(recording_by_feature)}"
    )
    for row in rows:
        for key in DISCOURSE_UTTERANCE_KEYS:
            matching = utterance_by_feature.get((row["utterance_id"], key), [])
            if math.isnan(row[key]):
                assert len(matching) == 1, (
                    f"NaN {key} of {row['utterance_id']} needs exactly one issue, got {len(matching)}"
                )
                utterance_by_feature.pop((row["utterance_id"], key))
            else:
                assert not matching, f"computed {key} of {row['utterance_id']} has a stray issue"
    assert utterance_by_feature == {}, (
        f"utterance issues without NaN rows: {sorted(utterance_by_feature)}"
    )


_DEP_STRUCTURE_KEYS = TASK9_DEP_KEYS + TASK9_STRUCTURE_KEYS


class TestAdultNeuroMorphosyntax:
    def test_hand_calculated_upos_ratios(self):
        # 25 target word tokens: NOUN x7, VERB x5, one each of
        # ADJ/ADV/AUX/ADP x2/INTJ/SCONJ/PRON/CCONJ/PROPN/NUM/DET/PART,
        # PUNCT/SYM/X absent; "noun" is case-normalised to NOUN.
        features, _, _ = extract_morphosyntax_features(_morph_rich_document(), target_speaker="p1")
        expected = {
            "morph_upos_adj_ratio": 1 / 25,
            "morph_upos_adp_ratio": 2 / 25,
            "morph_upos_adv_ratio": 1 / 25,
            "morph_upos_aux_ratio": 1 / 25,
            "morph_upos_cconj_ratio": 1 / 25,
            "morph_upos_det_ratio": 1 / 25,
            "morph_upos_intj_ratio": 1 / 25,
            "morph_upos_noun_ratio": 7 / 25,
            "morph_upos_num_ratio": 1 / 25,
            "morph_upos_part_ratio": 1 / 25,
            "morph_upos_pron_ratio": 1 / 25,
            "morph_upos_propn_ratio": 1 / 25,
            "morph_upos_punct_ratio": 0.0,
            "morph_upos_sconj_ratio": 1 / 25,
            "morph_upos_sym_ratio": 0.0,
            "morph_upos_verb_ratio": 5 / 25,
            "morph_upos_x_ratio": 0.0,
        }
        for key, value in expected.items():
            assert features[key] == pytest.approx(value), key
        assert sum(features[key] for key in TASK9_UPOS_KEYS) == pytest.approx(1.0)

    def test_hand_calculated_composition_ratios(self):
        features, _, _ = extract_morphosyntax_features(_morph_rich_document(), target_speaker="p1")
        # Content ADJ+ADV+NOUN+PROPN+VERB = 1+1+7+1+5 = 15/25; function
        # ADP+AUX+CCONJ+DET+PART+PRON+SCONJ = 2+1+1+1+1+1+1 = 8/25;
        # noun/verb = (NOUN+PROPN)/VERB = 8/5; pronoun/noun = PRON/(NOUN+PROPN) = 1/8.
        assert features["morph_content_word_ratio"] == pytest.approx(15 / 25)
        assert features["morph_function_word_ratio"] == pytest.approx(8 / 25)
        assert features["morph_noun_verb_ratio"] == pytest.approx(8 / 5)
        assert features["morph_pronoun_noun_ratio"] == pytest.approx(1 / 8)

    def test_hand_calculated_classifier_particle_code_switch_ratios(self):
        features, _, _ = extract_morphosyntax_features(_morph_rich_document(), target_speaker="p1")
        assert features["morph_classifier_ratio"] == pytest.approx(2 / 25)
        assert features["morph_particle_ratio"] == pytest.approx(2 / 25)
        assert features["morph_code_switch_ratio"] == pytest.approx(1 / 25)

    def test_dependency_ratios_partition_all_tokens(self):
        # Grand totals: root 2, subject 2, object 2, nominal modifier 2,
        # adverbial modifier 2, clausal complement 3, coordination 3,
        # function 5, other 4 -> 25 target word tokens, summing to one.
        features, _, _ = extract_morphosyntax_features(_morph_rich_document(), target_speaker="p1")
        expected = {
            "morph_dep_root_ratio": 2 / 25,
            "morph_dep_subject_ratio": 2 / 25,
            "morph_dep_object_ratio": 2 / 25,
            "morph_dep_nominal_modifier_ratio": 2 / 25,
            "morph_dep_adverbial_modifier_ratio": 2 / 25,
            "morph_dep_clausal_complement_ratio": 3 / 25,
            "morph_dep_coordination_ratio": 3 / 25,
            "morph_dep_function_ratio": 5 / 25,
            "morph_dep_other_ratio": 4 / 25,
        }
        for key, value in expected.items():
            assert features[key] == pytest.approx(value), key
        assert sum(features[key] for key in TASK9_DEP_KEYS) == pytest.approx(1.0)

    def test_hand_calculated_dependency_length_and_tree_depth(self):
        # 23 pooled edges summing to 64 (mean 64/23): ones x13, twos x3,
        # three, four x2, seven, eight, nine, ten. Depths over 25 tokens
        # sum to 43 (mean 43/25), maximum depth 4.
        features, _, _ = extract_morphosyntax_features(_morph_rich_document(), target_speaker="p1")
        edges = [1] * 13 + [2] * 3 + [3, 4, 4, 7, 8, 9, 10]
        assert len(edges) == 23 and sum(edges) == 64
        mean = 64 / 23
        sd = (sum((edge - mean) ** 2 for edge in edges) / len(edges)) ** 0.5
        assert features["morph_dependency_length_mean_tokens"] == pytest.approx(mean)
        assert features["morph_dependency_length_sd_tokens"] == pytest.approx(sd)
        assert features["morph_tree_depth_mean"] == pytest.approx(43 / 25)
        assert features["morph_tree_depth_max"] == 4.0

    def test_hand_calculated_clauses_and_subordination(self):
        # Clause heads (root, acl:relcl, advcl, xcomp, root) = 5; the three
        # non-root clause heads are subordinate; rate 5/2; subordination 3/5.
        features, _, _ = extract_morphosyntax_features(_morph_rich_document(), target_speaker="p1")
        assert features["morph_clause_count"] == 5.0
        assert features["morph_subordinate_clause_count"] == 3.0
        assert features["morph_clause_rate_per_utterance"] == pytest.approx(5 / 2)
        assert features["morph_subordination_ratio"] == pytest.approx(3 / 5)

    def test_target_isolation_examiner_content_never_leaks(self):
        features, rows, _ = extract_morphosyntax_features(
            _morph_rich_document(), target_speaker="p1"
        )
        assert features["morph_upos_noun_ratio"] == pytest.approx(7 / 25)
        assert features["morph_dep_function_ratio"] == pytest.approx(5 / 25)
        assert features["discourse_turn_count"] == 2.0
        assert [row["utterance_id"] for row in rows] == ["u1", "u2"]
        assert rows[1]["discourse_response_latency_s"] == 0.0
        assert rows[1]["discourse_turn_overlap_s"] == pytest.approx(0.5)
        assert rows[0]["discourse_turn_overlap_s"] == 0.0

    def test_nan_issue_pairing_on_rich_fixture(self):
        features, rows, issues = extract_morphosyntax_features(
            _morph_rich_document(), target_speaker="p1"
        )
        assert set(features) == set(TASK9_RECORDING_KEYS)
        _assert_task9_nan_issue_pairing(features, rows, issues)


class TestAdultNeuroMorphologyMissingLayers:
    def test_absent_upos_layer_flags_upos_and_composition_only(self):
        document = _morph_rich_document()
        document = SpeechDocument(
            document_id=document.document_id,
            speakers=document.speakers,
            utterances=document.utterances,
            annotations=tuple(a for a in document.annotations if a.layer != "upos"),
        )
        features, rows, issues = extract_morphosyntax_features(document, target_speaker="p1")
        upos_derived = TASK9_UPOS_KEYS + (
            "morph_content_word_ratio",
            "morph_function_word_ratio",
            "morph_noun_verb_ratio",
            "morph_pronoun_noun_ratio",
        )
        for key in upos_derived:
            assert math.isnan(features[key])
        # Independent layers stay defined despite the missing upos layer.
        assert features["morph_classifier_ratio"] == pytest.approx(2 / 25)
        assert features["morph_particle_ratio"] == pytest.approx(2 / 25)
        assert features["morph_code_switch_ratio"] == pytest.approx(1 / 25)
        assert features["morph_dep_root_ratio"] == pytest.approx(2 / 25)
        by_feature = _issues_by_feature(issues)
        for key in upos_derived:
            assert by_feature[key][0].code == "MISSING_ANNOTATION"
        _assert_task9_nan_issue_pairing(features, rows, issues)

    def test_unknown_upos_tag_flags_all_21_keys(self):
        document = _morph_rich_document()
        layer = next(a for a in document.annotations if a.layer == "upos")
        values = dict(layer.values)
        values["t25"] = "NOUNS"
        document = SpeechDocument(
            document_id=document.document_id,
            speakers=document.speakers,
            utterances=document.utterances,
            annotations=(
                AnnotationLayer(layer="upos", source="reviewed", values=values),
                *(a for a in document.annotations if a.layer != "upos"),
            ),
        )
        features, rows, issues = extract_morphosyntax_features(document, target_speaker="p1")
        upos_derived = TASK9_UPOS_KEYS + (
            "morph_content_word_ratio",
            "morph_function_word_ratio",
            "morph_noun_verb_ratio",
            "morph_pronoun_noun_ratio",
        )
        for key in upos_derived:
            assert math.isnan(features[key])
        assert features["morph_classifier_ratio"] == pytest.approx(2 / 25)
        by_feature = _issues_by_feature(issues)
        assert by_feature["morph_upos_adj_ratio"][0].code == "MISSING_ANNOTATION"
        assert by_feature["morph_noun_verb_ratio"][0].code == "MISSING_ANNOTATION"
        _assert_task9_nan_issue_pairing(features, rows, issues)

    def test_incomplete_upos_layer_flags_all_21_keys(self):
        document = _morph_rich_document()
        layer = next(a for a in document.annotations if a.layer == "upos")
        values = dict(layer.values)
        del values["t2"]
        document = SpeechDocument(
            document_id=document.document_id,
            speakers=document.speakers,
            utterances=document.utterances,
            annotations=(
                AnnotationLayer(layer="upos", source="reviewed", values=values),
                *(a for a in document.annotations if a.layer != "upos"),
            ),
        )
        features, rows, issues = extract_morphosyntax_features(document, target_speaker="p1")
        assert math.isnan(features["morph_upos_noun_ratio"])
        assert math.isnan(features["morph_content_word_ratio"])
        assert features["morph_code_switch_ratio"] == pytest.approx(1 / 25)
        by_feature = _issues_by_feature(issues)
        assert by_feature["morph_upos_noun_ratio"][0].code == "MISSING_ANNOTATION"
        _assert_task9_nan_issue_pairing(features, rows, issues)

    def test_missing_classifier_value_flags_classifier_only(self):
        document = _morph_rich_document()
        layer = next(a for a in document.annotations if a.layer == "classifier")
        values = dict(layer.values)
        del values["t1"]
        document = SpeechDocument(
            document_id=document.document_id,
            speakers=document.speakers,
            utterances=document.utterances,
            annotations=(
                *(a for a in document.annotations if a.layer != "classifier"),
                AnnotationLayer(layer="classifier", source="reviewed", values=values),
            ),
        )
        features, rows, issues = extract_morphosyntax_features(document, target_speaker="p1")
        assert math.isnan(features["morph_classifier_ratio"])
        assert features["morph_particle_ratio"] == pytest.approx(2 / 25)
        assert features["morph_upos_noun_ratio"] == pytest.approx(7 / 25)
        by_feature = _issues_by_feature(issues)
        assert by_feature["morph_classifier_ratio"][0].code == "MISSING_ANNOTATION"
        _assert_task9_nan_issue_pairing(features, rows, issues)

    def test_boolean_classifier_value_is_invalid(self):
        document = _morph_rich_document()
        layer = next(a for a in document.annotations if a.layer == "classifier")
        values = dict(layer.values)
        values["t1"] = True
        document = SpeechDocument(
            document_id=document.document_id,
            speakers=document.speakers,
            utterances=document.utterances,
            annotations=(
                *(a for a in document.annotations if a.layer != "classifier"),
                AnnotationLayer(layer="classifier", source="reviewed", values=values),
            ),
        )
        features, _, issues = extract_morphosyntax_features(document, target_speaker="p1")
        assert math.isnan(features["morph_classifier_ratio"])
        by_feature = _issues_by_feature(issues)
        assert by_feature["morph_classifier_ratio"][0].code == "MISSING_ANNOTATION"

    def test_non_binary_classifier_value_is_invalid(self):
        document = _morph_rich_document()
        layer = next(a for a in document.annotations if a.layer == "classifier")
        values = dict(layer.values)
        values["t1"] = 2
        document = SpeechDocument(
            document_id=document.document_id,
            speakers=document.speakers,
            utterances=document.utterances,
            annotations=(
                *(a for a in document.annotations if a.layer != "classifier"),
                AnnotationLayer(layer="classifier", source="reviewed", values=values),
            ),
        )
        features, _, issues = extract_morphosyntax_features(document, target_speaker="p1")
        assert math.isnan(features["morph_classifier_ratio"])
        by_feature = _issues_by_feature(issues)
        assert by_feature["morph_classifier_ratio"][0].code == "MISSING_ANNOTATION"

    def test_missing_particle_value_flags_particle_only(self):
        document = _morph_rich_document()
        layer = next(a for a in document.annotations if a.layer == "particle")
        values = dict(layer.values)
        del values["t12"]
        document = SpeechDocument(
            document_id=document.document_id,
            speakers=document.speakers,
            utterances=document.utterances,
            annotations=(
                *(a for a in document.annotations if a.layer != "particle"),
                AnnotationLayer(layer="particle", source="reviewed", values=values),
            ),
        )
        features, _, issues = extract_morphosyntax_features(document, target_speaker="p1")
        assert math.isnan(features["morph_particle_ratio"])
        assert features["morph_classifier_ratio"] == pytest.approx(2 / 25)
        by_feature = _issues_by_feature(issues)
        assert by_feature["morph_particle_ratio"][0].code == "MISSING_ANNOTATION"

    def test_missing_token_language_flags_code_switch_only(self):
        document = _morph_rich_document()
        u2 = document.utterances[1]
        u2 = DocumentUtterance(
            id=u2.id,
            speaker_id=u2.speaker_id,
            start_s=u2.start_s,
            end_s=u2.end_s,
            tokens=tuple(
                _morph_token(t.id, t.text, "NOUN", "discourse", "t17", language=None)
                if t.id == "t25"
                else t
                for t in u2.tokens
            ),
        )
        document = SpeechDocument(
            document_id=document.document_id,
            speakers=document.speakers,
            utterances=(document.utterances[0], u2, document.utterances[2]),
            annotations=document.annotations,
        )
        features, _, issues = extract_morphosyntax_features(document, target_speaker="p1")
        assert math.isnan(features["morph_code_switch_ratio"])
        assert features["morph_upos_noun_ratio"] == pytest.approx(7 / 25)
        assert features["morph_classifier_ratio"] == pytest.approx(2 / 25)
        by_feature = _issues_by_feature(issues)
        assert by_feature["morph_code_switch_ratio"][0].code == "MISSING_ANNOTATION"

    def test_dependency_two_roots_rejects_all_17_keys(self):
        document = _dep_rejection_document(
            DocumentUtterance(
                id="u1",
                speaker_id="p1",
                start_s=0.0,
                end_s=5.0,
                tokens=(
                    _morph_token("t1", "mèo", "NOUN", "root", None),
                    _morph_token("t2", "đi", "VERB", "root", None),
                ),
            ),
        )
        features, rows, issues = extract_morphosyntax_features(document)
        for key in _DEP_STRUCTURE_KEYS:
            assert math.isnan(features[key])
        assert features["morph_upos_noun_ratio"] == pytest.approx(1.0)
        by_feature = _issues_by_feature(issues)
        for key in _DEP_STRUCTURE_KEYS:
            assert by_feature[key][0].code == "MISSING_ANNOTATION"
        _assert_task9_nan_issue_pairing(features, rows, issues)

    def test_dependency_missing_head_rejects_all_17_keys(self):
        document = _dep_rejection_document(
            DocumentUtterance(
                id="u1",
                speaker_id="p1",
                start_s=0.0,
                end_s=5.0,
                tokens=(
                    _morph_token("t1", "mèo", "NOUN", "root", None),
                    _morph_token("t2", "đi", "VERB", "nsubj", None),
                ),
            ),
        )
        features, rows, issues = extract_morphosyntax_features(document)
        for key in _DEP_STRUCTURE_KEYS:
            assert math.isnan(features[key])
        by_feature = _issues_by_feature(issues)
        assert by_feature["morph_dep_subject_ratio"][0].code == "MISSING_ANNOTATION"
        _assert_task9_nan_issue_pairing(features, rows, issues)

    def test_dependency_cycle_rejects_all_17_keys(self):
        document = _dep_rejection_document(
            DocumentUtterance(
                id="u1",
                speaker_id="p1",
                start_s=0.0,
                end_s=5.0,
                tokens=(
                    _morph_token("t1", "mèo", "NOUN", "nsubj", "t2"),
                    _morph_token("t2", "đi", "VERB", "nsubj", "t1"),
                ),
            ),
        )
        features, rows, issues = extract_morphosyntax_features(document)
        for key in _DEP_STRUCTURE_KEYS:
            assert math.isnan(features[key])
        by_feature = _issues_by_feature(issues)
        assert by_feature["morph_tree_depth_mean"][0].code == "MISSING_ANNOTATION"
        _assert_task9_nan_issue_pairing(features, rows, issues)

    def test_dependency_cross_utterance_head_rejects_all_17_keys(self):
        document = _dep_rejection_document(
            DocumentUtterance(
                id="u1",
                speaker_id="p1",
                start_s=0.0,
                end_s=1.0,
                tokens=(_morph_token("t1", "mèo", "NOUN", "root", None),),
            ),
            DocumentUtterance(
                id="u2",
                speaker_id="p1",
                start_s=2.0,
                end_s=3.0,
                tokens=(_morph_token("t2", "đi", "VERB", "nsubj", "t1"),),
            ),
        )
        features, rows, issues = extract_morphosyntax_features(document)
        for key in _DEP_STRUCTURE_KEYS:
            assert math.isnan(features[key])
        by_feature = _issues_by_feature(issues)
        assert by_feature["morph_dependency_length_mean_tokens"][0].code == "MISSING_ANNOTATION"
        _assert_task9_nan_issue_pairing(features, rows, issues)

    def test_dependency_root_with_external_head_rejects_all_17_keys(self):
        document = _dep_rejection_document(
            DocumentUtterance(
                id="u1",
                speaker_id="p1",
                start_s=0.0,
                end_s=5.0,
                tokens=(
                    _morph_token("t1", "mèo", "NOUN", "root", "t2"),
                    _morph_token("t2", "đi", "VERB", "nsubj", "t1"),
                ),
            ),
        )
        features, rows, issues = extract_morphosyntax_features(document)
        for key in _DEP_STRUCTURE_KEYS:
            assert math.isnan(features[key])
        by_feature = _issues_by_feature(issues)
        assert by_feature["morph_dep_root_ratio"][0].code == "MISSING_ANNOTATION"
        _assert_task9_nan_issue_pairing(features, rows, issues)

    def test_dependency_empty_or_blank_relation_rejects_all_17_keys(self):
        for invalid in (None, "", "   "):
            document = _dep_rejection_document(
                DocumentUtterance(
                    id="u1",
                    speaker_id="p1",
                    start_s=0.0,
                    end_s=5.0,
                    tokens=(
                        _morph_token("t1", "mèo", "NOUN", "root", None),
                        _morph_token("t2", "đi", "VERB", invalid, "t1"),
                    ),
                ),
            )
            features, rows, issues = extract_morphosyntax_features(document)
            for key in _DEP_STRUCTURE_KEYS:
                assert math.isnan(features[key]), f"{invalid!r} left {key} computed"
            by_feature = _issues_by_feature(issues)
            for key in _DEP_STRUCTURE_KEYS:
                assert by_feature[key][0].code == "MISSING_ANNOTATION", f"{invalid!r} {key}"
            _assert_task9_nan_issue_pairing(features, rows, issues)


class TestAdultNeuroConversation:
    def test_hand_calculated_recording_timing_measures(self):
        features, _, _ = extract_morphosyntax_features(
            _conversation_document(), target_speaker="p1"
        )
        assert features["discourse_turn_count"] == 4.0
        assert features["discourse_examiner_prompt_ratio"] == pytest.approx(3 / 4)
        assert features["discourse_response_latency_mean_s"] == pytest.approx(1 / 3)
        expected_sd = ((1 / 9 + 1 / 9 + 4 / 9) / 3) ** 0.5
        assert features["discourse_response_latency_sd_s"] == pytest.approx(expected_sd)
        assert features["discourse_overlap_s"] == pytest.approx(0.6)
        assert features["discourse_overlap_ratio"] == pytest.approx(0.6 / 5.7)

    def test_hand_calculated_turn_lengths_with_grouping(self):
        # Words [4, 2, 2, 3] (xe/đạp grouped in u2), syllables [5, 2, 2, 3].
        features, _, _ = extract_morphosyntax_features(
            _conversation_document(), target_speaker="p1"
        )
        word_values = [4.0, 2.0, 2.0, 3.0]
        word_mean = 11 / 4
        word_sd = (sum((v - word_mean) ** 2 for v in word_values) / 4) ** 0.5
        assert features["discourse_turn_length_mean_words"] == pytest.approx(word_mean)
        assert features["discourse_turn_length_sd_words"] == pytest.approx(word_sd)
        syllable_values = [5.0, 2.0, 2.0, 3.0]
        syllable_mean = 3.0
        syllable_sd = (sum((v - syllable_mean) ** 2 for v in syllable_values) / 4) ** 0.5
        assert features["discourse_turn_length_mean_syllables"] == pytest.approx(syllable_mean)
        assert features["discourse_turn_length_sd_syllables"] == pytest.approx(syllable_sd)

    def test_exact_utterance_rows_in_document_order(self):
        features, rows, issues = extract_morphosyntax_features(
            _conversation_document(), target_speaker="p1"
        )
        assert [row["utterance_id"] for row in rows] == ["u2", "u3", "u5", "u7"]
        assert list(rows[0]) == [
            "utterance_id",
            "start_s",
            "end_s",
            "discourse_turn_word_count",
            "discourse_turn_syllable_count",
            "discourse_response_latency_s",
            "discourse_turn_overlap_s",
        ]
        assert rows[0] == {
            "utterance_id": "u2",
            "start_s": 0.8,
            "end_s": 3.0,
            "discourse_turn_word_count": 4.0,
            "discourse_turn_syllable_count": 5.0,
            "discourse_response_latency_s": 0.0,
            "discourse_turn_overlap_s": pytest.approx(0.2),
        }
        assert rows[1]["utterance_id"] == "u3"
        assert rows[1]["start_s"] == 2.5
        assert rows[1]["end_s"] == 4.0
        assert rows[1]["discourse_turn_word_count"] == 2.0
        assert rows[1]["discourse_turn_syllable_count"] == 2.0
        assert math.isnan(rows[1]["discourse_response_latency_s"])
        assert rows[1]["discourse_turn_overlap_s"] == pytest.approx(0.4)
        # u3 follows a target turn, so it is not prompted: its latency is
        # NaN with an utterance-scoped issue.
        latency_issues = [i for i in issues if i.feature == "discourse_response_latency_s"]
        assert [i.utterance_id for i in latency_issues] == ["u3"]
        assert latency_issues[0].code == "MISSING_ANNOTATION"
        assert rows[2] == {
            "utterance_id": "u5",
            "start_s": 3.8,
            "end_s": 5.5,
            "discourse_turn_word_count": 2.0,
            "discourse_turn_syllable_count": 2.0,
            "discourse_response_latency_s": 0.0,
            "discourse_turn_overlap_s": pytest.approx(0.1),
        }
        assert rows[3] == {
            "utterance_id": "u7",
            "start_s": 8.0,
            "end_s": 9.0,
            "discourse_turn_word_count": 3.0,
            "discourse_turn_syllable_count": 3.0,
            "discourse_response_latency_s": 1.0,
            "discourse_turn_overlap_s": 0.0,
        }
        _assert_task9_nan_issue_pairing(features, rows, issues)

    def test_overlap_union_not_sum(self):
        # p2 x e2 = [3.5, 3.9] and p3 x e2 = [3.8, 3.9] share the e2 tail:
        # recording overlap is the 0.4 s union, not 0.4 + 0.1.
        features, rows, _ = extract_morphosyntax_features(
            _conversation_document(), target_speaker="p1"
        )
        assert features["discourse_overlap_s"] == pytest.approx(0.6)
        assert rows[1]["discourse_turn_overlap_s"] == pytest.approx(0.4)
        assert rows[2]["discourse_turn_overlap_s"] == pytest.approx(0.1)

    def test_nan_issue_pairing_on_timing_fixture(self):
        features, rows, issues = extract_morphosyntax_features(
            _conversation_document(), target_speaker="p1"
        )
        assert set(features) == set(TASK9_RECORDING_KEYS)
        _assert_task9_nan_issue_pairing(features, rows, issues)

    def test_single_turn_length_sds_insufficient(self):
        features, rows, issues = extract_morphosyntax_features(_single_turn_conversation_document())
        assert features["discourse_turn_count"] == 1.0
        assert features["discourse_turn_length_mean_words"] == pytest.approx(2.0)
        assert math.isnan(features["discourse_turn_length_sd_words"])
        assert math.isnan(features["discourse_turn_length_sd_syllables"])
        by_feature = _issues_by_feature(issues)
        assert by_feature["discourse_turn_length_sd_words"][0].code == "INSUFFICIENT_TOKENS"
        _assert_task9_nan_issue_pairing(features, rows, issues)

    def test_no_prompted_turns_latency_unavailable(self):
        features, rows, issues = extract_morphosyntax_features(
            _no_prompt_conversation_document(), target_speaker="p1"
        )
        assert features["discourse_examiner_prompt_ratio"] == 0.0
        assert math.isnan(features["discourse_response_latency_mean_s"])
        assert math.isnan(features["discourse_response_latency_sd_s"])
        by_feature = _issues_by_feature(issues)
        assert by_feature["discourse_response_latency_mean_s"][0].code == "MISSING_ANNOTATION"
        assert by_feature["discourse_response_latency_sd_s"][0].code == "MISSING_ANNOTATION"
        # No intersection with the examiner turn is a known zero.
        assert features["discourse_overlap_s"] == 0.0
        assert features["discourse_overlap_ratio"] == 0.0
        assert rows[0]["discourse_turn_overlap_s"] == 0.0
        _assert_task9_nan_issue_pairing(features, rows, issues)


class TestAdultNeuroConversationMissing:
    def test_single_speaker_conversation_measures_unavailable(self):
        features, rows, issues = extract_morphosyntax_features(
            _single_speaker_conversation_document()
        )
        # Turn counts/lengths remain defined (words and syllables [1, 3]).
        assert features["discourse_turn_count"] == 2.0
        assert features["discourse_turn_length_mean_words"] == pytest.approx(2.0)
        assert features["discourse_turn_length_sd_words"] == pytest.approx(1.0)
        assert features["discourse_turn_length_mean_syllables"] == pytest.approx(2.0)
        assert features["discourse_turn_length_sd_syllables"] == pytest.approx(1.0)
        for key in (
            "discourse_examiner_prompt_ratio",
            "discourse_response_latency_mean_s",
            "discourse_response_latency_sd_s",
            "discourse_overlap_s",
            "discourse_overlap_ratio",
        ):
            assert math.isnan(features[key])
        by_feature = _issues_by_feature(issues)
        for key in (
            "discourse_examiner_prompt_ratio",
            "discourse_response_latency_mean_s",
            "discourse_response_latency_sd_s",
            "discourse_overlap_s",
            "discourse_overlap_ratio",
        ):
            assert by_feature[key][0].code == "MISSING_ANNOTATION"
        # Utterance rows keep defined counts but NaN latency/overlap.
        assert [row["utterance_id"] for row in rows] == ["u1", "u2"]
        assert rows[0]["discourse_turn_word_count"] == 1.0
        assert math.isnan(rows[0]["discourse_response_latency_s"])
        assert math.isnan(rows[0]["discourse_turn_overlap_s"])
        _assert_task9_nan_issue_pairing(features, rows, issues)

    def test_no_target_sample_all_51_keys_nan_with_one_issue_each(self):
        features, rows, issues = extract_morphosyntax_features(
            _no_target_conversation_document(), target_speaker="p1"
        )
        assert set(features) == set(TASK9_RECORDING_KEYS)
        assert all(math.isnan(value) for value in features.values())
        assert rows == []
        assert len(issues) == 51
        by_feature = _issues_by_feature(issues)
        assert set(by_feature) == set(TASK9_RECORDING_KEYS)
        assert all(issue.code == "MISSING_ANNOTATION" for issue in issues)
        assert all(issue.utterance_id is None for issue in issues)
        _assert_task9_nan_issue_pairing(features, rows, issues)

    def test_filler_only_sample_known_zeros_and_insufficient(self):
        document = SpeechDocument(
            document_id="rec-filler",
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
        features, rows, issues = extract_morphosyntax_features(document)
        # No word-kind tokens: morphology denominators are zero.
        for key in TASK9_UPOS_KEYS + TASK9_DEP_KEYS + TASK9_COMPOSITION_KEYS:
            assert math.isnan(features[key])
        for key in (
            "morph_dependency_length_mean_tokens",
            "morph_dependency_length_sd_tokens",
            "morph_tree_depth_mean",
            "morph_tree_depth_max",
            "morph_subordination_ratio",
        ):
            assert math.isnan(features[key])
        # Known zeros stay zero; conversation counts/lengths remain defined.
        assert features["morph_clause_count"] == 0.0
        assert features["morph_subordinate_clause_count"] == 0.0
        assert features["morph_clause_rate_per_utterance"] == 0.0
        assert features["discourse_turn_count"] == 1.0
        assert features["discourse_turn_length_mean_words"] == 0.0
        by_feature = _issues_by_feature(issues)
        assert by_feature["morph_upos_adj_ratio"][0].code == "INSUFFICIENT_TOKENS"
        assert by_feature["morph_noun_verb_ratio"][0].code == "INSUFFICIENT_TOKENS"
        assert by_feature["morph_classifier_ratio"][0].code == "INSUFFICIENT_TOKENS"
        assert by_feature["morph_dep_root_ratio"][0].code == "INSUFFICIENT_TOKENS"
        assert by_feature["morph_subordination_ratio"][0].code == "INSUFFICIENT_TOKENS"
        _assert_task9_nan_issue_pairing(features, rows, issues)


class TestAdultNeuroComposition:
    def test_extract_adult_neuro_composes_exactly_97_keys(self):
        document = _morph_rich_document()
        lexical, lexical_issues = extract_lexical_features(
            document, target_speaker="p1", recording_id="r1"
        )
        morph, rows, morph_issues = extract_morphosyntax_features(
            document, target_speaker="p1", recording_id="r1"
        )
        features, got_rows, issues = extract_adult_neuro_features(
            document, target_speaker="p1", recording_id="r1"
        )
        assert set(features) == set(ADULT_NEURO_KEYS + TASK9_RECORDING_KEYS)
        assert set(got_rows[0]) - {"utterance_id", "start_s", "end_s"} == set(
            DISCOURSE_UTTERANCE_KEYS
        )
        assert features == {**lexical, **morph}
        assert got_rows == rows
        assert list(issues) == list(lexical_issues) + list(morph_issues)

    def test_composition_preserves_nan_issue_pairing(self):
        features, rows, issues = extract_adult_neuro_features(
            _no_target_conversation_document(), target_speaker="p1"
        )
        assert set(features) == set(ADULT_NEURO_KEYS + TASK9_RECORDING_KEYS)
        assert all(math.isnan(value) for value in features.values())
        assert len(issues) == 93
        _assert_task9_nan_issue_pairing(features, rows, issues)
