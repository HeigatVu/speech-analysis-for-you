"""Behavioral tests for external task-spec scorers (Task 3).

Scorers are built purely from external task-spec JSON (picture concept/entity/
action aliases, recall idea aliases, phonemic initials/exclusions, semantic
item aliases/subcategories) and a participant ``Transcript``. Matching uses
exact NFC/casefold equality plus standard Levenshtein normalized similarity at
a documented conservative threshold. No diagnosis/labels and no NLP/ASR models.
"""

import math

import pytest

import speech_features.tasks as tasks
from speech_features.schema import Token, Transcript, Utterance


def _utt(*kinds_texts):
    tokens = [Token(kind=k, text=t) for k, t in kinds_texts]
    return Utterance(speaker="participant", start_s=0.0, end_s=1.0, tokens=tokens)


def _tr(*utts):
    return Transcript(utterances=utts)


# --------------------------------------------------------------------------- #
# Similarity / matching
# --------------------------------------------------------------------------- #
class TestLevenshteinSimilarity:
    def test_levenshtein_distance(self):
        assert tasks.levenshtein("kitten", "sitting") == 3
        assert tasks.levenshtein("mèo", "mèo") == 0
        assert tasks.levenshtein("", "") == 0
        assert tasks.levenshtein("abc", "") == 3

    def test_normalised_similarity_bounds(self):
        assert tasks.normalised_similarity("mèo", "mèo") == pytest.approx(1.0)
        assert tasks.normalised_similarity("cat", "dog") == pytest.approx(0.0)
        assert 0.0 <= tasks.normalised_similarity("con mèo", "con mêo") <= 1.0

    def test_normalised_similarity_empty_pair_is_zero(self):
        assert tasks.normalised_similarity("", "") == pytest.approx(0.0)

    def test_default_threshold_is_conservative(self):
        assert 0.0 <= tasks.SIMILARITY_THRESHOLD <= 1.0
        assert tasks.SIMILARITY_THRESHOLD >= 0.80


# --------------------------------------------------------------------------- #
# Picture
# --------------------------------------------------------------------------- #
def _picture_spec():
    return {
        "version": 1,
        "task": "picture_desc_1",
        "concept_aliases": {
            "cat": ["mèo", "con mèo"],
            "dog": ["chó"],
            "run": ["chạy"],
        },
        "entity_groups": {"animals": ["mèo", "con mèo", "chó"]},
        "action_groups": {"motion": ["chạy", "đang chạy"]},
    }


class TestPictureScorer:
    def test_finds_concepts_via_coverage_repeat_and_groups(self):
        tr = _tr(_utt(("word", "con mèo"), ("word", "mèo"), ("word", "chó"), ("word", "chạy")))
        r = tasks.score_picture(tr, _picture_spec())
        assert r["picture_concept_coverage"] == pytest.approx(3 / 3)
        # 4 matched tokens, 3 distinct concepts
        assert r["picture_concept_density"] == pytest.approx(3 / 4)
        assert r["picture_repeat_ratio"] == pytest.approx(1 / 4)
        # entity group "animals" fully covered; only one participant word matches
        assert r["picture_entity_coverage"] == pytest.approx(1.0)
        assert r["picture_action_coverage"] == pytest.approx(1.0)

    def test_partial_coverage_by_aliases(self):
        tr = _tr(_utt(("word", "con mèo")))
        r = tasks.score_picture(tr, _picture_spec())
        assert r["picture_concept_coverage"] == pytest.approx(1 / 3)

    def test_fuzzy_alias_match_via_levenshtein(self):
        # "con meo" (missing diacritic) is within the conservative threshold of
        # the alias "con mèo".
        tr = _tr(_utt(("word", "con meo")))
        r = tasks.score_picture(tr, _picture_spec())
        assert r["picture_concept_coverage"] == pytest.approx(1 / 3)

    def test_unmatched_words_produce_zero_not_error(self):
        tr = _tr(_utt(("word", "completely-unrelated")))
        r = tasks.score_picture(tr, _picture_spec())
        assert r["picture_concept_coverage"] == pytest.approx(0.0)
        assert r["picture_repeat_ratio"] == 0.0 or math.isnan(r["picture_repeat_ratio"])
        assert r["picture_entity_coverage"] == pytest.approx(0.0)

    def test_no_participant_words_returns_nans(self):
        tr = _tr(_utt(("filler", "ừm")))
        r = tasks.score_picture(tr, _picture_spec())
        assert math.isnan(r["picture_concept_density"])
        assert math.isnan(r["picture_repeat_ratio"])


# --------------------------------------------------------------------------- #
# Recall
# --------------------------------------------------------------------------- #
def _recall_spec():
    return {
        "version": 1,
        "task": "immediate_recall",
        # ideas in the presented reference order
        "idea_aliases": {
            "camera": ["camera", "máy ảnh"],
            "umbrella": ["ô", "dù"],
            "frog": ["ếch"],
        },
    }


class TestRecallScorer:
    def test_recall_coverage_density_repeat(self):
        tr = _tr(_utt(("word", "máy ảnh"), ("word", "ô"), ("word", "ếch"), ("word", "ếch")))
        r = tasks.score_recall(tr, _recall_spec())
        assert r["recall_idea_coverage"] == pytest.approx(3 / 3)
        assert r["recall_idea_density"] == pytest.approx(3 / 4)
        assert r["recall_repeat_ratio"] == pytest.approx(1 / 4)

    def test_recall_order_score_in_reference_order(self):
        tr = _tr(_utt(("word", "máy ảnh"), ("word", "ô"), ("word", "ếch")))
        r = tasks.score_recall(tr, _recall_spec())
        # matched in spec order -> perfect order score
        assert r["recall_order_score"] == pytest.approx(1.0)

    def test_recall_order_score_penalises_inversion(self):
        tr = _tr(_utt(("word", "ếch"), ("word", "ô"), ("word", "máy ảnh")))
        r = tasks.score_recall(tr, _recall_spec())
        assert 0.0 <= r["recall_order_score"] < 1.0

    def test_recall_partial_matches(self):
        tr = _tr(_utt(("word", "ô")))
        r = tasks.score_recall(tr, _recall_spec())
        assert r["recall_idea_coverage"] == pytest.approx(1 / 3)

    def test_recall_no_words_returns_nans(self):
        tr = _tr(_utt(("filler", "ừm")))
        r = tasks.score_recall(tr, _recall_spec())
        assert math.isnan(r["recall_idea_density"])
        assert math.isnan(r["recall_repeat_ratio"])
        assert math.isnan(r["recall_order_score"])


# --------------------------------------------------------------------------- #
# Phonemic fluency
# --------------------------------------------------------------------------- #
def _phonemic_spec():
    return {
        "version": 1,
        "task": "phonemic_fluency",
        "initials": ["c"],
        "exclusions": ["con gì"],  # excluded whole response
    }


class TestPhonemicScorer:
    def test_valid_responses_rate_repeats_intrusions(self):
        # One utterance is one response item.
        tr = _tr(
            _utt(("word", "cá"), ("word", "cây")),  # valid "cá cây"
            _utt(("word", "cây")),  # repeat
            _utt(("word", "con gì")),  # excluded -> intrusion
            _utt(("word", "bàn")),  # wrong initial -> intrusion
        )
        r = tasks.score_phonemic(tr, _phonemic_spec())
        # unique valid: {cá cây, cây} -> distinct... treat each response's first
        # word as its identity for uniqueness.
        assert r["fluency_valid_count"] == 2
        assert r["fluency_valid_unique"] == 2
        assert r["fluency_response_count"] == 4
        assert r["fluency_intrusions"] == 2
        assert r["fluency_rate"] == pytest.approx(2 / 4)

    def test_valid_response_respecting_initial(self):
        tr = _tr(_utt(("word", "con cá")), _utt(("word", "già")))
        r = tasks.score_phonemic(tr, _phonemic_spec())
        # "con cá" starts with c -> valid; "già" does not -> intrusion
        assert r["fluency_valid_count"] == 1
        assert r["fluency_intrusions"] == 1

    def test_half_time_and_production_change(self):
        tr = _tr(
            _utt(("word", "cá")),  # valid, 1st half
            _utt(("word", "con gì")),  # intrusion
            _utt(("word", "cây")),  # valid, 2nd half
        )
        r = tasks.score_phonemic(tr, _phonemic_spec())
        assert r["fluency_first_half_valid"] == 1
        assert r["fluency_second_half_valid"] == 1
        assert r["fluency_production_change"] == 0

    def test_no_utterances_returns_nans(self):
        tr = _tr()
        r = tasks.score_phonemic(tr, _phonemic_spec())
        assert math.isnan(r["fluency_rate"])
        assert r["fluency_valid_count"] == 0
        assert r["fluency_intrusions"] == 0


# --------------------------------------------------------------------------- #
# Semantic fluency
# --------------------------------------------------------------------------- #
def _semantic_spec():
    return {
        "version": 1,
        "task": "semantic_fluency",
        "item_aliases": {
            "cat": ["mèo", "con mèo"],
            "dog": ["chó", "con chó"],
            "apple": ["táo", "quả táo"],
        },
        "subcategories": {"cat": "animal", "dog": "animal", "apple": "fruit"},
    }


class TestSemanticScorer:
    def test_valid_unique_repeats_rate(self):
        tr = _tr(
            _utt(("word", "mèo")),
            _utt(("word", "chó")),
            _utt(("word", "mèo")),  # repeat
            _utt(("word", "táo")),
        )
        r = tasks.score_semantic(tr, _semantic_spec())
        assert r["fluency_valid_unique"] == 3
        assert r["fluency_repeats"] == 1
        assert r["fluency_response_count"] == 4
        assert r["fluency_rate"] == pytest.approx(3 / 4)

    def test_clusters_and_switches(self):
        # animal(1) -> animal(2) = same cluster, no switch;
        # animal -> fruit = a switch, fruit -> animal = another switch.
        tr = _tr(
            _utt(("word", "mèo")),  # animal
            _utt(("word", "chó")),  # animal  (within-cluster)
            _utt(("word", "táo")),  # fruit    (switch)
            _utt(("word", "con chó")),  # animal (switch)
        )
        r = tasks.score_semantic(tr, _semantic_spec())
        assert r["fluency_clusters"] == 3
        assert r["fluency_switches"] == 2

    def test_fuzzy_item_match(self):
        tr = _tr(_utt(("word", "con meo")))  # missing diacritic
        r = tasks.score_semantic(tr, _semantic_spec())
        assert r["fluency_valid_unique"] == 1
        assert r["fluency_valid_count"] == 1

    def test_no_items_returns_nans(self):
        tr = _tr(_utt(("word", "not-an-item")))
        r = tasks.score_semantic(tr, _semantic_spec())
        assert math.isnan(r["fluency_rate"])


# --------------------------------------------------------------------------- #
# Score-level guards
# --------------------------------------------------------------------------- #
class TestScorerGuards:
    def test_scores_never_use_diagnosis(self):
        # Scorers accept only a Transcript + spec; no diagnosis/label anywhere.
        assert "diagnosis" not in tasks.__all__
