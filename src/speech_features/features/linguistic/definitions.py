"""Stable feature definitions for the adult-neuro lexical/disfluency pack (Task 8).

Each definition carries the exact key, pack ``adult_neuro``, level
``recording``, unit, population applicability ``adult``, formula version
``1``, and reference ``SAY catalog v1`` required by the catalog contract.
Formula details are documented in
:mod:`speech_features.features.linguistic` and its helpers.
"""

from __future__ import annotations

from ...catalog import FeatureDefinition, register_feature

LEX_KEYS = (
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
)

SURFACE_DIVERSITY_KEYS = (
    "lex_token_ttr",
    "lex_token_mattr_20",
    "lex_token_mtld",
    "lex_token_hdd_42",
    "lex_token_hapax_ratio",
    "lex_token_brunet_w",
    "lex_token_honore_r",
    "lex_token_entropy",
)

LEMMA_DIVERSITY_KEYS = (
    "lex_lemma_ttr",
    "lex_lemma_mattr_20",
    "lex_lemma_mtld",
    "lex_lemma_hdd_42",
    "lex_lemma_hapax_ratio",
    "lex_lemma_brunet_w",
    "lex_lemma_honore_r",
    "lex_lemma_entropy",
)

DISFLUENCY_KEYS = (
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

ALL_KEYS = LEX_KEYS + SURFACE_DIVERSITY_KEYS + LEMMA_DIVERSITY_KEYS + DISFLUENCY_KEYS

_COMMON = dict(
    pack="adult_neuro",
    level="recording",
    population="adult",
    formula_version=1,
    reference="SAY catalog v1",
)

# (key, unit, prerequisites)
_SPEC = (
    ("lex_utterance_count", "count", ()),
    ("lex_token_count", "count", ()),
    ("lex_word_count", "count", ()),
    ("lex_syllable_count", "count", ()),
    ("lex_character_count", "count", ()),
    ("lex_unique_token_count", "count", ()),
    ("lex_mlu_words", "words/utterance", ("lex_word_count", "lex_utterance_count")),
    ("lex_mlu_syllables", "syllables/utterance", ("lex_syllable_count", "lex_utterance_count")),
    ("lex_token_length_mean_characters", "characters", ()),
    ("lex_token_length_sd_characters", "characters", ()),
    ("lex_word_length_mean_syllables", "syllables/word", ()),
    ("lex_word_length_sd_syllables", "syllables/word", ()),
    ("lex_token_ttr", "ratio", ()),
    ("lex_token_mattr_20", "ratio", ()),
    ("lex_token_mtld", "words", ()),
    ("lex_token_hdd_42", "ratio", ()),
    ("lex_token_hapax_ratio", "ratio", ()),
    ("lex_token_brunet_w", "index", ()),
    ("lex_token_honore_r", "ratio", ()),
    ("lex_token_entropy", "ratio", ()),
    ("lex_lemma_ttr", "ratio", ()),
    ("lex_lemma_mattr_20", "ratio", ()),
    ("lex_lemma_mtld", "words", ()),
    ("lex_lemma_hdd_42", "ratio", ()),
    ("lex_lemma_hapax_ratio", "ratio", ()),
    ("lex_lemma_brunet_w", "index", ()),
    ("lex_lemma_honore_r", "ratio", ()),
    ("lex_lemma_entropy", "ratio", ()),
    ("disfluency_filler_count", "count", ()),
    ("disfluency_filler_ratio", "ratio", ("disfluency_filler_count",)),
    ("disfluency_fragment_count", "count", ()),
    ("disfluency_fragment_ratio", "ratio", ("disfluency_fragment_count",)),
    ("disfluency_immediate_repetition_count", "count", ()),
    ("disfluency_immediate_repetition_ratio", "ratio", ("disfluency_immediate_repetition_count",)),
    ("disfluency_retracing_count", "count", ()),
    ("disfluency_retracing_ratio", "ratio", ("disfluency_retracing_count",)),
    ("disfluency_revision_count", "count", ()),
    ("disfluency_revision_ratio", "ratio", ("disfluency_revision_count",)),
    ("disfluency_maze_count", "count", ()),
    ("disfluency_maze_ratio", "ratio", ("disfluency_maze_count",)),
    ("disfluency_annotated_error_count", "count", ()),
    ("disfluency_annotated_error_ratio", "ratio", ("disfluency_annotated_error_count",)),
)

_DEFINITIONS = tuple(
    FeatureDefinition(key=key, unit=unit, prerequisites=prerequisites, **_COMMON)
    for key, unit, prerequisites in _SPEC
)

_registered = False


def register_linguistic_features() -> None:
    """Register the adult-neuro lexical/disfluency definitions (idempotent)."""
    global _registered
    if _registered:
        return
    for definition in _DEFINITIONS:
        register_feature(definition)
    _registered = True
