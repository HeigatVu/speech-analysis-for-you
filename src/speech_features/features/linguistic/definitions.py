"""Stable feature definitions for the adult-neuro lexical/disfluency and
morphosyntax/conversation packs (Tasks 8--9).

Each definition carries the exact key, pack ``adult_neuro``, level
``recording`` or ``utterance``, unit, population applicability ``adult``,
formula version ``1``, and reference ``SAY catalog v1`` required by the
catalog contract. Formula details are documented in
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

MORPH_UPOS_KEYS = (
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

MORPH_DEP_DIST_KEYS = (
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

MORPH_COMPOSITION_KEYS = (
    "morph_content_word_ratio",
    "morph_function_word_ratio",
    "morph_noun_verb_ratio",
    "morph_pronoun_noun_ratio",
    "morph_classifier_ratio",
    "morph_particle_ratio",
    "morph_code_switch_ratio",
)

MORPH_STRUCTURE_KEYS = (
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

MORPHOSYNTAX_KEYS = (
    MORPH_UPOS_KEYS + MORPH_DEP_DIST_KEYS + MORPH_COMPOSITION_KEYS + MORPH_STRUCTURE_KEYS
)
TASK9_RECORDING_KEYS = MORPHOSYNTAX_KEYS + DISCOURSE_RECORDING_KEYS
TASK9_KEYS = TASK9_RECORDING_KEYS + DISCOURSE_UTTERANCE_KEYS
ADULT_NEURO_KEYS = ALL_KEYS + TASK9_KEYS

_COMMON = dict(
    pack="adult_neuro",
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
    ("morph_upos_adj_ratio", "ratio", ()),
    ("morph_upos_adp_ratio", "ratio", ()),
    ("morph_upos_adv_ratio", "ratio", ()),
    ("morph_upos_aux_ratio", "ratio", ()),
    ("morph_upos_cconj_ratio", "ratio", ()),
    ("morph_upos_det_ratio", "ratio", ()),
    ("morph_upos_intj_ratio", "ratio", ()),
    ("morph_upos_noun_ratio", "ratio", ()),
    ("morph_upos_num_ratio", "ratio", ()),
    ("morph_upos_part_ratio", "ratio", ()),
    ("morph_upos_pron_ratio", "ratio", ()),
    ("morph_upos_propn_ratio", "ratio", ()),
    ("morph_upos_punct_ratio", "ratio", ()),
    ("morph_upos_sconj_ratio", "ratio", ()),
    ("morph_upos_sym_ratio", "ratio", ()),
    ("morph_upos_verb_ratio", "ratio", ()),
    ("morph_upos_x_ratio", "ratio", ()),
    ("morph_dep_root_ratio", "ratio", ()),
    ("morph_dep_subject_ratio", "ratio", ()),
    ("morph_dep_object_ratio", "ratio", ()),
    ("morph_dep_nominal_modifier_ratio", "ratio", ()),
    ("morph_dep_adverbial_modifier_ratio", "ratio", ()),
    ("morph_dep_clausal_complement_ratio", "ratio", ()),
    ("morph_dep_coordination_ratio", "ratio", ()),
    ("morph_dep_function_ratio", "ratio", ()),
    ("morph_dep_other_ratio", "ratio", ()),
    ("morph_content_word_ratio", "ratio", ()),
    ("morph_function_word_ratio", "ratio", ()),
    ("morph_noun_verb_ratio", "ratio", ()),
    ("morph_pronoun_noun_ratio", "ratio", ()),
    ("morph_classifier_ratio", "ratio", ()),
    ("morph_particle_ratio", "ratio", ()),
    ("morph_code_switch_ratio", "ratio", ()),
    ("morph_dependency_length_mean_tokens", "tokens", ()),
    ("morph_dependency_length_sd_tokens", "tokens", ()),
    ("morph_tree_depth_mean", "edges", ()),
    ("morph_tree_depth_max", "edges", ()),
    ("morph_clause_count", "count", ()),
    (
        "morph_clause_rate_per_utterance",
        "clauses/utterance",
        ("morph_clause_count", "lex_utterance_count"),
    ),
    ("morph_subordinate_clause_count", "count", ()),
    (
        "morph_subordination_ratio",
        "ratio",
        ("morph_subordinate_clause_count", "morph_clause_count"),
    ),
    ("discourse_turn_count", "count", ()),
    ("discourse_turn_length_mean_words", "words", ()),
    ("discourse_turn_length_sd_words", "words", ()),
    ("discourse_turn_length_mean_syllables", "syllables", ()),
    ("discourse_turn_length_sd_syllables", "syllables", ()),
    ("discourse_examiner_prompt_ratio", "ratio", ()),
    ("discourse_response_latency_mean_s", "s", ()),
    ("discourse_response_latency_sd_s", "s", ()),
    ("discourse_overlap_s", "s", ()),
    ("discourse_overlap_ratio", "ratio", ("discourse_overlap_s",)),
    ("discourse_turn_word_count", "words", (), "utterance"),
    ("discourse_turn_syllable_count", "syllables", (), "utterance"),
    ("discourse_response_latency_s", "s", (), "utterance"),
    ("discourse_turn_overlap_s", "s", (), "utterance"),
)


def _build_definition(spec):
    key, unit, prerequisites = spec[:3]
    level = spec[3] if len(spec) > 3 else "recording"
    return FeatureDefinition(
        key=key, unit=unit, prerequisites=prerequisites, level=level, **_COMMON
    )


_DEFINITIONS = tuple(_build_definition(spec) for spec in _SPEC)

_registered = False


def register_linguistic_features() -> None:
    """Register the adult-neuro lexical, morphosyntax, and conversation
    definitions (idempotent; 97 keys)."""
    global _registered
    if _registered:
        return
    for definition in _DEFINITIONS:
        register_feature(definition)
    _registered = True
