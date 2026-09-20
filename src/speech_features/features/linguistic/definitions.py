"""Stable feature definitions for adult-neuro linguistic and task measures.

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

STRUCTURAL_PSYCHOLINGUISTIC_KEYS = (
    "morph_sentence_count",
    "morph_t_unit_count",
    "morph_words_per_sentence",
    "morph_words_per_t_unit",
    "morph_words_per_clause",
    "morph_clauses_per_sentence",
    "morph_coordinate_phrase_count",
    "morph_complex_nominal_count",
    "morph_verb_phrase_count",
    "morph_embedding_count",
    "morph_dependent_clause_ratio",
    "morph_well_formed_sentence_ratio",
    "morph_incomplete_sentence_ratio",
    "morph_reduced_sentence_ratio",
    "morph_yngve_depth_mean",
    "morph_yngve_depth_max",
    "semantic_idea_density",
    "semantic_proposition_density",
    "lex_frequency_mean",
    "lex_log_frequency_mean",
    "lex_familiarity_mean",
    "lex_age_of_acquisition_mean",
    "lex_imageability_mean",
    "lex_concreteness_mean",
)

ERROR_TYPES = (
    "phonemic",
    "phonetic",
    "semantic",
    "visual",
    "morphological",
    "inflectional",
    "syntactic",
    "closed_class",
    "neologism",
    "perseveration",
    "circumlocution",
    "indefinite_term",
    "word_finding",
)
ERROR_KEYS = tuple(
    key
    for error_type in ERROR_TYPES
    for key in (
        f"disfluency_{error_type}_error_count",
        f"disfluency_{error_type}_error_ratio",
    )
)

DISCOURSE_CLINICAL_KEYS = (
    "discourse_referential_cohesion_ratio",
    "discourse_temporal_cohesion_ratio",
    "discourse_causal_cohesion_ratio",
    "discourse_correct_pronoun_ratio",
    "discourse_local_lexical_coherence",
    "discourse_global_coherence_ratio",
    "discourse_topic_maintenance_ratio",
    "discourse_marker_ratio",
    "discourse_relevant_detail_ratio",
    "discourse_irrelevant_detail_ratio",
    "discourse_microproposition_count",
    "discourse_macroproposition_count",
    "discourse_information_unit_count",
    "discourse_content_accuracy_ratio",
    "discourse_information_efficiency_per_min",
)

CLINICAL_LINGUISTIC_KEYS = STRUCTURAL_PSYCHOLINGUISTIC_KEYS + ERROR_KEYS + DISCOURSE_CLINICAL_KEYS

PICTURE_TASK_KEYS = (
    "task_picture_concept_coverage",
    "task_picture_concept_density",
    "task_picture_repeat_ratio",
    "task_picture_entity_coverage",
    "task_picture_action_coverage",
)
RECALL_TASK_KEYS = (
    "task_recall_idea_coverage",
    "task_recall_idea_density",
    "task_recall_repeat_ratio",
    "task_recall_order_score",
)
FLUENCY_TASK_KEYS = (
    "task_fluency_response_count",
    "task_fluency_valid_count",
    "task_fluency_valid_unique",
    "task_fluency_repeats",
    "task_fluency_intrusions",
    "task_fluency_first_half_valid",
    "task_fluency_second_half_valid",
    "task_fluency_production_change",
    "task_fluency_rate",
    "task_fluency_clusters",
    "task_fluency_cluster_size_mean",
    "task_fluency_switches",
)
TASK_KEYS = PICTURE_TASK_KEYS + RECALL_TASK_KEYS + FLUENCY_TASK_KEYS

_TASK5_COUNT_KEYS = (
    "morph_sentence_count",
    "morph_t_unit_count",
    "morph_coordinate_phrase_count",
    "morph_complex_nominal_count",
    "morph_verb_phrase_count",
    "morph_embedding_count",
    "discourse_microproposition_count",
    "discourse_macroproposition_count",
    "discourse_information_unit_count",
    "task_fluency_response_count",
    "task_fluency_valid_count",
    "task_fluency_valid_unique",
    "task_fluency_repeats",
    "task_fluency_intrusions",
    "task_fluency_first_half_valid",
    "task_fluency_second_half_valid",
    "task_fluency_production_change",
    "task_fluency_clusters",
    "task_fluency_cluster_size_mean",
    "task_fluency_switches",
) + tuple(f"disfluency_{error_type}_error_count" for error_type in ERROR_TYPES)

_TASK5_RATIO_KEYS = (
    "morph_dependent_clause_ratio",
    "morph_well_formed_sentence_ratio",
    "morph_incomplete_sentence_ratio",
    "morph_reduced_sentence_ratio",
    "semantic_idea_density",
    "semantic_proposition_density",
    "discourse_referential_cohesion_ratio",
    "discourse_temporal_cohesion_ratio",
    "discourse_causal_cohesion_ratio",
    "discourse_correct_pronoun_ratio",
    "discourse_local_lexical_coherence",
    "discourse_global_coherence_ratio",
    "discourse_topic_maintenance_ratio",
    "discourse_marker_ratio",
    "discourse_relevant_detail_ratio",
    "discourse_irrelevant_detail_ratio",
    "discourse_content_accuracy_ratio",
    "task_picture_concept_coverage",
    "task_picture_concept_density",
    "task_picture_repeat_ratio",
    "task_picture_entity_coverage",
    "task_picture_action_coverage",
    "task_recall_idea_coverage",
    "task_recall_idea_density",
    "task_recall_repeat_ratio",
) + tuple(f"disfluency_{error_type}_error_ratio" for error_type in ERROR_TYPES)

TASK5_UNITS = {key: "count" for key in _TASK5_COUNT_KEYS}
TASK5_UNITS.update({key: "ratio" for key in _TASK5_RATIO_KEYS})
TASK5_UNITS.update(
    {
        "morph_words_per_sentence": "words",
        "morph_words_per_t_unit": "words",
        "morph_words_per_clause": "words",
        "morph_clauses_per_sentence": "clauses/sentence",
        "morph_yngve_depth_mean": "index",
        "morph_yngve_depth_max": "index",
        "lex_frequency_mean": "score",
        "lex_log_frequency_mean": "score",
        "lex_familiarity_mean": "score",
        "lex_age_of_acquisition_mean": "score",
        "lex_imageability_mean": "score",
        "lex_concreteness_mean": "score",
        "discourse_information_efficiency_per_min": "count/min",
        "task_recall_order_score": "score",
        "task_fluency_rate": "count/min",
    }
)
if set(TASK5_UNITS) != set(CLINICAL_LINGUISTIC_KEYS + TASK_KEYS):
    raise RuntimeError("Task 5 units must cover every Task 5 key exactly")

ADULT_NEURO_RECORDING_KEYS = ALL_KEYS + TASK9_RECORDING_KEYS + CLINICAL_LINGUISTIC_KEYS + TASK_KEYS
ADULT_NEURO_KEYS = ADULT_NEURO_RECORDING_KEYS + DISCOURSE_UTTERANCE_KEYS

_COMMON = dict(
    pack="adult_neuro",
    population="adult",
    formula_version=1,
    reference="SAY catalog v1",
)

# Conservative source-supported metadata for the pre-expansion "SAY catalog
# v1" keys: lexical/morphosyntactic features are language_dependent, and
# disfluency/discourse features are language_sensitive. Disorder unions come
# from docs/2026-08-10/neurodegenerative-speech-feature-expansion/1/RESEARCH-2026-08-10.md: lexical and
# morphosyntactic from sources 2--4 and 9; disfluency from source 3; discourse
# from sources 2--3.
_LEXICAL_DISORDERS = (
    "ad",
    "als",
    "cbs",
    "dlb",
    "ftd",
    "hd",
    "mci",
    "pd",
    "pdd",
    "ppa",
    "psp",
)
_DISFLUENCY_DISORDERS = ("ad", "als", "cbs", "dlb", "ftd", "hd", "mci", "pd", "pdd", "ppa")
_DISCOURSE_DISORDERS = _DISFLUENCY_DISORDERS


def _legacy_metadata(key: str) -> dict:
    if key.startswith("lex_"):
        return dict(
            domain="lexical",
            language_scope="language_dependent",
            tasks=("connected_speech",),
            disorders=_LEXICAL_DISORDERS,
            evidence_level="derived_companion",
        )
    if key.startswith("morph_"):
        return dict(
            domain="morphosyntactic",
            language_scope="language_dependent",
            tasks=("connected_speech",),
            disorders=_LEXICAL_DISORDERS,
            evidence_level="derived_companion",
        )
    if key.startswith("disfluency_"):
        return dict(
            domain="disfluency",
            language_scope="language_sensitive",
            tasks=("connected_speech",),
            disorders=_DISFLUENCY_DISORDERS,
            evidence_level="derived_companion",
        )
    if key.startswith("discourse_"):
        return dict(
            domain="discourse",
            language_scope="language_sensitive",
            tasks=("connected_speech",),
            disorders=_DISCOURSE_DISORDERS,
            evidence_level="derived_companion",
        )
    raise ValueError(f"legacy adult-neuro key {key!r} has no backfilled metadata")


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
    metadata = _LEGACY_LINGUISTIC_METADATA(key)
    return FeatureDefinition(
        key=key,
        unit=unit,
        prerequisites=prerequisites,
        level=level,
        **_COMMON,
        **metadata,
    )


# Backfilled metadata for the pre-expansion "SAY catalog v1" keys (Tasks 1-5).
# Domains use the reviewed lexical/morphosyntactic/disfluency/discourse set;
# scopes are language-dependent for lexical and morphosyntactic measures and
# language-sensitive for disfluency and discourse. Disorders come from the
# conservative source-supported unions of the sources document.
_LEXICAL_DISORDERS = (
    "ad",
    "als",
    "cbs",
    "dlb",
    "ftd",
    "hd",
    "mci",
    "pd",
    "pdd",
    "ppa",
    "psp",
)
_MORPHOSYNTACTIC_DISORDERS = _LEXICAL_DISORDERS
_DISFLUENCY_DISORDERS = (
    "ad",
    "als",
    "cbs",
    "dlb",
    "ftd",
    "hd",
    "mci",
    "pd",
    "pdd",
    "ppa",
)
_DISCOURSE_DISORDERS = _DISFLUENCY_DISORDERS


def _LEGACY_LINGUISTIC_METADATA(key: str) -> dict:
    if key.startswith("lex_"):
        return dict(
            domain="lexical",
            language_scope="language_dependent",
            tasks=("connected_speech",),
            disorders=_LEXICAL_DISORDERS,
            evidence_level="derived_companion",
        )
    if key.startswith("morph_"):
        return dict(
            domain="morphosyntactic",
            language_scope="language_dependent",
            tasks=("connected_speech",),
            disorders=_MORPHOSYNTACTIC_DISORDERS,
            evidence_level="derived_companion",
        )
    if key.startswith("disfluency_"):
        return dict(
            domain="disfluency",
            language_scope="language_sensitive",
            tasks=("connected_speech",),
            disorders=_DISFLUENCY_DISORDERS,
            evidence_level="derived_companion",
        )
    if key.startswith("discourse_"):
        return dict(
            domain="discourse",
            language_scope="language_sensitive",
            tasks=("connected_speech",),
            disorders=_DISCOURSE_DISORDERS,
            evidence_level="derived_companion",
        )
    raise ValueError(f"legacy linguistic key {key!r} has no backfilled metadata")


_DEFINITIONS = tuple(_build_definition(spec) for spec in _SPEC)


def _new_definition(key):
    if key.startswith("morph_"):
        domain = "morphosyntactic"
    elif key.startswith("semantic_"):
        domain = "semantic"
    elif key.startswith("lex_"):
        domain = "psycholinguistic"
    elif key.startswith("disfluency_"):
        domain = "disfluency"
    elif key.startswith("discourse_"):
        domain = "discourse"
    else:
        domain = "task"
    tasks = ()
    if key.startswith("task_picture_"):
        tasks = ("picture_description",)
    elif key.startswith("task_recall_"):
        tasks = ("story_recall",)
    elif key.startswith("task_fluency_"):
        tasks = ("phonemic_fluency", "semantic_fluency")
    return FeatureDefinition(
        key=key,
        pack="adult_neuro",
        level="recording",
        unit=TASK5_UNITS[key],
        population="adult",
        reference="SAY catalog v1",
        formula_version=1,
        domain=domain,
        language_scope="language_sensitive",
        tasks=tasks,
    )


_DEFINITIONS += tuple(_new_definition(key) for key in CLINICAL_LINGUISTIC_KEYS + TASK_KEYS)

_registered = False


def register_linguistic_features() -> None:
    """Register the adult-neuro lexical, morphosyntax, and conversation
    definitions (idempotent)."""
    global _registered
    if _registered:
        return
    for definition in _DEFINITIONS:
        register_feature(definition)
    _registered = True
