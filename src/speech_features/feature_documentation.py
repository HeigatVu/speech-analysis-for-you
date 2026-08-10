"""Single source of feature-specific formula and missing-data catalog text.

``documentation_for`` resolves the exact ``(formula, missing_data)`` pair for
every live catalog key: the restored 170 legacy pairs from
:mod:`speech_features._legacy_feature_docs`, explicit per-key text for every
Task 2--6 key, and detail-bearing pattern text for the MFCC, error-type, and
eGeMAPS families. Unknown keys raise :class:`KeyError`; no generic fallback
text exists.
"""

from __future__ import annotations

import re

from ._legacy_feature_docs import LEGACY_FEATURE_DOCS
from .features.standardized.definitions import EGEMAPS_KEYS

STANDARDIZED_MISSING = (
    "NaN plus one MISSING_OPTIONAL_DEPENDENCY issue for the whole recording "
    "when the optional standardized-acoustic extra (openSMILE) is unavailable."
)

TIMING_COMPANION_DOCS = {
    "time_pause_total_s": (
        "Sum of pause durations (each at least pause_threshold_s, 0.2 s default) "
        "over the target intervals.",
        "0 when no pause meets pause_threshold_s; NaN plus NO_SPEECH when no voiced frame exists.",
    ),
    "time_pause_median_s": (
        "Median pause duration (pauses of at least pause_threshold_s).",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES when no pause meets pause_threshold_s; "
        "NaN plus NO_SPEECH when no voiced frame exists.",
    ),
    "time_pause_iqr_s": (
        "Interquartile range (P75 minus P25) of qualifying pause durations.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES when no pause meets pause_threshold_s; "
        "NaN plus NO_SPEECH when no voiced frame exists.",
    ),
    "time_pause_cv": (
        "Population SD divided by the mean of qualifying pause durations.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES when no pause meets pause_threshold_s; "
        "NaN plus NO_SPEECH when no voiced frame exists.",
    ),
    "time_pause_proportion": (
        "Total qualifying pause duration divided by the recording duration.",
        "0 when no pause meets pause_threshold_s; NaN plus NO_SPEECH when no voiced frame exists.",
    ),
    "time_between_utterance_pause_proportion": (
        "Duration of inter-utterance gaps of at least pause_threshold_s divided by "
        "the total qualifying pause duration.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES when no pause meets pause_threshold_s; "
        "NaN plus NO_SPEECH when no voiced frame exists.",
    ),
    "time_speech_segment_count": (
        "Number of merged target-speaker aligned interval segments.",
        "NaN plus MISSING_ANNOTATION when no aligned target intervals exist "
        "(without allow_unaligned).",
    ),
    "time_speech_segment_rate_per_min": (
        "Merged target-interval segment count divided by the recording duration in minutes.",
        "NaN plus MISSING_ANNOTATION when no aligned target intervals exist "
        "(without allow_unaligned).",
    ),
    "time_speech_segment_median_s": (
        "Median duration of the merged target-speaker aligned intervals.",
        "NaN plus MISSING_ANNOTATION when no aligned target intervals exist "
        "(without allow_unaligned).",
    ),
    "time_speech_segment_iqr_s": (
        "Interquartile range (P75 minus P25) of merged target interval durations.",
        "NaN plus MISSING_ANNOTATION when no aligned target intervals exist "
        "(without allow_unaligned).",
    ),
    "time_speech_segment_cv": (
        "Population SD divided by the mean of merged target interval durations.",
        "NaN plus MISSING_ANNOTATION when no aligned target intervals exist "
        "(without allow_unaligned).",
    ),
    "time_speech_segment_max_s": (
        "Maximum duration of a merged target-speaker aligned interval.",
        "NaN plus MISSING_ANNOTATION when no aligned target intervals exist "
        "(without allow_unaligned).",
    ),
    "time_max_local_speech_rate_wpm": (
        "Maximum words-per-minute over every sliding window of three consecutive "
        "target utterances with positive speech duration; words group by word_id "
        "within an utterance.",
        "NaN plus MISSING_ANNOTATION when fewer than three aligned utterances with "
        "word tokens exist.",
    ),
    "time_timing_event_rate_per_min": (
        "Coalesced voiced/unvoiced/pause timing-event count divided by the recording "
        "duration in minutes.",
        "NaN plus NO_SPEECH when no voiced frame exists.",
    ),
    "time_timing_event_entropy": (
        "Normalized Shannon entropy (log base 3) over voiced/unvoiced/pause event-class "
        "proportions.",
        "NaN plus NO_SPEECH when no voiced frame exists.",
    ),
    "time_timing_acceleration_per_min2": (
        "Second-half event rate minus first-half event rate divided by the recording "
        "duration in minutes.",
        "NaN plus NO_SPEECH when no voiced frame exists.",
    ),
}

VOICE_COMPANION_DOCS = {
    "voice_break_count": (
        "Voice-break count: unvoiced runs of at least pause_threshold_s bounded by a "
        "voiced frame on both sides.",
        "NaN plus INSUFFICIENT_VOICED_FRAMES when no voiced frame exists in the target audio.",
    ),
    "voice_break_rate_per_min": (
        "Voice-break count divided by the recording duration in minutes.",
        "NaN plus INSUFFICIENT_VOICED_FRAMES when no voiced frame exists in the target audio.",
    ),
    "voice_break_proportion": (
        "Total bounded unvoiced-run duration divided by the analyzed target duration.",
        "NaN plus INSUFFICIENT_VOICED_FRAMES when no voiced frame exists in the target audio.",
    ),
    "voice_f0_range_semitones": (
        "Max minus min of 12*log2(f0/median(f0)) over voiced frames.",
        "NaN plus INSUFFICIENT_VOICED_FRAMES when fewer than two voiced frames exist.",
    ),
    "voice_f0_mad_semitones": (
        "Median absolute deviation of 12*log2(f0/median(f0)) over voiced frames.",
        "NaN plus INSUFFICIENT_VOICED_FRAMES when fewer than two voiced frames exist.",
    ),
    "voice_intensity_range_db": (
        "Max minus min of frame intensity (20*log10(rms)) over voiced speech frames.",
        "NaN plus INSUFFICIENT_VOICED_FRAMES when fewer than two voiced frames exist.",
    ),
    "voice_intensity_cv": (
        "Population SD divided by the mean of speech-frame RMS amplitudes.",
        "NaN plus INSUFFICIENT_VOICED_FRAMES when fewer than two voiced frames exist.",
    ),
    "voice_nhr_mean_db": (
        "Mean of -HNR per frame (negative harmonics-to-noise ratio in dB).",
        "NaN plus INSUFFICIENT_VOICED_FRAMES when fewer than two voiced frames exist.",
    ),
    "voice_jitter_rap": (
        "Regional perturbation quotient with a 3-period window over the period series.",
        "NaN plus INSUFFICIENT_CYCLES when fewer than two voiced frames exist.",
    ),
    "voice_jitter_ppq5": (
        "Regional perturbation quotient with a 5-period window over the period series.",
        "NaN plus INSUFFICIENT_CYCLES when fewer than two voiced frames exist.",
    ),
    "voice_jitter_ddp": (
        "3 times voice_jitter_rap (triple perturbation).",
        "NaN plus INSUFFICIENT_CYCLES when fewer than two voiced frames exist.",
    ),
    "voice_shimmer_apq3": (
        "Regional perturbation quotient of RMS amplitudes with a 3-frame window.",
        "NaN plus INSUFFICIENT_CYCLES when fewer than two voiced frames exist.",
    ),
    "voice_shimmer_apq5": (
        "Regional perturbation quotient of RMS amplitudes with a 5-frame window.",
        "NaN plus INSUFFICIENT_CYCLES when fewer than two voiced frames exist.",
    ),
    "voice_shimmer_apq11": (
        "Regional perturbation quotient of RMS amplitudes with an 11-frame window.",
        "NaN plus INSUFFICIENT_CYCLES when fewer than two voiced frames exist.",
    ),
    "voice_shimmer_dda": (
        "3 times voice_shimmer_apq3.",
        "NaN plus INSUFFICIENT_CYCLES when fewer than two voiced frames exist.",
    ),
    "voice_pitch_period_entropy": (
        "Normalized Shannon entropy of detrended log periods over entropy_bins "
        "(default 100) fixed bins.",
        "NaN plus INSUFFICIENT_VOICING when fewer than nonlinear_min_periods "
        "(default 300) voiced periods exist.",
    ),
    "voice_rpde": (
        "Recurrence-period density entropy: normalized entropy of recurrence counts "
        "at lags 1 through min(100, n/2) with radius 0.1*SD.",
        "NaN plus INSUFFICIENT_VOICING when fewer than nonlinear_min_periods "
        "(default 300) voiced periods exist.",
    ),
    "voice_dfa": (
        "Detrended fluctuation analysis log-log slope over non-overlapping windows "
        "of 4 through 64 periods.",
        "NaN plus INSUFFICIENT_VOICING when fewer than nonlinear_min_periods "
        "(default 300) voiced periods exist.",
    ),
    "voice_correlation_dimension": (
        "Correlation-sum slope over 8 geometrically spaced radii between the 10th "
        "and 60th percentile distances of the delay-one embedding.",
        "NaN plus INSUFFICIENT_VOICING when fewer than nonlinear_min_periods "
        "(default 300) voiced periods exist.",
    ),
}

ADVANCED_SPECTRAL_DOCS = {
    "spectral_energy_mean_db": (
        "Mean over frames of 10*log10(total power) of the target speech frames.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES when no target speech frames exist.",
    ),
    "spectral_energy_sd_db": (
        "Population SD over frames of 10*log10(total power) of the target speech frames.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES when no target speech frames exist.",
    ),
    "spectral_skewness_mean": (
        "Mean over frames of the third standardized spectral moment around the centroid.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES when no target speech frames exist.",
    ),
    "spectral_skewness_sd": (
        "Population SD over frames of the third standardized spectral moment.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES when no target speech frames exist.",
    ),
    "spectral_kurtosis_mean": (
        "Mean over frames of the fourth standardized spectral moment.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES when no target speech frames exist.",
    ),
    "spectral_kurtosis_sd": (
        "Population SD over frames of the fourth standardized spectral moment.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES when no target speech frames exist.",
    ),
    "spectral_low_high_energy_ratio_db": (
        "10*log10 of total power below sample_rate/4 divided by total power above it.",
        "NaN plus INSUFFICIENT_SPEECH_FRAMES when no target speech frames exist.",
    ),
}

STRUCTURAL_PSYCHOLINGUISTIC_DOCS = {
    "morph_sentence_count": (
        "Number of distinct sentence_id values over target word tokens.",
        "NaN plus MISSING_ANNOTATION when sentence_id annotation is absent or incomplete "
        "for target words; MISSING_ANNOTATION without a target word sample.",
    ),
    "morph_t_unit_count": (
        "Number of distinct t_unit_id values over target word tokens.",
        "NaN plus MISSING_ANNOTATION when t_unit_id annotation is absent or incomplete "
        "for target words; MISSING_ANNOTATION without a target word sample.",
    ),
    "morph_words_per_sentence": (
        "lex_token_count divided by morph_sentence_count.",
        "NaN plus MISSING_ANNOTATION when sentence_id annotation is absent or incomplete "
        "for target words; MISSING_ANNOTATION without a target word sample.",
    ),
    "morph_words_per_t_unit": (
        "lex_token_count divided by morph_t_unit_count.",
        "NaN plus MISSING_ANNOTATION when t_unit_id annotation is absent or incomplete "
        "for target words; MISSING_ANNOTATION without a target word sample.",
    ),
    "morph_words_per_clause": (
        "lex_token_count divided by the number of distinct (sentence_id, clause_id) pairs.",
        "NaN plus MISSING_ANNOTATION when clause_id annotation is absent or incomplete "
        "for target words; MISSING_ANNOTATION without a target word sample.",
    ),
    "morph_clauses_per_sentence": (
        "Distinct (sentence_id, clause_id) pairs divided by morph_sentence_count.",
        "NaN plus MISSING_ANNOTATION when clause_id or sentence_id annotation is absent "
        "or incomplete for target words; MISSING_ANNOTATION without a target word sample.",
    ),
    "morph_coordinate_phrase_count": (
        "Count of coordinate phrase_type runs over target utterances (consecutive "
        "same-phrase tokens count once).",
        "NaN plus MISSING_ANNOTATION when phrase_type annotation is absent or incomplete "
        "for target words; MISSING_ANNOTATION without a target word sample.",
    ),
    "morph_complex_nominal_count": (
        "Count of complex_nominal phrase_type runs over target utterances (consecutive "
        "same-phrase tokens count once).",
        "NaN plus MISSING_ANNOTATION when phrase_type annotation is absent or incomplete "
        "for target words; MISSING_ANNOTATION without a target word sample.",
    ),
    "morph_verb_phrase_count": (
        "Count of verb_phrase phrase_type runs over target utterances (consecutive "
        "same-phrase tokens count once).",
        "NaN plus MISSING_ANNOTATION when phrase_type annotation is absent or incomplete "
        "for target words; MISSING_ANNOTATION without a target word sample.",
    ),
    "morph_embedding_count": (
        "Count of distinct clauses with clause_type embedded.",
        "NaN plus MISSING_ANNOTATION when clause_type annotation is absent, incomplete, "
        "or inconsistent for target words; MISSING_ANNOTATION without a target word sample.",
    ),
    "morph_dependent_clause_ratio": (
        "Distinct clauses of type dependent/embedded/subordinate divided by all distinct clauses.",
        "NaN plus MISSING_ANNOTATION when clause_type annotation is absent, incomplete, "
        "or inconsistent for target words; MISSING_ANNOTATION without a target word sample.",
    ),
    "morph_well_formed_sentence_ratio": (
        "Well-formed sentences divided by all sentences (sentence_status layer).",
        "NaN plus MISSING_ANNOTATION when sentence_status annotation is absent, incomplete, "
        "or inconsistent for target words; MISSING_ANNOTATION without a target word sample.",
    ),
    "morph_incomplete_sentence_ratio": (
        "Incomplete sentences divided by all sentences (sentence_status layer).",
        "NaN plus MISSING_ANNOTATION when sentence_status annotation is absent, incomplete, "
        "or inconsistent for target words; MISSING_ANNOTATION without a target word sample.",
    ),
    "morph_reduced_sentence_ratio": (
        "Reduced sentences divided by all sentences (sentence_status layer).",
        "NaN plus MISSING_ANNOTATION when sentence_status annotation is absent, incomplete, "
        "or inconsistent for target words; MISSING_ANNOTATION without a target word sample.",
    ),
    "morph_yngve_depth_mean": (
        "Mean of the yngve_depth annotation over target word tokens.",
        "NaN plus MISSING_ANNOTATION when yngve_depth annotation is absent or incomplete "
        "for target words; MISSING_ANNOTATION without a target word sample.",
    ),
    "morph_yngve_depth_max": (
        "Maximum of the yngve_depth annotation over target word tokens.",
        "NaN plus MISSING_ANNOTATION when yngve_depth annotation is absent or incomplete "
        "for target words; MISSING_ANNOTATION without a target word sample.",
    ),
    "semantic_idea_density": (
        "Number of distinct information_unit values divided by lex_token_count.",
        "NaN plus MISSING_ANNOTATION when information_unit annotation is absent; "
        "MISSING_ANNOTATION without a target word sample.",
    ),
    "semantic_proposition_density": (
        "Count of microproposition plus macroproposition discourse_role tokens divided "
        "by lex_token_count.",
        "NaN plus MISSING_ANNOTATION when discourse_role annotation is absent; "
        "MISSING_ANNOTATION without a target word sample.",
    ),
    "lex_frequency_mean": (
        "Mean of the frequency annotation over target word tokens.",
        "NaN plus MISSING_ANNOTATION when the frequency annotation layer is absent or "
        "incomplete for target words; MISSING_ANNOTATION without a target word sample.",
    ),
    "lex_log_frequency_mean": (
        "Mean of the log_frequency annotation over target word tokens.",
        "NaN plus MISSING_ANNOTATION when the log_frequency annotation layer is absent or "
        "incomplete for target words; MISSING_ANNOTATION without a target word sample.",
    ),
    "lex_familiarity_mean": (
        "Mean of the familiarity annotation over target word tokens.",
        "NaN plus MISSING_ANNOTATION when the familiarity annotation layer is absent or "
        "incomplete for target words; MISSING_ANNOTATION without a target word sample.",
    ),
    "lex_age_of_acquisition_mean": (
        "Mean of the age_of_acquisition annotation over target word tokens.",
        "NaN plus MISSING_ANNOTATION when the age_of_acquisition annotation layer is "
        "absent or incomplete for target words; MISSING_ANNOTATION without a target word sample.",
    ),
    "lex_imageability_mean": (
        "Mean of the imageability annotation over target word tokens.",
        "NaN plus MISSING_ANNOTATION when the imageability annotation layer is absent or "
        "incomplete for target words; MISSING_ANNOTATION without a target word sample.",
    ),
    "lex_concreteness_mean": (
        "Mean of the concreteness annotation over target word tokens.",
        "NaN plus MISSING_ANNOTATION when the concreteness annotation layer is absent or "
        "incomplete for target words; MISSING_ANNOTATION without a target word sample.",
    ),
}

DISCOURSE_CLINICAL_DOCS = {
    "discourse_referential_cohesion_ratio": (
        "Referential cohesion_type tokens divided by all cohesion_type tokens.",
        "NaN plus MISSING_ANNOTATION when cohesion_type annotation has no target "
        "observations; MISSING_ANNOTATION without a target word sample.",
    ),
    "discourse_temporal_cohesion_ratio": (
        "Temporal cohesion_type tokens divided by all cohesion_type tokens.",
        "NaN plus MISSING_ANNOTATION when cohesion_type annotation has no target "
        "observations; MISSING_ANNOTATION without a target word sample.",
    ),
    "discourse_causal_cohesion_ratio": (
        "Causal cohesion_type tokens divided by all cohesion_type tokens.",
        "NaN plus MISSING_ANNOTATION when cohesion_type annotation has no target "
        "observations; MISSING_ANNOTATION without a target word sample.",
    ),
    "discourse_correct_pronoun_ratio": (
        "Mean of the pronoun_reference_correct annotation (0 or 1) over target tokens.",
        "NaN plus MISSING_ANNOTATION when pronoun_reference_correct annotation has no "
        "valid target observations; MISSING_ANNOTATION without a target word sample.",
    ),
    "discourse_local_lexical_coherence": (
        "Mean Jaccard similarity (intersection over union) of normalized word sets "
        "between adjacent target utterances.",
        "NaN plus INSUFFICIENT_TOKENS when fewer than two adjacent target utterances "
        "with a defined union exist; MISSING_ANNOTATION without a target word sample.",
    ),
    "discourse_global_coherence_ratio": (
        "Mean of the topic_relevant annotation (0 or 1) over target word tokens.",
        "NaN plus MISSING_ANNOTATION when topic_relevant annotation is absent or "
        "incomplete for target words; MISSING_ANNOTATION without a target word sample.",
    ),
    "discourse_topic_maintenance_ratio": (
        "Share of target utterances with at least one topic-relevant word.",
        "NaN plus MISSING_ANNOTATION when topic_relevant annotation is absent or "
        "incomplete for target words; MISSING_ANNOTATION without a target word sample.",
    ),
    "discourse_marker_ratio": (
        "Marker discourse_role tokens divided by lex_token_count.",
        "NaN plus MISSING_ANNOTATION when discourse_role annotation is absent; "
        "MISSING_ANNOTATION without a target word sample.",
    ),
    "discourse_relevant_detail_ratio": (
        "Relevant-detail tokens divided by all detail tokens (relevant plus irrelevant).",
        "NaN plus INSUFFICIENT_TOKENS when no reviewed detail observations exist; "
        "MISSING_ANNOTATION when discourse_role annotation is absent or there is no "
        "target word sample.",
    ),
    "discourse_irrelevant_detail_ratio": (
        "Irrelevant-detail tokens divided by all detail tokens (relevant plus irrelevant).",
        "NaN plus INSUFFICIENT_TOKENS when no reviewed detail observations exist; "
        "MISSING_ANNOTATION when discourse_role annotation is absent or there is no "
        "target word sample.",
    ),
    "discourse_microproposition_count": (
        "Count of microproposition discourse_role tokens.",
        "NaN plus MISSING_ANNOTATION when discourse_role annotation is absent; "
        "MISSING_ANNOTATION without a target word sample.",
    ),
    "discourse_macroproposition_count": (
        "Count of macroproposition discourse_role tokens.",
        "NaN plus MISSING_ANNOTATION when discourse_role annotation is absent; "
        "MISSING_ANNOTATION without a target word sample.",
    ),
    "discourse_information_unit_count": (
        "Number of distinct information_unit values.",
        "NaN plus MISSING_ANNOTATION when information_unit annotation is absent; "
        "MISSING_ANNOTATION without a target word sample.",
    ),
    "discourse_content_accuracy_ratio": (
        "Distinct information units present in the task reference divided by all "
        "distinct information units.",
        "NaN plus MISSING_ANNOTATION when the task reference is unavailable; "
        "NaN plus INSUFFICIENT_TOKENS when no information units exist; "
        "MISSING_ANNOTATION without a target word sample.",
    ),
    "discourse_information_efficiency_per_min": (
        "Distinct information_unit values divided by the target utterance span in minutes.",
        "NaN plus INSUFFICIENT_TOKENS when the target timestamp span is zero; "
        "MISSING_ANNOTATION when information_unit annotation is absent or there is no "
        "target word sample.",
    ),
}

TASK_DOCS = {
    "task_picture_concept_coverage": (
        "Distinct matched concept_aliases divided by the task spec's concept count.",
        "NaN plus INSUFFICIENT_TOKENS when the concept count is zero; MISSING_ANNOTATION "
        "without a target word sample or task spec.",
    ),
    "task_picture_concept_density": (
        "Distinct matched concepts divided by the target word count.",
        "NaN plus INSUFFICIENT_TOKENS when the word count is zero; MISSING_ANNOTATION "
        "without a target word sample or task spec.",
    ),
    "task_picture_repeat_ratio": (
        "(Matched occurrences minus distinct concepts) divided by matched occurrences.",
        "NaN plus INSUFFICIENT_TOKENS when no concept matches; MISSING_ANNOTATION without "
        "a target word sample or task spec.",
    ),
    "task_picture_entity_coverage": (
        "Distinct matched entity_groups divided by the task spec's entity-group count.",
        "NaN plus INSUFFICIENT_TOKENS when the entity-group count is zero; "
        "MISSING_ANNOTATION without a target word sample or task spec.",
    ),
    "task_picture_action_coverage": (
        "Distinct matched action_groups divided by the task spec's action-group count.",
        "NaN plus INSUFFICIENT_TOKENS when the action-group count is zero; "
        "MISSING_ANNOTATION without a target word sample or task spec.",
    ),
    "task_recall_idea_coverage": (
        "Distinct matched idea_aliases divided by the task spec's idea count.",
        "NaN plus INSUFFICIENT_TOKENS when the idea count is zero; MISSING_ANNOTATION "
        "without a target word sample or task spec.",
    ),
    "task_recall_idea_density": (
        "Distinct matched ideas divided by the target word count.",
        "NaN plus INSUFFICIENT_TOKENS when the word count is zero; MISSING_ANNOTATION "
        "without a target word sample or task spec.",
    ),
    "task_recall_repeat_ratio": (
        "(Matched occurrences minus distinct ideas) divided by matched occurrences.",
        "NaN plus INSUFFICIENT_TOKENS when no ideas match; MISSING_ANNOTATION without a "
        "target word sample or task spec.",
    ),
    "task_recall_order_score": (
        "Longest common subsequence of the matched idea order divided by the reference idea count.",
        "NaN plus INSUFFICIENT_TOKENS when no ideas match or the reference is empty; "
        "MISSING_ANNOTATION without a target word sample or task spec.",
    ),
    "task_fluency_response_count": (
        "Number of target utterances containing at least one word.",
        "NaN plus MISSING_ANNOTATION without a target word sample or task spec.",
    ),
    "task_fluency_valid_count": (
        "Responses valid for the task (phonemic: starts with an initial and is not "
        "excluded; semantic: matches an item alias).",
        "NaN plus MISSING_ANNOTATION without a target word sample or task spec.",
    ),
    "task_fluency_valid_unique": (
        "Distinct valid responses.",
        "NaN plus MISSING_ANNOTATION without a target word sample or task spec.",
    ),
    "task_fluency_repeats": (
        "Valid responses minus distinct valid responses.",
        "NaN plus MISSING_ANNOTATION without a target word sample or task spec.",
    ),
    "task_fluency_intrusions": (
        "All responses minus valid responses.",
        "NaN plus MISSING_ANNOTATION without a target word sample or task spec.",
    ),
    "task_fluency_first_half_valid": (
        "Valid responses before the midpoint of the response span.",
        "NaN plus MISSING_ANNOTATION without a target word sample or task spec.",
    ),
    "task_fluency_second_half_valid": (
        "Valid responses at or after the midpoint of the response span.",
        "NaN plus MISSING_ANNOTATION without a target word sample or task spec.",
    ),
    "task_fluency_production_change": (
        "Second-half valid responses minus first-half valid responses.",
        "NaN plus MISSING_ANNOTATION without a target word sample or task spec.",
    ),
    "task_fluency_rate": (
        "Distinct valid responses divided by the response span in minutes.",
        "NaN plus INSUFFICIENT_TOKENS when the response span is zero; MISSING_ANNOTATION "
        "without a target word sample or task spec.",
    ),
    "task_fluency_clusters": (
        "Number of runs of the same semantic subcategory (semantic fluency only).",
        "NaN plus MISSING_ANNOTATION without a semantic-fluency task spec or target word sample.",
    ),
    "task_fluency_cluster_size_mean": (
        "Valid responses divided by the cluster count (semantic fluency only).",
        "NaN plus INSUFFICIENT_TOKENS when no clusters exist; MISSING_ANNOTATION without "
        "a semantic-fluency task spec or target word sample.",
    ),
    "task_fluency_switches": (
        "Adjacent response pairs whose semantic subcategory differs (semantic fluency only).",
        "NaN plus MISSING_ANNOTATION without a semantic-fluency task spec or target word sample.",
    ),
}

MOTOR_DOCS = {
    "artic_vowel_space_area_hz2": (
        "Area of the i/a/u vowel triangle in the (F1, F2) plane from median formants "
        "per vowel (shoelace polygon area).",
        "NaN plus INVALID_TASK_ANNOTATION when i/a/u formant or label annotations are "
        "missing or malformed; NaN plus INVALID_TASK_ANNOTATION when the VAI/FCR "
        "denominators are non-positive.",
    ),
    "artic_vowel_articulation_index": (
        "(F2_i + F1_a) divided by (F1_i + F1_u + F2_u + F2_a) over median i/a/u formants.",
        "NaN plus INVALID_TASK_ANNOTATION when i/a/u formant or label annotations are "
        "missing or malformed; NaN plus INVALID_TASK_ANNOTATION when the VAI/FCR "
        "denominators are non-positive.",
    ),
    "artic_formant_centralization_ratio": (
        "Inverse of the articulation index: (F1_i + F1_u + F2_u + F2_a) divided by (F2_i + F1_a).",
        "NaN plus INVALID_TASK_ANNOTATION when i/a/u formant or label annotations are "
        "missing or malformed; NaN plus INVALID_TASK_ANNOTATION when the VAI/FCR "
        "denominators are non-positive.",
    ),
    "artic_vowel_dispersion_mean_hz": (
        "Mean Euclidean distance of each vowel's (F1, F2) from the vowel centroid.",
        "NaN plus INVALID_TASK_ANNOTATION when vowel F1/F2 annotations are missing or "
        "malformed; MISSING_ANNOTATION without vowel tokens.",
    ),
    "artic_vowel_dispersion_sd_hz": (
        "Population SD of vowel distances from the vowel centroid.",
        "NaN plus INVALID_TASK_ANNOTATION when vowel F1/F2 annotations are missing or "
        "malformed; MISSING_ANNOTATION without vowel tokens.",
    ),
    "artic_f1_within_vowel_sd_hz": (
        "Mean over vowel labels of the within-label population SD of F1.",
        "NaN plus INVALID_TASK_ANNOTATION when vowel labels or F1/F2 annotations are "
        "missing or malformed; MISSING_ANNOTATION without vowel tokens.",
    ),
    "artic_f2_within_vowel_sd_hz": (
        "Mean over vowel labels of the within-label population SD of F2.",
        "NaN plus INVALID_TASK_ANNOTATION when vowel labels or F1/F2 annotations are "
        "missing or malformed; MISSING_ANNOTATION without vowel tokens.",
    ),
    "artic_formant_transition_slope_mean_hz_s": (
        "Mean of abs(F2 difference)/(onset time difference) over consecutive timed vowels.",
        "NaN plus INVALID_TASK_ANNOTATION when F2 annotations are missing or malformed "
        "or vowel onsets are not strictly increasing; MISSING_ANNOTATION with fewer than "
        "two timed vowels.",
    ),
    "artic_vot_mean_s": (
        "Mean of the vot_s annotation over stop tokens.",
        "NaN plus INVALID_TASK_ANNOTATION when vot_s annotations are missing or malformed.",
    ),
    "artic_vot_sd_s": (
        "Population SD of the vot_s annotation over stop tokens.",
        "NaN plus INVALID_TASK_ANNOTATION when vot_s annotations are missing or malformed.",
    ),
    "artic_stop_gap_mean_s": (
        "Mean of the stop_gap_s annotation over stop tokens.",
        "NaN plus INVALID_TASK_ANNOTATION when stop_gap_s annotations are missing or malformed.",
    ),
    "artic_stop_gap_sd_s": (
        "Population SD of the stop_gap_s annotation over stop tokens.",
        "NaN plus INVALID_TASK_ANNOTATION when stop_gap_s annotations are missing or malformed.",
    ),
    "artic_consonant_duration_mean_s": (
        "Mean duration (end_s minus start_s) of consonant/stop/fricative tokens.",
        "NaN plus MISSING_ANNOTATION when complete positive consonant token times are unavailable.",
    ),
    "artic_consonant_duration_sd_s": (
        "Population SD of consonant/stop/fricative token durations.",
        "NaN plus MISSING_ANNOTATION when complete positive consonant token times are unavailable.",
    ),
    "artic_fricative_m1_mean": (
        "Mean of the spectral_moment_1 annotation over fricative tokens.",
        "NaN plus INVALID_TASK_ANNOTATION when spectral_moment_1 annotations are missing "
        "or malformed.",
    ),
    "artic_fricative_m2_mean": (
        "Mean of the spectral_moment_2 annotation over fricative tokens.",
        "NaN plus INVALID_TASK_ANNOTATION when spectral_moment_2 annotations are missing "
        "or malformed.",
    ),
    "artic_fricative_m3_mean": (
        "Mean of the spectral_moment_3 annotation over fricative tokens.",
        "NaN plus INVALID_TASK_ANNOTATION when spectral_moment_3 annotations are missing "
        "or malformed.",
    ),
    "artic_fricative_m4_mean": (
        "Mean of the spectral_moment_4 annotation over fricative tokens.",
        "NaN plus INVALID_TASK_ANNOTATION when spectral_moment_4 annotations are missing "
        "or malformed.",
    ),
    "artic_resonance_attenuation_mean_db": (
        "Mean of the resonance_attenuation_db annotation over consonant tokens.",
        "NaN plus INVALID_TASK_ANNOTATION when resonance_attenuation_db annotations are "
        "missing or malformed.",
    ),
    "rhythm_percent_vocalic": (
        "100 times vowel-run duration divided by the sum of vowel-run and consonant-run "
        "durations over merged segment runs.",
        "NaN plus MISSING_ANNOTATION when complete positive vowel and consonant token "
        "times are unavailable.",
    ),
    "rhythm_varco_v": (
        "100 times the population SD divided by the mean of vowel-run durations.",
        "NaN plus MISSING_ANNOTATION when no vowel durations are available.",
    ),
    "rhythm_varco_c": (
        "100 times the population SD divided by the mean of consonant-run durations.",
        "NaN plus MISSING_ANNOTATION when no consonant durations are available.",
    ),
    "rhythm_rpvi_v": (
        "Mean absolute difference between consecutive vowel-run durations.",
        "NaN plus MISSING_ANNOTATION when fewer than two vowel durations are available.",
    ),
    "rhythm_rpvi_c": (
        "Mean absolute difference between consecutive consonant-run durations.",
        "NaN plus MISSING_ANNOTATION when fewer than two consonant durations are available.",
    ),
    "rhythm_npvi_v": (
        "Normalized pairwise variability index (nPVI) of vowel runs: 100 times the mean of abs(diff)/(pair mean) over consecutive vowel-run durations.",
        "NaN plus MISSING_ANNOTATION when fewer than two vowel durations are available.",
    ),
    "rhythm_npvi_c": (
        "100 times the mean of abs(diff)/(pair mean) over consecutive consonant-run durations.",
        "NaN plus MISSING_ANNOTATION when fewer than two consonant durations are available.",
    ),
    "task_ddk_rate_syllables_s": (
        "DDK syllable count divided by the span from the first to the last onset.",
        "NaN plus MISSING_ANNOTATION without task_spec task='ddk', complete token times, "
        "or at least two aligned DDK tokens; NaN plus INVALID_TASK_ANNOTATION when onsets "
        "are not strictly increasing.",
    ),
    "task_ddk_inter_onset_mean_s": (
        "Mean of consecutive DDK inter-onset intervals.",
        "NaN plus MISSING_ANNOTATION without task_spec task='ddk', complete token times, "
        "or at least two aligned DDK tokens; NaN plus INVALID_TASK_ANNOTATION when onsets "
        "are not strictly increasing.",
    ),
    "task_ddk_inter_onset_median_s": (
        "Median of consecutive DDK inter-onset intervals.",
        "NaN plus MISSING_ANNOTATION without task_spec task='ddk', complete token times, "
        "or at least two aligned DDK tokens; NaN plus INVALID_TASK_ANNOTATION when onsets "
        "are not strictly increasing.",
    ),
    "task_ddk_inter_onset_sd_s": (
        "Population SD of consecutive DDK inter-onset intervals.",
        "NaN plus MISSING_ANNOTATION without task_spec task='ddk', complete token times, "
        "or at least two aligned DDK tokens; NaN plus INVALID_TASK_ANNOTATION when onsets "
        "are not strictly increasing.",
    ),
    "task_ddk_inter_onset_cv": (
        "Population SD divided by the mean of DDK inter-onset intervals.",
        "NaN plus MISSING_ANNOTATION without task_spec task='ddk', complete token times, "
        "or at least two aligned DDK tokens; NaN plus INVALID_TASK_ANNOTATION when onsets "
        "are not strictly increasing.",
    ),
    "task_ddk_instability_s": (
        "Mean absolute difference between consecutive DDK inter-onset intervals.",
        "NaN plus MISSING_ANNOTATION when fewer than three DDK onsets are available.",
    ),
    "task_ddk_acceleration_syllables_s2": (
        "OLS slope of the reciprocal inter-onset interval over the interval midpoints.",
        "NaN plus MISSING_ANNOTATION when fewer than three DDK onsets are available.",
    ),
    "task_ddk_decay_ratio": (
        "First DDK inter-onset interval divided by the last.",
        "NaN plus MISSING_ANNOTATION without task_spec task='ddk', complete token times, "
        "or at least two aligned DDK tokens.",
    ),
    "task_ddk_voiced_interval_mean_s": (
        "Mean DDK token duration.",
        "NaN plus MISSING_ANNOTATION without task_spec task='ddk', complete token times, "
        "or at least two aligned DDK tokens.",
    ),
    "task_ddk_sequential_alternating_ratio": (
        "Share of adjacent DDK token pairs with different segment labels.",
        "NaN plus INVALID_TASK_ANNOTATION when DDK segment labels are missing or malformed; "
        "MISSING_ANNOTATION without task_spec task='ddk' or aligned DDK tokens.",
    ),
    "task_max_phonation_time_s": (
        "Maximum positive target utterance duration for the sustained_vowel task.",
        "NaN plus MISSING_ANNOTATION without task_spec task='sustained_vowel' or a positive utterance duration.",
    ),
    "voice_gaping_interval_rate_per_min": (
        "60 times the gaping/gaping_interval token count divided by the total segment duration.",
        "NaN plus MISSING_ANNOTATION without task_spec task='sustained_vowel' or complete "
        "segment types and token times.",
    ),
    "voice_subharmonic_interval_proportion": (
        "Duration of subharmonic/subharmonic_interval tokens divided by the total "
        "segment duration.",
        "NaN plus MISSING_ANNOTATION without task_spec task='sustained_vowel' or complete "
        "segment types and token times.",
    ),
    "voice_sustained_f0_sd_semitones": (
        "Population SD of 12*log2(f0) over voiced (non-gaping/non-silent) segments.",
        "NaN plus INVALID_TASK_ANNOTATION when f0_hz annotations are missing, malformed, "
        "or non-positive; MISSING_ANNOTATION without task_spec task='sustained_vowel' or "
        "voiced segments.",
    ),
    "voice_sustained_power_sd_db": (
        "Population SD of the power_db annotation over voiced (non-gaping/non-silent) segments.",
        "NaN plus INVALID_TASK_ANNOTATION when power_db annotations are missing or "
        "malformed; MISSING_ANNOTATION without task_spec task='sustained_vowel' or voiced "
        "segments.",
    ),
    "resp_breath_group_count": (
        "Number of contiguous breath_group label runs over the target tokens.",
        "NaN plus INVALID_TASK_ANNOTATION when breath_group annotations are missing or "
        "malformed; MISSING_ANNOTATION when complete positive token times are unavailable.",
    ),
    "resp_breath_group_mean_s": (
        "Mean duration of breath-group runs.",
        "NaN plus INVALID_TASK_ANNOTATION when breath_group annotations are missing or "
        "malformed; MISSING_ANNOTATION when complete positive token times are unavailable.",
    ),
    "resp_breath_group_sd_s": (
        "Population SD of breath-group run durations.",
        "NaN plus INVALID_TASK_ANNOTATION when breath_group annotations are missing or "
        "malformed; MISSING_ANNOTATION when complete positive token times are unavailable.",
    ),
    "resp_rate_per_min": (
        "60 times the breath-group count divided by the span from the first to the last "
        "token time.",
        "NaN plus INVALID_TASK_ANNOTATION when breath_group annotations are missing or "
        "malformed; MISSING_ANNOTATION when complete positive token times are unavailable.",
    ),
    "resp_pauses_per_breath": (
        "Pause/silence segment count divided by the breath-group count.",
        "NaN plus INVALID_TASK_ANNOTATION when timed breath groups are unavailable; "
        "MISSING_ANNOTATION when segment_type annotations are unavailable.",
    ),
    "resp_relative_loudness_db": (
        "Mean over tokens of speech_db minus respiration_db.",
        "NaN plus UNCALIBRATED_AUDIO when calibrated_amplitude is not true; NaN plus "
        "INVALID_TASK_ANNOTATION when respiration_db or speech_db annotations are missing "
        "or malformed.",
    ),
}

_MFCC_PATTERN = re.compile(r"spectral_mfcc_(\d+)_(mean|sd|skewness|kurtosis)")
_ERROR_PATTERN = re.compile(r"disfluency_(\w+)_error_(count|ratio)")

_EGEMAPS_STATS = {
    "amean": "arithmetic mean",
    "stddevNorm": "normalized standard deviation",
    "percentile20.0": "20th percentile",
    "percentile50.0": "50th percentile",
    "percentile80.0": "80th percentile",
    "pctlrange0-2": "percentile range 0-2",
    "meanRisingSlope": "mean rising slope",
    "stddevRisingSlope": "SD of the rising slope",
    "meanFallingSlope": "mean falling slope",
    "stddevFallingSlope": "SD of the falling slope",
}
_EGEMAPS_STANDALONE = {
    "loudnessPeaksPerSec": "loudness peaks per second",
    "VoicedSegmentsPerSec": "voiced segments per second",
    "MeanVoicedSegmentLengthSec": "mean voiced segment length",
    "StddevVoicedSegmentLengthSec": "SD of voiced segment length",
    "MeanUnvoicedSegmentLength": "mean unvoiced segment length",
    "StddevUnvoicedSegmentLength": "SD of unvoiced segment length",
    "equivalentSoundLevel_dBp": "equivalent continuous sound level",
}
_EGEMAPS_SIGNALS = {
    "F0semitoneFrom27.5Hz_sma3nz": "F0 in semitones from 27.5 Hz",
    "loudness_sma3": "auditory-model loudness",
    "spectralFlux_sma3": "spectral flux",
    "jitterLocal_sma3nz": "local jitter",
    "shimmerLocaldB_sma3nz": "local shimmer in dB",
    "HNRdBACF_sma3nz": "harmonics-to-noise ratio in dB",
    "logRelF0-H1-H2_sma3nz": "log relative F0-H1-H2 amplitude",
    "logRelF0-H1-A3_sma3nz": "log relative F0-H1-A3 amplitude",
    "F1frequency_sma3nz": "F1 frequency",
    "F1bandwidth_sma3nz": "F1 bandwidth",
    "F1amplitudeLogRelF0_sma3nz": "F1 amplitude relative to F0",
    "F2frequency_sma3nz": "F2 frequency",
    "F2bandwidth_sma3nz": "F2 bandwidth",
    "F2amplitudeLogRelF0_sma3nz": "F2 amplitude relative to F0",
    "F3frequency_sma3nz": "F3 frequency",
    "F3bandwidth_sma3nz": "F3 bandwidth",
    "F3amplitudeLogRelF0_sma3nz": "F3 amplitude relative to F0",
    "alphaRatioV_sma3nz": "alpha ratio (voiced)",
    "alphaRatioUV_sma3nz": "alpha ratio (unvoiced)",
    "hammarbergIndexV_sma3nz": "Hammarberg index (voiced)",
    "hammarbergIndexUV_sma3nz": "Hammarberg index (unvoiced)",
    "slopeV0-500_sma3nz": "spectral slope 0-500 Hz (voiced)",
    "slopeV500-1500_sma3nz": "spectral slope 500-1500 Hz (voiced)",
    "slopeUV0-500_sma3nz": "spectral slope 0-500 Hz (unvoiced)",
    "slopeUV500-1500_sma3nz": "spectral slope 500-1500 Hz (unvoiced)",
    "spectralFluxV_sma3nz": "spectral flux (voiced)",
    "spectralFluxUV_sma3nz": "spectral flux (unvoiced)",
    "mfcc1_sma3": "MFCC coefficient 1",
    "mfcc2_sma3": "MFCC coefficient 2",
    "mfcc3_sma3": "MFCC coefficient 3",
    "mfcc4_sma3": "MFCC coefficient 4",
    "mfcc1V_sma3nz": "MFCC coefficient 1 (voiced)",
    "mfcc2V_sma3nz": "MFCC coefficient 2 (voiced)",
    "mfcc3V_sma3nz": "MFCC coefficient 3 (voiced)",
    "mfcc4V_sma3nz": "MFCC coefficient 4 (voiced)",
}

_EGEMAPS_BY_KEY = None


def _egemaps_raw_columns() -> dict[str, str]:
    global _EGEMAPS_BY_KEY
    if _EGEMAPS_BY_KEY is None:
        from .features.standardized.definitions import EGEMAPS_KEYS, RAW_EGEMAPS_COLUMNS

        _EGEMAPS_BY_KEY = dict(zip(EGEMAPS_KEYS, RAW_EGEMAPS_COLUMNS, strict=True))
    return _EGEMAPS_BY_KEY


def _mfcc_docs(key: str) -> tuple[str, str]:
    match = _MFCC_PATTERN.fullmatch(key)
    assert match is not None, key
    coefficient, stat = match.groups()
    formula = (
        f"Distribution statistic {stat} of MFCC coefficient {coefficient} (of 13; "
        "26-triangular-mel filter bank, log power, orthonormal DCT-II) over target "
        "speech frames."
    )
    missing = (
        "NaN plus INSUFFICIENT_SPEECH_FRAMES when no target speech frames exist; "
        "skewness/kurtosis additionally require at least 3/4 finite frames."
    )
    return formula, missing


def _error_docs(key: str) -> tuple[str, str]:
    match = _ERROR_PATTERN.fullmatch(key)
    assert match is not None, key
    error_type, kind = match.groups()
    if kind == "count":
        formula = f"Count of target word tokens with reviewed error_type {error_type}."
        missing = (
            "0 for a present target sample with no such event; NaN plus "
            "MISSING_ANNOTATION when the error_type layer is absent or there is no "
            "target word sample."
        )
    else:
        formula = (
            f"Count of target word tokens with reviewed error_type {error_type} divided "
            "by lex_token_count."
        )
        missing = (
            "NaN plus INSUFFICIENT_TOKENS when the token count is zero; MISSING_ANNOTATION "
            "when the error_type layer is absent or there is no target word sample."
        )
    return formula, missing


def _egemaps_docs(key: str) -> tuple[str, str]:
    raw = _egemaps_raw_columns()[key]
    if raw in _EGEMAPS_STANDALONE:
        formula = f"openSMILE eGeMAPSv02 Functionals {_EGEMAPS_STANDALONE[raw]} (raw column {raw})."
        return formula, STANDARDIZED_MISSING
    for stem in sorted(_EGEMAPS_SIGNALS, key=len, reverse=True):
        if raw.startswith(stem + "_"):
            stat = _EGEMAPS_STATS.get(raw[len(stem) + 1 :])
            if stat is None:
                break
            formula = (
                f"openSMILE eGeMAPSv02 Functionals {stat} of {_EGEMAPS_SIGNALS[stem]} "
                f"(raw column {raw})."
            )
            return formula, STANDARDIZED_MISSING
    raise KeyError(f"no documented formula/missing-data pair for eGeMAPS key {key!r}")


_EXPLICIT_DOCS = {}
for _docs in (
    TIMING_COMPANION_DOCS,
    VOICE_COMPANION_DOCS,
    ADVANCED_SPECTRAL_DOCS,
    STRUCTURAL_PSYCHOLINGUISTIC_DOCS,
    DISCOURSE_CLINICAL_DOCS,
    TASK_DOCS,
    MOTOR_DOCS,
):
    _EXPLICIT_DOCS.update(_docs)


def _explicit_mfcc_keys():
    from .features.acoustic.definitions import SPECTRUM_KEYS

    return [key for key in SPECTRUM_KEYS if key.startswith("spectral_mfcc_")]


def _explicit_error_keys():
    from .features.linguistic.definitions import ERROR_KEYS

    return list(ERROR_KEYS)


ALL_DOCUMENTATION = {
    **LEGACY_FEATURE_DOCS,
    **_EXPLICIT_DOCS,
    **{key: _mfcc_docs(key) for key in _explicit_mfcc_keys()},
    **{key: _error_docs(key) for key in _explicit_error_keys()},
    **{key: _egemaps_docs(key) for key in EGEMAPS_KEYS},
}


def documentation_for(key: str) -> tuple[str, str]:
    """Return the exact (formula, missing_data) pair for one live feature key."""
    try:
        return ALL_DOCUMENTATION[key]
    except KeyError:
        raise KeyError(
            f"no documented formula/missing-data pair for live feature key {key!r}"
        ) from None


def validate_documentation(live_keys) -> None:
    """Check every live key is documented and every pair is unique."""
    pairs = {}
    for key in live_keys:
        pair = documentation_for(key)
        if pair in pairs:
            raise ValueError(
                f"duplicate formula/missing-data pair shared by {pairs[pair]!r} and {key!r}"
            )
        pairs[pair] = key


__all__ = [
    "ALL_DOCUMENTATION",
    "STANDARDIZED_MISSING",
    "documentation_for",
    "validate_documentation",
]
