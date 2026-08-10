"""Reviewed clinical linguistic measures over explicit document annotations."""

from __future__ import annotations

import math
from collections import Counter

from ...result import FeatureIssue
from ...schema import validate_task_spec
from . import _mean, _normalise, _resolve_target_speaker
from .definitions import (
    CLINICAL_LINGUISTIC_KEYS,
    ERROR_TYPES,
)

_DEPENDENT_CLAUSE_TYPES = frozenset({"dependent", "embedded", "subordinate"})
_CLAUSE_TYPES = frozenset(
    {"main", "independent", "coordinate", "dependent", "embedded", "subordinate"}
)
_SENTENCE_STATUSES = frozenset({"well_formed", "incomplete", "reduced"})
_NONE_LABELS = frozenset({"", "none", "null", "na", "n/a"})
_PSYCHOLINGUISTIC_LAYERS = {
    "lex_frequency_mean": "frequency",
    "lex_log_frequency_mean": "log_frequency",
    "lex_familiarity_mean": "familiarity",
    "lex_age_of_acquisition_mean": "age_of_acquisition",
    "lex_imageability_mean": "imageability",
    "lex_concreteness_mean": "concreteness",
}


def _issue(recording_id, speaker_id, code, message, feature):
    return FeatureIssue(
        recording_id=recording_id,
        speaker_id=speaker_id,
        code=code,
        severity="warning",
        message=message,
        feature=feature,
    )


def _unavailable(values, issues, keys, code, message, recording_id, speaker_id):
    issued = {issue.feature for issue in issues}
    for key in keys:
        values[key] = math.nan
        if key not in issued:
            issues.append(_issue(recording_id, speaker_id, code, message, key))
            issued.add(key)


def _layer(document, name):
    return next((layer for layer in document.annotations if layer.layer == name), None)


def _complete_strings(document, layer_name, word_tokens, *, allowed=None, rejected=()):
    layer = _layer(document, layer_name)
    if layer is None:
        return None
    values = []
    for token in word_tokens:
        value = layer.values.get(token.id)
        if not isinstance(value, str) or not value:
            return None
        value = _normalise(value)
        if value in rejected or (allowed is not None and value not in allowed):
            return None
        values.append(value)
    return values


def _complete_numbers(document, layer_name, word_tokens, *, binary=False):
    layer = _layer(document, layer_name)
    if layer is None:
        return None
    values = []
    for token in word_tokens:
        value = layer.values.get(token.id)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value = float(value)
        if not math.isfinite(value) or (binary and value not in (0.0, 1.0)):
            return None
        values.append(value)
    return values


def _sparse_strings(document, layer_name, word_tokens):
    layer = _layer(document, layer_name)
    if layer is None:
        return None
    target_ids = {token.id for token in word_tokens}
    values = []
    for token_id, value in layer.values.items():
        if token_id in target_ids and isinstance(value, str) and value:
            normalised = _normalise(value)
            if normalised not in _NONE_LABELS:
                values.append(normalised)
    return values


def _sparse_binary(document, layer_name, word_tokens):
    layer = _layer(document, layer_name)
    if layer is None:
        return None
    values = []
    for token in word_tokens:
        if token.id not in layer.values:
            continue
        value = layer.values[token.id]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in (0, 1):
            return None
        values.append(float(value))
    return values or None


def _group_labels(group_ids, labels):
    grouped = {}
    for group_id, label in zip(group_ids, labels):
        if group_id in grouped and grouped[group_id] != label:
            return None
        grouped[group_id] = label
    return grouped


def _task_reference_keys(task_spec):
    if task_spec is None:
        return None
    task = task_spec.get("task")
    field = {
        "picture_desc_1": "concept_aliases",
        "picture_desc_2": "concept_aliases",
        "immediate_recall": "idea_aliases",
        "delayed_recall": "idea_aliases",
        "semantic_fluency": "item_aliases",
    }.get(task)
    if field is None:
        return None
    return {_normalise(str(key)) for key in task_spec[field]}


def extract_clinical_linguistic_features(
    document,
    *,
    target_speaker=None,
    recording_id: str = "",
    task_spec=None,
    _validated_task_spec: bool = False,
) -> tuple[dict[str, float], tuple[FeatureIssue, ...]]:
    """Return all Task 5 structural, psycholinguistic, error, and discourse keys."""
    if task_spec is not None and not _validated_task_spec:
        validate_task_spec(task_spec)
    values = {key: math.nan for key in CLINICAL_LINGUISTIC_KEYS}
    issues = []
    speaker_id = _resolve_target_speaker(document, target_speaker)
    utterances = (
        [utterance for utterance in document.utterances if utterance.speaker_id == speaker_id]
        if document is not None and speaker_id
        else []
    )
    word_tokens = [
        token for utterance in utterances for token in utterance.tokens if token.kind == "word"
    ]
    if not word_tokens:
        _unavailable(
            values,
            issues,
            CLINICAL_LINGUISTIC_KEYS,
            "MISSING_ANNOTATION",
            "no explicit target-speaker word sample; clinical linguistic features unavailable",
            recording_id,
            speaker_id,
        )
        return values, tuple(issues)

    word_count = float(len(word_tokens))

    sentence_ids = _complete_strings(document, "sentence_id", word_tokens, rejected=_NONE_LABELS)
    sentence_keys = (
        "morph_sentence_count",
        "morph_words_per_sentence",
        "morph_clauses_per_sentence",
        "morph_well_formed_sentence_ratio",
        "morph_incomplete_sentence_ratio",
        "morph_reduced_sentence_ratio",
    )
    if sentence_ids is None:
        _unavailable(
            values,
            issues,
            sentence_keys,
            "MISSING_ANNOTATION",
            "sentence_id annotation is absent or incomplete for target words",
            recording_id,
            speaker_id,
        )
        sentence_count = None
    else:
        sentence_count = len(set(sentence_ids))
        values["morph_sentence_count"] = float(sentence_count)
        values["morph_words_per_sentence"] = word_count / sentence_count

    t_unit_ids = _complete_strings(document, "t_unit_id", word_tokens, rejected=_NONE_LABELS)
    if t_unit_ids is None:
        _unavailable(
            values,
            issues,
            ("morph_t_unit_count", "morph_words_per_t_unit"),
            "MISSING_ANNOTATION",
            "t_unit_id annotation is absent or incomplete for target words",
            recording_id,
            speaker_id,
        )
    else:
        count = len(set(t_unit_ids))
        values["morph_t_unit_count"] = float(count)
        values["morph_words_per_t_unit"] = word_count / count

    clause_ids = _complete_strings(document, "clause_id", word_tokens, rejected=_NONE_LABELS)
    qualified_clause_ids = (
        list(zip(sentence_ids, clause_ids))
        if sentence_ids is not None and clause_ids is not None
        else None
    )
    clause_count = len(set(qualified_clause_ids)) if qualified_clause_ids is not None else None
    if clause_count is None:
        _unavailable(
            values,
            issues,
            ("morph_words_per_clause", "morph_clauses_per_sentence"),
            "MISSING_ANNOTATION",
            "clause_id annotation is absent or incomplete for target words",
            recording_id,
            speaker_id,
        )
    else:
        values["morph_words_per_clause"] = word_count / clause_count
        if sentence_count is not None:
            values["morph_clauses_per_sentence"] = clause_count / sentence_count

    clause_types = _complete_strings(document, "clause_type", word_tokens, allowed=_CLAUSE_TYPES)
    grouped_clause_types = (
        _group_labels(qualified_clause_ids, clause_types)
        if qualified_clause_ids is not None and clause_types is not None
        else None
    )
    if grouped_clause_types is None:
        _unavailable(
            values,
            issues,
            ("morph_embedding_count", "morph_dependent_clause_ratio"),
            "MISSING_ANNOTATION",
            "clause_type annotation is absent, incomplete, or inconsistent",
            recording_id,
            speaker_id,
        )
    else:
        dependent = sum(label in _DEPENDENT_CLAUSE_TYPES for label in grouped_clause_types.values())
        values["morph_embedding_count"] = float(
            sum(label == "embedded" for label in grouped_clause_types.values())
        )
        values["morph_dependent_clause_ratio"] = dependent / len(grouped_clause_types)

    phrase_types = _complete_strings(
        document, "phrase_type", word_tokens, rejected=_NONE_LABELS - {"none"}
    )
    phrase_keys = (
        "morph_coordinate_phrase_count",
        "morph_complex_nominal_count",
        "morph_verb_phrase_count",
    )
    if phrase_types is None:
        _unavailable(
            values,
            issues,
            phrase_keys,
            "MISSING_ANNOTATION",
            "phrase_type annotation is absent or incomplete for target words",
            recording_id,
            speaker_id,
        )
    else:
        phrase_by_id = {
            token.id: phrase_type for token, phrase_type in zip(word_tokens, phrase_types)
        }
        counts = Counter()
        for utterance in utterances:
            previous = None
            for token in utterance.tokens:
                if token.kind != "word":
                    continue
                phrase_type = phrase_by_id[token.id]
                if phrase_type != "none" and phrase_type != previous:
                    counts[phrase_type] += 1
                previous = phrase_type
        values[phrase_keys[0]] = float(counts["coordinate"])
        values[phrase_keys[1]] = float(counts["complex_nominal"])
        values[phrase_keys[2]] = float(counts["verb_phrase"])

    statuses = _complete_strings(
        document, "sentence_status", word_tokens, allowed=_SENTENCE_STATUSES
    )
    grouped_statuses = (
        _group_labels(sentence_ids, statuses)
        if sentence_ids is not None and statuses is not None
        else None
    )
    status_keys = (
        "morph_well_formed_sentence_ratio",
        "morph_incomplete_sentence_ratio",
        "morph_reduced_sentence_ratio",
    )
    if grouped_statuses is None:
        _unavailable(
            values,
            issues,
            status_keys,
            "MISSING_ANNOTATION",
            "sentence_status annotation is absent, incomplete, or inconsistent",
            recording_id,
            speaker_id,
        )
    else:
        counts = Counter(grouped_statuses.values())
        denominator = len(grouped_statuses)
        values[status_keys[0]] = counts["well_formed"] / denominator
        values[status_keys[1]] = counts["incomplete"] / denominator
        values[status_keys[2]] = counts["reduced"] / denominator

    depths = _complete_numbers(document, "yngve_depth", word_tokens)
    if depths is None:
        _unavailable(
            values,
            issues,
            ("morph_yngve_depth_mean", "morph_yngve_depth_max"),
            "MISSING_ANNOTATION",
            "yngve_depth annotation is absent or incomplete for target words",
            recording_id,
            speaker_id,
        )
    else:
        values["morph_yngve_depth_mean"] = _mean(depths)
        values["morph_yngve_depth_max"] = max(depths)

    for key, layer_name in _PSYCHOLINGUISTIC_LAYERS.items():
        observations = _complete_numbers(document, layer_name, word_tokens)
        if observations is None:
            _unavailable(
                values,
                issues,
                (key,),
                "MISSING_ANNOTATION",
                f"{layer_name} annotation is absent or incomplete for target words",
                recording_id,
                speaker_id,
            )
        else:
            values[key] = _mean(observations)

    error_types = _sparse_strings(document, "error_type", word_tokens)
    if error_types is None:
        error_keys = tuple(
            key
            for error_type in ERROR_TYPES
            for key in (
                f"disfluency_{error_type}_error_count",
                f"disfluency_{error_type}_error_ratio",
            )
        )
        _unavailable(
            values,
            issues,
            error_keys,
            "MISSING_ANNOTATION",
            "error_type annotation is absent",
            recording_id,
            speaker_id,
        )
    else:
        counts = Counter(error_types)
        for error_type in ERROR_TYPES:
            count = float(counts[error_type])
            values[f"disfluency_{error_type}_error_count"] = count
            values[f"disfluency_{error_type}_error_ratio"] = count / word_count

    cohesion_types = _sparse_strings(document, "cohesion_type", word_tokens)
    cohesion_keys = (
        "discourse_referential_cohesion_ratio",
        "discourse_temporal_cohesion_ratio",
        "discourse_causal_cohesion_ratio",
    )
    if cohesion_types:
        counts = Counter(cohesion_types)
        denominator = len(cohesion_types)
        for cohesion_type, key in zip(("referential", "temporal", "causal"), cohesion_keys):
            values[key] = counts[cohesion_type] / denominator
    else:
        _unavailable(
            values,
            issues,
            cohesion_keys,
            "MISSING_ANNOTATION",
            "cohesion_type annotation has no target observations",
            recording_id,
            speaker_id,
        )

    pronoun_correct = _sparse_binary(document, "pronoun_reference_correct", word_tokens)
    if pronoun_correct is None:
        _unavailable(
            values,
            issues,
            ("discourse_correct_pronoun_ratio",),
            "MISSING_ANNOTATION",
            "pronoun_reference_correct annotation has no valid target observations",
            recording_id,
            speaker_id,
        )
    else:
        values["discourse_correct_pronoun_ratio"] = _mean(pronoun_correct)

    word_sets = [
        {_normalise(token.text) for token in utterance.tokens if token.kind == "word"}
        for utterance in utterances
    ]
    if len(word_sets) < 2 or any(
        not (left | right) for left, right in zip(word_sets, word_sets[1:])
    ):
        _unavailable(
            values,
            issues,
            ("discourse_local_lexical_coherence",),
            "INSUFFICIENT_TOKENS",
            "fewer than two adjacent target utterance word sets with a defined union",
            recording_id,
            speaker_id,
        )
    else:
        values["discourse_local_lexical_coherence"] = _mean(
            [len(left & right) / len(left | right) for left, right in zip(word_sets, word_sets[1:])]
        )

    topic_relevant = _complete_numbers(document, "topic_relevant", word_tokens, binary=True)
    if topic_relevant is None:
        _unavailable(
            values,
            issues,
            ("discourse_global_coherence_ratio", "discourse_topic_maintenance_ratio"),
            "MISSING_ANNOTATION",
            "topic_relevant annotation is absent or incomplete for target words",
            recording_id,
            speaker_id,
        )
    else:
        values["discourse_global_coherence_ratio"] = _mean(topic_relevant)
        relevant_by_id = {token.id: value for token, value in zip(word_tokens, topic_relevant)}
        values["discourse_topic_maintenance_ratio"] = sum(
            any(relevant_by_id[token.id] for token in utterance.tokens if token.kind == "word")
            for utterance in utterances
        ) / len(utterances)

    discourse_roles = _sparse_strings(document, "discourse_role", word_tokens)
    role_keys = (
        "discourse_marker_ratio",
        "discourse_relevant_detail_ratio",
        "discourse_irrelevant_detail_ratio",
        "discourse_microproposition_count",
        "discourse_macroproposition_count",
        "semantic_proposition_density",
    )
    if discourse_roles is None:
        _unavailable(
            values,
            issues,
            role_keys,
            "MISSING_ANNOTATION",
            "discourse_role annotation is absent",
            recording_id,
            speaker_id,
        )
    else:
        counts = Counter(discourse_roles)
        values["discourse_marker_ratio"] = counts["marker"] / word_count
        details = counts["relevant_detail"] + counts["irrelevant_detail"]
        if details:
            values["discourse_relevant_detail_ratio"] = counts["relevant_detail"] / details
            values["discourse_irrelevant_detail_ratio"] = counts["irrelevant_detail"] / details
        else:
            _unavailable(
                values,
                issues,
                ("discourse_relevant_detail_ratio", "discourse_irrelevant_detail_ratio"),
                "INSUFFICIENT_TOKENS",
                "no reviewed detail observations",
                recording_id,
                speaker_id,
            )
        micro = float(counts["microproposition"])
        macro = float(counts["macroproposition"])
        values["discourse_microproposition_count"] = micro
        values["discourse_macroproposition_count"] = macro
        values["semantic_proposition_density"] = (micro + macro) / word_count

    information_units = _sparse_strings(document, "information_unit", word_tokens)
    information_keys = (
        "discourse_information_unit_count",
        "discourse_content_accuracy_ratio",
        "discourse_information_efficiency_per_min",
        "semantic_idea_density",
    )
    if information_units is None:
        _unavailable(
            values,
            issues,
            information_keys,
            "MISSING_ANNOTATION",
            "information_unit annotation is absent",
            recording_id,
            speaker_id,
        )
    else:
        distinct_units = set(information_units)
        count = float(len(distinct_units))
        values["discourse_information_unit_count"] = count
        values["semantic_idea_density"] = count / word_count
        reference_keys = _task_reference_keys(task_spec)
        if reference_keys is None or not distinct_units:
            _unavailable(
                values,
                issues,
                ("discourse_content_accuracy_ratio",),
                "MISSING_ANNOTATION" if reference_keys is None else "INSUFFICIENT_TOKENS",
                "task reference or information units unavailable for content accuracy",
                recording_id,
                speaker_id,
            )
        else:
            values["discourse_content_accuracy_ratio"] = len(distinct_units & reference_keys) / len(
                distinct_units
            )
        duration_s = max(utterance.end_s for utterance in utterances) - min(
            utterance.start_s for utterance in utterances
        )
        if duration_s > 0:
            values["discourse_information_efficiency_per_min"] = count / (duration_s / 60.0)
        else:
            _unavailable(
                values,
                issues,
                ("discourse_information_efficiency_per_min",),
                "INSUFFICIENT_TOKENS",
                "zero target timestamp span",
                recording_id,
                speaker_id,
            )

    for key, value in values.items():
        if math.isnan(value):
            _unavailable(
                values,
                issues,
                (key,),
                "MISSING_ANNOTATION",
                "required reviewed annotation is unavailable",
                recording_id,
                speaker_id,
            )
    return values, tuple(issues)


__all__ = ["extract_clinical_linguistic_features"]
