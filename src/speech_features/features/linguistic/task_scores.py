"""Structured task scores over current ``SpeechDocument`` types."""

from __future__ import annotations

import math

from ...result import FeatureIssue
from ...schema import InvalidTaskSpecError, validate_task_spec
from . import _normalise, _resolve_target_speaker
from .definitions import TASK_KEYS

SIMILARITY_THRESHOLD = 0.85


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


def unavailable_structured_task_features(recording_id="", speaker_id=""):
    values = {key: math.nan for key in TASK_KEYS}
    issues = tuple(
        _issue(
            recording_id,
            speaker_id,
            "MISSING_ANNOTATION",
            "task spec is required for structured task features",
            key,
        )
        for key in TASK_KEYS
    )
    return values, issues


def _levenshtein(left, right):
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for index, left_character in enumerate(left, 1):
        current = [index]
        for right_index, right_character in enumerate(right, 1):
            cost = left_character != right_character
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def _similarity(left, right):
    denominator = max(len(left), len(right))
    return 1.0 - _levenshtein(left, right) / denominator if denominator else 0.0


def _match_key(text, aliases_by_key):
    normalised = _normalise(text)
    for key, aliases in aliases_by_key.items():
        if any(_normalise(alias) == normalised for alias in aliases):
            return key
    for key, aliases in aliases_by_key.items():
        if any(
            _similarity(normalised, _normalise(alias)) >= SIMILARITY_THRESHOLD for alias in aliases
        ):
            return key
    return None


def _lcs(left, right):
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for index, right_item in enumerate(right, 1):
            if left_item == right_item:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _responses(utterances):
    responses = []
    for utterance in utterances:
        words = [_normalise(token.text) for token in utterance.tokens if token.kind == "word"]
        if words:
            responses.append((" ".join(words), utterance.start_s, utterance.end_s))
    return responses


def _set(values, issues, key, value, recording_id, speaker_id):
    values[key] = float(value)
    if math.isnan(values[key]):
        _unavailable(
            values,
            issues,
            (key,),
            "INSUFFICIENT_TOKENS",
            "zero or absent structured-task denominator",
            recording_id,
            speaker_id,
        )


def _score_picture(values, issues, words, spec, recording_id, speaker_id):
    concepts = spec["concept_aliases"]
    entities = spec.get("entity_groups", {})
    actions = spec.get("action_groups", {})
    matched = [_match_key(word, concepts) for word in words]
    present = [key for key in matched if key is not None]
    distinct = set(present)
    entity_hits = {key for word in words if (key := _match_key(word, entities)) is not None}
    action_hits = {key for word in words if (key := _match_key(word, actions)) is not None}
    results = {
        "task_picture_concept_coverage": len(distinct) / len(concepts) if concepts else math.nan,
        "task_picture_concept_density": len(distinct) / len(words) if words else math.nan,
        "task_picture_repeat_ratio": (
            (len(present) - len(distinct)) / len(present) if present else math.nan
        ),
        "task_picture_entity_coverage": (
            len(entity_hits) / len(entities) if entities else math.nan
        ),
        "task_picture_action_coverage": (len(action_hits) / len(actions) if actions else math.nan),
    }
    for key, value in results.items():
        _set(values, issues, key, value, recording_id, speaker_id)


def _score_recall(values, issues, words, spec, recording_id, speaker_id):
    ideas = spec["idea_aliases"]
    present = [key for word in words if (key := _match_key(word, ideas)) is not None]
    distinct = set(present)
    ordered = list(dict.fromkeys(present))
    reference = list(ideas)
    results = {
        "task_recall_idea_coverage": len(distinct) / len(ideas) if ideas else math.nan,
        "task_recall_idea_density": len(distinct) / len(words) if words else math.nan,
        "task_recall_repeat_ratio": (
            (len(present) - len(distinct)) / len(present) if present else math.nan
        ),
        "task_recall_order_score": (
            _lcs(ordered, reference) / len(reference) if ordered and reference else math.nan
        ),
    }
    for key, value in results.items():
        _set(values, issues, key, value, recording_id, speaker_id)


def _score_fluency(values, issues, responses, spec, recording_id, speaker_id):
    task = spec["task"]
    response_texts = [response[0] for response in responses]
    matched_keys = []
    valid = []
    categories = []
    if task == "phonemic_fluency":
        initials = [_normalise(initial) for initial in spec["initials"]]
        exclusions = {_normalise(exclusion) for exclusion in spec.get("exclusions", [])}
        for response in response_texts:
            is_valid = (
                any(response.split()[0].startswith(initial) for initial in initials)
                and response not in exclusions
            )
            valid.append(is_valid)
            matched_keys.append(response if is_valid else None)
    else:
        items = spec["item_aliases"]
        subcategories = spec.get("subcategories", {})
        for response in response_texts:
            key = _match_key(response, items)
            matched_keys.append(key)
            valid.append(key is not None)
            if key is not None and key in subcategories:
                categories.append(subcategories[key])

    valid_keys = [key for key in matched_keys if key is not None]
    valid_count = len(valid_keys)
    valid_unique = len(set(valid_keys))
    if responses:
        start_s = min(start for _, start, _ in responses)
        end_s = max(end for _, _, end in responses)
        midpoint = (start_s + end_s) / 2.0
        first_half = sum(
            is_valid and start < midpoint for is_valid, (_, start, _) in zip(valid, responses)
        )
        second_half = valid_count - first_half
        duration_min = (end_s - start_s) / 60.0
    else:
        first_half = second_half = 0
        duration_min = 0.0

    common = {
        "task_fluency_response_count": len(responses),
        "task_fluency_valid_count": valid_count,
        "task_fluency_valid_unique": valid_unique,
        "task_fluency_repeats": valid_count - valid_unique,
        "task_fluency_intrusions": len(responses) - valid_count,
        "task_fluency_first_half_valid": first_half,
        "task_fluency_second_half_valid": second_half,
        "task_fluency_production_change": second_half - first_half,
        "task_fluency_rate": valid_unique / duration_min if duration_min > 0 else math.nan,
    }
    for key, value in common.items():
        _set(values, issues, key, value, recording_id, speaker_id)

    if task == "semantic_fluency":
        clusters = sum(
            index == 0 or category != categories[index - 1]
            for index, category in enumerate(categories)
        )
        switches = sum(left != right for left, right in zip(categories, categories[1:]))
        cluster_values = {
            "task_fluency_clusters": clusters,
            "task_fluency_cluster_size_mean": (
                len(categories) / clusters if clusters else math.nan
            ),
            "task_fluency_switches": switches,
        }
        for key, value in cluster_values.items():
            _set(values, issues, key, value, recording_id, speaker_id)


def extract_structured_task_features(
    document,
    task_spec,
    *,
    target_speaker=None,
    recording_id: str = "",
    _validated_task_spec: bool = False,
) -> tuple[dict[str, float], tuple[FeatureIssue, ...]]:
    """Return every registered task key using one validated version-1 task spec."""
    if not _validated_task_spec:
        validate_task_spec(task_spec)
    task = task_spec.get("task")
    scorers = {
        "picture_desc_1": "picture",
        "picture_desc_2": "picture",
        "immediate_recall": "recall",
        "delayed_recall": "recall",
        "phonemic_fluency": "fluency",
        "semantic_fluency": "fluency",
    }
    if task not in scorers:
        raise InvalidTaskSpecError("task spec must declare a supported structured task")

    values = {key: math.nan for key in TASK_KEYS}
    issues = []
    speaker_id = _resolve_target_speaker(document, target_speaker)
    utterances = (
        [utterance for utterance in document.utterances if utterance.speaker_id == speaker_id]
        if document is not None and speaker_id
        else []
    )
    words = [
        _normalise(token.text)
        for utterance in utterances
        for token in utterance.tokens
        if token.kind == "word"
    ]
    if not words:
        _unavailable(
            values,
            issues,
            TASK_KEYS,
            "MISSING_ANNOTATION",
            "no explicit target-speaker word sample; task features unavailable",
            recording_id,
            speaker_id,
        )
        return values, tuple(issues)

    scorer = scorers[task]
    if scorer == "picture":
        _score_picture(values, issues, words, task_spec, recording_id, speaker_id)
    elif scorer == "recall":
        _score_recall(values, issues, words, task_spec, recording_id, speaker_id)
    else:
        _score_fluency(
            values,
            issues,
            _responses(utterances),
            task_spec,
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
                f"{key} is not defined for task {task!r}",
                recording_id,
                speaker_id,
            )
    return values, tuple(issues)


__all__ = ["extract_structured_task_features"]
