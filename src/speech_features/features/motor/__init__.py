"""Annotation-driven articulation, rhythm, DDK, and respiratory features."""

from __future__ import annotations

import math

import numpy as np

from ...result import (
    FeatureIssue,
    InvalidConfigError,
    TargetSpeakerRequiredError,
)
from ...schema import nfc
from .definitions import (
    ALL_KEYS,
    ARTICULATION_KEYS,
    DDK_KEYS,
    RESPIRATORY_KEYS,
    RHYTHM_KEYS,
    SUSTAINED_VOWEL_KEYS,
    register_motor_features,
)
from .intervals import finite_mean_sd, layer_values, polygon_area, target_tokens, token_duration

register_motor_features()


def _normalise(value) -> str | None:
    return nfc(value).casefold().strip() if isinstance(value, str) else None


def _resolve_target_speaker(document, target_speaker) -> str:
    speaker_ids = {speaker.id for speaker in document.speakers} if document is not None else set()
    if target_speaker is not None:
        if document is None or target_speaker not in speaker_ids:
            raise InvalidConfigError(
                f"target speaker {target_speaker!r} is not a speaker of the speech document"
            )
        return target_speaker
    if len(speaker_ids) > 1:
        raise TargetSpeakerRequiredError(
            "speech document has multiple speakers; pass target_speaker to isolate one"
        )
    return next(iter(speaker_ids), "")


def _issue(recording_id: str, speaker_id: str, feature: str, code: str, message: str):
    return FeatureIssue(
        recording_id=recording_id,
        speaker_id=speaker_id,
        feature=feature,
        code=code,
        severity="warning",
        message=message,
    )


def _flag(
    values: dict[str, float],
    issues: list[FeatureIssue],
    keys,
    code: str,
    message: str,
    recording_id: str,
    speaker_id: str,
) -> None:
    flagged = {issue.feature for issue in issues}
    for key in keys:
        values[key] = math.nan
        if key not in flagged:
            issues.append(_issue(recording_id, speaker_id, key, code, message))
            flagged.add(key)


def _numeric_layer(document, name: str, token_ids: tuple[str, ...]):
    raw = layer_values(document, name, set(token_ids))
    if raw is None:
        return "missing", None
    numbers = []
    for token_id in token_ids:
        value = raw[token_id]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "invalid", None
        number = float(value)
        if not math.isfinite(number):
            return "invalid", None
        numbers.append(number)
    return "ok", numbers


def _labels(document, name: str, token_ids: tuple[str, ...]):
    raw = layer_values(document, name, set(token_ids))
    if raw is None:
        return "missing", None
    labels = [_normalise(raw[token_id]) for token_id in token_ids]
    if any(label is None or not label for label in labels):
        return "invalid", None
    return "ok", labels


def _annotation_code(*statuses: str) -> str:
    return "INVALID_TASK_ANNOTATION" if "invalid" in statuses else "MISSING_ANNOTATION"


def _set_mean_sd(
    values,
    issues,
    document,
    layer,
    token_ids,
    mean_key,
    sd_key,
    recording_id,
    speaker_id,
) -> None:
    status, numbers = _numeric_layer(document, layer, token_ids)
    if status != "ok" or not numbers:
        _flag(
            values,
            issues,
            (mean_key, sd_key),
            _annotation_code(status),
            f"complete finite {layer} annotations are required",
            recording_id,
            speaker_id,
        )
        return
    values[mean_key], values[sd_key] = finite_mean_sd(numbers)


def _articulation_features(document, tokens, types, values, issues, recording_id, speaker_id):
    if types is None:
        _flag(
            values,
            issues,
            ARTICULATION_KEYS,
            "MISSING_ANNOTATION",
            "complete segment_type annotations are required",
            recording_id,
            speaker_id,
        )
        return

    vowel_tokens = tuple(token for token, kind in zip(tokens, types) if kind == "vowel")
    vowel_ids = tuple(token.id for token in vowel_tokens)
    label_status, labels = _labels(document, "segment_label", vowel_ids)
    f1_status, f1 = _numeric_layer(document, "f1_hz", vowel_ids)
    f2_status, f2 = _numeric_layer(document, "f2_hz", vowel_ids)
    vowel_code = _annotation_code(label_status, f1_status, f2_status)
    vowel_message = "complete finite i/a/u formant annotations are required"
    space_keys = (
        "artic_vowel_space_area_hz2",
        "artic_vowel_articulation_index",
        "artic_formant_centralization_ratio",
    )
    if label_status != "ok" or f1_status != "ok" or f2_status != "ok" or not vowel_ids:
        _flag(values, issues, space_keys, vowel_code, vowel_message, recording_id, speaker_id)
    else:
        medians = {}
        for vowel in ("i", "a", "u"):
            indices = [index for index, label in enumerate(labels) if label == vowel]
            if indices:
                medians[vowel] = (
                    float(np.median([f1[index] for index in indices])),
                    float(np.median([f2[index] for index in indices])),
                )
        if set(medians) != {"i", "a", "u"}:
            _flag(
                values,
                issues,
                space_keys,
                "MISSING_ANNOTATION",
                vowel_message,
                recording_id,
                speaker_id,
            )
        else:
            f1_i, f2_i = medians["i"]
            f1_a, f2_a = medians["a"]
            f1_u, f2_u = medians["u"]
            numerator = f2_i + f1_a
            denominator = f1_i + f1_u + f2_u + f2_a
            if numerator <= 0.0 or denominator <= 0.0:
                _flag(
                    values,
                    issues,
                    space_keys,
                    "INVALID_TASK_ANNOTATION",
                    "vowel formants must make VAI and FCR denominators positive",
                    recording_id,
                    speaker_id,
                )
            else:
                values["artic_vowel_space_area_hz2"] = polygon_area(
                    (medians["i"], medians["a"], medians["u"])
                )
                values["artic_vowel_articulation_index"] = numerator / denominator
                values["artic_formant_centralization_ratio"] = denominator / numerator

    dispersion_keys = ("artic_vowel_dispersion_mean_hz", "artic_vowel_dispersion_sd_hz")
    if f1_status != "ok" or f2_status != "ok" or not vowel_ids:
        _flag(
            values,
            issues,
            dispersion_keys,
            _annotation_code(f1_status, f2_status),
            "complete finite vowel F1/F2 annotations are required",
            recording_id,
            speaker_id,
        )
    else:
        centroid = (float(np.mean(f1)), float(np.mean(f2)))
        distances = [math.hypot(x - centroid[0], y - centroid[1]) for x, y in zip(f1, f2)]
        (
            values["artic_vowel_dispersion_mean_hz"],
            values["artic_vowel_dispersion_sd_hz"],
        ) = finite_mean_sd(distances)

    for formants, status, key in (
        (f1, f1_status, "artic_f1_within_vowel_sd_hz"),
        (f2, f2_status, "artic_f2_within_vowel_sd_hz"),
    ):
        if label_status != "ok" or status != "ok" or not vowel_ids:
            _flag(
                values,
                issues,
                (key,),
                _annotation_code(label_status, status),
                "complete vowel labels and formants are required",
                recording_id,
                speaker_id,
            )
        else:
            grouped_sd = [
                float(np.std([value for value, item in zip(formants, labels) if item == label]))
                for label in sorted(set(labels))
            ]
            values[key] = float(np.mean(grouped_sd))

    transition_key = "artic_formant_transition_slope_mean_hz_s"
    starts = [token.start_s for token in vowel_tokens]
    if (
        f2_status != "ok"
        or len(vowel_tokens) < 2
        or any(start is None or not math.isfinite(start) for start in starts)
    ):
        _flag(
            values,
            issues,
            (transition_key,),
            _annotation_code(f2_status),
            "two timed vowels with finite F2 annotations are required",
            recording_id,
            speaker_id,
        )
    else:
        slopes = [
            abs(next_f2 - current_f2) / (next_start - current_start)
            for current_f2, next_f2, current_start, next_start in zip(
                f2, f2[1:], starts, starts[1:]
            )
            if next_start > current_start
        ]
        if slopes:
            values[transition_key] = float(np.mean(slopes))
        else:
            _flag(
                values,
                issues,
                (transition_key,),
                "INVALID_TASK_ANNOTATION",
                "vowel onsets must be strictly increasing",
                recording_id,
                speaker_id,
            )

    stop_ids = tuple(token.id for token, kind in zip(tokens, types) if kind == "stop")
    _set_mean_sd(
        values,
        issues,
        document,
        "vot_s",
        stop_ids,
        "artic_vot_mean_s",
        "artic_vot_sd_s",
        recording_id,
        speaker_id,
    )
    _set_mean_sd(
        values,
        issues,
        document,
        "stop_gap_s",
        stop_ids,
        "artic_stop_gap_mean_s",
        "artic_stop_gap_sd_s",
        recording_id,
        speaker_id,
    )

    consonants = tuple(
        token for token, kind in zip(tokens, types) if kind in {"consonant", "stop", "fricative"}
    )
    durations = [token_duration(token) for token in consonants]
    if not durations or any(duration is None for duration in durations):
        _flag(
            values,
            issues,
            ("artic_consonant_duration_mean_s", "artic_consonant_duration_sd_s"),
            "MISSING_ANNOTATION",
            "complete positive consonant token times are required",
            recording_id,
            speaker_id,
        )
    else:
        (
            values["artic_consonant_duration_mean_s"],
            values["artic_consonant_duration_sd_s"],
        ) = finite_mean_sd(durations)

    fricative_ids = tuple(token.id for token, kind in zip(tokens, types) if kind == "fricative")
    for moment in range(1, 5):
        key = f"artic_fricative_m{moment}_mean"
        status, numbers = _numeric_layer(document, f"spectral_moment_{moment}", fricative_ids)
        if status == "ok" and numbers:
            values[key] = float(np.mean(numbers))
        else:
            _flag(
                values,
                issues,
                (key,),
                _annotation_code(status),
                f"complete finite spectral_moment_{moment} annotations are required",
                recording_id,
                speaker_id,
            )

    consonant_ids = tuple(token.id for token in consonants)
    status, attenuation = _numeric_layer(document, "resonance_attenuation_db", consonant_ids)
    key = "artic_resonance_attenuation_mean_db"
    if status == "ok" and attenuation:
        values[key] = float(np.mean(attenuation))
    else:
        _flag(
            values,
            issues,
            (key,),
            _annotation_code(status),
            "complete finite resonance attenuation annotations are required",
            recording_id,
            speaker_id,
        )


def _rhythm_features(tokens, types, values, issues, recording_id, speaker_id):
    if types is None:
        _flag(
            values,
            issues,
            RHYTHM_KEYS,
            "MISSING_ANNOTATION",
            "complete segment_type annotations are required",
            recording_id,
            speaker_id,
        )
        return
    groups = {
        "v": [token_duration(token) for token, kind in zip(tokens, types) if kind == "vowel"],
        "c": [
            token_duration(token)
            for token, kind in zip(tokens, types)
            if kind in {"consonant", "stop", "fricative"}
        ],
    }
    all_durations = groups["v"] + groups["c"]
    if not all_durations or any(duration is None for duration in all_durations):
        _flag(
            values,
            issues,
            RHYTHM_KEYS,
            "MISSING_ANNOTATION",
            "complete positive vowel and consonant token times are required",
            recording_id,
            speaker_id,
        )
        return
    vowel_total = sum(groups["v"])
    consonant_total = sum(groups["c"])
    if vowel_total + consonant_total > 0.0:
        values["rhythm_percent_vocalic"] = 100.0 * vowel_total / (vowel_total + consonant_total)
    for suffix, durations in groups.items():
        varco_key = f"rhythm_varco_{suffix}"
        rpvi_key = f"rhythm_rpvi_{suffix}"
        npvi_key = f"rhythm_npvi_{suffix}"
        if durations:
            mean, sd = finite_mean_sd(durations)
            values[varco_key] = 100.0 * sd / mean
        else:
            _flag(
                values,
                issues,
                (varco_key,),
                "MISSING_ANNOTATION",
                f"no {suffix} durations are available",
                recording_id,
                speaker_id,
            )
        if len(durations) < 2:
            _flag(
                values,
                issues,
                (rpvi_key, npvi_key),
                "MISSING_ANNOTATION",
                f"two {suffix} durations are required",
                recording_id,
                speaker_id,
            )
        else:
            pairs = tuple(zip(durations, durations[1:]))
            values[rpvi_key] = float(np.mean([abs(a - b) for a, b in pairs]))
            values[npvi_key] = 100.0 * float(
                np.mean([abs(a - b) / ((a + b) / 2.0) for a, b in pairs])
            )


def _ddk_features(document, tokens, task_spec, values, issues, recording_id, speaker_id):
    if not isinstance(task_spec, dict) or task_spec.get("task") != "ddk":
        _flag(
            values,
            issues,
            DDK_KEYS,
            "MISSING_ANNOTATION",
            "DDK features require task_spec task='ddk'",
            recording_id,
            speaker_id,
        )
        return
    if not tokens or any(token_duration(token) is None for token in tokens):
        _flag(
            values,
            issues,
            DDK_KEYS,
            "MISSING_ANNOTATION",
            "DDK tokens require complete positive start/end times",
            recording_id,
            speaker_id,
        )
        return
    ordered = sorted(tokens, key=lambda token: token.start_s)
    onsets = np.asarray([token.start_s for token in ordered], dtype=float)
    durations = [token_duration(token) for token in ordered]
    span = ordered[-1].end_s - ordered[0].start_s
    if len(ordered) < 2 or span <= 0.0:
        _flag(
            values,
            issues,
            DDK_KEYS,
            "MISSING_ANNOTATION",
            "at least two aligned DDK tokens are required",
            recording_id,
            speaker_id,
        )
        return
    intervals = np.diff(onsets)
    if not np.all(intervals > 0.0):
        _flag(
            values,
            issues,
            DDK_KEYS,
            "INVALID_TASK_ANNOTATION",
            "DDK onsets must be finite and strictly increasing",
            recording_id,
            speaker_id,
        )
        return
    mean = float(np.mean(intervals))
    values["task_ddk_rate_syllables_s"] = len(ordered) / span
    values["task_ddk_inter_onset_mean_s"] = mean
    values["task_ddk_inter_onset_median_s"] = float(np.median(intervals))
    values["task_ddk_inter_onset_sd_s"] = float(np.std(intervals))
    values["task_ddk_inter_onset_cv"] = float(np.std(intervals)) / mean
    values["task_ddk_voiced_interval_mean_s"] = float(np.mean(durations))
    values["task_ddk_decay_ratio"] = float(intervals[0] / intervals[-1])
    variability = np.abs(np.diff(intervals))
    if variability.size:
        values["task_ddk_instability_s"] = float(np.mean(variability))
        midpoint = (onsets[:-1] + onsets[1:]) / 2.0
        values["task_ddk_acceleration_syllables_s2"] = float(
            np.polyfit(midpoint, 1.0 / intervals, 1)[0]
        )
    else:
        _flag(
            values,
            issues,
            ("task_ddk_instability_s", "task_ddk_acceleration_syllables_s2"),
            "MISSING_ANNOTATION",
            "three DDK onsets are required for change measures",
            recording_id,
            speaker_id,
        )

    label_status, labels = _labels(document, "segment_label", tuple(t.id for t in ordered))
    if label_status != "ok":
        _flag(
            values,
            issues,
            ("task_ddk_sequential_alternating_ratio",),
            _annotation_code(label_status),
            "complete DDK segment labels are required",
            recording_id,
            speaker_id,
        )
    else:
        values["task_ddk_sequential_alternating_ratio"] = float(
            np.mean([current != following for current, following in zip(labels, labels[1:])])
        )


def _respiratory_features(
    document, tokens, types, task_spec, values, issues, recording_id, speaker_id
):
    token_ids = tuple(token.id for token in tokens)
    group_status, groups = _labels(document, "breath_group", token_ids)
    breath_keys = RESPIRATORY_KEYS[:-1]
    if group_status != "ok" or not tokens:
        _flag(
            values,
            issues,
            breath_keys,
            _annotation_code(group_status),
            "complete breath_group annotations are required",
            recording_id,
            speaker_id,
        )
    elif any(token_duration(token) is None for token in tokens):
        _flag(
            values,
            issues,
            breath_keys,
            "MISSING_ANNOTATION",
            "breath groups require complete positive token times",
            recording_id,
            speaker_id,
        )
    else:
        unique_groups = tuple(dict.fromkeys(groups))
        group_durations = []
        for group in unique_groups:
            selected = [token for token, label in zip(tokens, groups) if label == group]
            group_durations.append(
                max(token.end_s for token in selected) - min(token.start_s for token in selected)
            )
        count = len(unique_groups)
        span = max(token.end_s for token in tokens) - min(token.start_s for token in tokens)
        values["resp_breath_group_count"] = float(count)
        values["resp_breath_group_mean_s"], values["resp_breath_group_sd_s"] = finite_mean_sd(
            group_durations
        )
        values["resp_rate_per_min"] = 60.0 * count / span
        if types is None:
            _flag(
                values,
                issues,
                ("resp_pauses_per_breath",),
                "MISSING_ANNOTATION",
                "complete segment_type annotations are required for pause counts",
                recording_id,
                speaker_id,
            )
        else:
            values["resp_pauses_per_breath"] = (
                sum(kind in {"pause", "silence"} for kind in types) / count
            )

    loudness_key = "resp_relative_loudness_db"
    if not isinstance(task_spec, dict) or task_spec.get("calibrated_amplitude") is not True:
        _flag(
            values,
            issues,
            (loudness_key,),
            "UNCALIBRATED_AUDIO",
            "relative loudness requires calibrated_amplitude=true",
            recording_id,
            speaker_id,
        )
        return
    respiration_status, respiration = _numeric_layer(document, "respiration_db", token_ids)
    speech_status, speech = _numeric_layer(document, "speech_db", token_ids)
    if respiration_status == "ok" and speech_status == "ok" and token_ids:
        values[loudness_key] = float(np.mean(np.asarray(speech) - np.asarray(respiration)))
    else:
        _flag(
            values,
            issues,
            (loudness_key,),
            _annotation_code(respiration_status, speech_status),
            "complete finite respiration_db and speech_db annotations are required",
            recording_id,
            speaker_id,
        )


def _sustained_features(
    document, tokens, types, task_spec, values, issues, recording_id, speaker_id
):
    if not isinstance(task_spec, dict) or task_spec.get("task") != "sustained_vowel":
        _flag(
            values,
            issues,
            SUSTAINED_VOWEL_KEYS,
            "MISSING_ANNOTATION",
            "sustained-vowel features require task_spec task='sustained_vowel'",
            recording_id,
            speaker_id,
        )
        return
    utterance_durations = [
        utterance.end_s - utterance.start_s
        for utterance in document.utterances
        if utterance.speaker_id == speaker_id
        and math.isfinite(utterance.end_s - utterance.start_s)
        and utterance.end_s > utterance.start_s
    ]
    if utterance_durations:
        values["task_max_phonation_time_s"] = max(utterance_durations)
    else:
        _flag(
            values,
            issues,
            ("task_max_phonation_time_s",),
            "MISSING_ANNOTATION",
            "a positive sustained-vowel utterance duration is required",
            recording_id,
            speaker_id,
        )

    interval_keys = (
        "voice_gaping_interval_rate_per_min",
        "voice_subharmonic_interval_proportion",
    )
    durations = [token_duration(token) for token in tokens]
    if types is None or not durations or any(duration is None for duration in durations):
        _flag(
            values,
            issues,
            interval_keys,
            "MISSING_ANNOTATION",
            "complete segment types and token times are required",
            recording_id,
            speaker_id,
        )
        voiced_ids = ()
    else:
        total = sum(durations)
        values["voice_gaping_interval_rate_per_min"] = (
            60.0 * sum(kind in {"gaping", "gaping_interval"} for kind in types) / total
        )
        values["voice_subharmonic_interval_proportion"] = (
            sum(
                duration
                for duration, kind in zip(durations, types)
                if kind in {"subharmonic", "subharmonic_interval"}
            )
            / total
        )
        voiced_ids = tuple(
            token.id
            for token, kind in zip(tokens, types)
            if kind not in {"gaping", "gaping_interval", "pause", "silence"}
        )

    for layer, key in (
        ("f0_hz", "voice_sustained_f0_sd_semitones"),
        ("power_db", "voice_sustained_power_sd_db"),
    ):
        status, numbers = _numeric_layer(document, layer, voiced_ids)
        if (
            status != "ok"
            or not numbers
            or (layer == "f0_hz" and any(value <= 0 for value in numbers))
        ):
            code = "INVALID_TASK_ANNOTATION" if status == "invalid" else "MISSING_ANNOTATION"
            if layer == "f0_hz" and numbers and any(value <= 0 for value in numbers):
                code = "INVALID_TASK_ANNOTATION"
            _flag(
                values,
                issues,
                (key,),
                code,
                f"complete finite {layer} sustained-vowel annotations are required",
                recording_id,
                speaker_id,
            )
        elif layer == "f0_hz":
            values[key] = float(np.std(12.0 * np.log2(numbers)))
        else:
            values[key] = float(np.std(numbers))


def extract_motor_features(
    document,
    *,
    target_speaker=None,
    recording_id: str = "",
    task_spec=None,
) -> tuple[dict[str, float], tuple[FeatureIssue, ...]]:
    """Extract every motor-neuro key from explicit reviewed annotations."""
    speaker_id = _resolve_target_speaker(document, target_speaker)
    values = {key: math.nan for key in ALL_KEYS}
    issues: list[FeatureIssue] = []
    if document is None:
        loudness_key = "resp_relative_loudness_db"
        _flag(
            values,
            issues,
            tuple(key for key in ALL_KEYS if key != loudness_key),
            "MISSING_ANNOTATION",
            "a speech document with reviewed annotations is required",
            recording_id,
            speaker_id,
        )
        _flag(
            values,
            issues,
            (loudness_key,),
            "MISSING_ANNOTATION"
            if isinstance(task_spec, dict) and task_spec.get("calibrated_amplitude") is True
            else "UNCALIBRATED_AUDIO",
            "relative loudness requires calibrated audio annotations",
            recording_id,
            speaker_id,
        )
        return values, tuple(issues)

    tokens = target_tokens(document, speaker_id)
    type_status, types = _labels(document, "segment_type", tuple(token.id for token in tokens))
    if type_status != "ok":
        types = None

    _articulation_features(document, tokens, types, values, issues, recording_id, speaker_id)
    _rhythm_features(tokens, types, values, issues, recording_id, speaker_id)
    _ddk_features(document, tokens, task_spec, values, issues, recording_id, speaker_id)
    _respiratory_features(
        document, tokens, types, task_spec, values, issues, recording_id, speaker_id
    )
    _sustained_features(
        document, tokens, types, task_spec, values, issues, recording_id, speaker_id
    )

    for key, value in values.items():
        if not math.isfinite(value):
            _flag(
                values,
                issues,
                (key,),
                "MISSING_ANNOTATION",
                "feature prerequisites are unavailable",
                recording_id,
                speaker_id,
            )
    return values, tuple(issues)


__all__ = ["ALL_KEYS", "extract_motor_features"]
