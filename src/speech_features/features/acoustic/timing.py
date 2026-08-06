"""Target-speaker isolation and timing measures (Task 5).

Speaker resolution
------------------
- A requested target must be a documented speaker, else ``INVALID_CONFIG``.
- With several documented speakers and no target, ``TARGET_SPEAKER_REQUIRED``.
- With exactly one documented speaker, that speaker is the implicit target.
- Without a document (or a target with no aligned utterances) analysis fails
  with ``MISSING_ANNOTATION`` unless ``allow_unaligned=True`` requests the
  whole-recording fallback, which emits an ``UNALIGNED_SPEAKER`` warning issue.

Isolation
---------
Normal operation analyses only the target speaker's merged utterance
intervals (overlapping intervals are merged; intervals are clipped to the
recording). Frame-level voice activity detection (VAD, shared helpers from
:mod:`speech_features.acoustic`) runs per interval on the clipped samples, so
examiner audio and the recording's global loudness never contaminate the
participant's voiced/pause statistics.

Timing formulas and denominators
-------------------------------
- ``time_speech_s`` = total duration of the target's merged aligned intervals.
- ``time_speech_ratio`` = speech seconds / recording duration.
- ``time_voiced_segment_{mean,sd}_s`` = mean/population-SD of voiced-run
  durations (frame size/hop from the config), pooled over target intervals.
- ``time_pause_count`` = unvoiced runs inside aligned intervals of at least
  ``pause_threshold_s``; ``time_pause_{mean,sd,max}_s`` summarize those runs;
  ``time_pause_rate_per_min`` = count / (recording duration in minutes);
  ``time_long_pause_count`` counts runs of at least ``long_pause_threshold_s``.
- ``time_response_latency_s`` = mean of ``utt.start - prev.end`` where the
  immediately preceding utterance (latest end <= utt.start) belongs to a
  non-target speaker.
- ``time_overlap_s`` = total seconds of non-target/target interval
  intersection.
- ``time_words_per_min`` = word count / speech minutes. Word tokens group by
  explicit ``word_id`` when present; ungrouped word tokens count one word
  each. ``time_syllables_per_min`` and ``time_articulation_rate_syllables_per_s``
  use the number of word-kind tokens (orthographic syllables; text is never
  split on whitespace).

Missing denominators or prerequisites yield ``NaN`` plus a structured
:class:`~speech_features.result.FeatureIssue`; no value is fabricated as zero.
"""

from __future__ import annotations

import math

import numpy as np

from ...acoustic import _energy_vad, _frames, _pauses
from ...result import (
    FeatureIssue,
    InvalidConfigError,
    MissingAnnotationError,
    TargetSpeakerRequiredError,
)
from ...schema import ExtractionConfig
from .definitions import TIMING_KEYS

VAD_KEYS = (
    "time_voiced_segment_mean_s",
    "time_voiced_segment_sd_s",
    "time_pause_count",
    "time_pause_rate_per_min",
    "time_pause_mean_s",
    "time_pause_sd_s",
    "time_pause_max_s",
    "time_long_pause_count",
)

TRANSCRIPT_KEYS = (
    "time_speech_s",
    "time_speech_ratio",
    "time_response_latency_s",
    "time_overlap_s",
    "time_words_per_min",
    "time_syllables_per_min",
    "time_articulation_rate_syllables_per_s",
)


def _issue(
    recording_id: str, speaker_id: str, code: str, message: str, feature: str | None = None
) -> FeatureIssue:
    return FeatureIssue(
        recording_id=recording_id,
        speaker_id=speaker_id,
        code=code,
        severity="warning",
        message=message,
        feature=feature,
    )


def _merge_raw(intervals, duration_s: float) -> list[tuple[float, float]]:
    """Merge (overlapping) intervals, clipped to the recording."""
    raw = sorted((start, end) for start, end in intervals)
    merged: list[tuple[float, float]] = []
    for start, end in raw:
        start = max(0.0, start)
        end = min(duration_s, end)
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _merged_intervals(document, speaker_id: str, duration_s: float) -> list[tuple[float, float]]:
    """Merge one speaker's utterance intervals, clipped to the recording."""
    return _merge_raw(
        ((u.start_s, u.end_s) for u in document.utterances if u.speaker_id == speaker_id),
        duration_s,
    )


def _resolve_target(
    document, target_speaker, allow_unaligned: bool, recording_id: str, duration_s: float
):
    """Resolve the target speaker and its aligned intervals.

    Returns ``(speaker_id, intervals, issues)`` where ``intervals`` is ``None``
    when the whole-recording fallback is in effect.
    """
    speaker_ids = {speaker.id for speaker in document.speakers} if document is not None else set()
    issues: list[FeatureIssue] = []

    if target_speaker is not None:
        if document is None or target_speaker not in speaker_ids:
            raise InvalidConfigError(
                f"target speaker {target_speaker!r} is not a speaker of the speech document"
            )
        speaker_id = target_speaker
    elif len(speaker_ids) > 1:
        raise TargetSpeakerRequiredError(
            "speech document has multiple speakers; pass target_speaker to isolate one"
        )
    elif len(speaker_ids) == 1:
        speaker_id = next(iter(speaker_ids))
    else:
        speaker_id = ""

    intervals = (
        _merged_intervals(document, speaker_id, duration_s)
        if document is not None and speaker_id
        else []
    )
    if not intervals:
        if not allow_unaligned:
            raise MissingAnnotationError(
                "no usable aligned target-speaker intervals; "
                "pass allow_unaligned=True for whole-recording analysis"
            )
        issues.append(
            _issue(
                recording_id,
                speaker_id,
                "UNALIGNED_SPEAKER",
                "no aligned target-speaker intervals; whole-recording fallback used",
            )
        )
    return speaker_id, intervals or None, issues


def _run_durations(mask: np.ndarray, hop_s: float) -> list[float]:
    """Durations (seconds) of maximal runs of True frames."""
    runs: list[float] = []
    run = 0
    for value in mask:
        if value:
            run += 1
        else:
            if run:
                runs.append(run * hop_s)
            run = 0
    if run:
        runs.append(run * hop_s)
    return runs


def _vad_features(
    audio,
    sample_rate: int,
    intervals,
    config: ExtractionConfig,
    duration_s: float,
    recording_id: str,
    speaker_id: str,
    issues: list[FeatureIssue],
) -> dict[str, float]:
    """Voiced-segment and pause summaries over the target intervals (or whole
    recording in fallback mode)."""
    features = {key: math.nan for key in VAD_KEYS}
    hop_s = config.hop_size / sample_rate
    regions = intervals if intervals else [(0.0, duration_s)]
    voiced_runs: list[float] = []
    pauses: list[float] = []
    any_voiced = False
    for start, end in regions:
        clip = audio[int(round(start * sample_rate)) : int(round(end * sample_rate))]
        if clip.size == 0:
            continue
        frames = _frames(clip, config.frame_size, config.hop_size)
        frame_energy = (frames**2).sum(axis=1)
        voiced = _energy_vad(frame_energy)
        any_voiced = any_voiced or bool(np.any(voiced))
        voiced_runs.extend(_run_durations(voiced, hop_s))
        pauses.extend(_pauses(voiced, hop_s, config.pause_threshold_s))

    if not any_voiced:
        for key in VAD_KEYS:
            issues.append(
                _issue(
                    recording_id,
                    speaker_id,
                    "NO_SPEECH",
                    "no voiced frames in target audio; feature unavailable",
                    feature=key,
                )
            )
        return features

    features["time_voiced_segment_mean_s"] = float(np.mean(voiced_runs))
    features["time_voiced_segment_sd_s"] = float(np.std(voiced_runs))
    pause_array = np.asarray(pauses, dtype=float)
    features["time_pause_count"] = float(pause_array.size)
    features["time_pause_rate_per_min"] = float(pause_array.size / (duration_s / 60.0))
    if pause_array.size:
        features["time_pause_mean_s"] = float(np.mean(pause_array))
        features["time_pause_sd_s"] = float(np.std(pause_array))
        features["time_pause_max_s"] = float(np.max(pause_array))
    features["time_long_pause_count"] = float(
        np.count_nonzero(pause_array >= config.long_pause_threshold_s)
    )
    return features


def _count_words_and_syllables(utterances):
    """Count explicit words and orthographic syllables from word tokens.

    Returns ``None`` when the speaker has no word-kind tokens at all (missing
    annotation); otherwise ``(words, syllables)``.
    """
    word_tokens = [token for utt in utterances for token in utt.tokens if token.kind == "word"]
    if not word_tokens:
        return None
    grouped = {token.word_id for token in word_tokens if token.word_id is not None}
    words = len(grouped) + sum(1 for token in word_tokens if token.word_id is None)
    return words, len(word_tokens)


def _transcript_features(
    document,
    speaker_id: str,
    intervals,
    duration_s: float,
    recording_id: str,
    issues: list[FeatureIssue],
) -> dict[str, float]:
    """Transcript-aligned timing: speech seconds/ratio, latency, overlap, rates."""
    features = {key: math.nan for key in TRANSCRIPT_KEYS}
    if document is None or not intervals:
        reason = "no speech document" if document is None else "no aligned target-speaker intervals"
        for key in TRANSCRIPT_KEYS:
            issues.append(
                _issue(
                    recording_id,
                    speaker_id,
                    "MISSING_ANNOTATION",
                    f"{reason}; feature unavailable",
                    feature=key,
                )
            )
        return features

    speech_s = sum(end - start for start, end in intervals)
    features["time_speech_s"] = float(speech_s)
    features["time_speech_ratio"] = float(speech_s / duration_s)

    utterances = [u for u in document.utterances if u.speaker_id == speaker_id]
    counted = _count_words_and_syllables(utterances)
    if counted is None:
        for key in (
            "time_words_per_min",
            "time_syllables_per_min",
            "time_articulation_rate_syllables_per_s",
        ):
            issues.append(
                _issue(
                    recording_id,
                    speaker_id,
                    "MISSING_ANNOTATION",
                    "no word tokens for the target speaker; feature unavailable",
                    feature=key,
                )
            )
    else:
        words, syllables = counted
        features["time_words_per_min"] = float(words / (speech_s / 60.0))
        features["time_syllables_per_min"] = float(syllables / (speech_s / 60.0))
        features["time_articulation_rate_syllables_per_s"] = float(syllables / speech_s)

    latencies = []
    for utterance in document.utterances:
        if utterance.speaker_id != speaker_id:
            continue
        candidates = [u for u in document.utterances if u.end_s <= utterance.start_s]
        if not candidates:
            continue
        previous = max(candidates, key=lambda u: u.end_s)
        # A target-speaker continuation right after the target's own turn is
        # not a response to the examiner; only an immediate non-target
        # predecessor counts as a response latency.
        if previous.speaker_id != speaker_id:
            latencies.append(utterance.start_s - previous.end_s)
    if latencies:
        features["time_response_latency_s"] = float(sum(latencies) / len(latencies))
    else:
        issues.append(
            _issue(
                recording_id,
                speaker_id,
                "MISSING_ANNOTATION",
                "no examiner-to-participant response pairs; feature unavailable",
                feature="time_response_latency_s",
            )
        )

    other_utterances = [u for u in document.utterances if u.speaker_id != speaker_id]
    if not other_utterances:
        issues.append(
            _issue(
                recording_id,
                speaker_id,
                "MISSING_ANNOTATION",
                "no non-target utterances; feature unavailable",
                feature="time_overlap_s",
            )
        )
    else:
        other_intervals = _merge_raw(((u.start_s, u.end_s) for u in other_utterances), duration_s)
        overlap = 0.0
        for p_start, p_end in intervals:
            for o_start, o_end in other_intervals:
                overlap += max(0.0, min(p_end, o_end) - max(p_start, o_start))
        features["time_overlap_s"] = float(overlap)

    return features


def timing_features(
    audio,
    sample_rate: int,
    *,
    document,
    target_speaker,
    allow_unaligned: bool,
    config: ExtractionConfig,
    recording_id: str,
) -> tuple[dict[str, float], list[FeatureIssue], str]:
    """Resolve the target speaker and compute all timing features.

    Returns ``(features, issues, speaker_id)``. Every timing key is present in
    ``features``; unavailable values are ``NaN`` and paired with an issue.
    """
    duration_s = float(audio.size) / sample_rate
    speaker_id, intervals, issues = _resolve_target(
        document, target_speaker, allow_unaligned, recording_id, duration_s
    )
    features = {key: math.nan for key in TIMING_KEYS}
    features.update(
        _vad_features(
            audio, sample_rate, intervals, config, duration_s, recording_id, speaker_id, issues
        )
    )
    features.update(
        _transcript_features(document, speaker_id, intervals, duration_s, recording_id, issues)
    )
    return features, issues, speaker_id
