"""Syllable-duration rhythm measures from explicit token times (Task 7).

Only explicit target-speaker ``word``-kind ``DocumentToken`` start/end times
are used; no text is split and no Vietnamese word/syllable boundary is ever
inferred. Each word-kind token with finite times and ``end_s > start_s``
contributes one syllable observation with duration ``end_s - start_s``;
tokens sharing a ``word_id`` are *not* collapsed (a multi-syllable word has
one token per syllable). Zero-duration tokens are degenerate and excluded.

Formulas
--------
- ``time_syllable_duration_mean_s`` — mean of all valid durations (requires
  at least one observation).
- ``time_syllable_duration_sd_s`` — population SD of the same (requires at
  least two observations).
- ``time_syllable_duration_cv`` — SD / mean (requires at least two
  observations and a positive mean).
- ``time_syllable_duration_npvi`` — the normalized pairwise variability index
  ``100 * mean(abs(d[i+1] - d[i]) / ((d[i+1] + d[i]) / 2))``. Pairs are
  formed only between consecutive tokens *inside the same utterance*, then
  pooled across the target speaker's utterances; a pair is never formed
  across an utterance boundary, so utterance-final and utterance-initial
  syllables do not interact. Requires at least one valid intra-utterance
  pair.

Sufficiency
-----------
Missing annotations (no document, no word-kind tokens with times, or too few
observations) yield float ``NaN`` with one ``MISSING_ANNOTATION`` issue per
affected key; no value is fabricated as zero.
"""

from __future__ import annotations

import math

import numpy as np

from ...result import FeatureIssue
from .definitions import RHYTHM_KEYS


def _issue(
    recording_id: str, speaker_id: str, code: str, message: str, feature: str
) -> FeatureIssue:
    return FeatureIssue(
        recording_id=recording_id,
        speaker_id=speaker_id,
        code=code,
        severity="warning",
        message=message,
        feature=feature,
    )


def _flag(
    features: dict[str, float],
    keys,
    code: str,
    message: str,
    recording_id: str,
    speaker_id: str,
    issues: list[FeatureIssue],
) -> None:
    for key in keys:
        features[key] = math.nan
        issues.append(_issue(recording_id, speaker_id, code, message, feature=key))


def rhythm_features(
    document,
    speaker_id: str,
    recording_id: str,
    issues: list[FeatureIssue],
) -> dict[str, float]:
    """Compute the four ``time_syllable_duration_*`` keys from token times.

    ``document`` is the (optional) :class:`~speech_features.document.SpeechDocument`
    and ``speaker_id`` the resolved target speaker; only that speaker's
    word-kind tokens are counted. Every key is present; unavailable values
    are ``NaN`` and paired with a per-key issue.
    """
    features = {key: math.nan for key in RHYTHM_KEYS}
    if document is None:
        _flag(
            features,
            RHYTHM_KEYS,
            "MISSING_ANNOTATION",
            "no speech document; syllable timing unavailable",
            recording_id,
            speaker_id,
            issues,
        )
        return features

    durations: list[list[float]] = []
    for utterance in document.utterances:
        if utterance.speaker_id != speaker_id:
            continue
        per_utterance = [
            token.end_s - token.start_s
            for token in utterance.tokens
            if token.kind == "word"
            and token.start_s is not None
            and token.end_s is not None
            and token.end_s > token.start_s
        ]
        if per_utterance:
            durations.append(per_utterance)

    pooled = [d for per_utterance in durations for d in per_utterance]
    if not pooled:
        _flag(
            features,
            RHYTHM_KEYS,
            "MISSING_ANNOTATION",
            "no word-kind tokens with explicit positive durations for the "
            "target speaker; syllable timing unavailable",
            recording_id,
            speaker_id,
            issues,
        )
        return features

    mean = float(np.mean(pooled))
    features["time_syllable_duration_mean_s"] = mean
    if len(pooled) < 2:
        _flag(
            features,
            (
                "time_syllable_duration_sd_s",
                "time_syllable_duration_cv",
                "time_syllable_duration_npvi",
            ),
            "MISSING_ANNOTATION",
            "fewer than two syllable durations; variability features unavailable",
            recording_id,
            speaker_id,
            issues,
        )
        return features

    sd = float(np.std(pooled))
    features["time_syllable_duration_sd_s"] = sd
    if mean > 0.0:
        features["time_syllable_duration_cv"] = sd / mean

    pairs: list[float] = []
    for per_utterance in durations:
        if len(per_utterance) >= 2:
            arr = np.asarray(per_utterance, dtype=float)
            pairs.extend(np.abs(arr[1:] - arr[:-1]) / ((arr[1:] + arr[:-1]) / 2.0))
    if not pairs:
        _flag(
            features,
            ("time_syllable_duration_npvi",),
            "MISSING_ANNOTATION",
            "no utterance holds two consecutive syllable tokens; nPVI unavailable",
            recording_id,
            speaker_id,
            issues,
        )
        return features

    features["time_syllable_duration_npvi"] = 100.0 * float(np.mean(pairs))
    return features
