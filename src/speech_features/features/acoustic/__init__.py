"""Acoustic quality, timing, phonation, resonance, spectrum, and rhythm (Tasks 5--7).

Public entry points:

- :func:`extract_acoustic_features` — quality, timing, phonation, resonance,
  spectrum, and rhythm features from a mono float array plus an optional
  :class:`~speech_features.document.SpeechDocument`.
- :func:`extract_acoustic_bundle` — WAV path (reusing the shared PCM reader)
  to a :class:`~speech_features.result.FeatureBundle` with deterministic
  recording columns.

The module registers the acoustic pack's stable quality/timing/phonation/
resonance/spectrum/rhythm definitions at import time (idempotent).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ...catalog import CATALOG_VERSION
from ...pipeline import _read_wav_with_width
from ...result import FeatureBundle, FeatureIssue, InvalidAudioError
from ...schema import ExtractionConfig
from .definitions import ALL_KEYS, register_acoustic_features
from .phonation import phonation_features
from .quality import quality_features
from .resonance import resonance_features
from .rhythm import rhythm_features
from .spectrum import spectrum_features
from .timing import timing_features

register_acoustic_features()

_UTTERANCE_COLUMNS = ("recording_id", "speaker_id", "utterance_id", "start_s", "end_s")
_ISSUE_COLUMNS = (
    "recording_id",
    "speaker_id",
    "utterance_id",
    "feature",
    "code",
    "severity",
    "message",
)


def _extract(
    audio,
    sample_rate: int,
    *,
    document,
    target_speaker,
    allow_unaligned: bool,
    config: ExtractionConfig,
    recording_id: str,
    clipping_boundary: float = 1.0,
) -> tuple[dict[str, float], list[FeatureIssue], str]:
    """Quality + timing + phonation features, issues, and the resolved speaker id."""
    timing, issues, speaker_id, intervals = timing_features(
        audio,
        sample_rate,
        document=document,
        target_speaker=target_speaker,
        allow_unaligned=allow_unaligned,
        config=config,
        recording_id=recording_id,
    )
    features = {
        **quality_features(audio, sample_rate, clipping_boundary=clipping_boundary),
        **timing,
    }
    features.update(
        phonation_features(
            audio,
            sample_rate,
            intervals=intervals,
            config=config,
            recording_id=recording_id,
            speaker_id=speaker_id,
            issues=issues,
        )
    )
    features.update(
        resonance_features(
            audio,
            sample_rate,
            intervals=intervals,
            config=config,
            recording_id=recording_id,
            speaker_id=speaker_id,
            issues=issues,
        )
    )
    features.update(
        spectrum_features(
            audio,
            sample_rate,
            intervals=intervals,
            config=config,
            recording_id=recording_id,
            speaker_id=speaker_id,
            issues=issues,
        )
    )
    features.update(
        rhythm_features(
            document,
            speaker_id,
            recording_id=recording_id,
            issues=issues,
        )
    )
    if math.isnan(features["audio_rms_dbfs"]):
        issues.append(
            FeatureIssue(
                recording_id=recording_id,
                speaker_id=speaker_id,
                code="NO_AUDIO",
                severity="warning",
                message="recording is digital silence; RMS dBFS is undefined",
                feature="audio_rms_dbfs",
            )
        )
    return features, issues, speaker_id


def extract_acoustic_features(
    audio,
    sample_rate: int,
    *,
    document=None,
    target_speaker=None,
    allow_unaligned: bool = False,
    config: ExtractionConfig | None = None,
    recording_id: str = "",
) -> tuple[dict[str, float], tuple[FeatureIssue, ...]]:
    """Extract the acoustic pack's recording-level quality, timing, phonation,
    and prosody features.

    ``audio`` is a mono float array already sampled at ``sample_rate`` (see
    :func:`speech_features.pipeline.read_wav`). ``document`` is an optional
    :class:`~speech_features.document.SpeechDocument` whose aligned utterances
    select the target speaker's audio; without one, whole-recording analysis
    requires ``allow_unaligned=True`` and emits an ``UNALIGNED_SPEAKER``
    warning. Multiple speakers with no ``target_speaker`` raise
    ``TARGET_SPEAKER_REQUIRED``; an unknown target raises ``INVALID_CONFIG``.

    Returns ``(features, issues)``: every registered key is present (``NaN``
    when its prerequisite is unavailable) and each ``NaN`` is paired with a
    structured issue. Formula and denominator details are documented in
    :mod:`speech_features.features.acoustic.quality`,
    :mod:`speech_features.features.acoustic.timing`, and
    :mod:`speech_features.features.acoustic.phonation`.
    """
    cfg = config if config is not None else ExtractionConfig()
    try:
        arr = np.asarray(audio, dtype=float)
        invalid = sample_rate <= 0 or arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr))
    except (ValueError, TypeError):
        invalid = True
    if invalid:
        raise InvalidAudioError(
            "acoustic input must be a finite non-empty mono float array with a positive sample rate"
        )
    features, issues, _ = _extract(
        arr,
        sample_rate,
        document=document,
        target_speaker=target_speaker,
        allow_unaligned=allow_unaligned,
        config=cfg,
        recording_id=recording_id,
    )
    return features, tuple(issues)


def extract_acoustic_bundle(
    audio_path,
    document=None,
    *,
    target_speaker=None,
    allow_unaligned: bool = False,
    config: ExtractionConfig | None = None,
    recording_id: str = "",
) -> FeatureBundle:
    """Read a standard PCM WAV and return a recording-level acoustic bundle.

    The shared :func:`speech_features.pipeline.read_wav` decoder is reused:
    malformed WAVs raise ``INVALID_AUDIO``; unsupported media (non-PCM,
    non-standard width or channel count) raise ``UNSUPPORTED_AUDIO``. The
    recordings table carries the identifier columns plus the sorted catalog
    keys with float values; the utterances table stays empty with its
    identifier columns; issues carry the stable codes above. Clipping is
    measured against the source width's positive full-scale boundary
    (``1 - 2^-(width*8-1)``), so positive and negative full-scale PCM of every
    supported width count as clipped.
    """
    cfg = config if config is not None else ExtractionConfig()
    audio, width = _read_wav_with_width(audio_path, sample_rate=cfg.sample_rate)
    clipping_boundary = 1.0 - 2.0 ** -(width * 8 - 1)
    features, issues, speaker_id = _extract(
        audio,
        cfg.sample_rate,
        document=document,
        target_speaker=target_speaker,
        allow_unaligned=allow_unaligned,
        config=cfg,
        recording_id=recording_id,
        clipping_boundary=clipping_boundary,
    )
    columns = ("recording_id", "speaker_id", *sorted(ALL_KEYS))
    recordings = pd.DataFrame(
        [[recording_id, speaker_id, *(features[key] for key in sorted(ALL_KEYS))]], columns=columns
    )
    utterances = pd.DataFrame(columns=_UTTERANCE_COLUMNS)
    issue_rows = [
        (i.recording_id, i.speaker_id, i.utterance_id, i.feature, i.code, i.severity, i.message)
        for i in issues
    ]
    issues_df = pd.DataFrame(issue_rows, columns=_ISSUE_COLUMNS)
    provenance = {
        "catalog_version": CATALOG_VERSION,
        "packs": ("acoustic",),
        "target_speaker": speaker_id or None,
        "alignment": "speaker"
        if not any(i.code == "UNALIGNED_SPEAKER" for i in issues)
        else "whole_recording",
        "allow_unaligned": bool(allow_unaligned),
    }
    return FeatureBundle(
        recordings=recordings, utterances=utterances, issues=issues_df, provenance=provenance
    )


__all__ = [
    "ALL_KEYS",
    "extract_acoustic_bundle",
    "extract_acoustic_features",
]
