"""Opt-in preprocessing A/B study: manifest-driven arms, scoring, and records.

Arms (SPEC "Comparison arms"): N0 baseline and N1 ``vad_asr`` run through the
exact ``say-transcribe compare`` code paths (``transcribe``; ``transcribe_windows``
+ ``result_from_windows``); P0 runs the preprocessing profile on the same
declared channel; PF/PD denoise the P0 signal through isolated-environment
workers. Nothing here re-implements ASR, VAD, or DSP stages. Every artifact this
module writes belongs in a private output directory.

Coverage metrics are pure functions of one arm's VAD windows and the frozen
reference intervals, so no arm can be scored differently from another. They work
in integer milliseconds on purpose: the SPEC's retention rule is "at least 50% of
the reference duration", and integer ``2 * overlap >= duration`` tests that
boundary exactly, where float seconds can land either side of it. The feature
package's interval helpers stay in float seconds because they measure speech
frames, not a thresholded boundary.
"""

from dataclasses import dataclass
from functools import lru_cache
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from say_transcribe.asr import (
    AsrResult,
    PhoWhisperBackend,
    compute_sha256,
    result_from_windows,
    transcribe,
    transcribe_windows,
)
from say_transcribe.audio import extract_channel, read_wav, resample_to_16kHz
from say_transcribe.denoise import DenoiserSpec, denoise_pcm
from say_transcribe.evaluate import extract_session_items, items_from_texts, score_uncapped
from say_transcribe.manifest import ManifestRow, load_denoiser_specs, load_manifest
from say_transcribe.profile import apply_p0_profile
from say_transcribe.vad import SAMPLE_RATE, get_speech_windows, merge_asr_windows
from speech_features.catalog import list_features
from speech_features.features.acoustic import extract_acoustic_features
from speech_features.features.standardized.opensmile_adapter import extract_egemaps_features
from speech_features.formats.chat import InvalidChatError, decode_chat

# Frozen per SPEC: value dev-selected in the v5 pilot; not tunable per arm.
STUDY_VAD_THRESHOLD = 0.2

# SPEC "Acoustic feature evaluation": eGeMAPS runs per PAR utterance of >= 1 s.
FEATURE_MIN_UTTERANCE_MS = 1000

# SPEC "Comparison arms" order; an arm is recorded only when its inputs exist.
ARM_ORDER = ("N0", "N1", "P0", "PF", "PD")


class StudyError(Exception):
    """A study run failed under a stable, redacted error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class AudioViews:
    channel_samples: np.ndarray
    sample_rate: int
    sample_width: int
    audio_16k: np.ndarray


def verify_source(row: ManifestRow) -> str:
    """Hash gate: verify the master before any decoder or model runs."""
    try:
        actual = compute_sha256(row.audio_path)
    except OSError:
        raise StudyError("INVALID_ARGUMENT", "master audio could not be read") from None
    if actual.lower() != row.sha256.lower():
        raise StudyError("SOURCE_HASH_MISMATCH", "Master audio does not match manifest SHA-256")
    return actual


def prepare_views(row: ManifestRow) -> AudioViews:
    """Decode once: declared channel at native rate plus the 16 kHz ASR view."""
    # ponytail: this decode+VAD glue mirrors cli.cmd_compare (whose tests patch
    # cli-module names); N0's transcribe() still re-decodes internally, same
    # parity as compare. Unify with cmd_compare if a third consumer appears.
    audio = read_wav(row.audio_path)
    channel_samples = extract_channel(audio, row.channel_index)
    audio_16k = resample_to_16kHz(channel_samples, audio.sample_rate, audio.sample_width)
    return AudioViews(
        channel_samples=channel_samples,
        sample_rate=audio.sample_rate,
        sample_width=audio.sample_width,
        audio_16k=audio_16k,
    )


def arm_vad_windows(audio_16k: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Frozen-threshold Silero speech windows, in samples, for one arm's signal.

    Computed once per arm and reused by both the ASR arm and the coverage metric,
    so the two can never disagree about which audio an arm selected.
    """
    return tuple(get_speech_windows(audio_16k, threshold=STUDY_VAD_THRESHOLD))


def compute_vad_arm(
    audio_16k: np.ndarray,
    source_sha256: str,
    backend: Any,
    windows: Sequence[tuple[int, int]] | None = None,
) -> AsrResult:
    """The compare command's ``vad_asr`` path on any 16 kHz mono signal."""
    speech_windows = arm_vad_windows(audio_16k) if windows is None else tuple(windows)
    asr_windows = merge_asr_windows(speech_windows)
    cached_windows = transcribe_windows(audio_16k, asr_windows, backend)
    return result_from_windows(audio_16k, source_sha256, cached_windows)


@dataclass(frozen=True)
class ReferenceIntervals:
    """Frozen reference timing: one interval per participant (``PAR``) utterance.

    ``zero_duration`` counts PAR utterances whose bullet has no duration. They
    cannot retain audio, so they are excluded from the denominators and reported
    separately rather than silently counted as retained or silently dropped.
    """

    utterances: tuple[tuple[int, int], ...]
    zero_duration: int


def load_reference_document(reference_text: str):
    """Strictly decode a verified reference; timing and features reuse the result."""
    try:
        return decode_chat(reference_text)
    except InvalidChatError:
        raise StudyError("INVALID_ARGUMENT", "reference transcript is not a valid CHAT document") from None


def reference_intervals(document) -> ReferenceIntervals:
    """Participant utterance intervals of a reference document, in milliseconds.

    ``decode_chat`` rejects a speaker tier without a media bullet and a reversed
    ``%xaud`` bullet, but it does not range-check the inline ``start_end`` bullet
    it also accepts, so a transposed one reaches here. Reversed timing would
    silently shrink the retention gate's denominator, so it fails loud instead;
    only a genuinely zero-length utterance is reported as ``zero_duration``.
    """
    participant: list[tuple[int, int]] = []
    for utterance in document.utterances:
        if utterance.speaker_id != "PAR":
            continue
        start = int(round(utterance.start_s * 1000))
        end = int(round(utterance.end_s * 1000))
        if end < start:
            raise StudyError(
                "INVALID_ARGUMENT", "reference participant utterance ends before it starts"
            )
        participant.append((start, end))
    if not participant:
        raise StudyError("INVALID_ARGUMENT", "reference transcript has no participant (PAR) intervals")
    usable = tuple((start, end) for start, end in participant if end > start)
    if not usable:
        raise StudyError("INVALID_ARGUMENT", "reference transcript has no usable participant intervals")
    return ReferenceIntervals(utterances=usable, zero_duration=len(participant) - len(usable))


def windows_to_ms(
    windows: Sequence[tuple[int, int]], sample_rate: int = SAMPLE_RATE
) -> tuple[tuple[int, int], ...]:
    """Convert sample-index windows to ``(start_ms, end_ms)``."""
    scale = 1000.0 / sample_rate
    return tuple((int(round(start * scale)), int(round(end * scale))) for start, end in windows)


def _merge_ms(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union of ``[start, end)`` millisecond intervals, sorted and disjoint."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _overlap_ms(start: int, end: int, merged: Sequence[tuple[int, int]]) -> int:
    """Total duration of ``[start, end)`` covered by a sorted, disjoint union."""
    total = 0
    for window_start, window_end in merged:
        if window_end <= start:
            continue
        if window_start >= end:
            break
        total += min(end, window_end) - max(start, window_start)
    return total


def coverage_metrics(
    vad_windows_ms: Sequence[tuple[int, int]], reference: ReferenceIntervals
) -> dict[str, Any]:
    """Speech coverage and retained-utterance fraction against frozen references.

    Coverage is retained participant-speech duration divided by reference
    participant-speech duration. An utterance is retained when at least half of
    its reference duration overlaps a VAD window (exactly 50% counts as
    retained). Arm-agnostic: only the windows and the reference enter.
    """
    reference_windows = _merge_ms(reference.utterances)
    selected = _merge_ms(vad_windows_ms)

    reference_ms = sum(end - start for start, end in reference_windows)
    retained_ms = sum(_overlap_ms(start, end, selected) for start, end in reference_windows)

    retained_utterances = sum(
        1
        for start, end in reference.utterances
        if 2 * _overlap_ms(start, end, selected) >= end - start
    )
    utterance_count = len(reference.utterances)
    return {
        "reference_speech_ms": reference_ms,
        "retained_speech_ms": retained_ms,
        "coverage": retained_ms / reference_ms if reference_ms else 0.0,
        "utterance_count": utterance_count,
        "retained_utterance_count": retained_utterances,
        "retained_utterance_fraction": (
            retained_utterances / utterance_count if utterance_count else 0.0
        ),
        "zero_duration_utterances": reference.zero_duration,
    }


def eligible_feature_intervals(reference: ReferenceIntervals) -> tuple[tuple[int, int], ...]:
    """Participant utterances long enough for a per-utterance eGeMAPS extraction."""
    return tuple(
        (start, end) for start, end in reference.utterances if end - start >= FEATURE_MIN_UTTERANCE_MS
    )


def _opensmile_available() -> bool:
    return importlib.util.find_spec("opensmile") is not None


def require_egemaps_available() -> None:
    """Fail before extracting anything when the eGeMAPS extra is absent.

    The shared adapter reports a missing optional dependency as all-NaN rows; a
    study run must not, because those NaNs would be indistinguishable from a real
    missingness finding.
    """
    if not _opensmile_available():
        raise StudyError(
            "FEATURE_EXTRACTION_FAILED",
            "opensmile is not installed, so this study run cannot extract eGeMAPS",
        )


@lru_cache(maxsize=None)
def _feature_domains(pack: str) -> dict[str, str]:
    """Catalog domain ("family") of every registered key of one pack."""
    return {definition.key: definition.domain for definition in list_features(pack=pack)}


def _is_finite(value: Any) -> bool:
    try:
        return bool(math.isfinite(value))
    except (TypeError, ValueError):
        return False


def _family_coverage(
    pack: str, values_by_key: dict[str, Sequence[float]]
) -> dict[str, dict[str, Any]]:
    """Finite-value coverage per feature family (catalog domain) for one pack."""
    domains = _feature_domains(pack)
    counts: dict[str, list[int]] = {}
    for key, values in values_by_key.items():
        entry = counts.setdefault(domains.get(key, "unregistered"), [0, 0])
        entry[0] += len(values)
        entry[1] += sum(1 for value in values if _is_finite(value))
    return {
        family: {
            "values": values,
            "finite": finite,
            "coverage": finite / values if values else 0.0,
        }
        for family, (values, finite) in sorted(counts.items())
    }


def _issue_codes(issues: Sequence[Any]) -> dict[str, int]:
    """Counts of structured issue codes; messages never reach the record."""
    codes: dict[str, int] = {}
    for issue in issues:
        code = getattr(issue, "code", "UNKNOWN")
        codes[code] = codes.get(code, 0) + 1
    return dict(sorted(codes.items()))


def extract_arm_features(
    audio_16k: np.ndarray,
    sample_rate: int,
    document: Any,
    reference: ReferenceIntervals,
) -> dict[str, Any]:
    """Acoustic pack and per-utterance eGeMAPS for one arm's 16 kHz mono signal.

    The array arrives already channel-selected, so the stereo-downmixing file
    reader is never in this path. Both extractions are scoped to the same human
    ``PAR`` intervals in every arm; eGeMAPS additionally requires an utterance of
    at least one second. Returns counts only: no participant identity, path, or
    transcript text. A vector is valid when every value is finite; eGeMAPS has no
    defined absolute range, so none is invented here.
    """
    require_egemaps_available()

    acoustic, acoustic_issues = extract_acoustic_features(
        audio_16k, sample_rate, document=document, target_speaker="PAR"
    )
    acoustic_finite = sum(1 for value in acoustic.values() if _is_finite(value))
    record: dict[str, Any] = {
        "sample_rate": sample_rate,
        "samples": int(audio_16k.shape[0]),
        "acoustic": {
            "values": len(acoustic),
            "finite": acoustic_finite,
            "coverage": acoustic_finite / len(acoustic) if acoustic else 0.0,
            "valid": acoustic_finite == len(acoustic) and bool(acoustic),
            "families": _family_coverage("acoustic", {key: [value] for key, value in acoustic.items()}),
            "issue_codes": _issue_codes(acoustic_issues),
        },
        "egemaps": {},
    }

    eligible = eligible_feature_intervals(reference)
    rows: list[dict[str, Any]] = []
    pooled: dict[str, list[float]] = {}
    issue_counts: dict[str, int] = {}
    for start_ms, end_ms in eligible:
        start = int(round(start_ms * sample_rate / 1000))
        stop = min(int(round(end_ms * sample_rate / 1000)), int(audio_16k.shape[0]))
        if stop <= start:
            rows.append({"start_ms": start_ms, "end_ms": end_ms, "valid": False, "reason": "TRUNCATED"})
            continue
        features, issues, _ = extract_egemaps_features(
            audio_16k[start:stop], sample_rate, recording_id="", speaker_id="PAR"
        )
        for code, count in _issue_codes(issues).items():
            issue_counts[code] = issue_counts.get(code, 0) + count
        for key, value in features.items():
            pooled.setdefault(key, []).append(value)
        rows.append(
            {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "valid": bool(features) and all(_is_finite(value) for value in features.values()),
                "reason": None,
            }
        )

    valid_rows = sum(1 for row in rows if row["valid"])
    vector_count = len(pooled)
    finite_values = sum(1 for values in pooled.values() for value in values if _is_finite(value))
    total_values = sum(len(values) for values in pooled.values())
    record["egemaps"] = {
        "utterances": len(eligible),
        "extracted": len(rows) - sum(1 for row in rows if row.get("reason") == "TRUNCATED"),
        "valid_utterances": valid_rows,
        "truncated_utterances": sum(1 for row in rows if row.get("reason") == "TRUNCATED"),
        "keys": vector_count,
        "values": total_values,
        "finite": finite_values,
        "coverage": finite_values / total_values if total_values else 0.0,
        "families": _family_coverage("standardized_acoustic", pooled),
        "issue_codes": dict(sorted(issue_counts.items())),
        "utterances_detail": rows,
    }
    return record


def compute_arm_results(
    row: ManifestRow,
    source_sha256: str,
    device: str,
    backend: Any,
    denoisers: dict[str, DenoiserSpec],
) -> dict[str, AsrResult]:
    """Compute every arm whose inputs are configured, in SPEC arm order."""
    baseline = transcribe(
        audio_path=row.audio_path,
        channel_index=row.channel_index,
        device=device,
        backend=backend,
    )
    if baseline.source_sha256.lower() != source_sha256.lower():
        raise StudyError("SOURCE_HASH_MISMATCH", "Master audio changed during processing")

    views = prepare_views(row)
    profile = apply_p0_profile(views.channel_samples, views.sample_rate, views.sample_width)

    signals: dict[str, np.ndarray] = {"N1": views.audio_16k, "P0": profile.samples}
    if "PF" in denoisers:
        signals["PF"] = denoise_pcm(profile.samples, profile.sample_rate, denoisers["PF"])
    if "PD" in denoisers:
        signals["PD"] = denoise_pcm(profile.samples, profile.sample_rate, denoisers["PD"])

    results: dict[str, AsrResult] = {"N0": baseline}
    for name in ARM_ORDER[1:]:
        if name in signals:
            results[name] = compute_vad_arm(signals[name], source_sha256, backend)
    return results


def session_record(row: ManifestRow, arms: dict[str, AsrResult]) -> dict[str, Any]:
    """Build the private per-session record: arm predictions plus uncapped scores."""
    try:
        reference_text = row.reference_path.read_text(encoding="utf-8")
    except OSError:
        raise StudyError("INVALID_ARGUMENT", "reference transcript could not be read") from None
    gold = extract_session_items(reference_text)
    if not gold["syllables"]:
        raise StudyError("INVALID_ARGUMENT", "reference transcript contains no scorable text")

    record: dict[str, Any] = {
        "session_id": row.session_id,
        "participant_id": row.participant_id,
        "split": row.split,
        "channel_index": row.channel_index,
        "sha256": row.sha256,
        "asr_revision": row.asr_revision,
        "arms": {},
    }
    for name in ARM_ORDER:
        if name not in arms:
            continue
        result = arms[name]
        hyp = items_from_texts([segment.text for segment in result.segments])
        record["arms"][name] = {
            "segments": [
                {"start_ms": s.start_ms, "end_ms": s.end_ms, "text": s.text}
                for s in result.segments
            ],
            "warnings": list(result.warnings),
            "scores": score_uncapped(gold, hyp),
        }
    return record


def write_session_record(record: dict[str, Any], out_dir: Path) -> Path:
    target = out_dir / f"{record['session_id']}.json"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        raise StudyError("STUDY_FAILED", "study output could not be written") from None
    return target


def run_study(
    manifest_path: Path,
    out_dir: Path,
    device: str = "cpu",
    asr_backend: Any = None,
) -> int:
    """Run the study arms for every manifest row into a private output directory."""
    rows = load_manifest(manifest_path)
    denoisers = load_denoiser_specs(manifest_path)
    if asr_backend is None:
        asr_backend = PhoWhisperBackend(device=device)
    for row in rows:
        source_sha256 = verify_source(row)
        arms = compute_arm_results(row, source_sha256, device, asr_backend, denoisers)
        # Re-verify after the (potentially hour-long) arm chain so a master swapped
        # mid-run cannot mix sources into one record.
        verify_source(row)
        write_session_record(session_record(row, arms), out_dir)
    return 0
