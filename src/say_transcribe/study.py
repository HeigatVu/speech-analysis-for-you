"""Opt-in preprocessing A/B study: manifest-driven arms, scoring, and records.

Arms (SPEC "Comparison arms"): N0 baseline and N1 ``vad_asr`` run through the
exact ``say-transcribe compare`` code paths (``transcribe``; ``transcribe_windows``
+ ``result_from_windows``); P0 runs the preprocessing profile on the same
declared channel; PF/PD denoise the P0 signal through isolated-environment
workers. Nothing here re-implements ASR, VAD, or DSP stages. Every artifact this
module writes belongs in a private output directory.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

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
from say_transcribe.vad import get_speech_windows, merge_asr_windows

# Frozen per SPEC: value dev-selected in the v5 pilot; not tunable per arm.
STUDY_VAD_THRESHOLD = 0.2

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


def compute_vad_arm(audio_16k: np.ndarray, source_sha256: str, backend: Any) -> AsrResult:
    """The compare command's ``vad_asr`` path on any 16 kHz mono signal."""
    speech_windows = get_speech_windows(audio_16k, threshold=STUDY_VAD_THRESHOLD)
    asr_windows = merge_asr_windows(speech_windows)
    cached_windows = transcribe_windows(audio_16k, asr_windows, backend)
    return result_from_windows(audio_16k, source_sha256, cached_windows)


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
