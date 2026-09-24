"""Opt-in preprocessing A/B study: manifest-driven arms, scoring, and records.

The native arms N0/N1 go through the exact ``say-transcribe compare`` code
paths (``transcribe`` for baseline, ``transcribe_windows`` +
``result_from_windows`` for ``vad_asr``); nothing here re-implements ASR or VAD.
Every artifact this module writes belongs in a private output directory.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from say_transcribe.asr import (
    AsrResult,
    PhoWhisperBackend,
    compute_sha256,
    result_from_windows,
    transcribe,
    transcribe_windows,
)
from say_transcribe.audio import extract_channel, read_wav, resample_to_16kHz
from say_transcribe.evaluate import extract_session_items, items_from_texts, score_uncapped
from say_transcribe.manifest import ManifestRow, load_manifest
from say_transcribe.vad import get_speech_windows, merge_asr_windows

# Frozen per SPEC: value dev-selected in the v5 pilot; not tunable per arm.
STUDY_VAD_THRESHOLD = 0.2


class StudyError(Exception):
    """A study run failed under a stable, redacted error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class NativeArms:
    baseline: AsrResult
    vad_asr: AsrResult


def verify_source(row: ManifestRow) -> str:
    """Hash gate: verify the master before any decoder or model runs."""
    try:
        actual = compute_sha256(row.audio_path)
    except OSError:
        raise StudyError("INVALID_ARGUMENT", "master audio could not be read") from None
    if actual.lower() != row.sha256.lower():
        raise StudyError("SOURCE_HASH_MISMATCH", "Master audio does not match manifest SHA-256")
    return actual


def compute_native_arms(row: ManifestRow, source_sha256: str, device: str, backend: Any) -> NativeArms:
    """Compute N0 (baseline) and N1 (vad_asr) via the compare command's code paths."""
    baseline = transcribe(
        audio_path=row.audio_path,
        channel_index=row.channel_index,
        device=device,
        backend=backend,
    )
    if baseline.source_sha256.lower() != source_sha256.lower():
        raise StudyError("SOURCE_HASH_MISMATCH", "Master audio changed during processing")
    # ponytail: decode+VAD glue mirrors cli.cmd_compare verbatim (its tests patch
    # cli-module names, so unifying now would churn them); N0's transcribe() call
    # re-decodes internally — same parity as compare. Unify both into one shared
    # prepare+arms helper when T2's profile reshapes this sequence for P0/PF/PD.
    audio = read_wav(row.audio_path)
    channel_samples = extract_channel(audio, row.channel_index)
    audio_16k = resample_to_16kHz(channel_samples, audio.sample_rate, audio.sample_width)
    speech_windows = get_speech_windows(audio_16k, threshold=STUDY_VAD_THRESHOLD)
    asr_windows = merge_asr_windows(speech_windows)
    cached_windows = transcribe_windows(audio_16k, asr_windows, backend)
    vad_asr = result_from_windows(audio_16k, source_sha256, cached_windows)
    return NativeArms(baseline=baseline, vad_asr=vad_asr)


def session_record(row: ManifestRow, arms: NativeArms) -> dict[str, Any]:
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
    for name, result in (("N0", arms.baseline), ("N1", arms.vad_asr)):
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
    """Run the native arms for every manifest row into a private output directory."""
    rows = load_manifest(manifest_path)
    if asr_backend is None:
        asr_backend = PhoWhisperBackend(device=device)
    for row in rows:
        source_sha256 = verify_source(row)
        arms = compute_native_arms(row, source_sha256, device, asr_backend)
        write_session_record(session_record(row, arms), out_dir)
    return 0
