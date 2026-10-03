"""Ablation arms for the ported ASR and VAD robustness behaviors.

Six arms share one pipeline and differ only in which ported behavior is active, so a
measured difference is attributable to that behavior: ``main`` is the shipped default,
``compatibility`` is the pre-port configuration, three arms enable one behavior each,
and ``combined`` enables all three. Scoring reuses the study path, so every arm is
scored in the spoken domain on the same references.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from say_transcribe.asr import AsrBackend, AsrResult, AsrSegment, make_asr_backend
from say_transcribe.asr import result_from_windows, transcribe_windows
from say_transcribe.evaluate import items_from_texts
from say_transcribe.manifest import load_manifest
from say_transcribe.study import (
    ReferenceIntervals,
    StudyError,
    coverage_metrics,
    load_reference_document,
    prepare_views,
    read_reference,
    reference_intervals,
    session_record,
    verify_source,
    windows_to_ms,
    write_session_record,
)
from say_transcribe.vad import (
    SAMPLE_RATE,
    Detector,
    get_speech_windows,
    merge_asr_windows,
)


@dataclass(frozen=True)
class Arm:
    """One ablation arm: which ported behavior is switched on."""

    name: str
    boost: bool
    retry: bool
    snap: bool


ARMS = (
    Arm("main", True, True, False),
    Arm("compatibility", False, False, False),
    Arm("quiet-vad", True, False, False),
    Arm("empty-retry", False, True, False),
    Arm("silence-cut", False, False, True),
    Arm("combined", True, True, True),
)
ABLATION_ARM_ORDER = tuple(arm.name for arm in ARMS)
ARM_BY_NAME = {arm.name: arm for arm in ARMS}
# Frozen for the same reason as the study threshold: a moving VAD threshold would make
# two runs of the same arm incomparable.
VAD_THRESHOLD = 0.2


@dataclass(frozen=True)
class ArmRun:
    """What one arm produced: its windows, its ASR result and its wall-clock cost."""

    windows: tuple[tuple[int, int], ...]
    result: AsrResult
    seconds: float


def arm_windows(
    audio_16k: np.ndarray,
    arm: Arm,
    *,
    detector: Detector | None = None,
    threshold: float = VAD_THRESHOLD,
) -> list[tuple[int, int]]:
    return get_speech_windows(
        audio_16k,
        detector=detector,
        threshold=threshold,
        boost=arm.boost,
        retry=arm.retry,
        snap=arm.snap,
    )


def run_arm(
    audio_16k: np.ndarray,
    source_sha256: str,
    backend: AsrBackend,
    arm: Arm,
    *,
    detector: Detector | None = None,
    threshold: float = VAD_THRESHOLD,
) -> ArmRun:
    """Transcribe the master through one arm's windows on the master timeline."""
    started = time.perf_counter()
    windows = arm_windows(audio_16k, arm, detector=detector, threshold=threshold)
    clips = merge_asr_windows(windows)
    results = transcribe_windows(audio_16k, clips, backend)
    result = result_from_windows(audio_16k, source_sha256, results)
    return ArmRun(tuple(windows), result, time.perf_counter() - started)


def result_counts(result: AsrResult) -> dict[str, int]:
    """Invalid and missing timing plus repetition suspicion, for one arm."""
    words = [word for segment in result.segments for word in segment.words]
    return {
        "words": len(words),
        "words_without_timing": sum(
            1 for word in words if word.start_ms is None or word.end_ms is None
        ),
        "utterances_without_timing": sum(
            1 for segment in result.segments if segment.start_ms is None or segment.end_ms is None
        ),
        "repetition_flags": sum(1 for warning in result.warnings if "REPETITION" in warning),
    }


def content_counts(segments: Sequence[AsrSegment]) -> dict[str, int]:
    """How much scorable content an arm produced, independent of the reference."""
    items = items_from_texts([segment.text for segment in segments])
    return {
        "words": len(items["words"]),
        "syllables": len(items["syllables"]),
        "chars": len(items["chars"]),
    }


def session_extras(run: ArmRun, reference: ReferenceIntervals) -> dict[str, Any]:
    """Coverage, invalid timing, produced content and runtime for one arm."""
    return {
        "coverage": coverage_metrics(windows_to_ms(run.windows), reference),
        "counts": result_counts(run.result),
        "content": content_counts(run.result.segments),
        "runtime_s": run.seconds,
    }


def _measured(record: dict[str, Any], *path: str) -> float:
    """Read one comparable number, loudly rather than by silent default."""
    value: Any = record
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise StudyError("INVALID_ARGUMENT", f"arm record is missing {'.'.join(path)}")
        value = value[key]
    if value is None:
        raise StudyError("INVALID_ARGUMENT", f"arm record has no {' '.join(path)}")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise StudyError(
            "INVALID_ARGUMENT", f"arm record has a non-numeric {'.'.join(path)}"
        ) from None


def promotion_verdict(
    sessions: dict[str, Any], primary: str = "p001"
) -> dict[str, dict[str, bool]]:
    """Ruling per arm against the primary session's ``main`` arm.

    An arm is promoted only when it is more accurate than ``main`` and regresses
    neither produced content, nor reference coverage, nor invalid word timing.
    """
    arms = (sessions.get(primary) or {}).get("arms") or {}
    baseline = arms.get("main")
    if not baseline:
        return {}
    verdict: dict[str, dict[str, bool]] = {}
    for name in ABLATION_ARM_ORDER[1:]:
        candidate = arms.get(name)
        if not candidate:
            continue
        accuracy_gain = _measured(candidate, "scores", "syer", "rate") < _measured(
            baseline, "scores", "syer", "rate"
        )
        content_ok = _measured(candidate, "content", "syllables") >= _measured(
            baseline, "content", "syllables"
        )
        coverage_ok = _measured(candidate, "coverage", "coverage") >= _measured(
            baseline, "coverage", "coverage"
        )
        timing_ok = _measured(candidate, "counts", "words_without_timing") <= _measured(
            baseline, "counts", "words_without_timing"
        )
        verdict[name] = {
            "promote": accuracy_gain and content_ok and coverage_ok and timing_ok,
            "accuracy_gain": accuracy_gain,
            "content_ok": content_ok,
            "coverage_ok": coverage_ok,
            "timing_ok": timing_ok,
        }
    return verdict


def _gpu_memory_bytes(device: str) -> int | None:
    """Peak GPU memory, or ``None`` on a CPU run where no device memory is involved."""
    if not device.startswith("cuda"):
        return None
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


def _arm_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "scores": entry.get("scores", {}),
            "coverage": entry.get("coverage"),
            "counts": entry.get("counts"),
            "content": entry.get("content"),
            "runtime_s": entry.get("runtime_s"),
        }
        for name, entry in record["arms"].items()
    }


def _session_summary(record: dict[str, Any], device: str) -> dict[str, Any]:
    """The redacted cohort-level view: no transcript text, no absolute paths."""
    return {
        "session_id": record["session_id"],
        "participant_id": record["participant_id"],
        "split": record["split"],
        "channel_index": record["channel_index"],
        "sha256": record["sha256"],
        "asr_revision": record["asr_revision"],
        "audio_seconds": record["audio_seconds"],
        "device": device,
        "gpu_memory_bytes": record["gpu_memory_bytes"],
        "arms": _arm_summary(record),
    }


def _number(value: Any, digits: int = 4) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    config = report["config"]
    lines = [
        "# Ablation: ported ASR and VAD robustness behaviors",
        "",
        f"- ASR model `{config['asr_model']}` at revision `{config['asr_revision']}`",
        f"- Device `{config['device']}`, channel {config['channel_index']}, "
        f"VAD threshold {config['vad_threshold']}",
        "- Uncapped error rates: SyER is the primary metric, CER and WER are reported "
        "alongside it.",
        "",
    ]
    for session_id, session in report["sessions"].items():
        gpu = session.get("gpu_memory_bytes")
        lines += [
            f"## {session_id} ({session['split']}, {_number(session['audio_seconds'], 1)} s)",
            "",
            f"Device `{session['device']}`, GPU memory "
            f"{'not applicable' if gpu is None else f'{gpu} bytes'}.",
            "",
            "| arm | SyER | CER | WER | coverage | words w/o timing "
            "| utterances w/o timing | repetition flags | syllables | runtime s |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name, arm in session["arms"].items():
            scores = arm["scores"]
            coverage = (arm["coverage"] or {}).get("coverage")
            counts = arm["counts"] or {}
            content = arm["content"] or {}
            lines.append(
                f"| {name} | {_number((scores.get('syer') or {}).get('rate'))} "
                f"| {_number((scores.get('cer') or {}).get('rate'))} "
                f"| {_number((scores.get('wer') or {}).get('rate'))} "
                f"| {_number(coverage)} | {counts.get('words_without_timing', '-')} "
                f"| {counts.get('utterances_without_timing', '-')} "
                f"| {counts.get('repetition_flags', '-')} "
                f"| {content.get('syllables', '-')} | {_number(arm['runtime_s'], 1)} |"
            )
        lines.append("")
    verdict = report["promotion"]
    if verdict:
        lines += ["## Promotion ruling", ""]
        for name, ruling in verdict.items():
            reasons = [
                label
                for label, key in (
                    ("accuracy", "accuracy_gain"),
                    ("content", "content_ok"),
                    ("coverage", "coverage_ok"),
                    ("timing", "timing_ok"),
                )
                if ruling[key]
            ]
            lines.append(
                f"- `{name}`: {'promote' if ruling['promote'] else 'do not promote'}"
                f" (holds: {', '.join(reasons) or 'nothing'})"
            )
        lines.append("")
    return "\n".join(lines)


def run_ablation(
    manifest_path: Path,
    out_dir: Path,
    *,
    model: str,
    revision: str,
    device: str = "cpu",
    backend: AsrBackend | None = None,
    detector: Detector | None = None,
) -> int:
    """Run every arm over every manifest row into a private output directory.

    Verifies each master hash before decoding it and again after the arms finish, so a
    master swapped mid-run cannot mix two sources into one record. Writes one private
    per-session record per row plus a redacted arm summary and its markdown view.
    """
    rows = load_manifest(Path(manifest_path))
    channels = {row.channel_index for row in rows}
    if len(channels) != 1:
        raise StudyError("INVALID_ARGUMENT", "ablation rows must share one channel_index")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if backend is None:
        backend = make_asr_backend(model, device=device, revision=revision)

    sessions: dict[str, Any] = {}
    for row in rows:
        source_sha256 = verify_source(row)
        views = prepare_views(row)
        reference = reference_intervals(load_reference_document(read_reference(row)))
        runs = {
            arm.name: run_arm(views.audio_16k, source_sha256, backend, arm, detector=detector)
            for arm in ARMS
        }
        extras = {name: session_extras(run, reference) for name, run in runs.items()}
        record = session_record(
            row,
            {name: run.result for name, run in runs.items()},
            extras=extras,
            order=ABLATION_ARM_ORDER,
        )
        record["audio_seconds"] = float(len(views.audio_16k)) / SAMPLE_RATE
        record["asr_model"] = model
        record["asr_revision"] = revision
        record["device"] = device
        record["gpu_memory_bytes"] = _gpu_memory_bytes(device)
        verify_source(row)
        write_session_record(record, out_dir / "records")
        sessions[row.session_id] = _session_summary(record, device)

    report = {
        "config": {
            "asr_model": model,
            "asr_revision": revision,
            "device": device,
            "channel_index": rows[0].channel_index,
            "vad_threshold": VAD_THRESHOLD,
            "arms": {
                arm.name: {"boost": arm.boost, "retry": arm.retry, "snap": arm.snap} for arm in ARMS
            },
        },
        "sessions": sessions,
        "promotion": promotion_verdict(sessions),
    }
    try:
        (out_dir / "ablation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / "ABLATION.md").write_text(render_markdown(report), encoding="utf-8")
    except OSError:
        raise StudyError("ABLATION_FAILED", "ablation report could not be written") from None
    return 0
