"""Benchmark ASR models and diarization backends against reference transcripts.

Every report row carries a pseudonymous session id, scores, and stable error
codes only — never paths, transcript text, or raw exception details.
"""

import json
from pathlib import Path
from typing import Any, Callable, Sequence

from say_transcribe.asr import (
    AsrError,
    compute_sha256,
    verify_source_sha256,
    make_asr_backend,
    transcribe,
)
from say_transcribe.audio import (
    AudioPreparationError,
    extract_channel,
    read_wav,
    resample_to_16kHz,
)
from say_transcribe.diarize import PyannoteBackend, WavlmClusterBackend, assign_speakers
from say_transcribe.evaluate import (
    compute_der,
    extract_session_items,
    items_from_texts,
    score_uncapped,
)
from say_transcribe.manifest import ManifestRow

# SPEC benchmark design: the ASR comparison pins vinai/phowhisper-large (the CLI
# default stays phowhisper-medium) against the new Qwen3-ASR backend.
ASR_BENCHMARK_MODELS = ("phowhisper-large", "qwen3-asr")

# The diarization comparison runs both backends over one fixed ASR front-end so
# the numbers isolate the diarization variable.
DIARIZATION_BENCHMARK_BACKENDS = ("pyannote", "wavlm")
DIARIZATION_ASR_MODEL = "phowhisper-large"

BenchmarkBackendFactory = Callable[[str, str], Any]


def _default_asr_factory(model: str, device: str) -> Any:
    # The benchmark scores text only: word timestamps decode identical text but cost
    # ~4.4GB more VRAM per window (measured 2026-10-03), which a 12GB card cannot hold
    # for phowhisper-large. Ask for segment timestamps.
    return make_asr_backend(model, device=device, timestamps="segment")


def _release_device_memory(device: str) -> None:
    """Return freed accelerator memory to the allocator between pinned models."""
    if not device.startswith("cuda"):
        return
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _default_diarizer_factory(name: str, device: str) -> Any:
    return WavlmClusterBackend(device=device) if name == "wavlm" else PyannoteBackend(device=device)


def _error_row(session_id: str, key: str, model_key: str, code: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        model_key: key,
        "status": "error",
        "code": code,
    }


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def run_asr_benchmark(
    rows: Sequence[ManifestRow],
    device: str = "cpu",
    backend_factory: BenchmarkBackendFactory | None = None,
) -> dict[str, Any]:
    """Score each pinned ASR model against each session's reference transcript."""
    factory = backend_factory or _default_asr_factory
    report_rows: list[dict[str, Any]] = []

    for row in rows:
        if compute_sha256(row.audio_path) != row.sha256:
            report_rows.append(
                _error_row(row.session_id, None, "model", "SOURCE_HASH_MISMATCH")
            )
            continue
        try:
            gold = extract_session_items(row.reference_path.read_text(encoding="utf-8"))
        except OSError:
            report_rows.append(_error_row(row.session_id, None, "model", "REFERENCE_UNREADABLE"))
            continue

        for model in ASR_BENCHMARK_MODELS:
            entry: dict[str, Any] = {"session_id": row.session_id, "model": model}
            backend: Any = None
            try:
                verify_source_sha256(row.audio_path, row.sha256)
                backend = factory(model, device)
                result = transcribe(
                    audio_path=row.audio_path,
                    channel_index=row.channel_index,
                    device=device,
                    backend=backend,
                    expected_sha256=row.sha256,
                )
                scores = score_uncapped(
                    gold, items_from_texts([segment.text for segment in result.segments])
                )
                entry.update(
                    status="ok",
                    wer=scores["wer"]["rate"],
                    cer=scores["cer"]["rate"],
                    syer=scores["syer"]["rate"],
                )
            except (AsrError, AudioPreparationError) as error:
                entry.update(status="error", code=error.code)
            finally:
                # Free one pinned model before the next loads; two coexisting
                # backends OOM the shared GPU (measured crash at 6.6 GiB).
                backend = None
                _release_device_memory(device)
            report_rows.append(entry)

    return {"task": "asr", "rows": report_rows, "aggregate": _asr_aggregate(report_rows)}


def _asr_aggregate(report_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for model in ASR_BENCHMARK_MODELS:
        ok = [r for r in report_rows if r.get("model") == model and r["status"] == "ok"]
        aggregate[model] = {
            "wer_mean": _mean([r["wer"] for r in ok]),
            "cer_mean": _mean([r["cer"] for r in ok]),
            "cells_scored": len(ok),
        }
    return aggregate


def run_diarization_benchmark(
    rows: Sequence[ManifestRow],
    device: str = "cpu",
    asr_factory: BenchmarkBackendFactory | None = None,
    diarizer_factory: BenchmarkBackendFactory | None = None,
) -> dict[str, Any]:
    """Score each diarization backend's PAR/INV turns against reference timing."""
    make_asr = asr_factory or _default_asr_factory
    make_diarizer = diarizer_factory or _default_diarizer_factory
    report_rows: list[dict[str, Any]] = []

    for row in rows:
        if compute_sha256(row.audio_path) != row.sha256:
            report_rows.append(
                _error_row(row.session_id, None, "diarizer", "SOURCE_HASH_MISMATCH")
            )
            continue
        try:
            gold = extract_session_items(row.reference_path.read_text(encoding="utf-8"))
        except OSError:
            report_rows.append(
                _error_row(row.session_id, None, "diarizer", "REFERENCE_UNREADABLE")
            )
            continue

        asr_backend: Any = None
        try:
            verify_source_sha256(row.audio_path, row.sha256)
            asr_backend = make_asr(DIARIZATION_ASR_MODEL, device)
            result = transcribe(
                audio_path=row.audio_path,
                channel_index=row.channel_index,
                device=device,
                backend=asr_backend,
                expected_sha256=row.sha256,
            )
            audio = read_wav(row.audio_path)
            channel_samples = extract_channel(audio, row.channel_index)
            audio_16k = resample_to_16kHz(
                channel_samples, audio.sample_rate, audio.sample_width
            )
        except (AsrError, AudioPreparationError) as error:
            for name in DIARIZATION_BENCHMARK_BACKENDS:
                report_rows.append(
                    _error_row(row.session_id, name, "diarizer", error.code)
                )
            continue
        finally:
            asr_backend = None
            _release_device_memory(device)

        for name in DIARIZATION_BENCHMARK_BACKENDS:
            entry: dict[str, Any] = {"session_id": row.session_id, "diarizer": name}
            backend = None
            try:
                backend = make_diarizer(name, device)
                if name == "wavlm":
                    turns = backend.diarize(audio_16k, segments=result.segments)
                else:
                    turns = backend.diarize(audio_16k)
                speakers = assign_speakers(result.segments, turns)
                hyp_intervals = [
                    (
                        turn.start_ms,
                        turn.end_ms,
                        (speakers.cluster_to_role or {}).get(turn.cluster_id, "UNKNOWN"),
                    )
                    for turn in turns
                ]
                entry.update(
                    status="ok",
                    der=compute_der(gold["intervals"], hyp_intervals),
                    n_speakers=len({turn.cluster_id for turn in turns}),
                    n_turns=len(turns),
                )
            except (AsrError, AudioPreparationError) as error:
                entry.update(status="error", code=error.code)
            finally:
                backend = None
                _release_device_memory(device)
            report_rows.append(entry)

    return {
        "task": "diarization",
        "rows": report_rows,
        "aggregate": _diarization_aggregate(report_rows),
    }


def _diarization_aggregate(report_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for name in DIARIZATION_BENCHMARK_BACKENDS:
        ok = [r for r in report_rows if r.get("diarizer") == name and r["status"] == "ok"]
        aggregate[name] = {
            "der_mean": _mean([r["der"] for r in ok]),
            "cells_scored": len(ok),
        }
    return aggregate


def render_markdown(report: dict[str, Any]) -> str:
    """Human-readable report: per-session rows first, then the aggregate."""
    task = report["task"]
    lines = [f"# Benchmark report: {task}", ""]
    if task == "asr":
        lines += ["| Session | Model | WER | CER | SyER | Status |", "| --- | --- | --- | --- | --- | --- |"]
        for row in report["rows"]:
            if row["status"] == "ok":
                lines.append(
                    f"| {row['session_id']} | {row['model']} | {row['wer']:.4f} "
                    f"| {row['cer']:.4f} | {row['syer']:.4f} | ok |"
                )
            else:
                lines.append(
                    f"| {row['session_id']} | {row.get('model') or '-'} | - | - | - "
                    f"| {row['code']} |"
                )
        lines += ["", "## Aggregate", "", "| Model | Mean WER | Mean CER | Cells scored |", "| --- | --- | --- | --- |"]
        for model, agg in report["aggregate"].items():
            wer = f"{agg['wer_mean']:.4f}" if agg["wer_mean"] is not None else "-"
            cer = f"{agg['cer_mean']:.4f}" if agg["cer_mean"] is not None else "-"
            lines.append(f"| {model} | {wer} | {cer} | {agg['cells_scored']} |")
    else:
        lines += [
            "| Session | Diarizer | DER | Speakers | Turns | Status |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in report["rows"]:
            if row["status"] == "ok":
                lines.append(
                    f"| {row['session_id']} | {row['diarizer']} | {row['der']:.4f} "
                    f"| {row['n_speakers']} | {row['n_turns']} | ok |"
                )
            else:
                lines.append(
                    f"| {row['session_id']} | {row.get('diarizer') or '-'} | - | - | - "
                    f"| {row['code']} |"
                )
        lines += ["", "## Aggregate", "", "| Diarizer | Mean DER | Cells scored |", "| --- | --- | --- |"]
        for name, agg in report["aggregate"].items():
            der = f"{agg['der_mean']:.4f}" if agg["der_mean"] is not None else "-"
            lines.append(f"| {name} | {der} | {agg['cells_scored']} |")
    lines.append("")
    return "\n".join(lines)


def write_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Write the JSON data report and its Markdown rendering."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"benchmark-{report['task']}.json"
    md_path = out_dir / f"benchmark-{report['task']}.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
