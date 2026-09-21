from __future__ import annotations


import hashlib
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from say_transcribe.annotations import (
    AnnotationValidationError,
    SessionAnnotations,
    TaskAnnotation,
)
from say_transcribe.audio import (
    Audio,
    AudioPreparationError,
    crop_task_clip,
    extract_channel,
    read_wav,
    resample_to_16kHz,
)


@dataclass(frozen=True)
class TaskClip:
    task_id: str
    path: Path
    asr_16k_mono: np.ndarray
    provenance: dict[str, Any]


def _write_mono_wav(
    path: Path, samples: np.ndarray, samples_rate: int, sample_width: int
) -> None:
    """Write a private mono PCM WAV without replacing an existing file."""
    created = False
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
        with os.fdopen(fd, "wb") as stream, wave.open(stream, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(sample_width)
            wf.setframerate(samples_rate)
            wf.writeframes(samples.tobytes())
    except (OSError, wave.Error):
        if created:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise AudioPreparationError("AUDIO_WRITE_FAILED", "Failed to write WAV clip") from None


def _redacted_provenance(
    annotations: SessionAnnotations,
    task: TaskAnnotation,
    audio: Audio,
    start_sample: int,
    end_sample: int,
) -> dict[str, Any]:
    """Provenance limited to pseudonymous IDs, stable metadata, and relative paths."""
    return {
        "relative_audio_path": annotations.source_audio,
        "source_sha256": annotations.source_sha256,
        "channel_index": annotations.channel_index,
        "start_sample": start_sample,
        "end_sample": end_sample,
        "sample_rate": audio.sample_rate,
        "task_id": task.task_id,
    }


def prepare_task_clips(
    annotations: SessionAnnotations,
    audio_root: Path,
    output_dir: Path,
) -> list[TaskClip]:
    """Verify master hash, extract selected channel, crop task bounds, and output native clips."""

    if annotations.review_status != "approved":
        raise AnnotationValidationError("UNAPPROVED_ANNOTATIONS", "review_status")
    if any(
        not task.task_id
        or not task.task_id.isprintable()
        or "/" in task.task_id
        or "\\" in task.task_id
        for task in annotations.tasks
    ):
        raise AnnotationValidationError("INVALID_ANNOTATIONS", "tasks[].task_id")

    if not annotations.source_audio.isprintable():
        raise AnnotationValidationError("INVALID_ANNOTATIONS", "source_audio")
    audio_path = (audio_root / annotations.source_audio).resolve()
    if not audio_path.is_relative_to(audio_root.resolve()) or not audio_path.is_file():
        raise AnnotationValidationError(
            "INVALID_ANNOTATIONS", "source_audio file not found"
        )

    # verify SHA256 before audio processing
    digest = hashlib.sha256()
    try:
        with audio_path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        raise AnnotationValidationError("INVALID_ANNOTATIONS", "source_audio unreadable") from None
    file_sha256 = digest.hexdigest()
    if file_sha256 != annotations.source_sha256:
        raise AnnotationValidationError(
            "SOURCE_HASH_MISMATCH", "source_sha256 does not match file"
        )

    # decode WAV audio
    audio = read_wav(audio_path)
    if annotations.sample_rate != audio.sample_rate:
        raise AnnotationValidationError("INVALID_ANNOTATIONS", "sample_rate")

    # Extract channel
    mono_samples = extract_channel(audio, annotations.channel_index)

    # Crop every task first so one task's invalid span aborts before any file is written
    cropped = [
        (task, *crop_task_clip(
            mono_samples, task.start_ms, task.end_ms, audio.sample_rate, audio.total_samples
        ))
        for task in annotations.tasks
    ]

    clip_paths = [output_dir / f"{task.task_id}.wav" for task in annotations.tasks]
    if len(set(clip_paths)) != len(clip_paths) or any(
        path.resolve() == audio_path or path.exists() or path.is_symlink()
        for path in clip_paths
    ):
        raise AnnotationValidationError("INVALID_ANNOTATIONS", "clip output path unavailable")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise AudioPreparationError("AUDIO_WRITE_FAILED", "Failed to create clip directory") from None
    clips: list[TaskClip] = []

    for (task, clip_samples, start_sample, end_sample), clip_path in zip(cropped, clip_paths):
        _write_mono_wav(clip_path, clip_samples, audio.sample_rate, audio.sample_width)

        # In-memory 16k Hz float32 view for ASR
        asr_view = resample_to_16kHz(
            clip_samples, audio.sample_rate, audio.sample_width
        )

        provenance = _redacted_provenance(annotations, task, audio, start_sample, end_sample)

        clips.append(
            TaskClip(
                task_id=task.task_id,
                path=clip_path,
                asr_16k_mono=asr_view,
                provenance=provenance,
            )
        )
    return clips
