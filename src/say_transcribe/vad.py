"""Speech activity windows for 16 kHz mono audio."""

from collections.abc import Callable, Iterable, Sequence
from functools import lru_cache

import numpy as np

SAMPLE_RATE = 16000
_MIN_SPEECH_SAMPLES = SAMPLE_RATE // 4
_MAX_SILENCE_SAMPLES = SAMPLE_RATE * 7 // 10
_PAD_SAMPLES = SAMPLE_RATE * 30 // 1000
_MAX_WINDOW_SAMPLES = SAMPLE_RATE * 20


class VADUnavailableError(Exception):
    """Silero VAD or its ONNX runtime could not run."""

    code = "VAD_UNAVAILABLE"

    def __init__(self) -> None:
        super().__init__(f"[{self.code}] VAD model or runtime unavailable")


Detector = Callable[[np.ndarray], Iterable[tuple[int, int]]]
DefaultDetector = Callable[[np.ndarray, float], Iterable[tuple[int, int]]]


@lru_cache(maxsize=1)
def _load_silero_detector() -> DefaultDetector:
    try:
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad

        model = load_silero_vad(onnx=True)
    except Exception as exc:
        raise VADUnavailableError() from exc

    def detect(audio: np.ndarray, threshold: float) -> list[tuple[int, int]]:
        spans = get_speech_timestamps(
            torch.from_numpy(audio),
            model,
            sampling_rate=SAMPLE_RATE,
            threshold=threshold,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
            speech_pad_ms=0,
        )
        return [(int(span["start"]), int(span["end"])) for span in spans]

    return detect


def merge_asr_windows(
    windows: Sequence[tuple[int, int]], *, max_samples: int = _MAX_WINDOW_SAMPLES
) -> list[tuple[int, int]]:
    """Greedily fold adjacent speech windows into <=20s ASR clips, regardless of the
    gap between them, so speech VAD missed inside a merged span still reaches ASR
    (WhisperX ``merge_chunks`` pattern)."""
    windows = list(windows)
    if not windows:
        return []
    merged: list[tuple[int, int]] = []
    current_start, current_end = windows[0]
    for start, end in windows[1:]:
        if end - current_start <= max_samples:
            current_end = end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


def get_speech_windows(
    audio_16k: np.ndarray,
    *,
    detector: Detector | None = None,
    threshold: float = 0.2,
) -> list[tuple[int, int]]:
    """Return padded, merged 16 kHz speech windows as half-open sample spans."""
    if not 0 < threshold < 1:
        raise ValueError("VAD threshold must be between 0 and 1")
    audio = np.asarray(audio_16k, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("audio_16k must be a one-dimensional mono array")
    if not np.isfinite(audio).all():
        raise ValueError("audio_16k must contain only finite samples")
    if len(audio) == 0:
        return []

    try:
        detected = list(
            _load_silero_detector()(audio, threshold)
            if detector is None
            else detector(audio)
        )
        bounded: list[tuple[int, int]] = []
        for span in detected:
            start, end = span
            start = max(0, min(len(audio), int(start)))
            end = max(0, min(len(audio), int(end)))
            if end - start >= _MIN_SPEECH_SAMPLES:
                bounded.append((start, end))
    except VADUnavailableError:
        raise
    except Exception as exc:
        raise VADUnavailableError() from exc

    if not bounded:
        return []

    bounded.sort()
    merged: list[tuple[int, int]] = []
    for start, end in bounded:
        if merged and start - merged[-1][1] <= _MAX_SILENCE_SAMPLES:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    windows: list[tuple[int, int]] = []
    for start, end in merged:
        start = max(0, start - _PAD_SAMPLES)
        end = min(len(audio), end + _PAD_SAMPLES)
        while start < end:
            window_end = min(start + _MAX_WINDOW_SAMPLES, end)
            windows.append((start, window_end))
            start = window_end
    return windows
