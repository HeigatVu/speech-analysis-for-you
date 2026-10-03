"""Speech activity windows for 16 kHz mono audio."""

from collections.abc import Callable, Iterable, Sequence
from functools import lru_cache

import numpy as np

SAMPLE_RATE = 16000
_MIN_SPEECH_SAMPLES = SAMPLE_RATE // 4
_MAX_SILENCE_SAMPLES = SAMPLE_RATE * 7 // 10
_PAD_SAMPLES = SAMPLE_RATE * 30 // 1000
_MAX_WINDOW_SAMPLES = SAMPLE_RATE * 20
_DEFAULT_MIN_SPEECH_MS = 250
_RETRY_MIN_SPEECH_MS = 150
# Quiet recordings (-33 to -55 LUFS) starve Silero, so detection sees a copy lifted to a
# -23 dBFS peak (sherpa-vietnamese-asr). Gain is capped at 40 dB so near-silence is not
# amplified into false speech. ASR and the audio on disk never see the boosted copy.
_BOOST_PEAK = 0.071
_MAX_BOOST_GAIN = 100.0


class VADUnavailableError(Exception):
    """Silero VAD or its ONNX runtime could not run."""

    code = "VAD_UNAVAILABLE"

    def __init__(self) -> None:
        super().__init__(f"[{self.code}] VAD model or runtime unavailable")


Detector = Callable[[np.ndarray], Iterable[tuple[int, int]]]
DefaultDetector = Callable[..., Iterable[tuple[int, int]]]


@lru_cache(maxsize=1)
def _load_silero_detector() -> DefaultDetector:
    try:
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad

        model = load_silero_vad(onnx=True)
    except Exception as exc:
        raise VADUnavailableError() from exc

    def detect(
        audio: np.ndarray, threshold: float, min_speech_ms: int = _DEFAULT_MIN_SPEECH_MS
    ) -> list[tuple[int, int]]:
        spans = get_speech_timestamps(
            torch.from_numpy(audio),
            model,
            sampling_rate=SAMPLE_RATE,
            threshold=threshold,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=100,
            speech_pad_ms=0,
        )
        return [(int(span["start"]), int(span["end"])) for span in spans]

    return detect


def _boost_for_detection(audio: np.ndarray) -> np.ndarray:
    peak = float(np.abs(audio).max())
    if peak == 0 or peak >= _BOOST_PEAK:
        return audio
    return audio * np.float32(min(_BOOST_PEAK / peak, _MAX_BOOST_GAIN))


def _bound_spans(
    spans: Iterable[tuple[int, int]], length: int, min_samples: int
) -> list[tuple[int, int]]:
    bounded: list[tuple[int, int]] = []
    for start, end in spans:
        start = max(0, min(length, int(start)))
        end = max(0, min(length, int(end)))
        if end - start >= min_samples:
            bounded.append((start, end))
    return bounded


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

    detect_audio = _boost_for_detection(audio)
    try:
        if detector is not None:
            bounded = _bound_spans(detector(detect_audio), len(audio), _MIN_SPEECH_SAMPLES)
        else:
            detect = _load_silero_detector()
            bounded = _bound_spans(detect(detect_audio, threshold), len(audio), _MIN_SPEECH_SAMPLES)
            if not bounded:
                bounded = _bound_spans(
                    detect(detect_audio, threshold / 2, _RETRY_MIN_SPEECH_MS),
                    len(audio),
                    SAMPLE_RATE * _RETRY_MIN_SPEECH_MS // 1000,
                )
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
