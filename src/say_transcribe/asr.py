from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from say_transcribe.audio import extract_channel, read_wav, resample_to_16kHz


class AsrError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class WordTiming:
    word: str
    start_ms: int | None
    end_ms: int | None


@dataclass(frozen=True)
class AsrSegment:
    start_ms: int | None
    end_ms: int | None
    text: str
    words: tuple[WordTiming, ...]


@dataclass(frozen=True)
class AsrResult:
    source_sha256: str
    segments: tuple[AsrSegment, ...]
    warnings: tuple[str, ...]


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file in chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _valid_span(start: Any, end: Any, duration_ms: int) -> tuple[int, int] | None:
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, (int, float))
        or not isinstance(end, (int, float))
        or not math.isfinite(start)
        or not math.isfinite(end)
    ):
        return None
    start_ms, end_ms = round(start), round(end)
    if start_ms < 0 or end_ms <= start_ms or end_ms > duration_ms:
        return None
    return start_ms, end_ms


def _seconds_to_ms(timestamp: Any) -> float | None:
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, float))
        or not math.isfinite(timestamp)
    ):
        return None
    return timestamp * 1000


def _is_lexical_word(word: Any) -> bool:
    return any(character.isalnum() for character in str(word))


def _is_suspected_repetition(words: Sequence[dict[str, Any]]) -> bool:
    """Flag a decode that looks like a Whisper repetition loop or an unrecognized
    token, rather than trusting it as real speech (real p001 pilot output: 59-65%
    of ASR syllables sat inside such loops on quiet audio)."""
    texts = [str(w.get("word", "")).strip().lower() for w in words]
    texts = [t for t in texts if t]
    if not texts:
        return False
    # Whisper word-level chunks attach trailing punctuation to the word itself
    # (e.g. "unk."), so match after stripping it rather than requiring an exact hit.
    if any(t.strip(".,!?…") in ("unk", "<unk>") for t in texts):
        return True
    return len(texts) >= 20 and len(set(texts)) / len(texts) < 0.3


def _has_complete_lexical_timing(words: Sequence[dict[str, Any]]) -> bool:
    lexical_words = [word for word in words if _is_lexical_word(word.get("word", ""))]
    return bool(lexical_words) and all(
        word.get("start_ms") is not None and word.get("end_ms") is not None
        for word in lexical_words
    )


def _normalize_segment(
    segment: dict[str, Any], duration_ms: int
) -> dict[str, Any]:
    segment_span = _valid_span(
        segment.get("start_ms"), segment.get("end_ms"), duration_ms
    )
    words = []
    timed_spans = []
    previous_start: int | None = None
    lexical_words_timed = True
    lexical_word_count = 0
    for word in segment.get("words", []):
        is_lexical = _is_lexical_word(word.get("word", ""))
        lexical_word_count += is_lexical
        span = _valid_span(word.get("start_ms"), word.get("end_ms"), duration_ms)
        if span is not None and segment_span is not None:
            if span[0] < segment_span[0] or span[1] > segment_span[1]:
                span = None
        if span is not None and previous_start is not None and span[0] < previous_start:
            span = None
        if span is None:
            start_ms = end_ms = None
            if is_lexical:
                lexical_words_timed = False
        else:
            start_ms, end_ms = span
            timed_spans.append(span)
            previous_start = start_ms
        words.append({**word, "start_ms": start_ms, "end_ms": end_ms})

    if segment_span is None and timed_spans and lexical_word_count and lexical_words_timed:
        segment_span = min(start for start, _ in timed_spans), max(end for _, end in timed_spans)
    return {
        **segment,
        "start_ms": None if segment_span is None else segment_span[0],
        "end_ms": None if segment_span is None else segment_span[1],
        "words": words,
    }


class PhoWhisperBackend:
    """Lazy-loaded PhoWhisper backend using HuggingFace transformers."""

    def __init__(
        self,
        model_id: str = "vinai/phowhisper-medium",
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._pipe: Any = None

    def load(self) -> None:
        if self.device == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    raise AsrError("GPU_UNAVAILABLE", "CUDA device requested but not available")
            except ImportError:
                raise AsrError("GPU_UNAVAILABLE", "PyTorch with CUDA not available") from None

        try:
            from transformers import pipeline

            device_arg = 0 if self.device == "cuda" else -1
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                device=device_arg,
                return_timestamps="word",
            )
            if self.device == "cuda":
                # ponytail: fp16 halves the word-timestamp attention cache (measured
                # 10.9GB -> 1.45GB per 30s window); model.half() is unambiguous, unlike
                # the deprecated torch_dtype kwarg which silently stayed fp32
                self._pipe.model.half()
        except AsrError:
            raise
        except Exception:
            raise AsrError("MODEL_UNAVAILABLE", "Failed to load PhoWhisper model") from None

    def transcribe_audio(self, audio_16k_mono: np.ndarray) -> Sequence[dict[str, Any]]:
        """Run ASR on 16kHz float32 mono audio array.

        Returns a list of segment dicts with 'start_ms', 'end_ms', 'text', and 'words'.
        """
        if self._pipe is None:
            self.load()

        # ponytail: 20s windows + 400-token cap bound the word-timestamp attention cache
        # (steps x layers x heads x src, fp32): dense 30s speech reached ~9GB reserved;
        # worst case here is ~0.8GB cache. No overlap; add stride+dedupe if boundary loss
        # shows in eval.
        chunk_len = int(20.0 * 16000)
        results: list[dict[str, Any]] = []
        try:
            for start in range(0, max(len(audio_16k_mono), 1), chunk_len):
                chunk = audio_16k_mono[start : start + chunk_len]
                if not len(chunk):
                    break
                max_tokens = min(400, max(32, math.ceil(len(chunk) / 16000 * 20)))
                result = self._pipe(
                    {"raw": chunk, "sampling_rate": 16000},
                    max_new_tokens=max_tokens,
                    # Standard Whisper decoding heuristic (openai/whisper decode_options):
                    # retry at higher temperature when the decode looks degenerate, instead
                    # of keeping a repetition loop or low-confidence guess.
                    generate_kwargs={
                        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                        "compression_ratio_threshold": 2.4,
                        "logprob_threshold": -1.0,
                        "no_speech_threshold": 0.6,
                        "condition_on_prev_tokens": False,
                    },
                )
                offset_ms = int(round(start / 16000 * 1000))
                chunk_duration_ms = round(len(chunk) * 1000 / 16000)
                for seg in self._parse_pipeline_output(result):
                    seg = _normalize_segment(seg, chunk_duration_ms)
                    results.append(self._offset_segment(seg, offset_ms))
                if self.device == "cuda":
                    import torch

                    torch.cuda.empty_cache()
            return results
        except Exception:
            raise AsrError("MODEL_UNAVAILABLE", "ASR inference execution failed") from None

    @staticmethod
    def _offset_segment(seg: dict[str, Any], offset_ms: int) -> dict[str, Any]:
        if not offset_ms:
            return seg
        words = [
            {
                **w,
                "start_ms": None if w.get("start_ms") is None else w["start_ms"] + offset_ms,
                "end_ms": None if w.get("end_ms") is None else w["end_ms"] + offset_ms,
            }
            for w in seg.get("words", [])
        ]
        return {
            **seg,
            "start_ms": None
            if seg.get("start_ms") is None
            else seg["start_ms"] + offset_ms,
            "end_ms": None
            if seg.get("end_ms") is None
            else seg["end_ms"] + offset_ms,
            "words": words,
        }

    @staticmethod
    def _finalize_segment(words: list[dict[str, Any]]) -> dict[str, Any]:
        """Build a segment dict from buffered word chunks. A suspected repetition
        loop or unrecognized token is replaced with CHAT's `xxx` (unintelligible
        speech) marker instead of being emitted as if it were real, timed text."""
        suspected = _is_suspected_repetition(words)
        if suspected:
            words = [{"word": "xxx", "start_ms": None, "end_ms": None}]
        starts = [w["start_ms"] for w in words if w.get("start_ms") is not None]
        ends = [w["end_ms"] for w in words if w.get("end_ms") is not None]
        segment_start = min(starts) if starts and _has_complete_lexical_timing(words) else None
        segment_end = max(ends) if segment_start is not None else None
        segment = {
            "start_ms": segment_start,
            "end_ms": segment_end,
            "text": " ".join(w["word"] for w in words),
            "words": words,
        }
        if suspected:
            segment["repetition_suspected"] = True
        return segment

    @staticmethod
    def _parse_pipeline_output(output: Any) -> Sequence[dict[str, Any]]:
        """Normalize transformers pipeline output into structured segment dicts."""
        if isinstance(output, dict) and "segments" in output:
            return output["segments"]

        # Default chunk parsing from transformers
        chunks = output.get("chunks", []) if isinstance(output, dict) else []
        if not chunks:
            text = output.get("text", "") if isinstance(output, dict) else ""
            if not text:
                return []
            return [{"start_ms": None, "end_ms": None, "text": text.strip(), "words": []}]

        # Group word chunks into utterance segments if chunks are words
        # In transformers with return_timestamps='word', each chunk is a word with timestamp (start, end)
        segments: list[dict[str, Any]] = []
        current_words: list[dict[str, Any]] = []
        current_start: int | None = None
        current_end: int | None = None

        for chunk in chunks:
            chunk_text = chunk.get("text", "").strip()
            ts = chunk.get("timestamp")
            w_start: int | None = None
            w_end: int | None = None
            if ts is not None and len(ts) == 2:
                span = _valid_span(
                    _seconds_to_ms(ts[0]), _seconds_to_ms(ts[1]), 2**63 - 1
                )
                if span is not None:
                    w_start, w_end = span

            # Check if this word indicates a segment break (e.g. pause > 700ms or terminal punct)
            if current_words and w_start is not None and current_end is not None:
                if w_start - current_end > 700:
                    # Flush current segment
                    segments.append(PhoWhisperBackend._finalize_segment(current_words))
                    current_words = []
                    current_start = None

            if current_start is None and w_start is not None:
                current_start = w_start
            if w_end is not None:
                current_end = w_end

            current_words.append({
                "word": chunk_text,
                "start_ms": w_start,
                "end_ms": w_end,
            })

            # Break on sentence-ending punctuation
            if chunk_text.endswith((".", "!", "?")):
                segments.append(PhoWhisperBackend._finalize_segment(current_words))
                current_words = []
                current_start = None
                current_end = None

        if current_words:
            segments.append(PhoWhisperBackend._finalize_segment(current_words))

        return segments


def transcribe(
    audio_path: Path,
    channel_index: int = 0,
    device: str = "cpu",
    backend: PhoWhisperBackend | None = None,
) -> AsrResult:
    """Transcribe a master WAV file on the declared channel.

    Loads audio, extracts channel without downmixing, resamples in-memory to 16 kHz,
    and runs PhoWhisper ASR to produce timed segments and word timestamps.
    """
    source_sha256 = compute_sha256(audio_path)
    audio = read_wav(audio_path)
    channel_samples = extract_channel(audio, channel_index)
    audio_16k = resample_to_16kHz(channel_samples, audio.sample_rate, audio.sample_width)

    if backend is None:
        backend = PhoWhisperBackend(device=device)

    raw_segments = backend.transcribe_audio(audio_16k)

    parsed_segments: list[AsrSegment] = []
    warnings: list[str] = []

    for i, seg in enumerate(raw_segments, start=1):
        seg_words: list[WordTiming] = []
        duration_ms = round(len(audio_16k) * 1000 / 16000)
        seg = _normalize_segment(seg, duration_ms)
        words_data = seg.get("words", [])
        has_valid_word_timing = bool(words_data)

        for w in words_data:
            w_text = w.get("word", "")
            w_start = w.get("start_ms")
            w_end = w.get("end_ms")
            if w_start is None or w_end is None:
                has_valid_word_timing = False
            seg_words.append(WordTiming(word=w_text, start_ms=w_start, end_ms=w_end))

        if not has_valid_word_timing:
            warnings.append(f"CHAT_WORD_TIMING_UNAVAILABLE:{i}")

        parsed_segments.append(
            AsrSegment(
                start_ms=seg.get("start_ms"),
                end_ms=seg.get("end_ms"),
                text=str(seg.get("text", "")),
                words=tuple(seg_words),
            )
        )
        if seg.get("start_ms") is None or seg.get("end_ms") is None:
            warnings.append(f"CHAT_UTTERANCE_TIMING_UNAVAILABLE:{i}")
        if seg.get("repetition_suspected"):
            warnings.append(f"ASR_REPETITION_SUSPECTED:{i}")

    return AsrResult(
        source_sha256=source_sha256,
        segments=tuple(parsed_segments),
        warnings=tuple(warnings),
    )


WindowResults = tuple[tuple[int, int, tuple[dict[str, Any], ...]], ...]


def transcribe_windows(
    audio_16k: np.ndarray,
    windows: Sequence[tuple[int, int]],
    backend: PhoWhisperBackend,
) -> WindowResults:
    """Run ASR once per VAD window, retaining sample offsets for both variants."""
    results: list[tuple[int, int, tuple[dict[str, Any], ...]]] = []
    previous_end = 0
    for start, end in windows:
        if start < previous_end or end <= start or end > len(audio_16k) or end - start > 20 * 16000:
            raise AsrError("INVALID_VAD_WINDOWS", "Speech windows must be ordered and in range")
        results.append((start, end, tuple(backend.transcribe_audio(audio_16k[start:end]))))
        previous_end = end
    return tuple(results)


def result_from_windows(
    audio_16k: np.ndarray,
    source_sha256: str,
    windows: WindowResults,
    *,
    aligner: Callable[
        [np.ndarray, Sequence[str]], Sequence[tuple[int | None, int | None]]
    ] | None = None,
) -> AsrResult:
    """Build an ASR result from cached speech-window output, optionally aligned."""
    raw_segments: list[dict[str, Any]] = []
    for start, end, window_segments in windows:
        segments: Sequence[dict[str, Any]] = window_segments
        if aligner is not None:
            words = [
                str(word.get("word", ""))
                for segment in window_segments
                for word in (segment.get("words") or [{"word": token} for token in str(segment.get("text", "")).split()])
            ]
            spans = aligner(audio_16k[start:end], words)
            if len(spans) != len(words):
                spans = ((None, None),) * len(words)
            chunks = [
                {
                    "text": word,
                    "timestamp": (
                        None if span[0] is None else span[0] / 1000,
                        None if span[1] is None else span[1] / 1000,
                    ),
                }
                for word, span in zip(words, spans)
            ]
            segments = PhoWhisperBackend._parse_pipeline_output({"chunks": chunks})

        window_duration_ms = round((end - start) * 1000 / 16000)
        offset_ms = round(start * 1000 / 16000)
        for segment in segments:
            bounded = _normalize_segment(segment, window_duration_ms)
            raw_segments.append(PhoWhisperBackend._offset_segment(bounded, offset_ms))

    duration_ms = round(len(audio_16k) * 1000 / 16000)
    parsed_segments: list[AsrSegment] = []
    warnings: list[str] = []
    for i, raw in enumerate(raw_segments, start=1):
        segment = _normalize_segment(raw, duration_ms)
        words = tuple(
            WordTiming(str(word.get("word", "")), word.get("start_ms"), word.get("end_ms"))
            for word in segment.get("words", ())
        )
        if not words or any(word.start_ms is None or word.end_ms is None for word in words):
            warnings.append(f"CHAT_WORD_TIMING_UNAVAILABLE:{i}")
        parsed_segments.append(
            AsrSegment(segment.get("start_ms"), segment.get("end_ms"), str(segment.get("text", "")), words)
        )
        if segment.get("start_ms") is None or segment.get("end_ms") is None:
            warnings.append(f"CHAT_UTTERANCE_TIMING_UNAVAILABLE:{i}")
        if segment.get("repetition_suspected"):
            warnings.append(f"ASR_REPETITION_SUSPECTED:{i}")
    return AsrResult(source_sha256, tuple(parsed_segments), tuple(warnings))
