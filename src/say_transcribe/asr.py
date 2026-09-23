from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Sequence

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
    start_ms: int
    end_ms: int
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

        try:
            result = self._pipe({"raw": audio_16k_mono, "sampling_rate": 16000})
            return self._parse_pipeline_output(result)
        except Exception:
            raise AsrError("MODEL_UNAVAILABLE", "ASR inference execution failed") from None

    def _parse_pipeline_output(self, output: Any) -> Sequence[dict[str, Any]]:
        """Normalize transformers pipeline output into structured segment dicts."""
        if isinstance(output, dict) and "segments" in output:
            return output["segments"]

        # Default chunk parsing from transformers
        chunks = output.get("chunks", []) if isinstance(output, dict) else []
        if not chunks:
            text = output.get("text", "") if isinstance(output, dict) else ""
            if not text:
                return []
            return [{"start_ms": 0, "end_ms": 0, "text": text.strip(), "words": []}]

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
            if ts is not None and len(ts) == 2 and ts[0] is not None and ts[1] is not None:
                w_start = int(round(ts[0] * 1000.0))
                w_end = int(round(ts[1] * 1000.0))

            # Check if this word indicates a segment break (e.g. pause > 700ms or terminal punct)
            if current_words and w_start is not None and current_end is not None:
                if w_start - current_end > 700:
                    # Flush current segment
                    seg_text = " ".join(w["word"] for w in current_words)
                    segments.append({
                        "start_ms": current_start or 0,
                        "end_ms": current_end or 0,
                        "text": seg_text,
                        "words": current_words,
                    })
                    current_words = []
                    current_start = None

            if current_start is None:
                current_start = w_start if w_start is not None else 0
            if w_end is not None:
                current_end = w_end

            current_words.append({
                "word": chunk_text,
                "start_ms": w_start,
                "end_ms": w_end,
            })

            # Break on sentence-ending punctuation
            if chunk_text.endswith((".", "!", "?")):
                seg_text = " ".join(w["word"] for w in current_words)
                segments.append({
                    "start_ms": current_start or 0,
                    "end_ms": current_end or (current_start or 0),
                    "text": seg_text,
                    "words": current_words,
                })
                current_words = []
                current_start = None
                current_end = None

        if current_words:
            seg_text = " ".join(w["word"] for w in current_words)
            segments.append({
                "start_ms": current_start or 0,
                "end_ms": current_end or (current_start or 0),
                "text": seg_text,
                "words": current_words,
            })

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
                start_ms=int(seg.get("start_ms", 0)),
                end_ms=int(seg.get("end_ms", 0)),
                text=str(seg.get("text", "")),
                words=tuple(seg_words),
            )
        )

    return AsrResult(
        source_sha256=source_sha256,
        segments=tuple(parsed_segments),
        warnings=tuple(warnings),
    )
