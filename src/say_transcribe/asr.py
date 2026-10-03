from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from say_transcribe.audio import extract_channel, read_wav, resample_to_16kHz
from say_transcribe.vad import _SNAP_FRAME_SAMPLES, _SNAP_SEARCH_SAMPLES, _quietest_cut


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


def verify_source_sha256(path: Path, expected_sha256: str) -> str:
    """Verify an approved master before decoding, without exposing file details."""
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in expected_sha256
    ):
        raise AsrError("INVALID_ARGUMENT", "Expected SHA-256 must be 64 hexadecimal characters")
    try:
        actual = compute_sha256(path)
    except OSError:
        raise AsrError("SOURCE_UNREADABLE", "Master audio could not be read") from None
    if actual != expected_sha256.lower():
        raise AsrError("SOURCE_HASH_MISMATCH", "Master audio does not match approved SHA-256")
    return actual


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


# Same set the repetition guard already strips: Whisper's word chunks carry sentence
# punctuation on the word itself. CHAT markup characters are deliberately absent, so a
# token like `&-ờ` or `[/]` can never be reshaped here.
_ASR_EDGE_PUNCTUATION = ".,!?…"


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
    if _longest_run(_loop_tokens(words)) >= _LOOP_RUN:
        return True
    return len(texts) >= 20 and len(set(texts)) / len(texts) < 0.3


# One syllable repeated this many times in a row is a decoder loop; shorter repeats
# ("một một bảy", "ừ ừ") are real Vietnamese, so nothing shorter is touched.
_LOOP_RUN = 8


def _loop_tokens(words: Sequence[dict[str, Any]]) -> list[str]:
    tokens = (str(w.get("word", "")).strip().lower().strip(".,!?…") for w in words)
    return [t for t in tokens if t]


def _longest_run(tokens: Sequence[str]) -> int:
    longest = run = 0
    previous = None
    for token in tokens:
        run = run + 1 if token == previous else 1
        previous = token
        longest = max(longest, run)
    return longest


def _collapse_loop_segments(segments: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive one-word segments that repeat the same token into one `xxx`.

    Whisper attaches "." to each looped word, so a loop is flushed as one segment per
    word and no single segment looks repetitive enough for the per-segment guard."""
    out: list[dict[str, Any]] = []
    group: list[dict[str, Any]] = []
    group_token: str | None = None

    def flush() -> None:
        count = sum(len(_loop_tokens(s.get("words", []))) for s in group)
        if count >= _LOOP_RUN:
            out.append(
                {
                    "start_ms": None,
                    "end_ms": None,
                    "text": "xxx",
                    "words": [{"word": "xxx", "start_ms": None, "end_ms": None}],
                    "repetition_suspected": True,
                }
            )
        else:
            out.extend(group)
        group.clear()

    for segment in segments:
        tokens = _loop_tokens(segment.get("words", []))
        token = tokens[0] if tokens and len(set(tokens)) == 1 else None
        if token is None or token != group_token:
            flush()
        group_token = token
        (group if token is not None else out).append(segment)
    flush()
    return out


_WINDOW_SAMPLES = 20 * 16000


def _window_bounds(
    audio_16k: np.ndarray, max_samples: int = _WINDOW_SAMPLES
) -> list[tuple[int, int]]:
    """Cut audio into windows of at most `max_samples`, ending each cut in the quietest
    300 ms of the last 2 s so a cut does not land mid-word (sherpa-vietnamese-asr
    chunking). A cut keeps the hard limit when nothing there is quieter."""
    total = len(audio_16k)
    bounds: list[tuple[int, int]] = []
    start = 0
    while start < total:
        end = start + max_samples
        if end >= total:
            end = total
        else:
            end = _quietest_cut(
                audio_16k, max(start + _SNAP_FRAME_SAMPLES, end - _SNAP_SEARCH_SAMPLES), end
            )
        bounds.append((start, end))
        start = end
    return bounds


def _has_complete_lexical_timing(words: Sequence[dict[str, Any]]) -> bool:
    lexical_words = [word for word in words if _is_lexical_word(word.get("word", ""))]
    return bool(lexical_words) and all(
        word.get("start_ms") is not None and word.get("end_ms") is not None
        for word in lexical_words
    )


def _without_sentence_punctuation(text: str) -> str:
    """Drop the sentence punctuation Whisper glues onto a word's edges.

    Real p002 output ended a segment with the word ``"thế."``. Left alone it makes the
    written main tier unparsable (chatter E316, then E316/E342 on ``%wor``) and counts
    as a phantom substitution against the reference, which compares bare tokens.
    """
    return text.strip(_ASR_EDGE_PUNCTUATION)


def _normalize_segment(
    segment: dict[str, Any], duration_ms: int
) -> dict[str, Any]:
    segment_span = _valid_span(
        segment.get("start_ms"), segment.get("end_ms"), duration_ms
    )
    text = " ".join(
        cleaned
        for cleaned in (
            _without_sentence_punctuation(token)
            for token in str(segment.get("text", "")).split()
        )
        if cleaned
    )
    words = []
    timed_spans = []
    previous_start: int | None = None
    lexical_words_timed = True
    lexical_word_count = 0
    for word in segment.get("words", []):
        token = _without_sentence_punctuation(str(word.get("word", "")))
        if not token:
            continue
        is_lexical = _is_lexical_word(token)
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
        words.append({**word, "word": token, "start_ms": start_ms, "end_ms": end_ms})

    if segment_span is None and timed_spans and lexical_word_count and lexical_words_timed:
        segment_span = min(start for start, _ in timed_spans), max(end for _, end in timed_spans)
    return {
        **segment,
        "start_ms": None if segment_span is None else segment_span[0],
        "end_ms": None if segment_span is None else segment_span[1],
        "text": text,
        "words": words,
    }


_TIMESTAMP_MODES = ("word", "segment")


class PhoWhisperBackend:
    """Lazy-loaded PhoWhisper backend using HuggingFace transformers."""

    def __init__(
        self,
        model_id: str = "vinai/phowhisper-medium",
        device: str = "cpu",
        revision: str | None = None,
        timestamps: str = "word",
        language: str | None = None,
    ) -> None:
        if timestamps not in _TIMESTAMP_MODES:
            raise ValueError(f"timestamps must be one of {_TIMESTAMP_MODES}")
        self.model_id = model_id
        self.device = device
        # SPEC: the study pins the checkpoint by revision SHA, not by name alone.
        self.revision = revision
        # ponytail: word timestamps align via cross-attention and cost ~4.4GB more per
        # window than segment ones (9.06GB vs 4.63GB peak, measured 2026-10-03) while
        # decoding byte-identical text, so text-only callers ask for "segment". The
        # default stays "word" because the chat pipeline prints %wor tiers.
        self.timestamps = timestamps
        # Multilingual checkpoints (whisper-large-v3) auto-detect the language unless
        # told; PhoWhisper is Vietnamese-only and needs no hint.
        self.language = language
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
            # ponytail: ask for fp16 at load time on CUDA. Loading fp32 and calling
            # model.half() afterwards copies the 6.2GB fp32 checkpoint onto the card
            # first and leaves the allocator holding ~6.6GB before any token is
            # generated; phowhisper-large then needs ~9.1GB for a word-timestamp
            # window (word timestamps cost ~4.4GB more than segment ones) and OOMs a
            # 12GB card that the fp16 load fits (3.1GB held) -- measured 2026-10-03.
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                device=device_arg,
                return_timestamps="word" if self.timestamps == "word" else True,
                revision=self.revision,
                **({"dtype": torch.float16} if self.device == "cuda" else {}),
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

        # ponytail: 20s windows + 400-token cap bound the word-timestamp attention cache
        # (steps x layers x heads x src, fp32): dense 30s speech reached ~9GB reserved;
        # worst case here is ~0.8GB cache. No overlap; add stride+dedupe if boundary loss
        # shows in eval.
        results: list[dict[str, Any]] = []
        try:
            for start, end in _window_bounds(audio_16k_mono):
                chunk = audio_16k_mono[start:end]
                max_tokens = min(400, max(32, math.ceil(len(chunk) / 16000 * 20)))
                result = self._decode(chunk, max_tokens)
                offset_ms = int(round(start / 16000 * 1000))
                chunk_duration_ms = round(len(chunk) * 1000 / 16000)
                segments = [
                    _normalize_segment(seg, chunk_duration_ms)
                    for seg in self._parse_pipeline_output(result)
                ]
                for seg in _collapse_loop_segments(segments):
                    results.append(self._offset_segment(seg, offset_ms))
                if self.device == "cuda":
                    import torch

                    torch.cuda.empty_cache()
            return results
        except Exception:
            raise AsrError("MODEL_UNAVAILABLE", "ASR inference execution failed") from None

    def _decode(self, chunk: np.ndarray, max_tokens: int) -> Any:
        return self._pipe(
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
                **({"language": self.language, "task": "transcribe"} if self.language else {}),
            },
        )

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


WAV2VEC2_VI_ID = "nguyenvulebinh/wav2vec2-base-vi-vlsp2020"
WAV2VEC2_VI_REVISION = "50a30dadb3ec98a0d4cdb1eb1ea315aff538f7c2"


class Wav2Vec2AsrBackend(PhoWhisperBackend):
    """Vietnamese wav2vec2 CTC arm: same windowing, offsets and word timing as the
    Whisper path, but a CTC pipeline takes no generation arguments."""

    def __init__(self, device: str = "cpu", revision: str | None = WAV2VEC2_VI_REVISION) -> None:
        super().__init__(model_id=WAV2VEC2_VI_ID, device=device, revision=revision)

    def _decode(self, chunk: np.ndarray, max_tokens: int) -> Any:
        return self._pipe({"raw": chunk, "sampling_rate": 16000})


_DIGITS = ("không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín")
_SCALES = ("", "nghìn", "triệu", "tỷ")


def _spell_group(n: int, full: bool) -> str:
    """Read 0 < n < 1000; `full` keeps "không trăm" for a non-leading group."""
    hundreds, rest = divmod(n, 100)
    tens, units = divmod(rest, 10)
    parts = [f"{_DIGITS[hundreds]} trăm"] if hundreds else (["không trăm"] if full else [])
    if tens == 0 and units:
        parts += (["lẻ"] if hundreds or full else []) + [_DIGITS[units]]
    elif tens == 1:
        parts += ["mười"] + ([{5: "lăm"}.get(units, _DIGITS[units])] if units else [])
    elif tens:
        unit = {1: "mốt", 4: "tư", 5: "lăm"}.get(units, _DIGITS[units]) if units else ""
        parts += [f"{_DIGITS[tens]} mươi"] + ([unit] if unit else [])
    return " ".join(parts)


def _spell_number(n: int) -> str:
    # ponytail: "mốt/tư/lăm/lẻ/nghìn" are the common spoken forms; regional variants
    # (ngàn, linh) are not modelled. Counts above 999,999,999,999 are left as digits.
    if n == 0:
        return _DIGITS[0]
    groups = []
    while n:
        n, group = divmod(n, 1000)
        groups.append(group)
    words = [
        f"{_spell_group(g, full=i < len(groups) - 1)} {_SCALES[i]}".strip()
        for i, g in reversed(list(enumerate(groups)))
        if g
    ]
    return " ".join(words)


def _spell_numbers(text: str) -> str:
    """Spell bare integers (and a trailing %) as Vietnamese words.

    Qwen3-ASR writes numerals and "%" that CHAT cannot parse (chatter E220/E316).
    Leading-zero strings and decimals are left alone: guessing them would invent speech.
    """
    out = []
    for token in text.split():
        core = token.rstrip(_ASR_EDGE_PUNCTUATION)
        tail = token[len(core) :]
        percent = core.endswith("%")
        digits = core[:-1] if percent else core
        if (digits.isascii() and digits.isdigit() and 0 < len(digits) <= 12
                and (digits == "0" or digits[0] != "0")):
            spoken = _spell_number(int(digits)) + (" phần trăm" if percent else "")
            out.append(spoken + tail)
        elif core == "%":
            out.append("phần trăm" + tail)
        else:
            out.append(token)
    return " ".join(out)


class Qwen3AsrBackend:
    """Lazy Qwen3-ASR backend using transformers-native checkpoints.

    Qwen3-ASR has no Vietnamese forced aligner, so segments carry their decode-window
    span and null word timings; scoring is text-based and unaffected."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-ASR-1.7B-hf",
        device: str = "cpu",
        revision: str | None = None,
        language: str = "Vietnamese",
        prompt: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.revision = revision
        self.language = language
        self.prompt = prompt
        self._processor: Any = None
        self._model: Any = None

    @staticmethod
    def _build(model_id: str, revision: str | None, device: str) -> tuple[Any, Any]:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        # fp16 at load time: loading fp32 then .half() peaks 0.6GB higher (4.70 vs 4.08GB).
        model = AutoModelForMultimodalLM.from_pretrained(
            model_id, revision=revision, **({"dtype": torch.float16} if device == "cuda" else {})
        )
        if device == "cuda":
            model = model.to("cuda")
        return processor, model

    def load(self) -> None:
        if self.device == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    raise AsrError("GPU_UNAVAILABLE", "CUDA device requested but not available")
            except ImportError:
                raise AsrError("GPU_UNAVAILABLE", "PyTorch with CUDA not available") from None
        try:
            self._processor, self._model = self._build(self.model_id, self.revision, self.device)
        except AsrError:
            raise
        except Exception:
            raise AsrError("MODEL_UNAVAILABLE", "Failed to load Qwen3-ASR model") from None

    def transcribe_audio(self, audio_16k_mono: np.ndarray) -> Sequence[dict[str, Any]]:
        """Run Qwen3-ASR on 16kHz float32 mono audio array.

        Returns one segment dict per decode spanning its window, with no words.
        """
        if self._processor is None or self._model is None:
            self.load()

        results: list[dict[str, Any]] = []
        try:
            for start, end in _window_bounds(audio_16k_mono):
                chunk = audio_16k_mono[start:end]
                max_tokens = min(400, max(32, math.ceil(len(chunk) / 16000 * 20)))
                inputs = self._processor.apply_transcription_request(
                    audio=np.asarray(chunk, dtype=np.float32),
                    language=self.language,
                    prompt=self.prompt,
                )
                # The CUDA half() model rejects host float32 feature tensors:
                # move every tensor onto the model device and cast floats to
                # the model dtype so audio-tower convs see matching types.
                prepared: dict[str, Any] = {}
                for key, value in inputs.items():
                    if hasattr(value, "to"):
                        value = value.to(self._model.device)
                        if value.is_floating_point():
                            value = value.to(self._model.dtype)
                    prepared[key] = value
                output_ids = self._model.generate(**prepared, max_new_tokens=max_tokens)
                generated = output_ids[:, prepared["input_ids"].shape[1] :]
                text = self._processor.decode(generated[0], return_format="transcription_only")
                text = _spell_numbers(str(text).strip())
                if text:
                    # ponytail: Qwen3 has no word timer, so the segment spans its decode
                    # window (coarse, but bounded). Word timing needs the CTC aligner.
                    results.append(
                        {
                            "start_ms": round(start * 1000 / 16000),
                            "end_ms": round(end * 1000 / 16000),
                            "text": text,
                            "words": [],
                        }
                    )
            return results
        except Exception:
            raise AsrError("MODEL_UNAVAILABLE", "ASR inference execution failed") from None


ASR_MODEL_CHOICES = (
    "phowhisper-medium",
    "phowhisper-large",
    "qwen3-asr",
    "whisper-large-v3",
    "wav2vec2-vi",
)

AsrBackend = PhoWhisperBackend | Qwen3AsrBackend


def make_asr_backend(
    model: str,
    device: str = "cpu",
    revision: str | None = None,
    *,
    timestamps: str = "word",
) -> AsrBackend:
    """Map a CLI model name to its ASR backend."""
    if model == "qwen3-asr":
        return Qwen3AsrBackend(device=device, revision=revision)
    if model == "phowhisper-medium":
        return PhoWhisperBackend(
            model_id="vinai/phowhisper-medium",
            device=device,
            revision=revision,
            timestamps=timestamps,
        )
    if model == "phowhisper-large":
        return PhoWhisperBackend(
            model_id="vinai/phowhisper-large",
            device=device,
            revision=revision,
            timestamps=timestamps,
        )
    if model == "whisper-large-v3":
        return PhoWhisperBackend(
            model_id="openai/whisper-large-v3",
            device=device,
            revision=revision,
            timestamps=timestamps,
            language="vi",
        )
    if model == "wav2vec2-vi":
        return Wav2Vec2AsrBackend(device=device, revision=revision or WAV2VEC2_VI_REVISION)
    raise AsrError("MODEL_UNKNOWN", f"Unknown ASR model: {model}")


def transcribe(
    audio_path: Path,
    channel_index: int = 0,
    device: str = "cpu",
    backend: AsrBackend | None = None,
    *,
    expected_sha256: str,
) -> AsrResult:
    """Transcribe a master WAV file on the declared channel.

    Loads audio, extracts channel without downmixing, resamples in-memory to 16 kHz,
    and runs PhoWhisper ASR to produce timed segments and word timestamps.
    """
    source_sha256 = verify_source_sha256(audio_path, expected_sha256)
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

    verify_source_sha256(audio_path, expected_sha256)
    return AsrResult(
        source_sha256=source_sha256,
        segments=tuple(parsed_segments),
        warnings=tuple(warnings),
    )


WindowResults = tuple[tuple[int, int, tuple[dict[str, Any], ...]], ...]


def transcribe_windows(
    audio_16k: np.ndarray,
    windows: Sequence[tuple[int, int]],
    backend: AsrBackend,
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
