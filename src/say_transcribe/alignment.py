"""Vietnamese CTC word alignment backed by a lazily loaded Wav2Vec2 model."""

from __future__ import annotations

from functools import lru_cache
import unicodedata
from typing import Protocol, Sequence

import numpy as np

_MODEL_ID = "nguyenvulebinh/wav2vec2-base-vi-vlsp2020"
_MODEL_REVISION = "50a30dadb3ec98a0d4cdb1eb1ea315aff538f7c2"
_SAMPLE_RATE = 16_000


class AlignmentError(Exception):
    """Alignment dependency or model failure with a stable, redacted code."""

    code = "ALIGNMENT_UNAVAILABLE"

    def __init__(self) -> None:
        super().__init__(f"[{self.code}] Vietnamese CTC alignment is unavailable")


class AlignmentBackend(Protocol):
    blank_id: int
    word_delimiter_id: int | None
    unk_id: int | None

    def encode(self, text: str) -> Sequence[int]: ...

    def emissions(self, audio_16k: np.ndarray) -> np.ndarray: ...

    def forced_align(self, log_probs: np.ndarray, targets: np.ndarray) -> np.ndarray: ...


class _Wav2Vec2Backend:
    def __init__(self) -> None:
        try:
            import torch
            import torchaudio
            from transformers import AutoFeatureExtractor, AutoModelForCTC, Wav2Vec2CTCTokenizer

            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
                _MODEL_ID, revision=_MODEL_REVISION, token=False
            )
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                _MODEL_ID, revision=_MODEL_REVISION, token=False
            )
            model = AutoModelForCTC.from_pretrained(
                _MODEL_ID, revision=_MODEL_REVISION, token=False
            )
            self._torch = torch
            self._torchaudio = torchaudio
            self._tokenizer = tokenizer
            self._feature_extractor = feature_extractor
            self._model = model
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device).eval()
            self.blank_id = int(tokenizer.pad_token_id)
            self.word_delimiter_id = tokenizer.word_delimiter_token_id
            self.unk_id = tokenizer.unk_token_id
        except Exception:
            raise AlignmentError() from None

    def encode(self, text: str) -> Sequence[int]:
        return self._tokenizer(text, add_special_tokens=False)["input_ids"]

    def emissions(self, audio_16k: np.ndarray) -> np.ndarray:
        try:
            inputs = self._feature_extractor(
                audio_16k,
                sampling_rate=_SAMPLE_RATE,
                return_tensors="pt",
            )
            input_values = inputs["input_values"].to(self._device)
            with self._torch.inference_mode():
                logits = self._model(input_values).logits
                log_probs = self._torch.log_softmax(logits, dim=-1)
            return log_probs[0].cpu().numpy()
        except Exception:
            raise AlignmentError() from None

    def forced_align(self, log_probs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        try:
            log_probs_tensor = self._torch.as_tensor(log_probs, dtype=self._torch.float32)
            targets_tensor = self._torch.as_tensor(targets, dtype=self._torch.int64)
            paths, _ = self._torchaudio.functional.forced_align(
                log_probs_tensor.unsqueeze(0),
                targets_tensor.unsqueeze(0),
                blank=self.blank_id,
            )
            return paths[0].cpu().numpy()
        except Exception:
            raise AlignmentError() from None


@lru_cache(maxsize=1)
def _load_backend() -> _Wav2Vec2Backend:
    return _Wav2Vec2Backend()


def _alignment_text(word: str) -> str:
    normalized = unicodedata.normalize("NFC", word).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _none_spans(words: Sequence[str]) -> tuple[tuple[None, None], ...]:
    return tuple((None, None) for _ in words)


def _token_frames(
    path: np.ndarray, targets: np.ndarray, blank_id: int
) -> tuple[tuple[int, int], ...] | None:
    spans: list[tuple[int, int]] = []
    target_index = 0
    run_start: int | None = None
    run_label: int | None = None

    def finish_run(run_end: int) -> bool:
        nonlocal target_index, run_start, run_label
        if run_start is None:
            return True
        if target_index >= len(targets) or run_label != int(targets[target_index]):
            return False
        spans.append((run_start, run_end))
        target_index += 1
        run_start = None
        run_label = None
        return True

    for frame, raw_label in enumerate(path):
        label = int(raw_label)
        if label == blank_id:
            if not finish_run(frame):
                return None
            continue
        if run_start is None:
            run_start, run_label = frame, label
        elif label != run_label:
            if not finish_run(frame):
                return None
            run_start, run_label = frame, label

    if not finish_run(len(path)) or target_index != len(targets):
        return None
    return tuple(spans)


def align_words(
    audio_16k: np.ndarray,
    words: Sequence[str],
    *,
    backend: AlignmentBackend | None = None,
) -> tuple[tuple[int | None, int | None], ...]:
    """Return relative millisecond spans for input words, preserving their order.

    Punctuation-only words and all words in a failed alignment receive ``None``
    spans. Runtime/model exceptions raise ``AlignmentError`` without raw details.
    """
    if not words:
        return ()
    try:
        audio = np.asarray(audio_16k, dtype=np.float32)
    except (TypeError, ValueError):
        return _none_spans(words)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        return _none_spans(words)

    lexical_indices: list[int] = []
    encoded_words: list[Sequence[int]] = []
    try:
        aligner = backend if backend is not None else _load_backend()
        for index, word in enumerate(words):
            text = _alignment_text(word)
            if not text:
                continue
            token_ids = tuple(int(token) for token in aligner.encode(text))
            if not token_ids or (aligner.unk_id is not None and aligner.unk_id in token_ids):
                # Word the aligner can't encode (digits, foreign script, <unk>) stays
                # untimed; don't let one bad word null out the whole window.
                continue
            lexical_indices.append(index)
            encoded_words.append(token_ids)

        if not lexical_indices:
            return _none_spans(words)

        targets: list[int] = []
        target_words: list[int | None] = []
        for position, (word_index, token_ids) in enumerate(zip(lexical_indices, encoded_words)):
            if position and aligner.word_delimiter_id is not None:
                targets.append(int(aligner.word_delimiter_id))
                target_words.append(None)
            targets.extend(token_ids)
            target_words.extend([word_index] * len(token_ids))

        target_array = np.asarray(targets, dtype=np.int64)
        emissions = np.asarray(aligner.emissions(audio), dtype=np.float32)
        if emissions.ndim != 2 or not len(emissions) or not np.isfinite(emissions).all():
            return _none_spans(words)
        required_frames = len(target_array) + sum(
            left == right for left, right in zip(target_array, target_array[1:])
        )
        if required_frames > len(emissions):
            return _none_spans(words)
        shifted = emissions - emissions.max(axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        path = np.asarray(aligner.forced_align(log_probs, target_array)).reshape(-1)
        token_spans = _token_frames(path, target_array, int(aligner.blank_id))
    except AlignmentError:
        raise
    except Exception:
        raise AlignmentError() from None

    if token_spans is None or len(token_spans) != len(targets):
        return _none_spans(words)

    frame_ms = len(audio) * 1000 / (_SAMPLE_RATE * len(path))
    output = list(_none_spans(words))
    for token_index, word_index in enumerate(target_words):
        if word_index is None:
            continue
        start_frame, end_frame = token_spans[token_index]
        start_ms = max(
            0, min(round(start_frame * frame_ms), round(len(audio) * 1000 / _SAMPLE_RATE))
        )
        end_ms = max(0, min(round(end_frame * frame_ms), round(len(audio) * 1000 / _SAMPLE_RATE)))
        previous = output[word_index]
        if previous[0] is None:
            output[word_index] = (start_ms, end_ms)
        else:
            output[word_index] = (previous[0], end_ms)

    previous_end = 0
    for start_ms, end_ms in output:
        if start_ms is None or end_ms is None:
            continue
        if start_ms < previous_end or end_ms <= start_ms:
            return _none_spans(words)
        previous_end = end_ms
    return tuple(output)
