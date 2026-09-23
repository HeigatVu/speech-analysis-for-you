from dataclasses import dataclass
from typing import Callable, Sequence
import unicodedata

from say_transcribe.asr import AsrSegment, WordTiming


class WordGroupingError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class GroupedWord:
    word: str
    start_ms: int | None
    end_ms: int | None
    syllables: tuple[WordTiming, ...]


def default_underthesea_tokenize(text: str) -> Sequence[str]:
    """Lazy-load underthesea and segment text into _-joined words."""
    try:
        from underthesea import word_tokenize

        # format="text" joins multi-syllable words with '_'
        tokenized_str = word_tokenize(text, format="text")
        return tokenized_str.split()
    except Exception:
        raise WordGroupingError("MODEL_UNAVAILABLE", "Failed to run Underthesea tokenizer") from None


def group_utterance_words(
    segment: AsrSegment,
    tokenizer: Callable[[str], Sequence[str]] | None = None,
) -> tuple[GroupedWord, ...]:
    """Group an ASR utterance segment's tokens into _-joined Vietnamese words.

    Guarantees:
    - Never groups tokens across utterance boundaries (only processes a single segment).
    - Preserves NFC, tone marks, and d/đ exactly.
    - Word span runs from first syllable start_ms to last syllable end_ms.
    - Utterance-final punctuation carries no span (start_ms=None, end_ms=None).
    - Raises WordGroupingError("WORD_GROUPING_UNALIGNED") on syllable mismatch.
    """
    if tokenizer is None:
        tokenizer = default_underthesea_tokenize

    norm_text = unicodedata.normalize("NFC", segment.text.strip())
    if not norm_text:
        return ()

    tokens = tokenizer(norm_text)
    norm_tokens = [unicodedata.normalize("NFC", t) for t in tokens]

    input_words = [
        WordTiming(
            word=unicodedata.normalize("NFC", w.word.strip()),
            start_ms=w.start_ms,
            end_ms=w.end_ms,
        )
        for w in segment.words
        if w.word.strip()
    ]

    # If no word-level timings were captured, produce grouped words without spans
    if not input_words:
        result: list[GroupedWord] = []
        for token in norm_tokens:
            result.append(GroupedWord(word=token, start_ms=None, end_ms=None, syllables=()))
        return tuple(result)

    # Validate syllable alignment
    # Deconstruct grouped tokens into syllables (split by '_')
    expanded_syllables: list[str] = []
    for token in norm_tokens:
        expanded_syllables.extend(token.split("_"))

    input_syllables = [w.word for w in input_words]

    if len(expanded_syllables) != len(input_syllables):
        raise WordGroupingError(
            "WORD_GROUPING_UNALIGNED",
            "Grouped word syllable count does not match input token count",
        )

    for exp_syl, in_syl in zip(expanded_syllables, input_syllables):
        if exp_syl.lower() != in_syl.lower():
            raise WordGroupingError(
                "WORD_GROUPING_UNALIGNED",
                "Grouped syllable text does not match input syllable text",
            )

    # Assign spans from input_words
    grouped_words: list[GroupedWord] = []
    curr_idx = 0

    for i, token in enumerate(norm_tokens):
        syl_parts = token.split("_")
        n_syl = len(syl_parts)
        matched_words = tuple(input_words[curr_idx : curr_idx + n_syl])
        curr_idx += n_syl

        # Check if this token is utterance-final punctuation
        is_final_punct = (i == len(norm_tokens) - 1) and token in {".", "!", "?", "...", "…"}

        if is_final_punct:
            span_start = None
            span_end = None
        else:
            span_start = matched_words[0].start_ms if matched_words else None
            span_end = matched_words[-1].end_ms if matched_words else None

        grouped_words.append(
            GroupedWord(
                word=token,
                start_ms=span_start,
                end_ms=span_end,
                syllables=matched_words,
            )
        )

    return tuple(grouped_words)
