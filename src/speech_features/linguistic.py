"""Vietnamese lexical and disfluency features (Task 3).

Participant-only transcript measures built on the immutable Task 1
``Transcript`` contract. Token normalization is deterministic, Unicode NFC +
``casefold``, preserving diacritics and the Vietnamese ``đ``/``Đ`` distinction.
Token classes are the explicit ``word``/``filler``/``fragment``/``noise`` kinds
the transcript contract already enforces.

All measures are computed only from the participant's own utterances. Lexical
diversity (TTR, hapax ratio, Brunet W, Honore R), repetition/filler rates, and
a simple external function-word-set ratio return ``NaN`` (never zero) when the
denominator degenerates, matching the plan's contract.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .schema import Token, Transcript, nfc

# Honore's R requires the ratio of once-occurring types; the quantity
# ``1 - V1 / V`` is the divisor and can vanish (single-type transcripts).
_HONORE_MIN_TYPES = 1


def normalise_token(text: str) -> str:
    """Deterministic NFC + casefold token normalization.

    ``casefold`` lower-cases (handling ``Đ`` -> ``đ``) while keeping the
    distinct ``d``/``đ`` graphemes apart; diacritics are preserved. Combining
    diacritics are composed via NFC so ``o`` + combining circumflex == ``ô``.
    """
    return nfc(text).casefold()


def participant_tokens(transcript: Transcript) -> tuple[Token, ...]:
    """All participant tokens in order (word/filler/fragment/noise)."""
    return tuple(
        tok for utt in transcript.utterances if utt.speaker == "participant" for tok in utt.tokens
    )


def participant_word_tokens(transcript: Transcript) -> tuple[Token, ...]:
    """Participant word-class tokens in order (fillers/fragments/noise pulled)."""
    return tuple(tok for tok in participant_tokens(transcript) if tok.kind == "word")


def _safe_div(num: float, den: float) -> float:
    return num / den if den else math.nan


def _honore_r(n: int, v: int, v1: int) -> float:
    """Honore's R = 100*log(N)/(1 - V1/V). NaN on a degenerate divisor."""
    if v <= 0 or v1 == v:  # divisor 1 - V1/V is zero when all types are hapax
        return math.nan
    return 100.0 * math.log(n) / (1.0 - v1 / v)


def token_stats(transcript: Transcript) -> dict[str, float]:
    """Raw token/character counts plus mean word-token length.

    Character counts sum the code points of participant word tokens only.
    """
    words = participant_word_tokens(transcript)
    tokens = participant_tokens(transcript)
    kinds = {"word": 0, "filler": 0, "fragment": 0, "noise": 0}
    for tok in tokens:
        kinds[tok.kind] += 1
    word_count = len(words)
    char_count = sum(len(tok.text) for tok in words)
    return {
        "lex_token_count": float(len(tokens)),
        "lex_word_count": float(word_count),
        "lex_unique_count": float(len({normalise_token(t.text) for t in words})),
        "lex_char_count": float(char_count),
        "lex_mean_token_length": _safe_div(char_count, word_count),
        "lex_filler_count": float(kinds["filler"]),
        "lex_fragment_count": float(kinds["fragment"]),
        "lex_noise_count": float(kinds["noise"]),
    }


def lexical_diversity(word_texts: Sequence[str]) -> dict[str, float]:
    """TTR, hapax ratio, Brunet W, and Honore R over normalized word forms."""
    forms = [normalise_token(t) for t in word_texts]
    n = len(forms)
    counts: dict[str, int] = {}
    for f in forms:
        counts[f] = counts.get(f, 0) + 1
    v = len(counts)
    v1 = sum(1 for c in counts.values() if c == 1)
    ttr = _safe_div(v, n)
    hapax = _safe_div(v1, n)
    brunet_w = v**0.172 if v > 0 else math.nan
    return {
        "lex_ttr": ttr,
        "lex_hapax_ratio": hapax,
        "lex_brunet_w": brunet_w,
        "lex_honore_r": _honore_r(n, v, v1),
    }


def repetition_ratio(word_texts: Sequence[str]) -> float:
    """Ratio of immediate (adjacent) repeated word tokens in participant speech."""
    forms = [normalise_token(t) for t in word_texts]
    if not forms:
        return math.nan
    repeats = sum(1 for a, b in zip(forms, forms[1:]) if a == b)
    return _safe_div(repeats, len(forms))


def raw_ratios(transcript: Transcript) -> dict[str, float]:
    """Filler/fragment/noise ratios measured against all participant tokens."""
    tokens = participant_tokens(transcript)
    total = len(tokens)
    n_filler = sum(1 for t in tokens if t.kind == "filler")
    n_fragment = sum(1 for t in tokens if t.kind == "fragment")
    n_noise = sum(1 for t in tokens if t.kind == "noise")
    return {
        "lex_filler_ratio": _safe_div(n_filler, total),
        "lex_fragment_ratio": _safe_div(n_fragment, total),
        "lex_noise_ratio": _safe_div(n_noise, total),
    }


def function_word_ratio(
    word_texts: Sequence[str], function_words: set[str] | frozenset[str] | None
) -> float:
    """Ratio of participant word tokens that appear in an external function-word set.

    The set is normalized (NFC/casefold) on load; ``None`` yields ``NaN`` so the
    feature is only defined when an external set is supplied.
    """
    if function_words is None:
        return math.nan
    norm_set = {normalise_token(w) for w in function_words}
    if not word_texts:
        return math.nan
    forms = [normalise_token(t) for t in word_texts]
    hits = sum(1 for f in forms if f in norm_set)
    return _safe_div(hits, len(forms))


def extract_linguistic(
    transcript: Transcript, *, function_words: set[str] | frozenset[str] | None = None
) -> dict[str, float]:
    """Compute the full ``lex_*`` feature set for a participant transcript."""
    stats = token_stats(transcript)
    words = [t.text for t in participant_word_tokens(transcript)]
    stats.update(lexical_diversity(words))
    stats.update(raw_ratios(transcript))
    stats["lex_repetition_ratio"] = repetition_ratio(words)
    stats["lex_function_word_ratio"] = function_word_ratio(words, function_words)
    return stats


__all__ = [
    "normalise_token",
    "participant_tokens",
    "participant_word_tokens",
    "token_stats",
    "lexical_diversity",
    "repetition_ratio",
    "raw_ratios",
    "function_word_ratio",
    "extract_linguistic",
]
