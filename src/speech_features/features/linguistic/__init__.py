"""Adult-neuro lexical and disfluency features over ``SpeechDocument`` (Task 8).

The single entry point :func:`extract_lexical_features` consumes only a
:class:`~speech_features.document.SpeechDocument`, a target speaker id, and a
recording id — never diagnosis, labels, age, cutoffs, task lexicons, or an
external tokenizer/tagger/parser/model. It returns ``(features, issues)``
where every one of the 42 registered keys is present; unavailable values are
``NaN`` and each is paired with exactly one per-key ``warning`` issue
(``MISSING_ANNOTATION`` for an absent target sample or lemma layer,
``INSUFFICIENT_TOKENS`` for too few or degenerate observations).

Counting rules
--------------
- Only the target speaker's utterances/tokens are used; examiner content
  never affects a value, and adjacent tokens are never compared across
  utterance boundaries.
- ``lex_token_count`` counts every explicit target token, including CHAT code
  tokens of any kind. Word/syllable observations count only explicit
  ``kind == "word"`` tokens; each is one explicit syllable observation and
  text is never split or segmented.
- Explicit words group within each utterance by non-null ``word_id``; an
  ungrouped word-kind token is its own word, and grouping never crosses
  utterance boundaries.
- Character counts and token lengths use NFC text code points; NFC + casefold
  is applied only for equality/type counting, preserving diacritics and the
  ``d``/``đ`` distinction. ``lex_utterance_count`` counts explicit target
  utterances; MLU divides explicit word/syllable counts by it.
- A known explicit absence (a present sample with no event of a kind) is a
  real zero; an absent target language sample is unavailable (``NaN`` +
  ``MISSING_ANNOTATION``), never a guessed zero.

Diversity formulas
------------------
For the normalized word-kind token text sequence (surface) or the ``lemma``
annotation layer values (lemma) of length ``N``, ``V`` types, ``V1`` hapax:

- TTR ``V/N``; MATTR-20 = mean TTR over every length-20 window (``N < 20``
  is ``NaN``); MTLD = mean of the forward/reverse factor-method values with
  threshold ``0.72`` and partial factor ``(1 - ttr)/(1 - 0.72)`` (a zero
  factor denominator is ``NaN``); HD-D-42 = ``sum_types(1 - C(N-f,42)/C(N,42))
  / 42`` via the bounded product ``product(k=0..41, (N-f-k)/(N-k))``
  (``N < 42`` is ``NaN``); hapax ratio ``V1/N``; Brunet W
  ``N ** (V ** -0.165)``; Honore R ``100*log(N)/(1 - V1/V)`` (zero divisor is
  ``NaN``); entropy = normalized Shannon ``-sum(p log p)/log(V)`` (a
  single-type sample is ``0``). SDs are population SDs (``ddof=0``).

The ``lemma`` layer must map every target word-kind token id to a non-empty
string; an absent/incomplete/non-string layer makes all eight lemma keys
``NaN`` with ``MISSING_ANNOTATION``, with no fallback to surface forms.

Disfluency
----------
Filler/fragment/retracing/revision/annotated-error counts count the explicit
token kinds ``filler``, ``fragment``, ``retracing``, ``revision``, and
``error``. Immediate repetition counts a word-kind token whose normalized text
equals the immediately previous word-kind token in the same utterance. Maze
events sum fillers, fragments, immediate repetitions, retracings, and
revisions (errors and noise excluded). Repetition ratio divides by the
explicit syllable count; every other ratio divides by ``lex_token_count``; a
zero denominator is ``NaN`` + ``INSUFFICIENT_TOKENS``.
"""

from __future__ import annotations

import math
from collections import Counter

from ...result import FeatureIssue, InvalidConfigError, TargetSpeakerRequiredError
from ...schema import nfc
from .definitions import ALL_KEYS, register_linguistic_features
from .diversity import (
    brunet_w,
    entropy,
    hapax_ratio,
    hdd_42,
    honore_r,
    mattr_20,
    mtld,
    ttr,
)

register_linguistic_features()

_DIVERSITY_BASES = (
    "ttr",
    "mattr_20",
    "mtld",
    "hdd_42",
    "hapax_ratio",
    "brunet_w",
    "honore_r",
    "entropy",
)
SURFACE_DIVERSITY_KEYS = tuple(f"lex_token_{base}" for base in _DIVERSITY_BASES)
LEMMA_DIVERSITY_KEYS = tuple(f"lex_lemma_{base}" for base in _DIVERSITY_BASES)


def _issue(
    recording_id: str, speaker_id: str, code: str, message: str, feature: str
) -> FeatureIssue:
    return FeatureIssue(
        recording_id=recording_id,
        speaker_id=speaker_id,
        code=code,
        severity="warning",
        message=message,
        feature=feature,
    )


def _flag(
    features: dict,
    keys,
    code: str,
    message: str,
    recording_id: str,
    speaker_id: str,
    issues: list,
) -> None:
    for key in keys:
        features[key] = math.nan
        issues.append(_issue(recording_id, speaker_id, code, message, feature=key))


def _normalise(text: str) -> str:
    """NFC + casefold for equality/type counting only; diacritics preserved."""
    return nfc(text).casefold()


def _resolve_target_speaker(document, target_speaker) -> str:
    """Resolve the target speaker id consistently with the acoustic pack.

    An explicit target must be a documented speaker (else ``INVALID_CONFIG``);
    several documented speakers without a target raise ``TARGET_SPEAKER_REQUIRED``;
    a single documented speaker is inferred; an empty document yields ``""``.
    """
    speaker_ids = {speaker.id for speaker in document.speakers} if document is not None else set()
    if target_speaker is not None:
        if document is None or target_speaker not in speaker_ids:
            raise InvalidConfigError(
                f"target speaker {target_speaker!r} is not a speaker of the speech document"
            )
        return target_speaker
    if len(speaker_ids) > 1:
        raise TargetSpeakerRequiredError(
            "speech document has multiple speakers; pass target_speaker to isolate one"
        )
    if len(speaker_ids) == 1:
        return next(iter(speaker_ids))
    return ""


def _word_syllable_lengths(utterances) -> list[int]:
    """One syllable count per explicit word, grouped per utterance only."""
    lengths: list[int] = []
    for utterance in utterances:
        per_word_id: dict[str, int] = {}
        ungrouped = 0
        for token in utterance.tokens:
            if token.kind != "word":
                continue
            if token.word_id is not None:
                per_word_id[token.word_id] = per_word_id.get(token.word_id, 0) + 1
            else:
                ungrouped += 1
        lengths.extend(per_word_id.values())
        lengths.extend([1] * ungrouped)
    return lengths


def _lemma_forms(document, word_tokens) -> list[str] | None:
    """Ordered lemma values for the target word tokens, or ``None`` when the
    ``lemma`` layer is absent, incomplete, or non-string."""
    layer = next((a for a in document.annotations if a.layer == "lemma"), None)
    if layer is None:
        return None
    forms: list[str] = []
    for token in word_tokens:
        value = layer.values.get(token.id)
        if not isinstance(value, str) or not value:
            return None
        forms.append(_normalise(value))
    return forms


def _fill_diversity(
    features: dict,
    issues: list,
    forms: list[str],
    keys,
    recording_id: str,
    speaker_id: str,
) -> None:
    values = (
        ttr(forms),
        mattr_20(forms),
        mtld(forms),
        hdd_42(forms),
        hapax_ratio(forms),
        brunet_w(forms),
        honore_r(forms),
        entropy(forms),
    )
    for key, value in zip(keys, values):
        features[key] = value
        if math.isnan(value):
            issues.append(
                _issue(
                    recording_id,
                    speaker_id,
                    "INSUFFICIENT_TOKENS",
                    "insufficient or degenerate token observations; feature unavailable",
                    feature=key,
                )
            )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _population_sd(values: list[float]) -> float:
    mean = _mean(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def extract_lexical_features(
    document,
    *,
    target_speaker=None,
    recording_id: str = "",
) -> tuple[dict[str, float], tuple[FeatureIssue, ...]]:
    """Extract the adult-neuro lexical and disfluency features for one target.

    ``document`` is a :class:`~speech_features.document.SpeechDocument`;
    ``target_speaker`` resolves like the acoustic pack. Returns
    ``(features, issues)``: every one of the 42 registered keys is present
    (``NaN`` when its prerequisite is unavailable) and each ``NaN`` is paired
    with exactly one per-key warning issue. Formula and denominator details
    are documented in this module's docstring.
    """
    features = {key: math.nan for key in ALL_KEYS}
    issues: list[FeatureIssue] = []
    speaker_id = _resolve_target_speaker(document, target_speaker)

    utterances = (
        [u for u in document.utterances if u.speaker_id == speaker_id]
        if document is not None and speaker_id
        else []
    )
    tokens = [token for utterance in utterances for token in utterance.tokens]
    if not tokens:
        _flag(
            features,
            ALL_KEYS,
            "MISSING_ANNOTATION",
            "no explicit target-speaker language sample; lexical features unavailable",
            recording_id,
            speaker_id,
            issues,
        )
        return features, tuple(issues)

    word_tokens = [token for token in tokens if token.kind == "word"]
    forms = [_normalise(token.text) for token in word_tokens]
    word_lengths = _word_syllable_lengths(utterances)
    token_count = float(len(tokens))
    syllable_count = float(len(word_tokens))
    word_count = float(len(word_lengths))
    utterance_count = float(len(utterances))

    features["lex_utterance_count"] = utterance_count
    features["lex_token_count"] = token_count
    features["lex_word_count"] = word_count
    features["lex_syllable_count"] = syllable_count
    features["lex_character_count"] = float(sum(len(token.text) for token in word_tokens))
    features["lex_unique_token_count"] = float(len(set(forms)))
    features["lex_mlu_words"] = word_count / utterance_count
    features["lex_mlu_syllables"] = syllable_count / utterance_count

    if word_tokens:
        char_lengths = [len(token.text) for token in word_tokens]
        features["lex_token_length_mean_characters"] = float(_mean(char_lengths))
        if len(char_lengths) >= 2:
            features["lex_token_length_sd_characters"] = float(_population_sd(char_lengths))
        else:
            _flag(
                features,
                ("lex_token_length_sd_characters",),
                "INSUFFICIENT_TOKENS",
                "fewer than two word tokens; token-length SD unavailable",
                recording_id,
                speaker_id,
                issues,
            )
    else:
        _flag(
            features,
            (
                "lex_token_length_mean_characters",
                "lex_token_length_sd_characters",
                "lex_word_length_mean_syllables",
                "lex_word_length_sd_syllables",
            ),
            "INSUFFICIENT_TOKENS",
            "no word-kind tokens; length features unavailable",
            recording_id,
            speaker_id,
            issues,
        )

    if word_lengths:
        features["lex_word_length_mean_syllables"] = float(_mean(word_lengths))
        if len(word_lengths) >= 2:
            features["lex_word_length_sd_syllables"] = float(_population_sd(word_lengths))
        else:
            _flag(
                features,
                ("lex_word_length_sd_syllables",),
                "INSUFFICIENT_TOKENS",
                "fewer than two explicit words; word-length SD unavailable",
                recording_id,
                speaker_id,
                issues,
            )
    else:
        _flag(
            features,
            ("lex_word_length_mean_syllables", "lex_word_length_sd_syllables"),
            "INSUFFICIENT_TOKENS",
            "no explicit words; word-length features unavailable",
            recording_id,
            speaker_id,
            issues,
        )

    _fill_diversity(features, issues, forms, SURFACE_DIVERSITY_KEYS, recording_id, speaker_id)

    if not word_tokens:
        _flag(
            features,
            LEMMA_DIVERSITY_KEYS,
            "INSUFFICIENT_TOKENS",
            "no word-kind tokens; lemma diversity unavailable",
            recording_id,
            speaker_id,
            issues,
        )
    else:
        lemma_forms = _lemma_forms(document, word_tokens)
        if lemma_forms is None:
            _flag(
                features,
                LEMMA_DIVERSITY_KEYS,
                "MISSING_ANNOTATION",
                "lemma annotation layer is absent or incomplete for the target word tokens",
                recording_id,
                speaker_id,
                issues,
            )
        else:
            _fill_diversity(
                features, issues, lemma_forms, LEMMA_DIVERSITY_KEYS, recording_id, speaker_id
            )

    kinds = Counter(token.kind for token in tokens)
    filler_count = float(kinds["filler"])
    fragment_count = float(kinds["fragment"])
    retracing_count = float(kinds["retracing"])
    revision_count = float(kinds["revision"])
    error_count = float(kinds["error"])
    repetition_count = 0.0
    for utterance in utterances:
        utterance_forms = [
            _normalise(token.text) for token in utterance.tokens if token.kind == "word"
        ]
        repetition_count += sum(
            1 for left, right in zip(utterance_forms, utterance_forms[1:]) if left == right
        )
    maze_count = filler_count + fragment_count + repetition_count + retracing_count + revision_count

    features["disfluency_filler_count"] = filler_count
    features["disfluency_fragment_count"] = fragment_count
    features["disfluency_retracing_count"] = retracing_count
    features["disfluency_revision_count"] = revision_count
    features["disfluency_annotated_error_count"] = error_count
    features["disfluency_immediate_repetition_count"] = repetition_count
    features["disfluency_maze_count"] = maze_count

    ratios = (
        ("disfluency_filler_ratio", filler_count, token_count),
        ("disfluency_fragment_ratio", fragment_count, token_count),
        ("disfluency_retracing_ratio", retracing_count, token_count),
        ("disfluency_revision_ratio", revision_count, token_count),
        ("disfluency_annotated_error_ratio", error_count, token_count),
        ("disfluency_maze_ratio", maze_count, token_count),
        ("disfluency_immediate_repetition_ratio", repetition_count, syllable_count),
    )
    for key, count, denominator in ratios:
        if denominator:
            features[key] = count / denominator
        else:
            _flag(
                features,
                (key,),
                "INSUFFICIENT_TOKENS",
                "zero ratio denominator; disfluency ratio unavailable",
                recording_id,
                speaker_id,
                issues,
            )

    return features, tuple(issues)


from .morphosyntax import (  # noqa: E402  (re-export after helpers resolve)
    extract_adult_neuro_features,
    extract_morphosyntax_features,
)

__all__ = [
    "ALL_KEYS",
    "LEMMA_DIVERSITY_KEYS",
    "SURFACE_DIVERSITY_KEYS",
    "extract_adult_neuro_features",
    "extract_lexical_features",
    "extract_morphosyntax_features",
]
