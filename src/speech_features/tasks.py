"""External task-spec scorers (Task 3).

Deterministic scoring of a participant ``Transcript`` against versioned
task-spec JSON: picture concept/entity/action aliases, recall idea aliases,
phonemic initials/exclusions, and semantic item aliases/subcategories.

Matching uses exact NFC/casefold equality first, then standard Levenshtein
edit-distance normalised similarity against a conservative, documented
threshold. No diagnosis/label is ever read and no NLP/ASR model is used.
"""

from __future__ import annotations

import math
from collections import defaultdict

from .linguistic import normalise_token, participant_word_tokens

# A response/word is accepted as a match when its normalised similarity to an
# alias is at least this threshold. 0.85 is deliberately conservative: it admits
# only near-exact variants (e.g. a dropped diacritic) and rejects genuinely
# unrelated words. This is a documented, fixed editorial choice, not a tuned
# hyperparameter.
SIMILARITY_THRESHOLD = 0.85


def levenshtein(a: str, b: str) -> int:
    """Classic Levenshtein edit distance (insert/delete/substitute)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[lb]


def normalised_similarity(a: str, b: str) -> float:
    """Levenshtein similarity in ``[0, 1]``; ``0`` for an empty pair.

    ``similarity = 1 - distance / max(len(a), len(b))``. An empty-vs-empty pair
    has no meaningful overlap and scores ``0`` (never a spurious match).
    """
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 0.0
    return 1.0 - levenshtein(a, b) / max(la, lb)


def match_alias(text: str, aliases, threshold: float = SIMILARITY_THRESHOLD) -> str | None:
    """Return the first alias the text matches (exact or similar enough), else ``None``."""
    norm = normalise_token(text)
    for alias in aliases:
        alias_norm = normalise_token(alias)
        if alias_norm == norm:
            return alias
    for alias in aliases:
        if normalised_similarity(norm, normalise_token(alias)) >= threshold:
            return alias
    return None


def match_key(
    text, aliases_by_key: dict, threshold: float = SIMILARITY_THRESHOLD
) -> tuple[str | None, str | None]:
    """Match text to a ``(key, alias)`` pair, exact aliases first.

    All aliases across every key are scanned for an exact NFC/casefold match
    before any fuzzy pass, so an earlier key's fuzzy alias never shadows a later
    key's exact one. Fuzzy matches fall back to the first hit in spec iteration
    order; each text maps to at most one key.
    """
    norm = normalise_token(text)
    for key, aliases in aliases_by_key.items():
        for alias in aliases:
            if normalise_token(alias) == norm:
                return key, alias
    for key, aliases in aliases_by_key.items():
        for alias in aliases:
            if normalised_similarity(norm, normalise_token(alias)) >= threshold:
                return key, alias
    return None, None


def _safe_div(num: float, den: float) -> float:
    return num / den if den else math.nan


def _lcs(a: list, b: list) -> int:
    """Length of the longest common subsequence of two sequences."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def score_picture(transcript, spec: dict) -> dict[str, float]:
    """Picture-description concept/entity/action coverage from alias groups."""
    concepts = spec["concept_aliases"]
    entities = spec.get("entity_groups", {})
    actions = spec.get("action_groups", {})
    words = [t.text for t in participant_word_tokens(transcript)]
    n_words = len(words)

    matched = []  # (concept_key,) in order of appearance
    for text in words:
        key, _ = match_key(text, concepts)
        matched.append(key)

    matched_keys = [k for k in matched if k is not None]
    distinct = set(matched_keys)
    concept_coverage = len(distinct) / len(concepts) if concepts else math.nan

    entity_hits = set()
    for text in words:
        key, _ = match_key(text, entities)
        if key is not None:
            entity_hits.add(key)
    action_hits = set()
    for text in words:
        key, _ = match_key(text, actions)
        if key is not None:
            action_hits.add(key)

    n_matched = len(matched_keys)
    return {
        "picture_concept_coverage": concept_coverage,
        "picture_concept_density": _safe_div(len(distinct), n_words),
        "picture_repeat_ratio": _safe_div(n_matched - len(distinct), n_matched),
        "picture_entity_coverage": len(entity_hits) / len(entities) if entities else math.nan,
        "picture_action_coverage": len(action_hits) / len(actions) if actions else math.nan,
    }


def score_recall(transcript, spec: dict) -> dict[str, float]:
    """Recall idea coverage/density/repeat plus a normalized order score."""
    ideas = spec["idea_aliases"]
    reference_order = list(ideas.keys())
    words = [t.text for t in participant_word_tokens(transcript)]
    n_words = len(words)

    matched_keys = []
    for text in words:
        key, _ = match_key(text, ideas)
        matched_keys.append(key)
    present = [k for k in matched_keys if k is not None]
    distinct = set(present)

    # Order score: LCS of the distinct matched ideas (in utterance order) against
    # the reference order, normalised by the number of reference ideas.
    ordered = []
    for k in present:
        if k not in ordered:
            ordered.append(k)
    if reference_order:
        order_score = _lcs(ordered, reference_order) / len(reference_order) if ordered else math.nan
    else:
        order_score = math.nan

    n_matched = len(present)
    return {
        "recall_idea_coverage": len(distinct) / len(ideas) if ideas else math.nan,
        "recall_idea_density": _safe_div(len(distinct), n_words),
        "recall_repeat_ratio": _safe_div(n_matched - len(distinct), n_matched),
        "recall_order_score": order_score,
    }


def _responses(transcript) -> list[str]:
    """One response item per participant utterance containing word tokens.

    The joined, normalised word text of an utterance is its response string; the
    utterance order is the response's time slot.
    """
    responses = []
    for utt in transcript.utterances:
        if utt.speaker != "participant":
            continue
        words = [normalise_token(t.text) for t in utt.tokens if t.kind == "word"]
        if words:
            responses.append(" ".join(words))
    return responses


def _response_initial(response: str) -> str:
    return response.split()[0]


def score_phonemic(transcript, spec: dict) -> dict[str, float]:
    """Phonemic fluency: valid/repeats/intrusions, half-time counts, change.

    A response is valid when its first word starts with one of the spec's
    initials and the whole response is not in ``exclusions``. Responses that
    fail either rule are intrusions. ``fluency_production_change`` is the
    second-half valid count minus the first-half valid count over the utterance
    time slots.
    """
    initials = [normalise_token(i) for i in spec["initials"]]
    exclusions = {normalise_token(e) for e in spec.get("exclusions", [])}
    responses = _responses(transcript)
    n = len(responses)

    valid_first = set()
    valid_count = 0
    valid_ids = []
    intrusion_ids = set()
    for i, resp in enumerate(responses):
        first = _response_initial(resp)
        starts_initial = any(first.startswith(ini) for ini in initials)
        excluded = resp in exclusions
        if starts_initial and not excluded:
            valid_count += 1
            valid_ids.append(i)
            valid_first.add(resp)
        else:
            intrusion_ids.add(i)

    valid_unique = len(valid_first)
    repeats = valid_count - valid_unique
    half = math.ceil(n / 2)
    first_half = [i for i in valid_ids if i < half]
    second_half = [i for i in valid_ids if i >= half]
    first_valid = len(first_half)
    second_valid = len(second_half)
    return {
        "fluency_response_count": float(n),
        "fluency_valid_count": float(valid_count),
        "fluency_valid_unique": float(valid_unique),
        "fluency_repeats": float(repeats),
        "fluency_intrusions": float(len(intrusion_ids)),
        "fluency_first_half_valid": float(first_valid),
        "fluency_second_half_valid": float(second_valid),
        "fluency_production_change": float(second_valid - first_valid),
        "fluency_rate": _safe_div(valid_unique, n),
    }


def score_semantic(transcript, spec: dict) -> dict[str, float]:
    """Semantic fluency: valid unique/repeats/rate plus clusters and switches.

    Each utterance's word text is matched to an item alias; a valid response is
    an item. ``fluency_clusters`` counts maximal runs of consecutive responses in
    the same subcategory; ``fluency_switches`` counts adjacent in-subcategory
    changes.
    """
    items = spec["item_aliases"]
    subcategories = spec.get("subcategories", {})
    responses = _responses(transcript)

    counts: dict[str, int] = defaultdict(int)
    for resp in responses:
        key, _ = match_key(resp, items)
        if key is not None:
            counts[key] += 1

    valid_unique = len(counts)
    valid_count = sum(counts.values())
    repeats = valid_count - valid_unique

    # Clusters/switches describe transitions between consecutive responses in the
    # same vs. different subcategory; responses whose matched item has no
    # declared subcategory are excluded from this sequence.
    matched_categories = []
    for resp in responses:
        key, _ = match_key(resp, items)
        if key is not None and key in subcategories:
            matched_categories.append(subcategories[key])

    clusters = 0
    switches = 0
    prev = None
    for cat in matched_categories:
        if cat != prev:
            clusters += 1  # a new run begins
        if prev is not None and cat != prev:
            switches += 1
        prev = cat
    n = len(responses)

    return {
        "fluency_response_count": float(n),
        "fluency_valid_count": float(valid_count),
        "fluency_valid_unique": float(valid_unique),
        "fluency_repeats": float(repeats),
        "fluency_clusters": float(clusters),
        "fluency_switches": float(switches),
        "fluency_rate": _safe_div(valid_unique, n) if n else math.nan,
    }


__all__ = [
    "SIMILARITY_THRESHOLD",
    "levenshtein",
    "normalised_similarity",
    "match_alias",
    "match_key",
    "score_picture",
    "score_recall",
    "score_phonemic",
    "score_semantic",
]
