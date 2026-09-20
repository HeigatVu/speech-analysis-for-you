"""Adult-neuro morphosyntax and conversation features (Task 9).

The single entry point :func:`extract_morphosyntax_features` consumes only a
:class:`~speech_features.document.SpeechDocument`, a target speaker id, and a
recording id — never diagnosis, labels, age, cutoffs, task lexicons, or an
external tokenizer/tagger/parser/model. It returns
``(recording_features, utterance_rows, issues)`` where every one of the 51
registered recording keys is present, each target turn appears as one plain
``dict`` row with the four registered utterance keys, and every emitted
``NaN`` (recording or utterance) is paired with exactly one per-key
``warning`` issue. Recording ``NaN`` issues carry no ``utterance_id``;
utterance ``NaN`` issues carry the owning row's ``utterance_id``.

No-inference boundary
---------------------
Morphology consumes only explicit annotation: the ``upos``, ``classifier``,
and ``particle`` :class:`AnnotationLayer` values and the explicit
``DocumentToken.dep_rel``/``dep_head``/``language`` fields of the target's
``kind == "word"`` tokens. ``%mor``/``%gra`` are never parsed and unavailable
morphology is never inferred.

- ``upos`` must hold a non-empty string for every representative target word token,
  case-normalised to uppercase, and each value must be one of the 17 locked
  Universal POS tags. An absent/incomplete/non-string/unknown layer makes all
  17 ``morph_upos_*_ratio`` keys and the four UPOS-derived composition keys
  ``NaN`` with ``MISSING_ANNOTATION``; the independent classifier, particle,
  code-switch, and dependency keys are unaffected.
- ``classifier`` and ``particle`` layers independently map every target word
  token id to numeric ``0`` or ``1`` (booleans are invalid); each
  missing/incomplete/invalid layer makes only its own ratio ``NaN``.
- Code switching requires every target word token to carry a non-empty
  ``token.language``; the ratio counts values not equal to the document
  language (NFC + casefold for comparison only).
- Dependency features use only explicit ``dep_rel``/``dep_head``. Every
  representative target word token needs a non-empty relation; an utterance's
  representative word tokens
  need exactly one root (``dep_rel == "root"`` with head ``None`` or itself);
  every other chain must reach that root through same-utterance heads without
  revisiting a token (cycles), or all 17 dependency distribution/structure
  keys are ``NaN`` with ``MISSING_ANNOTATION``.

Locked formulas (all ratios over target word tokens; population SDs)
--------------------------------------------------------------------
- UPOS ratios count each of the 17 tags over representative target word tokens.
  Content tags ``ADJ, ADV, NOUN, PROPN, VERB`` and function tags
  ``ADP, AUX, CCONJ, DET, PART, PRON, SCONJ`` divide by representative target word
  tokens; ``noun_verb = (NOUN + PROPN) / VERB`` and
  ``pronoun_noun = PRON / (NOUN + PROPN)``; a zero denominator is ``NaN`` +
  ``INSUFFICIENT_TOKENS``.
- Dependency relations are NFC + casefolded before grouping: root ``root``;
  subject ``nsubj, nsubj:pass, csubj, csubj:pass``; object ``obj, iobj``;
  nominal modifier ``amod, nmod, nmod:poss, appos, compound``; adverbial
  modifier ``advmod, obl, obl:agent, dislocated``; clausal complement
  ``ccomp, xcomp, advcl, acl, acl:relcl``; coordination ``conj, cc``;
  function ``case, mark, det, aux, aux:pass, cop, clf``; every remaining
  non-empty relation is ``other``. The nine ratios partition the target word
  tokens and sum to one.
- Dependency length is ``abs(dependent position - head position)`` in the
  same utterance for each non-root word token; the mean pools all edges and
  the SD is the population SD (mean needs one edge, SD two).
- Tree depth is the number of head edges from a word token to its utterance
  root (root depth 0); the mean covers representative target word tokens.
- Clause heads are tokens with relation in
  ``root, ccomp, xcomp, advcl, acl, acl:relcl, csubj, csubj:pass``;
  subordinate clause heads exclude ``root``. Clause rate divides by the
  target utterance count; subordination ratio divides by the clause count.
- Classifier/particle ratios are the mean of their complete ``0``/``1``
  layers.

Conversation
------------
Each explicit :class:`DocumentUtterance` is one turn; the complete document
is ordered by ``(start_s, end_s, id)`` and only target turns appear in output
rows and recording summaries. Word/syllable turn counts reuse the Task 8
explicit per-utterance word grouping. A target turn is examiner-prompted when
the immediately preceding ordered turn belongs to a non-target speaker;
prompt ratio divides prompted target turns by target turns. Response latency
is ``max(0, target.start_s - previous_non_target.end_s)`` for prompted target
turns only; the recording mean pools defined latencies and the SD needs two.
Utterance overlap is the union duration of intersections of the target
interval with all non-target intervals; recording overlap is the union of all
such intersections, and the overlap ratio divides by the union duration of
target intervals. With no non-target turns, prompt/latency/overlap values are
``NaN`` + ``MISSING_ANNOTATION`` while turn counts/lengths stay defined;
without a target language sample all 51 recording keys are ``NaN`` +
``MISSING_ANNOTATION``.
"""

from __future__ import annotations

import math
from collections import Counter

from ...result import FeatureIssue
from ...schema import TASK_SPEC_FIELDS, nfc, validate_task_spec
from . import (
    _mean,
    _normalise,
    _population_sd,
    _resolve_target_speaker,
    _word_representatives,
    _word_syllable_lengths,
)
from .definitions import (
    MORPH_COMPOSITION_KEYS,
    MORPH_DEP_DIST_KEYS,
    MORPH_STRUCTURE_KEYS,
    MORPH_UPOS_KEYS,
    TASK9_KEYS,
    TASK9_RECORDING_KEYS,
)

UPOS_TAGS = frozenset(
    {
        "ADJ",
        "ADP",
        "ADV",
        "AUX",
        "CCONJ",
        "DET",
        "INTJ",
        "NOUN",
        "NUM",
        "PART",
        "PRON",
        "PROPN",
        "PUNCT",
        "SCONJ",
        "SYM",
        "VERB",
        "X",
    }
)
UPOS_ORDER = (
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
)
CONTENT_TAGS = frozenset({"ADJ", "ADV", "NOUN", "PROPN", "VERB"})
FUNCTION_TAGS = frozenset({"ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ"})

_UPOS_COMPOSITION_KEYS = (
    "morph_content_word_ratio",
    "morph_function_word_ratio",
    "morph_noun_verb_ratio",
    "morph_pronoun_noun_ratio",
)

_CLAUSE_HEADS = frozenset(
    {"root", "ccomp", "xcomp", "advcl", "acl", "acl:relcl", "csubj", "csubj:pass"}
)

_DEP_GROUPS = {
    "root": frozenset({"root"}),
    "subject": frozenset({"nsubj", "nsubj:pass", "csubj", "csubj:pass"}),
    "object": frozenset({"obj", "iobj"}),
    "nominal_modifier": frozenset({"amod", "nmod", "nmod:poss", "appos", "compound"}),
    "adverbial_modifier": frozenset({"advmod", "obl", "obl:agent", "dislocated"}),
    "clausal_complement": frozenset({"ccomp", "xcomp", "advcl", "acl", "acl:relcl"}),
    "coordination": frozenset({"conj", "cc"}),
    "function": frozenset({"case", "mark", "det", "aux", "aux:pass", "cop", "clf"}),
}
_DEP_GROUP_ORDER = (
    "root",
    "subject",
    "object",
    "nominal_modifier",
    "adverbial_modifier",
    "clausal_complement",
    "coordination",
    "function",
    "other",
)


def _issue(
    recording_id: str,
    speaker_id: str,
    code: str,
    message: str,
    feature: str,
    utterance_id: str | None = None,
) -> FeatureIssue:
    return FeatureIssue(
        recording_id=recording_id,
        speaker_id=speaker_id,
        code=code,
        severity="warning",
        message=message,
        feature=feature,
        utterance_id=utterance_id,
    )


def _flag(
    features: dict,
    keys,
    code: str,
    message: str,
    recording_id: str,
    speaker_id: str,
    issues: list,
    utterance_id: str | None = None,
) -> None:
    for key in keys:
        features[key] = math.nan
        issues.append(
            _issue(recording_id, speaker_id, code, message, feature=key, utterance_id=utterance_id)
        )


def _dep_group(relation: str) -> str:
    """Map one casefolded relation to its locked dependency group."""
    for name, relations in _DEP_GROUPS.items():
        if relation in relations:
            return name
    return "other"


def _upos_forms(document, word_tokens) -> list[str] | None:
    """Uppercase UPOS values for every target word token, or ``None`` when the
    ``upos`` layer is absent, incomplete, non-string, or not one of the 17
    locked tags."""
    layer = next((a for a in document.annotations if a.layer == "upos"), None)
    if layer is None:
        return None
    forms: list[str] = []
    for token in word_tokens:
        value = layer.values.get(token.id)
        if not isinstance(value, str) or not value:
            return None
        form = nfc(value).upper()
        if form not in UPOS_TAGS:
            return None
        forms.append(form)
    return forms


def _binary_forms(document, word_tokens, layer_name: str) -> list[float] | None:
    """Numeric ``0``/``1`` values for every target word token, or ``None``
    when the layer is absent, incomplete, or holds non-numeric/non-binary
    values (booleans are invalid)."""
    layer = next((a for a in document.annotations if a.layer == layer_name), None)
    if layer is None:
        return None
    forms: list[float] = []
    for token in word_tokens:
        value = layer.values.get(token.id)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if value not in (0, 1):
            return None
        forms.append(float(value))
    return forms


def _languages_complete(word_tokens) -> bool:
    return all(isinstance(token.language, str) and token.language for token in word_tokens)


def _dependency_analysis(target_utterances):
    """Validate and trace the explicit dependency structure.

    Returns ``(depths, edges, clause_count, subordinate_count)`` over the
    target word tokens, or ``None`` when any target utterance with word
    tokens fails the locked validity rules: every word token needs a
    non-empty relation, exactly one root per utterance, and every head chain
    must reach that root through same-utterance heads without cycles.
    """
    depths: list[float] = []
    edges: list[float] = []
    clause_count = 0
    subordinate_count = 0
    for utterance in target_utterances:
        word_tokens = _word_representatives([utterance])
        if not word_tokens:
            continue
        if any(
            not isinstance(token.dep_rel, str) or not token.dep_rel.strip() for token in word_tokens
        ):
            return None
        positions = {token.id: index for index, token in enumerate(utterance.tokens)}
        by_id = {token.id: token for token in utterance.tokens}
        roots = [token for token in word_tokens if _normalise(token.dep_rel or "") == "root"]
        if len(roots) != 1:
            return None
        root = roots[0]
        if root.dep_head not in (None, root.id):
            return None
        depth_by_id: dict[str, int] = {}
        for token in word_tokens:
            depth = 0
            seen = {token.id}
            current = token
            while current.dep_head not in (None, current.id):
                head = by_id.get(current.dep_head)
                if head is None or head.id in seen:
                    return None
                seen.add(head.id)
                current = head
                depth += 1
            if current.id != root.id:
                return None
            depth_by_id[token.id] = depth
        depths.extend(depth_by_id.values())
        for token in word_tokens:
            if token.id == root.id:
                continue
            edges.append(float(abs(positions[token.id] - positions[token.dep_head])))
        clause_heads = [
            token for token in word_tokens if _normalise(token.dep_rel or "") in _CLAUSE_HEADS
        ]
        clause_count += len(clause_heads)
        subordinate_count += sum(
            1 for token in clause_heads if _normalise(token.dep_rel or "") != "root"
        )
    return depths, edges, clause_count, subordinate_count


def _union_duration(intervals) -> float:
    """Union duration of ``[start, end)`` intervals."""
    total = 0.0
    current_start = None
    current_end = None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        elif end > current_end:
            current_end = end
    if current_end is not None:
        total += current_end - current_start
    return total


def _turn_word_syllable_counts(utterance) -> tuple[int, int]:
    """Explicit Task 8 word grouping: (words, syllables) of one utterance."""
    lengths = _word_syllable_lengths([utterance])
    return len(lengths), sum(lengths)


def extract_morphosyntax_features(
    document,
    *,
    target_speaker=None,
    recording_id: str = "",
) -> tuple[dict[str, float], list[dict], tuple[FeatureIssue, ...]]:
    """Extract the adult-neuro morphosyntax and conversation features.

    ``document`` is a :class:`~speech_features.document.SpeechDocument`;
    ``target_speaker`` resolves like the acoustic and lexical packs. Returns
    ``(recording_features, utterance_rows, issues)``: every one of the 51
    registered recording keys is present (``NaN`` when its prerequisite is
    unavailable) and each ``NaN`` is paired with exactly one per-key warning
    issue. Each target turn yields one plain dict row with ``utterance_id``,
    ``start_s``, ``end_s``, then the four utterance keys. Formula and
    denominator details are documented in this module's docstring.
    """
    features = {key: math.nan for key in TASK9_RECORDING_KEYS}
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
            TASK9_RECORDING_KEYS,
            "MISSING_ANNOTATION",
            "no explicit target-speaker language sample; morphosyntax features unavailable",
            recording_id,
            speaker_id,
            issues,
        )
        return features, [], tuple(issues)

    word_tokens = _word_representatives(utterances)
    n_words = float(len(word_tokens))

    # UPOS distribution and the four UPOS-derived composition ratios.
    if not word_tokens:
        _flag(
            features,
            MORPH_UPOS_KEYS + _UPOS_COMPOSITION_KEYS,
            "INSUFFICIENT_TOKENS",
            "no word-kind tokens; UPOS ratios unavailable",
            recording_id,
            speaker_id,
            issues,
        )
    else:
        upos = _upos_forms(document, word_tokens)
        if upos is None:
            _flag(
                features,
                MORPH_UPOS_KEYS + _UPOS_COMPOSITION_KEYS,
                "MISSING_ANNOTATION",
                "upos annotation layer is absent or incomplete for the target word tokens",
                recording_id,
                speaker_id,
                issues,
            )
        else:
            counts = Counter(upos)
            for tag, key in zip(UPOS_ORDER, MORPH_UPOS_KEYS):
                features[key] = counts[tag] / n_words
            content = sum(counts[tag] for tag in CONTENT_TAGS)
            function = sum(counts[tag] for tag in FUNCTION_TAGS)
            verb = counts["VERB"]
            noun_propn = counts["NOUN"] + counts["PROPN"]
            features["morph_content_word_ratio"] = content / n_words
            features["morph_function_word_ratio"] = function / n_words
            if verb:
                features["morph_noun_verb_ratio"] = noun_propn / verb
            else:
                _flag(
                    features,
                    ("morph_noun_verb_ratio",),
                    "INSUFFICIENT_TOKENS",
                    "zero verb-token denominator; noun/verb ratio unavailable",
                    recording_id,
                    speaker_id,
                    issues,
                )
            if noun_propn:
                features["morph_pronoun_noun_ratio"] = counts["PRON"] / noun_propn
            else:
                _flag(
                    features,
                    ("morph_pronoun_noun_ratio",),
                    "INSUFFICIENT_TOKENS",
                    "zero noun+proper-noun denominator; pronoun/noun ratio unavailable",
                    recording_id,
                    speaker_id,
                    issues,
                )

    # Classifier/particle ratios over their complete 0/1 layers.
    for key, layer_name in (
        ("morph_classifier_ratio", "classifier"),
        ("morph_particle_ratio", "particle"),
    ):
        if not word_tokens:
            _flag(
                features,
                (key,),
                "INSUFFICIENT_TOKENS",
                "no word-kind tokens; ratio unavailable",
                recording_id,
                speaker_id,
                issues,
            )
            continue
        forms = _binary_forms(document, word_tokens, layer_name)
        if forms is None:
            _flag(
                features,
                (key,),
                "MISSING_ANNOTATION",
                f"{layer_name} annotation layer is absent or incomplete for the target word tokens",
                recording_id,
                speaker_id,
                issues,
            )
        else:
            features[key] = sum(forms) / n_words

    # Code-switch ratio over explicit per-token languages.
    if _languages_complete(word_tokens):
        if n_words:
            non_vie = sum(
                1
                for token in word_tokens
                if _normalise(token.language) != _normalise(document.language)
            )
            features["morph_code_switch_ratio"] = non_vie / n_words
        else:
            _flag(
                features,
                ("morph_code_switch_ratio",),
                "INSUFFICIENT_TOKENS",
                "no word-kind tokens; code-switch ratio unavailable",
                recording_id,
                speaker_id,
                issues,
            )
    else:
        _flag(
            features,
            ("morph_code_switch_ratio",),
            "MISSING_ANNOTATION",
            "token language is missing for target word tokens",
            recording_id,
            speaker_id,
            issues,
        )

    # Dependency distribution, structure, and clause keys.
    analysis = _dependency_analysis(utterances)
    if analysis is None:
        _flag(
            features,
            MORPH_DEP_DIST_KEYS + MORPH_STRUCTURE_KEYS,
            "MISSING_ANNOTATION",
            "dependency annotations are missing, cyclic, or malformed for the target word tokens",
            recording_id,
            speaker_id,
            issues,
        )
    else:
        depths, edges, clause_count, subordinate_count = analysis
        relations = [_normalise(token.dep_rel or "") for token in word_tokens]
        group_counts = Counter(_dep_group(relation) for relation in relations)
        if n_words:
            for group, key in zip(_DEP_GROUP_ORDER, MORPH_DEP_DIST_KEYS):
                features[key] = group_counts[group] / n_words
        else:
            _flag(
                features,
                MORPH_DEP_DIST_KEYS,
                "INSUFFICIENT_TOKENS",
                "no word-kind tokens; dependency ratios unavailable",
                recording_id,
                speaker_id,
                issues,
            )
        if edges:
            features["morph_dependency_length_mean_tokens"] = _mean(edges)
            if len(edges) >= 2:
                features["morph_dependency_length_sd_tokens"] = _population_sd(edges)
            else:
                _flag(
                    features,
                    ("morph_dependency_length_sd_tokens",),
                    "INSUFFICIENT_TOKENS",
                    "fewer than two dependency edges; dependency-length SD unavailable",
                    recording_id,
                    speaker_id,
                    issues,
                )
        else:
            _flag(
                features,
                ("morph_dependency_length_mean_tokens", "morph_dependency_length_sd_tokens"),
                "INSUFFICIENT_TOKENS",
                "no dependency edges; dependency length unavailable",
                recording_id,
                speaker_id,
                issues,
            )
        if depths:
            features["morph_tree_depth_mean"] = _mean(depths)
            features["morph_tree_depth_max"] = float(max(depths))
        else:
            _flag(
                features,
                ("morph_tree_depth_mean", "morph_tree_depth_max"),
                "INSUFFICIENT_TOKENS",
                "no word-kind tokens; tree depth unavailable",
                recording_id,
                speaker_id,
                issues,
            )
        features["morph_clause_count"] = float(clause_count)
        features["morph_subordinate_clause_count"] = float(subordinate_count)
        if utterances:
            features["morph_clause_rate_per_utterance"] = clause_count / len(utterances)
        else:
            _flag(
                features,
                ("morph_clause_rate_per_utterance",),
                "INSUFFICIENT_TOKENS",
                "no target utterances; clause rate unavailable",
                recording_id,
                speaker_id,
                issues,
            )
        if clause_count:
            features["morph_subordination_ratio"] = subordinate_count / clause_count
        else:
            _flag(
                features,
                ("morph_subordination_ratio",),
                "INSUFFICIENT_TOKENS",
                "zero clause count; subordination ratio unavailable",
                recording_id,
                speaker_id,
                issues,
            )

    # Conversation: ordered turns, prompts, latencies, and overlap unions.
    ordered = sorted(document.utterances, key=lambda u: (u.start_s, u.end_s, u.id))
    target_turns = [u for u in ordered if u.speaker_id == speaker_id]
    has_non_target = any(u.speaker_id != speaker_id for u in ordered)
    by_index = {u.id: index for index, u in enumerate(ordered)}

    word_lengths = []
    syllable_lengths = []
    for turn in target_turns:
        words, syllables = _turn_word_syllable_counts(turn)
        word_lengths.append(float(words))
        syllable_lengths.append(float(syllables))
    features["discourse_turn_count"] = float(len(target_turns))
    features["discourse_turn_length_mean_words"] = _mean(word_lengths)
    features["discourse_turn_length_mean_syllables"] = _mean(syllable_lengths)
    for key, values in (
        ("discourse_turn_length_sd_words", word_lengths),
        ("discourse_turn_length_sd_syllables", syllable_lengths),
    ):
        if len(values) >= 2:
            features[key] = _population_sd(values)
        else:
            _flag(
                features,
                (key,),
                "INSUFFICIENT_TOKENS",
                "fewer than two target turns; turn-length SD unavailable",
                recording_id,
                speaker_id,
                issues,
            )

    rows: list[dict] = []
    for turn in target_turns:
        index = by_index[turn.id]
        prompted = index > 0 and ordered[index - 1].speaker_id != speaker_id
        row = {
            "utterance_id": turn.id,
            "start_s": turn.start_s,
            "end_s": turn.end_s,
        }
        words, syllables = _turn_word_syllable_counts(turn)
        row["discourse_turn_word_count"] = float(words)
        row["discourse_turn_syllable_count"] = float(syllables)
        if has_non_target and prompted:
            row["discourse_response_latency_s"] = max(0.0, turn.start_s - ordered[index - 1].end_s)
        else:
            row["discourse_response_latency_s"] = math.nan
            issues.append(
                _issue(
                    recording_id,
                    speaker_id,
                    "MISSING_ANNOTATION",
                    "target turn has no preceding non-target turn; response latency unavailable",
                    feature="discourse_response_latency_s",
                    utterance_id=turn.id,
                )
            )
        if has_non_target:
            intersections = []
            for other in ordered:
                if other.speaker_id == speaker_id:
                    continue
                start = max(turn.start_s, other.start_s)
                end = min(turn.end_s, other.end_s)
                if end > start:
                    intersections.append((start, end))
            row["discourse_turn_overlap_s"] = _union_duration(intersections)
        else:
            row["discourse_turn_overlap_s"] = math.nan
            issues.append(
                _issue(
                    recording_id,
                    speaker_id,
                    "MISSING_ANNOTATION",
                    "no non-target turns; utterance overlap unavailable",
                    feature="discourse_turn_overlap_s",
                    utterance_id=turn.id,
                )
            )
        rows.append(row)

    if has_non_target:
        prompt_count = sum(
            1
            for turn in target_turns
            if by_index[turn.id] > 0 and ordered[by_index[turn.id] - 1].speaker_id != speaker_id
        )
        features["discourse_examiner_prompt_ratio"] = prompt_count / len(target_turns)
        latencies = [
            max(0.0, turn.start_s - ordered[by_index[turn.id] - 1].end_s)
            for turn in target_turns
            if by_index[turn.id] > 0 and ordered[by_index[turn.id] - 1].speaker_id != speaker_id
        ]
        if latencies:
            features["discourse_response_latency_mean_s"] = _mean(latencies)
            if len(latencies) >= 2:
                features["discourse_response_latency_sd_s"] = _population_sd(latencies)
            else:
                _flag(
                    features,
                    ("discourse_response_latency_sd_s",),
                    "INSUFFICIENT_TOKENS",
                    "fewer than two defined target latencies; latency SD unavailable",
                    recording_id,
                    speaker_id,
                    issues,
                )
        else:
            _flag(
                features,
                ("discourse_response_latency_mean_s", "discourse_response_latency_sd_s"),
                "MISSING_ANNOTATION",
                "no examiner-prompted target turns; response latency unavailable",
                recording_id,
                speaker_id,
                issues,
            )
        intersection_intervals = []
        for turn in target_turns:
            for other in ordered:
                if other.speaker_id == speaker_id:
                    continue
                start = max(turn.start_s, other.start_s)
                end = min(turn.end_s, other.end_s)
                if end > start:
                    intersection_intervals.append((start, end))
        overlap_s = _union_duration(intersection_intervals)
        features["discourse_overlap_s"] = overlap_s
        target_union = _union_duration([(turn.start_s, turn.end_s) for turn in target_turns])
        if target_union:
            features["discourse_overlap_ratio"] = overlap_s / target_union
        else:
            _flag(
                features,
                ("discourse_overlap_ratio",),
                "INSUFFICIENT_TOKENS",
                "zero target utterance duration; overlap ratio unavailable",
                recording_id,
                speaker_id,
                issues,
            )
    else:
        _flag(
            features,
            (
                "discourse_examiner_prompt_ratio",
                "discourse_response_latency_mean_s",
                "discourse_response_latency_sd_s",
                "discourse_overlap_s",
                "discourse_overlap_ratio",
            ),
            "MISSING_ANNOTATION",
            "no non-target turns; conversation measures unavailable",
            recording_id,
            speaker_id,
            issues,
        )

    return features, rows, tuple(issues)


def extract_adult_neuro_features(
    document,
    *,
    target_speaker=None,
    recording_id: str = "",
    task_spec=None,
    _validated_task_spec: bool = False,
) -> tuple[dict[str, float], list[dict], tuple[FeatureIssue, ...]]:
    """Compose lexical, morphosyntax, clinical-linguistic, and task features
    into the adult-neuro pack output, without building a
    :class:`~speech_features.result.FeatureBundle` (Task 10 owns that
    integration)."""
    if task_spec is not None and not _validated_task_spec:
        validate_task_spec(task_spec)

    from . import extract_lexical_features  # deferred: sibling entry point

    lexical, lexical_issues = extract_lexical_features(
        document, target_speaker=target_speaker, recording_id=recording_id
    )
    morphosyntax, rows, morphosyntax_issues = extract_morphosyntax_features(
        document, target_speaker=target_speaker, recording_id=recording_id
    )
    from .clinical import extract_clinical_linguistic_features
    from .task_scores import (
        extract_structured_task_features,
        unavailable_structured_task_features,
    )

    clinical, clinical_issues = extract_clinical_linguistic_features(
        document,
        target_speaker=target_speaker,
        recording_id=recording_id,
        task_spec=task_spec,
        _validated_task_spec=True,
    )
    if task_spec is None or task_spec.get("task") not in TASK_SPEC_FIELDS:
        speaker_id = _resolve_target_speaker(document, target_speaker)
        task, task_issues = unavailable_structured_task_features(recording_id, speaker_id)
    else:
        task, task_issues = extract_structured_task_features(
            document,
            task_spec,
            target_speaker=target_speaker,
            recording_id=recording_id,
            _validated_task_spec=True,
        )
    features = {**lexical, **morphosyntax, **clinical, **task}
    return (
        features,
        rows,
        lexical_issues + morphosyntax_issues + clinical_issues + task_issues,
    )


__all__ = [
    "MORPH_COMPOSITION_KEYS",
    "MORPH_DEP_DIST_KEYS",
    "MORPH_STRUCTURE_KEYS",
    "MORPH_UPOS_KEYS",
    "TASK9_KEYS",
    "extract_adult_neuro_features",
    "extract_morphosyntax_features",
]
