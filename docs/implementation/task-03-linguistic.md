# Task 03 — Vietnamese Linguistic & Task-Specific Features

Slug: `linguistic-tasks`. Owns `src/speech_features/linguistic.py`,
`src/speech_features/tasks.py`, `tests/speech_features/test_linguistic.py`,
`tests/speech_features/test_tasks.py`, and this report.

Goal (dispatch): *"Implement Vietnamese lexical/disfluency features and
picture, recall, and fluency scorers from external task specs."*

Scope (as instructed): participant-only transcript measures (token/character
counts, mean token length, TTR, hapax ratio, Brunet W, Honore R — all with safe
NaNs) plus repetition and filler rates and a simple external function-word-set
ratio; and external task-spec scorers for picture concepts, recall ideas,
phonemic initials/exclusions, and semantic items/subcategories, using exact
NFC/casefold match plus standard Levenshtein normalised similarity at a
documented conservative threshold. No diagnosis/labels and no NLP/ASR models.

## 1. OpenCode Implementation

### Approach (TDD)

Wrote the two behavioural test modules first, confirmed the red, then
implemented `linguistic.py` and `tasks.py`, then iterated to green.

**Red command / result (recorded before any production code):**

```
$ uv run pytest tests/speech_features/test_linguistic.py tests/speech_features/test_tasks.py -q
ERROR collecting tests/speech_features/test_linguistic.py
    import speech_features.linguistic as linguistic
E   ModuleNotFoundError: No module named 'speech_features.linguistic'
ERROR collecting tests/speech_features/test_tasks.py
    import speech_features.tasks as tasks
E   ModuleNotFoundError: No module named 'speech_features.tasks'
2 errors during collection
```

### What was built — `src/speech_features/linguistic.py`

- **`normalise_token(text)`** — deterministic Unicode NFC + `casefold`.
  `casefold` lower-cases (handling `Đ` → `đ`) while never conflating the
  distinct `d`/`đ` graphemes; diacritics are preserved; NFC composes combining
  marks (``o`` + combining circumflex == ``ô``). Idempotent.
- **`participant_tokens` / `participant_word_tokens`** — participant-only token
  streams; every measure below is computed exclusively from participant
  utterances (examiner speech is never used).
- **`token_stats`** — `lex_token_count`, `lex_word_count`, `lex_unique_count`,
  `lex_char_count` (sum of code points of participant word tokens only),
  `lex_mean_token_length`, and filler/fragment/noise counts.
- **`lexical_diversity`** — over normalised word forms: `lex_ttr` (V/N),
  `lex_hapax_ratio` (V1/N), `lex_brunet_w` (V^0.172), `lex_honore_r` =
  `100·log(N)/(1 − V1/V)`. All degeneracies return `NaN`: a zero-word transcript
  → all NaN; single-type transcript (V1 == V) → Honore denominator vanishes →
  NaN (never a bogus inf/zero).
- **`repetition_ratio`** — ratio of **adjacent** duplicate word tokens
  (immediate stutter-like repeats); spaced repeats do not count.
- **`raw_ratios`** — `lex_filler_ratio` / `lex_fragment_ratio` / `lex_noise_ratio`
  measured against all participant tokens.
- **`function_word_ratio`** — fraction of participant word tokens present in an
  externally supplied function-word set (NFC/casefolded on load); `None` set or
  no words → `NaN`.
- **`extract_linguistic(transcript, *, function_words=None)`** — the composite
  `lex_*` feature set.

### What was built — `src/speech_features/tasks.py`

- **`levenshtein(a, b)`** — classic edit distance (insert/delete/substitute).
- **`normalised_similarity(a, b)`** — `1 − distance / max(len(a), len(b))`,
  bounded `[0, 1]`; an empty-vs-empty pair scores `0` (never a spurious match).
- **`SIMILARITY_THRESHOLD = 0.85`** — the documented conservative matching
  threshold: it admits only near-exact variants (e.g. a dropped diacritic) and
  rejects genuinely unrelated words. Fixed editorial choice, not a tuned
  hyperparameter.
- **`match_alias` / `match_key`** — exact NFC/casefold equality first, then
  normalised-similarity at the threshold; the first hit in spec iteration order
  wins, so each word maps to at most one concept/idea/item.
- **`score_picture(transcript, spec)`** → `picture_concept_coverage` (matched
  distinct concepts / total), `picture_concept_density` (distinct matched /
  word count), `picture_repeat_ratio` ((matched − distinct)/matched), and
  `picture_entity_coverage` / `picture_action_coverage` from `entity_groups` /
  `action_groups`.
- **`score_recall(transcript, spec)`** → `recall_idea_coverage`, `recall_idea_density`,
  `recall_repeat_ratio`, and `recall_order_score` = normalized LCS of the
  matched-ideas sequence (utterance order) against the spec's reference idea
  order; NaN when nothing is recalled.
- **`score_phonemic(transcript, spec)`** — one participant utterance with word
  tokens is one response item. `fluency_response_count`, `fluency_valid_count`,
  `fluency_valid_unique`, `fluency_repeats`, `fluency_intrusions` (responses that
  fail the initial or hit `exclusions`), `fluency_first_half_valid` /
  `fluency_second_half_valid` (over utterance time slots),
  `fluency_production_change` (second − first half), and `fluency_rate`.
- **`score_semantic(transcript, spec)`** — `fluency_valid_unique`, `fluency_valid_count`,
  `fluency_repeats`, `fluency_clusters` (maximal runs of consecutive responses in
  the same `subcategory`), `fluency_switches` (adjacent subcategory changes),
  and `fluency_rate`.

Both modules import stdlib + internal `.schema`/`.linguistic` only. No
diagnosis/labels are read anywhere; no librosa/openSMILE/spaCy/torch/ASR.

## 2. Agy Review

Reviewer: `agy:code-reviewer`. Pending (this is the build commit; findings, if
any, are resolved below in Section 3 before approval).

## 3. Resolution

N/A for the initial build commit — no prior review findings. Three test-side
corrections were made during iteration (not production bugs): the NFC/casefold
vector expectation, the character-count total (Mèo+con = 6, not 7), and the
recall `order_score` returning NaN (not 0.0) when nothing was recalled. All
were already within the RED→GREEN TDD loop.

### RED

```
$ uv run pytest tests/speech_features/test_linguistic.py tests/speech_features/test_tasks.py -q
2 errors during collection (ModuleNotFoundError)   # modules absent
```

### GREEN (iterative)

After implementing both modules and the three test corrections:

```
$ uv run pytest tests/speech_features/test_linguistic.py tests/speech_features/test_tasks.py -q
45 passed in 0.03s
```

### Final verification

```
$ uv run pytest tests/speech_features -q
114 passed in 0.35s

$ uv run ruff check src/speech_features tests/speech_features
All checks passed!

$ uv run ruff format --check src/speech_features tests/speech_features
9 files already formatted

$ git diff --check
(clean)
```

No notebooks, no future-task files (`pipeline.py`, `evaluation.py`, docs
`task-02..06`), no plan/dispatch JSON, and no `.serena` were touched. Only the
four Task 3 owned paths and this report were modified.

## Changed files

- `src/speech_features/linguistic.py` (new)
- `src/speech_features/tasks.py` (new)
- `tests/speech_features/test_linguistic.py` (new)
- `tests/speech_features/test_tasks.py` (new)
- `docs/implementation/task-03-linguistic.md` (this report, new)

## Commit ID

- Build commit `<FILLED_ON_COMMIT>` — `feat: add vietnamese linguistic and task scorers`.

## Skipped scope

Deliberately not implemented: syllable counts, MATTR-20, normalized entropy,
mean utterance length, and the `time_*` (words/min) features that need
recording durations — these bridge into the Task 4 pipeline and are deferred
there. The semantic `inter-response intervals` time metric and the full AD
classification are out of Task 3's remit (Tasks 4/5).
