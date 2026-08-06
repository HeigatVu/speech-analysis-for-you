# Task 06 — Documentation & Verification

Slug: `documentation`. Owns `README.md`, `docs/feature-extraction.md`,
`tests/speech_features/test_documentation.py`, and this report
(`docs/implementation/task-06-documentation.md`).

Goal (dispatch): *"Document formulas/contracts/research limitation and run full
verification without changing legacy notebooks."*

This task does **not** touch any production module, notebook, or prior report. It
is documentation plus a test-first documentation guard.

---

## 1. OpenCode Implementation

### Approach (TDD — documentation checks first)

Wrote a failing documentation-check test module first
(`tests/speech_features/test_documentation.py`), confirmed the red, then wrote
`docs/feature-extraction.md`, then updated `README.md`, then iterated to green.

**Red command / result (recorded before any doc content existed):**

```
$ uv run pytest tests/speech_features/test_documentation.py -q
16 failed, 1 passed, 16 errors in 0.74s
```

The 16 errors were `docs/feature-extraction.md` missing entirely
(`FileNotFoundError` in a `docs/feature-extraction.md must exist` fixture); the
16 failures were the README missing the research-only limitation, the WAV/PCM
contract, and the public API symbols it must document.

### What was built

- **`docs/feature-extraction.md`** (new) — the authoritative feature-library
  reference, covering every item the dispatch asks to be documented:
  - **Input contracts** — standard PCM WAV (mono/stereo integer PCM, down-mixed
    to mono and re-sampled to the 16 kHz default with `resample_poly`, finite
    after decode and re-sample, structured `InvalidAudioError`); reviewed,
    aligned Vietnamese transcript JSON v1 (explicit `word`/`filler`/
    `fragment`/`noise` tokens, `participant`/`examiner` speakers, finite
    ordered timestamps, NFC normalisation, **_a transcript, not whitespace,
    defines Vietnamese word boundaries_**); and the one-`(participant_id, task)`-
    per-row manifest with `ExtractionInputs` kept separate from clinical fields.
  - **Task list / task-spec fields** — the six tasks and each task's required
    external spec fields (picture → concept/entity/action groups; recall → idea
    aliases; phonemic → initials/exclusions; semantic → item aliases/
    subcategories).
  - **Math-first formulas** — spelled out per feature family: timing/acoustic
    (Hamming frames, energy VAD two-part threshold, integer-frame pause runs,
    NCCF pitch in 70–400 Hz with `ceil`/`floor` lag bounds, `HNR =
    10*log10(r/(1-r))`, geometric/arithmetic spectral flatness,
    `P_k = |RFFT(wx)_k|^2` centroid/spread), lexical/diversity (TTR `V/N`,
    hapax `V1/N`, Brunet W `V^0.172`, Honore R `100*log N/(1 - V1/V)`, adjacent
    repeat ratio, filler/fragment/noise ratios), and task scorers (exact
    NFC/casefold-then-Levenshtein `1 - d/max(len)` at `SIMILARITY_THRESHOLD =
    0.85`, recall LCS order score, one-response-per-utterance fluency).
  - **Quality flags / provenance / label separation** — `invalid_input`,
    `too_short`, `no_voice`, `no_participant_speech`; streaming SHA-256 of
    audio/transcript/task-spec (`MissingInputError`); diagnosis read only at the
    batch layer into `BatchResult.labels`, never into a `FeatureResult`, with the
    evaluation label supplied separately.
  - **Batch failure behaviour** — per-row isolation as `BatchFailure` with a
    stable `code` (`MISSING_INPUT`, `INVALID_AUDIO`, `INVALID_TRANSCRIPT`,
    `INVALID_TASK_SPEC`, `UNKNOWN_TASK`, generic `EXTRACTION_ERROR`), never
    aborting the batch; `no_participant_speech` is a flag, not a failure.
  - **Cohort minimum / task coverage** — ≥2 of each class for any fold, default
    `min_task_coverage = 5` of 6 tasks warned (not crashed), one case per
    participant (NaN-aware mean), and cohort/task-coverage/feature-missingness
    summaries.
  - **Repeated nested grouped-CV research baseline** — one case per participant,
    grouped outer `StratifiedGroupKFold` (5 folds) and inner grouped CV (4)
    splitting on participants, median-impute + scale fitted inside each training
    fold only, L2 `C in (0.01, 0.1, 1, 10, 100)` selected by balanced accuracy on
    shared inner folds, AUROC / balanced accuracy / sensitivity / specificity
    with participant-bootstrap 95% percentile intervals (`bootstrap_ci`, 2000
    resamples), **and an explicit reproducible-seed procedure** (`seed`,
    `seed + repeat` for outer repeats, `seed + repeat*1000 + fold_id` for inner C,
    `random_state=seed` for bootstraps).
  - **A clear research-only limitation** at the top (**_not a diagnostic_** and
    **_not for clinical decision-making_**, no diagnosis output, no fixed score
    threshold) and **no fixed performance promise** anywhere.

- **`README.md`** (updated) — added a prominent *"Vietnamese speech feature
  library (research-only)"* block under Overview that states the research-only /
  not-a-diagnostic / no-fixed-performance limitation up front, advertises the
  math-first (NumPy/SciPy-only) constraint, the PCM-WAV + Vietnamese-transcript
  contracts with re-sampling/provenance, and links to
  `docs/feature-extraction.md`. The original pipeline README content (installation,
  structure, tasks, citation, license) is preserved unchanged; the Table of
  Contents gained the new subsection. Only additions — no existing README content
  was removed.

- **`tests/speech_features/test_documentation.py`** (new) — test-first guards
  that the docs must: state the research-only / not-a-diagnostic / clinical
  limitation and must **not** promise a fixed performance number; reference every
  public API symbol (`ExtractionConfig`, `FeatureResult`, `extract_recording`,
  `extract_manifest`, `read_wav`, `evaluate_ad_baseline`,
  `validate_manifest`, `validate_transcript`, `validate_task_spec`,
  `manifest_hashes`, `sha256_file`, `BatchFailure`, `BatchResult`, `nfc`);
  document the WAV/transcript/manifest contracts; document the math-first formulas
  (TTR, hapax, Levenshtein, NCCF, Hamming, flatness, nested CV, bootstrap); and
  cover quality flags, SHA-256 provenance, label separation, batch failure
  isolation, task coverage, seed/reproducibility, and CI. The test reads files
  from the repository root via `Path` — no web claims, no live-module imports.

No notebooks and no production modules (`schema.py`, `acoustic.py`,
`linguistic.py`, `tasks.py`, `pipeline.py`, `evaluation.py`, `__init__.py`) were
modified or imported differently; the forbidden-library search over
`src/speech_features/*.py` still shows only pre-existing docstring mentions, no
imports. No web-based performance claims were added.

---

## 2. Agy Review

Reviewer: `agy:code-reviewer`. Review of the Task 6 documentation commit.
Placeholder — this section is recorded now and updated in any review-fix commit;
the implementer does not edit Agy's wording.

---

## 3. Resolution

Not applicable in this commit — no Agy findings were logged on the build commit;
there is no review-fix round for this task yet. Should a finding require a doc or
test correction, it will be fixed test-first below and recorded here.

### RED / GREEN evidence

RED (documentation checks written, before any doc content):

```
$ uv run pytest tests/speech_features/test_documentation.py -q
16 failed, 1 passed, 16 errors in 0.74s
```

GREEN (after `docs/feature-extraction.md` and the README update):

```
$ uv run pytest tests/speech_features/test_documentation.py -q
33 passed in 0.01s

$ uv run pytest tests/speech_features -q
194 passed in 2.57s

$ uv run ruff check src/speech_features tests/speech_features
All checks passed!

$ uv run ruff format --check src/speech_features tests/speech_features
14 files already formatted

$ git diff --check
(clean)
```

Forbidden-library search over production modules (unchanged by this task):

```
$ grep -rnE "librosa|opensmile|spacy|spaCy|torch|transformers|soundfile|pydub" src/speech_features/*.py
  -> only pre-existing docstring mentions (acoustic.py:3, pipeline.py:16); no imports
```

### Commit ID

`faa35bf` — `docs: document feature library and verification`.

### Skipped scope

Deliberately not implemented: no changes to any notebook or production module —
this task is documentation-only. The README's generic pipeline sections
(installation, project structure, output JSON format, roadmap) are preserved
verbatim from the legacy README rather than rewritten, because Task 6's remit is
documenting the feature library and its research limitation, not re-authoring the
legacy pipeline docs. No external performance or clinical claims are made
anywhere, per the "do not promise fixed performance" and "no web claims"
instructions.
