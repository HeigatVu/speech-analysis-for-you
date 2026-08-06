# SAY Vietnamese Speech Feature Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: execute each task test-first. OpenCode implements with `deepseek/deepseek-v4-flash` at `max` effort. Agy reviews with `gemini-3.6-flash-high` at high effort. A task is incomplete until both specification and quality verdicts are approved.

**Goal:** Ship version 0.2.0 of SAY as a population-neutral Python speech feature library with first-party Vietnamese adult-neuro acoustic and linguistic packs.

**Architecture:** A versioned `SpeechDocument` is the boundary between manual/automated transcript workflows and feature extraction. Static first-party feature packs consume a shared extraction context and return recording/utterance tables plus structured issues and provenance. The current six-task AD workflow remains temporarily available under `speech_features.legacy.ad`.

**Tech Stack:** Python 3.10+, stdlib, NumPy, SciPy, pandas, pytest, Ruff, setuptools.

## Global Constraints

- OpenCode writes a focused failing pytest before production code and records the exact RED command/result in its task report.
- Production feature algorithms use stdlib, NumPy, and SciPy; do not import librosa, openSMILE, spaCy, torch, transformers, Batchalign, or ASR libraries from `src/speech_features`.
- All production logic is `.py`; no production module imports or executes a notebook.
- No diagnosis, label, clinical cutoff, normative score, patient data, or task lexicon enters the general extraction API or fixtures.
- Preserve Vietnamese NFC text, diacritics, tone marks, and `d`/`đ`; never infer word boundaries solely from whitespace.
- Unavailable denominators or prerequisites produce `NaN` plus a structured issue, never a fabricated zero.
- Only edit paths owned by the current task plus the task report. Preserve unrelated work.
- OpenCode runs focused tests and Ruff before committing. Agy receives the complete task diff and must approve specification compliance and code quality before the next task.
- Keep the implementation minimal: static built-in pack registration, stdlib `argparse`, no GUI/service/database/distributed executor/plugin discovery.

---

### Task 1: Specification and Packaging Boundary

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/speech_features/__init__.py`
- Create: `tests/speech_features/test_packaging.py`

**Produces:** installable version `0.2.0`, Python `>=3.10`, setuptools build backend, `say-features = speech_features.cli:main`, and dependency extras.

- [ ] Add tests that parse `pyproject.toml` and assert the exact version, Python floor, console script, build backend, core dependencies, optional extras, and absence of notebook/openSMILE/librosa runtime dependencies.
- [ ] Run the focused test and record the expected RED failure.
- [ ] Set core dependencies to `numpy>=1.26`, `scipy>=1.11`, and `pandas>=2.1`.
- [ ] Add `legacy-ad` with `scikit-learn>=1.3`; add `legacy-preprocessing` with `hydra-core>=1.3`, `omegaconf>=2.3`, `pydub>=0.25`, `matplotlib>=3.8`, and `tqdm>=4.67`; retain pytest/Ruff in `dev`.
- [ ] Add the console entry point even though `speech_features.cli` lands in Task 10; packaging tests must not import the missing module yet.
- [ ] Keep existing public imports working; do not reorganize production modules in this task.
- [ ] Run focused tests, the existing 194-test suite, and Ruff; commit `build: define SAY library package boundaries`.

### Task 2: Population-Neutral Document Model

**Files:**
- Create: `src/speech_features/document.py`
- Create: `src/speech_features/formats/__init__.py`
- Create: `src/speech_features/formats/json.py`
- Create: `tests/speech_features/test_document.py`
- Modify: `src/speech_features/__init__.py`

**Produces:** `AnnotationLayer`, `DocumentToken`, `DocumentUtterance`, `DocumentSpeaker`, `MediaRef`, `SpeechDocument`, `load_document`, and `save_document` for JSON v2/v1. These names deliberately avoid the deprecated AD `Token` and `Utterance` exports.

- [ ] Test frozen/deeply immutable values, duplicate IDs, finite ordered times, token-within-utterance times, valid dependency heads, annotation confidence in `[0, 1]`, and exactly ISO language `vie` by default.
- [ ] Test NFC preservation, `d` versus `đ`, multi-syllable tokens, shared `word_id`, token language for code switching, and no whitespace-based word splitting.
- [ ] Define JSON v2 with stable fields matching the design. Mapping fields must be immutable after validation.
- [ ] Migrate current JSON v1 deterministically: utterance IDs `u0001...`, token IDs `u0001_t0001...`, and each old word token becomes one word unless its source already declares grouping.
- [ ] Detect JSON format by extension/content when `format=None`; refuse overwrite unless `force=True`.
- [ ] Re-export only the new document API from the package root; do not remove current compatibility exports.
- [ ] Run focused/full tests and Ruff; commit `feat: add versioned speech document model`.

### Task 3: Clinical CHAT Interchange

**Files:**
- Create: `src/speech_features/formats/chat.py`
- Create: `tests/speech_features/test_chat.py`
- Modify: `src/speech_features/formats/__init__.py`
- Modify: `src/speech_features/document.py`

**Consumes:** Task 2 document classes and load/save dispatch.

**Produces:** CHAT read/write support for the documented clinical subset.

- [ ] Add Batchalign-shaped fixtures covering Vietnamese `@Languages: vie`, participant/examiner roles, `@Media`, utterance bullets, word bullets, `%mor`, `%gra`, fillers, fragments, `[/]`, `[//]`, errors, and an unknown dependent tier.
- [ ] Verify known data round-trips semantically; preserve unsupported lines/tiers verbatim and emit `UNSUPPORTED_CHAT_TIER`.
- [ ] Reject missing begin/end, malformed headers, unknown speaker references, malformed bullets, and incompatible `%mor`/`%gra` alignment with `INVALID_CHAT`.
- [ ] Never invoke Batchalign or merge separate transcript histories. Load the current file snapshot and record its input hash/origin.
- [ ] Run focused/full tests and Ruff; commit `feat: add clinical CHAT interchange`.

### Task 4: Feature Catalog and Result Contracts

**Files:**
- Create: `src/speech_features/catalog.py`
- Create: `src/speech_features/result.py`
- Create: `tests/speech_features/test_catalog.py`
- Modify: `src/speech_features/__init__.py`

**Produces:** `FeatureDefinition`, `FeatureIssue`, `FeatureBundle`, `ExtractionContext`, `FeaturePack`, `list_features`, stable error codes, and static pack registration.

- [ ] Test immutable definitions, unique stable keys, allowed levels (`recording`, `utterance`), allowed issue severities, DataFrame identifier columns, deterministic column ordering, and JSON-serializable provenance.
- [ ] Use feature prefixes from the design and catalog version `1`.
- [ ] Define stable errors: `INVALID_DOCUMENT`, `INVALID_CHAT`, `INVALID_AUDIO`, `UNSUPPORTED_AUDIO`, `MISSING_INPUT`, `TARGET_SPEAKER_REQUIRED`, `MISSING_ANNOTATION`, `UNKNOWN_PACK`, `INVALID_CONFIG`, and `EXTRACTION_ERROR`.
- [ ] Define one protocol shared by the two built-in packs and a static mapping; do not add entry-point discovery.
- [ ] Empty results retain identifier columns. Feature values are floats; missing values are `NaN` with issues.
- [ ] Run focused/full tests and Ruff; commit `feat: define feature catalog and result contracts`.

### Task 5: Acoustic Core, Audio Quality, and Timing

**Files:**
- Refactor: `src/speech_features/acoustic.py`
- Create as needed under: `src/speech_features/features/acoustic/`
- Extend: `tests/speech_features/test_acoustic.py`

**Consumes:** Task 2 document timings and Task 4 pack/result contracts.

**Produces:** shared acoustic analysis, PCM WAV handling, target-speaker isolation, quality measures, and timing measures.

- [ ] Preserve all current tested acoustic behavior unless the new speaker-isolation or formula contract explicitly supersedes it.
- [ ] Add synthetic tests for PCM widths/resampling, clipping/DC/RMS, silence, voiced-segment and pause summaries, and examiner/participant interval isolation.
- [ ] Use transcript-aligned target-speaker intervals. With multiple speakers and no target, raise `TARGET_SPEAKER_REQUIRED`. Whole-recording fallback requires explicit configuration and emits `UNALIGNED_SPEAKER`.
- [ ] Expose calibration only for sample rate, frame/hop seconds, pitch range, autocorrelation/VAD threshold, pause/long-pause threshold, and LPC order.
- [ ] Keep standard PCM WAV decoding and polyphase resampling. Unsupported media raises `UNSUPPORTED_AUDIO`.
- [ ] Register exact definition metadata for quality/timing keys and document formulas in docstrings.
- [ ] Run focused/full tests and Ruff; commit `feat: add acoustic quality and timing core`.

### Task 6: Acoustic Phonation and Prosody

**Files:**
- Extend modules under: `src/speech_features/features/acoustic/`
- Extend: `tests/speech_features/test_acoustic.py`

**Consumes:** Task 5 shared acoustic context and Task 4 result/catalog contracts.

**Produces:** F0, intensity, jitter, shimmer, HNR, and CPP features.

- [ ] Add synthetic tests for a 200 Hz tone, controlled F0 slope/change, amplitude modulation, period modulation, harmonic-plus-noise HNR monotonicity, and CPP tonal/noise ordering.
- [ ] Compute F0 mean, median, SD, CV, IQR, 5--95% range, slope, and absolute change; voiced ratio; intensity summaries/slope; local jitter/shimmer; HNR summaries; and CPP summaries.
- [ ] Require sufficient voiced frames/cycles; otherwise emit `NaN` and a specific issue rather than zero.
- [ ] Respect only the calibration controls fixed in Task 5; add no per-feature options.
- [ ] Register exact definition metadata and formula docstrings.
- [ ] Run focused/full tests and Ruff; commit `feat: add acoustic phonation and prosody features`.

### Task 7: Acoustic Resonance, Spectrum, and Rhythm

**Files:**
- Extend modules under: `src/speech_features/features/acoustic/`
- Extend: `tests/speech_features/test_acoustic.py`

**Consumes:** Tasks 5--6 acoustic context and Task 2 syllable timing.

**Produces:** LPC formants/bandwidths, spectral summaries, and rhythm features; completes built-in `acoustic` pack version `1`.

- [ ] Add synthetic source-filter vowel tests for F1--F3/bandwidth tolerances, tone/noise tests for centroid/spread/slope/rolloff/flux/flatness/entropy, and annotated syllable-duration variability tests.
- [ ] Require sufficient stable frames/formant roots/annotations; otherwise emit `NaN` and specific issues.
- [ ] Register exact metadata for every remaining acoustic key and make `list_features(pack="acoustic")` complete and deterministic.
- [ ] Run focused/full tests and Ruff; commit `feat: complete acoustic resonance spectrum and rhythm pack`.

### Task 8: Adult-Neuro Lexical and Disfluency Features

**Files:**
- Refactor: `src/speech_features/linguistic.py`
- Create as needed under: `src/speech_features/features/linguistic/`
- Create/extend: `tests/speech_features/test_linguistic.py`

**Consumes:** Task 2 annotations and Task 4 pack/result contracts.

**Produces:** adult-neuro lexical diversity, counts, lengths, MLU, and disfluency features.

- [ ] Add hand-calculated fixtures for counts, MLU words/syllables, TTR, MATTR with window 20, MTLD threshold `0.72`, HD-D sample size `42`, hapax, Brunet W, Honore R, entropy, and length summaries.
- [ ] Add fillers, fragments, immediate repetition, retracing, revisions, mazes, and annotated-error measures.
- [ ] Count explicit words and syllables independently. Preserve diacritics and `d`/`đ`; normalize only for comparisons.
- [ ] Each annotation-dependent feature returns `NaN` plus `MISSING_ANNOTATION`; do not add a tokenizer, tagger, parser, ASR, embedding, or task scorer.
- [ ] Register exact definition metadata for every emitted key and document formulas in docstrings.
- [ ] Run focused/full tests and Ruff; commit `feat: add Vietnamese adult neuro lexical features`.

### Task 9: Adult-Neuro Morphosyntax and Conversation Features

**Files:**
- Extend modules under: `src/speech_features/features/linguistic/`
- Extend: `tests/speech_features/test_linguistic.py`

**Consumes:** Task 8 linguistic context and Task 2 annotations/timing.

**Produces:** UPOS/dependency, Vietnamese composition, and conversation measures; completes built-in `adult_neuro` pack version `1`.

- [ ] Add reviewed-annotation fixtures for content/function, noun/verb, pronoun/noun, classifiers, particles, code switching, dependency distributions/length/tree depth, clauses, and subordination.
- [ ] Add turn counts/lengths, examiner-prompt ratio, response latency, and overlap at recording and utterance levels where defined.
- [ ] Missing required layers/timing emit `NaN` plus `MISSING_ANNOTATION`; never infer unavailable morphology or syntax.
- [ ] Register exact metadata and make `list_features(pack="adult_neuro")` complete and deterministic.
- [ ] Run focused/full tests and Ruff; commit `feat: complete adult neuro morphosyntax and conversation pack`.

### Task 10: Label-Free Pipeline and CLI

**Files:**
- Create: `src/speech_features/extraction.py`
- Create: `src/speech_features/cli.py`
- Create/extend: `tests/speech_features/test_pipeline.py`
- Create: `tests/speech_features/test_cli.py`
- Modify: `src/speech_features/__init__.py`

**Produces:** public `extract`, `extract_batch`, and four `say-features` subcommands.

- [ ] Define manifest v2 with `version`, non-empty `rows`, `recording_id`, `audio_path`, `transcript_path`, and optional `target_speakers`. Reject diagnosis and task fields in the general manifest.
- [ ] Compose selected packs once per recording, reuse shared audio/document context, and produce deterministic recording/utterance/issue tables.
- [ ] Batch row failures are isolated; write `recordings.csv`, `utterances.csv`, `issues.csv`, and `provenance.json` even after partial failure.
- [ ] Implement stdlib `argparse`: `validate`, `convert`, `extract`, and `list-features`; exit `0` for success/warnings, `1` for partial row failures, and `2` for invalid global input.
- [ ] Hash every input and include package/catalog versions, configuration, packs, target speakers, and annotation sources in provenance.
- [ ] No labels, demographics, tasks, or clinical outputs enter feature tables.
- [ ] Leave the existing AD-oriented `pipeline.py` unchanged until Task 11; export the new functions from the package root and use them from the CLI.
- [ ] Run focused/full tests, CLI smoke tests, and Ruff; commit `feat: add label-free extraction pipeline and CLI`.

### Task 11: Legacy AD Compatibility and Notebook Cleanup

**Files:**
- Create: `src/speech_features/legacy/__init__.py`
- Create/refactor under: `src/speech_features/legacy/ad/`
- Modify compatibility modules and `src/speech_features/__init__.py`
- Modify/delete notebooks only as specified below
- Create: `tests/speech_features/test_legacy.py`

**Produces:** functional legacy AD imports with deprecation warnings and no production notebook logic.

- [ ] Move current task scorers, task manifest pipeline, and AD evaluation behind `speech_features.legacy.ad` without changing their existing numerical behavior.
- [ ] Replace the old module paths with thin compatibility shims after the implementations are safely present in the legacy namespace; the new general pipeline remains in `speech_features.extraction`.
- [ ] Keep old top-level `extract_recording`, `extract_manifest`, and evaluation imports working through `0.2.x`; emit `DeprecationWarning` pointing to the legacy namespace.
- [ ] Preserve all current legacy tests and add warning/import tests.
- [ ] Keep `src/speech_features/acoustics.ipynb` and `src/speech_features/linguistic.ipynb` deleted.
- [ ] Remove `src/classification/classifiers.ipynb` as production source. Any retained `main.ipynb` or `notebooks/analysis/analyze.ipynb` must become a thin tutorial that only imports the public API and reads synthetic/example inputs.
- [ ] Remove `import_ipynb` usage, notebook `%pip` installs, hard-coded machine paths, duplicated feature algorithms, and reusable configuration from notebooks.
- [ ] Run full tests, Ruff, and an import search; commit `refactor: isolate legacy AD workflow and retire notebook logic`.

### Task 12: Documentation, Packaging, and Release Verification

**Files:**
- Modify: `README.md`
- Replace/update: `docs/feature-extraction.md`
- Create: `docs/transcript-formats.md`
- Create: `docs/feature-catalog-v1.md`
- Create: `docs/migration-0.2.md`
- Create: `docs/implementation/say-library/review-summary.md`
- Extend documentation tests

**Produces:** user-facing version 0.2 documentation and a clean installable wheel.

- [ ] Document installation, Python and CLI quick starts, JSON v2, the CHAT subset, manual/automated transcript workflow, target-speaker isolation, tables, issues, provenance, and all stable errors.
- [ ] Document each catalog key with formula, unit, level, prerequisites, applicability, reference, and missing-data behavior.
- [ ] State research/descriptive use prominently: no diagnosis, cutoff, normative range, or treatment recommendation.
- [ ] Add a minimal future pediatric-pack example using the existing `FeaturePack` contract; do not implement pediatric features.
- [ ] Document 0.1-to-0.2 migration and AD legacy imports/deprecation window.
- [ ] Summarize OpenCode task commits, RED/GREEN evidence, Agy approvals, deferred minors, and final review in the review summary.
- [ ] Build the wheel and install it in a clean temporary environment; import the API and run `say-features --help`.
- [ ] Run the complete acceptance suite and commit `docs: ship SAY 0.2 library documentation`.

## Final Acceptance

Run fresh after the final Agy whole-branch review and any OpenCode fix wave:

```bash
rtk proxy uv run pytest tests/speech_features
rtk proxy uv run ruff check src/speech_features tests/speech_features
rtk proxy uv build
rtk git diff --check
rtk proxy rg "import_ipynb|opensmile|librosa|spacy" src/speech_features
```

The goal is complete only when every command succeeds, Agy approves the whole
branch, the wheel installs in a clean environment, every catalog item is tested
and documented, and no production notebook logic or unsupported clinical claim
remains.
