# Task 11 — Legacy AD Compatibility and Notebook Retirement

Implements the locked Task 11 contract on top of `724adb6` per
`.superpowers/sdd/2026-08-06-say-vietnamese-speech-library/task-11-brief.md`
(Agy pre-review: APPROVE, no findings). Full evidence in the ignored
`.superpowers/sdd/2026-08-06-say-vietnamese-speech-library/task-11-report.md`.

## OpenCode implementation

### Changed and deleted files

- Created `src/speech_features/audio.py` — population-neutral PCM core moved
  out of the old AD pipeline: `InvalidAudioError`, `UnsupportedAudioError`,
  `read_wav`, and the private `_bytes_to_samples`, `_to_float`, `_resample`,
  `_read_wav_with_width`. One implementation only; every consumer imports it.
- Created `src/speech_features/legacy/__init__.py` (namespace marker) and
  `src/speech_features/legacy/ad/__init__.py` — lazy package `__getattr__`
  exposing the pipeline names (`BatchFailure`, `BatchResult`,
  `extract_recording`, `extract_manifest`) and the evaluation names
  (`C_GRID`, `EvaluationResult`, `KNOWN_TASKS`, `bootstrap_ci`,
  `evaluate_ad_baseline`); evaluation loads only when requested.
- Created `src/speech_features/legacy/ad/pipeline.py` — moved from
  `speech_features/pipeline.py` with only import paths fixed
  (`.acoustic`→`...acoustic`, `.schema`→`...schema`, `.linguistic`→
  `...linguistic`, `.tasks`→`.tasks`, PCM I/O from `...audio`). Numerical
  behavior, constants, dataclasses, errors, and `_TASK_SCORER` unchanged.
- Created `src/speech_features/legacy/ad/tasks.py` — moved from
  `speech_features/tasks.py`; only the `speech_features.linguistic` import
  was made package-relative.
- Created `src/speech_features/legacy/ad/evaluation.py` — moved verbatim
  from `speech_features/evaluation.py` (no relative imports needed).
- Replaced `src/speech_features/pipeline.py`, `src/speech_features/tasks.py`,
  and `src/speech_features/evaluation.py` with thin deprecation shims that
  only re-export the legacy objects and emit a `DeprecationWarning` naming
  the exact replacement import. No algorithm or copied business logic.
- Modified `src/speech_features/__init__.py` — eager legacy pipeline import
  removed; `read_wav`/`InvalidAudioError` come from `speech_features.audio`;
  `BatchFailure`, `BatchResult`, `extract_recording`, `extract_manifest`
  stay in `__all__` and resolve through a lazy module `__getattr__` that
  emits a replacement-naming `DeprecationWarning`.
- Modified `src/speech_features/result.py` — `InvalidAudioError` /
  `UnsupportedAudioError` now imported from `speech_features.audio`
  (identity preserved through `result`, root, and the pipeline shim).
- Modified `src/speech_features/extraction.py` — `_read_wav_with_width` now
  imported from `speech_features.audio`; docstring reference updated.
- Modified `src/speech_features/features/acoustic/__init__.py` — PCM reader
  import and two docstring references point at the shared audio core.
- Created `tests/speech_features/test_legacy.py` (19 tests: fresh-process
  import isolation, exact warning targets, object identity, `_TASK_SCORER`
  registry sharing, audio error identity, numerical parity, notebook
  retirement).
- Deleted five tracked notebooks via `git rm`:
  `src/speech_features/acoustics.ipynb`, `src/speech_features/linguistic.ipynb`,
  `src/classification/classifiers.ipynb`, `main.ipynb`,
  `notebooks/analysis/analyze.ipynb`. None of their obsolete, label-bound, or
  unfinished logic was converted into production modules.
- Untouched: `src/speech_features/acoustic.py` and
  `src/speech_features/linguistic.py` (shared tested helpers), `uv.lock`,
  `.serena/`, `.superpowers/` coordination artifacts, and all Task 12 files.

### RED (focused migration tests first, before any production edit)

```
$ uv run pytest tests/speech_features/test_legacy.py -q
17 failed, 2 passed in 5.23s
```

Failing: `TestImportIsolation::test_import_legacy_ad_is_warning_free_*`,
`test_root_read_wav_is_a_warning_free_core_export`, all three deprecated
module-warning tests, all four deprecated root-export tests, all object
identity tests, the `_TASK_SCORER` registry test, the audio error identity
test, both numerical parity tests, and `test_no_tracked_notebooks_remain`.
Only the import-ipynb source scan passed. Root cause: `legacy.ad` did not
exist, no shims warned, root exports were eager, and notebooks were tracked.

### GREEN (after moving implementations, adding shims, deleting notebooks)

```
$ uv run pytest tests/speech_features/test_legacy.py -q
19 passed in 5.59s

$ uv run pytest tests/speech_features -q
571 passed in 12.53s        (baseline was 552; all pre-existing tests unchanged)

$ uv run ruff check src/speech_features tests/speech_features
All checks passed!

$ uv run ruff format --check src/speech_features tests/speech_features
2 files reformatted (tasks.py shim, test_legacy.py), 45 already formatted;
check now passes.

$ uv run python -W error::DeprecationWarning -c "import speech_features; import speech_features.legacy.ad"
clean (no DeprecationWarning, no eager sklearn import)

$ git diff --check
clean
```

Forbidden-import scan (`import_ipynb|opensmile|librosa|spacy` under
`src/speech_features`) matches only the pre-existing acoustic.py docstring
prohibition statement ("No ... librosa/openSMILE"); no import anywhere.

### Compatibility decisions

- **One implementation, advisory shims.** Old module paths and the legacy
  namespace operate on the same underlying objects; shims contain only
  re-exports.
- **Object identity preserved.** `speech_features.pipeline.extract_recording
  is speech_features.legacy.ad.pipeline.extract_recording` (same for every
  task scorer, evaluation public name, and the private evaluation helpers
  `_default_pipe`/`_pick_c`/`_select_c` used by compatibility tests).
- **`_TASK_SCORER` is the same mutable dict object** through the shim and the
  legacy pipeline, so existing monkeypatch-based callers keep working.
- **Audio errors defined once** in `speech_features.audio`; the same class
  objects surface through `speech_features.audio`, `speech_features.result`,
  `speech_features`, and the deprecated `speech_features.pipeline` shim.
  `speech_features.read_wav` is a warning-free public core export.
- **Import isolation.** Plain `import speech_features` and
  `import speech_features.legacy.ad` are warning-free and never import
  scikit-learn; `speech_features.legacy.ad` loads evaluation lazily via
  package `__getattr__`.
- **Exact warning targets.** Each deprecated module path and root export
  emits a `DeprecationWarning` naming its precise replacement
  (e.g. `speech_features.legacy.ad.pipeline.extract_recording`), with
  removal in 0.3.0.
- **Numerical parity.** `extract_recording` through the shim and the legacy
  pipeline produces identical feature dicts, task, and input hashes.
- **No existing test changed.** All 552 baseline tests run unchanged; legacy
  tests exercise the shims and now observe (benign) deprecation warnings.
- `speech_features.acoustic` and `speech_features.linguistic` stay in place:
  their tested math helpers are shared by the new first-party packs and the
  legacy scorers; moving them would only create duplicate shim churn.

### Skipped scope

- Task 12 owns release documentation (README, feature-extraction,
  transcript-formats, catalog, migration guide, review summary) and the clean
  wheel install; nothing here.
- `uv.lock`, `.serena/`, and `.superpowers/` coordination artifacts were not
  touched; no dependencies were added.
- No tutorial/example notebooks or quickstart scaffolding were created.
- Legacy 0.1.x semantics (labels in batch results, AD evaluation) are
  preserved as-is; removal is deferred to 0.3.0.

## Agy review

Pending — reviewer (Agy) to verify specification compliance and code quality
against the brief; verdict to be recorded here.

## Resolution

Pending — controller to close the task after Agy approval.
