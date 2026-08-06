# Task 01 — Speech Feature Input Contracts

Slug: `contract`. Owns `pyproject.toml`, `src/speech_features/__init__.py`,
`src/speech_features/schema.py`, `tests/speech_features/test_schema.py`, and this
report.

Goal (dispatch): *"Create importable package, immutable extraction config/result
types, JSON/manifest validation, normalisation, provenance hashes, and structured
extraction errors."*

---

## 1. OpenCode Implementation

### Approach (TDD)

Wrote a failing test module first (`tests/speech_features/test_schema.py`),
confirmed the red, then implemented `schema.py` and the package `__init__.py`,
then iteration green.

**Red command / result (recorded before any production code):**

```
$ uv run pytest tests/speech_features/test_schema.py -q
ERROR tests/speech_features/test_schema.py
ModuleNotFoundError: No module named 'speech_features.schema'
1 error during collection
```

### What was built

- **`ExtractionConfig`** — frozen dataclass fixing the plan defaults: 16 kHz,
  25 ms frames (400 samples), 10 ms hop (160 samples), 70–400 Hz pitch range,
  0.30 pitch autocorrelation threshold, 0.20 s pause threshold, 2.0 s long-pause
  threshold. Overridable via keyword args.
- **`FeatureResult`** — frozen dataclass returning recording ID, participant ID,
  task, numeric features, quality flags, per-input SHA-256 hashes, and the
  config used. Nested `features` / `input_hashes` are wrapped in
  `MappingProxyType` and `quality_flags` as a `tuple` for true immutability.
- **`ManifestRow`** + `validate_manifest` — parses a `version == 1` manifest
  (one `(participant_id, task)` row; recording/audio/transcript IDs and paths;
  diagnosis `AD`/`HC`; age; sex; education years). Rejects unknown version,
  missing fields, unknown tasks (`UnknownTaskError`), invalid diagnosis, negative
  age/education, and duplicate `(participant_id, task)` pairs.
- **`Token` / `Utterance` + `validate_transcript`** — parses transcript JSON v1:
  ordered participant/examiner utterances with finite, ordered timestamps and
  `word`/`filler`/`fragment`/`noise` tokens.
- **`nfc`** — Unicode NFC normalisation wrapper.
- **`sha256_file`** + `manifest_hashes` — streaming SHA-256 provenance with a
  structured error on missing files; `(kind, path) → digest` for every unique
  asset path across all manifest rows.
- **Structured errors** — `FeatureExtractionError` base with
  `InvalidManifestError`, `InvalidTranscriptError`, `UnknownTaskError`
  subclasses.
- **`pyproject.toml`** — added `[tool.ruff]` (line length 100, ignore notebooks)
  and `[tool.pytest.ini_options]` (`pythonpath = ["src"]`, `testpaths`).
  No project dependencies changed.
- **`__init__.py`** — importable package re-exporting the public symbols.

Production imports are stdlib only (`hashlib`, `math`, `unicodedata`,
`dataclasses`, `pathlib`, `types`) — no librosa/openSMILE/spaCy/torch/ASR.

---

## 2. Agy Review

Reviewer: `agy:code-reviewer`. Full review of the exact diff (the four owned
files). Verdict: **REQUEST CHANGES** — no blockers; one major, several minors,
nits. Verbatim findings, abbreviated:

| # | Severity | Location | Finding |
|---|----------|----------|---------|
| R1 | Major | `schema.py` `manifest_hashes` | Docstring over-promises: hashes only the first row; raises bare `IndexError` on an empty `rows` list instead of a structured error; `first` variable dead. |
| R2 | Major | `schema.py` `validate_manifest` | Duplicate-key detection runs on the raw (pre-NFC) `participant_id`, so NFC-equivalent IDs (e.g. `"e\u0302"` vs `"\u00ea"`) bypass duplicate rejection while their stored forms collide. |
| R3 | Minor | `schema.py` | `education_years` is required but never type/range validated; `sex` unconstrained (unlike `diagnosis`). Inconsistent validation depth for a public contract. |
| R4 | Minor | `test_schema.py` | `test_rejects_unknown_task_spec_name` is misnamed and asserts the base `FeatureExtractionError` instead of concrete `UnknownTaskError`. |
| R5 | Nit | `test_schema.py` | `pytest.raises(Exception)` immutability assertions are too broad; prefer `FrozenInstanceError` / `TypeError`. |
| R6 | Nit | `schema.py` | `_finite` reinvents `math.isfinite`. |
| R7 | Nit | `schema.py` `ManifestRow` | `audio`/`transcript`/`task_spec` Path properties shadow `audio_id`/`transcript_id`. Cosmetic. |

Confirmed-wrong too: NFC-equivalent rows both parse (2 rows returned);
empty `rows` raises `IndexError`; string `education_years` passes validation.

---

## 3. Resolution

All findings addressed:

| Finding | Resolution |
|---------|-----------|
| R1 | Rewrote `manifest_hashes` to validate `version == 1`, reject empty/missing `rows` with `InvalidManifestError`, and hash **every unique asset path across all rows**, returning `(kind, path) → digest`. Docstring now matches behaviour; removed dead `first`. Covered by `test_manifest_hashes_all_rows` and `test_manifest_hashes_rejects_empty_rows`. |
| R2 | NFC-normalise `participant_id` **before** the duplicate check, then store the normalised value (`schema.py`): covered by `test_duplicate_detection_is_nfc_normalised`. |
| R3 | Added `isinstance(education_years, int) and >= 0` validation, mirroring `age`: `test_rejects_negative_education_years`. `sex` intentionally left unconstrained (plan does not restrict its values) — noted in `ManifestRow` docs. |
| R4 | Renamed to `test_rejects_unknown_task` and asserted `pytest.raises(schema.UnknownTaskError)`. |
| R5 | Tightened immutability assertions to `dataclasses.FrozenInstanceError` (attribute set) and `TypeError` (mapping mutation via `MappingProxyType`). |
| R6 | Replaced `_finite` with `math.isfinite`. |
| R7 | Accepted (cosmetic); no functional change. |

Non-trivial additions for resolved findings each include a runnable test.

### Final verification (after Resolution)

```
$ uv run pytest tests/speech_features/test_schema.py -q
28 passed in 0.03s

$ uv run ruff check src/speech_features tests/speech_features
All checks passed!

$ uv run ruff format --check src/speech_features tests/speech_features
All checks passed!

$ git diff --check
(clean)
```

No notebooks and no future-task files (`acoustic.py`, `linguistic.py`,
`tasks.py`, `pipeline.py`, `evaluation.py`, docs `task-02..06`) were touched.
Only Task 1 owned paths and this report were modified. Not committed by this
report; commit happens separately as `feat: define speech feature input contracts`.
