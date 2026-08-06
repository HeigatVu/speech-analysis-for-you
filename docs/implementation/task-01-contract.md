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
  age/education, and duplicate `(participant_id, task)` pairs. Each row exposes a
  frozen `ExtractionInputs` (audio/transcript/task-spec paths) separate from its
  clinical fields.
- **`validate_task_spec`** — validates a versioned task-spec JSON (`version == 1`)
  and the per-task required fields the plan documents (picture, recall, phonemic,
  semantic). Raises `InvalidTaskSpecError`.
- **`Token` / `Utterance` / `Transcript` + `validate_transcript`** — parses
  transcript JSON v1 into a frozen, deeply immutable `Transcript` carrying
  `transcript_id` and `language`, with ordered participant/examiner utterances,
  finite, ordered timestamps, `word`/`filler`/`fragment`/`noise` tokens, and
  optional token `start_s`/`end_s` that are preserved and validated.
- **`nfc`** — Unicode NFC normalisation wrapper.
- **`sha256_file`** + `manifest_hashes` — streaming SHA-256 provenance with a
  `MissingInputError` on missing files; returns a **JSON-serializable**,
  deduplicated `{kind: {path: sha256}}` mapping across all manifest rows.
- **Structured errors** — every error carries a stable machine-readable
  `code`: `FEATURE_EXTRACTION_ERROR`, `UNKNOWN_TASK`, `INVALID_MANIFEST`,
  `INVALID_TRANSCRIPT`, `INVALID_TASK_SPEC`, and `MISSING_INPUT`.
- **`pyproject.toml`** — added `[tool.ruff]` (line length 100, ignore notebooks)
  and `[tool.pytest.ini_options]` (`pythonpath = ["src"]`, `testpaths`).
  No project dependencies changed.
- **`__init__.py`** — importable package re-exporting the public symbols.

Production imports are stdlib only (`hashlib`, `math`, `unicodedata`,
`dataclasses`, `pathlib`, `types`) — no librosa/openSMILE/spaCy/torch/ASR.

---

## 2. Agy Review

Reviewer: `agy:code-reviewer`. Full review of the exact diff and report.
Verdict: **REQUEST CHANGES** — several contract gaps found. Findings:

| # | Severity | Location | Finding |
|---|----------|----------|---------|
| C1 | Major | `schema.py` | No versioned **task-spec** validation. Task-spec JSON is external and versioned per the plan, but `schema.py` never validates a spec's version or its per-task fields. |
| C2 | Major | `Utterance.tokens` | Transcript tokens are stored in a mutable `list`; `Utterance` is `frozen` only nominally. Containers must be immutable for a trusted contract. |
| C3 | Major | `manifest_hashes` | Provenance is not JSON-serializable (tuple keys), not clearly deduplicated, and the missing-file path raises the generic `FeatureExtractionError` with no machine-readable signal. |
| C4 | Minor | errors | Errors expose a human message only. There is no stable **machine-readable error code**, and no dedicated code for a missing input file. |
| C5 | Minor | `Token` | Optional **token timestamps** (`start_s`/`end_s`) are dropped: they are neither preserved nor validated. |
| C6 | Major | transcript result | `validate_transcript` returns a bare `list[Utterance]`; the transcript's **`transcript_id` and `language`** are discarded. No frozen `Transcript` result. |
| C7 | Minor | `validate_manifest` | `ManifestRow` bundles **extraction inputs** (audio/transcript/task-spec paths) with **diagnosis**. No structural separation of extraction inputs from clinical/diagnosis data. |

## 3. Resolution

Every finding addressed with TDD (new failing tests first):

| Finding | Resolution |
|---------|-----------|
| C1 | Added `validate_task_spec(spec) -> int` enforcing a supported integer `version` and the required per-task fields documented in the plan (picture → concept/entity/action groups; recall → idea aliases; phonemic → initials/exclusions; semantic → item aliases/subcategories). Raises `InvalidTaskSpecError`. Tests: `TestTaskSpecValidation`. |
| C2 | `Utterance.tokens` is now a `tuple` (coerced in `__post_init__`); `Token`, `Utterance`, and the new `Transcript` are `@dataclass(frozen=True)`. Test: `test_transcript_is_immutable_and_deeply_immutable`. |
| C3 | `manifest_hashes` now returns a plain, **JSON-serializable** nested dict `{kind: {path: sha256}}`, deduplicating shared paths by string key. `sha256_file` raises `MissingInputError`. Tests: `test_manifest_hashes_is_json_serializable`, `test_manifest_hashes_raises_missing_input_code`, `test_manifest_hashes_all_rows`. |
| C4 | Added a stable `code` attribute on every error class and a new `MissingInputError(code="MISSING_INPUT")`. Tests: `TestStructuredErrorCodes`. |
| C5 | `Token.start_s`/`end_s` are preserved (optional, `None` when absent) and validated finite/ordered. Tests: `test_optional_token_timestamps_preserved`, `test_rejects_bad_token_timestamps`. |
| C6 | `validate_transcript` returns a frozen `Transcript` carrying `transcript_id`, `language`, and a deeply immutable `utterances` tuple. Tests: `test_valid_transcript_v1_parses`, `test_preserves_transcript_id_and_language`. |
| C7 | Added frozen `ExtractionInputs(audio_path, transcript_path, task_spec_path)`. `ManifestRow` now holds `inputs: ExtractionInputs` plus separate clinical fields; `audio_path`/`transcript_path`/`task_spec_path` remain accessible as properties. Test: `TestExtractionInputsSeparation`. |

### Red run (missing behaviour, before implementation)

```
$ uv run pytest tests/speech_features/test_schema.py -q
21 failed, 24 passed
(collection/assertion failures across TestTaskSpecValidation,
 TestStructuredErrorCodes, TestExtractionInputsSeparation, and the
 manifest/transcript immutability + JSON-serializability tests)
```

### Final verification (after Resolution)

```
$ uv run pytest tests/speech_features/test_schema.py -q
46 passed in 0.04s

$ uv run ruff check src/speech_features tests/speech_features
All checks passed!

$ uv run ruff format --check src/speech_features tests/speech_features
All checks passed!

$ git diff --check
(clean)
```

No notebooks and no future-task files (`acoustic.py`, `linguistic.py`,
`tasks.py`, `pipeline.py`, `evaluation.py`, docs `task-02..06`) were touched.
Only Task 1 owned paths and this report were modified. Commit message:
`fix: complete speech feature contracts`.
