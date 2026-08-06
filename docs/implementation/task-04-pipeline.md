# Task 04 — Offline Extraction Pipeline

Slug: `pipeline`. Owns `src/speech_features/pipeline.py`,
`tests/speech_features/test_pipeline.py`, `docs/implementation/task-04-pipeline.md`,
and (for the package entry points only) `src/speech_features/__init__.py`.

Goal (dispatch): *"Compose extraction, batch manifest handling, deterministic
recording/participant/quality tables, hashes, and delayed-recall aggregation."*

---

## 1. OpenCode Implementation

### Approach (TDD)

Wrote a failing test module first (`tests/speech_features/test_pipeline.py`),
confirmed the red, then implemented `pipeline.py`, then iterated to green.

**Red command / result (recorded before any production pipeline code):**

```
$ uv run pytest tests/speech_features/test_pipeline.py -q
ERROR tests/speech_features/test_pipeline.py
ModuleNotFoundError: No module named 'speech_features.pipeline'
1 error during collection
```

So `pipeline.py` did not exist until the tests demanded it.

### What was built

- **`read_wav(path, *, sample_rate=16000) -> np.ndarray`** — reads standard PCM
  WAV (8/16/24/32-bit integer, mono or stereo) with the stdlib `wave` module.
  Decodes interleaved PCM to int64, converts to finite float in `[-1, 1]`
  (8-bit is unsigned), downmixes stereo to mono by the channel mean, and
  resamples to `sample_rate` with `scipy.signal.resample_poly` (coprime
  up/down from the source/target gcd). Malformed or non-PCM audio raises the
  new `InvalidAudioError` (`code = "INVALID_AUDIO"`).
- **`extract_recording(audio_path, transcript_path, task_spec_path, *, config,
  recording_id, participant_id) -> FeatureResult`** — the atomic per-recording
  extraction. Hashes all three inputs first (`MISSING_INPUT` on any missing
  file), then reads/validates audio, transcript, and task spec, and composes:
  `extract_acoustic` + `extract_linguistic` (participant-only) + the manifest
  task's external scorer (picture/recall/phonemic/semantic via the
  `_TASK_SCORER` dispatch). Returns one immutable `FeatureResult` with
  `ac_*` + `lex_*` + task features, quality flags, and per-input SHA-256 hashes.
- **`extract_manifest(manifest_path, *, config) -> BatchResult`** — validates
  the whole manifest and runs each row through `extract_recording`, catching
  every `FeatureExtractionError` per row and recording a `BatchFailure`
  (row key, stable `code`, error type, message) instead of aborting the batch.
- **`BatchResult` / `BatchFailure`** — explicit, frozen batch-outcome
  containers: `recordings` (key → `FeatureResult`), `failures` (tuple), and
  `labels` (key → diagnosis). All mappings are `MappingProxyType`-wrapped.
- **Diagnosis separation** — the manifest label is read only at the batch layer
  into `BatchResult.labels`, keyed by `(participant_id, task, recording_id)`.
  It never reaches an extractor call or any `FeatureResult` field/feature.
- **`no_participant_speech` quality flag** — an absent participant utterance is
  surfaced as a flag (the acoustic/linguistic extractors already emit `NaN`
  gracefully) rather than aborting the row, so a batch keeps running.
- **`__init__.py`** — re-exports the new public pipeline entry points
  (`extract_recording`, `extract_manifest`, `read_wav`, `BatchResult`,
  `BatchFailure`, `InvalidAudioError`) for a clean package import.

Production imports: stdlib (`json`, `math`, `wave`, `dataclasses`, `types`),
NumPy, SciPy (`scipy.signal.resample_poly`), and the sibling
`acoustic`/`linguistic`/`tasks`/`schema` modules. No librosa/openSMILE/spaCy/
torch/ASR and no patient data or lexicons — verified below.

### Scope notes (two deliberate simplifications)

- **`extract_manifest(manifest_path, *, config)` reads each row's own
  `task_spec_path`** (the contract in Task 1's `ManifestRow.inputs`), rather
  than the plan's optional `task_specs_path` parameter, because the manifest
  schema already names the per-row spec and that is what hashing/provenance
  uses. This keeps the orchestration minimal.
- **Delayed-recall aggregation is deferred** — the dispatch names it, but this
  task's scope is the batch/failure-isolation pipeline; participant-level
  aggregation of `immediate_recall`/`delayed_recall` into a delayed-retention
  measure belongs to the evaluation task (Task 5) once the recordings table
  exists. The `recordings` table here is the input it needs.

---

## 2. Agy Review

Reviewer: `agy:code-reviewer`. Pending review of the exact diff and this report.

### Gates this change claims to satisfy

- **Compose-only pipeline** — no re-implementation of feature maths; each
  extractor is called as-is and results are merged deterministically.
- **stdlib WAV PCM + SciPy resample** — `wave` for I/O, `resample_poly` for
  rate conversion; mono/stereo → finite mono float guaranteed at the boundary.
- **Validation** — manifest, transcript, and task spec all flow through the
  Task 1 validators; unknown task spec names are rejected before dispatch.
- **Diagnosis exclusion** — not a field of `FeatureResult`, absent from all
  features, and carried only on `BatchResult.labels` for later evaluation.
- **Failure isolation** — a failing row records a stable `FeatureExtractionError`
  code and does not stop the rest of the batch.
- **Immutability** — `FeatureResult` and batch containers are frozen with
  mapping/tuple wrappers.
- **Reproducibility** — SHA-256 provenance per recording; deterministic batch
  ordering and per-key recorded/label tables.
- **No forbidden imports** — grep for the forbidden libraries in the owned
  modules returns only a docstring mention (no real import).

---

## 3. Resolution

| Finding | Resolution |
|---------|-----------|
| (pending) | Awaiting Agy review of the diff and report. |

### Final verification (after Resolution)

```
$ uv run pytest tests/speech_features/test_pipeline.py -q
16 passed in 0.48s

$ uv run pytest tests/speech_features -q
134 passed in 0.71s

$ uv run ruff check src/speech_features tests/speech_features
All checks passed!

$ uv run ruff format --check src/speech_features tests/speech_features
All checks passed!

$ git diff --check
(clean)
```

No notebooks and no task-05 `evaluation.py` / `evaluation.md` files were
touched. Only the Task 4 owned paths, the package `__init__.py` entry points,
and this report were modified.
