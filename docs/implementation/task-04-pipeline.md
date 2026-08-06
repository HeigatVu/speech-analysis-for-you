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

**RED, review-fix regression tests (written before the production fixes):**

```
$ uv run pytest tests/speech_features/test_pipeline.py -q
3 failed, 21 passed in 0.58s

FAILED TestInvalidAudioIsolation::test_truncated_wav_raises_invalid_audio
FAILED TestInvalidAudioIsolation::test_truncated_wav_is_isolated_as_batch_failure
FAILED TestGenericRowFallback::test_arbitrary_row_exception_becomes_batch_failure
```

The three failures were the Agy-finding regressions: a truncated 24-bit WAV
leaked a raw `IndexError` instead of `InvalidAudioError` (finding 1, 6), and an
arbitrary row exception aborted the whole batch instead of being isolated
(finding 2). The 8/24-bit decode and post-resample finiteness tests passed even
on the build commit; they lock in the vectorised decode (finding 4) and the
post-resample finiteness guarantee (finding 3) as regressions.

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

Reviewer: `agy:code-reviewer`. Review of the Task 4 build commit `03d8bde`
raised one set of hardening findings on the WAV-decode and batch-isolation
paths. Each finding maps to a TDD fix below (RED regression test written
first, then production fix), recorded in Resolution.

1. **WAV decode/downmix/resample paths are not uniformly wrapped in
   `InvalidAudioError`.** Only I/O (`wave.open`) and header guard failures were
   raised as `InvalidAudioError`; a truncated payload that failed during
   PCM decode leaked raw `ValueError`/`IndexError` (e.g. stripping one byte
   from a 24-bit WAV → an uncaught `IndexError`). Every decode/downmix/
   resample step must yield a structured `InvalidAudioError` (`INVALID_AUDIO`).
2. **`extract_manifest` catches only `FeatureExtractionError`.** An arbitrary
   unexpected exception raised inside a row (a scorer bug, a `TypeError`, etc.)
   propagated out of `extract_recording` and terminated the whole batch instead
   of being recorded as a `BatchFailure`. The batch needs a generic per-row
   fallback so any row exception isolates without aborting.
3. **Finiteness is only asserted before resampling.** `read_wav` checks the
   decoded PCM is finite but does not re-assert finiteness on the output of
   `resample_poly`; the boundary guarantee should hold after resampling too.
4. **24-bit PCM decoding uses a Python sample loop.** `_bytes_to_samples` for
   `width == 3` iterated per-sample; it should assemble each 24-bit sample from
   its three little-endian bytes with NumPy vectorised ops.
5. **8-bit unsigned and 24-bit signed PCM decoding are untested.** The WAV
   contract advertises 8/16/24/32-bit support but only 16-bit had decoding
   tests.
6. **A malformed/truncated WAV is not tested as an isolated batch row** with a
   stable `INVALID_AUDIO` code.

Each of the above was converted into a regression test first (see Resolution),
confirmed RED against the build commit, then fixed.

---

## 3. Resolution

All six findings resolved test-first. The three behavioural regressions (1, 2,
6) genuinely failed on the build commit; the decode-coverage and vectorisation
findings (3, 4, 5) added missing guarantees/tests alongside the fix.

| # | Failing test (RED) | Fix |
|---|--------------------|-----|
| 1 | `TestInvalidAudioIsolation::test_truncated_wav_raises_invalid_audio` | `read_wav` wraps the whole decode/downmix/resample block in a `try` that converts `ValueError`/`IndexError`/`TypeError` into `InvalidAudioError`; a 24-bit payload not a multiple of three now raises `INVALID_AUDIO` at the boundary. |
| 2 | `TestGenericRowFallback::test_arbitrary_row_exception_becomes_batch_failure` | `extract_manifest` adds a generic `except Exception` fallback that records a `BatchFailure` (`code = "EXTRACTION_ERROR"`, `error_type` = exception class) instead of aborting the batch. |
| 6 | `TestInvalidAudioIsolation::test_truncated_wav_is_isolated_as_batch_failure` | Proves a truncated-WAV row in a manifest is isolated as `BatchFailure(key, INVALID_AUDIO)` while a sibling good row still extracts. |
| 3 | `TestInvalidAudioIsolation::test_read_wav_output_is_finite_after_resample` | `read_wav` re-asserts `np.all(np.isfinite(samples))` on the `resample_poly` output and raises `InvalidAudioError` if non-finite. |
| 4 | `TestPcmWidthDecoding::*` (24-bit round-trip + stereo downmix) | `_bytes_to_samples` for `width == 3` slices bytes into a `(N, 3)` array and assembles samples with vectorised `|`/`<<`/`np.where` — no per-sample loop. |
| 5 | `TestPcmWidthDecoding::test_8bit_unsigned_pcm_decodes_correctly`, `test_24bit_signed_pcm_decodes_correctly` | Added explicit 8-bit unsigned and 24-bit signed PCM decoding rounds, verifying the `[-1, 1]` mapping matches the writer. |

### Final verification (after Resolution)

```
$ uv run pytest tests/speech_features/test_pipeline.py -q
24 passed in 0.57s

$ uv run pytest tests/speech_features -q
142 passed in 0.80s

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
