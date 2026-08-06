# Task 02 — Math-First Acoustic Features

Slug: `acoustic`. Owns `src/speech_features/acoustic.py`,
`tests/speech_features/test_acoustic.py`, and this report.

Scope (as instructed): a minimal deterministic NumPy/SciPy extractor for mono
PCM float arrays — sample-rate/config checks, frame/hop analysis, energy VAD,
VAD-derived silence/pause summaries, normalized-autocorrelation F0 in
configured bounds with explicit unvoiced handling, voiced F0 mean/SD/CV,
frame-energy mean/SD, spectral centroid/spread/flatness, and quality flags for
too-short, no-voice, and invalid input. Plain numeric features + flags only;
no labels, ASR, external audio feature packages, WAV I/O, or pipeline
orchestration. It builds on the approved Task 1 public contract,
`ExtractionConfig` from `schema.py`.

## 1. OpenCode Implementation

### Approach (TDD)

Wrote the behavioral test module first, confirmed the red, then implemented
`acoustic.py`, then iterated to green.

**Red command / result (recorded before any production code):**

```
$ uv run pytest tests/speech_features/test_acoustic.py -q
ERROR tests/speech_features/test_acoustic.py
ImportError ... ModuleNotFoundError: No module named 'speech_features.acoustic'
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
```

### What was built

- **`AcousticResult`** — frozen dataclass carrying an immutable `features`
  mapping (`ac_*` floats) and a `flags` tuple
  (`invalid_input` / `too_short` / `no_voice`).
- **`extract_acoustic(audio, sample_rate, *, config=None)`** — public entry
  point returning an `AcousticResult`. Honors `ExtractionConfig` bounds.
- **Frame / hop analysis** — `_frames` zero-pads the tail (Hamming-windowed)
  so every sample influences exactly one frame: 25 ms frames / 10 ms hop at
  16 kHz by default.
- **Energy VAD** — `_energy_vad` marks a frame voiced when its energy exceeds
  the mean frame energy (adaptive, gain-agnostic).
- **Silence / pause summaries** — `_pauses` returns durations of maximal
  non-speech runs `>= pause_threshold_s`; derives count, rate/min, mean/SD/max,
  and long-pause count (`>= long_pause_threshold_s`).
- **F0** — `_f0_per_frame` computes, per frame, the mean-subtracted
  length-normalized cross-correlation (NCCF) only for lags whose period lies in
  `[pitch_min_hz, pitch_max_hz]`; the greatest-NCCF lag is the period, and a
  peak below `pitch_autocorr_threshold` is explicitly unvoiced (`NaN`). A tone
  outside the bounds never yields a voiced candidate.
- **Voiced F0 mean/SD/CV** and **frame-energy mean/SD**.
- **Spectral** — `_spectral` mean centroid, spread (bandwidth), and flatness
  from `P_k = |RFFT(wx)_k|^2` of the Hamming-windowed frames.
- **Quality flags** — invalid input (bad sample rate / non-finite / non-1-D /
  empty) → `invalid_input`; duration < 1.0 s → `too_short`; no voiced frames or
  no pitched F0 → `no_voice`. Unavailable statistics are `NaN`, never zero.

Production imports: stdlib + `numpy` only (via `.schema` for
`ExtractionConfig`). No WAV I/O, no ASR, no librosa/openSMILE/spaCy/torch.

## 2. Agy Review

_Left ready for the reviewer (agy:code-reviewer) as defined in the plan /
dispatch gate workflow — to be completed before the next task._

## 3. Resolution

No reviewing findings at commit time (single build commit submitted for
review). Section left in the state required by the workflow for the review
phase; findings, once raised by Agy, will be recorded in resolution rows with
TDD (failing tests first) exactly as Task 1 documented.

## Changed files

- `src/speech_features/acoustic.py` (new)
- `tests/speech_features/test_acoustic.py` (new)
- `docs/implementation/task-02-acoustic.md` (this report, new)

No other files, notebooks, future-task paths, plan/dispatch JSON, or `.serena`
were touched.

## RED / GREEN results

RED (before production code):
```
$ uv run pytest tests/speech_features/test_acoustic.py -q
ERROR ... ModuleNotFoundError: No module named 'speech_features.acoustic'
```

GREEN (final):
```
$ uv run pytest tests/speech_features/test_acoustic.py -q
9 passed in 0.14s

$ uv run pytest tests/speech_features -q
56 passed in 0.15s

$ uv run ruff check src/speech_features tests/speech_features
All checks passed!

$ uv run ruff format --check src/speech_features tests/speech_features
5 files already formatted

$ git diff --check
(clean)
```

## Commit ID

`feat: add math-first acoustic features` — commit will be recorded here after
commit (see Resolution / next-step note).

## Skipped scope

Deliberately not implemented in this task: HNR, pitch median/IQR/5–95% span/delta,
energy IQR/span, spectral entropy/flux, WAV loading/resampling, participant
interval gating, and pipeline/`FeatureResult` assembly — these belong to later
tasks (Task 4 pipeline) or are outside this task's narrow scope and can be added
when the corresponding task requires them.
