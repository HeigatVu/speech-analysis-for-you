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

Findings raised by the reviewer (agy:code-reviewer) on the Task 2 build commit
`d64a14c`. Each finding maps to a TDD fix below (RED test written first, then
production fix), recorded in Resolution.

1. **Spectral flatness is mis-defined.** Computed as an `exp(mean(log))` ratio
   that exceeds 1 (reported `~1.2e12` for a tone, `~2.4e23` for noise). Must be
   the geometric/arithmetic power ratio, bounded in `[0, 1]` (≈0 for a tone, ≈1
   for noise), per the plan (`Tone → flatness == 0, Noise → flatness == 1`).
2. **Energy VAD has no absolute silence floor.** Only a mean-based dynamic
   threshold, so quiet noise can be classed voiced. Must combine the dynamic
   range with an absolute silence floor and be tested with quiet noise →
   `no_voice`.
3. **Pause runs are computed by accumulating `hop_s` per frame**, so a silence
   run whose duration lands exactly on the threshold can drift below it in
   float. Runs must be counted as integer frames and converted to seconds once.
4. **Pitch-lag bounds use `int` truncation, not `ceil`/`floor`.** For a
   non-integer `sr / pitch_max`, `int()` admits a lag whose period exceeds the
   configured upper bound. Must use `ceil(sr/pitch_max)` (min lag) and
   `floor(sr/pitch_min)` (max lag) and test the configured upper bound.
5. **Plan-required summaries are missing.** The plan (`ac_*`) demands pitch
   median/IQR/5–95% span/delta, energy SD/IQR/span, and HNR median/IQR; the
   build only emitted pitch mean/SD/CV and energy mean/SD.
6. **NaN truthiness bug.** `if features["ac_pitch_voiced_mean"]:` uses boolean
   truthiness of a value that can be `NaN` (truthy), risking a garbage CV;
   the guard must be an explicit finite check.

## 3. Resolution

All six findings resolved test-first (failing regression test recorded as RED,
then production fix, then GREEN).

| # | Failing test (RED) | Fix |
|---|--------------------|-----|
| 1 | `TestSpectralFlatness::test_tone_flatness_near_zero_and_bounded`, `test_white_noise_flatness_bounded_near_one` | `_spectral` now returns `exp(mean(log(P_k + eps))) / mean(P_k)` — geometric/arithmetic power ratio — bounded `[0, 1]`. |
| 2 | `TestEnergyVADSilenceFloor::test_quiet_noise_yields_no_voice`, `test_very_low_noise_yields_no_voice` | `_energy_vad` requires a frame to clear the absolute `SILENCE_ENERGY_FLOOR` **and** the dynamic threshold (mean of above-floor frames). Quiet noise → `no_voice`. |
| 3 | `TestPauseFrameCounting::test_exact_threshold_pause_is_counted` | `_pauses` counts maximal non-speech runs as integer frame counts and multiplies by `hop_s` once, so an exact-threshold (0.20 s) run is never lost to float drift. |
| 4 | `TestPitchLagBounds::test_pitch_lag_min_is_ceil_of_sr_over_max` (+ `test_configured_upper_bound_tone_voiced`) | `min_lag = ceil(sr/pitch_max), max_lag = floor(sr/pitch_min)`; a tone at the configured upper bound stays voiced. |
| 5 | `TestPlanRequiredSummaries::*` | Added `ac_frame_energy_iqr/span`, `ac_pitch_voiced_median/iqr/span/delta`, and `ac_hnr_median/iqr` (HNR = `10*log10(r/(1-r))` from per-voiced-frame best NCCF, r clamped to 0.999). |
| 6 | `TestNanTruthinessAvoided::test_cv_not_evaluated_by_boolean_truth_of_nan` | CV guard is an explicit `math.isfinite(mean) and mean > 0` check, replacing boolean truthiness. |

Verified no I/O, no labels, and no forbidden imports were introduced (imports
remain stdlib + `numpy` + `pytest`).

## Changed files

Task 2 owned files (build commit `d64a14c`):
- `src/speech_features/acoustic.py` (new)
- `tests/speech_features/test_acoustic.py` (new)
- `docs/implementation/task-02-acoustic.md` (this report, new)

Review-fix commit (this commit): updated
- `src/speech_features/acoustic.py` (flatness, VAD floor, pause frames, lag
  bounds, plan summaries, finite guard)
- `tests/speech_features/test_acoustic.py` (new RED regression tests)

No other files, notebooks, future-task paths, plan/dispatch JSON, or `.serena`
were touched.

## RED / GREEN results

RED (build, before any production code — task 2 first commit):
```
$ uv run pytest tests/speech_features/test_acoustic.py -q
ERROR ... ModuleNotFoundError: No module named 'speech_features.acoustic'
```

RED (review fixes — regression tests written, before the corrected maths):
```
$ uv run pytest tests/speech_features/test_acoustic.py -q
8 failed, 14 passed in 0.38s
```
The 8 failures were the Agy-finding regressions: tone/noise spectral flatness
out of `[0,1]`, quiet-noise not reading as `no_voice`, missing pitch/energy/HNR
summaries, and `int()`-vs-`ceil()` lag bounds.

GREEN (build, task 2 first commit):
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

GREEN (review fixes, final — this commit):
```
$ uv run pytest tests/speech_features/test_acoustic.py -q
22 passed in 0.31s

$ uv run pytest tests/speech_features -q
69 passed in 0.32s

$ uv run ruff check src/speech_features tests/speech_features
All checks passed!

$ uv run ruff format --check src/speech_features tests/speech_features
5 files already formatted

$ git diff --check
(clean)
```

## Commit ID

- Build commit `d64a14c` — `feat: add math-first acoustic features`.
- Review-fix commit — `fix: correct acoustic feature math` (this commit).

## Skipped scope

Deliberately not implemented: spectral entropy/flux, WAV loading/resampling,
participant interval gating, and pipeline/`FeatureResult` assembly — these
belong to later tasks (Task 3/4 pipeline) or are outside this task's scope and
can be added when the corresponding task requires them. Energy SD/IQR/span,
pitch median/IQR/5–95% span/delta, and HNR median/IQR were originally deferred
here but are now implemented as part of the Agy review resolution (finding 5).
