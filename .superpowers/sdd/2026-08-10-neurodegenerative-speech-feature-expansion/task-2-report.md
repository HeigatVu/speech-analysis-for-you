# Task 2 implementation report

## Changed files

- `src/speech_features/features/acoustic/definitions.py`: registered 83 new acoustic keys with units, domains, language scopes, tasks, disorders, and evidence levels.
- `src/speech_features/features/acoustic/timing.py`: added pause, aligned speech-segment, local-rate, event-rate, entropy, and acceleration companions.
- `src/speech_features/features/acoustic/phonation.py`: added voice-break, semitone F0, intensity, and NHR companions.
- `src/speech_features/features/acoustic/spectrum.py`: shared the existing target-interval/VAD frame selection with expanded spectral extraction.
- `src/speech_features/features/acoustic/advanced.py`: added energy, spectral-shape, low/high energy, MFCC, and distribution-statistic extraction.
- `src/speech_features/features/acoustic/__init__.py`: routed expanded spectral extraction through the existing array and bundle APIs.
- `tests/speech_features/test_acoustic_neuro.py`: added Task 2 behavioral, schema, metadata, formula, and missing-data tests.
- `tests/speech_features/test_acoustic.py`: updated the pre-existing exact catalog/bundle key expectations from 73 to 156 keys; no behavioral assertion was weakened.

No dependency, plugin/discovery layer, diagnosis API, documentation count, unrelated production module, or lockfile was changed.

## Strict pytest TDD record

### RED

First production-free run:

```text
rtk proxy uv run pytest tests/speech_features/test_acoustic_neuro.py -q
FFFFFFF                                                                  [100%]
7 failed in 0.83s
```

The failures were the expected missing behavior: absent timing, voice, spectral, and MFCC keys plus the absent `speech_features.features.acoustic.advanced` module.

During self-review, a scale-invariance regression test was added before its fix:

```text
rtk proxy uv run pytest tests/speech_features/test_acoustic_neuro.py -q
....F..                                                                  [100%]
1 failed, 6 passed in 0.83s
```

It showed that the first near-constant guard incorrectly treated a nonconstant `1e-16..4e-16` distribution as insufficient. The guard was replaced by an exact constant check while retaining SciPy `bias=False` statistics.

### GREEN and regression

Latest focused result before this report:

```text
rtk proxy uv run pytest tests/speech_features/test_acoustic_neuro.py -q
.......                                                                  [100%]
7 passed in 0.84s
```

Latest focused plus acoustic regression result before final formatting:

```text
rtk proxy uv run pytest tests/speech_features/test_acoustic_neuro.py tests/speech_features/test_acoustic.py -q
119 passed, 1 warning in 4.55s
```

The warning is the pre-existing `speech_features.pipeline` deprecation warning emitted by `tests/speech_features/test_acoustic.py`.

### Lint and format

```text
rtk proxy uv run ruff check src/speech_features/features/acoustic tests/speech_features/test_acoustic_neuro.py
All checks passed!
```

The first format check identified five changed files. Ruff formatted only those files.

Final verification gate:

```text
rtk proxy uv run pytest tests/speech_features/test_acoustic_neuro.py tests/speech_features/test_acoustic.py -q
119 passed, 1 warning in 4.54s

rtk proxy uv run ruff check src/speech_features/features/acoustic tests/speech_features/test_acoustic_neuro.py
All checks passed!

rtk proxy uv run ruff format --check src/speech_features/features/acoustic/advanced.py src/speech_features/features/acoustic/definitions.py src/speech_features/features/acoustic/__init__.py src/speech_features/features/acoustic/phonation.py src/speech_features/features/acoustic/spectrum.py src/speech_features/features/acoustic/timing.py tests/speech_features/test_acoustic_neuro.py tests/speech_features/test_acoustic.py
8 files already formatted
```

## Formulas and edge behavior

### Timing

- Pause observations combine threshold-qualified VAD pauses inside target regions with threshold-qualified gaps between merged aligned target intervals.
- Pause total is the sum; median and IQR use NumPy percentiles; CV is population SD divided by mean; proportion is total pause seconds divided by recording seconds.
- Between-utterance pause proportion is aligned-gap pause seconds divided by all pause seconds.
- Speech-segment summaries use merged aligned target intervals. Count and rate use recording minutes; median/IQR/CV/max use interval durations.
- Maximum local speech rate is the maximum word count per summed speech minutes over each window of three consecutive aligned target utterances.
- Timing events are maximal `voiced`, short-`unvoiced`, and threshold-qualified `pause` runs plus aligned inter-utterance gaps. Event entropy is Shannon entropy normalized by `log(3)`. Acceleration is second-half event rate minus first-half event rate, divided by recording minutes.
- With voiced speech but no qualifying pause, total/proportion/count/rate remain finite zero; distributional pause values are `NaN` with one `INSUFFICIENT_SPEECH_FRAMES` issue each. Missing alignment or word annotations use one `MISSING_ANNOTATION` issue per unavailable key.

### Voice/prosody

- A voice break is a pitch-unvoiced frame run at least `pause_threshold_s`, bounded on both sides by pitch-voiced frames within the same target interval.
- Break rate uses recording minutes; break proportion uses analyzed target-frame time.
- F0 is converted to semitones relative to median F0. Range is max minus min; MAD is median absolute deviation in semitones.
- Intensity range uses voiced-frame dB values; intensity CV uses linear RMS population SD divided by mean RMS.
- NHR is the sign-reversed mean of the existing finite HNR dB observations.
- No voiced frames produce one `INSUFFICIENT_VOICED_FRAMES` issue for each new unavailable voice key. Breaks never bridge target-interval boundaries.

### Spectrum and MFCC

- Expanded features reuse the existing target-interval framing, Hamming windows, and energy VAD.
- MFCC uses the frame power spectrum, 26 triangular Mel filters from zero to Nyquist, machine-epsilon power floor, natural log, orthonormal DCT-II, coefficient zero dropped, and coefficients 1--13 retained.
- Each coefficient has population mean/SD and SciPy `bias=False` skewness/excess-kurtosis. Mean/SD require one finite observation; skewness requires three; kurtosis requires four; exact constant distributions return `NaN` for higher moments.
- Spectral energy is frame power in dB. Spectral skewness and raw kurtosis are power-weighted standardized frequency moments summarized by population mean/SD. Low/high energy is pooled power below/above half Nyquist, converted with `10*log10(low/high)`.
- Missing target speech returns every expanded spectral key as `NaN` with exactly one `INSUFFICIENT_SPEECH_FRAMES` issue. The focused test also verifies the full returned-key set equals the registered acoustic catalog and every `NaN` in a silence extraction has exactly one issue.

## Evidence metadata

- Timing, voice-break, and NHR companions are `language_independent`.
- F0, intensity, spectral-shape, energy, low/high energy, and MFCC features are `language_sensitive`.
- Timing uses the `timing` domain; F0/intensity use `prosody`; voice break/NHR use `phonation`; energy/spectral/MFCC use `spectral`.
- The shared adult-neuro descriptors are tagged for the catalog's supported neurodegenerative disorders. MFCC/spectral summaries use `standard_feature_set`; derived timing/voice companions use `derived_companion`.

## Self-review

- Correctness: exact key counts, hand-derived pause/segment values, helper math, API return coverage, per-key issue cardinality, and acoustic regressions are tested.
- Architecture: no new public extraction path was introduced; array and bundle extraction use the existing orchestration. `advanced.py` owns the new spectral calculations and reuses the spectrum module's target-frame selector.
- Readability: formulas stay in the module that owns them; generated definitions avoid 83 repetitive declarations while preserving immutable `FeatureDefinition` validation.
- Security/data boundaries: extraction remains label-free, performs no network or external I/O, and adds no diagnosis field or inference.
- Performance: NumPy/SciPy vectorize the filter bank and frame statistics. Existing spectrum and expanded spectrum currently select frames in separate calls; sharing cached frames would require a wider API refactor and is intentionally out of Task 2 scope.

## Concerns

- These are math-first descriptors, not Praat/openSMILE-equivalent clinical measures or diagnosis outputs.
- Timing event boundaries inherit the existing frame/hop observation model; the last partial frame is zero-padded by the existing helper.
- Disorder tags express catalog applicability of shared descriptors, not evidence that every feature is discriminative for every disorder.
