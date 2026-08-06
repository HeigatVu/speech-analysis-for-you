# Vietnamese Alzheimer Speech Feature Library

## Goal

Build a research-only, math-first Vietnamese speech feature package for AD versus healthy-control studies. It accepts participant WAV recordings plus reviewed, aligned Vietnamese transcripts for picture description, immediate/delayed recall, and phonemic/semantic fluency tasks. It is not a diagnostic product.

## Constraints

- Production extraction uses NumPy/SciPy primitives only; no librosa, openSMILE, spaCy, ASR, embeddings, or transformer models.
- Existing notebooks stay untouched.
- A transcript, not whitespace, defines Vietnamese word boundaries.
- Real task lexicons and patient data stay external to Git. Each result records SHA-256 provenance for audio, transcript, and task specification.
- Each task is test-first and is reviewed by Agy before the next starts.

## Public Contract

`ExtractionConfig` fixes the defaults: 16 kHz, 25 ms frames, 10 ms hop, 70--400 Hz pitch range, 0.30 pitch autocorrelation threshold, 0.20 s pause threshold, and 2.0 s long-pause threshold.

`FeatureResult` returns recording ID, participant ID, task, numeric features, quality flags, and input/spec hashes.

`extract_recording(audio_path, transcript_path, task_spec, *, config)` returns one `FeatureResult`.

`extract_manifest(manifest_path, task_specs_path, *, config)` returns recording, participant, and quality/failure tables.

`evaluate_ad_baseline(participant_features, *, seed=42, outer_splits=5, outer_repeats=10, inner_splits=4, bootstrap_resamples=2000)` returns metrics, predictions, coefficients, and cohort summaries.

The manifest has one `(participant_id, task)` row with recording/audio/transcript IDs and paths, diagnosis (`AD` or `HC`), age, sex, and education. Transcript JSON v1 contains ordered participant/examiner utterances with finite timestamps and `word`, `filler`, `fragment`, or `noise` tokens. One fluency utterance is one response item.

Task-spec JSON is external and versioned. Picture specs provide concept aliases and entity/action groups; recall specs provide ordered idea aliases; phonemic specs provide initials/exclusions; semantic specs provide item aliases and one subcategory per item.

## Features

For recording duration `T`, participant speech time `S`, lexical words `N`, fillers `F`, syllables `Y`, and pauses `p_j >= 0.20 s`:

- `time_*`: `S/T`, pause count/minute, pause time ratio, pause mean/SD/max, long-pause count divided by `N + F`, mean speech-segment duration, words/minute, syllables/minute, and articulation rate `60Y/S`.
- `lex_*`: word/unique counts, syllables per word, TTR, MATTR-20, normalized entropy, hapax ratio, mean utterance lengths, and filler/fragment/immediate-repetition ratios.
- `ac_*`: autocorrelation pitch median/IQR/5--95% span/delta; energy SD/IQR/span; HNR median/IQR; spectral centroid, entropy, and flux means/SDs. Use Hamming-windowed frames and `P_k = |RFFT(wx)_k|^2`.
- `picture_*`: concept coverage, concept density/efficiency, repeat ratio, entity and action coverage.
- `recall_*`: idea coverage/density/efficiency, repeat ratio, order score, plus participant-level delayed retention.
- `fluency_*`: valid unique responses, rate, repeats, intrusions, half-time counts, production change, and inter-response intervals. Semantic fluency adds clusters and switches.

Invalid denominators or too few observations return `NaN` with a quality flag; they never become zero. Constant or non-finite audio, invalid timestamps, unknown tasks, and absent participant speech are errors.

## Evaluation

Require at least five of six tasks per participant. Use fold-local median imputation without missingness indicators and repeat on complete cases. Compare demographics-only, timing-only, acoustic-only, linguistic/task-only, combined speech, and combined-plus-demographics sensitivity panels. Use repeated nested participant-stratified cross-validation with L2 logistic regression and `C` in `[0.01, 0.1, 1, 10, 100]`. Report AUROC, balanced accuracy, sensitivity, specificity, F1, participant-bootstrap 95% intervals, and coefficient sign stability. No deployable model or clinical cutoff is produced.

## Verification

- Hand-checked synthetic tests cover pitch, energy, spectrum, HNR, silence, clipping, and resampling.
- Vietnamese tests cover Unicode NFC, diacritics, multi-syllable word tokens, aliases, recall order, and fluency responses.
- Integration tests cover schema errors, deterministic tables, provenance, missing tasks, batch continuation, and fold isolation.
- Final commands: `pytest tests/speech_features`, `ruff check src/speech_features tests/speech_features`, `git diff --check`, and a search proving production package modules do not import forbidden feature/NLP libraries.

## Agent Gate

For every task: OpenCode implements test-first, writes a Markdown report, Agy reviews the exact diff and report, OpenCode resolves findings, Agy approves, then the next task starts. User reviews the complete branch only after all tasks and the final whole-branch review.
