# Vietnamese Speech Feature Extraction

Math-first feature extraction for AD-versus-healthy-control speech research.
It consumes participant WAV recordings and reviewed, aligned Vietnamese
transcripts across picture description, immediate/delayed recall, and
phonemic/semantic fluency tasks.

> **Research-only.** This library is a **research-only** tool for cohort
> characterisation and hypothesis exploration. It is **not a diagnostic** and
> it is **not for clinical decision-making.** It does not give a diagnosis and
> it defines **no fixed score threshold**; a decision based on these features
> for any individual is not supported and is out of scope. Do not use the
> outputs as clinical evidence. We do not promise any fixed performance; do not
> treat any metric shown in this documentation as a guarantee of real-world
> accuracy.

---

## 1. Input contracts

### Audio (standard PCM WAV)

- `read_wav(path, *, sample_rate=16000)` reads standard PCM WAV with the stdlib
  `wave` module: mono or stereo integer PCM (8/16/24/32-bit), required rate
  `>= 0` frames, non-empty.
- Stereo is down-mixed to mono by the channel mean; samples are mapped to a
  finite float domain in `[-1, 1]` and re-sampled to `ExtractionConfig.sample_rate`
  (default 16 kHz) with `scipy.signal.resample_poly` (coprime up/down from the
  source/target GCD). The decoded and re-sampled signals are both checked for
  finiteness.
- Non-PCM compression, unsupported width/channel counts, broken headers, and
  malformed or truncated payloads raise `InvalidAudioError`
  (`code = "INVALID_AUDIO"`).

### Transcript (reviewed, aligned Vietnamese JSON v1)

- Transcript JSON must declare `version: 1` and an `utterances` list.
- Each utterance has a `speaker` (`participant` or `examiner`), finite
  `start_s`/`end_s` with `end_s >= start_s`, and a `tokens` list.
- Each token has a `kind` from `word`, `filler`, `fragment`, `noise` and an
  optional finite, ordered `start_s`/`end_s`. Text is normalised to Unicode NFC
  via `nfc`.
- **_A transcript, not whitespace, defines Vietnamese word boundaries_**: a
  token's `text` is the unit of measurement. Acoustic duration, word counts, and
  all lexical statistics are computed from these explicit tokens — never by
  splitting on spaces. Tokens with the `filler`/`fragment`/`noise` kinds are
  counted as disfluencies and are not treated as content words.
- `validate_transcript` returns a frozen, deeply immutable `Transcript` carrying
  `transcript_id` and `language` (default `"vi"`).

### Manifest (one `(participant_id, task)` row)

A manifest is JSON with `version: 1` and a non-empty `rows` list. Each row
carries: `participant_id`, `task`, `recording_id`, `audio_id`, `transcript_id`,
`audio_path`, `transcript_path`, `task_spec_path`, `diagnosis` (`AD`/`HC`),
`age`, `sex`, `education_years`. `validate_manifest` rejects unknown versions,
missing fields, unknown tasks (`UnknownTaskError`), invalid diagnoses, negative
age/education, and duplicate `(participant_id, task)` pairs.

The three input paths are isolated on an `ExtractionInputs` object, separate
from the clinical `diagnosis`/`age`/`sex`/`education_years` fields, so inputs
can be passed around without leaking protected attributes.
`validate_task_spec` trims each row's external task-spec JSON against the per-task
contract in the next section.

## 2. Task list and task-spec fields

The six supported tasks and the required external task-spec JSON fields per task
(`version: 1`):

| Task | Required task-spec fields |
|------|---------------------------|
| `picture_desc_1`, `picture_desc_2` | `concept_aliases`, `entity_groups`, `action_groups` |
| `immediate_recall`, `delayed_recall` | `idea_aliases` |
| `phonemic_fluency` | `initials`, `exclusions` |
| `semantic_fluency` | `item_aliases`, `subcategories` |

When present, the optional task-spec `task` field must name a known task.
Task specs are external and versioned; patient data and real task lexicons
remain outside Git.

## 3. Feature families (math-first formulas)

Context for all acoustic measures: recording duration `T` (seconds), frame
energy `E_m`, frame voice activity `v_m in {0,1}`, pause durations
`p_j >= pause_threshold_s`, participant speech time `S`, lexical (content) words
`N`, fillers `F`, and sets of voiced pitches `F = {f_i}`. Frames use a
Hamming window and `P_k = |RFFT(wx)_k|^2`.

### Timing (`time_*`) and acoustic (`ac_*`)

- `ac_frame_energy_mean/sd/iqr/span` — mean/SD/between-percentile (IQR = P75−P25;
  span = P95−P5) of per-frame energy `E_m`.
- Voice activity: a frame is voiced when `E_m > SILENCE_ENERGY_FLOOR (=1e-6)`
  **and** `E_m >= mean(E_m | E_m > floor)`.
- Pauses: maximal runs of non-voiced frames with `run >= pause_threshold_s`;
  `ac_pause_count`, `ac_pause_rate_per_min` (count / `T` in minutes),
  `ac_pause_mean_s`, `ac_pause_sd_s`, `ac_pause_max_s`, and `ac_long_pause_count`
  (runs `>= long_pause_threshold_s = 2.0 s`).
- Pitch (`ac_pitch_voiced_*`): per-frame F0 from the normalised
  cross-correlation (NCCF) at lags whose period lies in
  `[pitch_min_hz, pitch_max_hz] = [70, 400]` Hz. The greatest-NCCF lag is the
  period; a peak below `pitch_autocorr_threshold = 0.30` is explicitly unvoiced.
  `mean/sd/cv (=sd/mean)/median/iqr/span/delta`, plus `ac_voice_ratio`.
- HNR (`ac_hnr_*`): `HNR = 10*log10(r / (1 - r))` in dB per voiced frame, from
  the best NCCF `r` clamped to `<= 0.999`; reports `median` and `iqr`.
- Spectral — `P_k = |RFFT(wx)_k|^2`:
  - centroid `mu_1 = sum(P_k * f_k) / sum(P_k)`,
  - spread `sigma = sqrt(sum(P_k * (f_k - mu_1)^2) / sum(P_k))`,
  - flatness `exp(mean(log(P_k))) / mean(P_k)` — geometric/arithmetic power
    ratio, bounded in `[0, 1]` (≈1 white noise, ≈0 tonal).
- Invalid input, `duration < 1.0 s`, or no voiced/pitched frames return `NaN`
  statistics together with a flag — never zero.

### Lexical and disfluency (`lex_*`)

Over participant word forms normalised as `NFC + casefold` (preserving
`d`/`đ` and diacritics):

- `lex_ttr = V / N` (type-token ratio), where `N` = word tokens, `V` = unique
  word types.
- `lex_hapax_ratio = V1 / N`, `V1` = hapax (once-occurring) types.
- `lex_brunet_w = V^0.172`.
- `lex_honore_r = 100*log(N) / (1 - V1/V)`; `NaN` when the divisor vanishes
  (single-type transcript).
- `lex_repetition_ratio` — share of word tokens that immediately repeat the
  prior token (adjacent repeats only).
- Filler/fragment/noise ratios measured against **all** participant tokens.
- `lex_function_word_ratio` — share of participant word tokens present in an
  external NFC/casefolded function-word set (`NaN` when no set is supplied).

### Task-specific (`picture_*`, `recall_*`, `fluency_*`)

Alias matching is exact NFC/casefold equality first, then standard Levenshtein
edit distance with `normalised_similarity = 1 - d / max(len(a), len(b))` at the
documented conservative `SIMILARITY_THRESHOLD = 0.85` (near-exact variants only,
never unrelated words; empty pairs score 0).

- **Picture** — `picture_concept_coverage` (distinct matched concepts / total),
  `picture_concept_density` (distinct matched / words), `picture_repeat_ratio`,
  `picture_entity_coverage`, `picture_action_coverage` (from `entity_groups` /
  `action_groups`).
- **Recall** — same coverage/density/repeat family over `idea_aliases`, plus
  `recall_order_score` = normalised LCS length of the matched-ideas sequence
  (in utterance order) against the reference idea order (`NaN` when nothing is
  recalled).
- **Phonemic fluency** — one participant utterance containing word tokens is one
  response item. A response is valid when its first word starts with an initial
  and the whole response is not excluded. Outputs: `fluency_response_count`,
  `fluency_valid_count`, `fluency_valid_unique`, `fluency_repeats`,
  `fluency_intrusions`, `fluency_first_half_valid`, `fluency_second_half_valid`,
  `fluency_production_change` (second − first half), `fluency_rate` =
  `valid_unique / response_count`.
- **Semantic fluency** — items matched to `item_aliases`; `fluency_valid_unique`,
  `fluency_valid_count`, `fluency_repeats`, `fluency_rate` (same denominator),
  plus `fluency_clusters` (maximal runs of consecutive responses in the same
  `subcategory`) and `fluency_switches` (adjacent subcategory changes). Matched
  responses without a declared subcategory still count as valid but are excluded
  from the cluster/switch sequence.

## 4. Quality flags, provenance, and label separation

- **Quality flags** ride on `FeatureResult.quality_flags` and never replace a
  value with a fabricated zero: `invalid_input`, `too_short` (audio < 1.0 s),
  `no_voice`, and `no_participant_speech`. Unavailable statistics are `NaN`.
- **Provenance** — every extracted recording records streaming SHA-256 digests
  of its audio, transcript, and task-spec files (`sha256_file` /
  `manifest_hashes`; a missing input raises `MissingInputError`, code
  `MISSING_INPUT`). This makes each result traceable to its exact inputs.
- **Label separation** — diagnosis is read only at the batch layer into
  `BatchResult.labels` (`AD`/`HC` keyed by `(participant_id, task,
  recording_id)`). It never enters a `FeatureResult` feature or field, and the
  evaluation function receives labels as a required, **separate** argument keyed
  by participant.

## 5. Batch failure behaviour

`extract_manifest(manifest_path, *, config)` validates the whole manifest, then
runs every row through `extract_recording`. A failing row is isolated as a
`BatchFailure` (row key, stable `code`, error type, message) instead of aborting
the batch, so the remaining rows still complete. Rows that fail with a
`FeatureExtractionError` are recorded with their stable code (`MISSING_INPUT`,
`INVALID_AUDIO`, `INVALID_TRANSCRIPT`, `INVALID_TASK_SPEC`, `UNKNOWN_TASK`); an
unexpected exception inside a row is caught by a generic fallback and recorded
with `code = "EXTRACTION_ERROR"`. Missing speaker speech (`no_participant_speech`)
is a flag, not a failure.

## 6. Cohort minimum and task coverage

The AD-vs-HC baseline requires at least two participants of each class to form
any fold, and it warns (rather than crashing) when the cohort is too small. By
default each participant must contribute at least **5 of the 6 tasks**
(`min_task_coverage=5`); participants below coverage are listed in `warnings`.
Recordings are aggregated to **one case per participant** (NaN-aware mean), so a
participant with more recordings is never duplicated or overweighted, and the
cohort/task-coverage and feature-missingness summaries are reported on
`EvaluationResult`.

## 7. Repeated nested grouped-CV research baseline

`evaluate_ad_baseline` runs the leakage-safe research baseline over in-memory
feature rows with binary labels supplied separately:

- **One case per participant** and **grouped splits** — outer
  `StratifiedGroupKFold` (default 5 folds) and inner grouped CV (default 4) both
  split on participants, so a participant never straddles a train/test boundary.
- **Preprocessing inside the training fold only** — median imputation and
  scaling are fit on each training fold alone; test participants never
  contribute to these statistics.
- **`C` selection** — L2 logistic regression with `C` drawn from
  `(0.01, 0.1, 1, 10, 100)` chosen by balanced accuracy on the inner grouped CV;
  all candidates are scored on the same materialised folds; ties fall back to the
  smaller numeric `C`.
- **Reproducible seed procedure** — the whole run is seed-controlled from a
  single `seed` (default 42): each of the `outer_repeats = 10` repeated outer
  splits uses `random_state = seed + repeat`, the inner `C` selection uses
  `seed + repeat*1000 + fold_id`, and bootstraps use `random_state=seed`. Passing
  the same `seed` reproduces the same folds, selected `C`s, metrics, and bootstrap
  intervals.
- **Metrics and CIs** — per-fold AUROC, balanced accuracy, sensitivity (AD
  recall), specificity, plus the mean across folds and participant-bootstrap 95%
  percentile intervals (`bootstrap_ci`, default 2000 resamples) for each metric.
- **Outputs** — `EvaluationResult` carries `fold_metrics`, `selected_cs`,
  `participant_predictions`, `feature_columns`, `cohort_summary`,
  `task_coverage`, `feature_missingness`, `metrics_summary`, and `warnings`.
  It does **not** claim diagnostic/clinical performance and defines **no fixed
  score target**.
