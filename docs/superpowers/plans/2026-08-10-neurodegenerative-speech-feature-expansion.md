# SAY Neurodegenerative Speech Feature Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand SAY into an evidence-traceable library of interpretable acoustic, motor-speech, cognitive-linguistic, structured-task, and standardized eGeMAPSv02 features for research on major neurodegenerative disorders.

**Architecture:** Preserve the population-neutral `SpeechDocument`, `FeatureBundle`, static registry, and existing keys. Add validated metadata and three capability paths: expanded shared acoustic extraction, annotation/task-driven `motor_neuro` and `adult_neuro` extraction, and a lazy optional `standardized_acoustic` adapter. Missing inputs produce `NaN` plus stable issues; diagnosis and clinical labels never enter extraction.

**Tech Stack:** Python 3.10--3.13, NumPy, SciPy, pandas, stdlib WAV/JSON/CSV, pytest, Ruff; optional `opensmile>=2.5,<3` only for eGeMAPSv02.

## Global Constraints

- Preserve distribution `speech-analysis-for-you`, namespace `speech_features`, CLI `say-features`, package version `0.2.0`, and every existing feature key/API behavior.
- Keep NumPy, SciPy, and pandas as the only core dependencies; openSMILE is a lazy optional extra.
- Accept reviewed PCM WAV, `SpeechDocument`, annotation layers, and versioned task specifications; never infer ASR, alignment, morphosyntax, diagnosis, age, or clinical labels.
- Normalize Vietnamese comparison text with NFC plus casefold while preserving diacritics and `d`/`đ`.
- Use stable `NaN` plus a structured issue for unavailable/undefined values; never guess zero or return infinity.
- Add no patient data, learned embeddings, open-ended plugin system, ComParE surface, librosa, spaCy, torch, or transformers.
- Use `rtk` before every shell command, pytest TDD for each slice, atomic commits, and no unrelated refactoring.
- All paths and interfaces below are relative to the repository root.

---

### Task 1: Catalog metadata and evidence inventory contract

**Files:**
- Modify: `src/speech_features/catalog.py`
- Create: `src/speech_features/evidence.py`
- Create: `tests/speech_features/test_catalog_metadata.py`
- Create: `docs/research/neurodegenerative-speech-feature-sources.md`

**Interfaces:**
- Consumes: existing `FeatureDefinition`, `PACKS`, `register_feature`, and `list_features`.
- Produces: `DOMAINS`, `LANGUAGE_SCOPES`, `TASK_IDS`, `DISORDERS`, `EVIDENCE_LEVELS`; additive immutable metadata fields; filters `domain`, `language_scope`, `task`, `disorder`, `evidence_level`; registered packs `motor_neuro` and `standardized_acoustic`; evidence records and inventory validation used by Tasks 2--7.

- [ ] **Step 1: Write failing catalog contract tests**

```python
def test_feature_metadata_is_normalized_immutable_and_filterable():
    definition = FeatureDefinition(
        key="resp_breath_group_count",
        pack="motor_neuro",
        level="recording",
        unit="count",
        population="adult",
        reference="https://pmc.ncbi.nlm.nih.gov/articles/PMC9950294/",
        domain="respiration",
        language_scope="language_independent",
        tasks=["connected_speech"],
        disorders=["als", "pd"],
        evidence_level="systematic_review",
    )
    assert definition.tasks == ("connected_speech",)
    assert definition.disorders == ("als", "pd")
    assert definition in list_features(domain="respiration", disorder="als")


@pytest.mark.parametrize(
    ("field", "value"),
    [("domain", "unknown"), ("language_scope", "neutral"),
     ("tasks", ("interview",)), ("disorders", ("dementia",)),
     ("evidence_level", "strong")],
)
def test_feature_metadata_rejects_unknown_values(field, value):
    kwargs = valid_definition_kwargs()
    kwargs[field] = value
    with pytest.raises(ValueError):
        FeatureDefinition(**kwargs)
```

- [ ] **Step 2: Run the focused tests and record the expected RED result**

Run: `rtk proxy uv run pytest tests/speech_features/test_catalog_metadata.py -q`

Expected: FAIL because the new fields, packs, constants, and filters do not exist.

- [ ] **Step 3: Add the minimal additive catalog contract**

```python
DOMAINS = frozenset({"audio_quality", "timing", "respiration", "phonation",
                     "prosody", "spectral", "articulation", "rhythm", "lexical",
                     "psycholinguistic", "morphosyntactic", "disfluency", "semantic",
                     "discourse", "task"})
LANGUAGE_SCOPES = frozenset({"language_independent", "language_sensitive",
                             "language_dependent", "language_specific"})
TASK_IDS = frozenset({"connected_speech", "picture_description", "story_recall",
                      "semantic_fluency", "phonemic_fluency", "reading",
                      "sustained_vowel", "ddk"})
DISORDERS = frozenset({"ad", "mci", "ppa", "ftd", "dlb", "pd", "pdd", "als",
                       "mnd", "hd", "ms", "ataxia", "psp", "msa", "cbs"})
EVIDENCE_LEVELS = frozenset({"systematic_review", "multi_study", "single_study",
                             "standard_feature_set", "derived_companion"})

@dataclass(frozen=True)
class FeatureDefinition:
    # existing fields remain in their existing order
    domain: str = "audio_quality"
    language_scope: str = "language_independent"
    tasks: tuple[str, ...] = ()
    disorders: tuple[str, ...] = ()
    evidence_level: str = "derived_companion"
```

Normalize tuple fields and validate them in `__post_init__`. Extend the static
`PACKS` mapping and `KEY_PREFIXES`; do not introduce discovery. Extend
`list_features` with single-value optional filters and reject unknown values.

- [ ] **Step 4: Add the evidence record contract and authoritative source list**

```python
@dataclass(frozen=True)
class EvidenceRecord:
    candidate: str
    domain: str
    language_scope: str
    tasks: tuple[str, ...]
    disorders: tuple[str, ...]
    source: str
    evidence_level: str
    status: str
    feature_keys: tuple[str, ...] = ()
    reason: str = ""

def validate_evidence(records, catalog_keys) -> tuple[EvidenceRecord, ...]:
    allowed_statuses = {"existing", "implemented", "optional", "deferred"}
    checked = []
    for record in records:
        if record.status not in allowed_statuses:
            raise ValueError(f"unknown evidence status: {record.status!r}")
        if not record.source:
            raise ValueError(f"missing evidence source: {record.candidate}")
        if record.status in {"existing", "implemented", "optional"}:
            if not record.feature_keys or not set(record.feature_keys) <= set(catalog_keys):
                raise ValueError(f"invalid catalog keys: {record.candidate}")
        if record.status == "deferred" and not record.reason:
            raise ValueError(f"deferred feature lacks reason: {record.candidate}")
        checked.append(record)
    return tuple(checked)
```

Allow statuses `existing`, `implemented`, `optional`, and `deferred`. Require a
source for every row, catalog keys for existing/implemented/optional rows, and a non-empty reason
for deferred rows. Populate `docs/research/neurodegenerative-speech-feature-sources.md`
with the exact sources in the design plus title, disorders, domains, and URL/DOI.

- [ ] **Step 5: Make tests GREEN and run catalog regression checks**

Run: `rtk proxy uv run pytest tests/speech_features/test_catalog.py tests/speech_features/test_catalog_metadata.py -q`

Expected: PASS with existing catalog ordering and error behavior unchanged.

- [ ] **Step 6: Run Ruff and commit**

Run: `rtk proxy uv run ruff check src/speech_features/catalog.py src/speech_features/evidence.py tests/speech_features/test_catalog_metadata.py`

Commit: `feat: add evidence-aware feature catalog metadata`

---

### Task 2: Expanded shared timing, prosody, spectrum, and MFCC features

**Files:**
- Modify: `src/speech_features/features/acoustic/timing.py`
- Modify: `src/speech_features/features/acoustic/phonation.py`
- Modify: `src/speech_features/features/acoustic/spectrum.py`
- Modify: `src/speech_features/features/acoustic/definitions.py`
- Modify: `src/speech_features/features/acoustic/__init__.py`
- Create: `src/speech_features/features/acoustic/advanced.py`
- Create: `tests/speech_features/test_acoustic_neuro.py`

**Interfaces:**
- Consumes: Task 1 metadata contract, existing target-speaker intervals,
  `ExtractionConfig`, frame helpers, and issue behavior.
- Produces: the keys below through the existing `extract_acoustic_features` and
  `extract_acoustic_bundle` APIs; `advanced_features(audio, sample_rate,
  intervals, config, recording_id, speaker_id, issues) -> dict[str, float]`.

- [ ] **Step 1: Write failing hand-calculated summary and schema tests**

```python
def test_pause_companion_summaries_are_hand_calculated(aligned_document):
    values, _ = extract_acoustic_features(
        np.ones(80000) * 0.1, 16000, document=aligned_document, target_speaker="PAR"
    )
    assert values["time_pause_total_s"] == pytest.approx(1.0)
    assert values["time_pause_median_s"] == pytest.approx(0.5)
    assert values["time_pause_iqr_s"] == pytest.approx(0.2)
    assert values["time_pause_proportion"] == pytest.approx(0.2)


def test_mfcc_statistics_have_fixed_52_key_schema():
    keys = {d.key for d in list_features(pack="acoustic")}
    assert {f"spectral_mfcc_{i}_{stat}" for i in range(1, 14)
            for stat in ("mean", "sd", "skewness", "kurtosis")} <= keys
```

- [ ] **Step 2: Run the focused tests and record RED**

Run: `rtk proxy uv run pytest tests/speech_features/test_acoustic_neuro.py -q`

Expected: FAIL on absent feature definitions and values.

- [ ] **Step 3: Implement timing/prosody companion features**

Add these exact keys with finite-or-`NaN` behavior:

```text
time_pause_total_s                  time_pause_median_s
time_pause_iqr_s                    time_pause_cv
time_pause_proportion               time_speech_segment_count
time_speech_segment_rate_per_min    time_speech_segment_median_s
time_speech_segment_iqr_s           time_speech_segment_cv
time_speech_segment_max_s           time_between_utterance_pause_proportion
time_max_local_speech_rate_wpm      time_timing_event_rate_per_min
time_timing_event_entropy           time_timing_acceleration_per_min2
voice_break_count                   voice_break_rate_per_min
voice_break_proportion              voice_f0_range_semitones
voice_f0_mad_semitones              voice_intensity_range_db
voice_intensity_cv                  voice_nhr_mean_db
```

Use the existing pause threshold and aligned speech intervals. Define a voice
break as an unvoiced run at least `pause_threshold_s` bounded by voiced frames.
Maximum local speech rate is the maximum WPM over any three consecutive aligned
utterances. Timing entropy is normalized Shannon entropy of event classes
`voiced`, `unvoiced`, and `pause`; acceleration is second-half event rate minus
first-half event rate divided by recording minutes.

- [ ] **Step 4: Implement standard MFCC 1--13 statistics without a new dependency**

```python
def mfcc_frames(frames: np.ndarray, sample_rate: int, n_mfcc: int = 13) -> np.ndarray:
    """Power spectrum -> 26 triangular Mel filters -> log -> orthonormal DCT-II."""

def distribution_stats(values: np.ndarray) -> tuple[float, float, float, float]:
    """Population mean/SD and scipy biased=False skew/kurtosis; NaN if insufficient."""
```

Use 26 Mel filters from 0 to Nyquist, a power floor of machine epsilon, DCT-II
with `norm="ortho"`, and coefficients 1--13 after dropping coefficient 0. Add
52 stable keys `spectral_mfcc_{1..13}_{mean|sd|skewness|kurtosis}` plus:

```text
spectral_energy_mean_db             spectral_energy_sd_db
spectral_skewness_mean              spectral_skewness_sd
spectral_kurtosis_mean              spectral_kurtosis_sd
spectral_low_high_energy_ratio_db
```

- [ ] **Step 5: Register evidence metadata and missing-data issues**

All timing/signal summaries use `language_independent`; F0, intensity, spectral,
and MFCC features use `language_sensitive`. Tag supported disorders from the
design sources. Every registered key must be returned and every `NaN` must have
one stable issue.

- [ ] **Step 6: Run focused and acoustic regression tests**

Run: `rtk proxy uv run pytest tests/speech_features/test_acoustic_neuro.py tests/speech_features/test_acoustic.py -q`

Expected: PASS.

- [ ] **Step 7: Run Ruff and commit**

Run: `rtk proxy uv run ruff check src/speech_features/features/acoustic tests/speech_features/test_acoustic_neuro.py`

Commit: `feat: expand shared neuro acoustic features`

---

### Task 3: Nonlinear phonation and perturbation measures

**Files:**
- Modify: `src/speech_features/schema.py`
- Modify: `src/speech_features/features/acoustic/advanced.py`
- Modify: `src/speech_features/features/acoustic/phonation.py`
- Modify: `src/speech_features/features/acoustic/definitions.py`
- Modify: `tests/speech_features/test_acoustic_neuro.py`

**Interfaces:**
- Consumes: Task 2 pitch-period and voiced-frame data.
- Produces: deterministic perturbation/nonlinear feature helpers and calibration
  fields on `ExtractionConfig` with conservative fixed defaults.

- [ ] **Step 1: Write failing synthetic-period tests**

```python
def test_period_perturbation_formulas_match_hand_calculation():
    periods = np.array([0.0100, 0.0101, 0.0099, 0.0102, 0.0098])
    rap_residuals = [abs(periods[i] - periods[i - 1:i + 2].mean()) for i in range(1, 4)]
    ppq5_residual = abs(periods[2] - periods.mean())
    assert jitter_rap(periods) == pytest.approx(np.mean(rap_residuals) / periods.mean())
    assert jitter_ppq5(periods) == pytest.approx(ppq5_residual / periods.mean())


def test_constant_period_signal_has_zero_pitch_period_entropy():
    assert pitch_period_entropy(np.ones(100)) == pytest.approx(0.0)
```

- [ ] **Step 2: Run RED**

Run: `rtk proxy uv run pytest tests/speech_features/test_acoustic_neuro.py -k "perturbation or entropy or nonlinear" -q`

Expected: FAIL because helpers and keys are absent.

- [ ] **Step 3: Implement exact perturbation keys**

```text
voice_jitter_rap                 voice_jitter_ppq5
voice_jitter_ddp                 voice_shimmer_apq3
voice_shimmer_apq5               voice_shimmer_apq11
voice_shimmer_dda                voice_pitch_period_entropy
voice_rpde                       voice_dfa
voice_correlation_dimension
```

RAP/PPQ/APQ compare each center sample with the local 3/5/11-sample moving
average and divide the mean absolute residual by the global mean. DDP is
`3 * RAP`; DDA is `3 * APQ3`. Pitch-period entropy is normalized Shannon
entropy over 32 fixed bins spanning the observed detrended log-period range.
RPDE uses recurrence lags 1--`min(100, n//2)` at radius `0.1 * SD` and returns
the normalized entropy of the recurrence-period histogram. DFA uses non-
overlapping window sizes `[4, 8, 16, 32, 64]` that fit and returns the
least-squares log-log slope. Correlation dimension uses a two-dimensional
delay embedding with delay one and the slope across correlation sums at eight
log-spaced radii between the 10th and 60th distance percentiles. Fewer than the
documented minimum observations returns `NaN` plus `INSUFFICIENT_VOICING`.

- [ ] **Step 4: Add physical calibration knobs**

Add `nonlinear_min_periods: int = 64`, `recurrence_radius_sd: float = 0.1`, and
`entropy_bins: int = 32` to the frozen config. Reject non-positive values at
the existing configuration boundary; do not add another config type.

- [ ] **Step 5: Run regression, Ruff, and commit**

Run: `rtk proxy uv run pytest tests/speech_features/test_schema.py tests/speech_features/test_acoustic_neuro.py tests/speech_features/test_acoustic.py -q`

Run: `rtk proxy uv run ruff check src/speech_features/schema.py src/speech_features/features/acoustic tests/speech_features/test_acoustic_neuro.py`

Commit: `feat: add interpretable nonlinear voice measures`

---

### Task 4: Annotation-driven motor-neuro feature pack

**Files:**
- Create: `src/speech_features/features/motor/__init__.py`
- Create: `src/speech_features/features/motor/definitions.py`
- Create: `src/speech_features/features/motor/intervals.py`
- Create: `tests/speech_features/test_motor_neuro.py`

**Interfaces:**
- Consumes: Task 1 catalog, `SpeechDocument`, token timing, and annotation
  layers `segment_type`, `segment_label`, `f1_hz`, `f2_hz`, `f3_hz`,
  `vot_s`, `stop_gap_s`, `spectral_moment_1`--`spectral_moment_4`,
  `resonance_attenuation_db`, and `breath_group`.
- Produces: `extract_motor_features(document, *, target_speaker=None,
  recording_id="", task_spec=None) -> tuple[dict[str, float], tuple[FeatureIssue, ...]]`
  and all registered `motor_neuro` keys.

- [ ] **Step 1: Write failing Vietnamese annotation-fixture tests**

```python
def test_vowel_space_and_centralization_from_reviewed_annotations(vowel_document):
    values, issues = extract_motor_features(vowel_document, target_speaker="PAR")
    # Fixture medians: /i/=(F1 300, F2 2400), /a/=(800, 1200), /u/=(350, 700).
    assert values["artic_vowel_space_area_hz2"] == pytest.approx(395_000.0)
    assert values["artic_vowel_articulation_index"] == pytest.approx(3200 / 2550)
    assert not issues


def test_missing_phone_annotations_are_nan_with_one_issue(document):
    values, issues = extract_motor_features(document, target_speaker="PAR")
    assert math.isnan(values["artic_vot_mean_s"])
    assert [(i.feature, i.code) for i in issues].count(
        ("artic_vot_mean_s", "MISSING_ANNOTATION")
    ) == 1
```

- [ ] **Step 2: Run RED**

Run: `rtk proxy uv run pytest tests/speech_features/test_motor_neuro.py -q`

Expected: import failure because the motor package does not exist.

- [ ] **Step 3: Implement safe annotation access and duration helpers**

```python
def layer_values(document, name: str, token_ids: set[str]) -> dict[str, str | float] | None:
    layer = next((item for item in document.annotations if item.layer == name), None)
    if layer is None or any(token_id not in layer.values for token_id in token_ids):
        return None
    return {token_id: layer.values[token_id] for token_id in token_ids}


def target_tokens(document, speaker_id: str) -> tuple[DocumentToken, ...]:
    return tuple(
        token
        for utterance in document.utterances
        if utterance.speaker_id == speaker_id
        for token in utterance.tokens
    )


def finite_mean_sd(values) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return math.nan, math.nan
    return float(array.mean()), float(array.std(ddof=0))


def polygon_area(points: tuple[tuple[float, float], ...]) -> float:
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2)
                   in zip(points, points[1:] + points[:1]))) / 2.0
```

Reject non-finite numeric annotations with `INVALID_TASK_ANNOTATION`; absent or
incomplete layers use `MISSING_ANNOTATION`. Group vowel tokens by normalized
`segment_label` values `i`, `a`, and `u` for the triangular formulas.

- [ ] **Step 4: Implement and register articulation/rhythm keys**

```text
artic_vowel_space_area_hz2          artic_vowel_articulation_index
artic_formant_centralization_ratio  artic_vowel_dispersion_mean_hz
artic_vowel_dispersion_sd_hz        artic_f1_within_vowel_sd_hz
artic_f2_within_vowel_sd_hz         artic_formant_transition_slope_mean_hz_s
artic_vot_mean_s                     artic_vot_sd_s
artic_stop_gap_mean_s                artic_stop_gap_sd_s
artic_consonant_duration_mean_s      artic_consonant_duration_sd_s
artic_fricative_m1_mean              artic_fricative_m2_mean
artic_fricative_m3_mean              artic_fricative_m4_mean
artic_resonance_attenuation_mean_db
rhythm_percent_vocalic               rhythm_varco_v
rhythm_varco_c                        rhythm_rpvi_v
rhythm_npvi_v                         rhythm_rpvi_c
rhythm_npvi_c
```

VAI is `(F2_i + F1_a) / (F1_i + F1_u + F2_u + F2_a)` and FCR is its reciprocal
denominator/numerator. `%V = 100 * vowel_duration / (vowel + consonant)`;
Varco is `100 * SD / mean`; rPVI is mean absolute adjacent difference; nPVI is
`100 * mean(abs(a-b) / ((a+b)/2))`.

- [ ] **Step 5: Implement DDK, respiratory, and sustained-vowel keys**

```text
task_ddk_rate_syllables_s            task_ddk_inter_onset_mean_s
task_ddk_inter_onset_median_s        task_ddk_inter_onset_sd_s
task_ddk_inter_onset_cv              task_ddk_instability_s
task_ddk_acceleration_syllables_s2   task_ddk_decay_ratio
task_ddk_voiced_interval_mean_s      task_ddk_sequential_alternating_ratio
task_max_phonation_time_s            resp_breath_group_count
resp_breath_group_mean_s             resp_breath_group_sd_s
resp_rate_per_min                     resp_pauses_per_breath
resp_relative_loudness_db             voice_gaping_interval_rate_per_min
voice_subharmonic_interval_proportion voice_sustained_f0_sd_semitones
voice_sustained_power_sd_db
```

Use token start/end for durations and ordered DDK onsets. Relative loudness is
available only when `task_spec["calibrated_amplitude"] is True` and numeric
`respiration_db`/`speech_db` annotations exist; otherwise emit
`UNCALIBRATED_AUDIO`. Sustained vowel and DDK features require the matching
`task_spec["task"]` and never fall back to connected speech.

- [ ] **Step 6: Run focused tests, Ruff, and commit**

Run: `rtk proxy uv run pytest tests/speech_features/test_motor_neuro.py tests/speech_features/test_catalog_metadata.py -q`

Run: `rtk proxy uv run ruff check src/speech_features/features/motor tests/speech_features/test_motor_neuro.py`

Commit: `feat: add annotation-driven motor speech pack`

---

### Task 5: Cognitive-linguistic, discourse, and structured-task features

**Files:**
- Modify: `src/speech_features/features/linguistic/definitions.py`
- Modify: `src/speech_features/features/linguistic/__init__.py`
- Create: `src/speech_features/features/linguistic/clinical.py`
- Create: `src/speech_features/features/linguistic/task_scores.py`
- Create: `tests/speech_features/test_neuro_linguistic.py`

**Interfaces:**
- Consumes: Task 1 metadata, existing lexical/morphosyntax extraction, reviewed
  annotation layers, and validated version-1 task specs.
- Produces: `extract_clinical_linguistic_features(document, *,
  target_speaker=None, recording_id="", task_spec=None)` and
  `extract_structured_task_features(document, task_spec, *, target_speaker=None,
  recording_id="")`; both return values plus structured issues.

- [ ] **Step 1: Write failing annotation and legacy-formula parity tests**

```python
def test_psycholinguistic_means_use_only_target_word_tokens(document_with_norms):
    values, _ = extract_clinical_linguistic_features(
        document_with_norms, target_speaker="PAR"
    )
    assert values["lex_frequency_mean"] == pytest.approx(3.5)
    assert values["lex_concreteness_mean"] == pytest.approx(4.0)


def test_picture_task_matches_existing_validated_formula(picture_document, picture_spec):
    values, _ = extract_structured_task_features(picture_document, picture_spec)
    assert values["task_picture_concept_coverage"] == pytest.approx(2 / 3)
    assert values["task_picture_repeat_ratio"] == pytest.approx(1 / 3)
```

- [ ] **Step 2: Run RED**

Run: `rtk proxy uv run pytest tests/speech_features/test_neuro_linguistic.py -q`

Expected: import/key failure.

- [ ] **Step 3: Implement structural and psycholinguistic aggregators**

Use token layers `sentence_id`, `t_unit_id`, `clause_id`, `clause_type`,
`phrase_type`, `sentence_status`, and `yngve_depth`. Implement:

```text
morph_sentence_count                  morph_t_unit_count
morph_words_per_sentence              morph_words_per_t_unit
morph_words_per_clause                morph_clauses_per_sentence
morph_coordinate_phrase_count         morph_complex_nominal_count
morph_verb_phrase_count               morph_embedding_count
morph_dependent_clause_ratio          morph_well_formed_sentence_ratio
morph_incomplete_sentence_ratio       morph_reduced_sentence_ratio
morph_yngve_depth_mean                morph_yngve_depth_max
semantic_idea_density                 semantic_proposition_density
lex_frequency_mean                    lex_log_frequency_mean
lex_familiarity_mean                  lex_age_of_acquisition_mean
lex_imageability_mean                 lex_concreteness_mean
```

Psycholinguistic keys consume numeric layers of the same base name. Missing or
incomplete target-token coverage is unavailable, not partially averaged.

- [ ] **Step 4: Implement reviewed error/discourse aggregators**

Use `error_type`, `discourse_role`, `cohesion_type`,
`pronoun_reference_correct`, `topic_relevant`, and `information_unit` layers.
Implement count and per-target-token ratio for error types `phonemic`,
`phonetic`, `semantic`, `visual`, `morphological`, `inflectional`, `syntactic`,
`closed_class`, `neologism`, `perseveration`, `circumlocution`,
`indefinite_term`, and `word_finding`. Implement:

```text
discourse_referential_cohesion_ratio   discourse_temporal_cohesion_ratio
discourse_causal_cohesion_ratio        discourse_correct_pronoun_ratio
discourse_local_lexical_coherence      discourse_global_coherence_ratio
discourse_topic_maintenance_ratio      discourse_marker_ratio
discourse_relevant_detail_ratio        discourse_irrelevant_detail_ratio
discourse_microproposition_count       discourse_macroproposition_count
discourse_information_unit_count       discourse_content_accuracy_ratio
discourse_information_efficiency_per_min
```

Local lexical coherence is mean Jaccard overlap of adjacent target utterance
word sets after Vietnamese normalization; an empty union is undefined.

- [ ] **Step 5: Promote structured task formulas without importing legacy APIs**

Copy the already-tested minimal formulas into `task_scores.py` under current
types and keys; do not call deprecated modules at runtime. Register:

```text
task_picture_concept_coverage          task_picture_concept_density
task_picture_repeat_ratio              task_picture_entity_coverage
task_picture_action_coverage           task_recall_idea_coverage
task_recall_idea_density               task_recall_repeat_ratio
task_recall_order_score                task_fluency_response_count
task_fluency_valid_count               task_fluency_valid_unique
task_fluency_repeats                   task_fluency_intrusions
task_fluency_first_half_valid          task_fluency_second_half_valid
task_fluency_production_change         task_fluency_rate
task_fluency_clusters                  task_fluency_cluster_size_mean
task_fluency_switches
```

Use only the target speaker's normalized word tokens and explicit timestamps.
Keep the existing picture, recall, phonemic, and semantic spec validation rules.

- [ ] **Step 6: Make the adult-neuro orchestrator return every new key**

Extend `extract_adult_neuro_features(document, *, target_speaker=None,
recording_id="", task_spec=None)` additively. Without a
task spec, task keys are `NaN` plus `MISSING_ANNOTATION`; all existing values
and utterance rows remain byte-for-byte compatible.

- [ ] **Step 7: Run focused/regression tests, Ruff, and commit**

Run: `rtk proxy uv run pytest tests/speech_features/test_neuro_linguistic.py tests/speech_features/test_linguistic.py tests/speech_features/test_tasks.py -q`

Run: `rtk proxy uv run ruff check src/speech_features/features/linguistic tests/speech_features/test_neuro_linguistic.py`

Commit: `feat: add neuro linguistic and task measures`

---

### Task 6: Optional eGeMAPSv02 standardized acoustic pack

**Files:**
- Modify: `pyproject.toml`
- Create: `src/speech_features/features/standardized/__init__.py`
- Create: `src/speech_features/features/standardized/definitions.py`
- Create: `src/speech_features/features/standardized/opensmile_adapter.py`
- Create: `tests/speech_features/test_standardized_acoustic.py`

**Interfaces:**
- Consumes: Task 1 catalog and mono audio array.
- Produces: `extract_egemaps_features(audio, sample_rate, *,
  recording_id="", speaker_id="") -> tuple[dict[str, float], tuple[FeatureIssue, ...], dict]`
  and the `standardized_acoustic` pack.

- [ ] **Step 1: Write failing lazy-import and stable-schema tests**

```python
def test_core_import_does_not_import_opensmile():
    subprocess.run([sys.executable, "-c",
                    "import speech_features, sys; assert 'opensmile' not in sys.modules"],
                   check=True)


def test_adapter_prefixes_exact_88_columns(fake_opensmile):
    values, issues, provenance = extract_egemaps_features(np.zeros(16000), 16000)
    assert len(values) == 88
    assert all(key.startswith("egemaps_") for key in values)
    assert provenance["opensmile_feature_set"] == "eGeMAPSv02"
    assert not issues
```

- [ ] **Step 2: Run RED**

Run: `rtk proxy uv run pytest tests/speech_features/test_standardized_acoustic.py -q`

Expected: missing package modules and optional extra.

- [ ] **Step 3: Add the optional extra and lazy adapter**

```toml
standardized-acoustic = [
    "opensmile>=2.5,<3",
]
```

Import openSMILE only inside the extraction call. Instantiate
`opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
feature_level=opensmile.FeatureLevel.Functionals)` and call `process_signal`.
Convert each returned column to `egemaps_` plus a lowercase ASCII-safe name.
Freeze the expected raw 88-column tuple in `definitions.py`; mismatches raise
`EXTRACTION_ERROR`. Absence of the dependency returns all 88 keys as `NaN` plus
one `MISSING_OPTIONAL_DEPENDENCY` issue and provenance declaring unavailable.

- [ ] **Step 4: Register standardized metadata and provenance**

Set `language_scope="language_sensitive"`,
`evidence_level="standard_feature_set"`, and the GeMAPS DOI. Record package
version, raw feature-set name, and feature-level name.

- [ ] **Step 5: Run focused/packaging tests, Ruff, and commit**

Run: `rtk proxy uv run pytest tests/speech_features/test_standardized_acoustic.py tests/speech_features/test_packaging.py -q`

Run: `rtk proxy uv run ruff check src/speech_features/features/standardized tests/speech_features/test_standardized_acoustic.py`

Commit: `feat: add optional egemaps acoustic pack`

---

### Task 7: Pipeline integration, generated inventory, documentation, and full verification

**Files:**
- Modify: `src/speech_features/extraction.py`
- Modify: `src/speech_features/cli.py`
- Modify: `src/speech_features/__init__.py`
- Modify: `src/speech_features/result.py`
- Modify: `README.md`
- Modify: `docs/feature-catalog-v1.md`
- Modify: `docs/feature-extraction.md`
- Create: `docs/neurodegenerative-feature-guide.md`
- Create: `tools/render_neuro_feature_inventory.py`
- Create: `docs/research/neurodegenerative-speech-feature-inventory.csv`
- Create: `docs/research/neurodegenerative-speech-feature-inventory.md`
- Modify: `tests/speech_features/test_pipeline.py`
- Modify: `tests/speech_features/test_cli.py`
- Modify: `tests/speech_features/test_documentation.py`
- Create: `tests/speech_features/test_evidence_inventory.py`

**Interfaces:**
- Consumes: all preceding pack extractors and evidence records.
- Produces: integrated `extract(..., task_spec=None)`, batch manifest optional
  `task_spec_path`, CLI metadata filters, root exports, rendered inventory, and
  user documentation.

- [ ] **Step 1: Write failing integration and inventory tests**

```python
def test_extract_composes_all_capability_packs(wav_path, annotated_document, task_spec):
    bundle = extract(wav_path, annotated_document,
                     packs=("acoustic", "adult_neuro", "motor_neuro"),
                     task_spec=task_spec)
    assert "spectral_mfcc_1_mean" in bundle.recordings
    assert "artic_vowel_space_area_hz2" in bundle.recordings
    assert "task_picture_concept_coverage" in bundle.recordings
    assert "diagnosis" not in bundle.provenance


def test_inventory_covers_every_catalog_key_and_explains_deferrals():
    rows = read_inventory_csv()
    catalog_keys = {definition.key for definition in list_features()}
    inventoried = {key for row in rows for key in row["feature_keys"].split("|") if key}
    assert catalog_keys <= inventoried
    assert all(row["reason"] for row in rows if row["status"] == "deferred")
```

- [ ] **Step 2: Run RED**

Run: `rtk proxy uv run pytest tests/speech_features/test_pipeline.py tests/speech_features/test_evidence_inventory.py -q`

Expected: new packs are not orchestrated and inventory files do not exist.

- [ ] **Step 3: Integrate pack data flow additively**

Add keyword-only `task_spec=None` to `extract`; validate it once with the
existing version-1 task-spec validator. Pass it only to adult/motor extractors.
Run `standardized_acoustic` through the audio array. Hash canonical task-spec
JSON into provenance. Add optional `task_spec_path` to manifest v2 rows, load it
per row, and preserve row-isolation behavior. Add new stable result issue codes:
`INVALID_TASK_ANNOTATION`, `UNCALIBRATED_AUDIO`, and
`MISSING_OPTIONAL_DEPENDENCY` without changing existing exception codes.

- [ ] **Step 4: Integrate CLI and root exports**

Expose motor, clinical-linguistic, structured-task, and eGeMAPS extraction
functions at the package root. Add repeatable `list-features` filters
`--domain`, `--language-scope`, `--task`, `--disorder`, and `--evidence-level`.
Add `--task-spec` to single and batch extraction where accepted. Preserve
existing JSON/table columns unless a filter is explicitly supplied.

- [ ] **Step 5: Render and validate the evidence inventory**

`tools/render_neuro_feature_inventory.py` imports the catalog and evidence
records, validates them, and writes deterministic CSV/Markdown sorted by
domain/candidate/key. Include all implemented catalog keys plus deferred rows
for learned embeddings, ASR-derived confidence/perplexity, semantic embedding
coherence, proprietary measures, ComParE, calibrated absolute loudness without
calibration, and features lacking a reproducible source formula.

Run: `rtk proxy uv run python tools/render_neuro_feature_inventory.py`

Run again and assert `rtk git diff --exit-code` for the two generated inventory
files after staging only the expected implementation files.

- [ ] **Step 6: Update user documentation**

Document pack selection, metadata queries, annotation layer names, task-spec
examples, optional installation, issue semantics, Vietnamese validation limits,
future child-pack extension, and the research-not-diagnosis boundary. Generate
the catalog table from `list_features`; do not hand-maintain a second formula
source of truth.

- [ ] **Step 7: Run focused integration tests**

Run: `rtk proxy uv run pytest tests/speech_features/test_pipeline.py tests/speech_features/test_cli.py tests/speech_features/test_documentation.py tests/speech_features/test_evidence_inventory.py -q`

Expected: PASS.

- [ ] **Step 8: Run full verification**

Run: `rtk proxy uv run pytest -q`

Run: `rtk proxy uv run ruff check src tests tools`

Run: `rtk proxy uv run ruff format --check src tests tools`

Run: `rtk proxy uv build`

Run wheel smoke test in a fresh temporary virtual environment and assert:

```python
import speech_features
assert len(speech_features.list_features()) > 170
assert {"acoustic", "adult_neuro", "motor_neuro", "standardized_acoustic"} <= set(
    speech_features.PACKS
)
```

- [ ] **Step 9: Review the final diff and commit**

Confirm no patient data, labels, credentials, build output, unrelated edits, or
eager optional imports are staged.

Commit: `feat: integrate neurodegenerative feature library expansion`
