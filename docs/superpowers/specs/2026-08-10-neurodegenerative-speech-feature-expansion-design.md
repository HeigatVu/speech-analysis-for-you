# SAY Neurodegenerative Speech Feature Expansion Design

## Approval and purpose

The user approved autonomous implementation on 2026-08-10 after reviewing the
capability-based architecture. This design expands SAY from its existing 170
Vietnamese adult-speech features into a broad, evidence-traceable research
library for Alzheimer disease (AD), mild cognitive impairment (MCI), primary
progressive aphasia (PPA), frontotemporal dementia (FTD), dementia with Lewy
bodies (DLB), Parkinson disease and Parkinson disease dementia (PD/PDD),
amyotrophic lateral sclerosis and motor-neuron disease (ALS/MND), Huntington
disease (HD), multiple sclerosis (MS), cerebellar and spinocerebellar ataxias,
progressive supranuclear palsy (PSP), multiple system atrophy (MSA), and
corticobasal syndrome (CBS).

SAY remains a descriptive feature-extraction library. It does not diagnose a
person, supply clinical cutoffs, train a disease classifier, or claim that an
association reported in another language is validated for Vietnamese.

## Evidence boundary

The catalog covers interpretable, hand-engineered speech and connected-language
measures reported by the primary systematic reviews and representative studies
listed in the research inventory. “All features” means all reviewed measures
that fit at least one of these reproducible contracts:

1. deterministically computed from PCM WAV and SAY's existing reviewed speaker
   alignment;
2. deterministically computed from the reviewed `SpeechDocument` transcript,
   token, word, POS, dependency, error, or discourse annotations;
3. deterministically scored from an explicit task specification, such as a
   picture concept list, story idea units, verbal-fluency category, sustained
   vowel, reading, or diadochokinetic task; or
4. returned verbatim by the versioned openSMILE eGeMAPSv02 standard feature set
   through an optional dependency.

Learned embeddings, end-to-end neural representations, automatic diagnosis,
ASR, automatic forced alignment, automatic POS/dependency parsing, proprietary
feature formulas, and ComParE's 6,373-functionals set are documented as
deferred rather than represented as equivalent hand-engineered features.
ComParE is deferred because exposing thousands of dynamically versioned columns
would overwhelm the stable catalog; users who need it can call openSMILE
directly until there is a concrete stable-key use case.

## Compatibility and architecture

The existing public functions, JSON version 2 document, result tables, feature
keys, `acoustic` pack, and `adult_neuro` pack remain valid. The design is
additive:

- `FeatureDefinition` gains immutable metadata with backward-compatible
  defaults: `domain`, `language_scope`, `tasks`, `disorders`, and
  `evidence_level`.
- `list_features` gains optional filters for those metadata fields while
  preserving key ordering, errors for unknown pack/level filters, and the
  existing return type.
- Stable prefixes are extended only where existing prefixes would be
  misleading: `resp_`, `artic_`, `rhythm_`, `semantic_`, `task_`, and
  `egemaps_`.
- `acoustic` remains the shared population-neutral signal pack.
- `adult_neuro` remains the reviewed adult cognitive-linguistic pack.
- `motor_neuro` contains task- or annotation-dependent motor-speech features.
- `standardized_acoustic` exposes eGeMAPSv02 behind the optional `opensmile`
  extra.

Pack names describe extraction capabilities, not diagnoses. A pause feature is
implemented once in `acoustic` and can carry evidence links to several
disorders. Extraction composes packs; it does not copy the same feature into a
pack for every disease.

No entry-point discovery, plugin manager, registry class hierarchy, or empty
`child_speech` placeholder is introduced. Static built-in registration remains
the smallest design that works. A future child-speech pack can reuse the same
population-neutral feature definitions and add child-specific task features
without changing the document or result schemas.

## Metadata contract

Each implemented `FeatureDefinition` declares:

- `domain`: `audio_quality`, `timing`, `respiration`, `phonation`, `prosody`,
  `spectral`, `articulation`, `rhythm`, `lexical`, `psycholinguistic`,
  `morphosyntactic`, `disfluency`, `semantic`, `discourse`, or `task`;
- `language_scope`: `language_independent`, `language_sensitive`,
  `language_dependent`, or `language_specific`;
- `tasks`: zero or more stable task identifiers, including
  `connected_speech`, `picture_description`, `story_recall`,
  `semantic_fluency`, `phonemic_fluency`, `reading`, `sustained_vowel`, and
  `ddk`;
- `disorders`: zero or more evidence tags from `ad`, `mci`, `ppa`, `ftd`,
  `dlb`, `pd`, `pdd`, `als`, `mnd`, `hd`, `ms`, `ataxia`, `psp`, `msa`, and
  `cbs`;
- `evidence_level`: `systematic_review`, `multi_study`, `single_study`,
  `standard_feature_set`, or `derived_companion`; and
- the existing unit, population, prerequisite feature keys, formula version,
  and reference URL or DOI.

Tuples are normalized in `__post_init__`, values are validated against frozen
allowed sets, and caller-supplied mutable sequences never remain attached to a
definition. Existing constructors remain valid through conservative defaults;
all built-in definitions are explicitly enriched rather than relying on those
defaults.

A generated CSV and Markdown inventory contain one row per reviewed candidate
with stable name, description, domain, language scope, task, disorders,
evidence source, implementation status, SAY key when implemented, and a precise
reason when deferred. The inventory distinguishes reported association from
Vietnamese validation.

## Feature families

### Shared audio, timing, respiration, and prosody

The existing duration, speech ratio, rate, pause, response-latency, overlap,
F0, intensity, jitter, shimmer, HNR, CPP, formant, spectral, and syllable-rhythm
features remain. The expansion adds review-supported companion summaries:

- total, median, IQR, coefficient of variation, and proportion for pauses and
  speech segments; between-utterance pause proportion; maximum local speech
  rate; timing-event rate, entropy, and acceleration;
- voice-break count, rate, and voiced-duration proportion; F0 range and robust
  semitone variability; intensity range and coefficient of variation;
- jitter RAP, PPQ5, and DDP; shimmer APQ3, APQ5, APQ11, and DDA; noise-to-
  harmonics ratio; pitch-period entropy, recurrence-period-density entropy,
  detrended fluctuation analysis, and correlation dimension;
- spectral energy, skewness, kurtosis, low/high spectral-energy ratio, and
  MFCC 1--13 mean, standard deviation, skewness, and kurtosis; and
- percentage-vocalic, VarcoV/VarcoC, raw and normalized pairwise variability
  indices when reviewed vowel/consonant interval annotations exist.

Loudness comparisons that require microphone calibration are emitted only when
the task specification declares calibrated amplitude. Otherwise absolute
between-speaker loudness features are unavailable with a structured issue.

### Motor speech, articulation, resonance, and task measures

The `motor_neuro` pack consumes explicit task and interval annotations. It
adds:

- vowel-space area, vowel-articulation index, formant-centralization ratio,
  vowel dispersion/distance, within-vowel formant variability, and formant
  transition slope;
- voice-onset time, stop-gap duration, consonant duration, fricative spectral
  moments, and resonance-frequency attenuation;
- DDK rate, inter-onset mean/median/SD/CV, instability, acceleration, decay,
  voiced-interval duration, and sequential-versus-alternating rate;
- maximum phonation time, breath-group count and duration, respiratory rate,
  pauses per breath, relative respiration loudness when calibrated, and
  gaping between voiced intervals; and
- subharmonic-interval proportion and sustained-vowel stability summaries.

Interval-dependent formulas never infer phoneme or breath boundaries. They use
reviewed annotation layers with stable names and emit `MISSING_ANNOTATION` for
missing or incomplete layers. A task feature is not silently computed from a
different elicitation task.

### Cognitive-linguistic and connected-speech measures

The existing token, word, syllable, lexical-diversity, POS, dependency,
disfluency, and conversation measures remain. The expansion adds:

- sentence, T-unit, clause, coordinate phrase, complex nominal, verb phrase,
  embedding, dependent clause, well-formed sentence, incomplete sentence,
  reduced sentence, words-per-clause, clauses-per-sentence, and Yngve depth
  measures when the corresponding reviewed annotations exist;
- idea/propositional density; corpus-frequency, log frequency, familiarity,
  age-of-acquisition, imageability, and concreteness summaries supplied by an
  explicit Vietnamese lexicon or token annotation layer;
- phonemic, phonetic, semantic, visual, morphological, inflectional,
  syntactic, closed-class, neologism, perseveration, circumlocution,
  indefinite-term, and word-finding error counts and rates from reviewed error
  annotations; and
- referential/temporal/causal cohesion, local lexical coherence, global/task
  coherence, correct-pronoun reference, topic maintenance, discourse markers,
  relevant/irrelevant details, micro/macropropositions, information units,
  content accuracy, and information efficiency from reviewed annotations and
  task specifications.

Local lexical coherence uses deterministic token-set overlap and is labeled a
derived companion. Semantic embedding coherence is deferred because this cycle
does not introduce a model dependency.

### Structured elicitation tasks

The current deterministic legacy scorers are promoted into the current catalog
without changing their formulas:

- picture description: concept/entity/action coverage, content density, and
  repeated-content measures;
- story recall: idea coverage, density, repetition, and order preservation;
- semantic and phonemic fluency: valid items, unique items, repetitions,
  intrusions, first/second-half production, production change, clusters,
  cluster size, and switches.

Task specifications remain external data and are hashed in provenance. Labels
used for later clinical evaluation are never extraction inputs.

### Standardized acoustic set

The optional `standardized_acoustic` pack imports `opensmile` lazily and uses
the exact `eGeMAPSv02` functionals configuration. The adapter preserves the 88
openSMILE output names under deterministic `egemaps_` keys, records the
openSMILE version, and rejects an incompatible column set rather than silently
changing the schema. If the extra is absent, the pack fails with one stable
`MISSING_OPTIONAL_DEPENDENCY` error; core packs continue to import and run.

## Data flow and failure behavior

`extract` validates requested packs and prerequisites, creates the existing
population-neutral `ExtractionContext`, runs each requested pack, joins stable
recording/utterance columns, and appends structured issues. Independent feature
failures remain isolated. A missing annotation yields `NaN` plus one
deduplicated issue per affected feature and scope. Mathematically undefined
values also yield `NaN`, never infinity or a guessed zero.

New stable issue codes are limited to `INVALID_TASK_ANNOTATION`,
`UNCALIBRATED_AUDIO`, and `MISSING_OPTIONAL_DEPENDENCY`; existing failure codes
remain unchanged. Batch extraction continues after row-local failures.

Catalog and inventory validation run at import/test time and reject duplicate
keys, unknown enum values, missing evidence links, broken prerequisite keys,
or status/key inconsistencies.

## Vietnamese and future population handling

Signal measures such as duration or jitter are labeled
`language_independent`; F0, rhythm, vowel, consonant, and voice-quality measures
are `language_sensitive` because Vietnamese tone, phonation, dialect, and
syllable structure affect their interpretation; lexical, morphosyntactic,
semantic, and discourse measures are `language_dependent`; explicit Vietnamese
lexicons or category rules are `language_specific`.

No reviewed source found in this research pass establishes Vietnamese clinical
cutoffs for these disorders. Documentation must say that cross-language
features are hypotheses for Vietnamese validation. Future child-speech work
reuses shared acoustic keys but provides separate child tasks, reference data,
and evidence tags; it must not add age or diagnosis fields to extraction input.

## Dependencies and packaging

Core dependencies remain NumPy, SciPy, and pandas on Python 3.10--3.13.
`opensmile` is an optional extra and is never imported during normal package
import. No librosa, spaCy, torch, transformers, ASR, forced-aligner, or schema
framework becomes a core dependency. Standard PCM WAV remains the core audio
format.

This is an additive feature release. Package-version and changelog changes are
made only if the repository already maintains those files as release sources;
no tag or remote release is created.

## Agent execution and integration

Implementation is decomposed into independently testable slices in one written
plan. OpenCode with `deepseek/deepseek-v4-flash` and `max` effort implements
contract/evidence and acoustic slices. Muse Spark 1.2 Contributor implements a
non-overlapping linguistic/task slice. Agy with `gemini-3.6-flash-high` reviews
the specification, plan, tests, and integrated diff. Codex owns worktree setup,
task boundaries, review resolution, integration, and final verification.

Each implementation slice starts with a failing behavioral test, records the
expected RED result, makes the minimum production change, runs focused tests
and Ruff, and commits atomically. Agents work in isolated git worktrees or
strictly non-overlapping files; no agent edits the user's main worktree
concurrently.

## Verification and completion

Completion requires all of the following evidence:

1. catalog contract tests prove immutable metadata, validation, filtering,
   backward compatibility, unique keys, valid prerequisites, and evidence
   inventory consistency;
2. hand-calculated arrays and synthetic PCM WAV prove every new acoustic and
   motor formula, including undefined and missing-annotation cases;
3. hand-annotated Vietnamese fixtures prove every linguistic, discourse, and
   task formula without patient data;
4. optional-backend tests prove lazy import, exact eGeMAPSv02 schema mapping,
   missing-extra behavior, and provenance using a stub, plus a real smoke test
   when the extra is installable;
5. focused tests, the complete test suite, Ruff check, Ruff format check,
   package build, wheel installation, and public API/CLI smoke tests pass; and
6. the final generated inventory reports every reviewed candidate as
   `existing`, `implemented`, `optional`, or `deferred`, with no blank reason
   for a deferred row.

No real patient recordings, clinical labels, credentials, generated build
artifacts, or unrelated cleanup are committed.

## Principal sources

- AD/MCI acoustic review: https://doi.org/10.3389/fpsyg.2021.620251
- AD speech/language classification review: https://pmc.ncbi.nlm.nih.gov/articles/PMC8772820/
- Connected speech across neurodegenerative disorders: https://pmc.ncbi.nlm.nih.gov/articles/PMC5337522/
- FTD cross-linguistic speech/language review: https://discovery.ucl.ac.uk/id/eprint/10201085/
- Motor-neurodegenerative articulation review: https://pmc.ncbi.nlm.nih.gov/articles/PMC9950294/
- MND digital speech review: https://doi.org/10.1038/s41746-023-00959-9
- MS dysarthria review: https://doi.org/10.1016/j.msard.2018.08.015
- PD/MSA/PSP acoustic indices: https://doi.org/10.1038/s41531-022-00389-6
- PPA markers review: https://doi.org/10.1044/2020_AJSLP-20-00008
- openSMILE standard sets: https://audeering.github.io/opensmile-python/
