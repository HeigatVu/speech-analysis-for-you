# Legacy AD Compatibility Namespace (`src/speech_features/legacy/`)

The supported **0.2.x** legacy namespace for the original Alzheimer's-disease
(AD) workflow. It keeps the 0.1.x pipeline, task scorers, and evaluation
baseline importable from the old module paths **with deprecation warnings**,
while the label-free `extract` / `extract_batch` API becomes the primary
interface.

The whole namespace is eligible for removal in **0.3.0** — new code should use
[`extract`/`extract_batch`](../extraction.py) instead.

---

## 1. Layout

| Path | Status |
|---|---|
| `legacy/` | Compatibility namespace; contains only `ad/` |
| `legacy/ad/` | The supported legacy AD surface |
| [`../pipeline.py`](../pipeline.py), [`../evaluation.py`](../evaluation.py), [`../tasks.py`](../tasks.py) | Top-level **deprecation shims** — re-export only, emit `DeprecationWarning`, hold no algorithm |

---

## 2. Lazy and Quiet by Design

`legacy/ad/__init__.py` resolves names through module-level `__getattr__`:

- `_PIPELINE_NAMES = ("BatchFailure", "BatchResult", "extract_manifest", "extract_recording")`
- `_EVALUATION_NAMES = ("C_GRID", "EvaluationResult", "KNOWN_TASKS", "bootstrap_ci", "evaluate_ad_baseline")`

A plain `import speech_features.legacy.ad` is therefore **warning-free and does
not import scikit-learn**. The evaluation module (and its scikit-learn
dependency) loads only when one of its names is actually accessed. This is what
keeps offline unit tests fast and dependency-light.

---

## 3. Modules

### `ad/pipeline.py` — manifest/task pipeline

Composes the acoustic and linguistic/task-spec extractors into a single
immutable `FeatureResult` with SHA-256 provenance. Audio is read through the
shared population-neutral [`audio`](../audio.py) PCM core.

Batch orchestration validates the manifest and runs each row through
`extract_recording`, isolating any failing row behind a stable
`FeatureExtractionError.code` **instead of aborting the batch**.

**Diagnosis never enters an extractor call or a `FeatureResult`.** The manifest
label is surfaced separately on `BatchResult.labels` for use by the evaluator
only — this separation is what makes the evaluation leakage-safe.

| Public name | Kind |
|---|---|
| `extract_recording` | function — one row to one `FeatureResult` |
| `extract_manifest` | function — manifest path to a `BatchResult` |
| `BatchResult` | class — successful rows plus labels, kept apart |
| `BatchFailure` | class — one isolated row failure and its stable code |

### `ad/tasks.py` — external task-spec scorers

Deterministic scoring of a participant `Transcript` against versioned task-spec
JSON: picture concept/entity/action aliases, recall idea aliases, phonemic
initials/exclusions, and semantic item aliases/subcategories.

Matching uses **exact NFC/casefold equality first**, then standard Levenshtein
edit-distance normalized similarity against a conservative, documented
threshold. **No diagnosis or label is ever read and no NLP/ASR model is used.**

| Public name | Kind |
|---|---|
| `score_picture`, `score_recall`, `score_phonemic`, `score_semantic` | task scorers |
| `levenshtein`, `normalised_similarity`, `match_alias`, `match_key` | matching primitives |

### `ad/evaluation.py` — AD-vs-HC baseline

A **leakage-safe participant-level** baseline over in-memory feature rows (one
or more recordings per participant), with binary labels supplied **separately**
from the features:

- `evaluate_ad_baseline` takes `labels` as a required argument keyed by
  participant; a row's feature dict never carries the label.
- ID / label / provenance / non-numeric / invalid feature columns are dropped
  before modelling, via precise token/key filtering that keeps domain features.
- `C_GRID` model selection uses grouped inner splits, so tuning never sees the
  held-out participants.

| Public name | Kind |
|---|---|
| `evaluate_ad_baseline` | function — baseline evaluation entry point |
| `EvaluationResult` | class — metrics and provenance of one evaluation |
| `bootstrap_ci` | function — confidence intervals |
| `C_GRID`, `KNOWN_TASKS` | constants |

---

## 4. Related

- [Package overview](../README.md) — the primary label-free API
- [Migration guide](../../../docs/2026-08-06/say-vietnamese-speech-library/1/migration-0.2.md) — moving off the 0.1.x names
- [Feature catalog](../../../docs/2026-08-06/say-vietnamese-speech-library/1/feature-catalog-v1.md) — the current feature contract
