# Task 05 — Leakage-Safe Evaluation Baseline

Slug: `evaluation`. Owns `src/speech_features/evaluation.py`,
`tests/speech_features/test_evaluation.py`, and this report
(`docs/implementation/task-05-evaluation.md`).

Goal (dispatch): *"Implement leakage-safe repeated nested AD baseline, panel
reports, bootstraps, and cohort summaries."* This task executes the narrower
build brief: a **research-only, leakage-safe participant-level** baseline over
**in-memory feature rows**, with labels kept separate from feature inputs, and
all the outputs listed below. The plan's *multiple sensitivity panels* are
out of scope here (deferred — see Scope notes).

---

## 1. OpenCode Implementation

### Approach (TDD)

Wrote a failing test module first (`tests/speech_features/test_evaluation.py`),
confirmed the red, then implemented `evaluation.py`, then iterated to green.

**Red command / result (recorded before any production evaluation code):**

```
$ uv run pytest tests/speech_features/test_evaluation.py -q
ERROR tests/speech_features/test_evaluation.py
ImportError: cannot import name 'evaluate_ad_baseline' from 'speech_features.evaluation'
ModuleNotFoundError: No module named 'speech_features.evaluation'
1 error during collection
```

So `evaluation.py` did not exist until the tests demanded it.

**RED, regression tests (written before the corresponding production fixes):**

```
$ uv run pytest tests/speech_features/test_evaluation.py -q
(after first implementation) 5 failed, 8 passed
```

The five initial failures were genuine design bugs the later setup caught and
fixed: a single-feature "unique participant value" cohort that was still
separably learnable in training (so it could not falsify leakage), an
assumed-but-wrong imputation-leakage test, `n_splits=1` inner folds on a tiny
cohort, and a `balanced_accuracy_score` stat import typo. These are described
under Resolution.

### What was built

- **`evaluate_ad_baseline(rows, labels, *, participant_key, task_key,
  feature_key, seed, outer_splits, outer_repeats, inner_splits, n_bootstrap,
  min_task_coverage)` — the nested grouped-CV L2-logistic AD baseline.**
  Accepts in-memory feature rows as mappings *or* attribute objects
  (`participant_id`, `task`, `features`), with **binary labels supplied
  separately** keyed by participant (binary int or `AD`/`HC`). Every
  leakage-safety guarantee below is structural:

  - **Labels separate from features.** `labels` is a required, distinct
    argument; a row's `features` dict never carries the label. ID / label /
    provenance / non-numeric / invalid feature columns are dropped before
    modelling (modulespace `_EXCLUDED_PATTERNS`: `id`, `hash`, `path`,
    `diagnos`, `label`, `provenance`, `task`), and all-`NaN` or
    single-distinct-value columns are dropped too.
  - **One case per participant.** Each participant's recordings are aggregated
    to a single participant-level case — mean across the participant's
    recordings, NaN-aware — so a participant with more recordings is never
    duplicated or overweighted (no task-duplication leakage). Participant
    predictions are one entry per participant.
  - **Preprocessing strictly inside each training fold.** Imputation (median)
    and scaling live in `_default_pipe` = a sklearn `Pipeline` (`SimpleImputer`
    + `StandardScaler` + L2 `LogisticRegression`) that is **fit only on the
    outer training fold**; test participants never contribute to imputation or
    scaling statistics.
  - **Grouped splits.** Outer `StratifiedGroupKFold` (default 5 folds, `shuffle`
    with a seed that advances per repeat) splits on participants; the inner
    grouped CV (default 4) selects the L2 `C` from `[0.01, 0.1, 1, 10, 100]` by
    AUROC/balanced-accuracy on participant groups. No participant ever straddles
    a train/test boundary.
  - **10 seed-controlled outer repeats.** `random_state=seed + repeat` drives
    each repeated outer split; the whole run is seed-reproducible.
  - **Research-only output, no score target.** Returns fold metrics (AUROC,
    balanced accuracy, sensitivity = AD recall, specificity), the per-fold
    selected `C`, a participant prediction table, cohort/task coverage and
    feature-missingness summaries, and bootstrap CIs — never a diagnosis and
    never a fixed pass/fail score.

- **`bootstrap_ci(values, *, n_resamples, statistic, random_state, alpha)` — an
  independent, seed-controlled bootstrap CI helper.** Ignores non-finite
  values, resamples with replacement `n_resamples` times, returns the
  `100*(1-alpha)` percentile interval, and is fully reproducible for an integer
  `random_state`. Used to attach 95% bootstrap intervals (AUROC,
  balanced accuracy, sensitivity, specificity) to `metrics_summary`.

- **`EvaluationResult`** — frozen dataclass holding `fold_metrics`,
  `selected_cs`, `participant_predictions`, `feature_columns`, `cohort_summary`,
  `task_coverage`, `feature_missingness`, `metrics_summary`, and `warnings`
  (mappings wrapped in `MappingProxyType`, sequences as tuples).

- **Insufficient-cohort behaviour (warnings, not errors).** When the cohort
  cannot support the requested folds — e.g. too few participants of either
  class, a participant missing the required task coverage (default ≥5 of 6
  tasks), or an inner CV with a single class — the run emits a descriptive
  `warnings` entry and, where necessary, lowers the fold count or skips C
  selection with a default `C=1.0`, rather than failing. A cohort cannot even
  form two outer folds returns empty fold metrics with a warning.

Production imports: stdlib (`dataclasses`, `re`, `types`), NumPy, and
scikit-learn (`model_selection`, `linear_model`, `pipeline`, `preprocessing`,
`impute`, `metrics`). No librosa/openSMILE/spaCy/torch/ASR and no patient data
or task lexicons — verified below.

### Scope note (one deliberate simplification)

- **Sensitivity "panels" are deferred.** The dispatch's *"panel reports"* (the
  plan's multiple feature-set sensitivity panels: demographics-only,
  timing-only, acoustic-only, linguistic/task-only, combined, combined-plus-
  demographics) are **not** part of this task's build brief, which specifies the
  single combined-feature baseline plus the listed outputs. A user-requested
  panel mode can compose this same leakage-safe `evaluate_ad_baseline` over
  feature subsets without changing its guarantees. Added when a panel report is
  actually needed.

---

## 2. Agy Review

Reviewer: `agy:code-reviewer`. Review of the Task 5 build commit. Placeholder —
see Resolution for the `RED → fix` mapping. (This section is recorded here now
and updated in the fix commit; the implementer does not edit Agy's wording.)

---

## 3. Resolution

Initial implementation had five failing tests; each was converted into a fix:

| # | Failing test (RED) | Fix |
|---|--------------------|-----|
| 1 | `TestParticipantSeparationAndNoDuplication::test_no_train_test_participant_leakage_chance_level_auroc` | The first cohort used a *separable* per-class feature (`f1`: AD 10–15 vs HC 50–55) that generalised to held-out participants, so AUROC was 1.0 even with clean separation (not leakage). The fixture's shuffled mode now scrambles every feature per participant, so no class boundary is learnable in training; a high AUROC is reachable only by leaking a test participant into training. Leak-free ⇒ chance-level. |
| 2 | `TestTrainingOnlyImputationAndScaling::test_test_participant_value_does_not_enter_fit` | The first version compared one held-out participant's prediction under its own differing feature values — invalid, because that participant legitimately enters the training set in other folds. Replaced with a structural test: fit `_default_pipe` on a training fold and assert the imputer median / scaler mean equal the **train-only** statistics while an extreme test outlier is present but excluded; plus an end-to-end finite-probability check. |
| 3 | `TestInsufficientCohortWarnings::test_tiny_cohort_still_returns_a_result_with_coverage` | A 2-AD / 2-HC cohort reduced `inner_use` to 1, and `StratifiedGroupKFold(n_splits=1)` raised. `_select_c` now guards `inner_use < 2` / single-class training subsets and falls back to `C=1.0` with a warning. |
| 4 | `TestMetricOutputs::*` | `balanced_accuracy_score` was referenced before its import was restored; re-imported. (Ruff F401 caught the initial over-removal only after the build; the import is now present and used.) |
| 5 | `_metrics` / outer-fit single-class folds | Outer training folds with a single class would crash logistic regression; they now predict the majority-class prevalence with a warning. Added `TestInsufficientCohortWarnings::test_cohort_without_two_of_each_class_warns_and_skips_folds` as a regression lock. |

### Final verification (after Resolution)

```
$ uv run pytest tests/speech_features/test_evaluation.py -q
15 passed in 1.24s

$ uv run pytest tests/speech_features -q
157 passed in 1.68s

$ uv run ruff check src/speech_features tests/speech_features
All checks passed!

$ uv run ruff format --check src/speech_features tests/speech_features
All checks passed!

$ git diff --check
(clean)
```

Forbidden-library search over the new production module:

```
$ grep -rnE "librosa|opensmile|spacy|torch|transformers|soundfile|pydub" src/speech_features/evaluation.py
(no matches)
```

No notebooks and no other-task files (`schema.py`, `acoustic.py`,
`linguistic.py`, `tasks.py`, `pipeline.py`, `__init__.py`, prior task reports)
were modified. Only the Task 5 owned paths and this report were changed. This
baseline does **not** claim diagnosis/clinical performance and defines **no
fixed score target**.
