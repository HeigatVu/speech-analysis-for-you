"""Legacy AD-vs-HC evaluation baseline (moved from ``speech_features.evaluation``).

Supported 0.2.x legacy namespace: ``C_GRID``, ``EvaluationResult``,
``KNOWN_TASKS``, ``bootstrap_ci``, and ``evaluate_ad_baseline``.

Leakage-safe participant-level baseline:

Runs on in-memory feature rows (one or more recordings per participant) with
binary labels supplied **separately** from the features. Notes on honesty:

* **Labels are separate.** ``evaluate_ad_baseline`` takes ``labels`` as a
  required argument keyed by participant; a row's feature dict never carries the
  label. ID / label / provenance / non-numeric / invalid feature columns are
  dropped before modelling, via precise token/key filtering that keeps domain
  features such as ``idea_*`` and ``valid_*``.
* **One case per participant.** Each participant's recordings are aggregated to
  a single participant-level case (mean across the participant's recordings,
  NaN-aware), so a participant with more recordings never gets duplicated
  weight or leaks through multiple rows.
* **Preprocessing fits inside the training fold only.** Imputation (median) and
  scaling live in an sklearn ``Pipeline`` that is ``fit`` on each training fold
  alone; test participants never contribute to imputation or scaling statistics.
* **Grouped splits.** The outer loop is a strata-grouped CV by participant and
  the inner L2-C selection is a grouped CV too, so no participant ever straddles
  a train/test boundary.

This is a **research-only baseline** for cohort characterisation. It reports
fold metrics (AUROC, balanced accuracy, sensitivity, specificity), the per-fold
selected L2 ``C``, a participant prediction table, cohort/task coverage and
feature-missingness summaries, and warnings when the data cannot support the
requested folds. It does **not** claim diagnostic/clinical performance and
defines **no fixed score target**.
Deprecated through 0.2.x; eligible for removal in 0.3.0.
"""

from __future__ import annotations

import dataclasses
import re
from types import MappingProxyType

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

KNOWN_TASKS = (
    "picture_desc_1",
    "picture_desc_2",
    "immediate_recall",
    "delayed_recall",
    "phonemic_fluency",
    "semantic_fluency",
)
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)

# Precise feature-key tokens that must never enter the model feature matrix.
# Filtering is token/key based (split on non-alpha), so a substring such as
# "id" never falsely drops a domain feature like *idea_coverage* or
# *valid_words* while still excluding genuine ID / hash / path / label /
# provenance / task metadata.
_EXCLUDED_TOKENS = frozenset(
    {
        "id",
        "hash",
        "hashes",
        "sha",
        "path",
        "diagnos",
        "diagnosis",
        "label",
        "provenance",
        "task",
        "recording",
        "participant",
    }
)


# ---------------------------------------------------------------------------
# Bootstrap CI helper
# ---------------------------------------------------------------------------
def _check_random_state(seed):
    if seed is None or isinstance(seed, int):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    if isinstance(seed, np.random.Generator):
        return seed
    raise ValueError(
        f"{seed!r} cannot be used as a random seed; pass an int, RandomState or Generator"
    )


def bootstrap_ci(
    values,
    *,
    n_resamples: int = 2000,
    statistic=np.mean,
    random_state=None,
    alpha: float = 0.05,
):
    """Bootstrap percentile confidence interval for ``statistic`` over ``values``.

    Ignores non-finite entries, resamples with replacement a fixed number of
    times, and returns the ``(lower, upper)`` 100*(1-alpha) percentile interval.
    Passing an integer ``random_state`` makes the interval reproducible.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return (float("nan"), float("nan"))
    rng = _check_random_state(random_state)
    estimates = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        estimates[i] = statistic(rng.choice(values, size=values.size, replace=True))
    lo, hi = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


# ---------------------------------------------------------------------------
# Row coercion
# ---------------------------------------------------------------------------
def _get(row, key):
    """Read ``key`` from a mapping or an attribute object."""
    if isinstance(row, MappingProxyType) or hasattr(row, "keys"):
        return row[key]
    return getattr(row, key)


# ---------------------------------------------------------------------------
# Participant aggregation (one case per participant, NaN-aware)
# ---------------------------------------------------------------------------
def _collect_participants(rows, participant_key, task_key, feature_key):
    """Normalise rows into ``{participant: {features: [dict, ...], tasks: {}}}``."""
    by_participant = {}
    for row in rows:
        pid = _get(row, participant_key)
        task = _get(row, task_key)
        features = _get(row, feature_key)
        if not isinstance(features, dict):
            features = (
                dict(getattr(features, "items", lambda: {})().__self__)
                if hasattr(features, "keys") and features is not None
                else {}
            )
        case = by_participant.setdefault(pid, {"feature_dicts": [], "tasks": {}})
        case["feature_dicts"].append(features)
        case["tasks"][task] = case["tasks"].get(task, 0) + 1
    return by_participant


# ---------------------------------------------------------------------------
# Feature selection / matrix
# ---------------------------------------------------------------------------
def _select_and_collect(by_participant, warnings_list):
    """Return ordered feature columns ``(cols, is_valid)`` and per-participant rows.

    Columns that are non-numeric, ID/label/provenance-named, or constant/invalid
    across participants are dropped. Returns the usable ordered ``cols`` and a
    numeric 2-D array ``X`` (participants x cols) plus participant IDs.
    """
    # Gather the union of feature keys and tag columns that are ever non-numeric.
    columns = {}
    for case in by_participant.values():
        for fd in case["feature_dicts"]:
            for name, value in fd.items():
                if name not in columns:
                    columns[name] = {"numeric": True}
                if not columns[name]["numeric"]:
                    continue
                if isinstance(value, (str, bytes, dict, list, tuple, set, bool, type(None))):
                    columns[name]["numeric"] = False

    def _keep(name):
        lower = name.lower()
        tokens = {t for t in re.split(r"[^a-z]", lower) if t}
        if tokens & _EXCLUDED_TOKENS:
            return False
        return columns[name]["numeric"]

    selected = [name for name in columns if _keep(name)]
    # Precompute per-participant means twice (NaN-aware) for a stable column set.
    raw = []  # participant id -> {col: nanmean over its recordings}
    for pid, case in by_participant.items():
        vals = {c: [] for c in selected}
        for fd in case["feature_dicts"]:
            for c in selected:
                vals[c].append(float(fd.get(c, float("nan"))))
        raw.append(
            (
                pid,
                {
                    c: float(np.nanmean(vs)) if any(np.isfinite(vs)) else float("nan")
                    for c, vs in vals.items()
                },
            )
        )

    # Drop all-NaN and <2-distinct-value columns (no discriminative signal).
    usable = []
    for c in selected:
        values = [r[c] for _, r in raw]
        finite = [v for v in values if np.isfinite(v)]
        if len(finite) < 2 or len(set(np.round(finite, 12))) < 2:
            continue
        usable.append(c)

    X = np.asarray([[r[c] for c in usable] for _, r in raw], dtype=float)
    pids = [pid for pid, _ in raw]
    return usable, X, pids


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def _default_pipe(c: float):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(penalty="l2", C=c, max_iter=2000),
    )


def _pick_c(scores):
    """Pick the best ``C`` from ``{C: score}`` (higher score wins, ties → smaller C)."""
    best = None
    for c, s in scores.items():
        if best is None or s > scores[best] or (s == scores[best] and c < best):
            best = c
    return best


def _select_c(Xt, yt, inner_groups, inner_use, *, c_grid, random_state, warnings_list):
    """Pick the best L2 ``C`` on the outer training fold via a grouped inner CV.

    Candidates are scored by balanced accuracy (threshold-based, so L2
    regularisation genuinely trades off across ``C``); inner folds whose
    training subset has a single class are skipped (logistic regression needs
    both); when no inner split is feasible it returns a default ``C`` and warns
    instead of failing. The inner splits are materialised once as a list so every
    ``C`` candidate is scored on the *same* folds.
    """
    if inner_use < 2 or np.unique(yt).size < 2:
        warnings_list.append("inner CV not feasible; using default C=1.0")
        return 1.0
    try:
        inner = StratifiedGroupKFold(n_splits=inner_use, shuffle=True, random_state=random_state)
        inner_splits = list(inner.split(Xt, yt, groups=inner_groups))
    except ValueError as exc:
        warnings_list.append(f"inner CV not feasible; using default C=1.0 ({exc})")
        return 1.0
    scores = {}
    for c in c_grid:
        c_scores = []
        for i_tr, i_va in inner_splits:
            if np.unique(yt[i_tr]).size < 2:
                continue  # skip a single-class training subset
            pipe = _default_pipe(c)
            pipe.fit(Xt[i_tr], yt[i_tr])
            y_va_true = yt[i_va]
            y_va_prob = pipe.predict_proba(Xt[i_va])[:, 1]
            if np.unique(y_va_true).size < 2:
                # Single-class validation fold: a constant prediction earns
                # balanced accuracy 1 only by chance, so score it at chance.
                c_scores.append(0.5)
            else:
                c_scores.append(float(balanced_accuracy_score(y_va_true, y_va_prob >= 0.5)))
        scores[c] = float(np.mean(c_scores)) if c_scores else float("-inf")
    return _pick_c(scores)


def _metrics(y_true, y_score, pos_label=1):
    """Fold metrics with graph-safe handling of degenerate folds."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if np.unique(y_true).size < 2:
        return None  # both classes required for these metrics
    auroc = float(roc_auc_score(y_true, y_score))
    pred = (y_score >= 0.5).astype(int)
    balanced = float(balanced_accuracy_score(y_true, pred))
    tp = float(np.sum((y_true == pos_label) & (pred == pos_label)))
    fn = float(np.sum((y_true == pos_label) & (pred != pos_label)))
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    tn = float(np.sum((y_true != pos_label) & (pred != pos_label)))
    fp = float(np.sum((y_true != pos_label) & (pred == pos_label)))
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "auroc": auroc,
        "balanced_accuracy": balanced,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def evaluate_ad_baseline(
    rows,
    labels,
    *,
    participant_key="participant_id",
    task_key="task",
    feature_key="features",
    seed=42,
    outer_splits=5,
    outer_repeats=10,
    inner_splits=4,
    n_bootstrap=2000,
    min_task_coverage=5,
):
    """Run the repeated nested grouped-CV L2-logistic AD baseline.

    ``rows`` is any iterable of mappings/objects exposing ``participant_id``,
    ``task`` and ``features`` (a ``{name: numeric}`` dict). ``labels`` is a
    separate mapping participant_id -> binary class. See the module docstring for
    the leakage-safety guarantees. Returns an :class:`EvaluationResult`.
    """
    warn: list[str] = []
    by_participant = _collect_participants(rows, participant_key, task_key, feature_key)
    if not by_participant:
        raise ValueError("evaluate_ad_baseline requires at least one participant row")

    # ---- labels: map participant -> 0/1, AD positive ----
    y_map = {}
    for pid in by_participant:
        raw = labels[pid]
        if isinstance(raw, str):
            mapped = 1 if raw.upper() in ("AD", "TRUE", "1", "POSITIVE") else 0
        else:
            mapped = int(raw)
        y_map[pid] = mapped
    n_ad = sum(v == 1 for v in y_map.values())
    n_hc = sum(v == 0 for v in y_map.values())
    if not (n_ad and n_hc):
        warn.append("insufficient cohort: neither AD nor HC has participants; cannot model")

    feature_columns, X, pids = _select_and_collect(by_participant, warn)
    y = np.asarray([y_map[p] for p in pids], dtype=int)

    # ---- task / coverage summaries ----
    task_coverage = {t: 0 for t in KNOWN_TASKS}
    below_coverage = []
    for pid, case in by_participant.items():
        n_tasks = len(case["tasks"])
        for t in set(case["tasks"]):
            task_coverage[t] = task_coverage.get(t, 0) + 1
        if n_tasks < min_task_coverage:
            below_coverage.append((pid, n_tasks))
    if below_coverage:
        warn.append(
            "participants with fewer than "
            f"{min_task_coverage} tasks: {', '.join(f'{p}({n})' for p, n in sorted(below_coverage))}"
        )

    feature_missingness = {}
    for j, c in enumerate(feature_columns):
        col = X[:, j]
        feature_missingness[c] = float(np.count_nonzero(~np.isfinite(col)) / len(pids))

    if len(pids) < 2:
        warn.append("insufficient cohort: fewer than two participants")

    fold_metrics: list[dict] = []
    selected_cs: list[dict] = []
    prob_rows = [[] for _ in pids]  # pids -> list of out-of-fold probabilities

    # ---- reduce outer/inner splits when the cohort is too small ----
    outer_use = min(outer_splits, int(y.size))
    if n_ad and n_hc:
        outer_use = min(outer_use, int(min(n_ad, n_hc)))
    if outer_use < outer_splits and (int(y.size) or n_ad + n_hc):
        warn.append(f"requested outer_splits={outer_splits} reduced to {outer_use} for this cohort")
    if outer_use < 2:
        warn.append("insufficient cohort: cannot form at least two outer folds; no fold metrics")

    if outer_use >= 2:
        for repeat in range(outer_repeats):
            outer = StratifiedGroupKFold(
                n_splits=outer_use, shuffle=True, random_state=seed + repeat
            )
            fold_id = 0
            try:
                outer_splits_iter = outer.split(X, y, groups=pids)
            except ValueError as exc:
                warn.append(f"outer split failed (repeat {repeat}): {exc}")
                break
            for train_idx, test_idx in outer_splits_iter:
                Xt, yt = X[train_idx], y[train_idx]
                Xe, ye, pe = X[test_idx], y[test_idx], [pids[i] for i in test_idx]

                # ---- inner grouped CV for L2-C selection ----
                inner_use = min(inner_splits, int(yt.size))
                inner_use = min(inner_use, int(np.bincount(yt).min()))
                if inner_use < inner_splits:
                    warn.append(f"requested inner_splits={inner_splits} reduced to {inner_use}")
                inner_groups = np.asarray([pids[i] for i in train_idx])
                best_c = _select_c(
                    Xt,
                    yt,
                    inner_groups,
                    inner_use,
                    c_grid=C_GRID,
                    random_state=seed + repeat * 1000 + fold_id,
                    warnings_list=warn,
                )

                # ---- fit on the whole outer training fold with best C ----
                if np.unique(yt).size < 2:
                    # A single-class training fold cannot fit logistic regression;
                    # predict the majority class with its training prevalence.
                    warn.append("outer training fold had a single class; predicted prevalence")
                    proba = np.full(Xe.shape[0], float(yt.mean()), dtype=float)
                else:
                    pipe = _default_pipe(best_c)
                    pipe.fit(Xt, yt)
                    proba = pipe.predict_proba(Xe)[:, 1]
                metrics = _metrics(ye, proba)
                if metrics is None:
                    fold_metrics.append(
                        {
                            "repeat": repeat,
                            "fold": fold_id,
                            "auroc": float("nan"),
                            "balanced_accuracy": float("nan"),
                            "sensitivity": float("nan"),
                            "specificity": float("nan"),
                        }
                    )
                else:
                    fold_metrics.append({"repeat": repeat, "fold": fold_id, **metrics})
                selected_cs.append({"repeat": repeat, "fold": fold_id, "c": best_c})
                for j, pid in enumerate(pe):
                    prob_rows[pids.index(pid)].append(float(proba[j]))
                fold_id += 1

    # ---- participant prediction table (mean out-of-fold probability) ----
    participant_predictions = {}
    for pid, probs in zip(pids, prob_rows):
        if not probs:
            continue
        mean_prob = float(np.mean(probs))
        participant_predictions[pid] = {
            "mean_probability": mean_prob,
            "predicted_label": "AD" if mean_prob >= 0.5 else "HC",
        }

    # ---- aggregate metrics + bootstrap CIs (participant-strict folds) ----
    def _ci(name):
        vals = [m.get(name) for m in fold_metrics]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if not vals:
            return (float("nan"), float("nan"))
        return bootstrap_ci(vals, n_resamples=n_bootstrap, random_state=seed, statistic=np.mean)

    def _mean(name):
        vals = [m.get(name) for m in fold_metrics]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    # ---- outcome check-summary ----
    outmet = {}
    for name in ("auroc", "balanced_accuracy", "sensitivity", "specificity"):
        outmet[name] = _mean(name)
    outmet["auc_ci"] = _ci("auroc")
    outmet["balanced_accuracy_ci"] = _ci("balanced_accuracy")
    outmet["sensitivity_ci"] = _ci("sensitivity")
    outmet["specificity_ci"] = _ci("specificity")

    result = EvaluationResult(
        fold_metrics=tuple(fold_metrics),
        selected_cs=tuple(selected_cs),
        participant_predictions=MappingProxyType(dict(participant_predictions)),
        feature_columns=tuple(feature_columns),
        cohort_summary={
            "n_participants": len(pids),
            "n_ad": n_ad,
            "n_hc": n_hc,
        },
        task_coverage=MappingProxyType(dict(task_coverage)),
        feature_missingness=MappingProxyType(dict(feature_missingness)),
        metrics_summary=MappingProxyType(dict(outmet)),
        warnings=tuple(dict.fromkeys(warn)),
    )
    return result


@dataclasses.dataclass(frozen=True)
class EvaluationResult:
    """Outcome of :func:`evaluate_ad_baseline`. See the function's module docstring.

    ``fold_metrics`` / ``selected_cs`` are per-(repeat, fold); ``metrics_summary``
    holds means and bootstrap CIs; ``participant_predictions`` is one entry per
    participant. Research-only: never a diagnosis, no fixed score target.
    """

    fold_metrics: tuple
    selected_cs: tuple
    participant_predictions: MappingProxyType
    feature_columns: tuple
    cohort_summary: MappingProxyType
    task_coverage: MappingProxyType
    feature_missingness: MappingProxyType
    metrics_summary: MappingProxyType
    warnings: tuple


__all__ = [
    "C_GRID",
    "EvaluationResult",
    "KNOWN_TASKS",
    "bootstrap_ci",
    "evaluate_ad_baseline",
]
