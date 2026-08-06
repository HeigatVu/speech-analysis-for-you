"""Behavioral tests for the leakage-safe evaluation baseline (Task 5).

The baseline runs on in-memory feature rows (participant-level), with binary
labels supplied *separately* from the features. It aggregates each
participant's recordings into a single participant-level case (so a
participant with more recordings never gets extra weight), embeds all
imputation/scaling strictly inside each training fold, and splits participants
with a stratified *grouped* outer CV plus a grouped inner CV for L2-C
selection, repeated over several seeded outer splits. This is a research-only
baseline: metrics and bootstrap intervals are reported for cohort
characterisation, never as a diagnosis or a clinical cutoff.

Only NumPy/SciPy/scikit-learn are used. No notebooks, WAVs, or transcripts.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from speech_features.evaluation import (
    C_GRID,
    _default_pipe,
    _pick_c,
    _select_c,
    bootstrap_ci,
    evaluate_ad_baseline,
)

TASKS = (
    "picture_desc_1",
    "picture_desc_2",
    "immediate_recall",
    "delayed_recall",
    "phonemic_fluency",
    "semantic_fluency",
)


def _cohort_rows(n_ad=6, n_hc=6, tasks=TASKS, noise=0.0, shuffled_f1=False):
    """Deterministic synthetic feature rows.

    ``f1`` carries the class signal; by default AD/HC values are separated by a
    wide gap. When ``shuffled_f1`` is set, each *participant* gets a unique,
    class-shuffled ``f1`` value -- no learnable boundary exists, so a model that
    performs well must be reading labels/participant identities it was not meant
    to see.
    """
    rng_shuf = np.random.RandomState(123)
    perm = rng_shuf.permutation(range(n_ad + n_hc))
    rows = []
    for i in range(n_ad):
        for t in tasks:
            rows.append(
                {
                    "participant_id": f"P{999 + i}",
                    "task": t,
                    "features": {
                        "f1": float(perm[i]),
                        "f2": float(1.0 + i),
                        "f3": float(2 * (i % 2)),
                    },
                }
            )
    for i in range(n_hc):
        for t in tasks:
            rows.append(
                {
                    "participant_id": f"C{999 + i}",
                    "task": t,
                    "features": {
                        "f1": float(perm[n_ad + i]),
                        "f2": float(20.0 + i),
                        "f3": float(2 * (i % 2)),
                    },
                }
            )
    if not shuffled_f1:
        for r in rows:
            base = int(r["participant_id"][1:]) - 999
            r["features"]["f1"] = float(
                10.0 + base if r["participant_id"].startswith("P") else 50.0 + base
            )
    else:
        # Scramble every common feature: no class boundary can be learned from a
        # training fold, so a good AUROC is only reachable by leaking a test
        # participant's identity/label into training.
        shuffled = rng_shuf.permutation(range(n_ad + n_hc)).astype(float)
        ci = 0
        for r in rows:
            r["features"] = {
                "f1": float(shuffled[ci % len(shuffled)]),
                "f2": float(shuffled[(ci + 3) % len(shuffled)]),
                "f3": float(-shuffled[(ci + 5) % len(shuffled)]),
            }
            ci += 1
    if noise:
        rng = np.random.RandomState(0)
        for r in rows:
            r["features"]["f1"] += float(rng.uniform(-noise, noise))
            r["features"]["f2"] += float(rng.uniform(-noise, noise))
    return rows


def _labels(rows):
    return {_pid(r): ("AD" if str(_pid(r)).startswith("P") else "HC") for r in rows}


def _pid(row):
    return row["participant_id"] if isinstance(row, dict) else row.participant_id


class TestParticipantSeparationAndNoDuplication:
    def test_one_row_per_participant_and_no_task_duplication(self):
        rows = _cohort_rows()
        labels = _labels(rows)
        result = evaluate_ad_baseline(
            rows, labels, seed=7, outer_splits=3, outer_repeats=2, inner_splits=2
        )
        participants = sorted({r["participant_id"] for r in rows})
        assert sorted(result.participant_predictions) == participants
        assert len(result.participant_predictions) == len(participants)

    def test_no_train_test_participant_leakage_chance_level_auroc(self):
        # Each participant owns a unique, class-shuffled feature value; a model
        # trained on train participants sees no learnable boundary and cannot
        # memorise test participants (they never appear in train). A near-perfect
        # AUROC would prove participant identity/labels leaked across folds.
        rows = _cohort_rows(n_ad=6, n_hc=6, shuffled_f1=True)
        labels = _labels(rows)
        result = evaluate_ad_baseline(
            rows, labels, seed=1, outer_splits=3, outer_repeats=4, inner_splits=2
        )
        aucs = np.array([m["auroc"] for m in result.fold_metrics if np.isfinite(m["auroc"])])
        assert aucs.size > 0
        assert float(np.mean(aucs)) < 0.8


class TestNoLabelLeakageAndFeatureFiltering:
    def test_id_label_provenance_and_non_numeric_features_excluded(self):
        rows = _cohort_rows()
        labels = _labels(rows)
        for r in rows:
            f = r["features"]
            f["diagnosis"] = float(0.0 if r["participant_id"].startswith("P") else 1.0)
            f["participant_id"] = r["participant_id"]
            f["input_hashes"] = {t: "abc" for t in TASKS}
            f["category"] = "word"
        result = evaluate_ad_baseline(
            rows, labels, seed=7, outer_splits=3, outer_repeats=1, inner_splits=2
        )
        for bad in ("diagnosis", "participant_id", "input_hashes", "category"):
            assert bad not in result.feature_columns
        assert set(result.feature_columns).issubset({"f1", "f2", "f3"})

    def test_labels_must_be_supplied_separately_from_features(self):
        rows = _cohort_rows()
        with pytest.raises(TypeError):
            evaluate_ad_baseline(rows)  # labels are a required, separate argument

    def test_domain_features_with_id_like_substrings_are_preserved(self):
        # The old broad substring patterns ("id", "task", ...) wrongly dropped
        # valid domain features such as idea_coverage, idea_density and
        # valid_* (which contain the substring "id" in "idea"/"valid"). These
        # must survive precise filtering.
        rows = _cohort_rows(noise=0.1)
        labels = _labels(rows)
        for i, r in enumerate(rows):
            f = r["features"]
            f["idea_coverage"] = float(0.1 * (i % 5))
            f["idea_density"] = float(0.2 + 0.05 * (i % 4))
            f["valid_words"] = float(50 + i)
            f["time_ratio"] = float(0.3 + 0.01 * (i % 7))
        result = evaluate_ad_baseline(
            rows, labels, seed=7, outer_splits=3, outer_repeats=1, inner_splits=2
        )
        for kept in ("idea_coverage", "idea_density", "valid_words", "time_ratio"):
            assert kept in result.feature_columns, f"{kept} was wrongly dropped"
        for dropped in ("participant_id", "input_hashes", "recording_path", "diagnosis"):
            assert dropped not in result.feature_columns, f"{dropped} leaked into features"

    def test_task_and_provenance_and_hash_fields_excluded_precisely(self):
        # Precise identifier filtering excludes genuine metadata/provenance/
        # label/path/hash/task fields even when they do not collide with a
        # domain feature's substring.
        rows = _cohort_rows()
        labels = _labels(rows)
        for r in rows:
            f = r["features"]
            f["recording_id"] = "rec"
            f["label"] = "AD"
            f["provenance"] = "p"
            f["audio_path"] = "/x"
            f["sha256"] = "ab" * 16
            f["task_id"] = "t"
        result = evaluate_ad_baseline(
            rows, labels, seed=7, outer_splits=3, outer_repeats=1, inner_splits=2
        )
        for dropped in ("recording_id", "label", "provenance", "audio_path", "sha256", "task_id"):
            assert dropped not in result.feature_columns, f"{dropped} leaked into features"
        assert set(result.feature_columns).issubset({"f1", "f2", "f3"})


class TestTrainingOnlyImputationAndScaling:
    def test_preprocessing_statistics_come_from_training_fold_only(self):
        # Imputation and scaling live in the sklearn Pipeline that is fit on the
        # training fold alone. A held-out test observation -- including an
        # extreme outlier that a leaky full-data fit would absorb into its
        # statistics -- must not contribute to the imputer median or the scaler
        # mean/std used to transform it.
        X_train = np.array([[1.0], [2.0], [3.0]])
        y_train = np.array([0, 0, 1])
        pipe = _default_pipe(c=1.0)
        pipe.fit(X_train, y_train)
        imputer = pipe.steps[0][1]
        scaler = pipe.steps[1][1]
        assert imputer.statistics_[0] == pytest.approx(2.0)  # median of train only
        assert scaler.mean_[0] == pytest.approx(np.mean([1.0, 2.0, 3.0]))

        X_test = np.array([[1e9]])  # extreme test value, never part of the fit
        transformed = scaler.transform(imputer.transform(X_test))
        assert np.isfinite(transformed).all()

    def test_out_of_fold_predictions_via_normal_normalization_only(self):
        # The outer-fit path predicts test participants through a pipeline fit
        # strictly on the outer training fold (see _default_pipe usage); running
        # the full baseline on a separable synthetic cohort must produce finite
        # out-of-fold probabilities for every participant.
        rows = _cohort_rows(n_ad=5, n_hc=5)
        labels = _labels(rows)
        res = evaluate_ad_baseline(
            rows, labels, seed=5, outer_splits=2, outer_repeats=2, inner_splits=2
        )
        for p in res.participant_predictions.values():
            assert np.isfinite(p["mean_probability"])


class TestInsufficientCohortWarnings:
    def test_requested_cv_reduced_and_warns_when_data_small(self):
        rows = _cohort_rows(n_ad=3, n_hc=3)
        labels = _labels(rows)
        result = evaluate_ad_baseline(
            rows, labels, seed=3, outer_splits=5, outer_repeats=2, inner_splits=4
        )
        assert result.warnings, "expected insuffficient-cohort warnings"
        joined = "\n".join(result.warnings).lower()
        assert "reduc" in joined or "insufficient" in joined

    def test_tiny_cohort_still_returns_a_result_with_coverage(self):
        rows = _cohort_rows(n_ad=2, n_hc=2)
        labels = _labels(rows)
        result = evaluate_ad_baseline(
            rows, labels, seed=1, outer_splits=2, outer_repeats=1, inner_splits=2
        )
        assert result.cohort_summary["n_participants"] == 4
        assert isinstance(result.feature_missingness, Mapping)

    def test_cohort_without_two_of_each_class_warns_and_skips_folds(self):
        # One HC participant cannot be stratified into two folds; the baseline
        # must warn and complete with empty fold metrics rather than crash.
        rows = _cohort_rows(n_ad=2, n_hc=1)
        labels = _labels(rows)
        result = evaluate_ad_baseline(
            rows, labels, seed=1, outer_splits=2, outer_repeats=2, inner_splits=2
        )
        assert result.warnings
        assert result.fold_metrics == ()


class TestMetricOutputs:
    def test_fold_metrics_and_selected_cs_and_summary_populated(self):
        rows = _cohort_rows(noise=0.5)
        labels = _labels(rows)
        result = evaluate_ad_baseline(
            rows, labels, seed=11, outer_splits=3, outer_repeats=2, inner_splits=2
        )
        assert result.fold_metrics and len(result.selected_cs) == len(result.fold_metrics)
        for m in result.fold_metrics:
            assert {"auroc", "balanced_accuracy", "sensitivity", "specificity"} <= set(m)
        for m in result.selected_cs:
            assert 0.001 < m["c"] <= 101
        assert "auroc" in result.metrics_summary and "auc_ci" in result.metrics_summary


class TestDeterminism:
    def test_same_seed_is_reproducible(self):
        rows = _cohort_rows(noise=0.3)
        labels = _labels(rows)
        a = evaluate_ad_baseline(
            rows, labels, seed=9, outer_splits=3, outer_repeats=2, inner_splits=2
        )
        b = evaluate_ad_baseline(
            rows, labels, seed=9, outer_splits=3, outer_repeats=2, inner_splits=2
        )
        assert a.participant_predictions == b.participant_predictions
        assert a.selected_cs == b.selected_cs
        assert [m["auroc"] for m in a.fold_metrics] == [m["auroc"] for m in b.fold_metrics]


class TestInnerCandidateEvaluation:
    def _training_block(self, n_per_class=50, signal=0.8):
        # One noisy, *overlapping* signal feature so that C=0.01's heavy L2
        # shrinkage genuinely under-fits the training subset relative to a
        # larger C when selecting on balanced accuracy over a grouped inner CV.
        # (A cleanly separable feature makes every C tie, since balanced
        # accuracy and AUROC are invariant to monotone scaling of the score.)
        rng = np.random.RandomState(0)
        x = np.concatenate(
            [rng.normal(signal, 2.0, n_per_class), rng.normal(-signal, 2.0, n_per_class)]
        )
        y = np.array([1] * n_per_class + [0] * n_per_class)
        groups = np.arange(2 * n_per_class)
        return x.reshape(-1, 1), y, groups

    def test_every_candidate_is_evaluated_and_best_c_selected(self):
        # Regression: the inner split iterator was consumed by the first C, so
        # only C=0.01 was ever measured and every fold picked 0.01 regardless of
        # the data. On a tuneable training block a stronger-but-regularized C
        # must be selectable.
        X, y, groups = self._training_block()
        best = _select_c(
            X,
            y,
            groups,
            inner_use=4,
            c_grid=C_GRID,
            random_state=42,
            warnings_list=[],
        )
        assert best > 0.01, f"larger C was never selected (got {best!r})"

    def test_tie_break_selects_smaller_c(self):
        # Deterministic tie-breaking: equal C scores resolve to the smaller
        # numeric C, independent of candidate-container ordering.
        assert _pick_c({1.0: 0.7, 0.01: 0.7, 10.0: 0.7}) == 0.01
        assert _pick_c({0.1: 0.9, 0.01: 0.9}) == 0.01
        assert _pick_c({10.0: 0.8, 100.0: 0.8}) == 10.0
        assert _pick_c({0.01: 0.6, 1.0: 0.9}) == 1.0  # higher score still wins


class TestBootstrapReproducibility:
    def test_same_random_state_gives_identical_ci(self):
        values = [0.5, 0.6, 0.7, 0.55, 0.62]
        lo1, hi1 = bootstrap_ci(values, n_resamples=200, random_state=42)
        lo2, hi2 = bootstrap_ci(values, n_resamples=200, random_state=42)
        assert lo1 == lo2 and hi1 == hi2

    def test_different_random_state_can_differ(self):
        values = [0.5, 0.6, 0.7, 0.55, 0.62, 0.58, 0.66]
        _, hi1 = bootstrap_ci(values, n_resamples=300, random_state=1)
        _, hi2 = bootstrap_ci(values, n_resamples=300, random_state=2)
        assert hi1 != hi2

    def test_ci_respects_alpha_and_ignores_nan(self):
        values = [float("nan"), 0.5, 0.6, 0.7]
        lo, hi = bootstrap_ci(values, n_resamples=100, random_state=3, alpha=0.10)
        assert 0.0 <= lo <= hi <= 1.0


class TestObjectRowsDuckTyping:
    def test_accepts_attribute_style_rows(self):
        class _Row:
            def __init__(self, participant_id, task, features):
                self.participant_id = participant_id
                self.task = task
                self.features = features

        rows = [
            _Row(r["participant_id"], r["task"], r["features"])
            for r in _cohort_rows(n_ad=4, n_hc=4)
        ]
        labels = _labels(rows)
        result = evaluate_ad_baseline(
            rows, labels, seed=2, outer_splits=2, outer_repeats=1, inner_splits=2
        )
        assert result.cohort_summary["n_participants"] == 8
