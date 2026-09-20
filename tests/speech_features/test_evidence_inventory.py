"""Generated neurodegenerative evidence inventory contracts (Task 7).

The generated artifacts (inventory CSV/Markdown and the catalog document) are
locked byte-for-byte against a fresh render; the exact live/pack/inventory/
status/deferred counts are locked explicitly; formula and missing-data cells
are feature-specific (the restored 170 legacy pairs plus per-family text for
every Task 2--6 key) and unique; the renderer refuses undocumented live keys;
and the standardized pack documents its single pack-level optional-dependency
missingness.
"""

from __future__ import annotations

import csv
import importlib.util
import re
from collections import Counter
from pathlib import Path

import pytest

import speech_features
from speech_features._legacy_feature_docs import LEGACY_FEATURE_DOCS

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "render_neuro_feature_inventory.py"
CSV_PATH = (
    REPO_ROOT
    / "docs"
    / "2026-08-10"
    / "neurodegenerative-speech-feature-expansion"
    / "1"
    / "neurodegenerative-speech-feature-inventory.csv"
)
MARKDOWN_PATH = (
    REPO_ROOT
    / "docs"
    / "2026-08-10"
    / "neurodegenerative-speech-feature-expansion"
    / "1"
    / "neurodegenerative-speech-feature-inventory.md"
)
CATALOG_PATH = (
    REPO_ROOT
    / "docs"
    / "2026-08-06"
    / "say-vietnamese-speech-library"
    / "1"
    / "feature-catalog-v1.md"
)

EXACT_LIVE_COUNT = 485
EXACT_PACK_COUNTS = {
    "acoustic": 167,
    "adult_neuro": 183,
    "motor_neuro": 47,
    "standardized_acoustic": 88,
}
EXACT_ROW_COUNT = 492
EXACT_STATUS_COUNTS = {"implemented": 397, "optional": 88, "deferred": 7}
EXACT_DEFERRED_CANDIDATES = {
    "ASR-derived confidence and perplexity",
    "ComParE feature set",
    "calibrated absolute loudness without calibration",
    "features lacking a reproducible source formula",
    "learned embeddings",
    "proprietary measures",
    "semantic embedding coherence",
}


def _rows():
    assert CSV_PATH.is_file(), "generated evidence inventory CSV is missing"
    with CSV_PATH.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _renderer():
    assert SCRIPT.is_file(), "inventory renderer is missing"
    spec = importlib.util.spec_from_file_location("render_neuro_feature_inventory", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _catalog_rows():
    """Parse the catalog markdown table into {key: row}."""
    text = CATALOG_PATH.read_text(encoding="utf-8")
    header = None
    rows = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        row = dict(zip(header, cells))
        assert row["key"] not in rows, f"duplicate catalog row for {row['key']}"
        rows[row["key"]] = row
    return rows


def test_inventory_covers_every_catalog_key_and_explains_deferrals():
    rows = _rows()
    catalog_keys = {definition.key for definition in speech_features.list_features()}
    inventoried = {key for row in rows for key in row["feature_keys"].split("|") if key}
    assert catalog_keys <= inventoried
    assert all(row["reason"] for row in rows if row["status"] == "deferred")


def test_inventory_has_required_metadata_and_exact_deferred_candidates():
    rows = _rows()
    assert rows
    required = {
        "candidate",
        "domain",
        "language_scope",
        "tasks",
        "disorders",
        "evidence_level",
        "status",
        "feature_keys",
        "reference",
        "reason",
    }
    assert set(rows[0]) == required
    deferred = {row["candidate"] for row in rows if row["status"] == "deferred"}
    assert deferred == EXACT_DEFERRED_CANDIDATES


def test_exact_live_pack_inventory_and_status_counts():
    definitions = speech_features.list_features()
    assert len(definitions) == EXACT_LIVE_COUNT
    assert Counter(definition.pack for definition in definitions) == EXACT_PACK_COUNTS
    rows = _rows()
    assert len(rows) == EXACT_ROW_COUNT
    assert Counter(row["status"] for row in rows) == EXACT_STATUS_COUNTS


def test_every_live_key_has_exactly_one_active_inventory_row():
    rows = _rows()
    occurrences = Counter(
        key
        for row in rows
        if row["status"] in {"implemented", "optional"}
        for key in row["feature_keys"].split("|")
        if key
    )
    live = {definition.key for definition in speech_features.list_features()}
    assert set(occurrences) == live
    assert all(count == 1 for count in occurrences.values())


def test_renderer_enforces_exact_counts():
    renderer = _renderer()
    assert renderer.EXPECTED_LIVE_COUNT == EXACT_LIVE_COUNT
    assert renderer.EXPECTED_PACK_COUNTS == EXACT_PACK_COUNTS
    assert renderer.EXPECTED_ROW_COUNT == EXACT_ROW_COUNT
    assert renderer.EXPECTED_STATUS_COUNTS == EXACT_STATUS_COUNTS
    assert renderer.EXPECTED_DEFERRED_CANDIDATES == EXACT_DEFERRED_CANDIDATES


def test_checked_in_artifacts_match_fresh_render_byte_for_byte(tmp_path):
    renderer = _renderer()
    csv_path = tmp_path / "inventory.csv"
    markdown_path = tmp_path / "inventory.md"
    catalog_path = tmp_path / "catalog.md"
    renderer.render(csv_path=csv_path, markdown_path=markdown_path, catalog_path=catalog_path)
    assert csv_path.read_bytes() == CSV_PATH.read_bytes(), (
        "fresh CSV render differs from checked-in"
    )
    assert markdown_path.read_bytes() == MARKDOWN_PATH.read_bytes(), (
        "fresh Markdown render differs from checked-in"
    )
    assert catalog_path.read_bytes() == CATALOG_PATH.read_bytes(), (
        "fresh catalog render differs from checked-in"
    )


def test_renderer_check_mode_detects_stale_artifacts(tmp_path):
    renderer = _renderer()
    assert renderer.check() == []
    marker = "| stale-marker |"
    original = CATALOG_PATH.read_text(encoding="utf-8")
    try:
        CATALOG_PATH.write_text(original + marker, encoding="utf-8")
        assert "catalog" in renderer.check()
    finally:
        CATALOG_PATH.write_text(original, encoding="utf-8")
    assert renderer.check() == []


def test_renderer_is_deterministic_and_writes_catalog_from_live_definitions(tmp_path):
    renderer = _renderer()
    csv_path = tmp_path / "inventory.csv"
    markdown_path = tmp_path / "inventory.md"
    catalog_path = tmp_path / "catalog.md"
    renderer.render(csv_path=csv_path, markdown_path=markdown_path, catalog_path=catalog_path)
    first = tuple(path.read_bytes() for path in (csv_path, markdown_path, catalog_path))
    renderer.render(csv_path=csv_path, markdown_path=markdown_path, catalog_path=catalog_path)
    second = tuple(path.read_bytes() for path in (csv_path, markdown_path, catalog_path))
    assert first == second
    catalog = catalog_path.read_text(encoding="utf-8")
    assert all(f"| {definition.key} |" in catalog for definition in speech_features.list_features())
    assert MARKDOWN_PATH.is_file(), "generated evidence inventory Markdown is missing"


class TestCatalogFormulaSpecificity:
    def test_legacy_catalog_cells_restore_the_exact_170_pairs(self):
        rows = _catalog_rows()
        legacy = {key: value for key, value in LEGACY_FEATURE_DOCS.items()}
        for key, (formula, missing_data) in legacy.items():
            row = rows[key]
            assert row["formula"] == formula, f"{key} formula drifted from the restored legacy text"
            assert row["missing_data"] == missing_data, (
                f"{key} missing-data text drifted from the restored legacy text"
            )

    def test_no_generic_fallback_text_in_catalog(self):
        rows = _catalog_rows()
        generic = (
            "Defined by the referenced extractor contract",
            "NaN plus a structured FeatureIssue when required input",
        )
        for key, row in rows.items():
            for column in ("formula", "missing_data"):
                assert not any(marker in row[column] for marker in generic), (
                    f"{key} {column} still uses generic fallback text"
                )

    def test_every_formula_missing_pair_is_unique(self):
        rows = _catalog_rows()
        pairs = [(row["formula"], row["missing_data"]) for row in rows.values()]
        assert len(pairs) == len(set(pairs)), "two live keys share identical formula/missing-data"

    def test_mfcc_family_cells_contain_coefficient_and_stat(self):
        rows = _catalog_rows()
        for key, row in rows.items():
            if not key.startswith("spectral_mfcc_"):
                continue
            coefficient, stat = key.rsplit("_", 1)
            coefficient = int(coefficient.split("_")[2])
            lower_formula = row["formula"].lower()
            assert str(coefficient) in lower_formula, f"{key} formula lacks coefficient"
            assert stat in lower_formula, f"{key} formula lacks stat {stat}"
            assert "insufficient_speech_frames" in row["missing_data"].lower()

    def test_error_family_cells_contain_error_type(self):
        rows = _catalog_rows()
        for key, row in rows.items():
            match = re.fullmatch(r"disfluency_(\w+)_error_(count|ratio)", key)
            if match is None or key in LEGACY_FEATURE_DOCS:
                continue
            error_type = match.group(1)
            lower_formula = row["formula"].lower()
            assert error_type in lower_formula, f"{key} formula lacks error type"
            assert "error_type" in lower_formula, f"{key} formula lacks the annotation layer"

    def test_egemaps_cells_contain_raw_functional_and_pack_level_missingness(self):
        rows = _catalog_rows()
        from speech_features.features.standardized.definitions import (
            EGEMAPS_KEYS,
            RAW_EGEMAPS_COLUMNS,
        )

        assert len(EGEMAPS_KEYS) == 88
        missing_texts = {rows[key]["missing_data"] for key in EGEMAPS_KEYS}
        assert len(missing_texts) == 1, "eGeMAPS missingness must be one pack-level text"
        assert "MISSING_OPTIONAL_DEPENDENCY" in next(iter(missing_texts))
        for raw, key in zip(RAW_EGEMAPS_COLUMNS, EGEMAPS_KEYS):
            row = rows[key]
            assert raw in row["formula"], f"{key} formula lacks the raw functional {raw}"

    @pytest.mark.parametrize(
        ("key", "needles", "missing_needle"),
        [
            ("time_pause_median_s", ("median", "pause_threshold"), ("NO_SPEECH",)),
            ("time_speech_segment_rate_per_min", ("segment", "minute"), ("MISSING_ANNOTATION",)),
            ("time_timing_event_entropy", ("entropy", "event"), ("NO_SPEECH",)),
            ("voice_break_count", ("break", "voiced frame"), ("INSUFFICIENT_VOICED_FRAMES",)),
            ("voice_jitter_rap", ("3", "period"), ("INSUFFICIENT_CYCLES",)),
            ("voice_dfa", ("detrended", "window"), ("INSUFFICIENT_VOICING",)),
            ("spectral_energy_mean_db", ("10*log10", "power"), ("INSUFFICIENT_SPEECH_FRAMES",)),
            (
                "spectral_low_high_energy_ratio_db",
                ("sample_rate/4", "power"),
                ("INSUFFICIENT_SPEECH_FRAMES",),
            ),
            ("morph_sentence_count", ("sentence_id", "distinct"), ("MISSING_ANNOTATION",)),
            ("morph_yngve_depth_mean", ("yngve_depth", "mean"), ("MISSING_ANNOTATION",)),
            ("semantic_proposition_density", ("proposition", "micro"), ("MISSING_ANNOTATION",)),
            ("lex_frequency_mean", ("frequency", "annotation"), ("MISSING_ANNOTATION",)),
            (
                "discourse_local_lexical_coherence",
                ("Jaccard", "adjacent"),
                ("INSUFFICIENT_TOKENS",),
            ),
            ("discourse_content_accuracy_ratio", ("reference", "information unit"), ()),
            (
                "task_picture_concept_coverage",
                ("concept_aliases", "distinct"),
                ("MISSING_ANNOTATION",),
            ),
            (
                "task_fluency_production_change",
                ("half", "valid"),
                ("MISSING_ANNOTATION",),
            ),
            ("artic_vowel_space_area_hz2", ("triangle", "i/a/u"), ("INVALID_TASK_ANNOTATION",)),
            ("artic_fricative_m1_mean", ("spectral_moment_1", "fricative"), ()),
            ("rhythm_npvi_v", ("npvi", "vowel"), ("MISSING_ANNOTATION",)),
            ("task_ddk_rate_syllables_s", ("ddk", "onset"), ("INVALID_TASK_ANNOTATION",)),
            ("resp_relative_loudness_db", ("speech_db", "respiration_db"), ("UNCALIBRATED_AUDIO",)),
            ("resp_pauses_per_breath", ("breath", "pause"), ()),
            ("task_max_phonation_time_s", ("utterance", "sustained_vowel"), ()),
            ("voice_sustained_power_sd_db", ("power_db", "sd"), ("INVALID_TASK_ANNOTATION",)),
        ],
    )
    def test_representative_new_cells_are_feature_specific(self, key, needles, missing_needle):
        rows = _catalog_rows()
        row = rows[key]
        lower_formula = row["formula"].lower()
        lower_missing = row["missing_data"].lower()
        for needle in needles:
            assert needle.lower() in lower_formula, (
                f"{key} formula lacks {needle!r}: {row['formula']}"
            )
        for needle in missing_needle:
            assert needle.lower() in lower_missing, (
                f"{key} missing_data lacks {needle!r}: {row['missing_data']}"
            )

    def test_renderer_raises_for_undocumented_live_key(self, monkeypatch, tmp_path):
        renderer = _renderer()

        def broken(key):
            raise KeyError(f"no documentation for {key!r}")

        monkeypatch.setattr(renderer, "documentation_for", broken)
        with pytest.raises(KeyError, match="no documentation"):
            renderer.render(
                csv_path=tmp_path / "i.csv",
                markdown_path=tmp_path / "i.md",
                catalog_path=tmp_path / "c.md",
            )


class TestLegacyMetadataBackfill:
    def test_legacy_acoustic_keys_use_reviewed_domains_and_scopes(self):
        legacy = set(LEGACY_FEATURE_DOCS)
        for definition in speech_features.list_features():
            if definition.pack != "acoustic" or definition.key not in legacy:
                continue
            assert definition.domain in {
                "audio_quality",
                "timing",
                "phonation",
                "prosody",
                "spectral",
            }, definition.key
            assert definition.language_scope in {
                "language_independent",
                "language_sensitive",
            }, definition.key
            assert definition.tasks, definition.key
            assert definition.disorders, definition.key

    def test_legacy_adult_keys_use_reviewed_domains_and_scopes(self):
        legacy = set(LEGACY_FEATURE_DOCS)
        for definition in speech_features.list_features():
            if definition.pack != "adult_neuro" or definition.key not in legacy:
                continue
            assert definition.domain in {
                "lexical",
                "morphosyntactic",
                "disfluency",
                "discourse",
            }, definition.key
            assert definition.language_scope in {
                "language_dependent",
                "language_sensitive",
            }, definition.key
            assert definition.tasks, definition.key
            assert definition.disorders, definition.key

    def test_timing_domain_includes_legacy_timing_keys(self):
        timing_keys = {d.key for d in speech_features.list_features(domain="timing")}
        assert {"time_pause_count", "time_speech_s", "time_words_per_min"} <= timing_keys
        assert "time_timing_event_rate_per_min" in timing_keys

    def test_audio_quality_domain_excludes_lexical_and_discourse(self):
        quality_keys = {d.key for d in speech_features.list_features(domain="audio_quality")}
        assert quality_keys
        assert not any(
            key.startswith(("lex_", "discourse_", "morph_", "disfluency_")) for key in quality_keys
        )
        assert "audio_duration_s" in quality_keys


class TestRendererEnforcement:
    def test_renderer_rejects_wrong_live_count(self, tmp_path, monkeypatch):
        renderer = _renderer()
        monkeypatch.setattr(renderer, "EXPECTED_LIVE_COUNT", 999)
        with pytest.raises(ValueError, match="live"):
            renderer.render(
                csv_path=tmp_path / "i.csv",
                markdown_path=tmp_path / "i.md",
                catalog_path=tmp_path / "c.md",
            )
