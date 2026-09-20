"""Documentation verification for the 0.2 user/release documentation (Task 12).

Semantic checks over ``README.md`` and ``docs/``: the five user/release
documents exist and cross-link; the research-only boundary and prohibited
clinical claims are stated; Python and CLI quick starts name real public APIs
and commands; transcript JSON v2, the CHAT subset, the manual-review workflow,
and 0.1-to-0.2 migration guidance exist; every stable error code is
documented; the catalog document parses to exactly the live registered
feature keys with matching metadata and non-placeholder formula/missing-data
cells; and the pediatric extension example is documentation-only with the
two-member ``FeaturePack`` contract. The checks are semantic and compact; they
never snapshot exact prose.
"""

import re
from pathlib import Path

import speech_features

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"

DOCS_2026_08_06 = DOCS / "2026-08-06" / "say-vietnamese-speech-library" / "1"
DOCS_2026_08_10 = DOCS / "2026-08-10" / "neurodegenerative-speech-feature-expansion" / "1"

DOC_PATHS = {
    "README": REPO_ROOT / "README.md",
    "feature-extraction": DOCS_2026_08_06 / "feature-extraction.md",
    "transcript-formats": DOCS_2026_08_06 / "transcript-formats.md",
    "feature-catalog-v1": DOCS_2026_08_06 / "feature-catalog-v1.md",
    "neurodegenerative-feature-guide": DOCS_2026_08_10 / "neurodegenerative-feature-guide.md",
    "migration-0.2": DOCS_2026_08_06 / "migration-0.2.md",
    "review-summary": DOCS_2026_08_06 / "REVIEW-2026-08-06.md",
}


def _doc(name: str) -> str:
    path = DOC_PATHS[name]
    assert path.is_file(), f"documentation file {path} must exist"
    return path.read_text(encoding="utf-8")


def _require(text: str, *needles: str, subject: str) -> None:
    words = [line.lstrip("> \t").strip() for line in (text or "").splitlines()]
    lower = " ".join(words).lower()
    missing = [needle for needle in needles if needle.lower() not in lower]
    assert not missing, f"{subject} is missing: {missing}"


class TestDocumentsExistAndCrossLink:
    def test_five_user_docs_exist(self):
        for name in (
            "README",
            "feature-extraction",
            "transcript-formats",
            "feature-catalog-v1",
            "migration-0.2",
        ):
            assert DOC_PATHS[name].is_file(), f"{name} document must exist"

    def test_review_summary_uses_locked_headings(self):
        headings = re.findall(r"(?m)^#\s+(.+)$", _doc("review-summary"))
        assert headings == ["OpenCode implementation", "Agy review", "Resolution"]

    def test_readme_links_to_every_guide(self):
        readme = _doc("README")
        for path in (
            "docs/2026-08-06/say-vietnamese-speech-library/1/feature-extraction.md",
            "docs/2026-08-06/say-vietnamese-speech-library/1/transcript-formats.md",
            "docs/2026-08-06/say-vietnamese-speech-library/1/feature-catalog-v1.md",
            "docs/2026-08-10/neurodegenerative-speech-feature-expansion/1/neurodegenerative-feature-guide.md",
            "docs/2026-08-06/say-vietnamese-speech-library/1/migration-0.2.md",
        ):
            assert path in readme, f"README must link to {path}"

    def test_guides_cross_link_to_readme_and_each_other(self):
        guides = ("feature-extraction", "transcript-formats", "feature-catalog-v1", "migration-0.2")
        for name in guides:
            text = _doc(name)
            assert "README" in text, f"{name} must reference the README"
        for name in ("transcript-formats", "migration-0.2"):
            text = _doc(name)
            assert "feature-extraction.md" in text, f"{name} must link feature-extraction"
        assert "migration-0.2.md" in _doc("transcript-formats"), (
            "transcript-formats must link migration-0.2"
        )


class TestResearchOnlyBoundary:
    def test_readme_declares_research_only(self):
        _require(_doc("README"), "research-only", "not a diagnostic", subject="README limitation")

    def test_readme_states_prohibited_claims(self):
        _require(
            _doc("README"),
            "no diagnosis",
            "not a screening tool",
            "no normative range",
            "no treatment recommendation",
            "no trained model",
            "no fixed clinical performance",
            subject="README prohibited claims",
        )

    def test_feature_doc_declares_research_only(self):
        _require(
            _doc("feature-extraction"),
            "research-only",
            "not a diagnostic",
            subject="feature-extraction",
        )

    def test_human_review_of_automated_annotations(self):
        _require(
            _doc("transcript-formats"),
            "human",
            "review",
            "automated",
            subject="human-review workflow",
        )

    def test_readme_documents_future_transcript_validation_boundary(self):
        _require(
            _doc("README"),
            "future",
            "automated transcription",
            "ground truth",
            "human review",
            "annotation provenance",
            subject="README future transcript validation",
        )


class TestQuickStartsNameRealContracts:
    def test_readme_python_quickstart_names_real_public_apis(self):
        readme = _doc("README")
        for symbol in ("load_document", "extract", "list_features"):
            assert symbol in readme, f"README quick start must show {symbol}"
            assert hasattr(speech_features, symbol), f"{symbol} must be a real public API"

    def test_readme_cli_quickstart_names_all_four_commands(self):
        readme = _doc("README")
        for command in ("validate", "convert", "extract", "list-features"):
            assert f"say-features {command}" in readme, (
                f"README must show the say-features {command} command"
            )

    def test_readme_documents_installation_floor_and_ceiling(self):
        _require(_doc("README"), "python 3.10", "3.13", "wheel", subject="README installation")

    def test_readme_documents_tables_issues_and_provenance(self):
        _require(
            _doc("README"),
            "recordings",
            "utterances",
            "issues",
            "provenance",
            subject="README outputs",
        )

    def test_readme_documents_testing_and_build_commands(self):
        _require(_doc("README"), "pytest", "ruff", subject="README testing commands")


class TestFeatureExtractionGuide:
    def test_documents_wav_and_resampling(self):
        _require(
            _doc("feature-extraction"), "pcm", "wav", "resampl", subject="feature-extraction audio"
        )

    def test_documents_target_speaker_isolation(self):
        _require(
            _doc("feature-extraction"),
            "target speaker",
            "TARGET_SPEAKER_REQUIRED",
            subject="feature-extraction isolation",
        )

    def test_allow_unaligned_only_on_direct_acoustic_pack(self):
        _require(
            _doc("feature-extraction"),
            "allow_unaligned",
            "acoustic",
            "UNALIGNED_SPEAKER",
            subject="feature-extraction unaligned fallback",
        )

    def test_documents_bundle_tables_and_prefixes(self):
        _require(
            _doc("feature-extraction"),
            "recording_id",
            "speaker_id",
            "utterance_id",
            subject="feature-extraction table prefixes",
        )

    def test_documents_pack_and_level_selection(self):
        _require(
            _doc("feature-extraction"),
            "packs",
            "levels",
            "recording",
            "utterance",
            subject="feature-extraction pack/level selection",
        )

    def test_documents_manifest_v2_row_shape(self):
        _require(
            _doc("feature-extraction"),
            "recording_id",
            "audio_path",
            "transcript_path",
            "target_speakers",
            subject="feature-extraction manifest v2",
        )

    def test_documents_batch_isolation(self):
        _require(
            _doc("feature-extraction"),
            "isolat",
            "severity",
            subject="feature-extraction batch isolation",
        )

    def test_documents_cli_exit_codes(self):
        _require(
            _doc("feature-extraction"),
            "exit",
            "0",
            "1",
            "2",
            subject="feature-extraction CLI exit codes",
        )

    def test_every_stable_error_code_is_documented(self):
        text = _doc("feature-extraction")
        missing = [code for code in speech_features.STABLE_ERROR_CODES if code not in text]
        assert not missing, f"stable error codes missing from the guide: {sorted(missing)}"

    def test_legacy_ad_evaluation_is_deprecated_compatibility_only(self):
        _require(
            _doc("feature-extraction"),
            "legacy",
            "deprecat",
            "evaluate_ad_baseline",
            "migration-0.2.md",
            subject="feature-extraction legacy section",
        )

    def test_documents_all_packs_task_specs_and_new_issue_codes(self):
        _require(
            _doc("feature-extraction"),
            "motor_neuro",
            "standardized_acoustic",
            "task_spec",
            "task_spec_path",
            "INVALID_TASK_ANNOTATION",
            "UNCALIBRATED_AUDIO",
            "MISSING_OPTIONAL_DEPENDENCY",
            subject="feature-extraction neuro packs",
        )


class TestNeurodegenerativeFeatureGuide:
    def test_guide_documents_layers_tasks_limits_and_extension_boundary(self):
        _require(
            _doc("neurodegenerative-feature-guide"),
            "research-only",
            "not a diagnostic",
            "Vietnamese",
            "validation",
            "annotation",
            "task spec",
            "motor_neuro",
            "standardized_acoustic",
            "child pack",
            "human review",
            subject="neurodegenerative feature guide",
        )

    def test_guide_names_reviewed_annotation_layers_and_optional_install(self):
        _require(
            _doc("neurodegenerative-feature-guide"),
            "lemma",
            "upos",
            "f0_hz",
            "segment_type",
            "information_unit",
            "standardized-acoustic",
            "opensmile",
            subject="neurodegenerative annotations and optional dependency",
        )

    def test_guide_uses_breath_group_layer_name(self):
        text = _doc("neurodegenerative-feature-guide")
        assert "breath_group" in text
        assert "breath_group_id" not in text, "guide must name the layer breath_group"

    def test_guide_documented_layers_match_extractor_contract(self):
        text = _doc("neurodegenerative-feature-guide")
        section = text.split("## Reviewed annotation layers", 1)[1]
        documented = set(re.findall(r"`([a-z0-9_]+)`", section.split("##", 1)[0]))
        assert "breath_group" in documented
        source = ""
        for path in sorted((REPO_ROOT / "src" / "speech_features" / "features").rglob("*.py")):
            source += path.read_text(encoding="utf-8")
        missing = [name for name in sorted(documented) if name not in source]
        assert not missing, (
            f"guide documents layer names absent from the extractor contract: {missing}"
        )


class TestTranscriptFormatsGuide:
    def test_documents_json_v2_shape_and_example(self):
        text = _doc("transcript-formats")
        _require(text, "version", "2", "document_id", subject="transcript-formats v2")
        assert re.search(r"```json[\s\S]*\"version\"\s*:\s*2", text), (
            "transcript-formats must contain a valid minimal JSON v2 example"
        )

    def test_documents_vietnamese_token_word_grouping(self):
        _require(
            _doc("transcript-formats"),
            "word_id",
            "syllable",
            "whitespace",
            subject="transcript-formats grouping",
        )

    def test_documents_nfc_casefold_and_d_duong(self):
        _require(
            _doc("transcript-formats"),
            "nfc",
            "casefold",
            "diacrit",
            "đ",
            subject="transcript-formats normalization",
        )

    def test_documents_chat_subset(self):
        _require(
            _doc("transcript-formats"),
            "@Begin",
            "@Languages",
            "@Participants",
            "@Media",
            "@End",
            "%mor",
            "%gra",
            "[/]",
            "[//]",
            "[*]",
            subject="transcript-formats CHAT subset",
        )

    def test_documents_unknown_tier_preservation_and_warnings(self):
        _require(
            _doc("transcript-formats"),
            "raw_tiers",
            "UNSUPPORTED_CHAT_TIER",
            subject="transcript-formats unknown tiers",
        )

    def test_documents_safe_overwrite(self):
        _require(
            _doc("transcript-formats"), "force", "overwrite", subject="transcript-formats overwrite"
        )

    def test_documents_manual_and_automated_workflows(self):
        _require(
            _doc("transcript-formats"),
            "manual",
            "automated",
            "human",
            subject="transcript-formats workflows",
        )


class TestMigrationGuide:
    def test_documents_names_and_api_table(self):
        _require(
            _doc("migration-0.2"),
            "speech-analysis-for-you",
            "speech_features",
            "extract_recording",
            "extract",
            "extract_manifest",
            "extract_batch",
            subject="migration names and API table",
        )

    def test_documents_manifest_v1_vs_v2(self):
        _require(
            _doc("migration-0.2"),
            "manifest",
            "version",
            "1",
            "2",
            "labels",
            subject="migration manifest contrast",
        )

    def test_documents_legacy_imports_and_extra(self):
        _require(
            _doc("migration-0.2"), "legacy.ad", "legacy-ad", subject="migration legacy imports"
        )

    def test_documents_deprecation_window(self):
        _require(
            _doc("migration-0.2"),
            "DeprecationWarning",
            "0.2.x",
            "0.3.0",
            subject="migration deprecation window",
        )

    def test_documents_notebook_retirement(self):
        _require(_doc("migration-0.2"), "notebook", subject="migration notebook retirement")

    def test_documents_removed_preprocessing_workflow(self):
        _require(
            _doc("migration-0.2"),
            "legacy-preprocessing",
            "removed",
            "Git history",
            subject="migration removed preprocessing workflow",
        )


def _catalog_rows(text):
    """Parse the catalog markdown table into {key: [row, ...]} plus the header."""
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
        rows.setdefault(row["key"], []).append(row)
    return header, rows


class TestCatalogV1:
    def test_catalog_rows_equal_live_registered_keys(self):
        header, rows = _catalog_rows(_doc("feature-catalog-v1"))
        live = {definition.key for definition in speech_features.list_features()}
        assert set(rows) == live, (
            f"catalog keys differ from live registry: "
            f"missing={sorted(live - set(rows))}, extra={sorted(set(rows) - live)}"
        )
        assert len(rows) == len(live)
        duplicates = {key: entries for key, entries in rows.items() if len(entries) != 1}
        assert not duplicates, f"catalog has duplicate rows: {sorted(duplicates)}"

    def test_catalog_metadata_matches_live_definitions(self):
        header, rows = _catalog_rows(_doc("feature-catalog-v1"))
        expected_columns = {
            "key",
            "pack",
            "level",
            "unit",
            "prerequisites",
            "population",
            "formula_version",
            "reference",
            "domain",
            "language_scope",
            "tasks",
            "disorders",
            "evidence_level",
            "formula",
            "missing_data",
        }
        assert set(header) == expected_columns, f"catalog columns differ: {header}"
        for definition in speech_features.list_features():
            row = rows[definition.key][0]
            assert row["pack"] == definition.pack
            assert row["level"] == definition.level
            assert row["unit"] == definition.unit
            assert row["population"] == definition.population
            assert row["formula_version"] == str(definition.formula_version)
            assert row["reference"] == definition.reference
            assert row["domain"] == definition.domain
            assert row["language_scope"] == definition.language_scope
            assert row["tasks"] == (", ".join(definition.tasks) or "—")
            assert row["disorders"] == (", ".join(definition.disorders) or "—")
            assert row["evidence_level"] == definition.evidence_level
            expected_prereqs = ", ".join(definition.prerequisites) or "—"
            assert row["prerequisites"] == expected_prereqs

    def test_catalog_formula_and_missing_data_cells_are_specific(self):
        header, rows = _catalog_rows(_doc("feature-catalog-v1"))
        placeholders = ("todo", "tbd", "placeholder", "xxx", "n/a")
        for key, entries in rows.items():
            for row in entries:
                for column in ("formula", "missing_data"):
                    cell = row[column]
                    assert cell.strip(), f"{key} {column} must be non-empty"
                    lower = cell.lower()
                    assert not any(token in lower for token in placeholders), (
                        f"{key} {column} contains a placeholder: {cell!r}"
                    )

    def test_pediatric_example_is_documentation_only(self):
        text = _doc("feature-catalog-v1")
        _require(
            text,
            "FeaturePack",
            "name",
            "version",
            "PACKS",
            "static",
            "entry-point",
            subject="catalog pediatric section",
        )
        assert "not implemented" in text.lower() or "no pediatric" in text.lower(), (
            "catalog must state pediatric features are not implemented"
        )
        match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
        assert match, "catalog must contain a python example"
        example = match.group(1)
        for line in example.splitlines():
            assignment = re.match(r"^\s*([a-zA-Z_]\w*)\s*=", line)
            if assignment:
                assert assignment.group(1) in ("name", "version"), (
                    f"pediatric example must only set name/version, got {assignment.group(1)!r}"
                )
