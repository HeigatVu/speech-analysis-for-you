"""Documentation verification for the feature library (Task 6).

Checks that the user-facing docs (``README.md`` and ``docs/feature-extraction.md``)
make the research-only limitation unambiguous, reference every public contract
symbol that exists in the package, and spell out the math-first formulas for
each feature family. These are test-first doc checks: they fail until the docs
actually say these things, guarding against a feature library that ships with an
old or over-promising README.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
FEATURE_DOC = REPO_ROOT / "docs" / "feature-extraction.md"


@pytest.fixture(scope="module")
def readme_text() -> str:
    assert README.is_file(), "README.md must exist"
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def feature_doc_text() -> str:
    assert FEATURE_DOC.is_file(), "docs/feature-extraction.md must exist"
    return FEATURE_DOC.read_text(encoding="utf-8")


def _require(text: str, *needles: str, subject: str) -> None:
    lower = "\n".join(line.lower() for line in (text or "").splitlines())
    missing = [n.lower() for n in needles if n.lower() not in lower]
    assert not missing, f"{subject} is missing: {missing}"


class TestResearchOnlyLimitation:
    """The docs must state the research-only / not-a-diagnostic limitation."""

    def test_readme_declares_research_only(self, readme_text):
        _require(
            readme_text,
            "research-only",
            "not a diagnostic",
            "clinical",
            subject="README research-only limitation",
        )

    def test_feature_doc_declares_research_only(self, feature_doc_text):
        _require(
            feature_doc_text,
            "research-only",
            "not a diagnostic",
            "clinical",
            subject="feature-extraction research-only limitation",
        )

    def test_no_fixed_performance_promise(self, readme_text, feature_doc_text):
        for name, text in (("README", readme_text), ("feature-extraction", feature_doc_text)):
            assert "accuracy" not in text.lower() or "do not promise" in text.lower(), (
                f"{name} must not promise a fixed performance number"
            )


class TestPublicContractReferences:
    """Every public API symbol the package exports must be referenced in the docs."""

    SOURCES = {
        "README": lambda: README.read_text(encoding="utf-8"),
        "feature-extraction": lambda: FEATURE_DOC.read_text(encoding="utf-8"),
    }

    @pytest.mark.parametrize(
        "symbol",
        [
            "ExtractionConfig",
            "FeatureResult",
            "extract_recording",
            "extract_manifest",
            "read_wav",
            "evaluate_ad_baseline",
            "validate_manifest",
            "validate_transcript",
            "validate_task_spec",
            "manifest_hashes",
            "sha256_file",
            "BatchFailure",
            "BatchResult",
            "nfc",
        ],
    )
    def test_public_api_symbol_is_documented(self, symbol):
        docs = "\n".join(src() for src in self.SOURCES.values())
        assert symbol in docs, f"public API symbol {symbol!r} is not documented"


class TestManifestAndTranscriptContracts:
    def test_readme_documents_wav_contract(self, readme_text):
        _require(readme_text, "WAV", "PCM", "resampl", subject="README WAV contract")

    def test_feature_doc_documents_transcript_contract(self, feature_doc_text):
        _require(
            feature_doc_text,
            "transcript",
            "word",
            "filler",
            "fragment",
            "noise",
            "participant",
            "examiner",
            subject="feature doc transcript contract",
        )

    def test_feature_doc_documents_manifest_fields(self, feature_doc_text):
        _require(
            feature_doc_text,
            "participant_id",
            "task",
            "recording_id",
            "diagnosis",
            "age",
            subject="feature doc manifest fields",
        )


class TestFormulaDocumentation:
    """Math-first formulas for the feature families must be written out."""

    def test_readme_and_feature_doc_link(self, readme_text):
        assert (
            "feature-extraction" in readme_text.lower()
            or "docs/feature-extraction.md" in readme_text
        ), "README must link to docs/feature-extraction.md"

    @pytest.mark.parametrize(
        "family_marker",
        [
            "ttr",  # lexical diversity
            "hapax",  # lexical diversity
            "leven",  # task-spec matching similarity
            "nccf",  # acoustic pitch autocorrelation
            "hamming",  # acoustic windowing
            "flatness",  # spectral flatness
            "nested",  # evaluation: repeated nested grouped CV
            "bootstrap",  # evaluation: participant-bootstrap CIs
        ],
    )
    def test_feature_family_formula_is_documented(self, feature_doc_text, family_marker):
        assert family_marker in feature_doc_text.lower(), (
            f"feature-extraction.md must document the formula for {family_marker!r}"
        )


class TestVerificationAndHonesty:
    def test_feature_doc_documents_quality_flags_and_provenance(self, feature_doc_text):
        _require(
            feature_doc_text,
            "quality",
            "sha-256",
            "provenance",
            subject="feature doc quality/provenance",
        )

    def test_feature_doc_documents_label_separation(self, feature_doc_text):
        _require(feature_doc_text, "label", "separate", subject="feature doc label separation")

    def test_feature_doc_documents_batch_failure(self, feature_doc_text):
        _require(feature_doc_text, "fail", "isolat", subject="feature doc batch failure behavior")

    def test_feature_doc_documents_cohort_and_seed(self, feature_doc_text):
        _require(
            feature_doc_text,
            "task coverage",
            "seed",
            "reproducible",  # covers the reproducible-seed procedure
            "ci",  # confidence interval
            subject="feature doc cohort minimum / task coverage / seed / CI",
        )
