import re
import tomllib
from pathlib import Path

import speech_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
PROJECT = PYPROJECT["project"]
PLACEHOLDER_DESCRIPTION = "Add your description here"
OBSOLETE_REPOSITORY_PATHS = (
    ".DS_Store",
    "src/.DS_Store",
    "environment.yml",
    "src/config/main.yaml",
    "src/preprocessing/audio-preprocessing.py",
    "src/preprocessing/clapperboard.py",
    "src/utils/audio.py",
    "src/utils/file_io.py",
    "src/utils/visualization.py",
)


def test_obsolete_nonpackage_research_tools_are_absent():
    remaining = [path for path in OBSOLETE_REPOSITORY_PATHS if (PROJECT_ROOT / path).exists()]
    assert remaining == []


def _names(deps):
    """Parse dependency names with stdlib only, agnostic to version operators,
    environment markers, and extras suffixes."""
    names = []
    for dep in deps:
        name = dep.split(";", 1)[0].strip()
        name = re.split(r"[<>=!~]", name, 1)[0].strip()
        name = name.split("[", 1)[0].strip()
        names.append(name)
    return sorted(names)


def test_version_and_python_floor():
    assert PROJECT["version"] == "0.2.0"
    assert PROJECT["requires-python"] == ">=3.10"
    assert speech_features.__version__ == "0.2.0"


def test_setuptools_build_backend_and_exact_requirement():
    assert PYPROJECT["build-system"]["build-backend"] == "setuptools.build_meta"
    assert PYPROJECT["build-system"]["requires"] == ["setuptools>=77"]


def test_mit_license_metadata_and_file():
    assert PROJECT["license"] == "MIT"
    assert PROJECT["license-files"] == ["LICENSE"]
    license_text = (PROJECT_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in license_text
    assert "SAY contributors" in license_text


def test_setuptools_discovery_is_restricted_to_speech_features():
    find = PYPROJECT["tool"]["setuptools"]["packages"]["find"]
    assert find["where"] == ["src"]
    assert find["include"] in (
        ["speech_features*"],
        ["speech_features*", "say_transcribe*"],
    )


def test_description_is_an_accurate_research_library_description():
    description = PROJECT["description"]
    assert description != PLACEHOLDER_DESCRIPTION
    assert description.strip()
    lower = description.lower()
    assert "speech" in lower
    assert any(marker in lower for marker in ("research", "feature library", "descriptive"))


def test_console_script():
    assert PROJECT["scripts"]["say-features"] == "speech_features.cli:main"


def test_core_dependencies():
    assert sorted(PROJECT["dependencies"]) == ["numpy>=1.26", "pandas>=2.1", "scipy>=1.11"]


def test_no_legacy_runtime_dependencies_in_core():
    names = _names(PROJECT["dependencies"])
    for banned in (
        "librosa",
        "opensmile",
        "import-ipynb",
        "ipykernel",
        "nbconvert",
        "noisereduce",
        "pyloudnorm",
        "soundfile",
    ):
        assert banned not in names


def test_optional_extras():
    extras = PROJECT["optional-dependencies"]
    assert set(extras) in (
        {"legacy-ad", "standardized-acoustic", "dev"},
        {"legacy-ad", "standardized-acoustic", "dev", "transcribe"},
    )
    assert _names(extras["legacy-ad"]) == ["scikit-learn"]
    assert "scikit-learn>=1.3" in extras["legacy-ad"]
    assert extras["standardized-acoustic"] == ["opensmile>=2.5,<3"]
    assert _names(extras["dev"]) == ["pytest", "ruff"]


def test_ruff_target_matches_declared_python_floor():
    ruff = PYPROJECT["tool"]["ruff"]
    assert ruff["target-version"] == "py310"
