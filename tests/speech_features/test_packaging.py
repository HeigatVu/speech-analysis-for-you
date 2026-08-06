import re
import tomllib
from pathlib import Path

import speech_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
PROJECT = PYPROJECT["project"]
PLACEHOLDER_DESCRIPTION = "Add your description here"


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
    assert PYPROJECT["build-system"]["requires"] == ["setuptools>=68"]


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
    assert _names(extras["legacy-ad"]) == ["scikit-learn"]
    assert "scikit-learn>=1.3" in extras["legacy-ad"]
    assert _names(extras["legacy-preprocessing"]) == sorted(
        ["hydra-core", "matplotlib", "omegaconf", "pydub", "tqdm"]
    )
    assert all(
        pin in extras["legacy-preprocessing"]
        for pin in (
            "hydra-core>=1.3",
            "omegaconf>=2.3",
            "pydub>=0.25",
            "matplotlib>=3.8",
            "tqdm>=4.67",
        )
    )
    assert _names(extras["dev"]) == ["pytest", "ruff"]


def test_ruff_target_matches_declared_python_floor():
    ruff = PYPROJECT["tool"]["ruff"]
    assert ruff["target-version"] == "py310"
