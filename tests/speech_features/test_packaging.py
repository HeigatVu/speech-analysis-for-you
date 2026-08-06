import tomllib
from pathlib import Path

import speech_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
PROJECT = PYPROJECT["project"]


def _names(deps):
    return sorted(d.split(">=", 1)[0] for d in deps)


def test_version_and_python_floor():
    assert PROJECT["version"] == "0.2.0"
    assert PROJECT["requires-python"] == ">=3.10"
    assert speech_features.__version__ == "0.2.0"


def test_setuptools_build_backend():
    assert PYPROJECT["build-system"]["build-backend"] == "setuptools.build_meta"


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
