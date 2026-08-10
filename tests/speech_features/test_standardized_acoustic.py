import math
import re
import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pytest

from speech_features import InvalidAudioError, list_features
from speech_features.catalog import KEY_PREFIXES
from speech_features.features.standardized import extract_egemaps_features
from speech_features.features.standardized.definitions import RAW_EGEMAPS_COLUMNS
from speech_features.result import ExtractionError


# Frozen independently from the authoritative opensmile-python eGeMAPSv02
# Functionals feature_names output.
EXPECTED_RAW_COLUMNS = (
    "F0semitoneFrom27.5Hz_sma3nz_amean",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "F0semitoneFrom27.5Hz_sma3nz_percentile20.0",
    "F0semitoneFrom27.5Hz_sma3nz_percentile50.0",
    "F0semitoneFrom27.5Hz_sma3nz_percentile80.0",
    "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
    "F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope",
    "F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope",
    "F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope",
    "F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope",
    "loudness_sma3_amean",
    "loudness_sma3_stddevNorm",
    "loudness_sma3_percentile20.0",
    "loudness_sma3_percentile50.0",
    "loudness_sma3_percentile80.0",
    "loudness_sma3_pctlrange0-2",
    "loudness_sma3_meanRisingSlope",
    "loudness_sma3_stddevRisingSlope",
    "loudness_sma3_meanFallingSlope",
    "loudness_sma3_stddevFallingSlope",
    "spectralFlux_sma3_amean",
    "spectralFlux_sma3_stddevNorm",
    "mfcc1_sma3_amean",
    "mfcc1_sma3_stddevNorm",
    "mfcc2_sma3_amean",
    "mfcc2_sma3_stddevNorm",
    "mfcc3_sma3_amean",
    "mfcc3_sma3_stddevNorm",
    "mfcc4_sma3_amean",
    "mfcc4_sma3_stddevNorm",
    "jitterLocal_sma3nz_amean",
    "jitterLocal_sma3nz_stddevNorm",
    "shimmerLocaldB_sma3nz_amean",
    "shimmerLocaldB_sma3nz_stddevNorm",
    "HNRdBACF_sma3nz_amean",
    "HNRdBACF_sma3nz_stddevNorm",
    "logRelF0-H1-H2_sma3nz_amean",
    "logRelF0-H1-H2_sma3nz_stddevNorm",
    "logRelF0-H1-A3_sma3nz_amean",
    "logRelF0-H1-A3_sma3nz_stddevNorm",
    "F1frequency_sma3nz_amean",
    "F1frequency_sma3nz_stddevNorm",
    "F1bandwidth_sma3nz_amean",
    "F1bandwidth_sma3nz_stddevNorm",
    "F1amplitudeLogRelF0_sma3nz_amean",
    "F1amplitudeLogRelF0_sma3nz_stddevNorm",
    "F2frequency_sma3nz_amean",
    "F2frequency_sma3nz_stddevNorm",
    "F2bandwidth_sma3nz_amean",
    "F2bandwidth_sma3nz_stddevNorm",
    "F2amplitudeLogRelF0_sma3nz_amean",
    "F2amplitudeLogRelF0_sma3nz_stddevNorm",
    "F3frequency_sma3nz_amean",
    "F3frequency_sma3nz_stddevNorm",
    "F3bandwidth_sma3nz_amean",
    "F3bandwidth_sma3nz_stddevNorm",
    "F3amplitudeLogRelF0_sma3nz_amean",
    "F3amplitudeLogRelF0_sma3nz_stddevNorm",
    "alphaRatioV_sma3nz_amean",
    "alphaRatioV_sma3nz_stddevNorm",
    "hammarbergIndexV_sma3nz_amean",
    "hammarbergIndexV_sma3nz_stddevNorm",
    "slopeV0-500_sma3nz_amean",
    "slopeV0-500_sma3nz_stddevNorm",
    "slopeV500-1500_sma3nz_amean",
    "slopeV500-1500_sma3nz_stddevNorm",
    "spectralFluxV_sma3nz_amean",
    "spectralFluxV_sma3nz_stddevNorm",
    "mfcc1V_sma3nz_amean",
    "mfcc1V_sma3nz_stddevNorm",
    "mfcc2V_sma3nz_amean",
    "mfcc2V_sma3nz_stddevNorm",
    "mfcc3V_sma3nz_amean",
    "mfcc3V_sma3nz_stddevNorm",
    "mfcc4V_sma3nz_amean",
    "mfcc4V_sma3nz_stddevNorm",
    "alphaRatioUV_sma3nz_amean",
    "hammarbergIndexUV_sma3nz_amean",
    "slopeUV0-500_sma3nz_amean",
    "slopeUV500-1500_sma3nz_amean",
    "spectralFluxUV_sma3nz_amean",
    "loudnessPeaksPerSec",
    "VoicedSegmentsPerSec",
    "MeanVoicedSegmentLengthSec",
    "StddevVoicedSegmentLengthSec",
    "MeanUnvoicedSegmentLength",
    "StddevUnvoicedSegmentLength",
    "equivalentSoundLevel_dBp",
)


def _canonical_key(raw: str) -> str:
    return "egemaps_" + re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")


EXPECTED_KEYS = tuple(_canonical_key(raw) for raw in EXPECTED_RAW_COLUMNS)


def _install_fake_opensmile(monkeypatch, result) -> None:
    class FeatureSet:
        eGeMAPSv02 = "eGeMAPSv02"

    class FeatureLevel:
        Functionals = "Functionals"

    class Smile:
        def __init__(self, *, feature_set, feature_level):
            if feature_set != FeatureSet.eGeMAPSv02 or feature_level != FeatureLevel.Functionals:
                raise AssertionError("adapter selected the wrong openSMILE schema")

        def process_signal(self, audio, sample_rate):
            return result

    fake = types.SimpleNamespace(
        FeatureLevel=FeatureLevel,
        FeatureSet=FeatureSet,
        Smile=Smile,
        __version__="9.9.9",
    )
    monkeypatch.setitem(sys.modules, "opensmile", fake)


def test_core_import_does_not_import_opensmile_and_registers_pack():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import speech_features, sys; "
            "assert 'opensmile' not in sys.modules; "
            "assert len(speech_features.list_features(pack='standardized_acoustic')) == 88",
        ],
        check=True,
    )


def test_raw_schema_and_canonical_keys_are_frozen_unique_and_ascii_safe():
    assert RAW_EGEMAPS_COLUMNS == EXPECTED_RAW_COLUMNS
    assert len(EXPECTED_KEYS) == len(set(EXPECTED_KEYS)) == 88
    assert all(re.fullmatch(r"egemaps_[a-z0-9_]+", key) for key in EXPECTED_KEYS)
    assert "egemaps_" in KEY_PREFIXES


def test_all_88_definitions_are_registered_with_conservative_metadata():
    definitions = list_features(pack="standardized_acoustic")
    assert {definition.key for definition in definitions} == set(EXPECTED_KEYS)
    assert {definition.language_scope for definition in definitions} == {"language_sensitive"}
    assert {definition.evidence_level for definition in definitions} == {"standard_feature_set"}
    assert {definition.reference for definition in definitions} == {
        "https://doi.org/10.1109/TAFFC.2015.2457417"
    }
    assert all(
        definition.domain in {"articulation", "phonation", "prosody", "spectral", "timing"}
        for definition in definitions
    )
    assert all(definition.unit and definition.disorders for definition in definitions)


def test_optional_dependency_absence_returns_one_pack_level_issue(monkeypatch):
    monkeypatch.setitem(sys.modules, "opensmile", None)

    values, issues, provenance = extract_egemaps_features(
        np.zeros(16000), 16000, recording_id="r1", speaker_id="s1"
    )

    assert tuple(values) == EXPECTED_KEYS
    assert all(math.isnan(value) for value in values.values())
    assert len(issues) == 1
    assert issues[0].code == "MISSING_OPTIONAL_DEPENDENCY"
    assert issues[0].feature is None
    assert issues[0].recording_id == "r1"
    assert issues[0].speaker_id == "s1"
    assert provenance == {
        "available": False,
        "opensmile_version": None,
        "opensmile_feature_set": "eGeMAPSv02",
        "opensmile_feature_level": "Functionals",
    }


def test_adapter_returns_exact_88_values_and_provenance(monkeypatch):
    expected_values = tuple(float(index) for index in range(88))
    frame = pd.DataFrame([expected_values], columns=EXPECTED_RAW_COLUMNS)
    _install_fake_opensmile(monkeypatch, frame)

    values, issues, provenance = extract_egemaps_features(np.zeros(16000), 16000)

    assert tuple(values) == EXPECTED_KEYS
    assert tuple(values.values()) == expected_values
    assert issues == ()
    assert provenance == {
        "available": True,
        "opensmile_version": "9.9.9",
        "opensmile_feature_set": "eGeMAPSv02",
        "opensmile_feature_level": "Functionals",
    }


@pytest.mark.parametrize(
    "result",
    [
        pd.DataFrame([[0.0] * 87], columns=EXPECTED_RAW_COLUMNS[:-1]),
        pd.DataFrame(
            [[0.0] * 88],
            columns=(EXPECTED_RAW_COLUMNS[1], EXPECTED_RAW_COLUMNS[0], *EXPECTED_RAW_COLUMNS[2:]),
        ),
        pd.DataFrame([[0.0] * 88, [1.0] * 88], columns=EXPECTED_RAW_COLUMNS),
        object(),
    ],
    ids=("column-mismatch", "column-order", "row-schema", "result-type"),
)
def test_adapter_rejects_raw_schema_drift_with_stable_error(monkeypatch, result):
    _install_fake_opensmile(monkeypatch, result)

    with pytest.raises(ExtractionError) as exc_info:
        extract_egemaps_features(np.zeros(16000), 16000)

    assert exc_info.value.code == "EXTRACTION_ERROR"


@pytest.mark.parametrize(
    ("audio", "sample_rate"),
    [
        ([], 16000),
        (np.zeros((2, 2)), 16000),
        ([0.0, math.nan], 16000),
        (["not-numeric"], 16000),
        ([0.0], 0),
        ([0.0], "16000"),
    ],
)
def test_adapter_rejects_invalid_audio_before_optional_import(monkeypatch, audio, sample_rate):
    monkeypatch.setitem(sys.modules, "opensmile", None)

    with pytest.raises(InvalidAudioError) as exc_info:
        extract_egemaps_features(audio, sample_rate)

    assert exc_info.value.code == "INVALID_AUDIO"
