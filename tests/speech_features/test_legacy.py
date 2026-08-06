"""Task 11: legacy AD compatibility and notebook retirement contracts.

Covers the deprecation shims for ``speech_features.pipeline``,
``speech_features.tasks`` and ``speech_features.evaluation``, the lazy
warning-free ``speech_features.legacy.ad`` namespace, shared PCM-core object
identity, the mutable ``_TASK_SCORER`` registry, numerical parity between the
old and new extraction paths, and the absence of tracked notebooks and
``import_ipynb`` usage in tracked source.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

SRC = str(Path(__file__).resolve().parents[2] / "src")


def _run_py(code: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter with ``src`` on the path."""
    return subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {SRC!r})\n{code}"],
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# Fresh-process import isolation
# ---------------------------------------------------------------------------
class TestImportIsolation:
    def test_import_speech_features_is_warning_free_and_sklearn_free(self):
        code = (
            "import sys, warnings\n"
            "with warnings.catch_warnings(record=True) as caught:\n"
            "    warnings.simplefilter('always')\n"
            "    import speech_features\n"
            "assert not caught, [str(w.message) for w in caught]\n"
            "assert 'sklearn' not in sys.modules\n"
        )
        result = _run_py(code)
        assert result.returncode == 0, result.stderr

    def test_import_legacy_ad_is_warning_free_and_sklearn_free(self):
        code = (
            "import sys, warnings\n"
            "with warnings.catch_warnings(record=True) as caught:\n"
            "    warnings.simplefilter('always')\n"
            "    import speech_features.legacy.ad as ad\n"
            "assert not caught, [str(w.message) for w in caught]\n"
            "assert 'sklearn' not in sys.modules\n"
            "assert callable(ad.extract_recording)\n"
            "assert callable(ad.extract_manifest)\n"
            "assert 'sklearn' not in sys.modules\n"
            "from speech_features.legacy.ad import evaluate_ad_baseline\n"
            "assert callable(evaluate_ad_baseline)\n"
            "assert 'sklearn' in sys.modules\n"
        )
        result = _run_py(code)
        assert result.returncode == 0, result.stderr

    def test_root_read_wav_is_a_warning_free_core_export(self):
        code = (
            "import warnings\n"
            "with warnings.catch_warnings(record=True) as caught:\n"
            "    warnings.simplefilter('always')\n"
            "    import speech_features\n"
            "    from speech_features import read_wav\n"
            "    from speech_features.audio import read_wav as core_read_wav\n"
            "assert not caught, [str(w.message) for w in caught]\n"
            "assert read_wav is core_read_wav\n"
        )
        result = _run_py(code)
        assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Deprecated module paths emit the exact migration target
# ---------------------------------------------------------------------------
_DEPRECATED_IMPORT = (
    "import warnings\n"
    "with warnings.catch_warnings(record=True) as caught:\n"
    "    warnings.simplefilter('always')\n"
    "    {statement}\n"
    "dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]\n"
    "assert dep, caught\n"
    "msg = ' '.join(str(w.message) for w in dep)\n"
    "assert {target!r} in msg, msg\n"
)


class TestDeprecatedModuleWarnings:
    def test_pipeline_module_warns_with_exact_targets(self):
        code = _DEPRECATED_IMPORT.format(
            statement="import speech_features.pipeline",
            target="speech_features.legacy.ad.pipeline",
        )
        result = _run_py(code)
        assert result.returncode == 0, result.stderr
        code = _DEPRECATED_IMPORT.format(
            statement="import speech_features.pipeline",
            target="speech_features.read_wav",
        )
        result = _run_py(code)
        assert result.returncode == 0, result.stderr

    def test_tasks_module_warns_with_exact_target(self):
        code = _DEPRECATED_IMPORT.format(
            statement="import speech_features.tasks",
            target="speech_features.legacy.ad.tasks",
        )
        result = _run_py(code)
        assert result.returncode == 0, result.stderr

    def test_evaluation_module_warns_with_exact_target(self):
        code = _DEPRECATED_IMPORT.format(
            statement="import speech_features.evaluation",
            target="speech_features.legacy.ad.evaluation",
        )
        result = _run_py(code)
        assert result.returncode == 0, result.stderr


class TestDeprecatedRootExportWarnings:
    @pytest.mark.parametrize(
        ("name", "target"),
        [
            ("BatchFailure", "speech_features.legacy.ad.pipeline.BatchFailure"),
            ("BatchResult", "speech_features.legacy.ad.pipeline.BatchResult"),
            ("extract_recording", "speech_features.legacy.ad.pipeline.extract_recording"),
            ("extract_manifest", "speech_features.legacy.ad.pipeline.extract_manifest"),
        ],
    )
    def test_root_export_warns_with_exact_target(self, name, target):
        code = _DEPRECATED_IMPORT.format(
            statement=f"from speech_features import {name}",
            target=target,
        )
        result = _run_py(code)
        assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Object identity: old shims delegate to the same legacy implementation
# ---------------------------------------------------------------------------
class TestObjectIdentity:
    def test_pipeline_shim_delegates_to_legacy_pipeline(self):
        from speech_features import pipeline
        from speech_features.legacy.ad import pipeline as legacy_pipeline

        for name in ("BatchFailure", "BatchResult", "extract_manifest", "extract_recording"):
            assert getattr(pipeline, name) is getattr(legacy_pipeline, name)
        assert pipeline.MissingInputError is legacy_pipeline.MissingInputError

    def test_tasks_shim_delegates_to_legacy_tasks(self):
        from speech_features import tasks
        from speech_features.legacy.ad import tasks as legacy_tasks

        assert tasks.__all__ == legacy_tasks.__all__
        for name in tasks.__all__:
            assert getattr(tasks, name) is getattr(legacy_tasks, name)

    def test_evaluation_shim_delegates_to_legacy_evaluation(self):
        from speech_features import evaluation
        from speech_features.legacy.ad import evaluation as legacy_evaluation

        for name in (
            "C_GRID",
            "EvaluationResult",
            "KNOWN_TASKS",
            "bootstrap_ci",
            "evaluate_ad_baseline",
        ):
            assert getattr(evaluation, name) is getattr(legacy_evaluation, name)
        for name in ("_default_pipe", "_pick_c", "_select_c"):
            assert getattr(evaluation, name) is getattr(legacy_evaluation, name)

    def test_task_scorer_registry_is_the_same_mutable_object(self, monkeypatch):
        from speech_features import pipeline
        from speech_features.legacy.ad import pipeline as legacy_pipeline

        assert pipeline._TASK_SCORER is legacy_pipeline._TASK_SCORER
        marker = object()
        monkeypatch.setitem(pipeline._TASK_SCORER, "semantic_fluency", marker)
        assert legacy_pipeline._TASK_SCORER["semantic_fluency"] is marker

    def test_audio_error_identity_across_all_paths(self):
        import speech_features
        from speech_features import audio, result
        from speech_features import pipeline

        assert audio.InvalidAudioError is result.InvalidAudioError
        assert audio.InvalidAudioError is speech_features.InvalidAudioError
        assert audio.InvalidAudioError is pipeline.InvalidAudioError
        assert audio.UnsupportedAudioError is result.UnsupportedAudioError
        assert audio.UnsupportedAudioError is pipeline.UnsupportedAudioError
        assert pipeline.read_wav is audio.read_wav
        assert speech_features.read_wav is audio.read_wav


# ---------------------------------------------------------------------------
# Numerical parity: old and new paths produce identical results
# ---------------------------------------------------------------------------
def _tone(freq, dur, sr=16000, amp=0.5):
    t = np.arange(int(round(sr * dur))) / sr
    return amp * np.sin(2 * math.pi * freq * t)


def _write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _write_wav(path, mono, sample_rate=16000):
    mono = np.clip(np.asarray(mono, dtype=float), -1.0, 1.0)
    pcm = np.round(mono * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.astype("<i2").tobytes())
    return str(path)


def _picture_spec():
    return {
        "version": 1,
        "task": "picture_desc_1",
        "concept_aliases": {"cat": ["con m\u00e8o"], "dog": ["ch\u00f3"]},
        "entity_groups": {"animals": ["con m\u00e8o", "ch\u00f3"]},
        "action_groups": {"motion": ["ch\u1ea1y"]},
    }


def _transcript(words):
    return {
        "version": 1,
        "transcript_id": "tr-1",
        "language": "vi",
        "utterances": [
            {
                "speaker": "examiner",
                "start_s": 0.0,
                "end_s": 1.0,
                "tokens": [{"kind": "word", "text": "Xin ch\u00e0o"}],
            },
            {
                "speaker": "participant",
                "start_s": 1.2,
                "end_s": 2.5,
                "tokens": [{"kind": "word", "text": w} for w in words],
            },
        ],
    }


class TestNumericalParity:
    def test_extract_recording_old_and_new_paths_are_identical(self, tmp_path):
        from speech_features import pipeline
        from speech_features.legacy.ad import pipeline as legacy_pipeline

        audio = _write_wav(tmp_path / "a.wav", _tone(145, 2.0), 22050)
        transcript = _write_json(tmp_path / "t.json", _transcript(["con m\u00e8o"]))
        spec = _write_json(tmp_path / "s.json", _picture_spec())

        old = pipeline.extract_recording(audio, transcript, spec, recording_id="r")
        new = legacy_pipeline.extract_recording(audio, transcript, spec, recording_id="r")
        assert old.features == new.features
        assert old.task == new.task == "picture_desc_1"
        assert old.input_hashes == new.input_hashes

    def test_scorer_parity_through_old_and_new_task_paths(self):
        from speech_features import tasks
        from speech_features.legacy.ad import tasks as legacy_tasks

        assert tasks.score_picture is legacy_tasks.score_picture
        assert tasks.score_semantic is legacy_tasks.score_semantic


# ---------------------------------------------------------------------------
# Notebook retirement
# ---------------------------------------------------------------------------
class TestNotebookRetirement:
    def test_no_tracked_notebooks_remain(self):
        result = subprocess.run(["git", "ls-files", "*.ipynb"], capture_output=True, text=True)
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_no_import_ipynb_in_tracked_source(self):
        result = subprocess.run(["git", "ls-files", "*.py"], capture_output=True, text=True)
        assert result.returncode == 0
        tracked = [p for p in result.stdout.split() if p.startswith("src/")]
        assert tracked, "expected tracked source files under src/"
        for rel in tracked:
            text = Path(rel).read_text(encoding="utf-8")
            assert "import_ipynb" not in text, f"{rel} imports a notebook"
