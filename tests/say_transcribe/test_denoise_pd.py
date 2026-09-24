import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from say_transcribe.denoise import DEFAULT_DF3_WORKER, DenoiseError, DenoiserSpec, denoise_pcm


def _stub_worker(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "stub_worker.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def _spec(tmp_path: Path, worker: Path) -> DenoiserSpec:
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"stub")
    return DenoiserSpec(
        name="deepfilternet3",
        python=Path(sys.executable),
        worker=worker,
        checkpoint=checkpoint,
    )


_PASSTHROUGH = """
    import sys
    assert "--rate" in sys.argv and "--checkpoint" in sys.argv
    sys.stdout.buffer.write(sys.stdin.buffer.read())
"""


def test_lazy_import_does_not_load_model_stack():
    code = (
        "import sys, say_transcribe.denoise\n"
        "for name in ('df', 'deepfilternet', 'torchaudio', 'torch'):\n"
        "    assert name not in sys.modules, f'{name} was eagerly imported'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"Import test failed:\n{result.stderr}"


def test_default_df3_worker_script_ships_and_never_fetches():
    assert DEFAULT_DF3_WORKER.is_file()
    source = DEFAULT_DF3_WORKER.read_text(encoding="utf-8")
    for token in ("urllib", "requests", "urlopen", "http:", "socket"):
        assert token not in source, f"worker source contains network primitive: {token}"


def test_stub_worker_preserves_duration_and_alignment(tmp_path: Path):
    spec = _spec(tmp_path, _stub_worker(tmp_path, _PASSTHROUGH))
    samples = np.linspace(-0.5, 0.5, 1600).astype(np.float32)

    enhanced = denoise_pcm(samples, 16000, spec)

    assert enhanced.dtype == np.float32
    assert np.array_equal(enhanced, samples)


def test_stub_worker_scaling_keeps_sample_alignment(tmp_path: Path):
    spec = _spec(
        tmp_path,
        _stub_worker(
            tmp_path,
            """
            import sys
            import numpy as np
            samples = np.frombuffer(sys.stdin.buffer.read(), dtype="<f4")
            sys.stdout.buffer.write((samples * 0.5).astype("<f4").tobytes())
            """,
        ),
    )
    samples = np.linspace(-0.5, 0.5, 1600).astype(np.float32)

    enhanced = denoise_pcm(samples, 16000, spec)

    assert np.array_equal(enhanced, samples * 0.5)


def test_missing_environment_raises_denoiser_unavailable(tmp_path: Path):
    spec = _spec(tmp_path, _stub_worker(tmp_path, _PASSTHROUGH))
    broken = DenoiserSpec(
        name=spec.name,
        python=tmp_path / "no-such-python",
        worker=spec.worker,
        checkpoint=spec.checkpoint,
    )

    with pytest.raises(DenoiseError) as excinfo:
        denoise_pcm(np.zeros(16, dtype=np.float32), 16000, broken)

    assert excinfo.value.code == "DENOISER_UNAVAILABLE"
    assert excinfo.value.__cause__ is None


def test_missing_worker_and_checkpoint_raise_denoiser_unavailable(tmp_path: Path):
    spec = _spec(tmp_path, _stub_worker(tmp_path, _PASSTHROUGH))
    variants = (
        DenoiserSpec(spec.name, spec.python, tmp_path / "no-worker.py", spec.checkpoint),
        DenoiserSpec(spec.name, spec.python, spec.worker, tmp_path / "no-checkpoint"),
    )

    for broken in variants:
        with pytest.raises(DenoiseError) as excinfo:
            denoise_pcm(np.zeros(16, dtype=np.float32), 16000, broken)
        assert excinfo.value.code == "DENOISER_UNAVAILABLE"


def test_worker_failure_raises_denoiser_unavailable(tmp_path: Path):
    spec = _spec(
        tmp_path,
        _stub_worker(
            tmp_path,
            """
            import sys
            sys.exit(1)
            """,
        ),
    )

    with pytest.raises(DenoiseError) as excinfo:
        denoise_pcm(np.zeros(16, dtype=np.float32), 16000, spec)

    assert excinfo.value.code == "DENOISER_UNAVAILABLE"


def test_worker_length_mismatch_raises_denoiser_unavailable(tmp_path: Path):
    spec = _spec(
        tmp_path,
        _stub_worker(
            tmp_path,
            """
            import sys
            data = sys.stdin.buffer.read()
            sys.stdout.buffer.write(data[: len(data) // 2])
            """,
        ),
    )

    with pytest.raises(DenoiseError) as excinfo:
        denoise_pcm(np.zeros(16, dtype=np.float32), 16000, spec)

    assert excinfo.value.code == "DENOISER_UNAVAILABLE"
    assert "length" in excinfo.value.message


def test_worker_timeout_raises_denoiser_unavailable(tmp_path: Path):
    spec = _spec(
        tmp_path,
        _stub_worker(
            tmp_path,
            """
            import time
            time.sleep(5)
            """,
        ),
    )

    with pytest.raises(DenoiseError) as excinfo:
        denoise_pcm(np.zeros(16, dtype=np.float32), 16000, spec, timeout=0.3)

    assert excinfo.value.code == "DENOISER_UNAVAILABLE"
    assert "timed out" in excinfo.value.message


def test_dispatch_never_touches_the_network(monkeypatch, tmp_path: Path):
    def no_network(*args, **kwargs):
        raise AssertionError("denoiser dispatch attempted network access")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    spec = _spec(tmp_path, _stub_worker(tmp_path, _PASSTHROUGH))

    enhanced = denoise_pcm(np.zeros(16, dtype=np.float32), 16000, spec)

    assert len(enhanced) == 16


def test_non_finite_input_raises_invalid_audio(tmp_path: Path):
    spec = _spec(tmp_path, _stub_worker(tmp_path, _PASSTHROUGH))
    samples = np.zeros(16, dtype=np.float32)
    samples[0] = np.nan

    with pytest.raises(DenoiseError) as excinfo:
        denoise_pcm(samples, 16000, spec)

    assert excinfo.value.code == "INVALID_AUDIO"


def test_empty_input_and_bad_rate_raise_invalid_audio(tmp_path: Path):
    spec = _spec(tmp_path, _stub_worker(tmp_path, _PASSTHROUGH))

    with pytest.raises(DenoiseError) as empty_error:
        denoise_pcm(np.zeros(0, dtype=np.float32), 16000, spec)
    assert empty_error.value.code == "INVALID_AUDIO"

    with pytest.raises(DenoiseError) as rate_error:
        denoise_pcm(np.zeros(16, dtype=np.float32), 0, spec)
    assert rate_error.value.code == "INVALID_AUDIO"


def test_worker_non_pcm_output_raises_denoiser_unavailable(tmp_path: Path):
    spec = _spec(
        tmp_path,
        _stub_worker(
            tmp_path,
            """
            import sys
            sys.stdout.buffer.write(b"abc")
            """,
        ),
    )

    with pytest.raises(DenoiseError) as excinfo:
        denoise_pcm(np.zeros(16, dtype=np.float32), 16000, spec)

    assert excinfo.value.code == "DENOISER_UNAVAILABLE"
    assert "PCM" in excinfo.value.message
