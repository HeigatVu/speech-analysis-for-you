"""PF arm: FullSubNet dispatch through the isolated-environment worker protocol."""

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from say_transcribe.denoise import (
    DEFAULT_FULLSUBNET_WORKER,
    DenoiseError,
    DenoiserSpec,
    denoise_pcm,
)
from say_transcribe.workers import pcm_chunks

FULLSUBNET_RATE = 16000


def _stub_worker(tmp_path: Path, body: str, name: str = "stub_worker.py") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def _spec(tmp_path: Path, worker: Path, config: Path | None = None) -> DenoiserSpec:
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"stub")
    return DenoiserSpec(
        name="fullsubnet",
        python=Path(sys.executable),
        worker=worker,
        checkpoint=checkpoint,
        config=config,
    )


_PASSTHROUGH = """
    import sys
    assert "--rate" in sys.argv and "--checkpoint" in sys.argv
    sys.stdout.buffer.write(sys.stdin.buffer.read())
"""


def test_default_worker_ships_and_never_fetches():
    assert DEFAULT_FULLSUBNET_WORKER.is_file()
    source = DEFAULT_FULLSUBNET_WORKER.read_text(encoding="utf-8")
    for token in ("urllib", "requests", "urlopen", "http:", "socket"):
        assert token not in source, f"worker source contains network primitive: {token}"


def test_worker_module_import_stays_light():
    """The worker module imports numpy and its chunk helper only, nothing heavier."""
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('w', r'{DEFAULT_FULLSUBNET_WORKER}')\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "for name in ('torch', 'audio_zen', 'toml'):\n"
        "    assert name not in sys.modules, f'{name} was eagerly imported'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"Import test failed:\n{result.stderr}"


def test_stub_worker_preserves_duration_and_alignment(tmp_path: Path):
    spec = _spec(tmp_path, _stub_worker(tmp_path, _PASSTHROUGH))
    samples = np.linspace(-0.5, 0.5, FULLSUBNET_RATE).astype(np.float32)

    enhanced = denoise_pcm(samples, FULLSUBNET_RATE, spec)

    assert enhanced.dtype == np.float32
    assert np.array_equal(enhanced, samples)


def test_config_flag_is_passed_only_when_pinned(tmp_path: Path):
    samples = np.zeros(16, dtype=np.float32)
    config = tmp_path / "inference.toml"
    config.write_text("[acoustics]\nsr = 16000\n", encoding="utf-8")
    requires_config = _stub_worker(
        tmp_path,
        """
        import sys
        assert "--config" in sys.argv, "config flag missing"
        assert sys.argv[sys.argv.index("--config") + 1].endswith("inference.toml")
        sys.stdout.buffer.write(sys.stdin.buffer.read())
        """,
        name="requires_config.py",
    )
    forbids_config = _stub_worker(
        tmp_path,
        """
        import sys
        assert "--config" not in sys.argv, "config flag passed without a pinned config"
        sys.stdout.buffer.write(sys.stdin.buffer.read())
        """,
        name="forbids_config.py",
    )

    pinned = _spec(tmp_path, requires_config, config)
    unpinned = _spec(tmp_path, forbids_config, None)

    assert np.array_equal(denoise_pcm(samples, FULLSUBNET_RATE, pinned), samples)
    assert np.array_equal(denoise_pcm(samples, FULLSUBNET_RATE, unpinned), samples)


def test_missing_config_raises_denoiser_unavailable(tmp_path: Path):
    spec = _spec(tmp_path, _stub_worker(tmp_path, _PASSTHROUGH), tmp_path / "absent.toml")

    with pytest.raises(DenoiseError) as excinfo:
        denoise_pcm(np.zeros(16, dtype=np.float32), FULLSUBNET_RATE, spec)

    assert excinfo.value.code == "DENOISER_UNAVAILABLE"
    assert "config" in excinfo.value.message


def test_missing_environment_worker_or_checkpoint_raises_denoiser_unavailable(tmp_path: Path):
    spec = _spec(tmp_path, _stub_worker(tmp_path, _PASSTHROUGH))
    variants = (
        DenoiserSpec(spec.name, tmp_path / "no-python", spec.worker, spec.checkpoint),
        DenoiserSpec(spec.name, spec.python, tmp_path / "no-worker.py", spec.checkpoint),
        DenoiserSpec(spec.name, spec.python, spec.worker, tmp_path / "no-checkpoint"),
    )

    for broken in variants:
        with pytest.raises(DenoiseError) as excinfo:
            denoise_pcm(np.zeros(16, dtype=np.float32), FULLSUBNET_RATE, broken)
        assert excinfo.value.code == "DENOISER_UNAVAILABLE"


def test_worker_failure_and_bad_output_raise_denoiser_unavailable(tmp_path: Path):
    failures = (
        "import sys\nsys.exit(1)\n",
        "import sys\ndata = sys.stdin.buffer.read()\nsys.stdout.buffer.write(data[: len(data) // 2])\n",
        "import sys\nsys.stdout.buffer.write(b'abc')\n",
    )

    for body in failures:
        spec = _spec(tmp_path, _stub_worker(tmp_path, body))
        with pytest.raises(DenoiseError) as excinfo:
            denoise_pcm(np.zeros(16, dtype=np.float32), FULLSUBNET_RATE, spec)
        assert excinfo.value.code == "DENOISER_UNAVAILABLE"


def test_real_worker_rejects_non_native_rate(tmp_path: Path):
    """The shipped worker fails loudly rather than resampling a 48 kHz signal."""
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"stub")
    config = tmp_path / "inference.toml"
    config.write_text("[acoustics]\nsr = 16000\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(DEFAULT_FULLSUBNET_WORKER),
            "--rate",
            "48000",
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(config),
        ],
        input=np.zeros(16, dtype="<f4").tobytes(),
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert b"16000" in result.stderr
    assert result.stdout == b""


def test_real_worker_rejects_a_config_outside_a_checkout(tmp_path: Path):
    """Without the pinned checkout on disk the worker fails, never guesses."""
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"stub")
    config = tmp_path / "inference.toml"
    config.write_text('[inferencer]\ntype = "full_band_crm_mask"\n', encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(DEFAULT_FULLSUBNET_WORKER),
            "--rate",
            str(FULLSUBNET_RATE),
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(config),
        ],
        input=np.zeros(16, dtype="<f4").tobytes(),
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert b"audio_zen" in result.stderr


class TestChunkArithmetic:
    """The one part of the worker that is verifiable without the model."""

    @pytest.mark.parametrize(
        "n_samples,chunk,hop",
        [(900, 400, 200), (1000, 400, 200), (400, 400, 200), (120, 400, 200), (1, 400, 200)],
    )
    def test_bounds_cover_the_signal_with_full_length_windows(self, n_samples, chunk, hop):
        bounds = pcm_chunks.chunk_bounds(n_samples, chunk, hop)

        assert bounds[0][0] == 0
        assert bounds[-1][1] == n_samples
        for start, stop in bounds:
            assert stop > start
            assert stop - start == min(chunk, n_samples)
        covered = np.zeros(n_samples, dtype=bool)
        for start, stop in bounds:
            covered[start:stop] = True
        assert covered.all()

    def test_bounds_reject_an_impossible_hop(self):
        with pytest.raises(ValueError):
            pcm_chunks.chunk_bounds(1000, 400, 500)

    @pytest.mark.parametrize("n_samples", [1, 120, 400, 900, 1000, 4801])
    def test_overlap_add_reconstructs_a_constant_signal(self, n_samples):
        value = 0.25
        rendered = pcm_chunks.overlap_add(
            n_samples, 400, 200, lambda start, stop: np.full(stop - start, value)
        )

        assert rendered.shape == (n_samples,)
        assert rendered.dtype == np.float32
        assert np.allclose(rendered, value, rtol=0, atol=1e-6)

    def test_overlap_add_rejects_wrong_or_non_finite_renders(self):
        with pytest.raises(ValueError):
            pcm_chunks.overlap_add(900, 400, 200, lambda start, stop: np.zeros(stop - start - 1))

        def bad_values(start: int, stop: int) -> np.ndarray:
            piece = np.zeros(stop - start)
            piece[0] = np.nan
            return piece

        with pytest.raises(ValueError):
            pcm_chunks.overlap_add(900, 400, 200, bad_values)
