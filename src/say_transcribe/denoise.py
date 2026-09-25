"""Denoiser dispatch: subprocess workers in isolated, pinned environments.

DeepFilterNet3 and FullSubNet carry incompatible dependency pins
(torchaudio>=2.11 breaks DeepFilterNet's ``AudioMetaData`` import; FullSubNet is
pinned to Python 3.10/PyTorch 1.12), so neither lives in this package's
environment. Each denoiser runs in its own venv through a standalone worker
script under ``say_transcribe/workers/``, executed by that venv's interpreter.

This module never imports torch/torchaudio/df and never touches the network: it
validates a :class:`DenoiserSpec`, pipes raw little-endian float32 mono PCM to
the worker's stdin, and reads the enhanced signal from stdout.

Worker protocol: ``<python> <worker> --rate R --checkpoint PATH [--config PATH]``;
stdin is raw little-endian float32 mono at R; stdout is the same format and the
same length. Workers own any model-specific resampling (DeepFilterNet3 runs at
48 kHz) and must return exactly the input length; the dispatch enforces that
contract. ``--config`` is passed only when the spec pins one (FullSubNet's recipe
TOML); it is the model configuration file, never audio.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess

import numpy as np

DEFAULT_DF3_WORKER = Path(__file__).resolve().parent / "workers" / "deepfilternet3_worker.py"
DEFAULT_FULLSUBNET_WORKER = Path(__file__).resolve().parent / "workers" / "fullsubnet_worker.py"


class DenoiseError(Exception):
    """A denoiser stage failed under a stable, redacted error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class DenoiserSpec:
    name: str
    python: Path  # interpreter of the isolated environment
    worker: Path  # standalone worker script
    checkpoint: Path  # local checkpoint; never downloaded at run time
    config: Path | None = None  # optional pinned model config file (e.g. recipe TOML)
    sha256: str | None = None  # optional expected SHA-256 for checkpoint verification


def denoise_pcm(
    samples: np.ndarray,
    sample_rate: int,
    spec: DenoiserSpec,
    *,
    timeout: float = 1800.0,
) -> np.ndarray:
    """Run the isolated-environment denoise worker on mono float32 samples."""
    if samples.size == 0 or samples.ndim != 1 or not np.isfinite(samples).all():
        raise DenoiseError("INVALID_AUDIO", "denoiser input must be finite mono samples")
    if sample_rate <= 0:
        raise DenoiseError("INVALID_AUDIO", "denoiser sample rate must be positive")

    # Missing environment/checkpoint fail here, before any process is spawned.
    if not spec.python.is_file():
        raise DenoiseError("DENOISER_UNAVAILABLE", "denoiser environment is missing")
    if not spec.worker.is_file():
        raise DenoiseError("DENOISER_UNAVAILABLE", "denoiser worker is missing")
    if not spec.checkpoint.exists():
        raise DenoiseError("DENOISER_UNAVAILABLE", "denoiser checkpoint is missing")
    if spec.config is not None and not spec.config.is_file():
        raise DenoiseError("DENOISER_UNAVAILABLE", "denoiser config is missing")
    if spec.sha256 is not None:
        if not spec.checkpoint.is_file():
            raise DenoiseError("DENOISER_UNAVAILABLE", "denoiser checkpoint is not a file")
        computed = hashlib.sha256(spec.checkpoint.read_bytes()).hexdigest()
        if computed.lower() != spec.sha256.lower():
            raise DenoiseError("DENOISER_UNAVAILABLE", "denoiser checkpoint hash mismatch")

    command = [
        str(spec.python),
        str(spec.worker),
        "--rate",
        str(sample_rate),
        "--checkpoint",
        str(spec.checkpoint),
    ]
    if spec.config is not None:
        command += ["--config", str(spec.config)]

    try:
        process = subprocess.run(
            command,
            input=samples.astype("<f4").tobytes(),
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise DenoiseError("DENOISER_UNAVAILABLE", "denoiser worker timed out") from None
    except OSError:
        raise DenoiseError("DENOISER_UNAVAILABLE", "denoiser worker could not be executed") from None
    if process.returncode != 0:
        raise DenoiseError("DENOISER_UNAVAILABLE", "denoiser worker failed")

    try:
        enhanced = np.frombuffer(process.stdout, dtype="<f4")
    except ValueError:
        raise DenoiseError("DENOISER_UNAVAILABLE", "denoiser output is not PCM") from None
    if len(enhanced) != len(samples):
        raise DenoiseError("DENOISER_UNAVAILABLE", "denoiser output length does not match input")
    return enhanced.astype(np.float32)
