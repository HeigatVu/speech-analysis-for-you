"""FullSubNet denoise worker for the isolated FullSubNet environment.

Executed by ``say_transcribe.denoise`` through the isolated venv's interpreter —
never imported by ``say_transcribe`` itself:

    <env-python> fullsubnet_worker.py --rate 16000 \
        --checkpoint <checkpoint.ckpt> --config <inference.toml>

stdin: raw little-endian float32 mono PCM at ``--rate``. stdout: same format and
exactly the same length, enhanced. FullSubNet is a 16 kHz model, so any other
input rate is rejected instead of being resampled: a silent rate change would
feed the model out-of-distribution audio and quietly change the study arm.

``--config`` is the pinned recipe TOML from the FullSubNet checkout
(``recipes/dns_interspeech_2020/fullsubnet/inference_cumulativeLaplaceNorm.toml``
by default there); its directory is added to ``sys.path`` for the recipe's
``model.Model`` and its repository root for ``audio_zen``, exactly as the
project's own ``inference.py`` driver does. Only the ``full_band_crm_mask``
inference type that FullSubNet was trained with is supported.

The checkpoint must already exist locally; this worker never fetches weights and
must run with no network access. Session-length audio is rendered in fixed
10-second, 50%-overlapping chunks (``pcm_chunks``) because FullSubNet's recurrent
model cannot run over tens of thousands of frames in one pass; the overlap-add is
unity-gain, so chunking does not change the level of the retained speech.

API shape targets the pinned checkout and its recipe TOML. The isolated
environment and the checkpoint are not pinned yet (task T0), so nothing here has
run against real FullSubNet weights: the API calls, the checkpoint state-dict
handling, and the chunked inference above are what CP1 must exercise for the PF
arm before any study conclusion uses it.
"""

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pcm_chunks import overlap_add  # noqa: E402  (sibling module in the worker directory)

_CHUNK_SECONDS = 10.0
_SUPPORTED_RATE = 16000
_SUPPORTED_INFERENCE_TYPE = "full_band_crm_mask"


def _fail(message: str) -> None:
    sys.stderr.write(f"fullsubnet_worker: {message}\n")
    raise SystemExit(2)


def _require_local(path: Path, label: str) -> Path:
    if not path.exists():
        # A bare pretrained name would resolve elsewhere and may fetch weights.
        _fail(f"{label} is missing")
    return path


def _find_repo_root(config_dir: Path) -> Path:
    """The nearest ancestor holding the FullSubNet ``audio_zen`` package."""
    for candidate in (config_dir, *config_dir.parents):
        if (candidate / "audio_zen" / "__init__.py").is_file():
            return candidate
    _fail("pinned config is not inside a FullSubNet checkout (no audio_zen package found)")
    raise AssertionError("unreachable")


def _load_model(config: dict, checkpoint: Path):
    import torch

    from audio_zen.utils import initialize_module

    model_config = config["model"]
    model = initialize_module(model_config["path"], args=model_config["args"], initialize=True)
    state = torch.load(checkpoint, map_location="cpu")
    if not isinstance(state, dict) or "model" not in state:
        _fail("checkpoint has no 'model' state dict")
    weights = {key.replace("module.", ""): value for key, value in state["model"].items()}
    model.load_state_dict(weights)
    model.eval()
    return model


def _make_renderer(samples: np.ndarray, config: dict, model):
    """A per-chunk renderer running the recipe's full-band cIRM mask inference."""
    import torch

    from audio_zen.acoustics.feature import istft, stft
    from audio_zen.acoustics.mask import decompress_cIRM

    acoustics = config["acoustics"]
    n_fft = int(acoustics["n_fft"])
    hop_length = int(acoustics["hop_length"])
    win_length = int(acoustics["win_length"])

    def render(start: int, stop: int) -> np.ndarray:
        chunk = torch.from_numpy(samples[start:stop].copy()).float().unsqueeze(0)
        with torch.no_grad():
            noisy_mag, _, noisy_real, noisy_imag = stft(chunk, n_fft, hop_length, win_length)
            predicted = model(noisy_mag.unsqueeze(1)).permute(0, 2, 3, 1)
            crm = decompress_cIRM(predicted)
            enhanced_real = crm[..., 0] * noisy_real - crm[..., 1] * noisy_imag
            enhanced_imag = crm[..., 1] * noisy_real + crm[..., 0] * noisy_imag
            enhanced = istft(
                (enhanced_real, enhanced_imag),
                n_fft,
                hop_length,
                win_length,
                length=chunk.size(-1),
                input_type="real_imag",
            )
        # No level renormalization here: the repository's file-writing inference
        # rescales to 0.8 full scale, which would defeat the loudness stage.
        return enhanced.detach().squeeze(0).cpu().numpy()

    return render


def main() -> int:
    parser = argparse.ArgumentParser(description="FullSubNet isolated-environment worker")
    parser.add_argument("--rate", type=int, required=True, help="PCM sample rate on stdin/stdout")
    parser.add_argument("--checkpoint", required=True, help="Local FullSubNet checkpoint file")
    parser.add_argument("--config", required=True, help="Pinned FullSubNet inference recipe TOML")
    args = parser.parse_args()

    if args.rate != _SUPPORTED_RATE:
        _fail(f"FullSubNet runs at {_SUPPORTED_RATE} Hz; got {args.rate}")

    checkpoint = _require_local(Path(args.checkpoint), "checkpoint")
    config_path = _require_local(Path(args.config), "config")
    config_dir = config_path.resolve().parent
    # Same two entries the project's own inference.py driver adds: the recipe
    # directory for "model.Model" and the checkout root for "audio_zen".
    repo_root = _find_repo_root(config_dir)
    sys.path.insert(0, str(config_dir))
    sys.path.insert(0, str(repo_root))

    import toml

    config = toml.load(config_path)
    inference_type = config.get("inferencer", {}).get("type")
    if inference_type != _SUPPORTED_INFERENCE_TYPE:
        _fail("pinned config is not a full_band_crm_mask recipe")

    raw = sys.stdin.buffer.read()
    samples = np.frombuffer(raw, dtype="<f4").copy()
    if samples.size == 0 or not np.isfinite(samples).all():
        _fail("input PCM is empty or non-finite")

    # The model stack prints to stdout; keep stdout pure PCM and send any stray
    # output to stderr, as the DeepFilterNet worker does for the same reason.
    saved_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        model = _load_model(config, checkpoint)
        render = _make_renderer(samples, config, model)
        enhanced = overlap_add(
            samples.size,
            int(_CHUNK_SECONDS * _SUPPORTED_RATE),
            int(_CHUNK_SECONDS * _SUPPORTED_RATE) // 2,
            render,
        )
    except Exception as error:  # noqa: BLE001 - isolated worker: report and exit non-zero
        sys.stderr.write(f"fullsubnet_worker: {type(error).__name__}\n")
        return 2
    finally:
        sys.stdout = saved_stdout

    sys.stdout.buffer.write(enhanced.astype("<f4").tobytes())
    return 0


if __name__ == "__main__":
    sys.exit(main())
