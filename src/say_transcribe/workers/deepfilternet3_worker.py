"""DeepFilterNet3 denoise worker for the isolated DeepFilterNet environment.

Executed by ``say_transcribe.denoise`` through the isolated venv's interpreter —
never imported by ``say_transcribe`` itself:

    <env-python> deepfilternet3_worker.py --rate 16000 --checkpoint <local-path>

stdin: raw little-endian float32 mono PCM at ``--rate``. stdout: same format and
exactly the same length, enhanced. DeepFilterNet3 operates at 48 kHz internally;
this worker owns the round-trip resampling via its own pinned torchaudio.

The checkpoint must already exist locally (DeepFilterNet3 checkpoint directory);
bare pretrained names are rejected because that path may fetch weights. This
worker never fetches weights and must run with no network access. API shape
targets ``df.enhance`` as pinned in the isolated environment; end-to-end
behavior is verified at the CP1 private run.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torchaudio

from df.enhance import enhance, init_df

# One STFT hop of tolerance for enhance() padding; anything larger is a
# time-scaling bug and must fail, never be papered over with silence.
_MAX_DRIFT_SAMPLES = 1024


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepFilterNet3 isolated-environment worker")
    parser.add_argument("--rate", type=int, required=True, help="PCM sample rate on stdin/stdout")
    parser.add_argument("--checkpoint", required=True, help="Local DeepFilterNet3 checkpoint directory")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        # A bare pretrained name would resolve inside df and may fetch weights.
        sys.exit(2)

    raw = sys.stdin.buffer.read()
    samples = np.frombuffer(raw, dtype="<f4").copy()
    audio = torch.from_numpy(samples).float().unsqueeze(0)

    # df's init_df registers a loguru sink on stdout and logs to enhance.log in
    # the checkpoint directory: both are disabled, and any stray print from the
    # model stack is diverted to stderr so stdout stays pure PCM.
    saved_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        result = init_df(str(checkpoint), log_level="none", log_file=None)
        model, state = result[0], result[1]
        if isinstance(state, dict):
            model_sr = int(state.get("sr", 48000))
        else:
            model_sr = int(getattr(state, "sr", 48000))
        if args.rate != model_sr:
            audio = torchaudio.functional.resample(audio, args.rate, model_sr)
        with torch.no_grad():
            enhanced = enhance(model, state, audio)
        if args.rate != model_sr:
            enhanced = torchaudio.functional.resample(enhanced, model_sr, args.rate)
    finally:
        sys.stdout = saved_stdout

    out = enhanced.squeeze(0).cpu().numpy().astype("<f4")
    drift = abs(len(out) - len(samples))
    if drift > _MAX_DRIFT_SAMPLES:
        sys.exit(2)
    if len(out) < len(samples):
        out = np.pad(out, (0, len(samples) - len(out)))
    else:
        out = out[: len(samples)]
    sys.stdout.buffer.write(out.tobytes())
    return 0


if __name__ == "__main__":
    sys.exit(main())
