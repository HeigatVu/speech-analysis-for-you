# Denoise Workers (`src/say_transcribe/workers/`)

Isolated-environment worker scripts for the **opt-in** audio profiles driven by
[`denoise.py`](../denoise.py).

These scripts are **never imported** by `say_transcribe`. They are executed as
subprocesses by a *separate*, purpose-built virtual environment interpreter,
because DeepFilterNet and FullSubNet pin conflicting torch/torchaudio versions
that cannot coexist with the transcription environment.

---

## 1. The Worker Contract

Every worker speaks the same protocol over stdin/stdout:

| Property | Value |
|---|---|
| Invocation | `<env-python> <worker>.py --rate <hz> ...` |
| stdin | Raw little-endian **float32 mono** PCM |
| stdout | Raw little-endian float32 mono PCM, **exactly the same length** |
| Network | Must run with **no network access** |
| Weights | Must resolve to a **local** checkpoint path — bare pretrained names are rejected because that code path may fetch weights |

Equal length in, equal length out is what lets the caller splice an enhanced
signal back in without touching timing, alignment, or the native audio file.

---

## 2. Workers

| Worker | Command | Notes |
|---|---|---|
| [`deepfilternet3_worker.py`](deepfilternet3_worker.py) | `--rate 16000 --checkpoint <local-path>` | DeepFilterNet3 operates at 48 kHz internally and owns its own resampling via pinned torchaudio; `main() -> int` |
| [`fullsubnet_worker.py`](fullsubnet_worker.py) | `--rate 16000 --checkpoint <checkpoint.ckpt> --config <inference.toml>` | FullSubNet is a 16 kHz model, so **any other input rate is rejected rather than resampled** — a silent rate change would feed the model out-of-distribution audio and quietly change the study arm; `main() -> int` |

`--config` is the pinned recipe TOML from the FullSubNet checkout
(`recipes/dns_interspeech_2020/fullsubnet/inference_cumulativeLaplaceNorm.toml`).
The worker adds that recipe's directory to `sys.path` for `model.Model` and the
repository root for `audio_zen`.

---

## 3. Chunking (`pcm_chunks.py`)

[`pcm_chunks.py`](pcm_chunks.py) holds **deterministic overlapping-chunk
arithmetic** for these workers. Full-band denoisers (FullSubNet's LSTM) cannot
process a whole session in one pass, so the worker renders fixed-length chunks
and overlap-adds them.

It is **numpy only**, which is what makes the chunking math verifiable offline —
tests check chunk boundaries, unity-gain reconstruction, and exact output length
without loading a model.

- Every chunk except a signal shorter than one chunk is exactly `chunk_samples`
  long.
- A Hann window is applied per chunk and normalized by the summed weights.
- The first chunk's ramp-up and the last chunk's ramp-down are forced to `1.0`
  so the signal's own onset and offset are preserved.

| Public name | Signature |
|---|---|
| `chunk_bounds` | `chunk_bounds(n_samples: int, chunk_samples: int, hop_samples: int) -> list[tuple[int, int]]` |
| `overlap_add` | `overlap_add(...)` — weighted reconstruction of the enhanced chunks |

---

## 4. Related

- [`denoise.py`](../denoise.py) — the orchestrator that spawns these workers
- [`pcm_chunks.py`](pcm_chunks.py) — the chunk arithmetic tested offline
- [Pipeline overview](../README.md) — where the optional profiles sit in the run
