# GAN Build Spec — `say_transcribe` v5

Authoritative sources (read these, do not re-derive):

- `docs/2026-09-23/vietnamese-chat-transcription/5/SPEC-2026-09-23.md` — contract
- `docs/2026-09-23/vietnamese-chat-transcription/5/PLAN-2026-09-23.md` — order and checkpoints
- `docs/2026-09-23/vietnamese-chat-transcription/5/TASKS-2026-09-23.md` — per-task acceptance
- `docs/2026-09-23/vietnamese-chat-transcription/5/tasks-2026-09-23.json` — dispatch state
- `AGENTS.md` — channel safety, redaction, Vietnamese text integrity, lazy imports

## What to build

One local-only path from a session master WAV to one Chatter-valid `.cha` file shaped exactly like
`/shared-data/dementiabank/Delaware/transcript` files: main tier (`*PAR`/`*INV` with
`•start_ms_end_ms•` bullets) + `%wor` (word-level `start_end` spans) + `%mor` + `%gra`.

Pipeline: PhoWhisper-medium ASR (timed segments + word timestamps) → pyannote diarization draft →
Underthesea 9.5.0 word grouping (`_`-joined, never across utterances) → Stanza `vi` UD-VTB
pretokenized, MWT disabled → render `%wor`/`%mor`/`%gra` → CHAT writer → CLI.

## Non-negotiables

- Never mean-downmix; explicit `--channel N`; in-memory 16 kHz float32 mono view for ASR; no
  resampled audio on disk.
- Logs/errors: stable codes only (`INVALID_AUDIO_CHANNEL`, `GPU_UNAVAILABLE`, `MODEL_UNAVAILABLE`,
  `WORD_GROUPING_UNALIGNED`, `CHAT_VALIDATION_FAILED`, `CHAT_MORPHOSYNTAX_UNAVAILABLE`,
  `CHAT_WORD_TIMING_UNAVAILABLE`). Never transcript text, tokens, absolute paths, raw exceptions.
- Vietnamese text integrity: NFC, tone marks, `d`/`đ` preserved; `_` kept in `%mor` lemmas (do NOT
  copy batchalign `render.py:921` underscore stripping); `,` renders `cm|cm`; `%gra` root-head-0/
  `REL == ROOT` joint invariant; `(n+1)|root|PUNCT` terminator.
- Lazy heavy imports (torch/transformers/pyannote/underthesea/stanza pipeline loads). Unit tests
  fully offline with fakes. No model downloads in CI/tests.
- Tasks T4→T12 in TASKS-2026-09-23.md, one at a time, each on its own branch per task table, each
  with tests + `.venv/bin/ruff check src/say_transcribe tests/say_transcribe` +
  `.venv/bin/pytest tests/say_transcribe -q` green before commit. Never merge to main.

## Test data (private; never copy content into git, logs, or reports)

- Masters: `/shared-data/hcmiu-bhl-corpus/pilot/audio/p00{1..5}/p00N_master.wav` (stereo;
  44.1/48 kHz; pick declared channel explicitly).
- Format reference (read-only, structure only): `/shared-data/dementiabank/Delaware/transcript`.
- Output: private output dir under `/shared-data/hcmiu-bhl-corpus/pilot/` (never tracked).
