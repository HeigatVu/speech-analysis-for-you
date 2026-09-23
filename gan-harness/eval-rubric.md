# Evaluation Rubric — code-only mode

Each criterion 1–10, weighted total out of 10. Pass threshold 7.0. The evaluator is ruthlessly
strict: only the happy path passing is a 5, never a pass.

## Correctness (weight 0.35)

- Tier rules exact: `%wor` word spans (final punct bare), `%mor` `pos|lemma[-Feat]` with `_` kept,
  `,`→`cm|cm`, `%gra` `i|head|REL` with root-head-0 joint invariant and `(n+1)|root|PUNCT`.
- Failure paths per utterance only (`CHAT_MORPHOSYNTAX_UNAVAILABLE`, `CHAT_WORD_TIMING_UNAVAILABLE`),
  main tier line + bullet always kept.
- No cross-utterance word grouping; NFC/tones/`d`/`đ` round-trip.

## Format fidelity (weight 0.25)

- `.cha` round-trips through `src/speech_features/formats/chat.py` preserving `%wor`/`%mor`/`%gra`.
- Header block matches SPEC §3 (one `source_sha256` and one draft-speaker `@Comment`, exactly once).
- Structural parity with `/shared-data/dementiabank/Delaware/transcript` layout (read-only diff of
  shape, never content).

## Safety & hygiene (weight 0.2)

- Lazy heavy imports proven by test (import `say_transcribe` without torch/transformers).
- No transcript text, tokens, absolute paths, or raw exceptions in any log/error/report.
- Channel selection enforced; no downmix; no resampled audio on disk.

## Tests & reliability (weight 0.2)

- Offline fake-backed tests for every task; `GPU_UNAVAILABLE`/`MODEL_UNAVAILABLE` distinct paths.
- `WORD_GROUPING_UNALIGNED`, `CHAT_MORPHOSYNTAX_UNAVAILABLE`, `CHAT_WORD_TIMING_UNAVAILABLE` tested.
- `.venv/bin/ruff check src/say_transcribe tests/say_transcribe` and
  `.venv/bin/pytest tests/say_transcribe -q` green.
