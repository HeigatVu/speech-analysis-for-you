# SAY documentation

This page is the documentation map. Start with **Current phase**; the detailed
task tracker is the source of truth for progress.

## Current phase

**Phase 0 - Contract, pre-implementation**

We are designing `say_transcribe`, a local-only companion that turns Vietnamese
audio into a draft JSON v2 or CHAT transcript. The draft is human-reviewed and
then consumed by the existing `speech_features` extractor. Version 2 of the
spec adds a TELL-derived audio preprocessing stage group (channel norm,
loudness norm, denoise, VAD) that feeds only the ASR/diarization branch.

v3 research (2026-09-20) found that this routing is likely wrong: channel
normalization and denoising measurably hurt ASR accuracy in the cited
literature, so `channel_norm` as wired in v2 has no consumer that benefits
from it today (acoustic features read the original audio; ASR is hurt by the
preprocessed audio). Only VAD is evidenced as beneficial for ASR. An addendum
to the research (§9) notes a planned second recording device gives
`channel_norm` a possible future consumer (cross-device comparability), so it
is built but shipped off by default rather than deleted.

The v3 **plan** (2026-09-20) resolves the routing question by measurement
instead of guessing: a timeboxed spike (T28) with a human checkpoint decides
whether `channel_norm`/`denoise` ship on or off, before either is enabled.
The plan also folds in the real corpus (5 participants, continuous
recordings, 11 tasks each, 44.1 kHz/12-bit) which doesn't fit
`speech_features`'s one-recording-per-task manifest yet — corpus intake and
an eval harness are now tracked as tasks (T21-T23) ahead of the preprocessing
work.

- v2 spec and research are written. v3 research (with the F3 addendum) is
  written. **v3 PLAN and TASKS are now written**, covering all 33 tasks
  (v1's 21 + v2 preprocessing + v3 reliability work).
- `src/say_transcribe/` does not exist yet — Phase 0 has not started.
- Next: human review of the plan, then T1 (package scaffold) and T21 (corpus
  intake) in parallel.

## Active project records

Read the **v3** plan first, then its research (it corrects a v2 decision),
then the v2 spec; v1 is superseded but kept as a historical decision record
(do not edit it):

1. [v3 Plan and checkpoints](2026-09-20/vietnamese-transcription-pipeline/3/PLAN-2026-09-20.md)
2. [v3 Tasks and acceptance tests](2026-09-20/vietnamese-transcription-pipeline/3/TASKS-2026-09-20.md)
3. [v3 Research (reliability methods, routing correction + device-change addendum)](2026-09-20/vietnamese-transcription-pipeline/3/RESEARCH-2026-09-20.md)
4. [v2 Research (preprocessing delta)](2026-09-20/vietnamese-transcription-pipeline/2/RESEARCH-2026-09-20.md)
5. [v2 Specification](2026-09-20/vietnamese-transcription-pipeline/2/SPEC-2026-09-20.md)

Superseded v1 (Q1-Q10, ASR/CHAT/LLM design - still authoritative for anything
v2 doesn't change):

4. [v1 Research](2026-09-02/vietnamese-transcription-pipeline/1/RESEARCH-2026-09-02.md)
5. [v1 Specification](2026-09-02/vietnamese-transcription-pipeline/1/SPEC-2026-09-02.md)
6. [v1 Plan and checkpoints](2026-09-02/vietnamese-transcription-pipeline/1/PLAN-2026-09-02.md)
7. [v1 Tasks and acceptance tests](2026-09-02/vietnamese-transcription-pipeline/1/TASKS-2026-09-02.md)

The active records use the dated path `docs/<date>/<slug>/<version>/`. Update
this page when the current phase changes; update the task tracker whenever a
task changes status.

## Built foundation

The existing `speech_features` package is the feature-extraction foundation:

- [Feature extraction](built-features/feature-extraction.md)
- [Transcript formats](built-features/transcript-formats.md)
- [Feature catalog](built-features/feature-catalog-v1.md)
- [Neurodegenerative feature guide](built-features/neurodegenerative-feature-guide.md)
- [Migration guide](built-features/migration-0.2.md)
- [Repository analysis](built-features/repo-analysis-2026-09.md)

## Evidence and history

- [Research evidence and inventory](research/)
- [Completed feature-library implementation reports](implementation/)
- [Earlier designs and plans](superpowers/)

The older folders are reference material from the completed feature-library
work. They are not the status tracker for `say_transcribe`.
