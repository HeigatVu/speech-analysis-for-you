---
slug: audio-preprocessing-experiment
date: 2026-09-25
version: 2
status: provisional-dry-run
---

# RESULTS (provisional) — real-data dry run, T1–T7 code

**Provisional means provisional.** Every score below is computed against the **draft** references
(`pilot/transcript/p00N.cha`), whose speaker labels are auto-diarized and whose utterance timing is a
fixed ~20 s window grid, not utterance boundaries. Task T0 (curate and verify references) is still
open, so nothing here is a study conclusion. What this run does establish is that the pipeline built
in T1–T7 runs end to end on real masters and real references without a silent failure.

## What ran

| Run | Input | Outcome |
| --- | --- | --- |
| Stage dry run | p001–p003 masters, full sessions | hash gate, channel selection, P0 profile, VAD, coverage, acoustic pack and eGeMAPS all produced output |
| End-to-end CLI | 30 s excerpt of p001 (native 44.1 kHz/32-bit/stereo) + slice reference | `say-transcribe preprocess-study` exit 0; N0/N1/P0 arms scored; session record and summary written |

Environment: CPU only (no CUDA device), `ffmpeg 6.1.1` with `libgsm`+`loudnorm`, PhoWhisper-medium
pinned at revision `55a7e3eb6c906de891f8f06a107754427dd3be79`, `opensmile 2.6.0` installed only for
this run (the declared `standardized-acoustic` extra) and removed afterwards. Denoiser arms (PF/PD)
could not run: T0's pinned checkpoints and isolated environments do not exist yet.

Private artifacts (not in Git): `pilot/study-v2/{manifest-stage.json,stage-report.json,slice30/,
out-slice30/}`.

## Findings

1. **Reference provenance checks out; reference timing does not.** All three draft `.cha` files carry
   an `@Comment: source_sha256` that matches the SHA-256 of the master they were built from, so the
   drafts describe these recordings and no other. Their timing is still unusable: VAD retained
   39–53 % of the participant speech the drafts claim, and p003's draft has 17 of 42 participant
   utterances with zero duration (p001: 3 of 53). The ≥95 % retention gate fails on references like
   these, so T0's curation — not the VAD — is the blocker.

2. **Loudness normalization fell back to dynamic mode on 3 of 3 development sessions**, despite
   `linear=true`. The target was still reached (measured output I ≈ −23.2 LUFS, TP ≈ −1.0 dBTP), but
   with time-varying gain rather than the intended single linear gain, which is exactly the
   confound the SPEC's LRA=50 choice was meant to avoid. The run flags it per session and in a
   cohort-level summary note, so it can never pass as silent. Freeze a decision before T8: raise the
   target for these very quiet recordings (input I ≈ −33 to −55 LUFS), apply a fixed pre-gain, or
   accept dynamic mode and exclude loudness-sensitive feature families from the comparison.

3. **The SPEC's per-vector feature-validity gate is unreachable on real data.** Per session the
   acoustic pack produced 161–162 finite values of 167 (96.4–97.0 %), because optional keys stay NaN
   without the annotations they need; eGeMAPS produced 100 % finite values. Under the SPEC sentence
   "a feature vector is valid when every value is finite", no acoustic vector is ever valid, so the
   ≥95 % gate cannot pass for any arm. The code now reports both numbers — per-vector validity (the
   gate) and value-level coverage — and this needs a SPEC decision: scope validity to keys whose
   prerequisites are present, or gate on value coverage.

4. **At 44.1 kHz the arms are one sample apart.** For p001 the native 16 kHz ASR view holds
   16,648,897 samples while the profile output holds 16,648,896 (0.06 ms). The 48 kHz sessions agree
   exactly. The mismatch is bounded and far below the metric resolution, but the two paths do not
   guarantee sample-identical input across all native rates; worth freezing in the SPEC if a future
   analysis needs sample-exact cross-arm alignment.

5. **Pipeline behaviour on real data.** Hash verification ran before any decode on all sessions; the
   re-verify-after-arms guard and the reference-bounds check were both exercised; per-arm runtime is
   now charged for the stages that arm needs (on 30 s of audio, CPU: N0 35 s, N1 163 s, P0 140 s);
   the redacted summary contains no session id, participant id, path, or transcript text (checked by
   pattern over the serialized file).

6. **Provisional scores are dominated by the draft reference, as expected.** On the 30 s slice the
   no-VAD baseline scored SyER 0.35 while the VAD arms scored 0.96–0.98: the draft's single 20 s
   window is mostly interviewer speech, so VAD correctly kept only 5–9 % of it and the scorer
   charged the rest as deletions. This is a statement about the draft timing, not about VAD.

## Gaps carried into T8

- T0: verified PAR/INV timing for all five references; pinned FullSubNet and DeepFilterNet3
  checkpoints with isolated environments; openSMILE licence sign-off recorded.
- The three SPEC decisions above (loudness mode, feature-validity unit, cross-rate alignment).
- `tests/speech_features/test_pipeline.py::TestLabelFreeExtraction::test_standardized_acoustic_uses_audio_and_records_adapter_provenance`
  asserts openSMILE is absent, so installing the study's own declared extra turns it red; it needs an
  environment-aware assertion before the formal run.
