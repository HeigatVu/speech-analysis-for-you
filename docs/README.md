# SAY documentation

## Current phase

The Vietnamese transcription pipeline version 4 is planned and ready for human review.
Implementation has not started, and `src/say_transcribe/` does not exist yet.

Version 4 is based on a read-only inspection of the pilot corpus. The data consists of five
continuous masters, not 55 ready clips. Rates and stereo layouts differ, most task/speaker
boundaries require curation, and only one session has a near-complete draft CHAT transcript. The
pipeline therefore starts with approved private annotations and explicit channel selection.

The production baseline is deliberately small: PhoWhisper-medium, model-native segments,
Vietnamese word grouping, optional draft diarization followed by human approval, existing SAY
JSON/CHAT serializers, provenance, and Vietnamese evaluation. A new `tell-inspired-v1` audio
profile is opt-in and evaluated in parallel; native selected-channel audio remains authoritative.
It uses pinned SoX/FFmpeg/DeepFilterNet3/Silero stages and publishes a separate timing feature
family only. Multiple ASR engines, LLM stages, forced alignment, morphosyntax, and automatic
role assignment remain deferred.

## Active records

Read in this order:

1. [v4 research and source-code guide](2026-09-21/vietnamese-transcription-pipeline/4/RESEARCH-2026-09-21.md)
2. [v4 specification](2026-09-21/vietnamese-transcription-pipeline/4/SPEC-2026-09-21.md)
3. [v4 implementation plan](2026-09-21/vietnamese-transcription-pipeline/4/PLAN-2026-09-21.md)
4. [v4 task acceptance criteria](2026-09-21/vietnamese-transcription-pipeline/4/TASKS-2026-09-21.md)
5. [v4 machine-readable tasks](2026-09-21/vietnamese-transcription-pipeline/4/tasks-2026-09-21.json)

T1–T11 are queued. T12 (the real pilot) is blocked until the local NVIDIA driver is repaired and
both `nvidia-smi` and a PyTorch CUDA smoke test succeed.

## Historical records

Versions 1 through 3 are retained as decision history and must not be used as the active contract:

- [v3 research](2026-09-20/vietnamese-transcription-pipeline/3/RESEARCH-2026-09-20.md)
- [v3 plan](2026-09-20/vietnamese-transcription-pipeline/3/PLAN-2026-09-20.md)
- [v3 tasks](2026-09-20/vietnamese-transcription-pipeline/3/TASKS-2026-09-20.md)
- [v2 specification](2026-09-20/vietnamese-transcription-pipeline/2/SPEC-2026-09-20.md)
- [v1 specification](2026-09-02/vietnamese-transcription-pipeline/1/SPEC-2026-09-02.md)

Other feature-library records remain under their dated directories.

## Reference material

- [talkbank-tools + chatter repo map](repo-map/batchalign3/README.md) — directory/file-level
  map of two upstream TalkBank repositories (Batchalign3 ML pipeline and the CHAT format
  authority), for reference when adopting CHAT serialization or pipeline patterns.
- [TalkBank/batchalign repo map](repo-map/batchalign-official/README.md) — a third,
  related repository: the same Batchalign3 product under the TalkBank org and a Bazel
  build, diverged from the map above.
