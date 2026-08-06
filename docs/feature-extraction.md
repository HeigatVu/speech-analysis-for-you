# Feature Extraction

This guide documents the 0.2 **label-free** extraction pipeline of SAY:
audio and document inputs, target-speaker isolation, the `FeatureBundle`
output tables, issues and provenance, pack/level selection, manifest v2,
batch behavior, CLI exit codes, and the stable error codes. It replaces the
0.1 AD-first workflow, which now lives only in the deprecated compatibility
section at the bottom of this page and in
[migration-0.2.md](migration-0.2.md). See the [README](../README.md) for
installation and the quick start.

> **Research-only.** SAY is a **research/descriptive** library. It is **not a
> diagnostic**; it gives no diagnosis, defines no screening cutoff or
> normative range, makes no treatment recommendation, and offers no fixed
> clinical performance. No trained model, ASR, diarization, forced alignment,
> or automatic morphosyntactic analysis is performed. Automated tools may
> prepare annotations externally, but a human reviews them before SAY
> consumes them. See [transcript-formats.md](transcript-formats.md).

## 1. Inputs

### Audio: standard PCM WAV

- `read_wav(path, *, sample_rate=16000)` decodes standard PCM WAV with the
  stdlib `wave` module: mono or stereo integer PCM (8/16/24/32-bit),
  non-empty, with a valid header.
- Stereo is down-mixed to mono by the channel mean; samples map to a finite
  float domain in `[-1, 1]` and are **resampled** with
  `scipy.signal.resample_poly` to the configured `ExtractionConfig.sample_rate`
  (default 16 kHz). Both the decoded and the resampled signals are checked
  for finiteness.
- Broken headers, non-PCM compression, unsupported widths/channels, and
  malformed or truncated payloads raise `InvalidAudioError`
  (`INVALID_AUDIO`); unknown formats raise `UnsupportedAudioError`
  (`UNSUPPORTED_AUDIO`).

### Document: reviewed transcript

`load_document(path)` accepts JSON v2 (native), JSON v1 (deterministically
migrated), or the CLAN-compatible CHAT subset. A human reviews the
transcript and its annotations before extraction. Full details, including
the exact JSON v2 shape and the CHAT subset, are in
[transcript-formats.md](transcript-formats.md).

## 2. Target-speaker isolation

- With **several documented speakers and no target**, extraction raises
  `TARGET_SPEAKER_REQUIRED`; pass `target_speaker` to isolate one speaker.
- With **exactly one documented speaker**, that speaker is the implicit
  target; an explicit target must be a documented speaker (`INVALID_CONFIG`
  otherwise).
- Only the target speaker's aligned utterance intervals are analyzed: frames
  are clipped to those intervals, so examiner audio and the recording's
  global loudness never contaminate the target's acoustic statistics.
- **`allow_unaligned` is available only on the direct acoustic pack
  functions** (`extract_acoustic_features` / `extract_acoustic_bundle`),
  **not** on the default high-level `extract`. It requests the
  whole-recording fallback when the target has no aligned intervals and
  emits an `UNALIGNED_SPEAKER` warning issue; without it, the default path
  fails with `MISSING_ANNOTATION`.

## 3. `FeatureBundle` output tables

`extract(audio_path, document, *, target_speaker=None,
packs=("acoustic", "adult_neuro"), levels=("recording", "utterance"),
config=None)` returns a frozen `FeatureBundle`:

- `recordings` — columns start with the identifier prefix
  `recording_id`, `speaker_id`, then one column per selected recording-level
  feature key, sorted lexicographically.
- `utterances` — the same identifiers plus the prefix
  `recording_id`, `speaker_id`, `utterance_id`, `start_s`, `end_s`, then the
  selected utterance-level feature keys.
- `issues` — exactly `recording_id`, `speaker_id`, `utterance_id`, `feature`,
  `code`, `severity`, `message`; severity is `warning` or `error`.
- `provenance` — JSON-serializable mapping with package/catalog versions,
  packs, levels, configuration, target speakers, input SHA-256 hashes
  (audio and transcript), the transcript hash kind (`source` for CHAT,
  `canonical_document` otherwise), and annotation sources (layer, source,
  confidence).

Missing values are `NaN` and each is paired with a structured `FeatureIssue`
(naming the exact feature key) — never a fabricated zero.

## 4. Pack and level selection

- `packs` selects built-in packs by name: `"acoustic"` (73 recording-level
  keys) and/or `"adult_neuro"` (93 recording-level and 4 utterance-level
  keys). The tuple must be non-empty and duplicate-free; an unknown pack
  raises `UNKNOWN_PACK`.
- `levels` selects `"recording"` and/or `"utterance"` output tables.
- Selections are canonicalized to static catalog order; when the
  `adult_neuro` pack is selected without `acoustic`, no audio is decoded.
- The full catalog is in [feature-catalog-v1.md](feature-catalog-v1.md);
  `list_features(pack=..., level=...)` queries it.

## 5. Manifest v2 and batch extraction

`extract_batch(manifest_path, *, packs=("acoustic", "adult_neuro"),
config=None)` runs manifest v2. The manifest is JSON with `version: 2` and a
non-empty `rows` list; every row has **exactly**:

- `recording_id` — non-empty string, unique across rows;
- `audio_path` — non-empty string, resolved relative to the manifest file;
- `transcript_path` — non-empty string, resolved relative to the manifest
  file;
- `target_speakers` — optional non-empty list of unique non-empty speaker
  ids; when omitted, the single documented speaker is used and multiple
  speakers without a declared target fail the row.

Any extra row key, unknown version, duplicate `recording_id`, or malformed
`target_speakers` raises `InvalidManifestError` before any extraction
begins. No diagnosis, labels, tasks, demographics, or clinical values are
accepted or emitted.

**Batch isolation.** Failures inside a structurally valid row are isolated:
a failure before a target is known emits one feature-less severity-`error`
issue with the recording id and an empty speaker id; a target-specific
failure emits one `error` issue with that target id. Remaining rows still
complete, and the batch always returns `recordings`/`utterances`/`issues`
plus a `provenance` mapping with per-recording entries (hashes, annotation
sources, error code/message when failed) and `counts` (`total`, `success`,
`failure`). Audio is hashed and decoded at most once per row, shared by all
declared targets.

## 6. CLI

`say-features` has four stdlib-`argparse` commands:

```bash
say-features validate INPUT                       # validate a document or manifest v2
say-features convert INPUT OUTPUT [--force]       # convert between JSON v2 and CHAT
say-features extract MANIFEST OUTPUT_DIR [--pack PACK ...] [--force]
say-features list-features [--pack PACK] [--level LEVEL]
```

- `extract` writes `recordings.csv`, `utterances.csv`, `issues.csv`, and
  `provenance.json`; pre-existing outputs are refused without `--force`.
- **Exit codes:** `0` for success or warnings-only output; `1` when any
  issue has severity `error` (partial row failures); `2` for invalid global
  input/configuration, argparse syntax errors, or missing input files, with
  one concise stderr message and no traceback.

## 7. Stable error codes

All ten stable codes are re-exported as `speech_features.STABLE_ERROR_CODES`
and attached to the matching exception classes:

| Code | Raised when |
|---|---|
| `INVALID_DOCUMENT` | a speech document JSON is malformed or invalid |
| `INVALID_CHAT` | a CHAT file violates the documented subset |
| `INVALID_AUDIO` | PCM WAV decoding fails (header/payload) |
| `UNSUPPORTED_AUDIO` | the media format or encoding is unsupported |
| `MISSING_INPUT` | a required input file is absent |
| `TARGET_SPEAKER_REQUIRED` | several speakers exist but no target is given |
| `MISSING_ANNOTATION` | a required annotation layer or alignment is absent |
| `UNKNOWN_PACK` | a pack selection names an unknown pack |
| `INVALID_CONFIG` | configuration or input types are invalid |
| `EXTRACTION_ERROR` | an unexpected single-feature failure |

Feature-specific warning codes (for example `UNSUPPORTED_CHAT_TIER`,
`UNALIGNED_SPEAKER`, `NO_SPEECH`, `INSUFFICIENT_VOICED_FRAMES`,
`INSUFFICIENT_TOKENS`) appear in the `issues` table and in the
[feature catalog](feature-catalog-v1.md).

## 8. Deprecated: legacy AD evaluation (0.1 compatibility)

The 0.1 AD-vs-healthy-control evaluation workflow (`evaluate_ad_baseline`,
task scorers, label-bearing manifest v1, the six-task cohort pipeline) is
**deprecated** and remains available only behind
`speech_features.legacy.ad` through `0.2.x`. It is not part of the 0.2
feature workflow, is not supported by the catalog above, and is eligible for
removal in `0.3.0`. Do not start new work on it — see
[migration-0.2.md](migration-0.2.md) for the full migration guide.
