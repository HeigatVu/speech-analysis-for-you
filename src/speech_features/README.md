# Label-Free Feature Extraction (`src/speech_features/`)

The **`speech_features`** package is the core SAY library (version `0.2.0`): a
math-first, population-neutral, **label-free** feature extractor for reviewed
Vietnamese speech data.

It reads a reviewed transcript plus one master PCM WAV and returns a
table-oriented `FeatureBundle`. It performs **no ASR, diarization, forced
alignment, or automatic morphosyntactic analysis** — those live in the separate
[`say_transcribe`](../say_transcribe/README.md) draft pipeline, and a human
reviews their output before this package consumes it.

---

## 1. Source Code Modules

| Module | Description | Key Public Names |
|---|---|---|
| [`__init__.py`](__init__.py) | Public API surface; eagerly imports the four built-in packs so `import speech_features` exposes the full catalog | `extract`, `extract_batch`, `load_document`, `save_document`, `list_features`, `PACKS` |
| [`document.py`](document.py) | Population-neutral `SpeechDocument` model (JSON v2 is the lossless native form) | `SpeechDocument`, `DocumentSpeaker`, `DocumentUtterance`, `DocumentToken`, `MediaRef`, `load_document`, `save_document` |
| [`extraction.py`](extraction.py) | Label-free extraction pipeline, row/target isolation, and provenance assembly | `extract`, `extract_batch` |
| [`result.py`](result.py) | Frozen result contracts and stable error/issue codes | `FeatureBundle`, `FeatureIssue`, `ExtractionContext`, `STABLE_ERROR_CODES` |
| [`catalog.py`](catalog.py) | Static feature registry, pack metadata, and deterministic catalog queries | `FeatureDefinition`, `FeaturePack`, `PACKS`, `list_features`, `CATALOG_VERSION` |
| [`audio.py`](audio.py) | Shared standard-PCM WAV decoding; stdlib `wave` only, mono/stereo to finite mono float, target-rate resampling | `read_wav`, `InvalidAudioError` |
| [`acoustic.py`](acoustic.py) | Math-first acoustic primitives over mono float arrays (NumPy/SciPy only; no WAV I/O, no labels, no librosa) | internal feature math used by the `acoustic` pack |
| [`linguistic.py`](linguistic.py) | Legacy Vietnamese lexical and disfluency measures over the immutable `Transcript` contract; NFC + `casefold`, diacritics and `đ`/`Đ` preserved | used by the legacy AD path |
| [`schema.py`](schema.py) | 0.1.x contract types retained for the legacy path: manifests, transcript validation, Unicode normalization, SHA-256 provenance | `ExtractionConfig`, `ManifestRow`, `Transcript`, `validate_manifest`, `sha256_file` |
| [`evidence.py`](evidence.py) | Immutable reviewed-candidate inventory rows backing the catalog evidence trail | `EvidenceRecord` |
| [`feature_documentation.py`](feature_documentation.py) | Single source of per-key `(formula, missing_data)` catalog text, resolved through `documentation_for` | `documentation_for` |
| [`_legacy_feature_docs.py`](_legacy_feature_docs.py) | Private store of the **170 catalog-v1 formula and missing-data texts** preserved so no key loses its documented formula; read by `feature_documentation.py`, not part of the public API | `LEGACY_FEATURE_DOCS` |
| [`cli.py`](cli.py) | `say-features` command-line interface (stdlib `argparse`) | `main` |
| [`pipeline.py`](pipeline.py), [`evaluation.py`](evaluation.py), [`tasks.py`](tasks.py) | **Deprecation shims** (removed in `0.3.0`): they re-export the legacy AD API and emit `DeprecationWarning`; no algorithm lives here | legacy re-exports |
| [`features/`](features/README.md) | The four built-in feature packs and their per-pack maps | see the [Feature Packs Map](features/README.md) |
| [`formats/`](formats/README.md) | Document serialization codecs and the static `FORMATS` registry | `FORMATS`, `encode_chat`, `decode_chat`, `encode_json`, `decode_json` |
| [`legacy/`](legacy/README.md) | Supported 0.2.x legacy AD namespace (lazy, warning-free on plain import) | `extract_recording`, `extract_manifest`, `evaluate_ad_baseline` |

---

## 2. Feature Packs

**485 registered features** (catalog version `1`) across four packs. Counts
below are read from the live registry:

| Pack | Features | Scope |
|---|---|---|
| `acoustic` | 167 | Audio quality, timing, phonation, resonance, spectrum, rhythm, CPP/CPPS |
| `adult_neuro` | 183 | Lexical diversity, disfluency, morphosyntax, discourse and task scoring |
| `motor_neuro` | 47 | Articulation (VSA/VAI/FCR), rhythm, DDK intervals, respiration |
| `standardized_acoustic` | 88 | eGeMAPSv02 functionals via the optional openSMILE adapter |

Per-pack module inventories, formulas, and prerequisites live in the
[feature pack maps](features/README.md).

---

## 3. Formats

`formats.FORMATS` is a **static** built-in mapping of format name to an
`{"encode", "decode"}` callable pair. Third-party discovery is deliberately
deferred until a real third-party format exists.

| Format | Codec | Notes |
|---|---|---|
| `json` | [`formats/json.py`](formats/json.py) | JSON v2 is lossless and native; JSON v1 is accepted through a deterministic migration reader |
| `chat` | [`formats/chat.py`](formats/chat.py) | The documented clinical CHAT **subset** — not a CLAN clone |

---

## 4. Missing-Data Semantics

An unavailable feature is returned as `NaN` **plus** a structured issue row in
`bundle.issues` — never a fabricated zero and never a guessed value.

- `STABLE_ERROR_CODES` (raise, abort the run): `EXTRACTION_ERROR`,
  `INVALID_AUDIO`, `INVALID_CHAT`, `INVALID_CONFIG`, `INVALID_DOCUMENT`,
  `MISSING_ANNOTATION`, `MISSING_INPUT`, `TARGET_SPEAKER_REQUIRED`,
  `UNKNOWN_PACK`, `UNSUPPORTED_AUDIO`.
- `STABLE_ISSUE_CODES` (record and continue): `INVALID_TASK_ANNOTATION`,
  `MISSING_OPTIONAL_DEPENDENCY`, `UNCALIBRATED_AUDIO`.

---

## 5. Usage

```python
import speech_features as sf

document = sf.load_document("session_001.cha")

bundle = sf.extract(
    audio_path="session_001.wav",
    document=document,
    channel_index=0,
    target_speakers={"PAR"},
    packs=("acoustic", "adult_neuro"),
)

print(bundle.recordings.shape, bundle.utterances.shape, len(bundle.issues))
print(bundle.provenance["audio_sha256"])
```

CLI:

```bash
say-features validate transcript.cha                  # validate a document or manifest v2
say-features convert transcript.cha transcript.json   # convert between JSON v2 and CHAT
say-features extract manifest.json output/ --task-spec task.json
say-features list-features --domain timing --disorder als
```

---

## 6. Boundaries

- **Research-only.** No diagnosis, no screening cutoff, no normative range, and
  no treatment recommendation. No trained model is embedded in this extractor.
- **No downmixing.** Multi-channel audio is never mean-downmixed to mono; the
  declared channel is extracted directly, because a silent channel would
  attenuate the speech signal.
- **Native PCM preserved.** Feature extraction reads native sample rate and bit
  depth on disk; any 16 kHz view is in-memory only.
- **Static packs.** `PACKS` is a static built-in mapping. Adding a pack is an
  intentional source change — there is no dynamic registry or entry-point
  discovery.

See the [repository README](../../README.md) for installation, the CLI surface,
and the full documentation index.
