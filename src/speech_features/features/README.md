# SAY Feature Packs Map (`src/speech_features/features/`)

This directory houses the built-in, label-free feature extraction packs for the **Speech Analysis for You (SAY)** library. SAY implements **485 registered features** across four primary packs designed for speech characterization and clinical hypothesis exploration.

---

## 1. Feature Pack Inventory

| Feature Pack | Level | Catalog Size | Primary Target / Disorders | Documentation Map |
|---|---|---|---|---|
| **`acoustic`** | Recording & Utterance | 167 features | Audio quality, prosody, timing, resonance, spectrum, rhythm (AD, MCI, neurodegenerative cohorts) | [`acoustic/README.md`](file:///home/ducvu/Heigat-home/project/speech-analysis-for-you/src/speech_features/features/acoustic/README.md) |
| **`adult_neuro`** | Recording & Utterance | 183 features | Lexical diversity, disfluency, morphosyntax, discourse coherence, task scoring (AD, PPA, MCI, FTD) | [`linguistic/README.md`](file:///home/ducvu/Heigat-home/project/speech-analysis-for-you/src/speech_features/features/linguistic/README.md) |
| **`motor_neuro`** | Recording | 47 features | Articulation (VSA, VAI, FCR), rhythm metrics, DDK intervals, respiratory measures (ALS, PD, Ataxia, HD) | [`motor/README.md`](file:///home/ducvu/Heigat-home/project/speech-analysis-for-you/src/speech_features/features/motor/README.md) |
| **`standardized_acoustic`** | Recording | 88 features | Standardized eGeMAPSv02 functionals via openSMILE adapter (cross-corpus comparability) | [`standardized/README.md`](file:///home/ducvu/Heigat-home/project/speech-analysis-for-you/src/speech_features/features/standardized/README.md) |
| **Total** | — | **485 features** | — | — |

---

## 2. Architectural Principles

1. **Static Built-in Registration:**
   All features are registered explicitly into the central catalog (`speech_features.catalog.REGISTRY`) via static `register_feature()` calls when modules are imported. No dynamic entry points or untyped reflection are used.

2. **Immutable Feature Bundles (`FeatureBundle`):**
   Extraction via `sf.extract()` produces a frozen bundle containing three pandas DataFrames and metadata:
   - `recordings`: Recording-level features indexed by `(recording_id, speaker_id)`.
   - `utterances`: Utterance-level features indexed by `(recording_id, speaker_id, utterance_id, start_s, end_s)`.
   - `issues`: Structured tracking table (`recording_id`, `speaker_id`, `utterance_id`, `feature`, `code`, `severity`, `message`).
   - `provenance`: Cryptographic SHA-256 hashes of master WAV and transcript files, runtime configuration, and package versions.

3. **Strict Missing Data Semantics:**
   Unavailable features are returned as `NaN` paired with a structured issue code in `bundle.issues` (e.g., `MISSING_ANNOTATION`, `INSUFFICIENT_TOKENS`, `NO_AUDIO`), **never** fabricated zeros or guessed values.

4. **Multi-Channel & Target Speaker Safety:**
   - Multi-channel audio is never mean-downmixed; the designated speaker channel is extracted directly.
   - Speech analysis isolates target speakers (`target_speakers`) to prevent interviewer cues or cross-talk from corrupting acoustic and linguistic metrics.

---

## 3. Directory Layout

```
src/speech_features/features/
├── README.md               # This top-level feature architecture map
├── acoustic/               # Built-in acoustic pack (167 features)
│   ├── README.md           # Acoustic sub-pack inventory & formulas
│   ├── definitions.py      # Catalog metadata & registration
│   ├── quality.py          # Duration, DC offset, clipping, RMS dBFS
│   ├── timing.py           # Speech/pause ratios, articulation rate, acceleration
│   ├── phonation.py        # F0 statistics, jitter, shimmer, HNR
│   ├── resonance.py        # F1-F4 formants, dispersion, bandwidths
│   ├── spectrum.py         # Centroid, tilt, roll-off, flux, alpha ratio, Hammarberg
│   ├── rhythm.py           # Vocalic/consonantal durations & variability
│   └── advanced.py         # Cepstral Peak Prominence (CPP/CPPS)
├── linguistic/             # Built-in adult_neuro pack (183 features)
│   ├── README.md           # Adult-neuro linguistic inventory & scoring
│   ├── definitions.py      # Catalog metadata & registration
│   ├── diversity.py        # TTR, MATTR-20, MTLD, HD-D-42, Brunet W, Honoré R, entropy
│   ├── clinical.py         # Fillers, fragments, immediate repetitions, revisions, retracings
│   ├── morphosyntax.py     # UPOS ratios, open/closed class, noun-to-verb ratio
│   └── task_scores.py      # Discourse cohesion, information units, concept scoring
├── motor/                  # Built-in motor_neuro pack (47 features)
│   ├── README.md           # Motor-neuro inventory (ALS, PD, Ataxia)
│   ├── definitions.py      # Catalog metadata & registration
│   └── intervals.py        # VSA, VAI, FCR, DDK syllable rates & regularity, breath groups
└── standardized/           # Built-in standardized_acoustic pack (88 features)
    ├── README.md           # eGeMAPSv02 feature map & openSMILE adapter
    ├── definitions.py      # Frozen eGeMAPSv02 functionals schema
    └── opensmile_adapter.py # Lazy openSMILE wrapper with graceful fallback
```

---

## 4. Extraction Example

```python
import speech_features as sf

# Load validated transcript (JSON v2 or CHAT subset)
document = sf.load_document("session_001.cha")

# Extract selected packs with explicit task specification
bundle = sf.extract(
    audio_path="session_001.wav",
    document=document,
    channel_index=0,
    target_speakers={"PAR"},
    packs=("acoustic", "adult_neuro", "motor_neuro", "standardized_acoustic"),
)

# Inspect outputs
print("Recording features:", bundle.recordings.shape)
print("Utterance features:", bundle.utterances.shape)
print("Logged issues:", len(bundle.issues))
print("Audio SHA-256:", bundle.provenance["audio_sha256"])
```
