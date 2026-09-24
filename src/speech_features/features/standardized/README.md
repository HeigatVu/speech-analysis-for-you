# Standardized Acoustic Feature Pack (`src/speech_features/features/standardized/`)

The **`standardized_acoustic`** pack provides **88 registered features** implementing the Geneva Minimalistic Acoustic Parameter Set version 02 (**eGeMAPSv02**) functional feature set via openSMILE.

All features are registered in [`definitions.py`](file:///home/ducvu/Heigat-home/project/speech-analysis-for-you/src/speech_features/features/standardized/definitions.py) under pack name `"standardized_acoustic"`.

---

## 1. Module Inventory

| Module | Features | Description | Key Indicators |
|---|---|---|---|
| [`definitions.py`](file:///home/ducvu/Heigat-home/project/speech-analysis-for-you/src/speech_features/features/standardized/definitions.py) | — | Frozen eGeMAPSv02 feature names and catalog metadata | Registration for all 88 eGeMAPSv02 functionals |
| [`opensmile_adapter.py`](file:///home/ducvu/Heigat-home/project/speech-analysis-for-you/src/speech_features/features/standardized/opensmile_adapter.py) | 88 | Lazy openSMILE wrapper extracting eGeMAPSv02 functionals over target speech segments | Pitch dynamics, energy functionals, spectral balance, and MFCCs 1–4 |

---

## 2. Parameter Set (eGeMAPSv02)

The 88 functionals capture:
- **Frequency parameters:** Fundamental frequency ($F_0$) semi-tone statistics, rising/falling pitch slopes, jitter, formants $F_1$, $F_2$, $F_3$ center frequencies and bandwidths.
- **Energy/Amplitude parameters:** Loudness functional distributions, rising/falling loudness slopes, local shimmer.
- **Spectral parameters:** Alpha ratio (50–1000 Hz vs 1000–5000 Hz), Hammarberg index, spectral slopes (0–500 Hz, 500–1500 Hz), relative harmonic amplitudes ($H_1-H_2$, $H_1-A_3$), spectral flux, and MFCCs 1–4.

---

## 3. Lazy Dependency Design

The `opensmile` dependency is optional. Core SAY features run completely independently of openSMILE.

- **To install:**
  ```bash
  uv pip install "speech-analysis-for-you[standardized-acoustic]"
  ```
- **Fallback behavior:**
  If the pack is requested without `opensmile` installed, extraction completes successfully with `NaN` values in eGeMAPS columns and an issue record `MISSING_OPTIONAL_DEPENDENCY:opensmile`.

---

## 4. Usage

```python
import speech_features as sf

document = sf.load_document("session_001.cha")
bundle = sf.extract(
    "session_001.wav",
    document,
    packs=("standardized_acoustic",),
    target_speakers={"PAR"},
)

# Inspect standardized columns
egemaps_cols = [c for c in bundle.recordings.columns if "_sma3" in c]
print(f"Extracted {len(egemaps_cols)} eGeMAPSv02 features")
```
