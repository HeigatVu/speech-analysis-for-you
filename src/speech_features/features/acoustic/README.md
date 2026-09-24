# Acoustic Feature Pack (`src/speech_features/features/acoustic/`)

The **`acoustic`** pack provides **167 registered features** covering digital audio quality, macro-timing, micro-prosody, phonation, resonance, spectral envelope, rhythm, and cepstral stability.

All features are registered in [`definitions.py`](file:///home/ducvu/Heigat-home/project/speech-analysis-for-you/src/speech_features/features/acoustic/definitions.py) under pack name `"acoustic"`.

---

## 1. Module Inventory

| Module | Features | Description | Key Indicators |
|---|---|---|---|
| [`quality.py`](file:///home/ducvu/Heigat-home/project/speech-analysis-for-you/src/speech_features/features/acoustic/quality.py) | 4 | Signal level, DC bias, and digital clipping | `audio_duration_s`, `audio_dc_offset`, `audio_clipping_ratio`, `audio_rms_dbfs` |
| [`timing.py`](file:///home/ducvu/Heigat-home/project/speech-analysis-for-you/src/speech_features/features/acoustic/timing.py) | 31 | Macro-temporal pacing, pauses, and rates | `time_speech_ratio`, `time_articulation_rate_syllables_per_s`, `time_pause_rate_per_min`, `time_long_pause_count`, `time_words_per_min`, `time_timing_event_entropy` |
| [`phonation.py`](file:///home/ducvu/Heigat-home/project/speech-analysis-for-you/src/speech_features/features/acoustic/phonation.py) | 36 | Fundamental frequency ($F_0$), cycle-to-cycle perturbation, and periodicity | `voice_f0_mean_hz`, `voice_f0_range_semitones`, `voice_jitter_local`, `voice_jitter_rap`, `voice_shimmer_local`, `voice_shimmer_apq5`, `voice_hnr_mean_db`, `voice_tremor_f0_hz` |
| [`resonance.py`](file:///home/ducvu/Heigat-home/project/speech-analysis-for-you/src/speech_features/features/acoustic/resonance.py) | 28 | Vocal tract formants ($F_1$--$F_4$) and vocal tract length proxies | `formant_f1_mean_hz`, `formant_f2_mean_hz`, `formant_f3_mean_hz`, `formant_f4_mean_hz`, `formant_dispersion_hz`, formant bandwidths |
| [`spectrum.py`](file:///home/ducvu/Heigat-home/project/speech-analysis-for-you/src/speech_features/features/acoustic/spectrum.py) | 48 | Spectral energy distribution, spectral moments, and glottal tilt | `spec_centroid_mean_hz`, `spec_spread_mean_hz`, `spec_skewness_mean`, `spec_kurtosis_mean`, `spec_rolloff85_mean_hz`, `spec_flux_mean`, `spec_alpha_ratio_db`, `spec_hammarberg_index_db` |
| [`rhythm.py`](file:///home/ducvu/Heigat-home/project/speech-analysis-for-you/src/speech_features/features/acoustic/rhythm.py) | 12 | Syllabic and vocalic timing variability | `rhythm_pvi_vocalic`, `rhythm_pvi_consonantal`, `rhythm_varco_vocalic`, `rhythm_varco_consonantal` |
| [`advanced.py`](file:///home/ducvu/Heigat-home/project/speech-analysis-for-you/src/speech_features/features/acoustic/advanced.py) | 8 | Cepstral Peak Prominence (CPP) and smoothed CPP (CPPS) | `voice_cpp_mean_db`, `voice_cpps_mean_db`, `voice_cpp_sd_db`, `voice_cpps_sd_db` |

---

## 2. Extraction Prerequisites & Levels

- **Native PCM Audio:** Features are extracted directly from native sample rate/bit-depth audio on the declared channel (`channel_index`). Downmixing is prohibited.
- **Levels:**
  - **Recording-level:** Aggregated over the target speaker's entire audio duration or speech turns.
  - **Utterance-level:** Computed per turn/utterance slice (`time_response_latency_s`, local $F_0$, local RMS, utterance pause rates).
- **Missing Data:** When silence or no valid pitch cycles exist, returns `NaN` with issue code `NO_AUDIO` or `INSUFFICIENT_VOICING`.

---

## 3. Usage

```python
import speech_features as sf

document = sf.load_document("sample.cha")
bundle = sf.extract("sample.wav", document, packs=("acoustic",))

# Retrieve acoustic columns
acoustic_cols = [c for c in bundle.recordings.columns if c.startswith(("audio_", "time_", "voice_", "formant_", "spec_", "rhythm_"))]
print(bundle.recordings[acoustic_cols].head())
```
