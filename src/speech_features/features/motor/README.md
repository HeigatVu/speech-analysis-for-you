# Motor-Neuro Feature Pack (`src/speech_features/features/motor/`)

The **`motor_neuro`** pack provides **47 registered features** specifically designed for motor speech evaluation, dysarthria profiling, and motor-neuron diseases (Amyotrophic Lateral Sclerosis / ALS, Parkinson's Disease / PD, Cerebellar Ataxia, Huntington's Disease / HD).

All features are registered in [`definitions.py`](file:///home/ducvu/Heigat-home/project/speech-analysis-for-you/src/speech_features/features/motor/definitions.py) under pack name `"motor_neuro"`.

---

## 1. Module Inventory

| Module | Features | Description | Key Indicators |
|---|---|---|---|
| [`definitions.py`](file:///home/ducvu/Heigat-home/project/speech-analysis-for-you/src/speech_features/features/motor/definitions.py) | — | Registration schema, units, levels, prerequisites, and formula versions | Metadata definitions for all 47 features |
| [`intervals.py`](file:///home/ducvu/Heigat-home/project/speech-analysis-for-you/src/speech_features/features/motor/intervals.py) | 47 | Fine-grained acoustic interval measurements across articulation, rhythm, DDK, and respiration | VSA, VAI, FCR, DDK regularity, breath group intervals |

---

## 2. Feature Domains

### Articulation (19 features)
- **Vowel Working Space:**
  - `artic_vowel_space_area_hz2`: Shoelace polygon area of the corner vowel triangle ($/i/, /u/, /a/$) in the $(F_1, F_2)$ plane.
  - `artic_vowel_articulation_index` (VAI) & `artic_formant_centralization_ratio` (FCR): Acoustic markers of vowel centralization and dysarthric acoustic compression.
  - `artic_vowel_dispersion_mean_hz` & `artic_vowel_dispersion_sd_hz`: Euclidean distance from the vowel centroid.
  - `artic_formant_transition_slope_mean_hz_s`: Dynamic articulatory transition rate.
- **Consonants & Obstacles:**
  - `artic_vot_mean_s`, `artic_vot_sd_s`: Voice Onset Time for stop consonants.
  - `artic_stop_gap_mean_s`, `artic_stop_gap_sd_s`: Silent occlusion intervals preceding release bursts.
  - `artic_fricative_m1_mean` to `m4_mean`: Spectral moments (center of gravity, variance, skewness, kurtosis) across fricative segments.

### Rhythm & Syllable Timing (7 features)
- `rhythm_percent_vocalic` ($\%V$): Proportion of vocalic duration.
- `rhythm_varco_v`, `rhythm_varco_c`: Rate-normalized variation coefficient of vocalic and consonantal durations.
- `rhythm_rpvi_v`, `rhythm_npvi_v`: Raw and normalized Pairwise Variability Indices for vowels.
- `rhythm_rpvi_c`, `rhythm_npvi_c`: Pairwise Variability Indices for consonants.

### Diadochokinetic Performance / DDK (10 features)
- `task_ddk_rate_syllables_s`: Syllables articulated per second during rapid repetition tasks (`/pa-ta-ka/` or `/pa/`).
- `task_ddk_inter_onset_mean_s`, `task_ddk_inter_onset_sd_s`, `task_ddk_inter_onset_cv`: Regularity and variability of inter-onset intervals.
- `task_ddk_instability_s`, `task_ddk_acceleration_syllables_s2`, `task_ddk_decay_ratio`: Fatigue, kinematic deceleration, or pace acceleration across trials.
- `task_ddk_sequential_alternating_ratio`: Relative timing performance comparing Sequential Motion Rates (SMR) and Alternating Motion Rates (AMR).

### Respiratory & Breath Groups (6 features)
- `resp_breath_group_count`, `resp_breath_group_mean_s`, `resp_breath_group_sd_s`: Frequency and duration of acoustic breath phrases.
- `resp_pauses_per_breath`: Frequency of atypical inhalation pauses within syntactic phrases.
- `resp_relative_loudness_db`: Subglottal air pressure decay proxy over breath groups.

---

## 3. Usage

```python
import speech_features as sf

document = sf.load_document("ddk_session.cha")
bundle = sf.extract(
    "ddk_session.wav",
    document,
    packs=("motor_neuro",),
    target_speakers={"PAR"},
)

motor_cols = [c for c in bundle.recordings.columns if c.startswith(("artic_", "rhythm_", "task_ddk_", "resp_"))]
print(bundle.recordings[motor_cols].head())
```
