# Migrating to SAY 0.2

Version 0.2.0 keeps the distribution name `speech-analysis-for-you`, the
import namespace `speech_features`, and the `say-features` console entry
point. The old AD-oriented pipeline moves behind
`speech_features.legacy.ad`; the new label-free API replaces it for new
work. See the [README](../README.md) for installation and
[feature-extraction.md](feature-extraction.md) for the new workflow.

## Old-to-new API

| 0.1 (deprecated in 0.2) | 0.2 replacement |
|---|---|
| `speech_features.extract_recording(...)` | `speech_features.extract(audio_path, document, ...)` |
| `speech_features.extract_manifest(...)` | `speech_features.extract_batch(manifest_path, ...)` |
| `speech_features.BatchResult` | `speech_features.FeatureBundle` (recordings/utterances/issues/provenance) |
| `speech_features.BatchFailure` | severity-`error` rows in `FeatureBundle.issues` |
| `speech_features.FeatureResult` | `FeatureBundle.recordings` / `FeatureBundle.utterances` |
| `speech_features.evaluate_ad_baseline(...)` | not replaced — legacy AD evaluation only |
| label-bearing manifest v1 | label-free manifest v2 |
| transcript JSON v1 | JSON v2 (v1 migrated deterministically) |

The old top-level names (`extract_recording`, `extract_manifest`,
`BatchResult`, `BatchFailure`) still work through `0.2.x` but emit an
advisory `DeprecationWarning` pointing at
`speech_features.legacy.ad.pipeline`. They are **eligible for removal in
0.3.0**.

## Manifest v1 versus v2

- **Manifest v1 (legacy, label-bearing)** carries `participant_id`, `task`,
  `diagnosis`, `age`, `sex`, `education_years`, and task-spec paths per row.
  It exists only for the deprecated AD workflow.
- **Manifest v2 (label-free)** rows contain exactly `recording_id`,
  `audio_path`, `transcript_path`, and optional `target_speakers`. No
  labels, tasks, demographics, or clinical values are accepted or emitted.

If you need AD evaluation or task scoring, keep your v1 manifests with the
legacy workflow; new extraction work should move to v2.

## Legacy imports and the `legacy-ad` extra

The legacy AD implementation (task scorers, task manifest pipeline, and the
AD evaluation baseline) lives under `speech_features.legacy.ad` with
**unchanged numerical behavior** — there is no hidden numerical change to
legacy paths. Install its optional dependency with:

```bash
uv pip install -e ".[legacy-ad]"
```

## Deprecation window

- Through all `0.2.x` releases the legacy imports keep working and emit
  `DeprecationWarning`; they are eligible for removal in `0.3.0`.
- The legacy `legacy-preprocessing` extra (Hydra/pydub preprocessing) is
  likewise retained only for compatibility and not part of the 0.2
  workflow.

## Notebook retirement

The old notebook workflows (`acoustics.ipynb`, `linguistic.ipynb`,
`classification/classifiers.ipynb`) are **retired**: no production logic
runs from notebooks, and no notebook dependency is part of the 0.2 package
or lockfile. Remaining notebooks may only be thin tutorials that call the
public API. Existing notebook users should move their analysis into plain
Python scripts using the API in the [README](../README.md) quick start.

## Verification

Migration is safe to run before committing: `say-features validate` checks
both old v1 documents (via migration) and new v2 documents/manifests.
