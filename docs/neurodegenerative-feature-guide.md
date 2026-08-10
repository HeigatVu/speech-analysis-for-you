# Neurodegenerative speech feature guide

SAY provides four additive descriptive packs for reviewed adult speech:
`acoustic`, `adult_neuro`, `motor_neuro`, and optional
`standardized_acoustic`. Select only the packs supported by the recording
task and available annotation layers; unavailable values remain `NaN` with a
structured issue. The generated [catalog](feature-catalog-v1.md) and
[evidence inventory](research/neurodegenerative-speech-feature-inventory.md)
are the feature-key and evidence references. See the [README](../README.md)
and [feature extraction guide](feature-extraction.md) for API and CLI usage.

> **Research-only.** These features are not a diagnostic, screening score,
> prognosis, normative range, or treatment recommendation. They describe
> reviewed observations and do not establish clinical performance.

## Packs and tasks

- `acoustic` uses target-aligned WAV audio for quality, timing, phonation,
  prosody, resonance, spectrum, MFCC, and nonlinear companion measures.
- `adult_neuro` uses reviewed transcript annotations for lexical,
  morphosyntactic, psycholinguistic, error, discourse, and structured-task
  measures. Picture description, story recall, semantic fluency, and
  phonemic fluency use a version-1 task spec.
- `motor_neuro` uses reviewed articulatory, rhythm, DDK, respiratory, and
  sustained-vowel observations. DDK and sustained-vowel task specs identify
  their task; calibrated amplitude must be declared before a relative
  loudness value is interpreted.
- `standardized_acoustic` uses the audio array for the frozen 88-value
  eGeMAPSv02 functionals. Install `.[standardized-acoustic]` to add openSMILE;
  openSMILE is imported lazily and is not a core dependency.

Example task spec:

```json
{
  "version": 1,
  "task": "picture_desc_1",
  "concept_aliases": {"cat": ["mèo"]},
  "entity_groups": {"animal": ["mèo"]},
  "action_groups": {"motion": ["chạy"]}
}
```

Pass it as `task_spec=` for one extraction, `task_spec_path` in a manifest
row, or `say-features extract ... --task-spec task.json` as a batch fallback.
Only its canonical SHA-256 is retained in provenance.

## Reviewed annotation layers

The packs never infer missing linguistic or motor annotations. Depending on
the selected feature, reviewed JSON v2 `AnnotationLayer` values include
`lemma`, `upos`, `classifier`, `particle`, `sentence_id`, `t_unit_id`,
`clause_id`, `yngve_depth`, psycholinguistic scores, `error_type`,
`cohesion_type`, `information_unit`, `discourse_role`, `topic_relevant`,
`segment_type`, `f1_hz`, `f2_hz`, `f0_hz`, `power_db`, `breath_group`,
and DDK/sustained-vowel event annotations. Annotation provenance records the
layer source and confidence. Automated preparation requires human review.

`INVALID_TASK_ANNOTATION` marks malformed reviewed task observations;
`UNCALIBRATED_AUDIO` marks loudness that lacks calibration; and
`MISSING_OPTIONAL_DEPENDENCY` marks an unavailable optional eGeMAPS runtime.
These are data-quality/result semantics, not clinical findings.

## Vietnamese validation limits

NFC normalization and casefold comparison preserve Vietnamese diacritics and
the `d`/`đ` distinction. That text correctness does not establish construct
validity. Validate each feature independently on held-out reviewed Vietnamese
data for the intended dialect, elicitation task, age/population, annotation
workflow, microphone, and recording environment. No current evidence here
defines Vietnamese clinical cutoffs.

## Future child pack

A future child pack can reuse the population-neutral `SpeechDocument`,
`FeatureBundle`, and `FeaturePack` name/version contract. It must add its own
reviewed formulas, definitions, extractor, evidence, and validation; adult
metadata must never be relabeled as child norms. `PACKS` remains static and
no child pack or dynamic entry-point discovery is implemented today.
