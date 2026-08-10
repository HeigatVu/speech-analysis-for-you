# Speech Analysis for You (SAY)

SAY is a **research/descriptive, Python-first library** for speech and sound
analysis of **reviewed Vietnamese speech data**. Version 0.2.0 ships a
versioned speech document model (JSON v2), a CLAN-compatible CHAT subset
(it is **not** a CLAN clone and does not claim full CHAT compatibility), a
label-free extraction pipeline, four built-in feature packs (`acoustic`,
`adult_neuro`, `motor_neuro`, `standardized_acoustic`) with a generated
485-feature catalog, and the
`say-features` command-line interface.

> **Research-only.** SAY is for cohort characterisation and hypothesis
> exploration. It is **not a diagnostic** and not for clinical
> decision-making: it gives **no diagnosis**, is **not a screening tool**,
> defines **no normative range** and **no screening or cutoff score**,
> makes **no treatment recommendation**, and offers **no fixed clinical
> performance**. No trained model is embedded. It performs **no ASR,
> diarization, forced alignment, or automatic morphosyntactic analysis**;
> automated tools may prepare annotations externally, but a **human reviews
> them before SAY consumes them**.

## Documentation

- [Feature extraction](docs/feature-extraction.md) — the label-free pipeline:
  PCM WAV input, target-speaker isolation, `FeatureBundle` tables, issues,
  provenance, pack/level selection, manifest v2, batch behavior, CLI exit
  codes, and all stable error codes.
- [Transcript formats](docs/transcript-formats.md) — JSON v2 and the CHAT
  subset, Vietnamese token/word grouping, normalization, annotation
  provenance, and the manual/automated transcription workflows.
- [Feature catalog v1](docs/feature-catalog-v1.md) — every registered feature
  key with unit, level, prerequisites, formula, and missing-data behavior.
- [Neurodegenerative feature guide](docs/neurodegenerative-feature-guide.md) —
  pack/task selection, reviewed annotation layers, evidence scope, and
  Vietnamese validation limits.
- [Migrating to 0.2](docs/migration-0.2.md) — 0.1-to-0.2 API changes, legacy
  AD imports, the deprecation window, and notebook retirement.

## Installation

Requires Python **3.10–3.13**. Core dependencies are NumPy, SciPy, and
pandas. From a source checkout:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Or install a built wheel:

```bash
uv build --wheel
uv pip install dist/speech_analysis_for_you-0.2.0-py3-none-any.whl
```

The `standardized_acoustic` pack is optional and keeps openSMILE lazy:

```bash
uv pip install "speech-analysis-for-you[standardized-acoustic]"
```

Without that extra, core imports and extraction still work; selecting the
pack returns `NaN` eGeMAPS columns plus one `MISSING_OPTIONAL_DEPENDENCY`
issue.

## Python quick start

```python
import speech_features as sf

document = sf.load_document("recording.json")          # JSON v2 (or CHAT)
features = sf.extract(                                 # choose any built-in packs
    "recording.wav",
    document,
    packs=("acoustic", "adult_neuro", "motor_neuro"),
    task_spec={
        "version": 1,
        "task": "picture_desc_1",
        "concept_aliases": {"cat": ["mèo"]},
        "entity_groups": {},
        "action_groups": {},
    },
)
print(features.recordings)                             # recording_id, speaker_id, feature columns
print(features.utterances)                             # + utterance_id, start_s, end_s
print(features.issues)                                 # structured warnings/issues
print(features.provenance)                             # hashes, config, versions

for definition in sf.list_features(pack="acoustic"):
    print(definition.key, definition.unit, definition.level)

for definition in sf.list_features(disorder="als", task="ddk"):
    print(definition.key, definition.evidence_level)
```

## CLI quick start

```bash
say-features validate transcript.cha                 # validate a document or manifest v2
say-features convert transcript.cha transcript.json # convert between JSON v2 and CHAT (--force to overwrite)
say-features extract manifest.json output/ --task-spec task.json
say-features list-features --domain timing --disorder als
```

The `extract` command writes `recordings.csv`, `utterances.csv`,
`issues.csv`, and `provenance.json` into the output directory. Exit codes:
`0` success or warnings, `1` partial row failures, `2` invalid global
input/configuration.

## Outputs

Every extraction returns a `FeatureBundle` with three pandas tables and
JSON-serializable provenance:

- `recordings` — `recording_id`, `speaker_id`, then stable feature columns.
- `utterances` — the same identifiers plus `utterance_id`, `start_s`, `end_s`.
- `issues` — `recording_id`, `speaker_id`, `utterance_id`, `feature`, `code`,
  `severity`, `message`; unavailable values are `NaN` plus an issue, never a
  fabricated zero.
- `provenance` — input SHA-256 hashes, configuration, package/catalog
  versions, target speakers, and annotation sources.

## Architecture and extension boundary

- `load_document`/`save_document` — versioned document model
  (`SpeechDocument`, JSON v2 native; JSON v1 migrated deterministically;
  CHAT subset read/write).
- `extract`/`extract_batch` — label-free pipeline over manifest v2, with
  row/target isolation and provenance.
- `list_features` — catalog queries over static built-in pack registration.
- `PACKS` is a **static** built-in mapping in 0.2; a future built-in pack
  (for example a child pack) requires an intentional source change and validation —
  no dynamic registry or entry-point discovery (see the
  [catalog](docs/feature-catalog-v1.md) for the minimal example).
- Legacy AD evaluation and task scorers remain available through
  `speech_features.legacy.ad` with a `DeprecationWarning`; see
  [migration-0.2](docs/migration-0.2.md).

## Future transcript automation

SAY does not currently transcribe audio. A future companion may produce draft JSON v2
or CHAT transcripts, but automated transcription remains outside the core feature
extractor. It should be evaluated on held-out human ground truth transcripts using at
least word and character error rates, plus speaker and timestamp accuracy when those
annotations are produced. Report results across relevant Vietnamese populations and
recording conditions, record annotation provenance, and require human review and
correction before SAY consumes the transcript.

## Development

```bash
uv run pytest tests/speech_features
uv run ruff check src/speech_features tests/speech_features
uv run ruff format --check src/speech_features tests/speech_features
```

## License

MIT — see the [LICENSE](LICENSE) file.
