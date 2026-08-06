# SAY Vietnamese Speech Feature Library Design

## Purpose

Refactor `speech_features` from a six-task Alzheimer research pipeline into a
Python-first, research/descriptive library for Vietnamese adult
neuro/geriatric speech. The core data and extraction contracts are
population-neutral so a pediatric pack can be added later without changing
the transcript or result schemas.

The library does not diagnose, provide normative cutoffs, recommend treatment,
or embed a trained clinical model. Batchalign, ASR, diarization, forced
alignment, and morphosyntactic taggers remain external. SAY consumes the
reviewed output of those tools and preserves enough provenance for a human to
edit and re-run extraction safely.

## Public API

```python
load_document(path, *, format=None) -> SpeechDocument
save_document(document, path, *, format=None, force=False) -> None

extract(
    audio_path,
    document,
    *,
    target_speaker=None,
    packs=("acoustic", "adult_neuro"),
    levels=("recording", "utterance"),
    config=None,
) -> FeatureBundle

extract_batch(manifest_path, *, packs=("acoustic", "adult_neuro"), config=None) -> BatchResult
list_features(*, pack=None, level=None) -> tuple[FeatureDefinition, ...]
```

The console entry point is `say-features` with `validate`, `convert`,
`extract`, and `list-features` subcommands.

## Document model

`SpeechDocument` JSON version 2 is the lossless native representation. It
contains document/media identifiers, ISO 639-3 language (`vie`), document
speakers/utterances, stable token IDs, optional time alignment, explicit word grouping,
linguistic annotations, annotation-layer provenance, and preserved raw CHAT
tiers.

Vietnamese words are never inferred by whitespace. A time-aligned token may be
one syllable and several tokens may share a `word_id`; a manually segmented
multi-syllable word may also be represented as a single token. Text is NFC
normalised and casefolded only for comparison. Diacritics and the `d`/`đ`
distinction are preserved.

JSON version 1 is accepted through a deterministic migration reader. Existing
word tokens become one word each and receive stable positional IDs.

The Python document types are named `DocumentSpeaker`, `DocumentUtterance`,
`DocumentToken`, and `MediaRef` so the deprecated AD `Token`/`Utterance`
contracts can remain available through `0.2.x` without ambiguous root exports.

The supported clinical CHAT subset includes `@Begin`, `@Languages`,
`@Participants`, `@ID`, `@Media`, `@End`, speaker tiers, media bullets,
`%mor`, `%gra`, and common fillers, fragments, retracing, revision, and error
codes. Unsupported dependent tiers and lines are preserved verbatim with a
structured warning. Conversion refuses to overwrite a source unless
`force=True` or `--force` is explicit.

## Feature results and catalog

`FeatureBundle` contains pandas tables:

- `recordings`: `recording_id`, `speaker_id`, then stable feature columns.
- `utterances`: the same identifiers plus `utterance_id`, `start_s`, and
  `end_s`.
- `issues`: identifiers plus `feature`, `code`, `severity`, and `message`.
- provenance metadata: input SHA-256 hashes, configuration, package version,
  catalog version, and annotation sources.

Every `FeatureDefinition` declares key, pack, level, unit, prerequisites,
population applicability, formula version, and reference. Stable prefixes are
`audio_`, `time_`, `voice_`, `spectral_`, `lex_`, `morph_`,
`disfluency_`, and `discourse_`.

A minimal built-in `FeaturePack` protocol has two version-1 implementations:
`acoustic` and `adult_neuro`. Registration is a static mapping. Third-party
entry-point discovery is deferred until an actual third-party pack exists.

## Adult neuro feature catalog

### Audio, timing, and quality

Duration, DC offset, clipping ratio, RMS dBFS, participant speech time and
ratio, voiced-segment duration summaries, pause count/rate and duration
summaries, long pauses, response latency, overlap, words/minute,
syllables/minute, and articulation rate. Participant-only acoustic measures
require aligned target-speaker intervals. Whole-recording analysis is allowed
only when explicitly requested and is flagged.

### Phonation, prosody, resonance, spectrum, and rhythm

F0 mean, median, standard deviation, coefficient of variation, IQR, 5--95%
range, slope, and absolute change; voiced ratio; intensity summaries and slope;
local jitter; local shimmer; HNR summaries; cepstral peak prominence; LPC F1,
F2, F3 and bandwidth summaries; spectral centroid, spread, slope, rolloff,
flux, flatness, and entropy summaries; voiced/unvoiced durations; and
annotation-derived syllable-duration variability.

### Language sample and Vietnamese annotation features

Utterance, token, word, syllable, character, and unique counts; MLU in words
and syllables; token/lemma TTR, MATTR-20, MTLD, HD-D, hapax ratio, Brunet W,
Honore R, entropy, and token-length summaries; UPOS and dependency
distributions; content/function, noun/verb, pronoun/noun, classifier, particle,
and code-switch ratios when reviewed annotations are present.

### Disfluency, morphosyntax, and conversation

Fillers, fragments, immediate repetitions, retracing, revisions, mazes,
annotated errors, dependency length and tree depth, clause and subordination
rates, turn counts and lengths, examiner-prompt ratio, response latency, and
overlap.

Unavailable prerequisites produce `NaN` and `MISSING_ANNOTATION`, never a
guessed zero. Insufficient acoustic evidence produces `NaN` and a specific
quality issue.

## Compatibility and packaging

Release `0.2.0` keeps the distribution name `speech-analysis-for-you`, import
namespace `speech_features`, and deprecated top-level AD imports. The existing
task scorers and AD evaluation move behind `speech_features.legacy.ad` and are
eligible for removal in `0.3.0`.

Core dependencies are NumPy, SciPy, and pandas. scikit-learn and the old Hydra/
pydub preprocessing workflow become optional extras. `import-ipynb`, librosa,
openSMILE, and nbconvert are not core runtime dependencies. Standard PCM WAV
is the only extraction audio format in version 1.

All production logic lives in `.py`. Remaining notebooks may only be thin
tutorials that call the public API; they contain no extraction algorithms,
package installation, reusable configuration, or hard-coded production paths.

## Failure behavior

Stable codes include `INVALID_DOCUMENT`, `INVALID_CHAT`, `INVALID_AUDIO`,
`UNSUPPORTED_AUDIO`, `MISSING_INPUT`, `TARGET_SPEAKER_REQUIRED`,
`MISSING_ANNOTATION`, `UNKNOWN_PACK`, `INVALID_CONFIG`, and
`EXTRACTION_ERROR`. Optional missing annotations fail only their dependent
features. Batch extraction isolates row failures and continues.

CLI exit codes are `0` for success or warnings, `1` for partial row failures,
and `2` for invalid global input/configuration.

## Verification

All logic is developed test-first. Synthetic audio validates signal formulas;
hand-calculated Vietnamese fixtures validate linguistic formulas. CHAT fixtures
cover Batchalign-shaped data, known-tier round trips, and unknown-tier
preservation. No patient data is committed. The finished wheel must install
and expose the API and CLI on Python 3.10--3.13.
