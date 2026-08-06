# Transcript Formats

SAY consumes **reviewed Vietnamese transcripts** as versioned speech
documents. The native representation is **JSON v2**; a deterministic reader
migrates JSON v1, and a CLAN-compatible **CHAT subset** is read and written
for interchange. SAY is **not a CLAN clone** and does **not** claim full
CHAT compatibility: only the documented subset below is supported, and
everything else is preserved verbatim with a structured warning.

> Automated tools (Batchalign, ASR, taggers, parsers) may prepare
> annotations externally, but a **human reviews** them before SAY consumes
> them. SAY never imports or invokes those tools itself.

## 1. JSON v2 (native)

A JSON v2 document is a single object with stable fields:

- `version` — exactly `2`.
- `document_id` — non-empty string.
- `language` — exactly `vie` (ISO 639-3).
- `media` — list of `{id, kind, path, sha256}` media references.
- `speakers` — list of `{id, name, role}`.
- `utterances` — list of `{id, speaker_id, start_s, end_s, tokens}` with
  `end_s >= start_s`.
- `annotations` — list of `{layer, source, confidence, values}` layers with
  `confidence` in `[0, 1]` and `values` mapping token ids to strings/numbers.
- `raw_tiers` — preserved unsupported CHAT lines, keyed by tier name.

Each token is `{id, text, kind, start_s, end_s, word_id, language, dep_head,
dep_rel}`. Token times must lie inside their utterance times; `dep_head`
must reference an existing token id.

Minimal valid example:

```json
{
  "version": 2,
  "document_id": "rec-001",
  "language": "vie",
  "media": [
    {"id": "rec-001.wav", "kind": "audio", "path": "rec-001.wav", "sha256": null}
  ],
  "speakers": [
    {"id": "PAR", "name": "Nguyễn Văn A", "role": "participant"},
    {"id": "EXA", "name": "Người khảo sát", "role": "examiner"}
  ],
  "utterances": [
    {
      "id": "u0001",
      "speaker_id": "PAR",
      "start_s": 1.5,
      "end_s": 4.0,
      "tokens": [
        {"id": "u0001_t0001", "text": "Tôi", "kind": "word", "start_s": 1.5, "end_s": 1.8},
        {"id": "u0001_t0002", "text": "đi", "kind": "word", "start_s": 1.9, "end_s": 2.1, "word_id": "w1"}
      ]
    }
  ],
  "annotations": [
    {"layer": "lemma", "source": "human", "confidence": 1.0, "values": {"u0001_t0001": "tôi", "u0001_t0002": "đi"}}
  ],
  "raw_tiers": {}
}
```

`save_document` serializes JSON v2 with a stable field order and **refuses
to overwrite an existing file unless `force=True`** (CLI: `--force`).

### JSON v1 migration (deterministic)

JSON v1 documents (`version: 1` with an `utterances` list) are migrated
deterministically: utterances receive ids `u0001...`, tokens
`u0001_t0001...`, and each old word token becomes exactly one word token —
**never split on whitespace** — unless the source already declares a
`word_id`, which is carried through. `language` becomes `vie`.

## 2. Vietnamese token/word grouping and normalization

- **A transcript, not whitespace, defines word boundaries.** A time-aligned
  token may be one syllable; several tokens may share a `word_id` to form a
  word. A manually segmented multi-syllable word may also be a single token.
- Text is normalized to **Unicode NFC** on load, and equality/type counting
  compares **casefolded** forms only — diacritics, tone marks, and the
  **`d` / `đ`** distinction are preserved in the text itself.
- Words group by non-null `word_id` within each utterance only; grouping
  never crosses utterance boundaries, and adjacent tokens are never compared
  across utterances.

## 3. Timing and annotation provenance

- Utterance and token times are finite and ordered; token times lie inside
  the owning utterance. Acoustic features analyze only the target speaker's
  aligned intervals.
- Every `AnnotationLayer` records `layer`, `source` (e.g. `human`, `chat`),
  and `confidence`; the extraction provenance carries the full annotation
  source list, plus SHA-256 hashes of the audio and of the transcript
  source file when the format records one (CHAT sets `source_sha256`).
- Document warnings (e.g. unsupported CHAT tiers) surface in the extraction
  `issues` table.

## 4. CHAT subset

Supported headers: `@Begin`, `@Languages` (exactly `vie`), `@Participants`
(three-letter codes), `@ID`, `@Media`, `@End`. Supported tiers: speaker
tiers (`*PAR:`), media bullets (`%xaud` with utterance or word time pairs),
`%mor` and `%gra` dependent tiers, and the common item codes: fillers
(`&x`), events (`&=x`), fragments (`x-`), retracing (`[/]`), revision
(`[//]`), and annotated errors (`[*]`). `%mor`/`%gra` must align one item
per content token.

**Unknown tiers.** Unsupported `@`/`%` lines are preserved verbatim in
`raw_tiers` and surfaced as a structured `UNSUPPORTED_CHAT_TIER` warning;
known data still round-trips semantically. Malformed input raises
`InvalidChatError` (`INVALID_CHAT`).

## 5. Workflows

- **Manual transcription:** transcribe and review the transcript (with
  word/syllable alignment and any linguistic annotations) in JSON v2 or
  CHAT, then run `say-features validate` before extraction.
- **External automation, then human review:** run Batchalign/ASR/taggers
  externally, export CHAT (or JSON), have a **human review and correct** the
  output, then let SAY consume the reviewed snapshot. SAY never merges
  transcript histories and never re-runs the automated tools.

## 6. Format conversion

```bash
say-features validate transcript.cha
say-features convert transcript.cha transcript.json
say-features convert transcript.json transcript.cha
```

Conversion detects the input format and writes by output extension; it
refuses to overwrite an existing file without `--force`. See
[feature-extraction.md](feature-extraction.md) for the extraction workflow,
the [README](../README.md) for installation, and
[migration-0.2.md](migration-0.2.md) for migrating existing 0.1 JSON v1
workflows.
