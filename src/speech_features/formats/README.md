# Document Formats (`src/speech_features/formats/`)

Serialization codecs for [`SpeechDocument`](../document.py). Each registered
format maps a name to an `{"encode", "decode"}` callable pair:

```python
encode(document: SpeechDocument) -> str
decode(text: str) -> SpeechDocument
```

The registry is a **static built-in mapping**. Third-party format discovery is
deliberately deferred until an actual third-party format exists.

---

## 1. Registry

```python
from speech_features.formats import FORMATS

FORMATS = {
    "chat": {"encode": encode_chat, "decode": decode_chat},
    "json": {"encode": encode_json, "decode": decode_json},
}
```

| Format | Module | Direction | Role |
|---|---|---|---|
| `json` | [`json.py`](json.py) | both | Lossless native representation (v2) plus a deterministic v1 migration reader |
| `chat` | [`chat.py`](chat.py) | both | The documented clinical CHAT **subset**, so SAY documents round-trip through human-readable transcripts |

The registry itself lives in [`__init__.py`](__init__.py), which imports both
codecs and exposes them as `FORMATS`.

---

## 2. JSON (`json.py`)

JSON **v2** is the lossless native form with stable fields — it is the format to
keep as the source of truth.

JSON **v1** (`"version": 1`) is accepted through a deterministic migration
reader. Migration assigns stable identifiers and never guesses word boundaries:

- utterances receive ids `u0001`, `u0002`, ...
- tokens receive ids `u0001_t0001`, ...
- each old word token becomes **exactly one** word token — it is never split on
  whitespace — unless the source already declares grouping via `word_id`.

| Public name | Signature |
|---|---|
| `encode_json` | `encode_json(document: SpeechDocument) -> str` |
| `decode_json` | `decode_json(text: str) -> SpeechDocument` |

---

## 3. CHAT (`chat.py`)

A codec for the documented **clinical CHAT subset** — not a CLAN clone.

**Written and read:** `@Begin`, `@Languages` (exactly `vie`), `@Participants`,
`@ID`, `@Media`, `@End`, speaker tiers, `%xaud` utterance and word media
bullets, `%mor`, `%gra`, and the common CHAT item codes — fillers `&x`, events
`&=x`, fragments `x-`, retracing `[/]`, revision `[//]`, error `[*]`.

**Deliberately not done here:** no Batchalign, ASR, diarization, or forced
alignment is ever imported or invoked, and only the current file snapshot is
parsed.

**Lossless-enough handling of the unknown:** any unsupported `@`/`%` line is
preserved verbatim in `raw_tiers` and surfaced as a structured
`UNSUPPORTED_CHAT_TIER` warning on `SpeechDocument.warnings`. Nothing is silently
dropped.

| Public name | Kind | Notes |
|---|---|---|
| `decode_chat` | function | `decode_chat(text: str, *, source: str = "", sha256: str \| None = None) -> SpeechDocument` |
| `encode_chat` | function | `encode_chat(document: SpeechDocument) -> str` |
| `InvalidChatError` | exception | Subclass of `InvalidDocumentError`; carries the stable code `INVALID_CHAT` on every malformed boundary case |
| `ChatTierWarning` | dataclass | Structured record of a preserved-but-unparsed tier |

---

## 4. Related

- [Document model](../document.py) — `SpeechDocument`, the object both codecs produce
- [Package overview](../README.md) — the full `speech_features` API and boundaries
- [Transcription pipeline](../../say_transcribe/README.md) — the draft `.cha` producer whose output feeds these codecs
