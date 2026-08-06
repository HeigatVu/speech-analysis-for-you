"""Population-neutral speech document model (Task 2).

``SpeechDocument`` JSON version 2 is the lossless native representation:
document/media identifiers, ISO 639-3 language (``vie``), speakers,
utterances with stable token IDs, optional time alignment, explicit word
grouping (shared ``word_id``), per-token language for code switching,
linguistic annotation layers with confidence, and preserved raw CHAT tiers.

The dataclasses here are deliberately named ``DocumentSpeaker``,
``DocumentUtterance``, ``DocumentToken``, and ``MediaRef`` so the deprecated
AD ``Token``/``Utterance`` contracts remain available without ambiguous root
exports. Only stdlib primitives are used.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .schema import MissingInputError, nfc

LANGUAGE = "vie"


class InvalidDocumentError(ValueError):
    """Raised when a speech document JSON is malformed or invalid."""

    code = "INVALID_DOCUMENT"


# ---------------------------------------------------------------------------
# Frozen value types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MediaRef:
    """One media attachment (typically an audio file) of the document."""

    id: str
    kind: str = "audio"
    path: str = ""
    sha256: str | None = None


@dataclass(frozen=True)
class DocumentSpeaker:
    """One documented speaker. ``role`` is a free string (e.g. participant)."""

    id: str
    name: str = ""
    role: str = ""


@dataclass(frozen=True)
class DocumentToken:
    """One token. ``word_id`` groups several tokens into one word; ``language``
    records code switching; ``dep_head`` references another token id."""

    id: str
    text: str
    kind: str = "word"
    start_s: float | None = None
    end_s: float | None = None
    word_id: str | None = None
    language: str | None = None
    dep_head: str | None = None
    dep_rel: str | None = None


@dataclass(frozen=True)
class DocumentUtterance:
    """One utterance with stable id and ordered tokens."""

    id: str
    speaker_id: str
    start_s: float
    end_s: float
    tokens: tuple = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tokens", tuple(self.tokens))


@dataclass(frozen=True)
class AnnotationLayer:
    """One annotation layer: ``values`` maps token ids to values, ``source``
    records provenance, ``confidence`` is in [0, 1]."""

    layer: str
    source: str = ""
    confidence: float = 1.0
    values: Mapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


@dataclass(frozen=True)
class SpeechDocument:
    """Validated speech document. All collection fields are deeply immutable."""

    document_id: str
    language: str = LANGUAGE
    media: tuple = field(default_factory=tuple)
    speakers: tuple = field(default_factory=tuple)
    utterances: tuple = field(default_factory=tuple)
    annotations: tuple = field(default_factory=tuple)
    raw_tiers: Mapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "media", tuple(self.media))
        object.__setattr__(self, "speakers", tuple(self.speakers))
        object.__setattr__(self, "utterances", tuple(self.utterances))
        object.__setattr__(self, "annotations", tuple(self.annotations))
        object.__setattr__(self, "raw_tiers", MappingProxyType(dict(self.raw_tiers)))


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _require_str(data: dict, key: str, *, allow_empty: bool = True, default=None) -> str | None:
    value = data.get(key, default)
    if value is None:
        return None
    if not isinstance(value, str) or (not allow_empty and not value):
        raise InvalidDocumentError(f"{key} must be a non-empty string, got {value!r}")
    return value


def _require_number(data: dict, key: str, *, default=None) -> float | None:
    value = data.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidDocumentError(f"{key} must be numeric, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise InvalidDocumentError(f"{key} must be finite, got {value!r}")
    return value


def _parse_media(media: list) -> tuple[MediaRef, ...]:
    ids = set()
    parsed = []
    for entry in media:
        if not isinstance(entry, dict):
            raise InvalidDocumentError("media entries must be objects")
        mid = _require_str(entry, "id", allow_empty=False)
        if mid in ids:
            raise InvalidDocumentError(f"duplicate media id: {mid!r}")
        ids.add(mid)
        sha = entry.get("sha256")
        if sha is not None and not isinstance(sha, str):
            raise InvalidDocumentError(f"media {mid} sha256 must be a string")
        parsed.append(
            MediaRef(
                id=mid,
                kind=_require_str(entry, "kind", default="audio") or "audio",
                path=_require_str(entry, "path", default="") or "",
                sha256=sha,
            )
        )
    return tuple(parsed)


def _parse_speakers(speakers: list) -> tuple[DocumentSpeaker, ...]:
    ids = set()
    parsed = []
    for entry in speakers:
        if not isinstance(entry, dict):
            raise InvalidDocumentError("speaker entries must be objects")
        sid = _require_str(entry, "id", allow_empty=False)
        if sid in ids:
            raise InvalidDocumentError(f"duplicate speaker id: {sid!r}")
        ids.add(sid)
        parsed.append(
            DocumentSpeaker(
                id=sid,
                name=nfc(_require_str(entry, "name", default="") or ""),
                role=_require_str(entry, "role", default="") or "",
            )
        )
    return tuple(parsed)


def _parse_tokens(tokens: list, utterance: dict, known_ids: set) -> tuple[DocumentToken, ...]:
    utt_start = float(utterance["start_s"])
    utt_end = float(utterance["end_s"])
    parsed = []
    for entry in tokens:
        if not isinstance(entry, dict):
            raise InvalidDocumentError("token entries must be objects")
        tid = _require_str(entry, "id", allow_empty=False)
        if tid in known_ids:
            raise InvalidDocumentError(f"duplicate token id: {tid!r}")
        known_ids.add(tid)
        start = _require_number(entry, "start_s")
        end = _require_number(entry, "end_s")
        if start is not None and end is not None and end < start:
            raise InvalidDocumentError(f"token {tid} end before start")
        if start is not None and start < utt_start:
            raise InvalidDocumentError(f"token {tid} start_s outside utterance times")
        if end is not None and end > utt_end:
            raise InvalidDocumentError(f"token {tid} end_s outside utterance times")
        parsed.append(
            DocumentToken(
                id=tid,
                text=nfc(_require_str(entry, "text", default="") or ""),
                kind=_require_str(entry, "kind", default="word") or "word",
                start_s=start,
                end_s=end,
                word_id=_require_str(entry, "word_id"),
                language=_require_str(entry, "language"),
                dep_head=_require_str(entry, "dep_head"),
                dep_rel=_require_str(entry, "dep_rel"),
            )
        )
    return tuple(parsed)


def _parse_utterances(
    utterances: list, speaker_ids: set, token_ids: set
) -> tuple[DocumentUtterance, ...]:
    ids = set()
    parsed = []
    for entry in utterances:
        if not isinstance(entry, dict):
            raise InvalidDocumentError("utterance entries must be objects")
        uid = _require_str(entry, "id", allow_empty=False)
        if uid in ids:
            raise InvalidDocumentError(f"duplicate utterance id: {uid!r}")
        ids.add(uid)
        speaker_id = _require_str(entry, "speaker_id", allow_empty=False)
        if speaker_id not in speaker_ids:
            raise InvalidDocumentError(f"utterance {uid} references unknown speaker {speaker_id!r}")
        start = _require_number(entry, "start_s")
        end = _require_number(entry, "end_s")
        if end < start:
            raise InvalidDocumentError(f"utterance {uid} end before start")
        tokens = _parse_tokens(
            entry.get("tokens", []) if isinstance(entry.get("tokens"), list) else [],
            entry,
            token_ids,
        )
        parsed.append(
            DocumentUtterance(
                id=uid, speaker_id=speaker_id, start_s=start, end_s=end, tokens=tokens
            )
        )
    return tuple(parsed)


def _parse_annotations(annotations: list) -> tuple[AnnotationLayer, ...]:
    parsed = []
    for entry in annotations:
        if not isinstance(entry, dict):
            raise InvalidDocumentError("annotation entries must be objects")
        layer = _require_str(entry, "layer", allow_empty=False)
        confidence = _require_number(entry, "confidence", default=1.0)
        if not 0.0 <= confidence <= 1.0:
            raise InvalidDocumentError(f"annotation {layer} confidence must be in [0, 1]")
        values = entry.get("values", {})
        if not isinstance(values, dict):
            raise InvalidDocumentError(f"annotation {layer} values must be an object")
        for tid, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                raise InvalidDocumentError(
                    f"annotation {layer} value for {tid!r} must be str or number"
                )
        parsed.append(
            AnnotationLayer(
                layer=layer,
                source=_require_str(entry, "source", default="") or "",
                confidence=confidence,
                values=values,
            )
        )
    return tuple(parsed)


def validate_document(data: dict) -> SpeechDocument:
    """Validate a JSON v2 document mapping and return a frozen
    :class:`SpeechDocument` with deeply immutable collection fields."""
    if not isinstance(data, dict):
        raise InvalidDocumentError(f"document must be an object, got {type(data).__name__}")
    if data.get("version") != 2:
        raise InvalidDocumentError("document must have version == 2")
    document_id = _require_str(data, "document_id", allow_empty=False)
    if document_id is None:
        raise InvalidDocumentError("document_id must be a non-empty string")
    language = _require_str(data, "language", default=LANGUAGE)
    if language != LANGUAGE:
        raise InvalidDocumentError(
            f"document language must be exactly {LANGUAGE!r}, got {language!r}"
        )

    raw_media = data.get("media", [])
    raw_speakers = data.get("speakers", [])
    raw_utterances = data.get("utterances", [])
    for name, raw in (
        ("media", raw_media),
        ("speakers", raw_speakers),
        ("utterances", raw_utterances),
    ):
        if not isinstance(raw, list):
            raise InvalidDocumentError(f"document {name} must be a list")

    token_ids: set = set()
    media = _parse_media(raw_media)
    speakers = _parse_speakers(raw_speakers)
    speaker_ids = {s.id for s in speakers}
    utterances = _parse_utterances(raw_utterances, speaker_ids, token_ids)

    for token in (t for u in utterances for t in u.tokens):
        if token.dep_head is not None and token.dep_head not in token_ids:
            raise InvalidDocumentError(
                f"token {token.id} dep_head references unknown token {token.dep_head!r}"
            )

    raw_annotations = data.get("annotations", [])
    if not isinstance(raw_annotations, list):
        raise InvalidDocumentError("document annotations must be a list")
    annotations = _parse_annotations(raw_annotations)

    raw_tiers = data.get("raw_tiers", {})
    if not isinstance(raw_tiers, dict):
        raise InvalidDocumentError("document raw_tiers must be an object")
    for name, text in raw_tiers.items():
        if not isinstance(name, str) or not isinstance(text, str):
            raise InvalidDocumentError("raw_tiers entries must map tier name to text")

    return SpeechDocument(
        document_id=document_id,
        language=language,
        media=media,
        speakers=speakers,
        utterances=utterances,
        annotations=annotations,
        raw_tiers=raw_tiers,
    )


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def _detect_format(path: Path) -> str:
    if path.suffix.lower() == ".json":
        return "json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        raise InvalidDocumentError(f"cannot detect format of {path}; pass format=")
    if isinstance(data, dict) and data.get("version") in (1, 2):
        return "json"
    raise InvalidDocumentError(f"cannot detect format of {path}; pass format=")


def load_document(path, *, format=None) -> SpeechDocument:
    """Load a speech document, detecting the format by extension or content
    when ``format`` is None. JSON v1 is migrated deterministically to v2."""
    from .formats import FORMATS  # deferred: formats/json imports this module

    p = Path(path)
    if not p.is_file():
        raise MissingInputError(f"document file not found: {p}")
    fmt = format or _detect_format(p)
    if fmt not in FORMATS:
        raise InvalidDocumentError(f"unknown document format: {fmt!r}")
    return FORMATS[fmt]["decode"](p.read_text(encoding="utf-8"))


def save_document(document: SpeechDocument, path, *, format=None, force=False) -> None:
    """Serialise ``document`` as JSON v2. Refuses to overwrite an existing
    file unless ``force`` is True."""
    from .formats import FORMATS  # deferred: formats/json imports this module

    if not isinstance(document, SpeechDocument):
        raise InvalidDocumentError(
            f"document must be a SpeechDocument, got {type(document).__name__}"
        )
    p = Path(path)
    fmt = format or ("json" if p.suffix.lower() == ".json" else None)
    if fmt is None:
        raise InvalidDocumentError(f"cannot infer format from extension {p.suffix!r}; pass format=")
    if fmt not in FORMATS:
        raise InvalidDocumentError(f"unknown document format: {fmt!r}")
    if p.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing file: {p} (pass force=True)")
    p.write_text(FORMATS[fmt]["encode"](document), encoding="utf-8")
