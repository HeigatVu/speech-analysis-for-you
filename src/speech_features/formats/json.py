"""JSON v1/v2 codec for :class:`SpeechDocument` (Task 2).

JSON v2 is the lossless native representation with stable fields. JSON v1
(``version == 1``) is accepted through a deterministic migration reader:
utterances receive ids ``u0001...``, tokens ``u0001_t0001...``, and each old
word token becomes exactly one word token (never split on whitespace) unless
the source already declares grouping via ``word_id``.
"""

from __future__ import annotations

import json

from ..document import (
    DocumentToken,
    InvalidDocumentError,
    SpeechDocument,
    validate_document,
)


def _token_to_dict(token: DocumentToken) -> dict:
    return {
        "id": token.id,
        "text": token.text,
        "kind": token.kind,
        "start_s": token.start_s,
        "end_s": token.end_s,
        "word_id": token.word_id,
        "language": token.language,
        "dep_head": token.dep_head,
        "dep_rel": token.dep_rel,
    }


def _document_to_dict(document: SpeechDocument) -> dict:
    return {
        "version": 2,
        "document_id": document.document_id,
        "language": document.language,
        "media": [
            {"id": m.id, "kind": m.kind, "path": m.path, "sha256": m.sha256} for m in document.media
        ],
        "speakers": [{"id": s.id, "name": s.name, "role": s.role} for s in document.speakers],
        "utterances": [
            {
                "id": u.id,
                "speaker_id": u.speaker_id,
                "start_s": u.start_s,
                "end_s": u.end_s,
                "tokens": [_token_to_dict(t) for t in u.tokens],
            }
            for u in document.utterances
        ],
        "annotations": [
            {
                "layer": a.layer,
                "source": a.source,
                "confidence": a.confidence,
                "values": dict(a.values),
            }
            for a in document.annotations
        ],
        "raw_tiers": dict(document.raw_tiers),
    }


def encode_json(document: SpeechDocument) -> str:
    """Serialise a document as JSON v2 with a stable field order."""
    return json.dumps(_document_to_dict(document), ensure_ascii=False, indent=2)


def decode_json(text: str) -> SpeechDocument:
    """Parse JSON v2, or migrate JSON v1 deterministically to v2."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidDocumentError(f"document is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise InvalidDocumentError("document must be a JSON object")
    version = data.get("version")
    if version == 2:
        return validate_document(data)
    if version == 1:
        return validate_document(_migrate_v1(data))
    raise InvalidDocumentError(f"unsupported document version: {version!r}")


def _migrate_v1(data: dict) -> dict:
    """Translate a JSON v1 transcript mapping into a JSON v2 mapping.

    Deterministic positional ids: ``u0001...`` for utterances and
    ``u0001_t0001...`` for tokens. Each v1 token becomes exactly one v2
    token with its text preserved verbatim; a ``word_id`` already declared on
    a source token is carried through.
    """
    utterances_v1 = data.get("utterances")
    if not isinstance(utterances_v1, list):
        raise InvalidDocumentError("v1 document must contain an 'utterances' list")

    speakers: dict[str, str] = {}
    utterances_v2 = []
    for i, utt in enumerate(utterances_v1):
        if not isinstance(utt, dict):
            raise InvalidDocumentError(f"utterance {i} is not an object")
        speaker = utt.get("speaker")
        if not isinstance(speaker, str) or not speaker:
            raise InvalidDocumentError(f"utterance {i} missing speaker")
        speakers.setdefault(speaker, speaker)

        uid = f"u{i + 1:04d}"
        raw_tokens = utt.get("tokens", [])
        if not isinstance(raw_tokens, list):
            raise InvalidDocumentError(f"utterance {uid} must contain a 'tokens' list")
        tokens = []
        for j, tok in enumerate(raw_tokens):
            if not isinstance(tok, dict):
                raise InvalidDocumentError(f"utterance {uid} token {j} is not an object")
            tokens.append(
                {
                    "id": f"{uid}_t{j + 1:04d}",
                    "text": tok.get("text"),
                    "kind": tok.get("kind", "word"),
                    "start_s": tok.get("start_s"),
                    "end_s": tok.get("end_s"),
                    "word_id": tok.get("word_id"),
                    "language": tok.get("language"),
                    "dep_head": tok.get("dep_head"),
                    "dep_rel": tok.get("dep_rel"),
                }
            )
        utterances_v2.append(
            {
                "id": uid,
                "speaker_id": speaker,
                "start_s": utt.get("start_s"),
                "end_s": utt.get("end_s"),
                "tokens": tokens,
            }
        )

    return {
        "version": 2,
        "document_id": data.get("transcript_id", "document"),
        "language": "vie",
        "media": [],
        "speakers": [{"id": sid, "name": name, "role": ""} for sid, name in speakers.items()],
        "utterances": utterances_v2,
        "annotations": [],
        "raw_tiers": {},
    }
