"""Clinical CHAT subset codec for :class:`SpeechDocument` (Task 3).

Reads and writes the documented clinical CHAT subset: ``@Begin``,
``@Languages`` (exactly ``vie``), ``@Participants``, ``@ID``, ``@Media``,
``@End``, speaker tiers, ``%xaud`` utterance and word media bullets,
``%mor``, ``%gra``, and the common CHAT item codes (fillers ``&x``, events
``&=x``, fragments ``x-``, retracing ``[/]``, revision ``[//]``, error
``[*]``).

No Batchalign, ASR, diarization, or forced alignment is ever imported or
invoked; only the current file snapshot is parsed. Unsupported ``@``/``%``
lines are preserved verbatim in ``raw_tiers`` and surfaced as a structured
``UNSUPPORTED_CHAT_TIER`` warning on ``SpeechDocument.warnings``. All
malformed boundary cases raise :class:`InvalidChatError` with the stable
code ``INVALID_CHAT``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..document import (
    AnnotationLayer,
    DocumentSpeaker,
    DocumentToken,
    DocumentUtterance,
    InvalidDocumentError,
    MediaRef,
    SpeechDocument,
    nfc,
)

UNSUPPORTED_CHAT_TIER = "UNSUPPORTED_CHAT_TIER"

_KNOWN_HEADERS = frozenset({"@Begin", "@End", "@Languages", "@Participants", "@ID", "@Media"})
_CONTENT_KINDS = frozenset({"word", "filler", "fragment"})


class InvalidChatError(InvalidDocumentError):
    """Raised when a CHAT file is malformed; carries the stable code
    ``INVALID_CHAT``."""

    code = "INVALID_CHAT"


@dataclass(frozen=True)
class ChatTierWarning:
    """Structured warning for one unsupported CHAT line, preserved verbatim."""

    code: str = UNSUPPORTED_CHAT_TIER
    tier: str = ""
    line: str = ""


def _invalid(message: str) -> InvalidChatError:
    return InvalidChatError(message)


def _item_kind(item: str) -> str:
    if item.startswith("&"):
        return "noise" if item.startswith("&=") else "filler"
    if item.endswith("-") and len(item) > 1:
        return "fragment"
    if item == "[/]":
        return "retracing"
    if item == "[//]":
        return "revision"
    if item == "[*]":
        return "error"
    return "word"


_UNTRANSCRIBED = frozenset({"xxx", "yyy", "www"})


def tier_roles(items: Sequence[str]) -> list[str]:
    """Role of each main-tier item for dependent-tier alignment.

    Follows chatter/Batchalign3 (validated with ``chatter validate``): ``%mor`` and
    ``%gra`` align to ``word`` items only (punctuation separators and the terminator
    are ``word``); ``%wor`` also takes ``filler`` (``&-um``) and ``retraced`` items
    (words before ``[/]``/``[//]``). ``untranscribed`` (``xxx``) and ``skip``
    (``&+``/``&~``/``&=``, ``(.)``, ``[: x]`` and other annotations) are in neither."""
    roles = ["word"] * len(items)
    in_annotation = False
    for i, item in enumerate(items):
        if in_annotation:
            roles[i] = "skip"
            in_annotation = not item.endswith("]")
        elif item.startswith("["):
            roles[i] = "skip"
            in_annotation = not item.endswith("]")
        elif re.fullmatch(r"\((?:\.+|\d+(?:\.\d+)?)\)", item):
            roles[i] = "skip"
        elif item.startswith("&-"):
            roles[i] = "filler"
        elif item.startswith("&") or (item.endswith("-") and len(item) > 1):
            roles[i] = "skip"
        elif item.strip(".,!?…").lower() in _UNTRANSCRIBED:
            roles[i] = "untranscribed"
    for i, item in enumerate(items):
        if item not in ("[/]", "[//]") or i == 0:
            continue
        start = i - 1
        while start >= 0 and roles[start] == "skip":
            start -= 1
        if start < 0:
            continue
        end = start + 1
        if items[start].endswith(">"):
            while start > 0 and not items[start].startswith("<"):
                start -= 1
            if not items[start].startswith("<"):
                start = i - 1
        for j in range(start, end):
            if roles[j] == "word":
                roles[j] = "retraced"
    return roles


def mor_projection(items: Sequence[str]) -> tuple[tuple[int, str], ...]:
    """Canonical corrected MOR surfaces with original main-tier source indices.

    Scope markup is removed only in this view. Corrections may contribute several
    words at one source index; the original main tier is never rewritten.
    """
    roles = tier_roles(items)
    projected: list[tuple[int, str]] = []
    i = 0
    while i < len(items):
        item = items[i]
        if item.startswith("[:"):
            annotation = [item[2:]]
            while not items[i].endswith("]") and i + 1 < len(items):
                i += 1
                annotation.append(items[i])
            replacement = " ".join(annotation).removesuffix("]").split()
            previous = i - len(annotation)
            while previous >= 0 and roles[previous] == "skip":
                previous -= 1
            start = previous
            if previous >= 0 and items[previous].endswith(">"):
                while start > 0 and not items[start].startswith("<"):
                    start -= 1
                if not items[start].startswith("<"):
                    start = previous
            if previous >= 0 and roles[previous] == "word":
                projected = [(j, text) for j, text in projected if j < start]
                projected.extend((start, text) for text in replacement)
        elif roles[i] == "word":
            surface = item.strip("<>")
            if surface:
                projected.append((i, surface))
        i += 1
    return tuple(projected)


def _split_header(line: str) -> tuple[str, str]:
    if ":" in line:
        name, _, rest = line.partition(":")
        return name, rest.strip()
    return line, ""


def _fmt(value: float) -> str:
    return f"{value:g}"


def decode_chat(text: str, *, source: str = "", sha256: str | None = None) -> SpeechDocument:
    """Parse a CHAT text snapshot into a :class:`SpeechDocument`.

    ``source`` records the origin and ``sha256`` the input hash; both are
    stored on the returned document. Raises :class:`InvalidChatError`
    (code ``INVALID_CHAT``) for malformed input.
    """
    if text.startswith("\ufeff"):
        text = text[1:]
    lines = [line.rstrip("\r\n") for line in text.splitlines() if line.strip()]

    # CLAN writes preamble headers (@UTF8, @Window, @TimeDuration, ...)
    # before @Begin; accept and ignore them.
    while lines and lines[0].startswith("@") and lines[0] != "@Begin":
        lines = lines[1:]

    if not lines or lines[0] != "@Begin":
        raise _invalid("CHAT file must start with @Begin")
    if lines.count("@Begin") != 1:
        raise _invalid("duplicate @Begin header")
    if lines[-1] != "@End" or lines.count("@End") != 1:
        raise _invalid("CHAT file must end with exactly one @End")

    language = "vie"
    media: list[MediaRef] = []
    participants: list[tuple[str, str]] = []
    id_info: dict[str, tuple[str, str]] = {}
    saw_participants = False
    saw_media = False
    raw_tiers: dict[str, list[str]] = {}
    warnings: list[ChatTierWarning] = []

    utterances: list[DocumentUtterance] = []
    annotations: dict[str, dict[str, str]] = {"wor": {}, "mor": {}, "gra": {}}

    current: dict | None = None
    current_bullets: list[tuple[str, list[str]]] = []

    def finish_utterance() -> None:
        nonlocal current
        if current is None:
            return
        word_pairs: list[tuple[float, float]] | None = None
        for role, parts in current_bullets:
            pairs = _parse_bullet(parts, current, role)
            if role == "utterance":
                current["start_s"], current["end_s"] = pairs[0]
            else:
                word_pairs = pairs
        if current["start_s"] is None:
            raise _invalid(f"speaker tier {current['tier']!r} has no media bullet")
        if word_pairs is not None:
            for token, (start, end) in zip(current["content"], word_pairs):
                if start < current["start_s"] or end > current["end_s"]:
                    raise _invalid(f"token {token.id} times outside utterance times")
                object.__setattr__(token, "start_s", start)
                object.__setattr__(token, "end_s", end)
        roles = tier_roles([t.text for t in current["tokens"]])
        for name, layer_key in (("%wor", "wor"), ("%mor", "mor"), ("%gra", "gra")):
            if name in current["tiers"]:
                if name == "%wor":
                    items = _chunk_wor(current["tiers"][name])
                else:
                    items = current["tiers"][name].split()
                projection = mor_projection([t.text for t in current["tokens"]])
                members = (
                    [current["tokens"][index] for index, _ in projection]
                    if name != "%wor" and len(projection) == len(items)
                    else _tier_members(name, current["tokens"], roles, len(items))
                )
                if name == "%wor" and len(items) != len(members):
                    raw_tiers.setdefault("__wor_" + current["id"], []).append("%wor:\t" + current["tiers"][name])
                    warnings.append(ChatTierWarning(code="UNTRUSTED_CHAT_WOR", tier="%wor"))
                    continue
                if len(items) != len(members):
                    raise _invalid(
                        f"incompatible {name} alignment: {len(items)} items for "
                        f"{len(members)} alignable tokens"
                    )
                if name == "%wor":
                    if not _apply_wor_timing(current, members, items, warnings):
                        raw_tiers.setdefault("__wor_" + current["id"], []).append("%wor:\t" + current["tiers"][name])
                        continue
                for token, item in zip(members, items):
                    previous = annotations[layer_key].get(token.id)
                    annotations[layer_key][token.id] = (previous + " " + item) if previous else item
        utterances.append(
            DocumentUtterance(
                id=current["id"],
                speaker_id=current["speaker"],
                start_s=current["start_s"],
                end_s=current["end_s"],
                tokens=current["tokens"],
            )
        )
        current = None
        current_bullets.clear()

    last_raw_name: str | None = None

    for line in lines[:-1]:
        if line.startswith("@"):
            name, rest = _split_header(line)
            if name == "@Begin":
                continue
            if name == "@Languages":
                if rest != "vie":
                    raise _invalid(f"@Languages must be exactly 'vie', got {rest!r}")
                continue
            if name == "@Participants":
                if saw_participants:
                    raise _invalid("duplicate @Participants header")
                saw_participants = True
                for part in rest.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    code, _, speaker_name = part.partition(" ")
                    if not re.fullmatch(r"[A-Z]{3}", code):
                        raise _invalid(f"malformed @Participants entry {part!r}")
                    participants.append((code, nfc(speaker_name.strip())))
                if not participants:
                    raise _invalid("@Participants must list at least one speaker")
                continue
            if name == "@ID":
                fields = [f.strip() for f in rest.split("|")]
                if len(fields) < 3 or not fields[1]:
                    raise _invalid(f"malformed @ID header {line!r}")
                if fields[0] != "vie":
                    raise _invalid(f"@ID language must be 'vie', got {fields[0]!r}")
                # Support standard TalkBank (@ID: lang|corpus|CODE|...) and legacy (@ID: lang|CODE|...)
                participant_codes = {c for c, _ in participants}
                if len(fields) >= 4 and fields[2] in participant_codes:
                    spk_code = fields[2]
                    spk_name = fields[7] if len(fields) > 7 and fields[7] else fields[3]
                    spk_role = fields[8] if len(fields) > 8 and fields[8] else fields[3]
                else:
                    spk_code = fields[1]
                    spk_name = fields[2] if len(fields) > 2 else ""
                    spk_role = fields[3] if len(fields) > 3 else ""
                id_info[spk_code] = (nfc(spk_name), spk_role)
                continue
            if name == "@Media":
                saw_media = True
                # Accept both chat.py's "path | kind" and CHAT/Delaware's "name, kind";
                # the comma form is what real TalkBank files use.
                sep = "|" if "|" in rest else ","
                fields = [f.strip() for f in rest.split(sep)]
                if not fields[0]:
                    raise _invalid(f"malformed @Media header {line!r}")
                path = fields[0]
                kind = fields[1] if len(fields) > 1 and fields[1] else "audio"
                media.append(MediaRef(id=path, kind=kind, path=path))
                continue
            finish_utterance()
            raw_tiers.setdefault(name, []).append(line)
            warnings.append(ChatTierWarning(tier=name, line=line))
            last_raw_name = name
            continue
        if line.startswith("%"):
            name, rest = _split_header(line)
            if current is None:
                raise _invalid(f"dependent tier {name} without a speaker tier")
            if name == "%xaud":
                parts = rest.split()
                if len(parts) < 2:
                    raise _invalid(f"malformed media bullet {line!r}")
                role = "utterance" if len(parts) == 3 else "word"
                if any(role == r for r, _ in current_bullets):
                    raise _invalid(
                        f"duplicate {role} media bullet for speaker tier {current['tier']!r}"
                    )
                current_bullets.append((role, parts))
                continue
            if name in ("%wor", "%mor", "%gra"):
                if name in current["tiers"]:
                    raise _invalid(f"duplicate {name} tier for speaker tier {current['tier']!r}")
                current["tiers"][name] = rest
                continue
            raw_tiers.setdefault(name, []).append(line)
            warnings.append(ChatTierWarning(tier=name, line=line))
            last_raw_name = name
            continue
        if line.startswith("*"):
            finish_utterance()
            if ":" not in line:
                raise _invalid(f"malformed speaker tier {line!r}")
            tier, _, content = line.partition(":")
            code = tier[1:]
            if code and code not in {c for c, _ in participants}:
                raise _invalid(f"speaker tier references unknown speaker {code!r}")
            if not code:
                # Recoverable source slip: an unlabeled "*:" tier keeps its
                # text and times but carries no speaker attribution.
                warnings.append(ChatTierWarning(tier="*", line="unlabeled speaker tier"))

            # Extract optional inline media bullet: \x15start_end\x15 or •start_end•
            inline_start_s = None
            inline_end_s = None
            bullet_match = re.search(r"[\x15•](\d+)_(\d+)[\x15•]", content)
            if bullet_match:
                inline_start_s = int(bullet_match.group(1)) / 1000.0
                inline_end_s = int(bullet_match.group(2)) / 1000.0
                content = content[: bullet_match.start()] + content[bullet_match.end() :]

            items = content.strip().split()
            tokens = []
            for i, item in enumerate(items):
                tokens.append(
                    DocumentToken(
                        id=f"u{len(utterances) + 1:04d}_t{i + 1:04d}",
                        text=nfc(item),
                        kind=_item_kind(item),
                    )
                )
            current = {
                "id": f"u{len(utterances) + 1:04d}",
                "tier": tier,
                "speaker": code,
                "tokens": tokens,
                "content": [t for t in tokens if t.kind in _CONTENT_KINDS],
                "tiers": {},
                "start_s": inline_start_s,
                "end_s": inline_end_s,
            }
            continue
        if line[0].isspace() and current is not None:
            # Wrapped utterance: indented lines continue the open speaker tier
            # and may carry its timing bullet (CHAT wraps long utterances).
            bullet_match = re.search(r"[\x15•](\d+)_(\d+)[\x15•]", line)
            content = line
            if bullet_match:
                current["start_s"] = int(bullet_match.group(1)) / 1000.0
                current["end_s"] = int(bullet_match.group(2)) / 1000.0
                content = line[: bullet_match.start()] + line[bullet_match.end() :]
            for item in content.strip().split():
                token = DocumentToken(
                    id=f"u{len(utterances) + 1:04d}_t{len(current['tokens']) + 1:04d}",
                    text=nfc(item),
                    kind=_item_kind(item),
                )
                current["tokens"].append(token)
                if token.kind in _CONTENT_KINDS:
                    current["content"].append(token)
            continue
        if line[0].isspace() and last_raw_name is not None:
            # Header continuation lines (e.g. a wrapped @Comment) are indented.
            raw_tiers[last_raw_name][-1] += " " + line.strip()
            continue
        raise _invalid(f"unexpected line {line!r}")

    finish_utterance()

    if not saw_participants:
        raise _invalid("missing @Participants header")
    if not saw_media:
        raise _invalid("missing @Media header")
    unknown_ids = set(id_info) - {c for c, _ in participants}
    if unknown_ids:
        raise _invalid(f"@ID references unknown speakers: {sorted(unknown_ids)}")

    speakers = []
    for code, name in participants:
        id_name, role = id_info.get(code, (name, ""))
        speakers.append(DocumentSpeaker(id=code, name=nfc(id_name or name), role=role))

    token_ids = {t.id for u in utterances for t in u.tokens}
    for annotation in annotations.values():
        unknown = set(annotation) - token_ids
        if unknown:
            raise _invalid(f"annotation references unknown token ids: {sorted(unknown)}")

    layers = tuple(
        AnnotationLayer(layer=layer, source="chat", values=values)
        for layer, values in annotations.items()
        if values
    )
    return SpeechDocument(
        document_id=media[0].path or media[0].id,
        language=language,
        media=media,
        speakers=speakers,
        utterances=utterances,
        annotations=layers,
        raw_tiers={name: "\n".join(lines) for name, lines in raw_tiers.items()},
        source=source,
        source_sha256=sha256,
        warnings=tuple(warnings),
    )


def _tier_members(
    tier: str, tokens: Sequence[DocumentToken], roles: Sequence[str], item_count: int
) -> list[DocumentToken]:
    """Tokens a dependent tier aligns to, the first candidate whose size matches.

    chatter/Batchalign3 rules come first: ``%mor``/``%gra`` take words only; ``%wor``
    also takes fillers and retraced words. Then ``%wor`` with `xxx` entries (older
    drafts wrote them), then the legacy convention (every word, filler and fragment
    token, as CLAN-style files and earlier drafts wrote them)."""
    wor = ("word", "filler", "retraced")
    candidates = (
        [("word",)]
        if tier != "%wor"
        else [wor, (*wor, "untranscribed")]
    )
    members = [[t for t, r in zip(tokens, roles) if r in kinds] for kinds in candidates]
    members.append([t for t in tokens if t.kind in _CONTENT_KINDS])
    return next((m for m in members if len(m) == item_count), members[0])


_WOR_CHUNK = re.compile(r"\S+(?:\s+[\x15•]\d+_\d+[\x15•])?")


def _chunk_wor(rest: str) -> list[str]:
    """Split %wor into one chunk per content token: `token` or `token \\x15start_end\\x15`."""
    return _WOR_CHUNK.findall(rest)


def _wor_surface(text: str) -> str:
    surface = text.strip("<>")
    return surface[2:] if surface.startswith("&-") else surface


def _apply_wor_timing(current: dict, members: Sequence[DocumentToken], items: Sequence[str],
                      warnings: list[ChatTierWarning]) -> bool:
    """Admit timing only after lexical identity and interval validation."""
    pairs: list[tuple[DocumentToken, float, float]] = []
    valid = True
    previous_end = current["start_s"]
    for token, item in zip(members, items):
        display = item.split()[0]
        if _wor_surface(token.text) != _wor_surface(display):
            valid = False
        match = re.search(r"[\x15•](\d+)_(\d+)[\x15•]", item)
        if match:
            start, end = (int(value) / 1000 for value in match.groups())
            if not (previous_end <= start < end <= current["end_s"]):
                valid = False
            previous_end = end
            pairs.append((token, start, end))
        elif any(c.isalnum() for c in display) and any("\x15" in entry or "•" in entry for entry in items):
            valid = False
    if not valid:
        warnings.append(ChatTierWarning(code="UNTRUSTED_CHAT_WOR", tier="%wor"))
        return False
    for token, start, end in pairs:
        object.__setattr__(token, "start_s", start)
        object.__setattr__(token, "end_s", end)
    return True


def _parse_bullet(parts: list[str], current: dict, role: str) -> list[tuple[float, float]]:
    # Trailing numeric run = the timing payload; a media path may contain spaces
    # ("p001, audio"), so anything before the floats is path text.
    floats: list[float] = []
    for token in reversed(parts[1:]):
        try:
            floats.append(float(token))
        except ValueError:
            break
    floats.reverse()
    if not floats:
        raise _invalid(f"malformed media bullet {parts!r}")
    if role == "utterance":
        if len(floats) != 2 or floats[1] < floats[0]:
            raise _invalid(f"malformed utterance bullet {parts!r}")
        return [(floats[0], floats[1])]
    content = current["content"]
    if len(floats) != 2 * len(content):
        raise _invalid(f"word bullet {parts!r} incompatible with {len(content)} content tokens")
    pairs = [(floats[i], floats[i + 1]) for i in range(0, len(floats), 2)]
    if any(end < start for start, end in pairs):
        raise _invalid(f"word bullet end before start: {parts!r}")
    return pairs


def encode_chat(document: SpeechDocument) -> str:
    """Serialise a document as the documented clinical CHAT subset."""
    if not isinstance(document, SpeechDocument):
        raise _invalid(f"document must be a SpeechDocument, got {type(document).__name__}")
    if not document.media:
        raise _invalid("cannot encode a document without @Media to CHAT")
    media_path = document.media[0].path or document.media[0].id

    lines = ["@Begin", f"@Languages:\t{document.language}"]
    lines.append("@Participants:\t" + ", ".join(f"{s.id} {s.name}" for s in document.speakers))
    for speaker in document.speakers:
        lines.append(f"@ID:\t{document.language}|{speaker.id}|{speaker.name}|{speaker.role}|")
    for ref in document.media:
        lines.append(f"@Media:\t{ref.path or ref.id} | {ref.kind}")

    layers = {a.layer: a.values for a in document.annotations}
    for utterance in document.utterances:
        lines.append(f"*{utterance.speaker_id}:\t" + " ".join(t.text for t in utterance.tokens))
        lines.append(f"%xaud:\t{media_path} {_fmt(utterance.start_s)} {_fmt(utterance.end_s)}")
        content = [t for t in utterance.tokens if t.kind in _CONTENT_KINDS]
        # a single-token word bullet would be indistinguishable from an
        # utterance bullet on reload; skip it rather than misround-trip
        if len(content) >= 2 and all(
            t.start_s is not None and t.end_s is not None for t in content
        ):
            pairs = " ".join(f"{_fmt(t.start_s)} {_fmt(t.end_s)}" for t in content)
            lines.append(f"%xaud:\t{media_path} {pairs}")
        roles = tier_roles([t.text for t in utterance.tokens])
        raw_wor = document.raw_tiers.get("__wor_" + utterance.id)
        if raw_wor:
            lines.append(raw_wor)
        for name in ("%wor", "%mor", "%gra"):
            values = layers.get(name[1:], {})
            if name != "%wor":
                canonical = list(dict.fromkeys(index for index, _ in mor_projection(
                    [t.text for t in utterance.tokens]
                )))
                canonical_members = [utterance.tokens[index] for index in canonical]
                present_ids = {t.id for t in utterance.tokens if t.id in values}
                if canonical_members and present_ids == {t.id for t in canonical_members}:
                    lines.append(name + ":\t" + " ".join(values[t.id] for t in canonical_members))
                    continue
            members = _tier_members(
                name,
                utterance.tokens,
                roles,
                sum(t.id in values for t in utterance.tokens),
            )
            if members and all(t.id in values for t in members):
                lines.append(name + ":\t" + " ".join(values[t.id] for t in members))
    for name, text in document.raw_tiers.items():
        if name in _KNOWN_HEADERS or name.startswith("__wor_"):
            continue
        for tier_line in text.split("\n"):
            lines.append(tier_line)
    lines.append("@End")
    return "\n".join(lines) + "\n"
