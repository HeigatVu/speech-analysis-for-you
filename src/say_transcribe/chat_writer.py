from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from speech_features.formats.chat import tier_roles

from say_transcribe.morphosyntax import GraItem, MorItem, UtteranceMorphosyntax
from say_transcribe.word_grouping import GroupedWord


@dataclass(frozen=True)
class UtteranceRecord:
    speaker: str  # "PAR" or "INV"
    start_ms: int | None
    end_ms: int | None
    text: str
    words: tuple[GroupedWord, ...]
    morphosyntax: UtteranceMorphosyntax | None


# (speaker code, role) declared in every generated transcript's header.
PARTICIPANTS: tuple[tuple[str, str], ...] = (
    ("PAR", "Participant"),
    ("INV", "Investigator"),
)


def _has_timing(utterances: Sequence[UtteranceRecord]) -> bool:
    """chatter (E544) requires `unlinked` on @Media when nothing carries timing."""
    return any(
        (utt.start_ms is not None and utt.end_ms is not None)
        or any(w.start_ms is not None for w in utt.words)
        for utt in utterances
    )


def format_chat_session(
    session_id: str,
    source_sha256: str,
    utterances: Sequence[UtteranceRecord],
    *,
    diarization_available: bool = True,
) -> str:
    """Format a session's utterances into a Delaware-shaped CHAT string."""
    lines = [
        "@UTF8",
        "@Begin",
        "@Languages:\tvie",
        "@Participants:\t" + ", ".join(f"{code} {role}" for code, role in PARTICIPANTS),
        *(f"@ID:\tvie|corpus|{code}|||||{role}|||" for code, role in PARTICIPANTS),
        f"@Media:\t{session_id}, audio{'' if _has_timing(utterances) else ', unlinked'}",
        f"@Comment:\tsource_sha256 {source_sha256}",
        (
            "@Comment:\tspeaker labels draft, auto-diarized; review before use"
            if diarization_available
            else "@Comment:\tspeaker labels unavailable; defaulted to PAR; review before use"
        ),
    ]

    if any(utt.start_ms is None or utt.end_ms is None for utt in utterances):
        lines.append("@Comment:\tutterance timing incomplete; align before timing-based analysis")

    for utt in utterances:
        tokens = [w.word for w in utt.words] if utt.words else [t for t in utt.text.strip().split() if t]
        if not tokens:
            continue

        # Enforce utterance-final punctuation; the appended terminator also lands in
        # %mor/%gra as (n+1)|root|PUNCT so tier item counts always match main tokens.
        if tokens[-1] not in {".", "?", "!", "...", "…"}:
            tokens.append(".")
            if utt.morphosyntax is not None and utt.morphosyntax.gra_items:
                gra = utt.morphosyntax.gra_items
                root = next((g.index for g in gra if g.head == 0), 1)
                utt = replace(
                    utt,
                    morphosyntax=UtteranceMorphosyntax(
                        mor_items=(*utt.morphosyntax.mor_items, MorItem(pos="", lemma=".")),
                        gra_items=(*gra, GraItem(index=len(gra) + 1, head=root, rel="PUNCT")),
                    ),
                )

        # Main tier line
        main_text = " ".join(tokens)
        timing = ""
        if utt.start_ms is not None and utt.end_ms is not None:
            timing = f" \x15{utt.start_ms}_{utt.end_ms}\x15"  # chatter rejects a tab here
        lines.append(f"*{utt.speaker}:\t{main_text}{timing}")

        roles = tier_roles(tokens)

        # %wor tier: words, fillers and retraced words (brackets stripped); not `xxx`
        has_timing = any(w.start_ms is not None for w in utt.words)
        if has_timing:
            wor_items: list[str] = []
            for i, (token, role) in enumerate(zip(tokens, roles)):
                if role not in ("word", "filler", "retraced"):
                    continue
                if role == "retraced":
                    token = token.removeprefix("<").removesuffix(">")
                word = utt.words[i] if i < len(utt.words) else None
                if word is not None and word.start_ms is not None and word.end_ms is not None:
                    wor_items.append(f"{token} \x15{word.start_ms}_{word.end_ms}\x15")
                else:
                    wor_items.append(token)
            lines.append(f"%wor:\t{' '.join(wor_items)}")

        # %mor and %gra tiers (chatter: none when no word is analysable)
        analysable = any(
            role == "word" and any(c.isalnum() for c in token)
            for token, role in zip(tokens, roles)
        )
        if utt.morphosyntax is not None and analysable:
            lines.append(utt.morphosyntax.mor_line())
            lines.append(utt.morphosyntax.gra_line())

    lines.append("@End\n")
    return "\n".join(lines)


def write_chat_file(
    output_path: Path,
    session_id: str,
    source_sha256: str,
    utterances: Sequence[UtteranceRecord],
    *,
    diarization_available: bool = True,
) -> Path:
    """Write one .cha file for a session."""
    content = format_chat_session(
        session_id=session_id,
        source_sha256=source_sha256,
        utterances=utterances,
        diarization_available=diarization_available,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite a manual transcript or an earlier automatic version (SPEC §8).
    with open(output_path, "x", encoding="utf-8") as f:
        f.write(content)
    return output_path
