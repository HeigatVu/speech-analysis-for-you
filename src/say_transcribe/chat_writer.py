from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

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
        "@Participants:\tPAR Participant, INV Investigator",
        "@ID:\tvie|corpus|PAR|||||Participant|||",
        "@ID:\tvie|corpus|INV|||||Investigator|||",
        f"@Media:\t{session_id}, audio",
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
            timing = f"\t\x15{utt.start_ms}_{utt.end_ms}\x15"
        lines.append(f"*{utt.speaker}:\t{main_text}{timing}")

        # %wor tier
        has_timing = any(w.start_ms is not None for w in utt.words)
        if has_timing:
            wor_items: list[str] = []
            for i, token in enumerate(tokens):
                is_final_punct = (i == len(tokens) - 1) and token in {".", "?", "!", "...", "…"}
                if is_final_punct:
                    wor_items.append(token)
                else:
                    if i < len(utt.words) and utt.words[i].start_ms is not None and utt.words[i].end_ms is not None:
                        wor_items.append(f"{token} \x15{utt.words[i].start_ms}_{utt.words[i].end_ms}\x15")
                    else:
                        wor_items.append(token)
            lines.append(f"%wor:\t{' '.join(wor_items)}")

        # %mor and %gra tiers
        if utt.morphosyntax is not None:
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
