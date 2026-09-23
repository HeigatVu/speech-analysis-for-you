from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from say_transcribe.morphosyntax import UtteranceMorphosyntax
from say_transcribe.word_grouping import GroupedWord


@dataclass(frozen=True)
class UtteranceRecord:
    speaker: str  # "PAR" or "INV"
    start_ms: int
    end_ms: int
    text: str
    words: tuple[GroupedWord, ...]
    morphosyntax: UtteranceMorphosyntax | None


def format_chat_session(
    session_id: str,
    source_sha256: str,
    utterances: Sequence[UtteranceRecord],
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
        "@Comment:\tspeaker labels draft, auto-diarized; review before use",
    ]

    for utt in utterances:
        tokens = [w.word for w in utt.words] if utt.words else [t for t in utt.text.strip().split() if t]
        if not tokens:
            continue

        # Enforce utterance-final punctuation
        if tokens[-1] not in {".", "?", "!", "...", "…"}:
            tokens.append(".")

        # Main tier line
        main_text = " ".join(tokens)
        lines.append(f"*{utt.speaker}:\t{main_text}\t\x15{utt.start_ms}_{utt.end_ms}\x15")

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
) -> Path:
    """Write one .cha file for a session."""
    content = format_chat_session(
        session_id=session_id,
        source_sha256=source_sha256,
        utterances=utterances,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path
