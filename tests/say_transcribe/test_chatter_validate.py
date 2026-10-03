"""Generated .cha files must pass `chatter validate` (TalkBank/chatter, the CHAT
reference validator Batchalign3 uses). Skipped only when no chatter binary exists:
put one on PATH or point CHATTER_BIN at it."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from say_transcribe.chat_writer import UtteranceRecord, format_chat_session
from say_transcribe.morphosyntax import GraItem, MorItem, UtteranceMorphosyntax
from say_transcribe.word_grouping import GroupedWord

CHATTER = os.environ.get("CHATTER_BIN") or shutil.which("chatter")
NEEDS_CHATTER = pytest.mark.skipif(CHATTER is None, reason="chatter binary not found")


def _word(text: str, start: int | None = None, end: int | None = None) -> GroupedWord:
    return GroupedWord(word=text, start_ms=start, end_ms=end, syllables=())


def _utterance(words, mor=None, gra=None, timed=True) -> UtteranceRecord:
    return UtteranceRecord(
        speaker="PAR",
        start_ms=0 if timed else None,
        end_ms=3000 if timed else None,
        text=" ".join(w.word for w in words),
        words=tuple(words),
        morphosyntax=UtteranceMorphosyntax(tuple(mor), tuple(gra)) if mor else None,
    )


def _plain() -> UtteranceRecord:
    return _utterance(
        [_word("tôi", 0, 400), _word("đi", 400, 800), _word("học", 800, 1200), _word(".")],
        [MorItem("pron", "tôi"), MorItem("verb", "đi"), MorItem("verb", "học"), MorItem("punct", ".")],
        [GraItem(1, 2, "NSUBJ"), GraItem(2, 0, "ROOT"), GraItem(3, 2, "XCOMP"), GraItem(4, 2, "PUNCT")],
    )


def _comma() -> UtteranceRecord:
    return _utterance(
        [_word("vâng", 0, 300), _word(","), _word("tôi", 400, 700), _word(".")],
        [MorItem("intj", "vâng"), MorItem("cm", "cm"), MorItem("pron", "tôi"), MorItem("punct", ".")],
        [GraItem(1, 3, "DISCOURSE"), GraItem(2, 3, "PUNCT"), GraItem(3, 0, "ROOT"), GraItem(4, 3, "PUNCT")],
    )


def _feats() -> UtteranceRecord:
    return _utterance(
        [_word("chúng_tôi", 0, 500), _word("đi", 500, 900), _word(".")],
        [MorItem("pron", "chúng_tôi", "Number=Plur|Person=1"), MorItem("verb", "đi"), MorItem("punct", ".")],
        [GraItem(1, 2, "NSUBJ"), GraItem(2, 0, "ROOT"), GraItem(3, 2, "PUNCT")],
    )


def _untimed_main_tier_only() -> UtteranceRecord:
    return _utterance([_word("tôi"), _word("đi"), _word(".")], timed=False)


def _unintelligible_only() -> UtteranceRecord:
    return _utterance([_word("xxx"), _word(".")], timed=False)


CASES = {
    "plain": _plain,
    "comma": _comma,
    "ud_feats": _feats,
    "main_tier_only": _untimed_main_tier_only,
    "unintelligible_only": _unintelligible_only,
}


def validate(tmp_path: Path, utterances: list[UtteranceRecord]) -> subprocess.CompletedProcess:
    # chatter requires the @Media name to match the file name.
    path = tmp_path / "s1.cha"
    path.write_text(format_chat_session("s1", "0" * 64, utterances), encoding="utf-8")
    return subprocess.run(
        [CHATTER, "validate", "--force", "-f", "text", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )


@NEEDS_CHATTER
@pytest.mark.parametrize("name", sorted(CASES))
def test_generated_session_is_chatter_valid(tmp_path: Path, name: str) -> None:
    result = validate(tmp_path, [CASES[name]()])

    assert result.returncode == 0, result.stdout[-1500:]


@NEEDS_CHATTER
def test_repo_chatter_fixture_is_chatter_valid() -> None:
    fixture = Path(__file__).parent / "fixtures" / "chatter_mor_gra_fixture.cha"

    result = subprocess.run(
        [CHATTER, "validate", "--force", "-f", "text", str(fixture)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout[-1500:]
