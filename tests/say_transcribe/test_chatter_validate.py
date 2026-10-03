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


def _feats_variants() -> UtteranceRecord:
    return _utterance(
        [_word("nhà", 0, 300), _word("của", 300, 500), _word("tôi", 500, 800), _word(".")],
        [
            MorItem("noun", "nhà", "Number=Plur,Sing"),
            MorItem("adp", "của", "Case=Gen"),
            MorItem("pron", "tôi", "Person=1"),
            MorItem("punct", "."),
        ],
        [GraItem(1, 0, "ROOT"), GraItem(2, 1, "CASE"), GraItem(3, 2, "OBL"), GraItem(4, 1, "PUNCT")],
    )


def _untimed_main_tier_only() -> UtteranceRecord:
    return _utterance([_word("tôi"), _word("đi"), _word(".")], timed=False)


def _unintelligible_only() -> UtteranceRecord:
    return _utterance([_word("xxx"), _word(".")], timed=False)


def _markup() -> UtteranceRecord:
    words = [
        _word("&-ờ", 0, 100),
        _word("<tôi", 100, 400),
        _word("đi>", 400, 800),
        _word("[/]"),
        _word("tôi", 800, 1100),
        _word("đi", 1100, 1500),
        _word("xxx"),
        _word("(.)"),
        _word("."),
    ]
    return _utterance(
        words,
        [MorItem("pron", "tôi"), MorItem("verb", "đi"), MorItem("punct", ".")],
        [GraItem(1, 2, "NSUBJ"), GraItem(2, 0, "ROOT"), GraItem(3, 2, "PUNCT")],
    )


def _markup_without_analysable_word() -> UtteranceRecord:
    return _utterance([_word("&-ờ", 0, 100), _word("xxx"), _word(".")])


CASES = {
    "retrace_filler_untranscribed_pause": _markup,
    "no_analysable_word": _markup_without_analysable_word,
    "plain": _plain,
    "comma": _comma,
    "ud_feats": _feats,
    "ud_feats_multi_value_and_function_pos": _feats_variants,
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


class _FakeWord:
    def __init__(self, index: int, text: str) -> None:
        self.id = index
        self.text = text
        self.lemma = text
        self.upos = "noun"
        self.feats = None
        self.head = 0 if index == 1 else 1
        self.deprel = "root" if index == 1 else "nsubj"


class _FakeStanza:
    def __init__(self, **kwargs) -> None:
        pass

    def parse_pretokenized(self, tokens):
        words = [_FakeWord(i, t) for i, t in enumerate(tokens, start=1)]
        return type("D", (), {"sentences": [type("S", (), {"words": words})()]})()


@NEEDS_CHATTER
def test_tag_output_for_reviewed_markup_is_chatter_valid(tmp_path: Path, monkeypatch) -> None:
    from say_transcribe.cli import main

    reviewed = tmp_path / "review" / "s1.cha"
    reviewed.parent.mkdir()
    reviewed.write_text(format_chat_session("s1", "0" * 64, [_markup()]), encoding="utf-8")
    monkeypatch.setattr("say_transcribe.cli.StanzaBackend", _FakeStanza)
    final = tmp_path / "final" / "s1.cha"

    assert main(["tag", str(reviewed), "--out", str(final)]) == 0

    result = subprocess.run(
        [CHATTER, "validate", "--force", "-f", "text", str(final)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-1500:]
