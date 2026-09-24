from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

from say_transcribe.morphosyntax import (
    GraItem,
    MorItem,
    StanzaBackend,
    UtteranceMorphosyntax,
    project_morphosyntax,
)
from say_transcribe.word_grouping import GroupedWord


@dataclass
class FakeWord:
    id: int
    text: str
    lemma: str
    upos: str
    feats: str | None
    head: int
    deprel: str


@dataclass
class FakeSentence:
    words: list[FakeWord]


@dataclass
class FakeDoc:
    sentences: list[FakeSentence]


class FakeStanzaBackend(StanzaBackend):
    def __init__(self, fake_doc: FakeDoc | None = None, fail: bool = False) -> None:
        super().__init__(lang="vi", device="cpu")
        self._fake_doc = fake_doc
        self._fail = fail

    def load(self) -> None:
        pass

    def parse_pretokenized(self, tokens: Sequence[str]) -> Any:
        if self._fail:
            raise RuntimeError("Stanza fake error")
        return self._fake_doc


def test_lazy_import_does_not_load_stanza():
    code = (
        "import sys, say_transcribe\n"
        "assert 'stanza' not in sys.modules, 'stanza was eagerly imported'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"Import test failed:\n{result.stderr}"


def test_chatter_fixture_file_exists_and_has_content():
    fixture_path = (
        Path(__file__).parent / "fixtures" / "chatter_mor_gra_fixture.cha"
    )
    assert fixture_path.exists()
    content = fixture_path.read_text(encoding="utf-8")
    assert "@Begin" in content
    assert "@Languages:\tvie" in content
    assert "%mor:\t" in content
    assert "%gra:\t" in content
    assert "học_sinh" in content
    assert "@End" in content


def test_mor_keeps_underscore_joiner_in_lemma():
    grouped = [
        GroupedWord(word="học_sinh", start_ms=0, end_ms=500, syllables=()),
        GroupedWord(word=".", start_ms=None, end_ms=None, syllables=()),
    ]
    fake_doc = FakeDoc(
        sentences=[
            FakeSentence(
                words=[
                    FakeWord(
                        id=1,
                        text="học_sinh",
                        lemma="học_sinh",
                        upos="NOUN",
                        feats=None,
                        head=0,
                        deprel="root",
                    ),
                    FakeWord(
                        id=2,
                        text=".",
                        lemma=".",
                        upos="PUNCT",
                        feats=None,
                        head=1,
                        deprel="punct",
                    ),
                ]
            )
        ]
    )
    backend = FakeStanzaBackend(fake_doc)
    res = project_morphosyntax(grouped, backend=backend)

    assert isinstance(res, UtteranceMorphosyntax)
    assert len(res.mor_items) == 2
    assert res.mor_items[0] == MorItem(pos="noun", lemma="học_sinh", feats=None)
    assert res.mor_items[0].format_mor() == "noun|học_sinh"
    assert res.mor_line() == "%mor:\tnoun|học_sinh ."


def test_comma_renders_as_cm_cm():
    grouped = [
        GroupedWord(word="vâng", start_ms=0, end_ms=200, syllables=()),
        GroupedWord(word=",", start_ms=200, end_ms=300, syllables=()),
        GroupedWord(word=".", start_ms=None, end_ms=None, syllables=()),
    ]
    fake_doc = FakeDoc(
        sentences=[
            FakeSentence(
                words=[
                    FakeWord(
                        id=1,
                        text="vâng",
                        lemma="vâng",
                        upos="INTJ",
                        feats=None,
                        head=0,
                        deprel="root",
                    ),
                    FakeWord(
                        id=2,
                        text=",",
                        lemma=",",
                        upos="PUNCT",
                        feats=None,
                        head=1,
                        deprel="punct",
                    ),
                    FakeWord(
                        id=3,
                        text=".",
                        lemma=".",
                        upos="PUNCT",
                        feats=None,
                        head=1,
                        deprel="punct",
                    ),
                ]
            )
        ]
    )
    backend = FakeStanzaBackend(fake_doc)
    res = project_morphosyntax(grouped, backend=backend)

    assert res is not None
    assert res.mor_items[1] == MorItem(pos="cm", lemma="cm", feats=None)
    assert res.mor_items[1].format_mor() == "cm|cm"


def test_gra_enforces_root_head_zero_joint_invariant_and_terminator():
    grouped = [
        GroupedWord(word="tôi", start_ms=0, end_ms=200, syllables=()),
        GroupedWord(word="đi", start_ms=200, end_ms=400, syllables=()),
        GroupedWord(word="học", start_ms=400, end_ms=600, syllables=()),
        GroupedWord(word=".", start_ms=None, end_ms=None, syllables=()),
    ]
    fake_doc = FakeDoc(
        sentences=[
            FakeSentence(
                words=[
                    FakeWord(
                        id=1,
                        text="tôi",
                        lemma="tôi",
                        upos="PRON",
                        feats=None,
                        head=2,
                        deprel="nsubj",
                    ),
                    # Stanza had root with head=0 and deprel="root"
                    FakeWord(
                        id=2,
                        text="đi",
                        lemma="đi",
                        upos="VERB",
                        feats=None,
                        head=0,
                        deprel="root",
                    ),
                    FakeWord(
                        id=3,
                        text="học",
                        lemma="học",
                        upos="VERB",
                        feats=None,
                        head=2,
                        deprel="xcomp",
                    ),
                    # Terminator punctuation
                    FakeWord(
                        id=4,
                        text=".",
                        lemma=".",
                        upos="PUNCT",
                        feats=None,
                        head=2,
                        deprel="punct",
                    ),
                ]
            )
        ]
    )
    backend = FakeStanzaBackend(fake_doc)
    res = project_morphosyntax(grouped, backend=backend)

    assert res is not None
    assert res.gra_items[0] == GraItem(index=1, head=2, rel="NSUBJ")
    # Root invariant: head == 0 and rel == "ROOT"
    assert res.gra_items[1] == GraItem(index=2, head=0, rel="ROOT")
    assert res.gra_items[2] == GraItem(index=3, head=2, rel="XCOMP")
    # Terminator rule: (n+1)|root|PUNCT -> 4|2|PUNCT
    assert res.gra_items[3] == GraItem(index=4, head=2, rel="PUNCT")
    assert res.gra_line() == "%gra:\t1|2|NSUBJ 2|0|ROOT 3|2|XCOMP 4|2|PUNCT"


def test_stanza_failure_returns_none():
    grouped = [GroupedWord(word="lỗi", start_ms=0, end_ms=200, syllables=())]
    backend = FakeStanzaBackend(fail=True)
    res = project_morphosyntax(grouped, backend=backend)
    assert res is None
