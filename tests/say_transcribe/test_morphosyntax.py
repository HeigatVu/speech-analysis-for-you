from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import pytest

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


def test_mor_renders_ud_feats_as_chat_suffixes_with_a_single_pipe():
    item = MorItem(pos="pron", lemma="tôi", feats="Number=Plur|Person=1")

    assert item.format_mor() == "pron|tôi-P1"
    assert item.format_mor().count("|") == 1


def test_mor_members_keep_words_and_separators_but_not_markup():
    from say_transcribe.morphosyntax import mor_members
    from say_transcribe.word_grouping import GroupedWord

    words = [
        GroupedWord(text, None, None, ())
        for text in ["&-ờ", "<tôi", "đi>", "[/]", "tôi", "đi", ",", "xxx", "(.)", "."]
    ]

    assert [w.word for w in mor_members(words)] == ["tôi", "đi", ",", "."]


@pytest.mark.parametrize("pos", ["adp", "adv", "cconj", "intj", "num", "part", "sconj", "sym", "x", "punct"])
def test_mor_writes_no_features_for_function_word_pos(pos):
    assert MorItem(pos=pos, lemma="của", feats="Case=Gen").format_mor() == f"{pos}|của"


def test_mor_keeps_comma_joined_feature_values():
    assert MorItem("noun", "nhà", "Number=Plur,Sing").format_mor() == "noun|nhà-Plur,Sing"


@pytest.mark.parametrize("pos,feats,suffix", [
    ("verb", "Person=0|Tense=Pres|Mood=Ind|Number=Plur|VerbForm=Fin", "Fin-Ind-Pres-P4"),
    ("pron", "Person=1|Case=Nom|Number=Plur|PronType=Prs|Reflex=Yes", "Prs-Nom-reflx-P1"),
    ("det", "PersonPsor=1|NumberPsor=Sing|Number=Plur|PronType=Dem|Definite=Def|Gender=Com", "Def-Dem-Plur-S1"),
    ("adj", "Number=Sing|Degree=Pos|Case=Nom", "Nom-S"),
    ("noun", "PronType=Prs|Case=Nom|Number=Sing|Gender=Com", "Nom-Prs"),
])
def test_batchalign_features_use_pos_specific_order(pos, feats, suffix):
    assert MorItem(pos, "đẹp", feats).format_mor() == f"{pos}|đẹp-{suffix}"


def test_mor_projection_corrections_scope_and_annotated_retrace():
    from say_transcribe.morphosyntax import mor_members
    words = [GroupedWord(t, None, None, ()) for t in
             ["<tôi", "đi>", "[:", "chúng", "tôi]", "[=", "ghi", "chú]", "[*]", "[/]", "<đến", "nhà>", "(1.2)", "."]]
    assert [w.word for w in mor_members(words)] == ["đến", "nhà", "."]


def test_mor_projection_multiple_replacement_words():
    from say_transcribe.morphosyntax import mor_members
    words = [GroupedWord(t, None, None, ()) for t in ["tôi", "[:", "chúng", "tôi]", "đi", "."]]
    assert [w.word for w in mor_members(words)] == ["chúng", "tôi", "đi", "."]
