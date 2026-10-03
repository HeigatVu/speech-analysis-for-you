import pytest

from say_transcribe.disfluency import mark_disfluencies
from say_transcribe.word_grouping import GroupedWord


def _words(text: str) -> list[GroupedWord]:
    return [
        GroupedWord(w, i * 100, i * 100 + 90, ()) for i, w in enumerate(text.split())
    ]


def _marked(text: str) -> str:
    return " ".join(w.word for w in mark_disfluencies(_words(text)))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("tôi tôi đi .", "tôi [/] tôi đi ."),
        ("tôi đi tôi đi học .", "<tôi đi> [/] tôi đi học ."),
        ("Tôi tôi đi .", "Tôi [/] tôi đi ."),
        ("ờ tôi đi .", "&-ờ tôi đi ."),
        ("Ờ tôi ừm đi .", "&-Ờ tôi &-ừm đi ."),
        ("tôi ờ tôi đi .", "tôi [/] &-ờ tôi đi ."),
        ("ờ tôi ờ tôi đi .", "&-ờ tôi [/] &-ờ tôi đi ."),
        ("a a a b .", "<a a> [/] a b ."),  # adjacent retraced words form one group
        ("tôi , tôi đi .", "tôi , tôi đi ."),
        ("xxx xxx .", "xxx xxx ."),
        ("tôi đi học .", "tôi đi học ."),
        ("", ""),
    ],
)
def test_mark_disfluencies(text: str, expected: str) -> None:
    assert _marked(text) == expected


def test_words_keep_their_timings_and_none_are_removed() -> None:
    words = _words("tôi đi tôi đi học .")

    marked = mark_disfluencies(words)

    real = [w for w in marked if w.word not in {"[/]"}]
    assert [(w.start_ms, w.end_ms) for w in real] == [(w.start_ms, w.end_ms) for w in words]
    assert [w.word.strip("<>") for w in real] == [w.word for w in words]
    assert next(w for w in marked if w.word == "[/]").start_ms is None
