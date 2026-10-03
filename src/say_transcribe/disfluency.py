"""Retrace and filled-pause markup for draft transcripts.

Follows Batchalign3 (`retrace.rs`, `cleanup.rs`): an exact repeated n-gram, compared
case-insensitively, is marked `[/]` on its first copy; fillers are written `&-X` and
are never marked as retraced. Words are only marked, never removed, so `%wor` keeps
every spoken word."""

import unicodedata
from dataclasses import replace
from typing import Sequence

from say_transcribe.word_grouping import GroupedWord

# Vietnamese filled pauses. Batchalign3 only lists English ones (um, ur, uh), so this
# list is a local choice; extend it as the reviewer finds more.
FILLERS = frozenset({"ờ", "ừ", "à", "ơ", "ừm", "ờm"})
_MAX_NGRAM = 4
_UNTRANSCRIBED = frozenset({"xxx", "yyy", "www"})


def _key(word: str) -> str:
    return unicodedata.normalize("NFC", word).casefold()


def _is_lexical(word: str) -> bool:
    return any(c.isalnum() for c in word) and _key(word) not in FILLERS | _UNTRANSCRIBED


def _retraced(words: Sequence[GroupedWord]) -> set[int]:
    """Indices of words in the first copy of an immediately repeated n-gram.

    Punctuation and `xxx` end a stretch (a repeat across a comma is not marked);
    fillers are transparent, so `tôi ờ tôi` still repeats `tôi`."""
    # ponytail: `một một bảy` (117) is marked as a retrace; the reviewer fixes it in
    # phase 1. Add a numeral stop-list if that turns out to be noisy.
    retraced: set[int] = set()
    stretch: list[int] = []

    def scan() -> None:
        keys = [_key(words[i].word) for i in stretch]
        i = 0
        while i < len(stretch):
            for n in range(min(_MAX_NGRAM, (len(stretch) - i) // 2), 0, -1):
                if keys[i : i + n] == keys[i + n : i + 2 * n]:
                    retraced.update(stretch[i : i + n])
                    i += n
                    break
            else:
                i += 1
        stretch.clear()

    for index, word in enumerate(words):
        if _is_lexical(word.word):
            stretch.append(index)
        elif _key(word.word) not in FILLERS:
            scan()
    scan()
    return retraced


def mark_disfluencies(words: Sequence[GroupedWord]) -> tuple[GroupedWord, ...]:
    """Return the words with `[/]` retrace markup and `&-` fillers added."""
    retraced = _retraced(words)
    out: list[GroupedWord] = []
    index = 0
    while index < len(words):
        word = words[index]
        if index in retraced:
            end = index
            while end + 1 in retraced:
                end += 1
            run = list(words[index : end + 1])
            if len(run) > 1:
                run[0] = replace(run[0], word="<" + run[0].word)
                run[-1] = replace(run[-1], word=run[-1].word + ">")
            out.extend(run)
            out.append(GroupedWord("[/]", None, None, ()))
            index = end + 1
            continue
        if _key(word.word) in FILLERS:
            word = replace(word, word="&-" + word.word)
        out.append(word)
        index += 1
    return tuple(out)


def drop_invalid_commas(words: Sequence[GroupedWord]) -> tuple[GroupedWord, ...]:
    """Drop commas chatter rejects: one before any spoken word (fillers and `xxx` do not
    count, E259) or directly after another comma (E258). Only punctuation is removed."""
    out: list[GroupedWord] = []
    spoken = False
    for word in words:
        if word.word == ",":
            if not spoken or (out and out[-1].word == ","):
                continue
        elif _is_lexical(word.word):
            spoken = True
        out.append(word)
    return tuple(out)
