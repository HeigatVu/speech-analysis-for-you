import numpy as np
import pytest

from say_transcribe.alignment import AlignmentError, align_words


class FakeBackend:
    blank_id = 0
    word_delimiter_id = 99
    unk_id = -1

    def __init__(self, path):
        self.path = np.asarray(path)
        self.encoded = []

    def encode(self, text):
        self.encoded.append(text)
        return [ord(character) for character in text]

    def emissions(self, audio):
        return np.zeros((len(self.path), 100), dtype=np.float32)

    def forced_align(self, log_probs, targets):
        return self.path


def test_align_words_maps_vietnamese_words_and_skips_punctuation():
    backend = FakeBackend([273, 273, 0, 7865, 0, 112, 99, 97, 0])

    spans = align_words(
        np.zeros(2880, dtype=np.float32),
        ["ĐẸP,", "...", "A"],
        backend=backend,
    )

    assert backend.encoded == ["đẹp", "a"]
    assert spans == ((0, 120), (None, None), (140, 160))


def test_align_words_returns_no_spans_when_ctc_path_is_impossible():
    backend = FakeBackend([ord("a"), 0])

    spans = align_words(np.zeros(1600, dtype=np.float32), ["aa"], backend=backend)

    assert spans == ((None, None),)


def test_align_words_leaves_an_unencodable_word_untimed_but_aligns_the_rest():
    class DigitBlindBackend(FakeBackend):
        def encode(self, text):
            self.encoded.append(text)
            if text.isdigit():
                return []
            return [ord(character) for character in text]

    backend = DigitBlindBackend([273, 273, 0, 7865, 0, 112, 99, 97, 0])

    spans = align_words(
        np.zeros(2880, dtype=np.float32),
        ["ĐẸP,", "5", "A"],
        backend=backend,
    )

    assert backend.encoded == ["đẹp", "5", "a"]
    assert spans == ((0, 120), (None, None), (140, 160))


def test_align_words_returns_no_spans_for_unsupported_characters():
    backend = FakeBackend([])

    spans = align_words(np.zeros(1600, dtype=np.float32), ["🙂"], backend=backend)

    assert spans == ((None, None),)


def test_align_words_redacts_backend_failures_with_stable_code():
    class BrokenBackend(FakeBackend):
        def emissions(self, audio):
            raise RuntimeError("private path /data/person.wav")

    with pytest.raises(AlignmentError) as error:
        align_words(
            np.zeros(1600, dtype=np.float32),
            ["a"],
            backend=BrokenBackend([]),
        )

    assert error.value.code == "ALIGNMENT_UNAVAILABLE"
    assert "/data/person.wav" not in str(error.value)
