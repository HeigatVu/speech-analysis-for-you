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


def _segment(text, start_ms, end_ms, words=()):
    from say_transcribe.asr import AsrSegment

    return AsrSegment(start_ms, end_ms, text, tuple(words))


def test_align_segments_times_words_on_the_master_timeline_and_tightens_the_span():
    from say_transcribe.alignment import align_segments

    calls = []

    def fake_aligner(audio, words):
        calls.append((len(audio), list(words)))
        return ((100, 400), (500, 900))

    audio = np.zeros(16000 * 10, dtype=np.float32)
    [segment] = align_segments(audio, [_segment("chào bạn", 2000, 6000)], aligner=fake_aligner)

    assert calls == [(16000 * 4, ["chào", "bạn"])]  # only the segment's own samples
    assert [(w.word, w.start_ms, w.end_ms) for w in segment.words] == [
        ("chào", 2100, 2400),
        ("bạn", 2500, 2900),
    ]
    assert (segment.start_ms, segment.end_ms) == (2100, 2900)
    assert segment.text == "chào bạn"


def test_align_segments_keeps_the_window_span_when_any_word_stays_untimed():
    from say_transcribe.alignment import align_segments

    def fake_aligner(audio, words):
        return ((100, 400), (None, None))

    audio = np.zeros(16000 * 10, dtype=np.float32)
    [segment] = align_segments(audio, [_segment("chào bạn", 2000, 6000)], aligner=fake_aligner)

    assert [(w.start_ms, w.end_ms) for w in segment.words] == [(2100, 2400), (None, None)]
    assert (segment.start_ms, segment.end_ms) == (2000, 6000)


def test_align_segments_leaves_timed_or_unbounded_segments_alone():
    from say_transcribe.alignment import align_segments
    from say_transcribe.asr import WordTiming

    def boom(audio, words):
        raise AssertionError("must not align")

    timed = _segment("chào", 0, 1000, [WordTiming("chào", 0, 900)])
    unbounded = _segment("chào", None, None)
    audio = np.zeros(16000 * 2, dtype=np.float32)

    assert align_segments(audio, [timed, unbounded], aligner=boom) == (timed, unbounded)


def test_align_segments_rejects_a_span_count_mismatch_without_fabricating_times():
    from say_transcribe.alignment import align_segments

    audio = np.zeros(16000 * 4, dtype=np.float32)
    [segment] = align_segments(
        audio, [_segment("a b", 0, 3000)], aligner=lambda a, w: ((0, 100),)
    )

    assert segment.words == ()
    assert (segment.start_ms, segment.end_ms) == (0, 3000)
