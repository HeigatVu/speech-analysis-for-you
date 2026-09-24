"""Behavioral checks for in-memory pilot transcription variants."""

import numpy as np
import pytest

from say_transcribe.asr import AsrError, result_from_windows, transcribe_windows


class FakeAsr:
    def __init__(self, segments):
        self.segments = segments
        self.clip_lengths = []

    def transcribe_audio(self, clip):
        self.clip_lengths.append(len(clip))
        return self.segments


def test_vad_window_asr_restores_master_timeline_without_second_asr_call():
    audio = np.zeros(4 * 16000, dtype=np.float32)
    backend = FakeAsr(
        [
            {
                "start_ms": 100,
                "end_ms": 700,
                "text": "xin chào",
                "words": [
                    {"word": "xin", "start_ms": 100, "end_ms": 300},
                    {"word": "chào", "start_ms": 400, "end_ms": 700},
                ],
            }
        ]
    )

    windows = transcribe_windows(audio, [(16000, 3 * 16000)], backend)
    plain = result_from_windows(audio, "a" * 64, windows)
    aligned = result_from_windows(
        audio,
        "a" * 64,
        windows,
        aligner=lambda clip, words: ((100, 300), (400, 700)),
    )

    assert backend.clip_lengths == [2 * 16000]
    assert (plain.segments[0].start_ms, plain.segments[0].end_ms) == (1100, 1700)
    assert (aligned.segments[0].start_ms, aligned.segments[0].end_ms) == (1100, 1700)
    assert [word.word for word in aligned.segments[0].words] == ["xin", "chào"]


def test_alignment_splits_after_asr_and_keeps_unalignable_text_untimed():
    audio = np.zeros(3 * 16000, dtype=np.float32)
    backend = FakeAsr(
        [
            {
                "start_ms": None,
                "end_ms": None,
                "text": "xin chào. cảm ơn",
                "words": [
                    {"word": "xin", "start_ms": None, "end_ms": None},
                    {"word": "chào.", "start_ms": None, "end_ms": None},
                    {"word": "cảm", "start_ms": None, "end_ms": None},
                    {"word": "ơn", "start_ms": None, "end_ms": None},
                ],
            }
        ]
    )
    windows = transcribe_windows(audio, [(16000, 3 * 16000)], backend)
    result = result_from_windows(
        audio,
        "a" * 64,
        windows,
        aligner=lambda clip, words: ((100, 250), (300, 450), (None, None), (1400, 1600)),
    )

    assert [segment.text for segment in result.segments] == ["xin chào.", "cảm ơn"]
    assert (result.segments[0].start_ms, result.segments[0].end_ms) == (1100, 1450)
    assert result.segments[1].start_ms is None
    assert result.segments[1].end_ms is None
    assert "CHAT_UTTERANCE_TIMING_UNAVAILABLE:2" in result.warnings


@pytest.mark.parametrize(
    "invalid_windows",
    [
        [(2000, 1000)],  # end <= start
        [(0, 2000), (1500, 3000)],  # overlapping / out of order
        [(0, 5 * 16000)],  # exceeds audio duration
        [(0, 21 * 16000)],  # window length > 20s
    ],
)
def test_transcribe_windows_raises_on_invalid_or_unordered_windows(invalid_windows):
    audio = np.zeros(4 * 16000, dtype=np.float32)
    backend = FakeAsr([])
    with pytest.raises(AsrError) as exc_info:
        transcribe_windows(audio, invalid_windows, backend)
    assert exc_info.value.code == "INVALID_VAD_WINDOWS"
