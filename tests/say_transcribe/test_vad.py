import numpy as np
import pytest

from say_transcribe.vad import VADUnavailableError, get_speech_windows, merge_asr_windows


def test_returns_empty_for_recording_without_speech() -> None:
    audio = np.zeros(16000, dtype=np.float32)

    assert get_speech_windows(audio, detector=lambda _: []) == []


def test_asr_windows_join_nearby_speech_for_context_without_exceeding_20_seconds() -> None:
    rate = 16000
    speech = [
        (0, 2 * rate),
        (6 * rate, 8 * rate),
        (15 * rate, 19 * rate),
        (21 * rate, 23 * rate),
        (30 * rate, 31 * rate),
    ]

    # Greedy: keep folding the next span in as long as the window stays <= 20s,
    # regardless of the gap before it (WhisperX merge_chunks pattern).
    assert merge_asr_windows(speech) == [
        (0, 19 * rate),
        (21 * rate, 31 * rate),
    ]


def test_merge_asr_windows_starts_a_new_window_when_the_span_itself_exceeds_20_seconds() -> None:
    rate = 16000
    speech = [(0, 5 * rate), (6 * rate, 27 * rate)]

    assert merge_asr_windows(speech) == [(0, 5 * rate), (6 * rate, 27 * rate)]


def test_calibrated_threshold_reaches_default_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[float] = []

    def detect(_audio: np.ndarray, threshold: float) -> list[tuple[int, int]]:
        observed.append(threshold)
        return [(0, 5000)]

    monkeypatch.setattr("say_transcribe.vad._load_silero_detector", lambda: detect)
    assert get_speech_windows(np.zeros(16000, dtype=np.float32), threshold=0.2)
    assert observed == [0.2]


def test_discards_speech_shorter_than_250ms() -> None:
    audio = np.zeros(16000, dtype=np.float32)

    assert get_speech_windows(audio, detector=lambda _: [(1000, 4999)]) == []


def test_pads_and_merges_speech_within_700ms() -> None:
    audio = np.zeros(3 * 16000, dtype=np.float32)
    spans = [(16000, 20000), (30000, 34000)]  # 625 ms of silence

    assert get_speech_windows(audio, detector=lambda _: spans) == [(15520, 34480)]


def test_keeps_speech_separate_when_silence_exceeds_700ms() -> None:
    audio = np.zeros(3 * 16000, dtype=np.float32)
    spans = [(16000, 20000), (32001, 36001)]  # 750 ms of silence

    assert get_speech_windows(audio, detector=lambda _: spans) == [
        (15520, 20480),
        (31521, 36481),
    ]


def test_splits_long_speech_into_adjacent_windows_without_dropping_samples() -> None:
    audio = np.zeros(45 * 16000, dtype=np.float32)
    speech_end = 40 * 16000

    windows = get_speech_windows(audio, detector=lambda _: [(0, speech_end - 480)])

    assert windows == [(0, 20 * 16000), (20 * 16000, speech_end)]
    assert all(start < end for start, end in windows)


def test_rejects_audio_that_is_not_mono() -> None:
    stereo = np.zeros((16000, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="one-dimensional"):
        get_speech_windows(stereo, detector=lambda _: [])


def test_hides_detector_errors_behind_stable_vad_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable() -> None:
        raise RuntimeError("private model path and token")

    monkeypatch.setattr("say_transcribe.vad._load_silero_detector", unavailable)

    with pytest.raises(VADUnavailableError) as caught:
        get_speech_windows(np.zeros(16000, dtype=np.float32))

    assert caught.value.code == "VAD_UNAVAILABLE"
    assert "private model path" not in str(caught.value)
