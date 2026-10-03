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


def _quiet(peak: float, samples: int = 16000) -> np.ndarray:
    audio = np.zeros(samples, dtype=np.float32)
    audio[::7] = peak
    audio[3::7] = -peak
    return audio


def test_quiet_audio_is_boosted_to_a_minus_23_dbfs_peak_for_detection_only() -> None:
    audio = _quiet(0.02)
    original = audio.copy()
    seen: list[np.ndarray] = []

    def detector(clip: np.ndarray) -> list[tuple[int, int]]:
        seen.append(clip)
        return [(0, 8000)]

    get_speech_windows(audio, detector=detector)

    assert float(np.abs(seen[0]).max()) == pytest.approx(0.071, rel=1e-3)
    assert np.array_equal(audio, original)


def test_loud_audio_is_not_boosted() -> None:
    audio = _quiet(0.5)
    seen: list[np.ndarray] = []

    def detector(clip: np.ndarray) -> list[tuple[int, int]]:
        seen.append(clip)
        return [(0, 8000)]

    get_speech_windows(audio, detector=detector)

    assert float(np.abs(seen[0]).max()) == pytest.approx(0.5)


def test_boost_gain_is_capped_so_near_silence_is_not_amplified_into_speech() -> None:
    seen: list[np.ndarray] = []

    def detector(clip: np.ndarray) -> list[tuple[int, int]]:
        seen.append(clip)
        return []

    get_speech_windows(_quiet(1e-5), detector=detector)

    assert float(np.abs(seen[0]).max()) == pytest.approx(1e-3, rel=1e-3)


def test_empty_first_pass_retries_once_with_lower_threshold_and_shorter_speech(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[float, int]] = []

    def detect(_audio: np.ndarray, threshold: float, min_speech_ms: int = 250):
        calls.append((threshold, min_speech_ms))
        return [] if len(calls) == 1 else [(0, 3000)]  # 187 ms: kept only by the retry

    monkeypatch.setattr("say_transcribe.vad._load_silero_detector", lambda: detect)

    assert get_speech_windows(_quiet(0.02), threshold=0.2) == [(0, 3480)]
    assert calls == [(0.2, 250), (0.1, 150)]


def test_no_retry_when_the_first_pass_finds_speech(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[float] = []

    def detect(_audio: np.ndarray, threshold: float, min_speech_ms: int = 250):
        calls.append(threshold)
        return [(0, 8000)]

    monkeypatch.setattr("say_transcribe.vad._load_silero_detector", lambda: detect)
    get_speech_windows(_quiet(0.02), threshold=0.2)

    assert calls == [0.2]


# The three ported robustness behaviors are separable so the ablation study can
# turn one off at a time. Defaults keep every earlier caller on today's behavior.


def test_boost_can_be_turned_off_for_an_ablation_arm() -> None:
    seen: list[np.ndarray] = []

    def detector(clip: np.ndarray) -> list[tuple[int, int]]:
        seen.append(clip)
        return [(0, 8000)]

    get_speech_windows(_quiet(0.02), detector=detector, boost=False)

    assert float(np.abs(seen[0]).max()) == pytest.approx(0.02)


def test_retry_can_be_turned_off_for_an_ablation_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[float] = []

    def detect(_audio: np.ndarray, threshold: float, min_speech_ms: int = 250):
        calls.append(threshold)
        return []

    monkeypatch.setattr("say_transcribe.vad._load_silero_detector", lambda: detect)

    assert get_speech_windows(_quiet(0.02), threshold=0.2, retry=False) == []
    assert calls == [0.2]


def _long_speech_with_a_quiet_patch() -> tuple[np.ndarray, int]:
    """45 s of loud speech with 500 ms of silence at 18.5 s, inside the 2 s search."""
    rate = 16000
    audio = np.full(45 * rate, 0.5, dtype=np.float32)
    quiet_start = 18 * rate + rate // 2
    audio[quiet_start : quiet_start + rate // 2] = 0.0
    return audio, quiet_start


def test_a_hard_cut_is_the_default_until_an_arm_asks_for_snapping() -> None:
    audio, _ = _long_speech_with_a_quiet_patch()

    windows = get_speech_windows(audio, detector=lambda _: [(0, 40 * 16000)])

    assert windows[0] == (0, 20 * 16000)


def test_snapping_cuts_at_the_quietest_frame_before_the_hard_limit() -> None:
    rate = 16000
    audio, quiet_start = _long_speech_with_a_quiet_patch()

    snapped = get_speech_windows(audio, detector=lambda _: [(0, 40 * rate)], snap=True)
    hard = get_speech_windows(audio, detector=lambda _: [(0, 40 * rate)])

    # The cut moves into the quiet patch, never past it and never past the hard limit.
    assert quiet_start <= snapped[0][1] <= quiet_start + rate // 2
    assert snapped[0][1] < hard[0][1]
    # Snapping re-cuts the same audio, so nothing is dropped or duplicated.
    assert snapped[0][0] == hard[0][0] == 0
    assert all(left[1] == right[0] for left, right in zip(snapped, snapped[1:]))
    assert snapped[-1][1] == hard[-1][1]
